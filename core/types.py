"""Value types shared by the partitioner, manifest writer, and job generator.

Plain dataclasses with explicit `to_dict`/`from_dict` rather than a serialisation
library: the on-disk manifest is a stable contract that outlives this package's
internals, so the mapping between them is written out by hand on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

Vec3 = tuple[float, float, float]


def _as_vec3(values: Sequence[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"expected 3 components, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box in world space."""

    min: Vec3
    max: Vec3

    def __post_init__(self) -> None:
        for lo, hi, axis in zip(self.min, self.max, "xyz"):
            if hi < lo:
                raise ValueError(f"AABB inverted on {axis}: {lo} > {hi}")

    @property
    def extent(self) -> Vec3:
        return tuple(hi - lo for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    @property
    def center(self) -> Vec3:
        return tuple((lo + hi) / 2.0 for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    def contains(self, point: Sequence[float]) -> bool:
        p = _as_vec3(point)
        return all(lo <= c <= hi for lo, c, hi in zip(self.min, p, self.max))

    def expanded(self, margin: float) -> AABB:
        """Uniform dilation. Used to derive a context box from a core box."""
        return AABB(
            min=tuple(c - margin for c in self.min),  # type: ignore[arg-type]
            max=tuple(c + margin for c in self.max),  # type: ignore[arg-type]
        )

    def intersects(self, other: AABB) -> bool:
        return all(
            self.min[i] <= other.max[i] and other.min[i] <= self.max[i] for i in range(3)
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"min": list(self.min), "max": list(self.max)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AABB:
        return cls(min=_as_vec3(data["min"]), max=_as_vec3(data["max"]))


@dataclass(frozen=True)
class CameraRef:
    """A camera identified by its COLMAP image id, plus what we need for assignment.

    Deliberately not a full intrinsics record: block assignment needs position and
    look direction, and the authoritative intrinsics stay in the COLMAP model that
    gets copied into each block dataset.
    """

    image_id: int
    name: str
    position: Vec3
    forward: Vec3

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "name": self.name,
            "position": list(self.position),
            "forward": list(self.forward),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraRef:
        return cls(
            image_id=int(data["image_id"]),
            name=str(data["name"]),
            position=_as_vec3(data["position"]),
            forward=_as_vec3(data["forward"]),
        )


@dataclass
class Block:
    """One trainable unit: a core volume to keep, a context volume to train on.

    `core_box` tiles the scene without gaps and defines what survives the Phase 3
    merge crop. `context_box` is the dilated region whose cameras and points feed
    training, so blocks agree near their shared seams.
    """

    block_id: str
    core_box: AABB
    context_box: AABB
    camera_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.context_box.intersects(self.core_box):
            raise ValueError(f"{self.block_id}: context box does not contain core box")

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "core_box": self.core_box.to_dict(),
            "context_box": self.context_box.to_dict(),
            "camera_ids": list(self.camera_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        return cls(
            block_id=str(data["block_id"]),
            core_box=AABB.from_dict(data["core_box"]),
            context_box=AABB.from_dict(data["context_box"]),
            camera_ids=[int(i) for i in data.get("camera_ids", [])],
        )


@dataclass
class BlockSet:
    """The full partition of one scene. Produced by `core.partition`."""

    run_id: str
    blocks: list[Block] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def block(self, block_id: str) -> Block:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        raise KeyError(block_id)

    def assigned_camera_ids(self) -> set[int]:
        return {cid for b in self.blocks for cid in b.camera_ids}
