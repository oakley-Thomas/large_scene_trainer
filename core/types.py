"""Value types shared by the partitioner, manifest writer, and job generator.

Plain dataclasses with explicit `to_dict`/`from_dict` rather than a serialisation
library: the on-disk manifest is a stable contract that outlives this package's
internals, so the mapping between them is written out by hand on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

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
    """The pose and intrinsics needed by trajectory partitioning.

    ``R_world_cam`` follows COLMAP's convention: it maps world coordinates into
    camera coordinates. Its rows are consequently the camera axes in world space.
    """

    image_id: int
    name: str
    camera_id: int
    C_world: np.ndarray
    R_world_cam: np.ndarray
    fx: float
    fy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        centre = np.asarray(self.C_world, dtype=np.float64)
        rotation = np.asarray(self.R_world_cam, dtype=np.float64)
        if centre.shape != (3,):
            raise ValueError(f"C_world must have shape (3,), got {centre.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"R_world_cam must have shape (3, 3), got {rotation.shape}")
        object.__setattr__(self, "C_world", centre)
        object.__setattr__(self, "R_world_cam", rotation)

    @property
    def right_world(self) -> np.ndarray:
        return self.R_world_cam[0, :]

    @property
    def down_world(self) -> np.ndarray:
        return self.R_world_cam[1, :]

    @property
    def forward_world(self) -> np.ndarray:
        return self.R_world_cam[2, :]


@dataclass(frozen=True)
class GroundFrame:
    """Right-handed trajectory frame with rows expressed in world space."""

    R: np.ndarray
    origin: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.R, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {rotation.shape}")
        if origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {origin.shape}")
        object.__setattr__(self, "R", rotation)
        object.__setattr__(self, "origin", origin)

    def to_ground(self, pts_world: np.ndarray) -> np.ndarray:
        """Transform one or more world-space points into ``(u, v, h)``."""
        points = np.asarray(pts_world, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError(f"points must end in dimension 3, got {points.shape}")
        return (points - self.origin) @ self.R.T

    def to_world(self, pts_ground: np.ndarray) -> np.ndarray:
        """Transform one or more ``(u, v, h)`` points into world space."""
        points = np.asarray(pts_ground, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError(f"points must end in dimension 3, got {points.shape}")
        return points @ self.R + self.origin

    def to_dict(self) -> dict[str, Any]:
        return {"R": self.R.tolist(), "origin": self.origin.tolist(), "version": 1}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundFrame:
        if data.get("version") != 1:
            raise ValueError(f"unsupported ground-frame version {data.get('version')!r}")
        return cls(
            R=np.asarray(data["R"], dtype=np.float64),
            origin=np.asarray(data["origin"], dtype=np.float64),
        )


@dataclass(frozen=True)
class PlaneDiagnostics:
    """Observability and quality information emitted by trajectory fitting."""

    estimator_used: str
    normal_orientation: np.ndarray | None
    orientation_dispersion_deg: float | None
    normal_pca: np.ndarray | None
    pca_condition_ratio: float | None
    pca_degenerate: bool
    agreement_deg: float | None
    height_residual_rms: float
    height_residual_max: float
    trajectory_arc_length: float
    median_inter_frame_spacing: float
    extent_ground: tuple[float, float, float]
    n_cameras: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_used": self.estimator_used,
            "normal_orientation": (
                None if self.normal_orientation is None else self.normal_orientation.tolist()
            ),
            "orientation_dispersion_deg": self.orientation_dispersion_deg,
            "normal_pca": None if self.normal_pca is None else self.normal_pca.tolist(),
            "pca_condition_ratio": self.pca_condition_ratio,
            "pca_degenerate": self.pca_degenerate,
            "agreement_deg": self.agreement_deg,
            "height_residual_rms": self.height_residual_rms,
            "height_residual_max": self.height_residual_max,
            "trajectory_arc_length": self.trajectory_arc_length,
            "median_inter_frame_spacing": self.median_inter_frame_spacing,
            "extent_ground": list(self.extent_ground),
            "n_cameras": self.n_cameras,
            "warnings": list(self.warnings),
        }


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
