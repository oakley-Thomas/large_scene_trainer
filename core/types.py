"""Value types shared by trajectory fitting, partitioning, and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _array3(values: Sequence[float] | np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CameraRef:
    """The pose and intrinsics needed by trajectory partitioning."""

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
        centre = _array3(self.C_world, "C_world")
        rotation = np.asarray(self.R_world_cam, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"R_world_cam must have shape (3, 3), got {rotation.shape}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("R_world_cam must contain only finite values")
        rotation = rotation.copy()
        rotation.setflags(write=False)
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
        if rotation.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {rotation.shape}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("R must contain only finite values")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "R", rotation)
        object.__setattr__(self, "origin", _array3(self.origin, "origin"))

    def to_ground(self, pts_world: np.ndarray) -> np.ndarray:
        points = np.asarray(pts_world, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError(f"points must end in dimension 3, got {points.shape}")
        return (points - self.origin) @ self.R.T

    def to_world(self, pts_ground: np.ndarray) -> np.ndarray:
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


@dataclass(frozen=True)
class BlockBox:
    """Axis-aligned box in the oriented ``(u, v, h)`` ground frame."""

    min_g: np.ndarray
    max_g: np.ndarray

    def __post_init__(self) -> None:
        minimum = _array3(self.min_g, "min_g")
        maximum = _array3(self.max_g, "max_g")
        if np.any(maximum < minimum):
            raise ValueError(f"BlockBox inverted: {minimum.tolist()} > {maximum.tolist()}")
        object.__setattr__(self, "min_g", minimum)
        object.__setattr__(self, "max_g", maximum)

    def contains(self, pts_g: np.ndarray) -> np.ndarray:
        """Inclusive geometric containment for one or more ground-frame points."""
        points = np.asarray(pts_g, dtype=np.float64)
        if points.shape == (3,):
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"pts_g must have shape (N, 3), got {points.shape}")
        return np.all((points >= self.min_g) & (points <= self.max_g), axis=1)

    def overlaps(self, other: "BlockBox") -> bool:
        return bool(np.all(self.overlap_extents(other) > 0.0))

    def overlap_extents(self, other: "BlockBox") -> np.ndarray:
        return np.minimum(self.max_g, other.max_g) - np.maximum(self.min_g, other.min_g)

    @property
    def extent(self) -> np.ndarray:
        return self.max_g - self.min_g

    def dilated(self, amounts: np.ndarray) -> "BlockBox":
        margin = _array3(amounts, "amounts")
        if np.any(margin < 0.0):
            raise ValueError("dilation amounts must be non-negative")
        return BlockBox(min_g=self.min_g - margin, max_g=self.max_g + margin)

    def to_dict(self) -> dict[str, list[float]]:
        return {"min_g": self.min_g.tolist(), "max_g": self.max_g.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockBox":
        return cls(
            min_g=np.asarray(data["min_g"], dtype=np.float64),
            max_g=np.asarray(data["max_g"], dtype=np.float64),
        )


@dataclass(frozen=True)
class Block:
    """A settled trainable block, authored entirely in ``GroundFrame`` coordinates."""

    block_id: int
    core: BlockBox
    context: BlockBox
    camera_indices: tuple[int, ...]
    frame_span: tuple[int, int]
    provenance: str

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.camera_indices)
        if not indices:
            raise ValueError("a block must own at least one camera")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("camera_indices must be sorted and unique")
        span = tuple(int(index) for index in self.frame_span)
        if len(span) != 2 or span[0] > span[1]:
            raise ValueError("frame_span must be a (first, last) pair")
        if not (
            np.all(self.context.min_g <= self.core.min_g)
            and np.all(self.context.max_g >= self.core.max_g)
        ):
            raise ValueError("context must contain the core")
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        object.__setattr__(self, "camera_indices", indices)
        object.__setattr__(self, "frame_span", span)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "core": self.core.to_dict(),
            "context": self.context.to_dict(),
            "camera_indices": list(self.camera_indices),
            "frame_span": list(self.frame_span),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        return cls(
            block_id=int(data["block_id"]),
            core=BlockBox.from_dict(data["core"]),
            context=BlockBox.from_dict(data["context"]),
            camera_indices=tuple(int(index) for index in data["camera_indices"]),
            frame_span=tuple(int(index) for index in data["frame_span"]),
            provenance=str(data["provenance"]),
        )


@dataclass(frozen=True)
class SegmentationConfig:
    core_extent_su: float
    max_cameras_per_block: int
    lateral_margin_su: float
    context_margin_frac: float
    z_pad_down_su: float
    z_pad_up_su: float
    merge_slack: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "core_extent_su",
            "lateral_margin_su",
            "context_margin_frac",
            "z_pad_down_su",
            "z_pad_up_su",
            "merge_slack",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.merge_slack < 1.0:
            raise ValueError("merge_slack must be at least 1.0")
        count = int(self.max_cameras_per_block)
        if count < 3:
            raise ValueError("max_cameras_per_block must be at least 3")
        object.__setattr__(self, "max_cameras_per_block", count)


@dataclass(frozen=True)
class SegmentationDiagnostics:
    n_blocks: int
    cameras_per_block: list[int]
    core_extents: list[tuple[float, float, float]]
    actions: list[str]
    orphan_cameras: int
    multi_owned_cameras: int
    grade_pct: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_blocks": self.n_blocks,
            "cameras_per_block": list(self.cameras_per_block),
            "core_extents": [list(extent) for extent in self.core_extents],
            "actions": list(self.actions),
            "orphan_cameras": self.orphan_cameras,
            "multi_owned_cameras": self.multi_owned_cameras,
            "grade_pct": self.grade_pct,
            "warnings": list(self.warnings),
        }
