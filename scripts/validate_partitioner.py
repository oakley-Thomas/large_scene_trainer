#!/usr/bin/env python3
"""Validate an existing partition before starting per-block GPU training.

Example:

    python scripts/validate_partitioner.py \
        --dataset-root ~/LichtFeld-Studio/datasets/small_city/camera_calibration/train

The script is read-only. It checks that every parent camera is assigned to at
least one block, each camera centre belongs to exactly one core box, and each
block's actual ``points3D.bin`` count matches its manifest and export guards.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys
from typing import Sequence

import numpy as np


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core import manifest
from large_scene_trainer.core.colmap_io import load_cameras
from large_scene_trainer.core.export import BLOCKS_DIRNAME, MIN_POINT_FRACTION, MIN_POINTS_PER_BLOCK
from large_scene_trainer.core.preview_layout import load_exported_layout
from large_scene_trainer.core.types import Block, CameraRef, GroundFrame


class PartitionerValidationError(RuntimeError):
    """Raised when a block export is unsafe to send to training."""


def count_colmap_points(path: Path) -> int:
    """Read COLMAP's uint64 ``points3D.bin`` record count without loading points."""
    try:
        with path.open("rb") as handle:
            header = handle.read(8)
    except OSError as exc:
        raise PartitionerValidationError(f"cannot read {path}: {exc}") from exc
    if len(header) != 8:
        raise PartitionerValidationError(f"{path} is too short to be a COLMAP points3D.bin")
    return int(struct.unpack("<Q", header)[0])


def validate_camera_partition(
    blocks: Sequence[Block], cameras: Sequence[CameraRef], frame: GroundFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Check context assignment coverage and unique core ownership.

    Returns per-camera context-assignment and core-ownership counts for
    reporting after all mandatory checks pass.
    """
    if not cameras:
        raise PartitionerValidationError("parent dataset has no cameras")
    assignment_counts = np.zeros(len(cameras), dtype=np.intp)
    for block in blocks:
        for camera_index in block.camera_indices:
            if camera_index < 0 or camera_index >= len(cameras):
                raise PartitionerValidationError(
                    f"block {block.block_id} references out-of-range camera index {camera_index}"
                )
            assignment_counts[camera_index] += 1
    missing_assignments = np.flatnonzero(assignment_counts == 0)
    if len(missing_assignments):
        raise PartitionerValidationError(
            f"{len(missing_assignments)} camera(s) are assigned to no block; "
            f"first index {int(missing_assignments[0])}"
        )

    centres_world = np.stack([camera.C_world for camera in cameras])
    centres_ground = frame.to_ground(centres_world)
    core_ownership_counts = np.zeros(len(cameras), dtype=np.intp)
    for block in blocks:
        core_ownership_counts += block.core.contains(centres_ground)
    core_gaps = np.flatnonzero(core_ownership_counts == 0)
    if len(core_gaps):
        raise PartitionerValidationError(
            f"core boxes leave {len(core_gaps)} camera centre(s) uncovered; "
            f"first index {int(core_gaps[0])}"
        )
    core_overlaps = np.flatnonzero(core_ownership_counts > 1)
    if len(core_overlaps):
        raise PartitionerValidationError(
            f"core boxes multiply own {len(core_overlaps)} camera centre(s); "
            f"first index {int(core_overlaps[0])}"
        )
    return assignment_counts, core_ownership_counts


def validate_point_counts(
    dataset_root: Path,
    blocks: Sequence[Block],
    *,
    minimum_points: int,
    minimum_fraction: float,
) -> tuple[tuple[int, int, float], ...]:
    """Check actual per-block point counts against manifests and safety guards."""
    parent_points = count_colmap_points(dataset_root / "sparse" / "0" / "points3D.bin")
    if parent_points == 0:
        raise PartitionerValidationError("parent points3D.bin is empty")
    rows: list[tuple[int, int, float]] = []
    blocks_dir = dataset_root / BLOCKS_DIRNAME
    for block in blocks:
        block_dir = blocks_dir / f"block_{block.block_id:03d}"
        data = manifest.load(block_dir)
        actual_count = count_colmap_points(block_dir / "sparse" / "0" / "points3D.bin")
        expected_count = data.get("counts", {}).get("points3D")
        if expected_count != actual_count:
            raise PartitionerValidationError(
                f"block {block.block_id} manifest has {expected_count!r} points but file has "
                f"{actual_count}"
            )
        fraction = actual_count / parent_points
        expected_fraction = data.get("counts", {}).get("points3D_fraction")
        if not isinstance(expected_fraction, (int, float)) or not math.isclose(
            expected_fraction, fraction, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise PartitionerValidationError(
                f"block {block.block_id} manifest point fraction does not match its point count"
            )
        if actual_count < minimum_points or fraction < minimum_fraction:
            raise PartitionerValidationError(
                f"block {block.block_id} has {actual_count} points ({fraction:.3%}); need at least "
                f"{minimum_points} and {minimum_fraction:.3%}"
            )
        rows.append((block.block_id, actual_count, fraction))
    return tuple(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--minimum-points", type=int, default=MIN_POINTS_PER_BLOCK)
    parser.add_argument("--minimum-point-fraction", type=float, default=MIN_POINT_FRACTION)
    args = parser.parse_args()
    if args.minimum_points < 1:
        parser.error("--minimum-points must be positive")
    if not 0.0 < args.minimum_point_fraction <= 1.0:
        parser.error("--minimum-point-fraction must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    blocks, frame = load_exported_layout(root)
    cameras = load_cameras(root / "sparse" / "0")
    assignment_counts, core_counts = validate_camera_partition(blocks, cameras, frame)
    point_rows = validate_point_counts(
        root,
        blocks,
        minimum_points=args.minimum_points,
        minimum_fraction=args.minimum_point_fraction,
    )

    print(
        f"PASS: {len(cameras)} cameras; assignment range "
        f"{assignment_counts.min()}–{assignment_counts.max()} block(s)/camera; "
        f"core ownership={core_counts.min()} exactly once"
    )
    print("block  cameras  points3D  parent_fraction")
    rows_by_id = {block_id: (count, fraction) for block_id, count, fraction in point_rows}
    for block in blocks:
        count, fraction = rows_by_id[block.block_id]
        print(f"{block.block_id:>5}  {len(block.camera_indices):>7}  {count:>8}  {fraction:>14.3%}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PartitionerValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
