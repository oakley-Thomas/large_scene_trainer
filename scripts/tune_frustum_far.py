#!/usr/bin/env python3
"""Measure DT.1.3 frustum-camera duplication without exporting a dataset.

Example:

    python scripts/tune_frustum_far.py \
        --dataset-root /path/to/small_city/camera_calibration/train

The script only reads the source COLMAP model. It reports one row per far-plane
candidate and recommends the largest candidate satisfying the DT.1.3 gate:
mean blocks per camera no greater than 2.0 and every block meeting the requested
minimum camera count.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


# The checkout directory is the package; direct script execution needs its
# parent directory on sys.path, just like validate_export.py.
REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core.camera_assign import (
    assign_cameras,
    frustum_config_from_diagnostics,
)
from large_scene_trainer.core.colmap_io import load_cameras
from large_scene_trainer.core.partition import config_from_diagnostics, partition_trajectory
from large_scene_trainer.core.trajectory_diagnostics import load_trajectory_diagnostics


# Start below the smallest observed block extent, then extend through the
# original coarse range. The lower candidates matter when a long trajectory
# produces a large segmentation limit but compact spatial blocks.
DEFAULT_MULTIPLIERS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--target-blocks", type=int, default=4)
    parser.add_argument("--minimum-cameras", type=int, default=32)
    parser.add_argument(
        "--far-multiplier",
        type=float,
        action="append",
        dest="far_multipliers",
        help="Repeat to override the default sweep multipliers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_cameras < 1:
        raise ValueError("--minimum-cameras must be positive")
    multipliers = tuple(args.far_multipliers or DEFAULT_MULTIPLIERS)
    if not multipliers or any(multiplier <= 0.0 for multiplier in multipliers):
        raise ValueError("far multipliers must be positive")

    sparse_dir = args.dataset_root.expanduser().resolve() / "sparse" / "0"
    cameras = load_cameras(sparse_dir)
    trajectory = load_trajectory_diagnostics(args.dataset_root)
    frame = trajectory.frame
    plane_diagnostics = trajectory.diagnostics
    segmentation_config = config_from_diagnostics(
        plane_diagnostics, target_blocks=args.target_blocks
    )
    blocks, _ = partition_trajectory(cameras, frame, plane_diagnostics, segmentation_config)

    print(f"core_extent_su: {segmentation_config.core_extent_su:.9g}")
    print(f"median_inter_frame_spacing: {plane_diagnostics.median_inter_frame_spacing:.9g}")
    print(f"blocks before assignment: {len(blocks)}")
    print("far_su  cameras/block  mean_blocks/camera  max  min  orphans  multi  seconds  gate")

    passing: list[float] = []
    for multiplier in sorted(set(multipliers)):
        far_su = multiplier * segmentation_config.core_extent_su
        config = frustum_config_from_diagnostics(
            plane_diagnostics,
            frustum_far_su=far_su,
            # This is a measurement tool: report undersized blocks rather than
            # raising before their counts can be compared.
            min_cameras_per_block=0,
        )
        started = time.perf_counter()
        _, diagnostics = assign_cameras(cameras, blocks, frame, config)
        elapsed = time.perf_counter() - started
        min_count = min(diagnostics.cameras_per_block)
        gate = (
            diagnostics.mean_blocks_per_camera <= 2.0
            and min_count >= args.minimum_cameras
            and not diagnostics.orphaned_camera_indices
            and not diagnostics.multiply_owned_camera_indices
        )
        if gate:
            passing.append(far_su)
        counts = ",".join(str(count) for count in diagnostics.cameras_per_block)
        print(
            f"{far_su:.9g}  {counts}  {diagnostics.mean_blocks_per_camera:.6g}  "
            f"{diagnostics.max_blocks_per_camera}  {min_count}  "
            f"{len(diagnostics.orphaned_camera_indices)}  "
            f"{len(diagnostics.multiply_owned_camera_indices)}  {elapsed:.3f}  "
            f"{'pass' if gate else 'fail'}"
        )

    if passing:
        print(f"recommended frustum_far_su: {max(passing):.9g}")
    else:
        print("no candidate passed; retain CameraAssignmentConfig(predicate='centre')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
