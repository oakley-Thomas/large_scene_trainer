"""Tests for the read-only partition correctness validation script."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from large_scene_trainer.core import manifest
from large_scene_trainer.core.types import Block, BlockBox, CameraRef, GroundFrame
from large_scene_trainer.scripts.validate_partitioner import (
    PartitionerValidationError,
    count_colmap_points,
    validate_camera_partition,
    validate_point_counts,
)


def _camera(index: int, centre: tuple[float, float, float]) -> CameraRef:
    return CameraRef(
        image_id=index,
        name=f"{index:03d}.jpg",
        camera_id=1,
        C_world=np.array(centre),
        R_world_cam=np.eye(3),
        fx=1.0,
        fy=1.0,
        width=1,
        height=1,
    )


def _block(block_id: int, minimum: float, maximum: float, indices: tuple[int, ...]) -> Block:
    core = BlockBox(np.array((minimum, 0.0, 0.0)), np.array((maximum, 1.0, 1.0)))
    return Block(
        block_id=block_id,
        core=core,
        context=core.dilated(np.array((0.1, 0.1, 0.1))),
        camera_indices=indices,
        frame_span=(indices[0], indices[-1]),
        provenance="test",
    )


def _write_point_count(path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", count))


def test_validate_camera_partition_accepts_complete_unique_cores():
    blocks = (_block(0, 0.0, 1.0, (0,)), _block(1, 1.0, 2.0, (1,)))
    cameras = (_camera(0, (0.5, 0.5, 0.5)), _camera(1, (1.5, 0.5, 0.5)))

    assignments, ownership = validate_camera_partition(
        blocks, cameras, GroundFrame(np.eye(3), np.zeros(3))
    )

    assert assignments.tolist() == [1, 1]
    assert ownership.tolist() == [1, 1]


def test_validate_camera_partition_rejects_unassigned_camera():
    blocks = (_block(0, 0.0, 1.0, (0,)),)
    cameras = (_camera(0, (0.5, 0.5, 0.5)), _camera(1, (1.5, 0.5, 0.5)))

    with pytest.raises(PartitionerValidationError, match="assigned to no block"):
        validate_camera_partition(blocks, cameras, GroundFrame(np.eye(3), np.zeros(3)))


def test_validate_camera_partition_rejects_core_overlap():
    blocks = (_block(0, 0.0, 2.0, (0,)), _block(1, 1.0, 3.0, (0,)))
    cameras = (_camera(0, (1.5, 0.5, 0.5)),)

    with pytest.raises(PartitionerValidationError, match="multiply own"):
        validate_camera_partition(blocks, cameras, GroundFrame(np.eye(3), np.zeros(3)))


def test_validate_point_counts_matches_files_and_manifests(tmp_path):
    root = tmp_path / "dataset"
    _write_point_count(root / "sparse" / "0" / "points3D.bin", 100)
    block = _block(0, 0.0, 1.0, (0,))
    block_dir = root / "blocks" / "block_000"
    _write_point_count(block_dir / "sparse" / "0" / "points3D.bin", 20)
    manifest.save(
        {
            **manifest.build(
                block,
                ground_frame=GroundFrame(np.eye(3), np.zeros(3)),
                run_id="test",
                plugin_version="0.1.0",
                source_scene="synthetic",
            ),
            "counts": {"points3D": 20, "points3D_fraction": 0.2},
        },
        block_dir,
    )

    rows = validate_point_counts(root, (block,), minimum_points=10, minimum_fraction=0.1)

    assert rows == ((0, 20, 0.2),)
    assert count_colmap_points(block_dir / "sparse" / "0" / "points3D.bin") == 20


def test_validate_point_counts_rejects_manifest_mismatch(tmp_path):
    root = tmp_path / "dataset"
    _write_point_count(root / "sparse" / "0" / "points3D.bin", 100)
    block = _block(0, 0.0, 1.0, (0,))
    block_dir = root / "blocks" / "block_000"
    _write_point_count(block_dir / "sparse" / "0" / "points3D.bin", 20)
    manifest.save(
        {
            **manifest.build(
                block,
                ground_frame=GroundFrame(np.eye(3), np.zeros(3)),
                run_id="test",
                plugin_version="0.1.0",
                source_scene="synthetic",
            ),
            "counts": {"points3D": 19, "points3D_fraction": 0.19},
        },
        block_dir,
    )

    with pytest.raises(PartitionerValidationError, match="manifest has 19"):
        validate_point_counts(root, (block,), minimum_points=10, minimum_fraction=0.1)
