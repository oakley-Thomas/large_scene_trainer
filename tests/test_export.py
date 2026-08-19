"""Synthetic coverage for deterministic, oriented COLMAP block export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from large_scene_trainer.core.camera_assign import (
    AssignmentDiagnostics,
    CameraAssignmentConfig,
    assign_cameras,
)
from large_scene_trainer.core.colmap_io import (
    ColmapCamera,
    ColmapImage,
    load_cameras,
    read_points3D_bin,
    write_cameras_bin,
    write_images_bin,
    write_points3D_bin,
)
from large_scene_trainer.core.export import ExportError, export_blocks, filter_points_for_block
from large_scene_trainer.core.types import (
    Block,
    BlockBox,
    CameraRef,
    GroundFrame,
    Point3D,
    SegmentationConfig,
)


def _frame() -> GroundFrame:
    return GroundFrame(R=np.eye(3), origin=np.array([0.0, 0.0, 0.0]))


def _cameras() -> list[CameraRef]:
    return [
        CameraRef(11, "a.jpg", 7, [-1.0, 0.0, 0.0], np.eye(3), 100.0, 100.0, 640, 480),
        CameraRef(31, "b.jpg", 7, [1.0, 0.0, 0.0], np.eye(3), 100.0, 100.0, 640, 480),
    ]


def _config() -> SegmentationConfig:
    return SegmentationConfig(10.0, 10, 1.0, 0.1, 1.0, 2.0)


def _block() -> Block:
    box = BlockBox(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0]))
    return Block(0, box, box, (0, 1), (0, 1), "synthetic")


def _assignment() -> AssignmentDiagnostics:
    return AssignmentDiagnostics("centre", 2, 1, (2,), 1.0, 1, (), ())


def _make_parent_dataset(root: Path, point_count: int = 1200) -> None:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (root / "images").mkdir()
    write_cameras_bin(
        {7: ColmapCamera(7, 1, 640, 480, np.array([100.0, 100.0, 320.0, 240.0]))},
        sparse / "cameras.bin",
    )
    write_images_bin(
        {
            31: ColmapImage(31, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 7, "b.jpg"),
            11: ColmapImage(11, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 7, "a.jpg"),
        },
        sparse / "images.bin",
    )
    points = {
        point_id: Point3D(
            point_id,
            [0.0, float(point_id % 7) / 10.0, 0.0],
            [point_id % 256, 9, 17],
            0.1,
            ((11, point_id), (31, point_id + 1)),
        )
        for point_id in range(1000, 1000 + point_count)
    }
    write_points3D_bin(points, sparse / "points3D.bin")


def _export(root: Path):
    return export_blocks(
        root, [_block()], _cameras(), _frame(), _config(), _assignment(), git_sha="test"
    )


def _tree(root: Path) -> dict[str, bytes | str]:
    result: dict[str, bytes | str] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = f"link:{path.readlink()}"
        elif path.is_file():
            result[relative] = path.read_bytes()
    return result


def test_export_determinism_symlink_manifest_and_reload(tmp_path):
    first_root = tmp_path / "first" / "dataset"
    second_root = tmp_path / "second" / "dataset"
    _make_parent_dataset(first_root)
    _make_parent_dataset(second_root)
    first = _export(first_root)
    second = _export(second_root)
    assert _tree(first.blocks_dir) == _tree(second.blocks_dir)

    block_dir = first.blocks_dir / "block_000"
    assert block_dir.joinpath("images").readlink() == Path("../../images")
    assert block_dir.joinpath("images").resolve() == first_root / "images"
    data = json.loads((block_dir / "manifest.json").read_text())
    assert data["ground_frame"] == _frame().to_dict()
    assert data["generator"]["points2D_mode"] == "zeroed"
    assert data["counts"]["cameras"] == 2
    assert data["counts"]["camera_models"] == 1
    assert data["counts"]["points3D"] == 1200
    assert len(data["source"]["points3D_sha256"]) == hashlib.sha256().digest_size * 2
    assert json.loads((first.blocks_dir / "index.json").read_text())["blocks"] == [
        {"block_id": 0, "camera_count": 2, "directory": "block_000"}
    ]
    written_cameras = load_cameras(block_dir / "sparse" / "0")
    assert [camera.name for camera in written_cameras] == ["a.jpg", "b.jpg"]
    assert [camera.image_id for camera in written_cameras] == [11, 31]
    assert read_points3D_bin(block_dir / "sparse" / "0" / "points3D.bin")[1000].track == (
        (11, 1000),
        (31, 1001),
    )


def test_point_filter_intersection_and_track_pruning():
    points = {
        1: Point3D(1, [0, 0, 0], [0, 0, 0], 0.0, ((11, 0), (31, 1))),
        2: Point3D(2, [0, 0, 0], [0, 0, 0], 0.0, ((99, 0), (98, 1))),
        3: Point3D(3, [10, 0, 0], [0, 0, 0], 0.0, ((11, 0), (31, 1))),
        4: Point3D(4, [0, 0, 0], [0, 0, 0], 0.0, ((11, 0),)),
    }
    box = BlockBox(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    kept = filter_points_for_block(points, {11, 31}, box, _frame())
    assert list(kept) == [1]


def test_degenerate_block_raises_without_publishing_partial_export(tmp_path):
    root = tmp_path / "dataset"
    _make_parent_dataset(root, point_count=999)
    with pytest.raises(ExportError, match="retained 999"):
        _export(root)
    assert not (root / "blocks").exists()
    assert not list(root.glob(".blocks-export-*"))


def test_existing_export_is_preserved(tmp_path):
    root = tmp_path / "dataset"
    _make_parent_dataset(root)
    (root / "blocks").mkdir()
    marker = root / "blocks" / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ExportError, match="refusing to replace"):
        _export(root)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_camera_assignment_has_complete_union_and_unique_core_ownership():
    cameras = _cameras()
    first_core = BlockBox(np.array([-2.0, -1.0, -1.0]), np.array([0.0, 1.0, 1.0]))
    second_core = BlockBox(np.array([0.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0]))
    blocks = [
        Block(0, first_core, first_core.dilated(np.array([0.5, 0.0, 0.0])), (0,), (0, 0), "test"),
        Block(1, second_core, second_core.dilated(np.array([0.5, 0.0, 0.0])), (1,), (1, 1), "test"),
    ]
    assigned, diagnostics = assign_cameras(
        cameras, blocks, _frame(), CameraAssignmentConfig(min_cameras_per_block=1)
    )
    assert set().union(*(set(block.camera_indices) for block in assigned)) == {0, 1}
    assert not diagnostics.orphaned_camera_indices
    assert not diagnostics.multiply_owned_camera_indices
