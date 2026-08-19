"""Unit coverage for G1 validation script filesystem comparisons."""

from __future__ import annotations

import pytest

import json

from large_scene_trainer.scripts.validate_export import (
    GateError,
    assert_same_tree,
    tree_fingerprint,
    verify_export,
)
from large_scene_trainer.core.export import MIN_POINT_FRACTION, MIN_POINTS_PER_BLOCK


def test_tree_fingerprint_covers_files_and_relative_symlinks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "model.bin").write_bytes(b"model")
        (root / "images").symlink_to("../shared")
    assert tree_fingerprint(first) == tree_fingerprint(second)
    assert_same_tree(first, second)

    (second / "model.bin").write_bytes(b"changed")
    with pytest.raises(GateError, match="model.bin"):
        assert_same_tree(first, second)


def test_verify_export_accepts_parent_images_symlink(tmp_path):
    dataset_root = tmp_path / "dataset"
    shared_images = tmp_path / "shared-images"
    shared_images.mkdir()
    dataset_root.mkdir()
    (dataset_root / "images").symlink_to("../shared-images")
    sparse = dataset_root / "blocks" / "block_000" / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"test")
    block_dir = sparse.parents[1]
    (block_dir / "images").symlink_to("../../images")
    manifest = {
        "ground_frame": {"version": 1},
        "counts": {
            "points3D": MIN_POINTS_PER_BLOCK,
            "points3D_fraction": MIN_POINT_FRACTION,
        },
        "generator": {"points2D_mode": "zeroed"},
        "image_ids": [7],
    }
    (block_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (dataset_root / "blocks" / "index.json").write_text(
        json.dumps({"blocks": [{"directory": "block_000"}]}), encoding="utf-8"
    )
    assert verify_export(dataset_root, {"version": 1}, {7}) == [block_dir]
