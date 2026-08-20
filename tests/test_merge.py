"""Tests for Phase 3 manifest integrity, ownership, and PLY artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from large_scene_trainer.core.merge import (
    MergeError,
    crop_is_complete,
    crop_receipt_matches,
    crop_paths,
    load_block_artifacts,
    merge_cropped_plys,
    merge_is_complete,
    merge_receipt_matches,
    merged_paths,
    ownership_ids,
    ownership_mask,
    validate_source_hashes,
    write_crop_receipt,
    write_merge_receipt,
)
from large_scene_trainer.core.types import Block, BlockBox


def _block(block_id: int, minimum: float, maximum: float) -> Block:
    box = BlockBox([minimum, 0, 0], [maximum, 1, 1])
    return Block(block_id, box, box, (block_id,), (block_id, block_id), "test")


def _write_ply(path: Path, x_values: tuple[float, ...]) -> None:
    payload = np.asarray(x_values, dtype="<f4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(x_values)}\nproperty float x\nend_header\n"
        ).encode("ascii")
        + payload
    )


def _dataset(root: Path) -> tuple:
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    source = {}
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = sparse / name
        path.write_bytes(name.encode("ascii"))
        source[f"{name.removesuffix('.bin')}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = []
    for block_id in range(2):
        directory = root / "blocks" / f"block_{block_id:03d}"
        directory.mkdir(parents=True)
        document = {
            "schema": 2,
            "block_id": block_id,
            "run_id": "test",
            "ground_frame": {"version": 1, "R": np.eye(3).tolist(), "origin": [0, 0, 0]},
            "core": {"min_g": [block_id, 0, 0], "max_g": [block_id + 1, 1, 1]},
            "context": {"min_g": [block_id, 0, 0], "max_g": [block_id + 1, 1, 1]},
            "camera_indices": [block_id],
            "frame_span": [block_id, block_id],
            "provenance": "test",
            "source": source,
        }
        (directory / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
        entries.append({"block_id": block_id, "directory": directory.name})
    (root / "blocks" / "index.json").write_text(
        json.dumps({"schema": 1, "blocks": entries}), encoding="utf-8"
    )
    return load_block_artifacts(root)


def test_half_open_ownership_retains_outer_boundary_without_shared_face_duplicates():
    blocks = (_block(0, 0, 1), _block(1, 1, 2))
    points = np.array([[0, 0.5, 0.5], [1, 0.5, 0.5], [2, 0.5, 0.5], [3, 0.5, 0.5]])

    assert ownership_ids(points, blocks).tolist() == [0, 1, 1, -1]
    assert ownership_mask(points, 0, blocks).tolist() == [True, False, False, False]
    assert ownership_mask(points, 1, blocks).tolist() == [False, True, True, False]


def test_manifest_source_mismatch_fails_before_crop(tmp_path):
    _, artifacts = _dataset(tmp_path)
    validate_source_hashes(tmp_path, artifacts)
    (tmp_path / "sparse" / "0" / "images.bin").write_bytes(b"changed")

    with pytest.raises(MergeError, match="does not match parent sparse model"):
        validate_source_hashes(tmp_path, artifacts)


def test_crop_and_merge_receipts_are_content_addressed_and_idempotent(tmp_path):
    _, artifacts = _dataset(tmp_path / "dataset")
    run_dir = tmp_path / "run"
    for artifact in artifacts:
        paths = crop_paths(run_dir, artifact.block.block_id)
        _write_ply(paths.ply, (float(artifact.block.block_id),))
        write_crop_receipt(paths, artifact)
        assert crop_is_complete(paths, artifact)
        assert crop_receipt_matches(paths, artifact)

    output, _ = merged_paths(run_dir)
    merge_cropped_plys([crop_paths(run_dir, item.block.block_id).ply for item in artifacts], output)
    write_merge_receipt(run_dir, artifacts)
    assert merge_is_complete(run_dir, artifacts)
    assert merge_receipt_matches(run_dir, artifacts)
    assert output.read_bytes().count(np.asarray([0.0], dtype="<f4").tobytes()) >= 1

    output.write_bytes(b"tampered")
    assert not merge_is_complete(run_dir, artifacts)
    # The fast status path intentionally avoids reading large PLY payloads.
    assert merge_receipt_matches(run_dir, artifacts)
