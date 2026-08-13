"""Schema-v2 manifest contract tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from large_scene_trainer.core import manifest
from large_scene_trainer.core.types import Block, BlockBox, GroundFrame


def make_block(block_id: int = 0) -> Block:
    core = BlockBox(min_g=np.array([0.0, 0.0, -2.0]), max_g=np.array([50.0, 50.0, 20.0]))
    return Block(
        block_id=block_id,
        core=core,
        context=core.dilated(np.array([10.0, 10.0, 0.0])),
        camera_indices=(1, 2, 3),
        frame_span=(1, 3),
        provenance="sequential",
    )


def make_frame() -> GroundFrame:
    return GroundFrame(R=np.eye(3), origin=np.array([10.0, 20.0, 30.0]))


def build_one(block: Block | None = None) -> dict:
    return manifest.build(
        block or make_block(),
        ground_frame=make_frame(),
        run_id="smallcity_test",
        plugin_version="0.1.0",
        source_scene="smallcity/segment_a",
        params={"core_extent_su": 50.0, "context_margin_frac": 0.1},
    )


def test_round_trip_through_disk(tmp_path):
    data = build_one()
    path = manifest.save(data, tmp_path / "block_000")
    assert manifest.load(path) == data
    assert manifest.load(tmp_path / "block_000") == data


def test_to_block_and_ground_frame_recover_geometry():
    block = make_block()
    data = build_one(block)
    assert manifest.to_block(data).to_dict() == block.to_dict()
    assert manifest.to_ground_frame(data).to_dict() == make_frame().to_dict()


def test_schema_v1_is_rejected_loudly():
    data = build_one()
    data["schema"] = 1
    with pytest.raises(manifest.ManifestError, match="regenerate"):
        manifest.validate(data)
    with pytest.raises(manifest.ManifestError, match="schema 1"):
        manifest.validate({"schema": 1, "core_box": {}})


def test_missing_keys_are_reported_together():
    data = build_one()
    del data["core"]
    del data["run_id"]
    with pytest.raises(manifest.ManifestError) as exc:
        manifest.validate(data)
    assert "core" in str(exc.value)
    assert "run_id" in str(exc.value)


def test_block_with_no_cameras_is_rejected():
    data = build_one()
    data["camera_indices"] = []
    with pytest.raises(manifest.ManifestError, match="camera_indices"):
        manifest.validate(data)


def test_corrupt_json_names_the_file(tmp_path):
    path = tmp_path / manifest.MANIFEST_NAME
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="invalid JSON"):
        manifest.load(path)


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    manifest.save(build_one(), tmp_path)
    assert json.loads((tmp_path / manifest.MANIFEST_NAME).read_text())
    assert not list(tmp_path.glob("*.tmp"))


def test_inverted_box_is_caught_at_construction():
    with pytest.raises(ValueError, match="inverted"):
        BlockBox(min_g=np.zeros(3), max_g=np.array([-1.0, 5.0, 5.0]))
