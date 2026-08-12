"""Manifest contract tests — runnable with plain pytest, no GUI, no CUDA."""

import json

import pytest

from large_scene_trainer.core import manifest
from large_scene_trainer.core.types import AABB, Block


def make_block(block_id: str = "block_000") -> Block:
    core = AABB(min=(0.0, 0.0, 0.0), max=(50.0, 50.0, 20.0))
    return Block(
        block_id=block_id,
        core_box=core,
        context_box=core.expanded(10.0),
        camera_ids=[1, 2, 3],
    )


def build_one(block: Block | None = None) -> dict:
    return manifest.build(
        block or make_block(),
        run_id="smallcity_test",
        plugin_version="0.1.0",
        source_scene="smallcity/segment_a",
        params={"core_extent_m": 50.0, "overlap_m": 10.0},
    )


def test_round_trip_through_disk(tmp_path):
    data = build_one()
    path = manifest.save(data, tmp_path / "block_000")

    loaded = manifest.load(path)
    assert loaded == data

    # A block directory is also a valid argument.
    assert manifest.load(tmp_path / "block_000") == data


def test_to_block_recovers_geometry():
    block = make_block()
    recovered = manifest.to_block(build_one(block))

    assert recovered.block_id == block.block_id
    assert recovered.core_box == block.core_box
    assert recovered.camera_ids == block.camera_ids


def test_unknown_schema_is_rejected_loudly():
    data = build_one()
    data["schema"] = 99

    with pytest.raises(manifest.ManifestError, match="unsupported"):
        manifest.validate(data)


def test_missing_keys_are_reported_together():
    data = build_one()
    del data["core_box"]
    del data["run_id"]

    with pytest.raises(manifest.ManifestError) as exc:
        manifest.validate(data)
    assert "core_box" in str(exc.value)
    assert "run_id" in str(exc.value)


def test_block_with_no_cameras_is_rejected():
    """An empty block trains on nothing and silently produces a hole in the merge."""
    data = build_one()
    data["camera_ids"] = []

    with pytest.raises(manifest.ManifestError, match="no cameras"):
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
        AABB(min=(0.0, 0.0, 0.0), max=(-1.0, 5.0, 5.0))
