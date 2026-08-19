"""Tests for loading a prior export into the viewport preview."""

from __future__ import annotations

import json

import numpy as np
import pytest

from large_scene_trainer.core import manifest
from large_scene_trainer.core.preview_layout import PreviewLayoutError, load_exported_layout
from large_scene_trainer.core.types import Block, BlockBox, GroundFrame


def _block(block_id: int) -> Block:
    core = BlockBox(np.array((block_id, 0.0, 0.0)), np.array((block_id + 1.0, 1.0, 1.0)))
    return Block(
        block_id=block_id,
        core=core,
        context=core.dilated(np.array((0.1, 0.1, 0.1))),
        camera_indices=(block_id,),
        frame_span=(block_id, block_id),
        provenance="test",
    )


def _write_export(root, *, second_frame: GroundFrame | None = None):
    frame = GroundFrame(np.eye(3), np.array((10.0, 20.0, 30.0)))
    blocks_dir = root / "blocks"
    entries = []
    for block_id in range(2):
        block = _block(block_id)
        directory = f"block_{block_id:03d}"
        data = manifest.build(
            block,
            ground_frame=second_frame if block_id == 1 and second_frame else frame,
            run_id="test",
            plugin_version="0.1.0",
            source_scene="synthetic",
        )
        manifest.save(data, blocks_dir / directory)
        entries.append({"block_id": block_id, "directory": directory, "camera_count": 1})
    (blocks_dir / "index.json").write_text(
        json.dumps({"schema": 1, "blocks": entries}), encoding="utf-8"
    )
    return frame


def test_load_exported_layout_recovers_blocks_and_shared_frame(tmp_path):
    expected_frame = _write_export(tmp_path)

    blocks, frame = load_exported_layout(tmp_path)

    assert [block.block_id for block in blocks] == [0, 1]
    assert frame.to_dict() == expected_frame.to_dict()


def test_load_exported_layout_rejects_mixed_ground_frames(tmp_path):
    _write_export(tmp_path, second_frame=GroundFrame(np.eye(3), np.zeros(3)))

    with pytest.raises(PreviewLayoutError, match="share one ground frame"):
        load_exported_layout(tmp_path)


def test_load_exported_layout_requires_an_index(tmp_path):
    with pytest.raises(PreviewLayoutError, match="no block export index"):
        load_exported_layout(tmp_path)
