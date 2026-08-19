"""Tests for human-readable assignment report data."""

import csv

import numpy as np

from large_scene_trainer.core.assignment_report import (
    block_assignment_rows,
    camera_assignment_rows,
    write_camera_assignment_csv,
)
from large_scene_trainer.core.types import Block, BlockBox


def _block(block_id: int, camera_indices: tuple[int, ...]) -> Block:
    core = BlockBox(np.array((block_id, 0.0, 0.0)), np.array((block_id + 1.0, 1.0, 1.0)))
    return Block(
        block_id=block_id,
        core=core,
        context=core.dilated(np.array((0.5, 0.5, 0.5))),
        camera_indices=camera_indices,
        frame_span=(camera_indices[0], camera_indices[-1]),
        provenance="test",
    )


def test_assignment_rows_are_sorted_and_retain_context_overlap():
    blocks = (_block(1, (1, 2)), _block(0, (0, 1)))

    summaries = block_assignment_rows(blocks)
    assignments = camera_assignment_rows(blocks)

    assert [row.block_id for row in summaries] == [0, 1]
    assert summaries[0].camera_count == 2
    assert assignments == ((0, (0,)), (1, (0, 1)), (2, (1,)))


def test_write_camera_assignment_csv(tmp_path):
    target = write_camera_assignment_csv(tmp_path / "assignments.csv", (_block(0, (2, 3)),))

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["camera_index", "assigned_block_ids", "blocks_per_camera"],
        ["2", "0", "1"],
        ["3", "0", "1"],
    ]
