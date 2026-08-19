"""Host-independent summaries and CSV output for block camera assignment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .types import Block


@dataclass(frozen=True)
class BlockAssignmentRow:
    """Compact one-row summary for a block assignment table."""

    block_id: int
    camera_count: int
    frame_start: int
    frame_end: int
    core_min: tuple[float, float, float]
    core_max: tuple[float, float, float]
    context_min: tuple[float, float, float]
    context_max: tuple[float, float, float]


def block_assignment_rows(blocks: Sequence[Block]) -> tuple[BlockAssignmentRow, ...]:
    """Return deterministic per-block summaries for the panel and reports."""
    return tuple(
        BlockAssignmentRow(
            block_id=block.block_id,
            camera_count=len(block.camera_indices),
            frame_start=block.frame_span[0],
            frame_end=block.frame_span[1],
            core_min=tuple(float(value) for value in block.core.min_g),
            core_max=tuple(float(value) for value in block.core.max_g),
            context_min=tuple(float(value) for value in block.context.min_g),
            context_max=tuple(float(value) for value in block.context.max_g),
        )
        for block in sorted(blocks, key=lambda item: item.block_id)
    )


def camera_assignment_rows(blocks: Sequence[Block]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return each camera index and every context block that contains it."""
    assignments: dict[int, list[int]] = {}
    for block in sorted(blocks, key=lambda item: item.block_id):
        for camera_index in block.camera_indices:
            assignments.setdefault(camera_index, []).append(block.block_id)
    return tuple(
        (camera_index, tuple(block_ids))
        for camera_index, block_ids in sorted(assignments.items())
    )


def write_camera_assignment_csv(path: str | Path, blocks: Sequence[Block]) -> Path:
    """Write the full camera-to-context-block table for offline inspection."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("camera_index", "assigned_block_ids", "blocks_per_camera"))
        for camera_index, block_ids in camera_assignment_rows(blocks):
            writer.writerow((camera_index, ";".join(map(str, block_ids)), len(block_ids)))
    return target
