"""Read an existing block export for the non-destructive viewport preview."""

from __future__ import annotations

import json
from pathlib import Path

from . import manifest
from .export import BLOCKS_DIRNAME
from .types import Block, GroundFrame


class PreviewLayoutError(ValueError):
    """Raised when an existing block export cannot supply a safe preview."""


def load_exported_layout(dataset_root: str | Path) -> tuple[tuple[Block, ...], GroundFrame]:
    """Load all blocks and their shared ground frame from an existing export.

    The exporter intentionally never overwrites ``blocks/``.  This reader lets
    a later plugin session preview that protected export without re-running it.
    """
    blocks_dir = Path(dataset_root).expanduser().resolve() / BLOCKS_DIRNAME
    index_path = blocks_dir / "index.json"
    if not index_path.is_file():
        raise PreviewLayoutError(f"no block export index at {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreviewLayoutError(f"{index_path}: invalid JSON: {exc}") from exc
    if not isinstance(index, dict) or index.get("schema") != 1:
        raise PreviewLayoutError(f"{index_path}: unsupported block export index")
    entries = index.get("blocks")
    if not isinstance(entries, list) or not entries:
        raise PreviewLayoutError(f"{index_path}: blocks must be a non-empty list")

    blocks: list[Block] = []
    frame: GroundFrame | None = None
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("directory"), str):
            raise PreviewLayoutError(f"{index_path}: malformed block entry")
        directory = Path(entry["directory"])
        if directory.is_absolute() or len(directory.parts) != 1:
            raise PreviewLayoutError(f"{index_path}: block directory must be a direct child")
        data = manifest.load(blocks_dir / directory)
        block = manifest.to_block(data)
        if entry.get("block_id") != block.block_id:
            raise PreviewLayoutError(
                f"{index_path}: block id does not match manifest for {directory}"
            )
        block_frame = manifest.to_ground_frame(data)
        if frame is None:
            frame = block_frame
        elif block_frame.to_dict() != frame.to_dict():
            raise PreviewLayoutError("block manifests do not share one ground frame")
        blocks.append(block)

    blocks.sort(key=lambda block: block.block_id)
    if [block.block_id for block in blocks] != list(range(len(blocks))):
        raise PreviewLayoutError("block IDs must be contiguous and start at zero")
    assert frame is not None
    return tuple(blocks), frame
