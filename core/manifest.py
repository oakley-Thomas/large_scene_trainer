"""Per-block manifest: the on-disk record of what a block is.

One manifest is written per block by the Phase 1 exporter and read back by the
Phase 3 merge, which needs `core_box` to crop the trained PLY. Because those two
steps can be separated by a machine boundary and days of wall-clock, the format
carries an explicit `schema` version from the very first commit — migrating a
versioned format is routine, guessing at an unversioned one is not.

Compatibility policy: readers accept any manifest whose schema version they know
and reject the rest loudly. Never silently coerce an unknown version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import AABB, Block

SCHEMA_VERSION = 1
SUPPORTED_SCHEMAS = frozenset({1})

MANIFEST_NAME = "manifest.json"


class ManifestError(ValueError):
    """Raised when a manifest is malformed or of an unsupported schema version."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build(
    block: Block,
    *,
    run_id: str,
    plugin_version: str,
    source_scene: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict for one block. Pure — does not touch the filesystem."""
    return {
        "schema": SCHEMA_VERSION,
        "block_id": block.block_id,
        "run_id": run_id,
        "created": _utc_now(),
        "plugin_version": plugin_version,
        "source_scene": source_scene,
        "core_box": block.core_box.to_dict(),
        "context_box": block.context_box.to_dict(),
        "camera_ids": list(block.camera_ids),
        "camera_count": len(block.camera_ids),
        "params": dict(params or {}),
    }


_REQUIRED_KEYS = ("schema", "block_id", "run_id", "core_box", "context_box", "camera_ids")


def validate(data: Any) -> dict[str, Any]:
    """Check a decoded manifest. Returns it unchanged, or raises ManifestError."""
    if not isinstance(data, dict):
        raise ManifestError(f"manifest must be an object, got {type(data).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ManifestError(f"manifest missing required keys: {', '.join(missing)}")

    schema = data["schema"]
    if schema not in SUPPORTED_SCHEMAS:
        raise ManifestError(
            f"manifest schema {schema!r} unsupported "
            f"(this build reads {sorted(SUPPORTED_SCHEMAS)}) — upgrade the plugin "
            f"or regenerate the block"
        )

    if not isinstance(data["camera_ids"], list):
        raise ManifestError("camera_ids must be a list")
    if not data["camera_ids"]:
        raise ManifestError(f"block {data['block_id']!r} has no cameras assigned")

    # Surfaces inverted or malformed boxes here rather than mid-merge.
    core = AABB.from_dict(data["core_box"])
    context = AABB.from_dict(data["context_box"])
    if not context.intersects(core):
        raise ManifestError(f"block {data['block_id']!r}: context box excludes core box")

    return data


def save(data: dict[str, Any], directory: Path) -> Path:
    """Write `manifest.json` into a block directory. Atomic via temp + replace."""
    validate(data)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def load(path: Path) -> dict[str, Any]:
    """Read and validate a manifest from a file or a block directory."""
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"no manifest at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON — {exc}") from exc
    return validate(data)


def to_block(data: dict[str, Any]) -> Block:
    """Reconstruct the in-memory Block from a validated manifest."""
    validate(data)
    return Block(
        block_id=data["block_id"],
        core_box=AABB.from_dict(data["core_box"]),
        context_box=AABB.from_dict(data["context_box"]),
        camera_ids=[int(i) for i in data["camera_ids"]],
    )
