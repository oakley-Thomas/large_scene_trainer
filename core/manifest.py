"""Schema-v2 per-block manifests authored in the shared ground frame."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Block, GroundFrame


SCHEMA_VERSION = 2
SUPPORTED_SCHEMAS = frozenset({SCHEMA_VERSION})
MANIFEST_NAME = "manifest.json"


class ManifestError(ValueError):
    """Raised when a manifest is malformed or has an unsupported schema."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build(
    block: Block,
    *,
    ground_frame: GroundFrame,
    run_id: str,
    plugin_version: str,
    source_scene: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one schema-v2 manifest without touching the filesystem."""
    if block.block_id < 0:
        raise ManifestError("cannot serialize a provisional block before final IDs are assigned")
    return {
        "schema": SCHEMA_VERSION,
        "block_id": block.block_id,
        "run_id": run_id,
        "created": _utc_now(),
        "plugin_version": plugin_version,
        "source_scene": source_scene,
        "ground_frame": ground_frame.to_dict(),
        "core": block.core.to_dict(),
        "context": block.context.to_dict(),
        "camera_indices": list(block.camera_indices),
        "frame_span": list(block.frame_span),
        "provenance": block.provenance,
        "camera_count": len(block.camera_indices),
        "params": dict(params or {}),
    }


_REQUIRED_KEYS = (
    "schema",
    "block_id",
    "run_id",
    "ground_frame",
    "core",
    "context",
    "camera_indices",
    "frame_span",
    "provenance",
)


def validate(data: Any) -> dict[str, Any]:
    """Validate a decoded v2 manifest and return it unchanged."""
    if not isinstance(data, dict):
        raise ManifestError(f"manifest must be an object, got {type(data).__name__}")
    schema = data.get("schema")
    if not isinstance(schema, int) or schema not in SUPPORTED_SCHEMAS:
        raise ManifestError(
            f"manifest schema {schema!r} unsupported "
            f"(this build reads {sorted(SUPPORTED_SCHEMAS)}); regenerate the block"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ManifestError(f"manifest missing required keys: {', '.join(missing)}")
    if not isinstance(data["camera_indices"], list) or not data["camera_indices"]:
        raise ManifestError("camera_indices must be a non-empty list")
    try:
        block = Block.from_dict(
            {
                "block_id": data["block_id"],
                "core": data["core"],
                "context": data["context"],
                "camera_indices": data["camera_indices"],
                "frame_span": data["frame_span"],
                "provenance": data["provenance"],
            }
        )
        frame = GroundFrame.from_dict(data["ground_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"malformed schema-v2 manifest: {exc}") from exc
    if block.block_id < 0:
        raise ManifestError("manifest block_id must be a final non-negative ID")
    if data.get("camera_count", len(block.camera_indices)) != len(block.camera_indices):
        raise ManifestError("camera_count does not match camera_indices")
    # Constructing these values is validation; retain local names to make the
    # frame/box contract explicit and prevent accidental world-AABB fallback.
    _ = frame, block
    return data


def save(data: dict[str, Any], directory: Path) -> Path:
    validate(data)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def load(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"no manifest at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON: {exc}") from exc
    return validate(data)


def to_block(data: dict[str, Any]) -> Block:
    validate(data)
    return Block.from_dict(
        {
            "block_id": data["block_id"],
            "core": data["core"],
            "context": data["context"],
            "camera_indices": data["camera_indices"],
            "frame_span": data["frame_span"],
            "provenance": data["provenance"],
        }
    )


def to_ground_frame(data: dict[str, Any]) -> GroundFrame:
    """Reconstruct the frame required to interpret a manifest's boxes."""
    validate(data)
    return GroundFrame.from_dict(data["ground_frame"])
