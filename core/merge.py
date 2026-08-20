"""Validated artifacts and deterministic ownership for Phase 3 crop/merge.

This module deliberately knows nothing about LichtFeld.  It describes which
trained block owns a world-space Gaussian and guards the on-disk contract used
by the host adapter and GPU worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence
import uuid

import numpy as np

from . import manifest
from .types import Block, GroundFrame


CROP_RECEIPT_SCHEMA = 1
MERGE_RECEIPT_SCHEMA = 1


class MergeError(ValueError):
    """Raised when crop/merge inputs are stale, incomplete, or inconsistent."""


@dataclass(frozen=True)
class BlockArtifact:
    """One block's immutable ownership and provenance inputs."""

    block: Block
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    data: dict[str, Any]


@dataclass(frozen=True)
class CropPaths:
    """Stable output locations for a cropped block and its completion receipt."""

    ply: Path
    receipt: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MergeError(f"cannot read {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MergeError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"{description} must be a JSON object: {path}")
    return value


def load_block_artifacts(dataset_root: str | Path) -> tuple[GroundFrame, tuple[BlockArtifact, ...]]:
    """Load every indexed manifest and require one identical persisted frame."""
    root = Path(dataset_root).expanduser().resolve()
    index = _read_json(root / "blocks" / "index.json", "block index")
    entries = index.get("blocks")
    if index.get("schema") != 1 or not isinstance(entries, list) or not entries:
        raise MergeError("blocks/index.json must be schema 1 with a non-empty blocks list")

    artifacts: list[BlockArtifact] = []
    seen_ids: set[int] = set()
    reference_frame: GroundFrame | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise MergeError("blocks/index.json contains a non-object block entry")
        block_id = entry.get("block_id")
        directory = entry.get("directory")
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0:
            raise MergeError("block index entry has an invalid block_id")
        if block_id in seen_ids:
            raise MergeError(f"blocks/index.json repeats block_id {block_id}")
        seen_ids.add(block_id)
        if not isinstance(directory, str) or not directory:
            raise MergeError(f"block {block_id} index entry has no directory")
        relative = Path(directory)
        if relative.is_absolute() or ".." in relative.parts:
            raise MergeError(f"block {block_id} directory is unsafe")
        block_dir = root / "blocks" / relative
        manifest_path = block_dir / manifest.MANIFEST_NAME
        try:
            data = manifest.load(manifest_path)
            block = manifest.to_block(data)
            frame = manifest.to_ground_frame(data)
        except (OSError, ValueError) as exc:
            raise MergeError(f"invalid manifest for block {block_id}: {exc}") from exc
        if block.block_id != block_id:
            raise MergeError(
                f"block index ID {block_id} disagrees with manifest ID {block.block_id}"
            )
        if reference_frame is None:
            reference_frame = frame
        elif not (
            np.array_equal(reference_frame.R, frame.R)
            and np.array_equal(reference_frame.origin, frame.origin)
        ):
            raise MergeError("block manifests do not share an identical ground frame")
        artifacts.append(
            BlockArtifact(block, block_dir, manifest_path, _sha256(manifest_path), data)
        )
    assert reference_frame is not None
    return reference_frame, tuple(sorted(artifacts, key=lambda item: item.block.block_id))


def validate_source_hashes(dataset_root: str | Path, artifacts: Iterable[BlockArtifact]) -> None:
    """Assert that all manifests were authored from the current parent COLMAP model."""
    sparse_dir = Path(dataset_root).expanduser().resolve() / "sparse" / "0"
    actual = {
        "cameras_sha256": _sha256(sparse_dir / "cameras.bin"),
        "images_sha256": _sha256(sparse_dir / "images.bin"),
        "points3D_sha256": _sha256(sparse_dir / "points3D.bin"),
    }
    for artifact in artifacts:
        source = artifact.data.get("source")
        if not isinstance(source, dict):
            raise MergeError(f"block {artifact.block.block_id} manifest has no source metadata")
        for field, value in actual.items():
            if source.get(field) != value:
                raise MergeError(
                    f"block {artifact.block.block_id} source {field} does not match "
                    "parent sparse model"
                )


def ownership_ids(points_g: np.ndarray, blocks: Sequence[Block]) -> np.ndarray:
    """Return the deterministic owner block ID for each ground-frame point.

    Core boxes use [min, max) containment.  A point on the outer boundary has
    no half-open candidate, so it is assigned to the lowest-ID closed-box
    candidate.  This retains global maxima without reintroducing shared-face
    duplicates.  Overlap is resolved by block ID as a final stable tie-break.
    """
    points = np.asarray(points_g, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise MergeError(f"points_g must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise MergeError("points_g must contain only finite values")
    ordered = tuple(sorted(blocks, key=lambda block: block.block_id))
    if not ordered:
        raise MergeError("cannot assign ownership without blocks")
    owners = np.full(len(points), -1, dtype=np.int64)
    for block in ordered:
        half_open = np.all(
            (points >= block.core.min_g) & (points < block.core.max_g), axis=1
        )
        owners[(owners < 0) & half_open] = block.block_id
    # Only unassigned points can be exterior maxima.  This deliberately never
    # overrides a half-open owner at an internal shared boundary.
    for block in ordered:
        closed = np.all(
            (points >= block.core.min_g) & (points <= block.core.max_g), axis=1
        )
        owners[(owners < 0) & closed] = block.block_id
    return owners


def ownership_mask(points_g: np.ndarray, block_id: int, blocks: Sequence[Block]) -> np.ndarray:
    """Boolean mask selecting the unique splats owned by ``block_id``."""
    return ownership_ids(points_g, blocks) == block_id


def crop_paths(run_dir: str | Path, block_id: int) -> CropPaths:
    directory = Path(run_dir).expanduser().resolve() / "merged" / "crops"
    stem = f"block_{block_id:03d}"
    return CropPaths(directory / f"{stem}.ply", directory / f"{stem}.json")


def write_crop_receipt(paths: CropPaths, artifact: BlockArtifact) -> None:
    if not paths.ply.is_file():
        raise MergeError(f"cropped PLY was not exported: {paths.ply}")
    _write_json_atomically(
        paths.receipt,
        {
            "schema": CROP_RECEIPT_SCHEMA,
            "block_id": artifact.block.block_id,
            "manifest_sha256": artifact.manifest_sha256,
            "ply": paths.ply.name,
            "ply_sha256": _sha256(paths.ply),
        },
    )


def crop_is_complete(paths: CropPaths, artifact: BlockArtifact) -> bool:
    if not paths.ply.is_file() or not paths.receipt.is_file():
        return False
    try:
        receipt = _read_json(paths.receipt, "crop receipt")
        return (
            receipt.get("schema") == CROP_RECEIPT_SCHEMA
            and receipt.get("block_id") == artifact.block.block_id
            and receipt.get("manifest_sha256") == artifact.manifest_sha256
            and receipt.get("ply") == paths.ply.name
            and receipt.get("ply_sha256") == _sha256(paths.ply)
        )
    except (OSError, MergeError):
        return False


def crop_receipt_matches(paths: CropPaths, artifact: BlockArtifact) -> bool:
    """Cheap status-only check that deliberately does not read/hash the PLY."""
    if not paths.ply.is_file() or not paths.receipt.is_file():
        return False
    try:
        receipt = _read_json(paths.receipt, "crop receipt")
        return (
            receipt.get("schema") == CROP_RECEIPT_SCHEMA
            and receipt.get("block_id") == artifact.block.block_id
            and receipt.get("manifest_sha256") == artifact.manifest_sha256
            and receipt.get("ply") == paths.ply.name
        )
    except (OSError, MergeError):
        return False


def require_complete_crops(
    run_dir: str | Path, artifacts: Iterable[BlockArtifact]
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for artifact in artifacts:
        crop = crop_paths(run_dir, artifact.block.block_id)
        if not crop_is_complete(crop, artifact):
            raise MergeError(f"block {artifact.block.block_id} has no valid cropped artifact")
        paths.append(crop.ply)
    return tuple(paths)


def merged_paths(run_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(run_dir).expanduser().resolve() / "merged"
    return directory / "scene.ply", directory / "scene.json"


def write_merge_receipt(run_dir: str | Path, artifacts: Iterable[BlockArtifact]) -> None:
    output, receipt = merged_paths(run_dir)
    if not output.is_file():
        raise MergeError(f"merged PLY was not exported: {output}")
    _write_json_atomically(
        receipt,
        {
            "schema": MERGE_RECEIPT_SCHEMA,
            "ply": output.name,
            "ply_sha256": _sha256(output),
            "blocks": [
                {"block_id": artifact.block.block_id, "manifest_sha256": artifact.manifest_sha256}
                for artifact in artifacts
            ],
        },
    )


def merge_is_complete(run_dir: str | Path, artifacts: Iterable[BlockArtifact]) -> bool:
    artifacts = tuple(artifacts)
    output, receipt_path = merged_paths(run_dir)
    if not output.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path, "merge receipt")
        expected = [
            {"block_id": artifact.block.block_id, "manifest_sha256": artifact.manifest_sha256}
            for artifact in artifacts
        ]
        return (
            receipt.get("schema") == MERGE_RECEIPT_SCHEMA
            and receipt.get("ply") == output.name
            and receipt.get("ply_sha256") == _sha256(output)
            and receipt.get("blocks") == expected
        )
    except (OSError, MergeError):
        return False


def merge_receipt_matches(run_dir: str | Path, artifacts: Iterable[BlockArtifact]) -> bool:
    """Cheap status-only check that deliberately does not read/hash the merged PLY."""
    artifacts = tuple(artifacts)
    output, receipt_path = merged_paths(run_dir)
    if not output.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = _read_json(receipt_path, "merge receipt")
        expected = [
            {"block_id": artifact.block.block_id, "manifest_sha256": artifact.manifest_sha256}
            for artifact in artifacts
        ]
        return (
            receipt.get("schema") == MERGE_RECEIPT_SCHEMA
            and receipt.get("ply") == output.name
            and receipt.get("blocks") == expected
        )
    except (OSError, MergeError):
        return False


_PLY_SCALAR_BYTES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def _read_binary_ply(path: Path) -> tuple[list[bytes], int, bytes]:
    """Read a vertex-only binary PLY without adding a runtime dependency."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise MergeError(f"cannot read cropped PLY: {path}") from exc
    marker = b"end_header\n"
    position = raw.find(marker)
    if position < 0:
        raise MergeError(f"cropped PLY has no end_header: {path}")
    header = raw[: position + len(marker)].splitlines(keepends=True)
    if not (
        header
        and header[0].strip() == b"ply"
        and any(line.startswith(b"format binary_little_endian ") for line in header)
    ):
        raise MergeError(f"cropped PLY is not binary_little_endian: {path}")
    vertex_count: int | None = None
    vertex_properties: list[str] = []
    in_vertex = False
    extra_elements = False
    for line in header[1:]:
        words = line.decode("ascii", errors="strict").strip().split()
        if not words:
            continue
        if words[0] == "element":
            in_vertex = len(words) == 3 and words[1] == "vertex"
            if not in_vertex:
                extra_elements = True
            elif vertex_count is None:
                try:
                    vertex_count = int(words[2])
                except ValueError as exc:
                    raise MergeError(f"cropped PLY has invalid vertex count: {path}") from exc
        elif in_vertex and words[0] == "property":
            if len(words) != 3 or words[1] == "list" or words[1] not in _PLY_SCALAR_BYTES:
                raise MergeError(f"cropped PLY has unsupported vertex property: {path}")
            vertex_properties.append(words[1])
    if vertex_count is None or not vertex_properties or extra_elements:
        raise MergeError(f"cropped PLY must contain one scalar vertex element: {path}")
    stride = sum(_PLY_SCALAR_BYTES[item] for item in vertex_properties)
    payload = raw[position + len(marker) :]
    if len(payload) != vertex_count * stride:
        raise MergeError(f"cropped PLY payload does not match its vertex count: {path}")
    return header, vertex_count, payload


def merge_cropped_plys(inputs: Sequence[Path], output: str | Path) -> Path:
    """Atomically concatenate compatible cropped Gaussian PLY vertex records."""
    if not inputs:
        raise MergeError("cannot merge an empty crop set")
    first_header, _, first_payload = _read_binary_ply(Path(inputs[0]))
    template = []
    for line in first_header:
        if line.startswith(b"element vertex "):
            template.append(b"element vertex {count}\n")
        elif line.startswith(b"comment "):
            # Export timestamps and source comments need not match.
            continue
        else:
            template.append(line)
    total = 0
    payloads = [first_payload]
    _, count, _ = _read_binary_ply(Path(inputs[0]))
    total += count
    expected = b"".join(line for line in template if line != b"element vertex {count}\n")
    for path in inputs[1:]:
        header, count, payload = _read_binary_ply(Path(path))
        normalized = []
        for line in header:
            if line.startswith(b"element vertex "):
                normalized.append(b"element vertex {count}\n")
            elif line.startswith(b"comment "):
                continue
            else:
                normalized.append(line)
        comparable = b"".join(line for line in normalized if line != b"element vertex {count}\n")
        if comparable != expected:
            raise MergeError(f"cropped PLY vertex schemas differ: {path}")
        total += count
        payloads.append(payload)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = [line.replace(b"{count}", str(total).encode("ascii")) for line in template]
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.writelines(rendered)
            for payload in payloads:
                handle.write(payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
