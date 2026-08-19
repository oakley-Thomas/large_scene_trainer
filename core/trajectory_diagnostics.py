"""Persist and load the canonical trajectory frame authored by ticket 1.1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .types import GroundFrame, PlaneDiagnostics


TRAJECTORY_DIAGNOSTICS_NAME = "large_scene_trajectory_diagnostics.json"
_SCHEMA_VERSION = 1


class TrajectoryDiagnosticsError(ValueError):
    """Raised when the persisted 1.1 artifact is missing, stale, or malformed."""


@dataclass(frozen=True)
class TrajectoryDiagnosticsArtifact:
    """The canonical frame and diagnostics tied to one COLMAP pose file."""

    frame: GroundFrame
    diagnostics: PlaneDiagnostics
    images_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images_path(dataset_root: Path) -> Path:
    return dataset_root / "sparse" / "0" / "images.bin"


def write_trajectory_diagnostics(
    dataset_root: str | Path, frame: GroundFrame, diagnostics: PlaneDiagnostics
) -> Path:
    """Atomically write the canonical 1.1 artifact for one source dataset."""
    root = Path(dataset_root).expanduser().resolve()
    images_path = _images_path(root)
    if not images_path.is_file():
        raise TrajectoryDiagnosticsError(f"COLMAP poses missing: {images_path}")
    payload = {
        "schema": _SCHEMA_VERSION,
        "source": {"images_sha256": _sha256(images_path)},
        "ground_frame": frame.to_dict(),
        "plane_diagnostics": diagnostics.to_dict(),
    }
    target = root / TRAJECTORY_DIAGNOSTICS_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def load_trajectory_diagnostics(dataset_root: str | Path) -> TrajectoryDiagnosticsArtifact:
    """Load the canonical frame and reject it if the source poses changed."""
    root = Path(dataset_root).expanduser().resolve()
    target = root / TRAJECTORY_DIAGNOSTICS_NAME
    if not target.is_file():
        raise TrajectoryDiagnosticsError(
            f"missing {target}; run Validate dataset or scripts/write_trajectory_diagnostics.py"
        )
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryDiagnosticsError(f"cannot read {target}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA_VERSION:
        raise TrajectoryDiagnosticsError(f"unsupported trajectory diagnostics schema in {target}")
    try:
        recorded_hash = payload["source"]["images_sha256"]
        frame = GroundFrame.from_dict(payload["ground_frame"])
        diagnostics = PlaneDiagnostics.from_dict(payload["plane_diagnostics"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrajectoryDiagnosticsError(
            f"malformed trajectory diagnostics in {target}: {exc}"
        ) from exc
    if not isinstance(recorded_hash, str) or recorded_hash != _sha256(_images_path(root)):
        raise TrajectoryDiagnosticsError(
            "trajectory diagnostics do not match sparse/0/images.bin; rerun Validate dataset"
        )
    return TrajectoryDiagnosticsArtifact(frame, diagnostics, recorded_hash)
