#!/usr/bin/env python3
"""Export and validate a COLMAP block dataset through the G1 loader gate.

Run from the plugin checkout, for example:

    python scripts/validate_export.py \
        --dataset-root ~/LichtFeld-Studio/datasets/small_city/camera_calibration/train \
        --lichtfeld-bin ~/LichtFeld-Studio/build/LichtFeld-Studio

The exporter writes ``<dataset-root>/blocks``. Logs and trainer checkpoints
live under ``--output-dir`` so validation artifacts do not enter the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from typing import Iterable

import numpy as np


# The checkout directory is the package; Python needs its parent on sys.path
# when this file is launched directly as ``python scripts/validate_export.py``.
REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core.camera_assign import CameraAssignmentConfig
from large_scene_trainer.core.colmap_io import load_cameras
from large_scene_trainer.core.export import (
    BLOCKS_DIRNAME,
    MIN_POINT_FRACTION,
    MIN_POINTS_PER_BLOCK,
    ExportError,
    export_dataset,
)
from large_scene_trainer.core.partition import config_from_diagnostics
from large_scene_trainer.core.trajectory_diagnostics import load_trajectory_diagnostics


class GateError(RuntimeError):
    """Raised when an exported block fails a G1 acceptance check."""


def _ground_frames_match(actual: object, expected: dict) -> bool:
    """Compare persisted frame values, allowing JSON float round-off in old exports."""
    if not isinstance(actual, dict):
        return False
    if set(expected) <= {"version"}:
        return actual == expected
    try:
        return (
            actual.get("version") == expected.get("version")
            and np.allclose(actual["R"], expected["R"], rtol=0.0, atol=1e-12)
            and np.allclose(actual["origin"], expected["origin"], rtol=0.0, atol=1e-12)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> dict[str, tuple[str, str]]:
    """Return a deterministic fingerprint for files and relative symlinks."""
    fingerprint: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            fingerprint[relative] = ("symlink", str(path.readlink()))
        elif path.is_file():
            fingerprint[relative] = ("file", _sha256(path))
    return fingerprint


def assert_same_tree(first: Path, second: Path) -> None:
    """Raise with a concise diagnostic when two exports differ."""
    first_tree = tree_fingerprint(first)
    second_tree = tree_fingerprint(second)
    if first_tree == second_tree:
        return
    differing = sorted(
        relative
        for relative in set(first_tree) | set(second_tree)
        if first_tree.get(relative) != second_tree.get(relative)
    )
    raise GateError(
        f"exports are not byte-identical; first differing path: {differing[0]!r} "
        f"({len(differing)} path(s) differ)"
    )


def _load_manifests(blocks_dir: Path) -> list[tuple[Path, dict]]:
    index_path = blocks_dir / "index.json"
    if not index_path.is_file():
        raise GateError(f"missing block index: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid block index: {exc}") from exc
    entries = index.get("blocks")
    if not isinstance(entries, list) or not entries:
        raise GateError("blocks/index.json must contain a non-empty blocks list")
    manifests: list[tuple[Path, dict]] = []
    for entry in entries:
        directory = entry.get("directory")
        if not isinstance(directory, str):
            raise GateError("block index entry has no directory")
        block_dir = blocks_dir / directory
        try:
            manifest = json.loads((block_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError(f"cannot read manifest for {directory}: {exc}") from exc
        manifests.append((block_dir, manifest))
    return manifests


def verify_export(
    dataset_root: Path, expected_frame: dict, parent_image_ids: set[int]
) -> list[Path]:
    """Check export layout and manifest acceptance criteria before training."""
    blocks_dir = dataset_root / BLOCKS_DIRNAME
    manifests = _load_manifests(blocks_dir)
    exported_image_ids: set[int] = set()
    block_dirs: list[Path] = []
    for block_dir, manifest in manifests:
        sparse_dir = block_dir / "sparse" / "0"
        required = [
            sparse_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise GateError(f"block {block_dir.name} is missing: {', '.join(missing)}")
        images_link = block_dir / "images"
        if not images_link.is_symlink() or images_link.readlink().is_absolute():
            raise GateError(f"block {block_dir.name} images must be a relative symlink")
        if images_link.resolve() != (dataset_root / "images").resolve():
            raise GateError(
                f"block {block_dir.name} images symlink does not resolve to parent images"
            )
        if not _ground_frames_match(manifest.get("ground_frame"), expected_frame):
            raise GateError(f"block {block_dir.name} ground frame differs from the source frame")
        counts = manifest.get("counts", {})
        point_count = counts.get("points3D")
        point_fraction = counts.get("points3D_fraction")
        if not isinstance(point_count, int) or point_count < MIN_POINTS_PER_BLOCK:
            raise GateError(f"block {block_dir.name} fails the {MIN_POINTS_PER_BLOCK}-point guard")
        if not isinstance(point_fraction, (int, float)) or point_fraction < MIN_POINT_FRACTION:
            raise GateError(
                f"block {block_dir.name} fails the {MIN_POINT_FRACTION:.2%} point-fraction guard"
            )
        if manifest.get("generator", {}).get("points2D_mode") != "zeroed":
            raise GateError(f"block {block_dir.name} did not record zeroed points2D mode")
        image_ids = manifest.get("image_ids")
        if not isinstance(image_ids, list):
            raise GateError(f"block {block_dir.name} manifest has no image_ids list")
        exported_image_ids.update(image_ids)
        block_dirs.append(block_dir)
    if exported_image_ids != parent_image_ids:
        missing = sorted(parent_image_ids - exported_image_ids)
        extra = sorted(exported_image_ids - parent_image_ids)
        raise GateError(
            "block image union differs from source images; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return block_dirs


def run_headless_checks(
    blocks: Iterable[Path], lichtfeld_bin: Path, output_dir: Path, iterations: int
) -> None:
    """Train each block to the requested iteration count, retaining a log per block."""
    for block_dir in blocks:
        block_output = output_dir / block_dir.name
        block_output.mkdir(parents=True, exist_ok=False)
        log_path = output_dir / f"{block_dir.name}.log"
        command = [
            str(lichtfeld_bin),
            "--headless",
            "--train",
            "--data-path",
            str(block_dir),
            "--output-path",
            str(block_output),
            "--iter",
            str(iterations),
            "--log-level",
            "info",
        ]
        print(f"G1 training check: {block_dir.name}")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise GateError(f"{block_dir.name} failed; inspect {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Training completed successfully" not in log_text:
            raise GateError(
                f"{block_dir.name} returned zero without success marker; inspect {log_path}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--lichtfeld-bin", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="Logs and checkpoint outputs (outside dataset root)"
    )
    parser.add_argument("--target-blocks", type=int, default=4)
    parser.add_argument("--min-cameras-per-block", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--skip-export", action="store_true", help="Validate existing blocks instead"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Do not run headless LichtFeld checks"
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Re-export once and compare; preserves a backup if the exports differ",
    )
    args = parser.parse_args()
    if args.target_blocks < 1 or args.min_cameras_per_block < 1 or args.iterations < 1:
        parser.error("target blocks, minimum cameras, and iterations must be positive")
    if args.verify_determinism and args.skip_export:
        parser.error("--verify-determinism cannot be combined with --skip-export")
    return args


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    lichtfeld_bin = args.lichtfeld_bin.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_root.parent / f"{dataset_root.name}-g1-validation"
    )
    if not lichtfeld_bin.is_file():
        raise GateError(f"LichtFeld executable does not exist: {lichtfeld_bin}")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise GateError("output directory must be outside the dataset root")
    output_dir.mkdir(parents=True, exist_ok=False)

    cameras = load_cameras(dataset_root / "sparse" / "0")
    trajectory = load_trajectory_diagnostics(dataset_root)
    frame = trajectory.frame
    diagnostics = trajectory.diagnostics
    config = config_from_diagnostics(diagnostics, target_blocks=args.target_blocks)
    assignment = CameraAssignmentConfig(
        predicate="centre", min_cameras_per_block=args.min_cameras_per_block
    )
    blocks_dir = dataset_root / BLOCKS_DIRNAME
    prior_export_backup: Path | None = None
    try:
        if args.verify_determinism and blocks_dir.exists():
            prior_export_backup = dataset_root / f"{BLOCKS_DIRNAME}.pre-g1-{uuid.uuid4().hex}"
            blocks_dir.replace(prior_export_backup)
            print(f"Saved prior export before deterministic passes: {prior_export_backup}")
        if not args.skip_export:
            print(f"Exporting {args.target_blocks}-target block dataset to {blocks_dir}")
            export_dataset(dataset_root, config, assignment)
        block_dirs = verify_export(
            dataset_root, frame.to_dict(), {camera.image_id for camera in cameras}
        )
        if args.verify_determinism:
            backup_dir = dataset_root / f"{BLOCKS_DIRNAME}.determinism-{uuid.uuid4().hex}"
            blocks_dir.replace(backup_dir)
            try:
                export_dataset(dataset_root, config, assignment)
                assert_same_tree(backup_dir, blocks_dir)
            except Exception:
                print(f"Determinism backup retained for inspection: {backup_dir}", file=sys.stderr)
                raise
            else:
                shutil.rmtree(backup_dir)
                if prior_export_backup is not None:
                    shutil.rmtree(prior_export_backup)
                print("Determinism check passed")
            block_dirs = verify_export(
                dataset_root, frame.to_dict(), {camera.image_id for camera in cameras}
            )
        if not args.skip_training:
            run_headless_checks(block_dirs, lichtfeld_bin, output_dir, args.iterations)
        print(f"G1 validation passed; logs and outputs: {output_dir}")
        return 0
    except (ExportError, GateError, OSError, ValueError) as exc:
        print(f"G1 validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
