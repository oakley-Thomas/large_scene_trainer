#!/usr/bin/env python3
"""Fit and persist the canonical ticket-1.1 trajectory diagnostics artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core.colmap_io import load_cameras
from large_scene_trainer.core.trajectory_diagnostics import write_trajectory_diagnostics
from large_scene_trainer.core.trajectory_plane import fit_trajectory_plane


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    frame, diagnostics = fit_trajectory_plane(load_cameras(root / "sparse" / "0"))
    target = write_trajectory_diagnostics(root, frame, diagnostics)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
