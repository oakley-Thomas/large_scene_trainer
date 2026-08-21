#!/usr/bin/env python3
"""Run generated LichtFeld block-training jobs across local NVIDIA GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core.gpu_workers import (
    QueueStore,
    WorkerRunnerError,
    discover_gpu_ids,
    parse_gpu_ids,
    run_workers,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--gpu-ids", help="Comma-separated GPU IDs; default: detect with nvidia-smi"
    )
    parser.add_argument("--trainer-mode", choices=("auto", "container", "native"), default="auto")
    parser.add_argument("--image", help="Container image for container or auto mode")
    parser.add_argument(
        "--lichtfeld-bin", help="Native LichtFeld executable for native or auto mode"
    )
    parser.add_argument("--container-engine", default="docker")
    parser.add_argument("--container-bin", default="LichtFeld-Studio")
    parser.add_argument("--retry-failed", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Validate queue and commands without training"
    )
    mode.add_argument(
        "--status", action="store_true", help="Print existing queue state without changing it"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.status:
            counts = QueueStore(args.run_dir).counts()
            print(" ".join(f"{name}={count}" for name, count in sorted(counts.items())))
            return 1 if counts["failed"] else 0
        gpu_ids = parse_gpu_ids(args.gpu_ids) if args.gpu_ids else discover_gpu_ids()
        summary = run_workers(
            args.dataset_root,
            args.run_dir,
            gpu_ids=gpu_ids,
            trainer_mode=args.trainer_mode,
            image=args.image,
            lichtfeld_bin=args.lichtfeld_bin,
            container_engine=args.container_engine,
            container_bin=args.container_bin,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    except WorkerRunnerError as exc:
        print(f"GPU worker runner failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"mode={summary.mode} "
        + " ".join(f"{name}={count}" for name, count in sorted(summary.counts.items()))
        + f" merge={summary.merge_state}"
    )
    return 0 if summary.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
