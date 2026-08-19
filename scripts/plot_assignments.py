#!/usr/bin/env python3
"""Write an offline block-assignment CSV and top-down matplotlib plot.

Example:

    python scripts/plot_assignments.py \
        --dataset-root ~/LichtFeld-Studio/datasets/small_city/camera_calibration/train \
        --output-dir /tmp/small-city-assignment-report

The script reads an existing ``blocks/`` export and never changes it. Install
the optional report dependency first when needed:

    uv pip install --python "$(which python)" 'matplotlib>=3.7'
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from large_scene_trainer.core.assignment_report import write_camera_assignment_csv
from large_scene_trainer.core.colmap_io import load_cameras
from large_scene_trainer.core.preview_layout import load_exported_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _core_owner_indices(blocks, centres_g: np.ndarray) -> np.ndarray:
    owners = np.full(len(centres_g), -1, dtype=np.intp)
    for block in blocks:
        hits = block.core.contains(centres_g)
        if np.any(owners[hits] >= 0):
            raise ValueError("core boxes overlap; cannot assign plot colors uniquely")
        owners[hits] = block.block_id
    if np.any(owners < 0):
        raise ValueError("some camera centres lie outside every core box")
    return owners


def write_assignment_plot(dataset_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create a PNG top-down plot and full camera assignment CSV."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required in this Python environment. Install it with: "
            f"uv pip install --python {sys.executable} 'matplotlib>=3.7'"
        ) from exc

    root = dataset_root.expanduser().resolve()
    blocks, frame = load_exported_layout(root)
    cameras = load_cameras(root / "sparse" / "0")
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in cameras]))
    owners = _core_owner_indices(blocks, centres_g)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_camera_assignment_csv(output_dir / "camera_assignments.csv", blocks)
    figure, axis = plt.subplots(figsize=(12, 9), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for block in blocks:
        color = palette(block.block_id % 10)
        context_extent = block.context.extent
        core_extent = block.core.extent
        axis.add_patch(
            Rectangle(
                block.context.min_g[:2],
                context_extent[0],
                context_extent[1],
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=1.25,
                alpha=0.7,
            )
        )
        axis.add_patch(
            Rectangle(
                block.core.min_g[:2],
                core_extent[0],
                core_extent[1],
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                label=f"Block {block.block_id}",
            )
        )
        points = centres_g[owners == block.block_id]
        axis.scatter(points[:, 0], points[:, 1], s=3, color=color, alpha=0.7)
        centre = (block.core.min_g[:2] + block.core.max_g[:2]) / 2.0
        axis.annotate(str(block.block_id), centre, color=color, ha="center", va="center")

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("ground u (scene units)")
    axis.set_ylabel("ground v (scene units)")
    axis.set_title("Block camera assignment (solid: core, dashed: context)")
    axis.legend(loc="best", title="Core owner")
    png_path = output_dir / "assignment_map.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return png_path, csv_path


def main() -> int:
    args = parse_args()
    png_path, csv_path = write_assignment_plot(args.dataset_root, args.output_dir)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
