"""Deterministic export of settled blocks as ordinary COLMAP datasets."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Sequence

import numpy as np

from . import manifest
from .camera_assign import AssignmentDiagnostics, CameraAssignmentConfig, assign_cameras
from .colmap_io import (
    ColmapImage,
    load_cameras,
    read_cameras_bin,
    read_images_bin,
    read_points3D_bin,
    write_cameras_bin,
    write_images_bin,
    write_points3D_bin,
)
from .partition import partition_trajectory
from .trajectory_diagnostics import load_trajectory_diagnostics
from .types import Block, BlockBox, CameraRef, GroundFrame, Point3D, SegmentationConfig


MIN_POINTS_PER_BLOCK = 1000
MIN_POINT_FRACTION = 0.005
BLOCKS_DIRNAME = "blocks"


class ExportError(ValueError):
    """Raised when an input dataset or requested block export is invalid."""


@dataclass(frozen=True)
class BlockExportSummary:
    block_id: int
    directory: Path
    cameras: int
    images: int
    points3D: int
    points3D_parent: int


@dataclass(frozen=True)
class ExportSummary:
    dataset_root: Path
    blocks_dir: Path
    blocks: tuple[BlockExportSummary, ...]
    ground_frame: GroundFrame
    blocks_definition: tuple[Block, ...]
    assignment: AssignmentDiagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _validate_dataset_root(dataset_root: str | Path) -> tuple[Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    sparse_dir = root / "sparse" / "0"
    if not root.is_dir():
        raise ExportError(f"dataset root does not exist: {root}")
    if not (root / "images").is_dir():
        raise ExportError(f"dataset root has no images directory: {root / 'images'}")
    missing = [
        name
        for name in ("cameras.bin", "images.bin", "points3D.bin")
        if not (sparse_dir / name).is_file()
    ]
    if missing:
        raise ExportError(f"sparse model missing {', '.join(missing)} under {sparse_dir}")
    blocks_dir = root / BLOCKS_DIRNAME
    if blocks_dir.exists() or blocks_dir.is_symlink():
        raise ExportError(f"refusing to replace existing export directory: {blocks_dir}")
    return root, sparse_dir


def filter_points_for_block(
    points: dict[int, Point3D],
    retained_image_ids: set[int],
    context: BlockBox,
    frame: GroundFrame,
) -> dict[int, Point3D]:
    """Return points in ``context`` with at least two retained observations."""
    if not points:
        return {}
    point_ids = sorted(points)
    xyz_world = np.stack([points[point_id].xyz for point_id in point_ids])
    in_context = context.contains(frame.to_ground(xyz_world))
    kept: dict[int, Point3D] = {}
    for point_id, is_in_context in zip(point_ids, in_context):
        if not is_in_context:
            continue
        point = points[point_id]
        track = tuple(item for item in point.track if item[0] in retained_image_ids)
        if len(track) >= 2:
            kept[point_id] = replace(point, track=track)
    return kept


def _subset_images(
    cameras: Sequence[CameraRef], block: Block, images: dict[int, ColmapImage]
) -> dict[int, ColmapImage]:
    selected: dict[int, ColmapImage] = {}
    for index in block.camera_indices:
        try:
            image_id = cameras[index].image_id
            image = images[image_id]
        except IndexError as exc:
            raise ExportError(f"block {block.block_id} references camera index {index}") from exc
        except KeyError as exc:
            raise ExportError(
                f"block {block.block_id} camera image id {image_id} is absent from images.bin"
            ) from exc
        # The selected loader branch ignores points2D.  Deliberately write empty
        # arrays while retaining points3D tracks for point filtering and provenance.
        selected[image_id] = replace(image, points2D=())
    return selected


def _manifest_data(
    *,
    block: Block,
    frame: GroundFrame,
    segmentation_config: SegmentationConfig,
    assignment: AssignmentDiagnostics,
    source_hashes: dict[str, str],
    parent_point_count: int,
    images: dict[int, ColmapImage],
    cameras_count: int,
    points_count: int,
    source_scene: str,
    git_sha: str,
) -> dict:
    data = manifest.build(
        block,
        ground_frame=frame,
        run_id="colmap_block_export",
        plugin_version="0.1.0",
        source_scene=source_scene,
        params=dataclasses.asdict(segmentation_config),
        include_created=False,
    )
    image_ids = sorted(images)
    data.update(
        {
            "segmentation_config": dataclasses.asdict(segmentation_config),
            "assignment": assignment.to_dict(),
            "source": {
                "sparse_dir": "../../sparse/0",
                "images_dir": "../../images",
                **source_hashes,
            },
            "counts": {
                "cameras": cameras_count,
                "images": len(images),
                "points3D": points_count,
                "points3D_parent": parent_point_count,
                "points3D_fraction": points_count / parent_point_count,
            },
            "image_ids": image_ids,
            "image_names": [images[image_id].name for image_id in image_ids],
            "generator": {"git_sha": git_sha, "points2D_mode": "zeroed"},
        }
    )
    return data


def export_blocks(
    dataset_root: str | Path,
    blocks: Sequence[Block],
    cameras: Sequence[CameraRef],
    frame: GroundFrame,
    segmentation_config: SegmentationConfig,
    assignment: AssignmentDiagnostics,
    *,
    git_sha: str = "unknown",
) -> ExportSummary:
    """Export already-partitioned blocks below the parent dataset root.

    The function is host-independent and leaves no visible ``blocks/`` directory
    if validation or any block guard fails.
    """
    root, sparse_dir = _validate_dataset_root(dataset_root)
    ordered_cameras = tuple(sorted(cameras, key=lambda camera: camera.name))
    if not blocks:
        raise ExportError("no blocks supplied for export")
    ordered_blocks = tuple(sorted(blocks, key=lambda block: block.block_id))
    if [block.block_id for block in ordered_blocks] != list(range(len(ordered_blocks))):
        raise ExportError("block IDs must be contiguous and start at zero")

    parent_cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    parent_images = read_images_bin(sparse_dir / "images.bin")
    parent_points = read_points3D_bin(sparse_dir / "points3D.bin")
    if not parent_points:
        raise ExportError("parent points3D.bin is empty")
    source_hashes = {
        "cameras_sha256": _sha256(sparse_dir / "cameras.bin"),
        "images_sha256": _sha256(sparse_dir / "images.bin"),
        "points3D_sha256": _sha256(sparse_dir / "points3D.bin"),
    }

    stage = Path(tempfile.mkdtemp(prefix=".blocks-export-", dir=root))
    summaries: list[BlockExportSummary] = []
    try:
        index_blocks: list[dict] = []
        for block in ordered_blocks:
            block_dir = stage / f"block_{block.block_id:03d}"
            output_sparse = block_dir / "sparse" / "0"
            output_sparse.mkdir(parents=True)
            selected_images = _subset_images(ordered_cameras, block, parent_images)
            retained_image_ids = set(selected_images)
            selected_cameras = {
                camera_id: parent_cameras[camera_id]
                for camera_id in sorted({image.camera_id for image in selected_images.values()})
            }
            selected_points = filter_points_for_block(
                parent_points, retained_image_ids, block.context, frame
            )
            point_fraction = len(selected_points) / len(parent_points)
            if len(selected_points) < MIN_POINTS_PER_BLOCK or point_fraction < MIN_POINT_FRACTION:
                raise ExportError(
                    f"block {block.block_id} retained {len(selected_points)} / "
                    f"{len(parent_points)} "
                    f"points ({point_fraction:.4%}); need at least {MIN_POINTS_PER_BLOCK} and "
                    f"{MIN_POINT_FRACTION:.2%}"
                )
            write_cameras_bin(selected_cameras, output_sparse / "cameras.bin")
            write_images_bin(selected_images, output_sparse / "images.bin")
            write_points3D_bin(selected_points, output_sparse / "points3D.bin")
            # ``block_NNN`` is two levels below the dataset root: blocks/block_NNN.
            (block_dir / "images").symlink_to("../../images", target_is_directory=True)
            data = _manifest_data(
                block=block,
                frame=frame,
                segmentation_config=segmentation_config,
                assignment=assignment,
                source_hashes=source_hashes,
                parent_point_count=len(parent_points),
                images=selected_images,
                cameras_count=len(selected_cameras),
                points_count=len(selected_points),
                source_scene=root.name,
                git_sha=git_sha,
            )
            manifest.save(data, block_dir)
            summaries.append(
                BlockExportSummary(
                    block.block_id,
                    Path(f"block_{block.block_id:03d}"),
                    len(selected_cameras),
                    len(selected_images),
                    len(selected_points),
                    len(parent_points),
                )
            )
            index_blocks.append(
                {
                    "block_id": block.block_id,
                    "directory": f"block_{block.block_id:03d}",
                    "camera_count": len(selected_images),
                }
            )
        _write_json(stage / "index.json", {"schema": 1, "blocks": index_blocks})
        stage.replace(root / BLOCKS_DIRNAME)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return ExportSummary(
        dataset_root=root,
        blocks_dir=root / BLOCKS_DIRNAME,
        blocks=tuple(summaries),
        ground_frame=frame,
        blocks_definition=ordered_blocks,
        assignment=assignment,
    )


def export_dataset(
    dataset_root: str | Path,
    segmentation_config: SegmentationConfig,
    assignment_config: CameraAssignmentConfig,
    *,
    git_sha: str = "unknown",
) -> ExportSummary:
    """Build and export blocks directly from a parent COLMAP dataset."""
    root, sparse_dir = _validate_dataset_root(dataset_root)
    cameras = load_cameras(sparse_dir)
    trajectory = load_trajectory_diagnostics(root)
    frame = trajectory.frame
    plane_diagnostics = trajectory.diagnostics
    blocks, _ = partition_trajectory(cameras, frame, plane_diagnostics, segmentation_config)
    assigned_blocks, assignment = assign_cameras(cameras, blocks, frame, assignment_config)
    return export_blocks(
        root,
        assigned_blocks,
        cameras,
        frame,
        segmentation_config,
        assignment,
        git_sha=git_sha,
    )
