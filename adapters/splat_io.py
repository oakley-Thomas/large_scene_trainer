"""LichtFeld-only Phase 3 splat crop and inspection helpers.

The training callback is intentionally tiny: it registers during trainer
startup, obtains the scene only at ``on_training_end``, and delegates all
geometry/provenance decisions to :mod:`core.merge`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import lichtfeld as lf
import numpy as np

from ..core.merge import (
    MergeError,
    crop_paths,
    load_block_artifacts,
    ownership_mask,
    validate_source_hashes,
    write_crop_receipt,
)


class SplatIOError(RuntimeError):
    """Raised when the host's splat API cannot perform a requested action."""


def world_positions(means_raw, world_transform) -> np.ndarray:
    """Convert node-local means through LichtFeld's column-major transform."""
    means = np.asarray(means_raw, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise SplatIOError(f"means_raw must have shape (N, 3), got {means.shape}")
    matrix = np.asarray(world_transform, dtype=np.float64)
    if matrix.size != 16:
        raise SplatIOError("world_transform must contain 16 values")
    matrix = matrix.reshape((4, 4), order="F")
    homogeneous = np.concatenate((means, np.ones((len(means), 1))), axis=1)
    transformed = homogeneous @ matrix.T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise SplatIOError("world_transform produced zero homogeneous coordinates")
    return transformed[:, :3] / transformed[:, 3:4]


def _splat_nodes(scene) -> list:
    nodes = []
    for node in scene.get_nodes():
        try:
            if node.splat_data() is not None:
                nodes.append(node)
        except (AttributeError, RuntimeError):
            continue
    return nodes


def _only_trained_splat(scene):
    nodes = _splat_nodes(scene)
    if len(nodes) != 1:
        raise SplatIOError(f"expected exactly one trained splat node, found {len(nodes)}")
    return nodes[0]


def _soft_delete(node, deleted: np.ndarray) -> None:
    tensor = lf.Tensor.from_numpy(np.ascontiguousarray(deleted, dtype=np.bool_)).cuda()
    splat = node.splat_data()
    for target in (splat, node):
        method = getattr(target, "soft_delete", None)
        if callable(method):
            method(tensor)
            return
    raise SplatIOError("the current LichtFeld build exposes no splat soft_delete method")


def _export_node_ply(scene, node, output: Path) -> None:
    """Use the host exporter while supporting the validated API spellings."""
    attempts: list[Callable[[], object]] = []
    io = getattr(lf, "io", None)
    save_ply = getattr(io, "save_ply", None)
    if callable(save_ply):
        attempts.append(lambda: save_ply(node.splat_data(), str(output)))
    export_scene = getattr(lf, "export_scene", None)
    if callable(export_scene):
        attempts.append(lambda: export_scene(0, str(output), [node.name], 3))
    for target in (node, node.splat_data(), scene, lf):
        for name in ("export_ply", "save_ply"):
            method = getattr(target, name, None)
            if not callable(method):
                continue
            if target is scene:
                attempts.extend(
                    (
                        lambda method=method: method(str(output), [node.name]),
                        lambda method=method: method(str(output), [node]),
                    )
                )
            else:
                attempts.append(lambda method=method: method(str(output)))
    failures: list[str] = []
    for attempt in attempts:
        try:
            attempt()
            if output.is_file():
                return
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(str(exc))
    detail = failures[-1] if failures else "no supported PLY exporter found"
    raise SplatIOError(f"could not export cropped splat PLY: {detail}")


def crop_trained_scene(dataset_root: str | Path, run_dir: str | Path, block_id: int) -> Path:
    """Soft-delete non-owned splats in the current training scene and export it."""
    frame, artifacts = load_block_artifacts(dataset_root)
    validate_source_hashes(dataset_root, artifacts)
    by_id = {artifact.block.block_id: artifact for artifact in artifacts}
    try:
        artifact = by_id[block_id]
    except KeyError as exc:
        raise SplatIOError(f"no manifest exists for trained block {block_id}") from exc
    scene = lf.get_scene()
    if scene is None:
        raise SplatIOError("training finished without an active LichtFeld scene")
    node = _only_trained_splat(scene)
    splat = node.splat_data()
    positions_w = world_positions(splat.means_raw, node.world_transform)
    keep = ownership_mask(
        frame.to_ground(positions_w), block_id, [item.block for item in artifacts]
    )
    _soft_delete(node, ~keep)
    paths = crop_paths(run_dir, block_id)
    _export_node_ply(scene, node, paths.ply)
    try:
        write_crop_receipt(paths, artifact)
    except MergeError as exc:
        raise SplatIOError(str(exc)) from exc
    return paths.ply


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SplatIOError(f"missing required {name} environment variable")
    return Path(value)


def _on_training_end(callback: Callable[[], None]) -> None:
    """Register against the callback spelling verified by supported host builds."""
    hook = getattr(lf, "on_training_end", None)
    if callable(hook):
        hook(callback)
        return
    events = getattr(lf, "events", None)
    hook = getattr(events, "on_training_end", None)
    if callable(hook):
        hook(callback)
        return
    raise SplatIOError("LichtFeld exposes no on_training_end callback hook")


def register_training_crop() -> None:
    """Register the environment-configured crop callback for a trainer process."""
    dataset_root = _env_path("LST_DATASET_ROOT")
    run_dir = _env_path("LST_RUN_DIR")
    try:
        block_id = int(os.environ["LST_BLOCK_ID"])
    except (KeyError, ValueError) as exc:
        raise SplatIOError("LST_BLOCK_ID must be a non-negative integer") from exc
    if block_id < 0:
        raise SplatIOError("LST_BLOCK_ID must be a non-negative integer")

    def _crop(*_args) -> None:
        output = crop_trained_scene(dataset_root, run_dir, block_id)
        lf.log.info(f"Large Scene Trainer: cropped block {block_id} to {output}")

    _on_training_end(_crop)


def load_cropped_blocks(
    crop_files: Sequence[str | Path], *, group_name: str = "LST Cropped Blocks"
):
    """Load individual crop PLYs as separate scene nodes for interactive seam inspection."""
    scene = lf.get_scene()
    if scene is None:
        raise SplatIOError("open a LichtFeld scene before loading cropped blocks")
    group_id = scene.add_group(group_name)
    loaded = []
    for path in crop_files:
        crop_path = Path(path)
        # ``lf.load_file`` replaces the current scene.  ``lf.io.load`` returns
        # data only, allowing every crop to be added as an independent node.
        result = lf.io.load(str(crop_path))
        splat = result.splat_data
        if splat is None:
            raise SplatIOError(f"cropped PLY contains no splat data: {crop_path}")
        node_id = scene.add_splat(
            crop_path.stem,
            splat.means_raw,
            splat.sh0_raw,
            splat.shN_raw,
            splat.scaling_raw,
            splat.rotation_raw,
            splat.opacity_raw,
            sh_degree=splat.active_sh_degree,
            scene_scale=splat.scene_scale,
            parent=group_id,
        )
        node = scene.get_node_by_id(node_id)
        if node is None:
            raise SplatIOError(f"LichtFeld did not create a node for cropped PLY: {crop_path}")
        loaded.append(node)
    scene.notify_changed()
    return tuple(loaded)
