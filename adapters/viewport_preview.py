"""LichtFeld scene adapter for non-destructive block wireframe previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import lichtfeld as lf
import numpy as np

from ..core.preview_geometry import box_wireframe_points
from ..core.types import Block, GroundFrame


PREVIEW_GROUP_NAME = "LST Block Preview"
_CORE_COLOR = np.array((0.15, 0.75, 1.0), dtype=np.float32)
_CONTEXT_COLOR = np.array((1.0, 0.55, 0.05), dtype=np.float32)


class ViewportPreviewError(RuntimeError):
    """Raised when a block preview cannot be added to the active scene."""


@dataclass(frozen=True)
class ViewportPreviewSummary:
    """Counts added by one preview refresh."""

    block_count: int
    core_point_count: int
    context_point_count: int
    already_visible: bool = False


def _as_tensor(values: np.ndarray):
    return lf.Tensor.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _preview_exists(scene) -> bool:
    """Whether this loaded scene already has this plugin's wireframe nodes."""
    return any(
        node.name == PREVIEW_GROUP_NAME
        or node.name.startswith("lst_preview_core_")
        or node.name.startswith("lst_preview_context_")
        for node in scene.get_nodes()
    )


def show_block_preview(
    blocks: Sequence[Block], frame: GroundFrame, *, samples_per_edge: int = 16
) -> ViewportPreviewSummary:
    """Add one colored core/context wireframe preview to the active scene.

    The current host build can freeze or crash if a plugin mutates or deletes
    point-cloud scene nodes after adding them.  Therefore this adapter never
    changes existing preview nodes. Reload the parent dataset to clear or
    refresh the preview.
    """
    if not blocks:
        raise ViewportPreviewError("there are no blocks to preview")
    scene = lf.get_scene()
    if scene is None:
        raise ViewportPreviewError("load the parent dataset into the viewport first")
    if _preview_exists(scene):
        return ViewportPreviewSummary(
            block_count=len(blocks), core_point_count=0, context_point_count=0, already_visible=True
        )

    group_id = scene.add_group(PREVIEW_GROUP_NAME)
    core_point_count = 0
    context_point_count = 0
    for block in sorted(blocks, key=lambda item: item.block_id):
        core_points = box_wireframe_points(
            block.core, frame, samples_per_edge=samples_per_edge
        ).astype(np.float32)
        context_points = box_wireframe_points(
            block.context, frame, samples_per_edge=samples_per_edge
        ).astype(np.float32)
        scene.add_point_cloud(
            f"lst_preview_core_{block.block_id:03d}",
            _as_tensor(core_points),
            _as_tensor(np.broadcast_to(_CORE_COLOR, core_points.shape).copy()),
            parent=group_id,
        )
        scene.add_point_cloud(
            f"lst_preview_context_{block.block_id:03d}",
            _as_tensor(context_points),
            _as_tensor(np.broadcast_to(_CONTEXT_COLOR, context_points.shape).copy()),
            parent=group_id,
        )
        core_point_count += len(core_points)
        context_point_count += len(context_points)

    scene.notify_changed()
    return ViewportPreviewSummary(
        block_count=len(blocks),
        core_point_count=core_point_count,
        context_point_count=context_point_count,
    )
