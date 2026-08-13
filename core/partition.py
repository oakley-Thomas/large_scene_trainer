"""Deterministic trajectory segmentation in the oriented ground frame.

Only camera centres participate in this module.  Gaussian positions are
deliberately excluded: floaters make them unsuitable for defining scene bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from .types import (
    Block,
    BlockBox,
    CameraRef,
    GroundFrame,
    PlaneDiagnostics,
    SegmentationConfig,
    SegmentationDiagnostics,
)


_MAX_DEOVERLAP_ITERATIONS = 100


@dataclass(frozen=True)
class _SequentialResult:
    blocks: list[Block]
    actions: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class _SpatialNode:
    indices: tuple[int, ...]
    min_uv: np.ndarray
    max_uv: np.ndarray


def config_from_diagnostics(
    diag: PlaneDiagnostics, target_blocks: int = 4, **overrides: object
) -> SegmentationConfig:
    """Build scene-unit partition defaults from trajectory diagnostics."""
    if target_blocks < 1:
        raise ValueError("target_blocks must be positive")
    if diag.n_cameras < 3:
        raise ValueError("at least three cameras are required for segmentation")
    if not np.isfinite(diag.trajectory_arc_length) or diag.trajectory_arc_length <= 0.0:
        raise ValueError("trajectory_arc_length must be finite and positive")
    if not np.isfinite(diag.median_inter_frame_spacing) or diag.median_inter_frame_spacing <= 0.0:
        raise ValueError("median_inter_frame_spacing must be finite and positive")

    spacing = diag.median_inter_frame_spacing
    values: dict[str, object] = {
        "core_extent_su": diag.trajectory_arc_length / target_blocks,
        "max_cameras_per_block": max(3, math.ceil(diag.n_cameras / target_blocks)),
        "lateral_margin_su": 5.0 * spacing,
        "context_margin_frac": 0.10,
        # The spacing multiplier keeps the intended scene-scale behaviour, but
        # a core must always include the camera centres it owns.  The fitted
        # global-plane residual is therefore a hard lower bound on the small
        # downward pad.
        "z_pad_down_su": max(2.0 * spacing, diag.height_residual_max),
        "z_pad_up_su": 12.0 * spacing,
        "merge_slack": 1.0,
    }
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise TypeError(f"unknown segmentation override(s): {', '.join(unknown)}")
    values.update(overrides)
    return SegmentationConfig(**values)  # type: ignore[arg-type]


def _ordered_cameras(cams: Sequence[CameraRef]) -> list[CameraRef]:
    ordered = sorted(cams, key=lambda camera: camera.name)
    names = [camera.name for camera in ordered]
    if len(set(names)) != len(names):
        raise ValueError("camera names must be unique for deterministic segmentation")
    return ordered


def _core_for_indices(
    indices: tuple[int, ...], centres_g: np.ndarray, cfg: SegmentationConfig
) -> BlockBox:
    points = centres_g[list(indices)]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    minimum[:2] -= cfg.lateral_margin_su
    maximum[:2] += cfg.lateral_margin_su
    minimum[2] = -cfg.z_pad_down_su
    maximum[2] = cfg.z_pad_up_su
    return BlockBox(min_g=minimum, max_g=maximum)


def _centre_spans(indices: tuple[int, ...], centres_g: np.ndarray) -> np.ndarray:
    return np.ptp(centres_g[list(indices)], axis=0)


def _within_limits(
    indices: tuple[int, ...], centres_g: np.ndarray, cfg: SegmentationConfig, slack: float
) -> bool:
    return (
        len(indices) <= cfg.max_cameras_per_block * slack
        and float(np.max(_centre_spans(indices, centres_g)[:2])) <= cfg.core_extent_su * slack
    )


def _provisional_block(
    indices: tuple[int, ...], centres_g: np.ndarray, cfg: SegmentationConfig, provenance: str
) -> Block:
    core = _core_for_indices(indices, centres_g, cfg)
    return Block(
        block_id=-1,
        core=core,
        context=core,
        camera_indices=indices,
        frame_span=(indices[0], indices[-1]),
        provenance=provenance,
    )


def _segment(
    ordered: list[CameraRef], frame: GroundFrame, cfg: SegmentationConfig
) -> _SequentialResult:
    if len(ordered) < 3:
        raise ValueError("at least three cameras are required for segmentation")
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in ordered]))
    groups: list[tuple[int, ...]] = []
    open_group: tuple[int, ...] = (0,)
    for index in range(1, len(ordered)):
        candidate = open_group + (index,)
        if not _within_limits(candidate, centres_g, cfg, 1.0):
            groups.append(open_group)
            open_group = (index,)
        else:
            open_group = candidate
    groups.append(open_group)

    actions: list[str] = []
    warnings: list[str] = []
    stub_threshold = 0.25 * cfg.max_cameras_per_block
    if len(groups) > 1 and len(groups[-1]) < stub_threshold:
        merged_indices = groups[-2] + groups[-1]
        if _within_limits(merged_indices, centres_g, cfg, cfg.merge_slack):
            predecessor, tail = len(groups) - 2, len(groups) - 1
            groups[-2] = merged_indices
            groups.pop()
            actions.append(f"merge_tail({predecessor},{tail})")
        elif len(groups[-1]) < 3:
            raise ValueError(
                "trailing segment has fewer than three cameras and cannot merge within merge_slack"
            )
        else:
            warnings.append(
                "retained undersized trailing segment because merging would exceed merge_slack"
            )
    elif len(groups[-1]) < 3:
        raise ValueError(
            "segmentation emitted fewer than three cameras in its only trailing segment"
        )

    blocks = []
    for group_index, indices in enumerate(groups):
        provenance = "sequential"
        if (
            actions
            and group_index == len(groups) - 1
            and actions[-1].startswith("merge_tail")
        ):
            provenance = actions[-1].replace("merge_tail", "merged")
        blocks.append(_provisional_block(indices, centres_g, cfg, provenance))
    return _SequentialResult(blocks=blocks, actions=actions, warnings=warnings)


def segment_sequential(
    cams: Sequence[CameraRef], frame: GroundFrame, cfg: SegmentationConfig
) -> list[Block]:
    """Build provisional sequential core blocks from name-sorted cameras."""
    return _segment(_ordered_cameras(cams), frame, cfg).blocks


def _spatial_core(node: _SpatialNode, cfg: SegmentationConfig) -> BlockBox:
    return BlockBox(
        min_g=np.array([node.min_uv[0], node.min_uv[1], -cfg.z_pad_down_su]),
        max_g=np.array([node.max_uv[0], node.max_uv[1], cfg.z_pad_up_su]),
    )


def _spatial_split(node: _SpatialNode, centres_g: np.ndarray, cfg: SegmentationConfig) -> tuple[_SpatialNode, _SpatialNode, int, float]:
    """Bisect a node at a deterministic gap between camera-centre coordinates."""
    points = centres_g[list(node.indices), :2]
    spans = np.ptp(points, axis=0)
    ratios = spans / cfg.core_extent_su
    axis = int(np.argmax(ratios if np.max(ratios) > 1.0 else spans))
    coordinates = points[:, axis]
    order = np.lexsort((np.asarray(node.indices), coordinates))
    allowed = [
        split
        for split in range(3, len(order) - 2)
        if coordinates[order[split - 1]] < coordinates[order[split]]
    ]
    if not allowed:
        raise ValueError(
            "spatial partition cannot split a leaf while retaining at least three cameras per block"
        )
    target = len(order) / 2.0
    split = min(allowed, key=lambda candidate: (abs(candidate - target), candidate))
    low_indices = tuple(sorted(node.indices[position] for position in order[:split]))
    high_indices = tuple(sorted(node.indices[position] for position in order[split:]))
    plane = float((coordinates[order[split - 1]] + coordinates[order[split]]) / 2.0)
    low_min, low_max = node.min_uv.copy(), node.max_uv.copy()
    high_min, high_max = node.min_uv.copy(), node.max_uv.copy()
    low_max[axis] = plane
    high_min[axis] = plane
    return (
        _SpatialNode(low_indices, low_min, low_max),
        _SpatialNode(high_indices, high_min, high_max),
        axis,
        plane,
    )


def _spatial_leaf_needs_split(
    node: _SpatialNode, centres_g: np.ndarray, cfg: SegmentationConfig
) -> bool:
    points = centres_g[list(node.indices), :2]
    return (
        len(node.indices) > cfg.max_cameras_per_block
        or float(np.max(np.ptp(points, axis=0))) > cfg.core_extent_su
    )


def segment_spatial(
    cams: Sequence[CameraRef], frame: GroundFrame, cfg: SegmentationConfig
) -> tuple[list[Block], list[str]]:
    """Create a non-overlapping spatial kd partition in the ground frame.

    Leaf boundaries are inherited from their parent split planes, so adjacent
    cores meet exactly and camera ownership is spatial rather than temporal.
    """
    ordered = _ordered_cameras(cams)
    if len(ordered) < 3:
        raise ValueError("at least three cameras are required for segmentation")
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in ordered]))
    minimum = centres_g[:, :2].min(axis=0) - cfg.lateral_margin_su
    maximum = centres_g[:, :2].max(axis=0) + cfg.lateral_margin_su
    leaves = [_SpatialNode(tuple(range(len(ordered))), minimum, maximum)]
    actions: list[str] = []
    position = 0
    while position < len(leaves):
        node = leaves[position]
        if not _spatial_leaf_needs_split(node, centres_g, cfg):
            position += 1
            continue
        low, high, axis, plane = _spatial_split(node, centres_g, cfg)
        leaves[position : position + 1] = [low, high]
        actions.append(f"spatial_split({position},{'uv'[axis]},{plane:.17g})")

    blocks = []
    for node in leaves:
        core = _spatial_core(node, cfg)
        blocks.append(
            Block(
                block_id=-1,
                core=core,
                context=core,
                camera_indices=node.indices,
                frame_span=(node.indices[0], node.indices[-1]),
                provenance="spatial",
            )
        )
    return blocks, actions


def _replace_core(block: Block, core: BlockBox, provenance: str | None = None) -> Block:
    return replace(block, core=core, context=core, provenance=provenance or block.provenance)


def _merge_blocks(
    first: Block, second: Block, centres_g: np.ndarray, cfg: SegmentationConfig, i: int, j: int
) -> Block:
    indices = tuple(sorted(first.camera_indices + second.camera_indices))
    return _provisional_block(indices, centres_g, cfg, f"merged({i},{j})")


def _trim_pair(first: Block, second: Block, axis: int) -> tuple[Block, Block, float]:
    overlap_lo = max(first.core.min_g[axis], second.core.min_g[axis])
    overlap_hi = min(first.core.max_g[axis], second.core.max_g[axis])
    midpoint = (overlap_lo + overlap_hi) / 2.0
    first_min, first_max = first.core.min_g.copy(), first.core.max_g.copy()
    second_min, second_max = second.core.min_g.copy(), second.core.max_g.copy()
    if first.core.min_g[axis] <= second.core.min_g[axis]:
        first_max[axis] = midpoint
        second_min[axis] = midpoint
    else:
        first_min[axis] = midpoint
        second_max[axis] = midpoint
    axis_name = "uvh"[axis]
    return (
        _replace_core(first, BlockBox(first_min, first_max), f"trimmed({axis_name})"),
        _replace_core(second, BlockBox(second_min, second_max), f"trimmed({axis_name})"),
        midpoint,
    )


def _ownership_counts(blocks: Sequence[Block], centres_g: np.ndarray) -> np.ndarray:
    """Half-open ownership; only a partition-wide outer maximum is inclusive."""
    outer_max = np.max(np.stack([block.core.max_g for block in blocks]), axis=0)
    counts = np.zeros(len(centres_g), dtype=np.int64)
    for block in blocks:
        lower = centres_g >= block.core.min_g
        upper = (centres_g < block.core.max_g) | (
            (centres_g == block.core.max_g) & (block.core.max_g == outer_max)
        )
        counts += np.all(lower & upper, axis=1)
    return counts


def _repair_first_orphan(
    blocks: list[Block], centres_g: np.ndarray, orphan_index: int
) -> None:
    owner_positions = [
        position for position, block in enumerate(blocks) if orphan_index in block.camera_indices
    ]
    if len(owner_positions) != 1:
        raise RuntimeError(f"camera {orphan_index} has no unique provenance owner")
    position = owner_positions[0]
    block = blocks[position]
    point = centres_g[orphan_index]
    core = BlockBox(np.minimum(block.core.min_g, point), np.maximum(block.core.max_g, point))
    blocks[position] = _replace_core(block, core)


def deoverlap(
    blocks: list[Block], centres_g: np.ndarray, cfg: SegmentationConfig
) -> tuple[list[Block], list[str]]:
    """Merge or trim overlapping cores until they have unique camera ownership."""
    if not blocks:
        return [], []
    centres = np.asarray(centres_g, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError(f"centres_g must have shape (N, 3), got {centres.shape}")
    settled = list(blocks)
    actions: list[str] = []
    for _ in range(_MAX_DEOVERLAP_ITERATIONS):
        changed = False
        for i in range(len(settled)):
            for j in range(i + 1, len(settled)):
                overlap = settled[i].core.overlap_extents(settled[j].core)
                if np.any(overlap <= 0.0):
                    continue
                union = tuple(sorted(settled[i].camera_indices + settled[j].camera_indices))
                if _within_limits(union, centres, cfg, cfg.merge_slack):
                    settled[i] = _merge_blocks(settled[i], settled[j], centres, cfg, i, j)
                    settled.pop(j)
                    actions.append(f"merge({i},{j})")
                else:
                    axis = int(np.argmin(overlap))
                    first, second, midpoint = _trim_pair(settled[i], settled[j], axis)
                    settled[i], settled[j] = first, second
                    actions.append(f"trim({i},{j},{'uvh'[axis]},{midpoint:.17g})")
                changed = True
                break
            if changed:
                break
        if changed:
            continue

        counts = _ownership_counts(settled, centres)
        orphans = np.flatnonzero(counts == 0)
        if len(orphans):
            orphan = int(orphans[0])
            _repair_first_orphan(settled, centres, orphan)
            actions.append(f"reclaim({orphan})")
            continue
        if np.any(counts > 1):
            raise RuntimeError(
                "core boxes have multi-owned camera centres without positive overlap"
            )
        return settled, actions
    raise RuntimeError(
        f"de-overlap did not converge within {_MAX_DEOVERLAP_ITERATIONS} iterations"
    )


def _with_context_and_ids(blocks: Sequence[Block], cfg: SegmentationConfig) -> list[Block]:
    result: list[Block] = []
    for block_id, block in enumerate(blocks):
        amounts = cfg.context_margin_frac * block.core.extent
        amounts[2] = 0.0
        result.append(replace(block, block_id=block_id, context=block.core.dilated(amounts)))
    return result


def partition_trajectory(
    cams: Sequence[CameraRef],
    frame: GroundFrame,
    plane_diagnostics: PlaneDiagnostics,
    cfg: SegmentationConfig,
) -> tuple[list[Block], SegmentationDiagnostics]:
    """Produce settled blocks and diagnostics for a fitted trajectory frame."""
    ordered = _ordered_cameras(cams)
    if len(ordered) != plane_diagnostics.n_cameras:
        raise ValueError("plane diagnostics camera count does not match segmentation cameras")
    if plane_diagnostics.trajectory_arc_length <= 0.0:
        raise ValueError("trajectory_arc_length must be positive for segmentation")
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in ordered]))
    spatial_blocks, spatial_actions = segment_spatial(ordered, frame, cfg)
    # Spatial leaves are constructed to meet at split planes.  Keep the
    # de-overlap pass as an invariant check and future safety net, but it does
    # not alter a valid spatial partition.
    settled, deoverlap_actions = deoverlap(spatial_blocks, centres_g, cfg)
    final_blocks = _with_context_and_ids(settled, cfg)
    counts = _ownership_counts(final_blocks, centres_g)
    grade_pct = (
        100.0
        * plane_diagnostics.height_residual_rms
        / plane_diagnostics.trajectory_arc_length
    )
    warnings: list[str] = []
    if grade_pct > 3.0:
        warnings.append(
            "global-plane z-extents invalid for this scene "
            f"(grade {grade_pct:.2f}%); per-block z must derive from local camera heights"
        )
    diagnostics = SegmentationDiagnostics(
        n_blocks=len(final_blocks),
        cameras_per_block=[len(block.camera_indices) for block in final_blocks],
        core_extents=[tuple(float(value) for value in block.core.extent) for block in final_blocks],
        actions=spatial_actions + deoverlap_actions,
        orphan_cameras=int(np.count_nonzero(counts == 0)),
        multi_owned_cameras=int(np.count_nonzero(counts > 1)),
        grade_pct=grade_pct,
        warnings=warnings,
    )
    if diagnostics.orphan_cameras or diagnostics.multi_owned_cameras:
        raise RuntimeError("de-overlap completed without unique core ownership")
    return final_blocks, diagnostics
