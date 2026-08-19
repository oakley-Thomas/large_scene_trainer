"""Camera assignment: which cameras train which block.

Ticket 1.3. This module is landed AHEAD of 1.3's real work so that ticket 1.4
(per-block dataset export) has a stable interface to build against and is never
blocked on which assignment path ships.

WHAT IS LANDED NOW
------------------
`assign_cameras()` with the ticket's stated fallback predicate implemented:
camera centre inside the context box. This is functional — 1.4 can export real
blocks against it today.

DEPLOYMENT DEFAULT
------------------
The frustum predicate is available, but `CameraAssignmentConfig.predicate`
remains "centre" until a scene-specific far-plane sweep meets the training-cost
gate. Both predicates share the same `Block.camera_indices` interface.

TWO MEMBERSHIP RELATIONS — DO NOT CONFLATE THEM
-----------------------------------------------
1. TRAINING MEMBERSHIP (what this module computes, stored in
   `Block.camera_indices`): a camera belongs to a block if it sees into that
   block's CONTEXT box. Cameras deliberately appear in SEVERAL blocks. That
   duplication is the entire point of the context margin — it is what stops
   block edges from being reconstructed from one-sided observation.

2. OWNERSHIP (an invariant, not an output): each camera CENTRE lies in exactly
   one CORE box. Cores tile without overlapping (guaranteed by 1.2's de-overlap
   pass), so this is a true partition. Phase 3 crops trained Gaussians to core
   boxes and concatenates; if ownership were not a partition, volume would be
   merged twice.

Getting these backwards produces blocks that look plausible and merge wrong.
`assignment_diagnostics()` checks (2) explicitly so the mistake fails loudly.

HARD RULES INHERITED FROM 1.1 / 1.2
-----------------------------------
- Nothing under `core/` may import `lichtfeld` or `lfs_plugins`.
- numpy only. No SciPy, no Open3D, no PyTorch.
- float64 throughout.
- Cameras are processed in name-sorted order; indices are into that list.
- Deterministic: identical input produces byte-identical output.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .types import Block, BlockBox, CameraRef, GroundFrame, PlaneDiagnostics


# CameraRef intentionally carries only focal lengths and image dimensions. The
# small inflation covers the absent principal point without changing that
# cross-ticket interface.
_FOV_INFLATE = 1.05
_SAT_EPSILON_SCALE = 1e-12


@dataclass(frozen=True)
class CameraAssignmentConfig:
    """Configuration for camera-to-block assignment.

    predicate:
        "centre"  — camera centre inside the context box. Ticket 1.3's stated
                    fallback. Implemented, deterministic, cheap.
        "frustum" — camera frustum intersects the context box, unioned with
                    camera-centre membership to preserve core ownership.

    frustum_near_su / frustum_far_su:
        Near and far planes of the truncated frustum, in scene units. Only read
        when predicate == "frustum". Derive from 1.1's diagnostics
        (`median_inter_frame_spacing` for near, a multiple of the block extent
        for far) rather than hard-coding — scene units are not metres.

    min_cameras_per_block:
        A block with fewer assigned cameras than this is a defect, not a small
        block. Raises rather than warns; 1.4 will not export it.
    """

    predicate: str = "centre"
    frustum_near_su: float = 0.0
    frustum_far_su: float = 0.0
    min_cameras_per_block: int = 32

    def __post_init__(self) -> None:
        if self.predicate not in ("centre", "frustum"):
            raise ValueError(f"unknown predicate: {self.predicate!r}")
        if self.predicate == "frustum":
            if not (0.0 <= self.frustum_near_su < self.frustum_far_su):
                raise ValueError(
                    "frustum predicate requires 0 <= near < far; got "
                    f"near={self.frustum_near_su}, far={self.frustum_far_su}"
                )


@dataclass(frozen=True)
class AssignmentDiagnostics:
    """Recorded to the diagnostics JSON and echoed into each block manifest."""

    predicate: str
    n_cameras: int
    n_blocks: int
    cameras_per_block: tuple[int, ...]
    mean_blocks_per_camera: float
    max_blocks_per_camera: int
    orphaned_camera_indices: tuple[int, ...]
    multiply_owned_camera_indices: tuple[int, ...]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["cameras_per_block"] = list(d["cameras_per_block"])
        d["orphaned_camera_indices"] = list(d["orphaned_camera_indices"])
        d["multiply_owned_camera_indices"] = list(d["multiply_owned_camera_indices"])
        return d


def frustum_config_from_diagnostics(
    diagnostics: PlaneDiagnostics,
    *,
    frustum_far_su: float,
    min_cameras_per_block: int = 32,
) -> CameraAssignmentConfig:
    """Build a frustum configuration from 1.1's scene-unit diagnostics.

    The near plane is tied to capture spacing, not a world-unit literal.
    ``frustum_far_su`` remains explicit because its safe value must be selected
    from a measured assignment sweep for each scene.
    """
    near = float(diagnostics.median_inter_frame_spacing)
    if not np.isfinite(near) or near <= 0.0:
        raise ValueError("median_inter_frame_spacing must be finite and positive")
    return CameraAssignmentConfig(
        predicate="frustum",
        frustum_near_su=near,
        frustum_far_su=frustum_far_su,
        min_cameras_per_block=min_cameras_per_block,
    )


def assign_cameras(
    cams: list[CameraRef],
    blocks: list[Block],
    frame: GroundFrame,
    cfg: CameraAssignmentConfig,
) -> tuple[list[Block], AssignmentDiagnostics]:
    """Assign training cameras to blocks by the configured predicate.

    Args:
        cams:   name-sorted camera list (the canonical ordering from 1.1).
        blocks: blocks from 1.2, with core and context boxes already authored
                in G. Their `camera_indices` are 1.2's sequential grouping and
                are REPLACED by this function.
        frame:  the ground frame from 1.1. Must be the same object 1.2 used;
                pass it through, never re-fit.
        cfg:    assignment configuration.

    Returns:
        (blocks, diagnostics) — new Block instances with `camera_indices`
        replaced. Input blocks are not mutated (Block is frozen).

    Raises:
        ValueError: if any block ends up with fewer than
            `cfg.min_cameras_per_block` cameras, if any camera centre lies in
            no core box, or if any camera centre lies in more than one core box.
    """
    if not blocks:
        raise ValueError("no blocks to assign cameras to")

    centres_world = np.asarray([c.C_world for c in cams], dtype=np.float64)
    centres_g = frame.to_ground(centres_world)  # (N, 3) as (u, v, h)

    assigned: list[list[int]] = []
    for blk in blocks:
        if cfg.predicate == "centre":
            member = blk.context.contains(centres_g)  # (N,) bool
        else:
            frustum_member = np.asarray(
                [
                    _frustum_intersects_box(cams[i], centres_g[i], blk.context, frame, cfg)
                    for i in range(len(cams))
                ],
                dtype=bool,
            )
            # A truncated frustum can exclude its own centre (behind the view
            # direction or inside the near plane). Context membership is the
            # required permissive union and protects unique core ownership.
            member = frustum_member | blk.context.contains(centres_g)
        # Ascending index order, so output is deterministic regardless of the
        # predicate's internal evaluation order.
        assigned.append(sorted(int(i) for i in np.flatnonzero(member)))

    diag = assignment_diagnostics(centres_g, blocks, assigned, cfg.predicate)

    for blk, idxs in zip(blocks, assigned):
        if len(idxs) < cfg.min_cameras_per_block:
            raise ValueError(
                f"block {blk.block_id} assigned only {len(idxs)} cameras "
                f"(minimum {cfg.min_cameras_per_block}). The box is wrong, or "
                f"the context margin is too small — do not export this block."
            )
    if diag.orphaned_camera_indices:
        raise ValueError(
            f"{len(diag.orphaned_camera_indices)} camera centre(s) lie in no core "
            f"box, first at index {diag.orphaned_camera_indices[0]}. 1.2's cores "
            f"must tile the trajectory; a gap means the de-overlap trim pass "
            f"orphaned a camera."
        )
    if diag.multiply_owned_camera_indices:
        raise ValueError(
            f"{len(diag.multiply_owned_camera_indices)} camera centre(s) lie in "
            f"more than one core box, first at index "
            f"{diag.multiply_owned_camera_indices[0]}. Cores must not overlap — "
            f"Phase 3 would merge that volume twice."
        )

    out = [
        dataclasses.replace(blk, camera_indices=tuple(idxs))
        for blk, idxs in zip(blocks, assigned)
    ]
    return out, diag


def assignment_diagnostics(
    centres_g: np.ndarray,
    blocks: list[Block],
    assigned: list[list[int]],
    predicate: str,
) -> AssignmentDiagnostics:
    """Compute assignment statistics and check the core-ownership partition.

    Training membership (context boxes) is expected to overlap across blocks.
    Ownership (core boxes) must be a partition: every camera centre in exactly
    one core. This function reports violations; the caller decides to raise.
    """
    n = centres_g.shape[0]
    core_hits = np.zeros(n, dtype=np.int64)
    for blk in blocks:
        core_hits += blk.core.contains(centres_g).astype(np.int64)

    blocks_per_camera = np.zeros(n, dtype=np.int64)
    for idxs in assigned:
        blocks_per_camera[np.asarray(idxs, dtype=np.int64)] += 1

    return AssignmentDiagnostics(
        predicate=predicate,
        n_cameras=n,
        n_blocks=len(blocks),
        cameras_per_block=tuple(len(i) for i in assigned),
        mean_blocks_per_camera=float(blocks_per_camera.mean()),
        max_blocks_per_camera=int(blocks_per_camera.max()),
        orphaned_camera_indices=tuple(int(i) for i in np.flatnonzero(core_hits == 0)),
        multiply_owned_camera_indices=tuple(int(i) for i in np.flatnonzero(core_hits > 1)),
    )


def _frustum_intersects_box(
    cam: CameraRef,
    centre_g: np.ndarray,
    box: BlockBox,
    frame: GroundFrame,
    cfg: CameraAssignmentConfig,
) -> bool:
    """Does this camera's truncated frustum intersect `box`?

    This is an 8-axis, deliberately permissive SAT: the three AABB axes, the
    view-direction axis, and four side-plane normals. The full SAT also has
    box-axis/frustum-edge cross products; omitting them can create false
    positives but cannot create false negatives, which is the intended trade
    for training-camera assignment.

    The pinhole pyramid assumes a centred principal point. `_FOV_INFLATE`
    provides a small guard band for the missing principal point, but cannot
    make this valid for fisheye cameras or fields of view around/above 120
    degrees. Use the `centre` predicate for such captures.
    """
    corners_g = _frustum_corners_ground(cam, frame, cfg)
    axes = _cheap_sat_axes(corners_g, centre_g)
    return _sat_intersects_aabb(corners_g, box, axes)


def _frustum_corners_ground(
    cam: CameraRef, frame: GroundFrame, cfg: CameraAssignmentConfig
) -> np.ndarray:
    """Return near then far frustum corners in ground coordinates.

    Corner order within each plane is top-left, top-right, bottom-right,
    bottom-left in COLMAP camera coordinates. The public frustum predicate
    validates its configuration before reaching this helper.
    """
    if not (
        np.isfinite(cam.fx) and np.isfinite(cam.fy) and cam.fx > 0.0 and cam.fy > 0.0
    ):
        raise ValueError(
            f"camera {cam.name!r} needs finite positive focal lengths for frustum use"
        )
    if cam.width <= 0 or cam.height <= 0:
        raise ValueError(f"camera {cam.name!r} needs positive image dimensions for frustum use")

    half_angle_x = np.arctan((0.5 * float(cam.width)) / cam.fx) * _FOV_INFLATE
    half_angle_y = np.arctan((0.5 * float(cam.height)) / cam.fy) * _FOV_INFLATE
    planes = np.asarray((cfg.frustum_near_su, cfg.frustum_far_su), dtype=np.float64)
    half_widths = planes * np.tan(half_angle_x)
    half_heights = planes * np.tan(half_angle_y)
    corners_camera = np.asarray(
        [
            (-half_widths[0], -half_heights[0], planes[0]),
            (half_widths[0], -half_heights[0], planes[0]),
            (half_widths[0], half_heights[0], planes[0]),
            (-half_widths[0], half_heights[0], planes[0]),
            (-half_widths[1], -half_heights[1], planes[1]),
            (half_widths[1], -half_heights[1], planes[1]),
            (half_widths[1], half_heights[1], planes[1]),
            (-half_widths[1], half_heights[1], planes[1]),
        ],
        dtype=np.float64,
    )
    # For row-vector batches, the column-vector transform R.T @ X + C becomes
    # X @ R + C. CameraRef.R_world_cam is COLMAP's world-to-camera rotation.
    corners_world = corners_camera @ cam.R_world_cam + cam.C_world
    return frame.to_ground(corners_world)


def _cheap_sat_axes(corners_g: np.ndarray, centre_g: np.ndarray) -> np.ndarray:
    """Return the three box and five unique frustum-face SAT axes."""
    centre = np.asarray(centre_g, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"centre_g must have shape (3,), got {centre.shape}")
    far_centre = corners_g[4:].mean(axis=0)
    side_pairs = ((0, 3), (1, 2), (0, 1), (3, 2))
    side_normals = [
        np.cross(corners_g[first] - centre, corners_g[second] - centre)
        for first, second in side_pairs
    ]
    return np.vstack((np.eye(3, dtype=np.float64), far_centre - centre, side_normals))


def _sat_intersects_aabb(corners_g: np.ndarray, box: BlockBox, axes: np.ndarray) -> bool:
    """Return whether all supplied SAT axes have overlapping projections."""
    scale = max(
        1.0,
        float(np.max(np.abs(corners_g))),
        float(np.max(np.abs(box.min_g))),
        float(np.max(np.abs(box.max_g))),
    )
    epsilon = _SAT_EPSILON_SCALE * scale
    box_centre = (box.min_g + box.max_g) * 0.5
    box_half_extent = (box.max_g - box.min_g) * 0.5
    for raw_axis in np.asarray(axes, dtype=np.float64):
        norm = float(np.linalg.norm(raw_axis))
        if norm <= np.finfo(np.float64).eps:
            # Degenerate axes cannot separate convex volumes. This occurs at
            # an apex when the allowed near plane is zero.
            continue
        axis = raw_axis / norm
        frustum_projection = corners_g @ axis
        box_projection_centre = float(box_centre @ axis)
        box_radius = float(np.abs(axis) @ box_half_extent)
        if (
            float(np.max(frustum_projection)) < box_projection_centre - box_radius - epsilon
            or float(np.min(frustum_projection)) > box_projection_centre + box_radius + epsilon
        ):
            return False
    return True
