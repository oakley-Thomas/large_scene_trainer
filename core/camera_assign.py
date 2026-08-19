"""Camera assignment: which cameras train which block.

Ticket 1.3. This module is landed AHEAD of 1.3's real work so that ticket 1.4
(per-block dataset export) has a stable interface to build against and is never
blocked on which assignment path ships.

WHAT IS LANDED NOW
------------------
`assign_cameras()` with the ticket's stated fallback predicate implemented:
camera centre inside the context box. This is functional — 1.4 can export real
blocks against it today.

WHAT 1.3 STILL OWES
-------------------
`_frustum_intersects_box()` currently raises. Implementing it and flipping
`CameraAssignmentConfig.predicate` to "frustum" is the whole of 1.3's remaining
work. No signature changes. No new fields. If frustum intersection does not land
by the G1 gate, "centre" ships and nothing downstream notices.

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

from .types import Block, BlockBox, CameraRef, GroundFrame


@dataclass(frozen=True)
class CameraAssignmentConfig:
    """Configuration for camera-to-block assignment.

    predicate:
        "centre"  — camera centre inside the context box. Ticket 1.3's stated
                    fallback. Implemented, deterministic, cheap.
        "frustum" — camera frustum intersects the context box. Ticket 1.3's
                    target. Not yet implemented.

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
            member = np.asarray(
                [
                    _frustum_intersects_box(cams[i], centres_g[i], blk.context, frame, cfg)
                    for i in range(len(cams))
                ],
                dtype=bool,
            )
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

    TICKET 1.3'S REMAINING WORK. Implementing this and setting
    `CameraAssignmentConfig.predicate = "frustum"` completes the ticket.

    Suggested approach — separating-axis test between the frustum's 8 corners
    and the box, both expressed in G:

      1. Build the 8 frustum corners in CAMERA space from fx, fy, w, h and the
         near/far planes:
             x = +/- (w / 2) * z / fx,  y = +/- (h / 2) * z / fy,
             for z in (near, far).
         COLMAP camera axes are +x right, +y down, +z forward (view direction).
      2. Transform to world:  X_world = R_world_cam.T @ X_cam + C_world.
         (`CameraRef.R_world_cam` is COLMAP's world->camera rotation, so its
         transpose maps camera->world.)
      3. Transform to G via `frame.to_ground()`.
      4. Test the box against the frustum's corner cloud. The cheap, correct-
         enough version: check the 3 box axes and the 6 frustum face normals as
         separating axes. Full SAT (15 axes incl. edge cross-products) is
         available if the cheap version proves too permissive, but a slightly
         over-inclusive test is HARMLESS here — an extra camera in a block costs
         training time, a missing one costs reconstruction quality. Bias
         permissive.

    Whatever is implemented must be deterministic and float64 throughout.
    """
    raise NotImplementedError(
        "Ticket 1.3: frustum-vs-context-box intersection is not implemented. "
        "Use CameraAssignmentConfig(predicate='centre') until it lands."
    )
