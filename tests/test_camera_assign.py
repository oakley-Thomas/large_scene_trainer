"""Analytic and invariant coverage for DT.1.3 frustum assignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from large_scene_trainer.core.camera_assign import (
    CameraAssignmentConfig,
    _cheap_sat_axes,
    _frustum_corners_ground,
    _frustum_intersects_box,
    _sat_intersects_aabb,
    assign_cameras,
    frustum_config_from_diagnostics,
)
from large_scene_trainer.core.partition import partition_trajectory
from large_scene_trainer.core.trajectory_plane import fit_trajectory_plane
from large_scene_trainer.core.types import (
    Block,
    BlockBox,
    CameraRef,
    GroundFrame,
    SegmentationConfig,
)
from .fixtures.trajectories import straight_run


def _frame(rotation: np.ndarray | None = None, origin: np.ndarray | None = None) -> GroundFrame:
    return GroundFrame(
        R=np.eye(3) if rotation is None else rotation,
        origin=np.zeros(3) if origin is None else origin,
    )


def _camera(
    centre: np.ndarray | None = None, rotation: np.ndarray | None = None
) -> CameraRef:
    return CameraRef(
        image_id=1,
        name="camera.jpg",
        camera_id=1,
        C_world=np.zeros(3) if centre is None else centre,
        R_world_cam=np.eye(3) if rotation is None else rotation,
        fx=100.0,
        fy=100.0,
        width=200,
        height=200,
    )


def _frustum_config(near: float = 1.0, far: float = 10.0) -> CameraAssignmentConfig:
    return CameraAssignmentConfig(
        predicate="frustum",
        frustum_near_su=near,
        frustum_far_su=far,
        min_cameras_per_block=0,
    )


def _box(
    minimum: tuple[float, float, float], maximum: tuple[float, float, float]
) -> BlockBox:
    return BlockBox(np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64))


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (_box((-0.5, -0.5, 5.0), (0.5, 0.5, 6.0)), True),
        (_box((-0.5, -0.5, -6.0), (0.5, 0.5, -5.0)), False),
        (_box((-0.5, -0.5, 11.0), (0.5, 0.5, 12.0)), False),
        (_box((-0.1, -0.1, 0.1), (0.1, 0.1, 0.5)), False),
        (_box((-0.1, -0.1, 0.5), (0.1, 0.1, 1.5)), True),
        (_box((5.0, -0.2, 5.0), (5.2, 0.2, 5.5)), True),
        (_box((6.0, -0.2, 5.0), (6.2, 0.2, 5.5)), False),
    ],
    ids=(
        "ahead",
        "behind",
        "beyond_far",
        "inside_near",
        "straddles_near",
        "inside_fov",
        "outside_fov",
    ),
)
def test_frustum_intersection_analytic_cases(box: BlockBox, expected: bool):
    frame = _frame()
    camera = _camera()
    assert _frustum_intersects_box(camera, np.zeros(3), box, frame, _frustum_config()) is expected


def _full_sat_intersects(corners_g: np.ndarray, centre_g: np.ndarray, box: BlockBox) -> bool:
    """Test-only 26-axis frustum-vs-AABB SAT reference."""
    cheap_axes = _cheap_sat_axes(corners_g, centre_g)
    edge_directions = np.asarray(
        [
            corners_g[1] - corners_g[0],
            corners_g[3] - corners_g[0],
            corners_g[4] - corners_g[0],
            corners_g[5] - corners_g[1],
            corners_g[6] - corners_g[2],
            corners_g[7] - corners_g[3],
        ],
        dtype=np.float64,
    )
    edge_cross_axes = np.asarray(
        [np.cross(box_axis, edge) for box_axis in np.eye(3) for edge in edge_directions],
        dtype=np.float64,
    )
    axes = np.vstack((cheap_axes, edge_cross_axes))
    assert axes.shape == (26, 3)
    return _sat_intersects_aabb(corners_g, box, axes)


def test_cheap_sat_known_false_positive_is_intentionally_retained():
    angle = math.radians(20.0)
    rotation = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    frame = _frame()
    camera = _camera(rotation=rotation)
    box = BlockBox(
        np.array([-12.88592915, 9.60034908, 1.49897024]),
        np.array([-11.29609763, 12.09533686, 4.96426818]),
    )
    config = _frustum_config()
    corners_g = _frustum_corners_ground(camera, frame, config)

    assert _frustum_intersects_box(camera, np.zeros(3), box, frame, config)
    assert not _full_sat_intersects(corners_g, np.zeros(3), box)


def test_cheap_sat_never_excludes_a_full_sat_intersection():
    frame = _frame()
    camera = _camera()
    config = _frustum_config()
    corners_g = _frustum_corners_ground(camera, frame, config)
    rng = np.random.default_rng(13)
    for _ in range(100):
        centre = rng.uniform(-12.0, 12.0, size=3)
        half_extent = rng.uniform(0.05, 2.0, size=3)
        box = BlockBox(centre - half_extent, centre + half_extent)
        full = _full_sat_intersects(corners_g, np.zeros(3), box)
        cheap = _frustum_intersects_box(camera, np.zeros(3), box, frame, config)
        assert not full or cheap


def test_fov_result_is_invariant_to_world_frame_rotation():
    config = _frustum_config()
    box = _box((4.5, -0.5, 4.5), (5.5, 0.5, 5.5))
    baseline = _frustum_intersects_box(_camera(), np.zeros(3), box, _frame(), config)

    angle = math.radians(37.0)
    ground_rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    frame = _frame(ground_rotation, np.array([8.0, -3.0, 2.0]))
    centre_g = np.zeros(3)
    transformed = _frustum_intersects_box(
        _camera(frame.to_world(centre_g), ground_rotation),
        centre_g,
        box,
        frame,
        config,
    )
    assert transformed is baseline


def test_fov_result_is_uniform_scale_invariant():
    frame = _frame()
    camera = _camera()
    box = _box((4.5, -0.5, 4.5), (5.5, 0.5, 5.5))
    config = _frustum_config()
    baseline = _frustum_intersects_box(camera, np.zeros(3), box, frame, config)

    scale = 17.0
    scaled_box = BlockBox(box.min_g * scale, box.max_g * scale)
    scaled_config = _frustum_config(
        config.frustum_near_su * scale, config.frustum_far_su * scale
    )
    assert _frustum_intersects_box(camera, np.zeros(3), scaled_box, frame, scaled_config) is baseline


def test_frustum_assignment_unions_centre_membership_and_is_deterministic():
    frame = _frame()
    camera = _camera()
    core = _box((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    block = Block(0, core, core, (0,), (0, 0), "synthetic")
    config = _frustum_config()

    first, first_diagnostics = assign_cameras([camera], [block], frame, config)
    second, second_diagnostics = assign_cameras([camera], [block], frame, config)
    assert first[0].camera_indices == (0,)
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics.predicate == "frustum"


def test_predicate_change_never_changes_core_ownership_diagnostics():
    cameras = straight_run(72)
    frame, plane_diagnostics = fit_trajectory_plane(cameras)
    segmentation = SegmentationConfig(12.0, 16, 2.0, 0.20, 1.0, 6.0, 6.0)
    blocks, _ = partition_trajectory(cameras, frame, plane_diagnostics, segmentation)

    centre_blocks, centre_diagnostics = assign_cameras(
        cameras, blocks, frame, CameraAssignmentConfig(min_cameras_per_block=0)
    )
    frustum_blocks, frustum_diagnostics = assign_cameras(
        cameras, blocks, frame, _frustum_config(0.5, 10.0)
    )
    for centre_block, frustum_block in zip(centre_blocks, frustum_blocks):
        assert set(centre_block.camera_indices) <= set(frustum_block.camera_indices)
    assert frustum_diagnostics.predicate == "frustum"
    assert frustum_diagnostics.orphaned_camera_indices == centre_diagnostics.orphaned_camera_indices
    assert (
        frustum_diagnostics.multiply_owned_camera_indices
        == centre_diagnostics.multiply_owned_camera_indices
    )


def test_frustum_config_uses_diagnostic_spacing_for_near_plane():
    _, diagnostics = fit_trajectory_plane(straight_run(40))
    config = frustum_config_from_diagnostics(diagnostics, frustum_far_su=20.0)
    assert CameraAssignmentConfig().predicate == "centre"
    assert config.predicate == "frustum"
    assert config.frustum_near_su == diagnostics.median_inter_frame_spacing
    assert config.frustum_far_su == 20.0
