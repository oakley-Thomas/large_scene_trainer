"""Synthetic trajectory tests for ground-frame fitting."""

from __future__ import annotations

import json
import math

import numpy as np

from large_scene_trainer.core.trajectory_plane import fit_trajectory_plane
from large_scene_trainer.core.types import CameraRef, GroundFrame


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip(abs(float(np.dot(unit(a), unit(b)))), -1.0, 1.0)))


def make_camera(
    centre: np.ndarray,
    forward: np.ndarray = np.array([1.0, 0.0, 0.0]),
    up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    image_id: int = 0,
    name: str | None = None,
) -> CameraRef:
    up = unit(np.asarray(up, dtype=np.float64))
    forward = unit(np.asarray(forward, dtype=np.float64))
    right = unit(np.cross(forward, up))
    down = -up
    rotation = np.stack([right, down, forward])
    # This is the COLMAP relation inverted for the requested world-space centre.
    translation = -rotation @ centre
    np.testing.assert_allclose(-rotation.T @ translation, centre, atol=1e-12)
    return CameraRef(
        image_id=image_id,
        name=name or f"frame_{image_id:04d}.jpg",
        camera_id=1,
        C_world=np.asarray(centre, dtype=np.float64),
        R_world_cam=rotation,
        fx=1000.0,
        fy=1000.0,
        width=1920,
        height=1080,
    )


def loop_cameras(n: int = 100, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> list[CameraRef]:
    up = unit(up)
    basis_a = unit(np.cross(np.array([0.0, 1.0, 0.0]), up))
    if np.linalg.norm(basis_a) < 1e-8:
        basis_a = unit(np.cross(np.array([1.0, 0.0, 0.0]), up))
    basis_b = unit(np.cross(up, basis_a))
    result = []
    for index, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)):
        radial = math.cos(theta) * basis_a + math.sin(theta) * basis_b
        tangent = -math.sin(theta) * basis_a + math.cos(theta) * basis_b
        result.append(make_camera(20.0 * radial, tangent, up, index))
    return result


def test_axis_row_identity():
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]))
    for index in range(3):
        axis = np.zeros(3)
        axis[index] = 1.0
        np.testing.assert_allclose(rotation.T @ axis, rotation[index, :])


def test_straight_line_triggers_degeneracy():
    cameras = [make_camera(np.array([float(i), 0.0, 0.0]), image_id=i) for i in range(100)]
    _, diagnostics = fit_trajectory_plane(cameras, scene_units_per_metre=1.0)
    assert diagnostics.pca_degenerate
    assert diagnostics.estimator_used == "orientation"
    assert angle_deg(diagnostics.normal_orientation, np.array([0.0, 0.0, 1.0])) < 1.0


def test_l_turn_and_loop_have_well_conditioned_pca():
    l_centres = [np.array([float(i), 0.0, 0.0]) for i in range(25)]
    l_centres.extend(np.array([24.0, float(i), 0.0]) for i in range(1, 25))
    l_cameras = [make_camera(centre, image_id=index) for index, centre in enumerate(l_centres)]
    _, l_diagnostics = fit_trajectory_plane(l_cameras, scene_units_per_metre=1.0)
    _, loop_diagnostics = fit_trajectory_plane(loop_cameras(), scene_units_per_metre=1.0)
    for diagnostics in (l_diagnostics, loop_diagnostics):
        assert not diagnostics.pca_degenerate
        assert angle_deg(diagnostics.normal_orientation, np.array([0.0, 0.0, 1.0])) < 1.0
        assert diagnostics.agreement_deg is not None and diagnostics.agreement_deg < 5.0


def test_tilted_ground_and_corrupted_poses_remain_correct():
    up = unit(np.array([0.0, -math.sin(math.radians(20.0)), math.cos(math.radians(20.0))]))
    cameras = loop_cameras(up=up)
    _, diagnostics = fit_trajectory_plane(cameras, scene_units_per_metre=1.0)
    assert angle_deg(diagnostics.normal_orientation, up) < 1.0

    rng = np.random.default_rng(42)
    corrupted = list(cameras)
    for index in rng.choice(len(corrupted), size=5, replace=False):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        old = corrupted[index]
        corrupted[index] = CameraRef(
            image_id=old.image_id, name=old.name, camera_id=old.camera_id,
            C_world=old.C_world, R_world_cam=rotation, fx=old.fx, fy=old.fy,
            width=old.width, height=old.height,
        )
    _, diagnostics = fit_trajectory_plane(corrupted, scene_units_per_metre=1.0)
    assert angle_deg(diagnostics.normal_orientation, up) < 2.0


def test_pitched_rig_uses_nullspace_fallback():
    cameras = loop_cameras()
    pitched: list[CameraRef] = []
    for index, camera in enumerate(cameras):
        rotation = camera.R_world_cam.copy()
        if index % 2:
            # Pitch about camera right; varied headings around the loop keep the
            # null-space estimate centred on the true gravity direction.
            theta = math.radians(30.0)
            down, forward = rotation[1], rotation[2]
            rotation[1] = math.cos(theta) * down - math.sin(theta) * forward
            rotation[2] = math.sin(theta) * down + math.cos(theta) * forward
        pitched.append(
            CameraRef(camera.image_id, camera.name, camera.camera_id, camera.C_world,
                      rotation, camera.fx, camera.fy, camera.width, camera.height)
        )
    _, diagnostics = fit_trajectory_plane(pitched, scene_units_per_metre=1.0)
    assert diagnostics.estimator_used == "nullspace"
    assert angle_deg(diagnostics.normal_orientation, np.array([0.0, 0.0, 1.0])) < 2.0


def test_frame_roundtrip_orthonormality_determinism_and_u_direction():
    cameras = [make_camera(np.array([float(i), 0.0, 0.0]), image_id=i) for i in range(20)]
    frame, _ = fit_trajectory_plane(cameras, scene_units_per_metre=1.0)
    points = np.random.default_rng(0).normal(size=(1000, 3))
    np.testing.assert_allclose(frame.to_ground(frame.to_world(points)), points, atol=1e-9)
    np.testing.assert_allclose(frame.R @ frame.R.T, np.eye(3), atol=1e-12, rtol=0.0)
    assert np.linalg.det(frame.R) > 0.0
    assert angle_deg(frame.R[0], np.array([1.0, 0.0, 0.0])) < 1.0
    assert frame.R[0, 0] > 0.0
    _, diagnostics = fit_trajectory_plane(cameras)
    assert set(diagnostics.to_dict()) == {
        "estimator_used", "normal_orientation", "orientation_dispersion_deg",
        "normal_pca", "pca_condition_ratio", "pca_degenerate", "agreement_deg",
        "height_residual_rms", "height_residual_max", "trajectory_arc_length",
        "median_inter_frame_spacing", "extent_ground", "n_cameras", "warnings",
    }
    json.dumps(diagnostics.to_dict())

    shuffled = list(reversed(cameras))
    shuffled_frame, _ = fit_trajectory_plane(shuffled, scene_units_per_metre=1.0)
    assert frame.to_dict() == shuffled_frame.to_dict()
    restored = GroundFrame.from_dict(frame.to_dict())
    np.testing.assert_array_equal(restored.R, frame.R)
    np.testing.assert_array_equal(restored.origin, frame.origin)
