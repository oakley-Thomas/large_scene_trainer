"""Deterministic synthetic camera trajectories for partitioning tests."""

from __future__ import annotations

import numpy as np

from large_scene_trainer.core.types import CameraRef


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _camera(centre: np.ndarray, index: int) -> CameraRef:
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = _unit(np.cross(forward, up))
    rotation = np.stack([right, -up, forward])
    return CameraRef(
        image_id=index,
        name=f"frame_{index:05d}.jpg",
        camera_id=1,
        C_world=np.asarray(centre, dtype=np.float64),
        R_world_cam=rotation,
        fx=1000.0,
        fy=1000.0,
        width=1920,
        height=1080,
    )


def _from_points(points: np.ndarray) -> list[CameraRef]:
    return [_camera(point, index) for index, point in enumerate(points)]


def straight_run(n: int = 72) -> list[CameraRef]:
    return _from_points(np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)]))


def figure_eight(n: int = 96) -> list[CameraRef]:
    t = np.linspace(-np.pi, np.pi, n)
    return _from_points(np.column_stack([20.0 * np.sin(t), 10.0 * np.sin(2.0 * t), np.zeros(n)]))


def grid_loop(n: int = 96) -> list[CameraRef]:
    corners = np.array(
        [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [10.0, 20.0], [10.0, 5.0], [25.0, 5.0], [25.0, 25.0], [0.0, 25.0], [0.0, 0.0]]
    )
    distances = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    samples = np.linspace(0.0, cumulative[-1], n)
    xy = np.empty((n, 2))
    for index, distance in enumerate(samples):
        segment = min(np.searchsorted(cumulative, distance, side="right") - 1, len(distances) - 1)
        alpha = (distance - cumulative[segment]) / distances[segment]
        xy[index] = corners[segment] * (1.0 - alpha) + corners[segment + 1] * alpha
    return _from_points(np.column_stack([xy, np.zeros(n)]))


def l_turn(n: int = 72) -> list[CameraRef]:
    first = n // 2
    points = np.empty((n, 3))
    points[:first, 0] = np.arange(first)
    points[:first, 1] = 0.0
    points[first:, 0] = first - 1.0
    points[first:, 1] = np.arange(1, n - first + 1)
    points[:, 2] = 0.0
    return _from_points(points)
