"""Coverage for host-independent block wireframe geometry."""

import numpy as np
import pytest

from large_scene_trainer.core.preview_geometry import box_corners, box_wireframe_points
from large_scene_trainer.core.types import BlockBox, GroundFrame


def test_box_corners_cover_each_min_max_combination():
    box = BlockBox(np.array((-1.0, 2.0, 3.0)), np.array((4.0, 5.0, 6.0)))

    corners = box_corners(box)

    assert corners.shape == (8, 3)
    assert {tuple(point) for point in corners} == {
        (-1.0, 2.0, 3.0),
        (4.0, 2.0, 3.0),
        (-1.0, 5.0, 3.0),
        (4.0, 5.0, 3.0),
        (-1.0, 2.0, 6.0),
        (4.0, 2.0, 6.0),
        (-1.0, 5.0, 6.0),
        (4.0, 5.0, 6.0),
    }


def test_wireframe_samples_world_space_edges():
    box = BlockBox(np.zeros(3), np.ones(3))
    frame = GroundFrame(np.eye(3), np.array((10.0, -2.0, 5.0)))

    points = box_wireframe_points(box, frame, samples_per_edge=3)

    assert points.shape == (36, 3)
    assert np.all(points >= np.array((10.0, -2.0, 5.0)))
    assert np.all(points <= np.array((11.0, -1.0, 6.0)))
    assert np.allclose(points[1], np.array((10.5, -2.0, 5.0)))


def test_wireframe_requires_at_least_two_samples_per_edge():
    box = BlockBox(np.zeros(3), np.ones(3))
    frame = GroundFrame(np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="at least 2"):
        box_wireframe_points(box, frame, samples_per_edge=1)
