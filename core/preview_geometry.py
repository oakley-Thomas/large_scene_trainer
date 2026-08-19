"""Host-independent geometry for block viewport previews.

The partitioner stores boxes in the oriented ground frame.  Viewport adapters
use these helpers to turn the boxes into world-space point samples without
pulling the LichtFeld host into :mod:`core`.
"""

from __future__ import annotations

import numpy as np

from .types import BlockBox, GroundFrame


_CORNER_SIGNS = np.array(
    (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
    dtype=np.float64,
)

_EDGE_INDICES = np.array(
    (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ),
    dtype=np.intp,
)


def box_corners(box: BlockBox) -> np.ndarray:
    """Return the eight box corners in ground-frame coordinates."""
    return box.min_g + _CORNER_SIGNS * box.extent


def box_wireframe_points(
    box: BlockBox, frame: GroundFrame, *, samples_per_edge: int = 16
) -> np.ndarray:
    """Return evenly spaced world-space samples on all twelve box edges.

    LichtFeld's public scene API has colored point clouds but no world-space
    line primitive.  A small point cloud sampled along each edge gives the
    viewport adapter a durable, non-destructive wireframe proxy.
    """
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least 2")

    corners_world = frame.to_world(box_corners(box))
    edges = corners_world[_EDGE_INDICES]
    fraction = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float64)
    return (
        edges[:, 0, None, :] * (1.0 - fraction)[None, :, None]
        + edges[:, 1, None, :] * fraction[None, :, None]
    ).reshape(-1, 3)
