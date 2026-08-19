"""Tests for the persisted canonical ticket-1.1 diagnostics artifact."""

from __future__ import annotations

import numpy as np
import pytest

from large_scene_trainer.core.trajectory_diagnostics import (
    TRAJECTORY_DIAGNOSTICS_NAME,
    TrajectoryDiagnosticsError,
    load_trajectory_diagnostics,
    write_trajectory_diagnostics,
)
from large_scene_trainer.core.trajectory_plane import fit_trajectory_plane
from large_scene_trainer.tests.fixtures.trajectories import straight_run


def _source_root(tmp_path):
    root = tmp_path / "dataset"
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "images.bin").write_bytes(b"source-poses")
    return root


def test_persisted_diagnostics_round_trip(tmp_path):
    root = _source_root(tmp_path)
    frame, diagnostics = fit_trajectory_plane(straight_run(8))

    target = write_trajectory_diagnostics(root, frame, diagnostics)
    restored = load_trajectory_diagnostics(root)

    assert target == root / TRAJECTORY_DIAGNOSTICS_NAME
    assert restored.frame.to_dict() == frame.to_dict()
    assert restored.diagnostics.to_dict() == diagnostics.to_dict()


def test_persisted_diagnostics_rejects_changed_pose_file(tmp_path):
    root = _source_root(tmp_path)
    frame, diagnostics = fit_trajectory_plane(straight_run(8))
    write_trajectory_diagnostics(root, frame, diagnostics)
    (root / "sparse" / "0" / "images.bin").write_bytes(b"changed-poses")

    with pytest.raises(TrajectoryDiagnosticsError, match="do not match"):
        load_trajectory_diagnostics(root)
