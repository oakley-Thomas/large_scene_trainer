"""Trajectory segmentation invariants on deterministic synthetic captures."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from large_scene_trainer.core import partition
from large_scene_trainer.core.partition import (
    config_from_diagnostics,
    deoverlap,
    partition_trajectory,
)
from large_scene_trainer.core.trajectory_plane import fit_trajectory_plane
from large_scene_trainer.core.types import Block, SegmentationConfig
from .fixtures.trajectories import figure_eight, grid_loop, l_turn, straight_run


def test_config_from_diagnostics_uses_scene_unit_defaults():
    _, diagnostics = fit_trajectory_plane(straight_run(40))
    config = config_from_diagnostics(diagnostics)
    assert config.core_extent_su == diagnostics.trajectory_arc_length / 4
    assert config.max_cameras_per_block == 10
    assert config.lateral_margin_su == 5.0 * diagnostics.median_inter_frame_spacing
    assert config.z_pad_down_su == max(
        2.0 * diagnostics.median_inter_frame_spacing, diagnostics.height_residual_max
    )
    assert config.z_pad_up_su == 12.0 * diagnostics.median_inter_frame_spacing
    assert config.context_margin_frac == 0.10


def _config() -> SegmentationConfig:
    return SegmentationConfig(
        core_extent_su=12.0,
        max_cameras_per_block=16,
        lateral_margin_su=2.0,
        context_margin_frac=0.20,
        z_pad_down_su=1.0,
        z_pad_up_su=6.0,
        # The loop fixtures deliberately interleave sequential blocks.  Their
        # revisit unions are allowed to coalesce; the strict default remains
        # covered by the configuration test above.
        merge_slack=6.0,
    )


def _partition(cameras):
    frame, plane_diagnostics = fit_trajectory_plane(cameras)
    return partition_trajectory(cameras, frame, plane_diagnostics, _config())


@pytest.mark.parametrize("factory", [straight_run, figure_eight, grid_loop, l_turn])
def test_cores_are_disjoint_and_all_cameras_have_one_owner(factory):
    cameras = factory()
    blocks, diagnostics = _partition(cameras)
    assert diagnostics.orphan_cameras == 0
    assert diagnostics.multi_owned_cameras == 0
    for index, first in enumerate(blocks):
        for second in blocks[index + 1 :]:
            assert not first.core.overlaps(second.core)


def test_figure_eight_runs_the_revisit_pass():
    blocks, diagnostics = _partition(figure_eight())
    assert blocks
    assert any(action.startswith("spatial_split(") for action in diagnostics.actions)


@pytest.mark.parametrize("factory", [straight_run, figure_eight, grid_loop, l_turn])
def test_blocks_respect_slack_limited_camera_and_spatial_bounds(factory):
    cameras = factory()
    frame, _ = fit_trajectory_plane(cameras)
    blocks, _ = _partition(cameras)
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in cameras]))
    config = _config()
    for block in blocks:
        assert len(block.camera_indices) <= config.max_cameras_per_block * config.merge_slack
        span = np.ptp(centres_g[list(block.camera_indices)], axis=0)
        assert max(span[:2]) <= config.core_extent_su * config.merge_slack + 1e-12


def test_shuffling_input_is_byte_identical_and_final_ids_are_sequential():
    cameras = figure_eight()
    blocks, diagnostics = _partition(cameras)
    shuffled_blocks, shuffled_diagnostics = _partition(list(reversed(cameras)))
    assert [block.block_id for block in blocks] == list(range(len(blocks)))
    assert [block.to_dict() for block in blocks] == [block.to_dict() for block in shuffled_blocks]
    assert diagnostics.to_dict() == shuffled_diagnostics.to_dict()


def test_contexts_contain_cores_and_adjacent_contexts_overlap():
    cameras = straight_run()
    frame, plane_diagnostics = fit_trajectory_plane(cameras)
    blocks, _ = partition_trajectory(
        cameras, frame, plane_diagnostics, replace(_config(), merge_slack=1.0)
    )
    for block in blocks:
        assert block.context.contains(block.core.min_g)[0]
        assert block.context.contains(block.core.max_g)[0]
    assert any(first.context.overlaps(second.context) for first, second in zip(blocks, blocks[1:]))


def test_block_serialisation_round_trips():
    blocks, _ = _partition(l_turn())
    assert [Block.from_dict(block.to_dict()).to_dict() for block in blocks] == [
        block.to_dict() for block in blocks
    ]


def test_grade_warning_is_embedded_in_partition_diagnostics():
    cameras = straight_run()
    frame, plane_diagnostics = fit_trajectory_plane(cameras)
    poor_grade = replace(
        plane_diagnostics,
        height_residual_rms=0.04 * plane_diagnostics.trajectory_arc_length,
    )
    _, diagnostics = partition_trajectory(cameras, frame, poor_grade, _config())
    assert diagnostics.grade_pct == pytest.approx(4.0)
    assert any("global-plane z-extents invalid" in warning for warning in diagnostics.warnings)


def test_deoverlap_reports_non_convergence_when_its_cap_is_exhausted(monkeypatch):
    cameras = straight_run()
    frame, _ = fit_trajectory_plane(cameras)
    centres_g = frame.to_ground(np.stack([camera.C_world for camera in cameras]))
    initial = partition.segment_sequential(cameras, frame, _config())
    monkeypatch.setattr(partition, "_MAX_DEOVERLAP_ITERATIONS", 0)
    with pytest.raises(RuntimeError, match="did not converge"):
        deoverlap(initial, centres_g, _config())
