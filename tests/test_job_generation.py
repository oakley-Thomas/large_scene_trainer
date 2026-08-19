"""Tests for portable, disk-backed training job generation."""

from __future__ import annotations

import json

import pytest

from large_scene_trainer.core.job_generation import (
    JobGenerationConfig,
    JobGenerationError,
    generate_jobs,
)


def _write_block(root, block_id: int, camera_count: int) -> None:
    block_dir = root / "blocks" / f"block_{block_id:03d}"
    block_dir.mkdir(parents=True)
    (block_dir / "manifest.json").write_text(
        json.dumps({"counts": {"cameras": camera_count}}), encoding="utf-8"
    )


def _write_index(root, entries: list[dict]) -> None:
    blocks_dir = root / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    (blocks_dir / "index.json").write_text(
        json.dumps({"schema": 1, "blocks": entries}), encoding="utf-8"
    )


def test_generate_jobs_uses_sorted_index_entries_and_portable_argv(tmp_path):
    _write_block(tmp_path, 7, 4)
    _write_block(tmp_path, 2, 3)
    _write_index(
        tmp_path,
        [
            {"block_id": 7, "directory": "block_007"},
            {"block_id": 2, "directory": "block_002"},
        ],
    )

    summary = generate_jobs(
        tmp_path, JobGenerationConfig(strategy="igs+", max_cap_per_camera=3_000, max_width=3840)
    )

    assert [job.block_id for job in summary.jobs] == [2, 7]
    assert json.loads(summary.job_file.read_text(encoding="utf-8")) == {
        "schema": 1,
        "jobs": [
            {
                "job_id": "block_002",
                "block_id": 2,
                "camera_count": 3,
                "max_cap": 9_000,
                "data_path": "blocks/block_002",
                "output_path": "outputs/block_002",
                "argv": [
                    "--headless",
                    "--train",
                    "--data-path",
                    "blocks/block_002",
                    "--output-path",
                    "outputs/block_002",
                    "--strategy",
                    "igs+",
                    "--max-cap",
                    "9000",
                    "--max-width",
                    "3840",
                ],
            },
            {
                "job_id": "block_007",
                "block_id": 7,
                "camera_count": 4,
                "max_cap": 12_000,
                "data_path": "blocks/block_007",
                "output_path": "outputs/block_007",
                "argv": [
                    "--headless",
                    "--train",
                    "--data-path",
                    "blocks/block_007",
                    "--output-path",
                    "outputs/block_007",
                    "--strategy",
                    "igs+",
                    "--max-cap",
                    "12000",
                    "--max-width",
                    "3840",
                ],
            },
        ],
    }
    assert not (tmp_path / "outputs").exists()


def test_generator_ignores_stray_directories(tmp_path):
    _write_block(tmp_path, 1, 2)
    _write_block(tmp_path, 9, 99)
    _write_index(tmp_path, [{"block_id": 1, "directory": "block_001"}])

    summary = generate_jobs(tmp_path)

    assert [job.block_id for job in summary.jobs] == [1]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("{", "invalid manifest"),
        (json.dumps({"counts": {"cameras": 0}}), "invalid counts.cameras"),
        (json.dumps({}), "has no counts object"),
    ],
)
def test_generator_failure_preserves_existing_jobs_file(tmp_path, manifest, message):
    _write_block(tmp_path, 0, 1)
    _write_index(tmp_path, [{"block_id": 0, "directory": "block_000"}])
    target = tmp_path / "jobs.json"
    target.write_text("existing jobs", encoding="utf-8")
    (tmp_path / "blocks" / "block_000" / "manifest.json").write_text(manifest, encoding="utf-8")

    with pytest.raises(JobGenerationError, match=message):
        generate_jobs(tmp_path)

    assert target.read_text(encoding="utf-8") == "existing jobs"


def test_generator_rejects_missing_block_directory(tmp_path):
    _write_index(tmp_path, [{"block_id": 0, "directory": "block_000"}])

    with pytest.raises(JobGenerationError, match="directory is missing"):
        generate_jobs(tmp_path)


def test_job_config_rejects_invalid_values():
    with pytest.raises(JobGenerationError, match="max_cap_per_camera"):
        JobGenerationConfig(max_cap_per_camera="3000")
    with pytest.raises(JobGenerationError, match="max_width"):
        JobGenerationConfig(max_width="3840")
