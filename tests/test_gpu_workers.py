"""Tests for the host-independent, filesystem-backed GPU worker queue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from large_scene_trainer.core.gpu_workers import (
    LaunchConfig,
    QueueStore,
    WorkerRunnerError,
    build_launch_command,
    load_job_status,
    load_jobs,
    parse_gpu_ids,
    resolve_mount_ancestor,
    run_workers,
)


def _write_jobs(root: Path, job_ids: tuple[str, ...] = ("block_000", "block_001")) -> None:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, job_id in enumerate(job_ids):
        block = root / "blocks" / job_id
        block.mkdir(parents=True, exist_ok=True)
        if not (block / "images").exists():
            (block / "images").symlink_to("../../images")
        data_path = f"blocks/{job_id}"
        output_path = f"outputs/{job_id}"
        jobs.append(
            {
                "job_id": job_id,
                "block_id": index,
                "data_path": data_path,
                "output_path": output_path,
                "argv": [
                    "--headless",
                    "--train",
                    "--data-path",
                    data_path,
                    "--output-path",
                    output_path,
                ],
            }
        )
    (root / "jobs.json").write_text(json.dumps({"schema": 1, "jobs": jobs}), encoding="utf-8")


def _native_config(root: Path, run_dir: Path) -> LaunchConfig:
    return LaunchConfig(
        dataset_root=root.resolve(),
        run_dir=run_dir.resolve(),
        trainer_mode="native",
        lichtfeld_bin="/bin/true",
    )


def test_load_jobs_rejects_duplicate_ids_and_path_disagreement(tmp_path):
    _write_jobs(tmp_path, ("block_000", "block_000"))
    with pytest.raises(WorkerRunnerError, match="repeats job_id"):
        load_jobs(tmp_path)

    _write_jobs(tmp_path, ("block_000",))
    document = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
    document["jobs"][0]["argv"][3] = "../unsafe"
    (tmp_path / "jobs.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorkerRunnerError, match="differs from its declared path"):
        load_jobs(tmp_path)

    document["jobs"][0]["job_id"] = "../unsafe"
    (tmp_path / "jobs.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorkerRunnerError, match="unsafe job_id"):
        load_jobs(tmp_path)


def test_queue_claim_recovery_and_explicit_failed_retry(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jobs(dataset, ("block_000",))
    jobs = load_jobs(dataset)
    queue = QueueStore(tmp_path / "run")
    queue.initialize(jobs, dataset.resolve(), retry_failed=False)
    claimed = queue.claim("worker", "0")
    assert claimed is not None
    job, running_path = claimed
    state = json.loads(running_path.read_text(encoding="utf-8"))
    state["process_id"] = 999_999_999
    running_path.write_text(json.dumps(state), encoding="utf-8")

    QueueStore(tmp_path / "run").initialize(jobs, dataset.resolve(), retry_failed=False)
    assert queue.counts() == {"pending": 1, "running": 0, "succeeded": 0, "failed": 0}

    claimed = queue.claim("worker", "0")
    assert claimed is not None
    _, running_path = claimed
    queue.complete(running_path, command=["false"], log_path=tmp_path / "job.log", exit_code=1)
    assert queue.counts()["failed"] == 1
    queue.initialize(jobs, dataset.resolve(), retry_failed=True)
    assert queue.counts() == {"pending": 1, "running": 0, "succeeded": 0, "failed": 0}
    assert job.job_id == "block_000"


def test_native_workers_complete_jobs_and_keep_succeeded_jobs_on_restart(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jobs(dataset)
    run_dir = tmp_path / "run"

    first = run_workers(
        dataset,
        run_dir,
        gpu_ids=("0", "1"),
        trainer_mode="native",
        lichtfeld_bin="/bin/true",
    )
    assert first.successful
    assert first.counts == {"pending": 0, "running": 0, "succeeded": 2, "failed": 0}
    assert (run_dir / "logs" / "block_000.log").is_file()

    second = run_workers(
        dataset,
        run_dir,
        gpu_ids=("0",),
        trainer_mode="native",
        lichtfeld_bin="/bin/true",
    )
    assert second.counts == first.counts


def test_failed_jobs_are_only_retried_when_requested(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jobs(dataset, ("block_000",))
    run_dir = tmp_path / "run"
    failed = run_workers(
        dataset,
        run_dir,
        gpu_ids=("0",),
        trainer_mode="native",
        lichtfeld_bin="/bin/false",
    )
    assert failed.counts["failed"] == 1

    unchanged = run_workers(
        dataset,
        run_dir,
        gpu_ids=("0",),
        trainer_mode="native",
        lichtfeld_bin="/bin/true",
    )
    assert unchanged.counts["failed"] == 1

    retried = run_workers(
        dataset,
        run_dir,
        gpu_ids=("0",),
        trainer_mode="native",
        lichtfeld_bin="/bin/true",
        retry_failed=True,
    )
    assert retried.successful


def test_resolve_mount_ancestor_follows_multiple_image_links(tmp_path):
    calibration = tmp_path / "camera_calibration"
    dataset = calibration / "train"
    _write_jobs(dataset, ("block_000",))
    (dataset / "images").rmdir()
    (calibration / "images").mkdir()
    block_images = dataset / "blocks" / "block_000" / "images"
    block_images.unlink()
    block_images.symlink_to("../../../images")
    jobs = load_jobs(dataset)

    assert resolve_mount_ancestor(dataset, jobs) == calibration.resolve()


def test_container_command_uses_safe_mount_and_in_container_image_assertion(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jobs(dataset, ("block_000",))
    job = load_jobs(dataset)[0]
    config = LaunchConfig(
        dataset_root=dataset.resolve(),
        run_dir=(tmp_path / "run").resolve(),
        trainer_mode="container",
        image="example/lf:latest",
    )
    command, environment = build_launch_command(
        job, config, "container", resolve_mount_ancestor(dataset, (job,))
    )

    assert "--entrypoint" in command
    assert "/bin/sh" in command
    assert any(entry.endswith(":/input:ro") for entry in command)
    assert "BLOCK_IMAGES=/input/blocks/block_000/images" in command
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "missing block images" in command[command.index("-c") + 1]
    assert environment == {"CUDA_VISIBLE_DEVICES": ""}


def test_parse_gpu_ids_rejects_empty_and_duplicate_values():
    assert parse_gpu_ids("0, 2") == ("0", "2")
    with pytest.raises(WorkerRunnerError, match="comma-separated"):
        parse_gpu_ids("0,0")
    with pytest.raises(WorkerRunnerError, match="comma-separated"):
        parse_gpu_ids(" , ")


def test_load_job_status_combines_jobs_with_read_only_queue_state(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jobs(dataset)
    jobs = load_jobs(dataset)
    queue = QueueStore(tmp_path / "run")
    queue.initialize(jobs, dataset.resolve(), retry_failed=False)
    job, running_path = queue.claim("worker", "0")
    queue.complete(running_path, command=["true"], log_path=tmp_path / "job.log", exit_code=0)

    rows = load_job_status(dataset, tmp_path / "run")

    assert [(row.job_id, row.state, row.exit_code) for row in rows] == [
        (job.job_id, "succeeded", 0),
        ("block_001", "pending", None),
    ]
