"""Restart-safe local GPU workers for generated LichtFeld training jobs.

The module deliberately uses only the Python standard library so the same
runner works on a desktop machine and a remote GPU host.  Runtime state is a
small filesystem queue: moving a job between state directories is its claim
operation, making the queue inspectable without a service or cloud SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Iterable, Sequence
import uuid

from .merge import (
    BlockArtifact,
    MergeError,
    crop_is_complete,
    crop_receipt_matches,
    crop_paths,
    load_block_artifacts,
    merge_cropped_plys,
    merge_is_complete,
    merge_receipt_matches,
    merged_paths,
    require_complete_crops,
    validate_source_hashes,
    write_merge_receipt,
)


JOBS_SCHEMA_VERSION = 1
QUEUE_SCHEMA_VERSION = 1
QUEUE_STATES = ("pending", "running", "succeeded", "failed")

_CROP_BOOTSTRAP = '''import lichtfeld as lf
lf.plugins.load("large_scene_trainer")
from lfs_plugins.large_scene_trainer.adapters.splat_io import register_training_crop
register_training_crop()
'''


class WorkerRunnerError(ValueError):
    """Raised when a jobs file, queue, or launch configuration is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_crop_bootstrap(run_dir: Path) -> Path:
    """Publish the tiny host-interpreter callback script mounted into containers."""
    target = run_dir / "runtime" / "post_train_crop.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_CROP_BOOTSTRAP, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkerRunnerError(f"cannot read {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkerRunnerError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerRunnerError(f"{description} must be a JSON object: {path}")
    return value


def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerRunnerError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise WorkerRunnerError(f"{field} must be a relative path without '..'")
    return path


@dataclass(frozen=True)
class WorkerJob:
    """Validated schema-1 ``jobs.json`` entry consumed by a GPU worker."""

    job_id: str
    block_id: int
    data_path: Path
    output_path: Path
    argv: tuple[str, ...]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> "WorkerJob":
        if not isinstance(value, dict):
            raise WorkerRunnerError("jobs.json contains a non-object job entry")
        job_id = value.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise WorkerRunnerError("job entry has an invalid job_id")
        if Path(job_id).name != job_id or job_id in {".", ".."}:
            raise WorkerRunnerError("job entry has an unsafe job_id")
        block_id = value.get("block_id")
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0:
            raise WorkerRunnerError(f"job {job_id!r} has an invalid block_id")
        argv = value.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
            raise WorkerRunnerError(f"job {job_id!r} has an invalid argv")
        data_path = _relative_path(value.get("data_path"), f"job {job_id!r} data_path")
        output_path = _relative_path(value.get("output_path"), f"job {job_id!r} output_path")
        for flag, expected in (
            ("--data-path", data_path.as_posix()),
            ("--output-path", output_path.as_posix()),
        ):
            try:
                position = argv.index(flag)
                actual = argv[position + 1]
            except (ValueError, IndexError) as exc:
                raise WorkerRunnerError(f"job {job_id!r} argv is missing {flag}") from exc
            if actual != expected:
                raise WorkerRunnerError(
                    f"job {job_id!r} {flag} differs from its declared path"
                )
        return cls(job_id, block_id, data_path, output_path, tuple(argv), dict(value))


def load_jobs(dataset_root: str | Path) -> tuple[WorkerJob, ...]:
    """Load the immutable, schema-1 job description authored by ticket 2.1."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise WorkerRunnerError(f"dataset root does not exist: {root}")
    document = _read_json(root / "jobs.json", "jobs file")
    if document.get("schema") != JOBS_SCHEMA_VERSION:
        raise WorkerRunnerError("jobs.json has an unsupported schema")
    values = document.get("jobs")
    if not isinstance(values, list) or not values:
        raise WorkerRunnerError("jobs.json must contain a non-empty jobs list")
    jobs = tuple(WorkerJob.from_dict(value) for value in values)
    seen = set()
    for job in jobs:
        if job.job_id in seen:
            raise WorkerRunnerError(f"jobs.json repeats job_id {job.job_id!r}")
        seen.add(job.job_id)
    return tuple(sorted(jobs, key=lambda job: (job.block_id, job.job_id)))


def discover_gpu_ids(nvidia_smi: str = "nvidia-smi") -> tuple[str, ...]:
    """Return visible GPU indices using the host's NVIDIA utility."""
    executable = shutil.which(nvidia_smi)
    if executable is None:
        raise WorkerRunnerError(f"cannot detect GPUs: {nvidia_smi!r} is not on PATH")
    result = subprocess.run(
        [executable, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown nvidia-smi error"
        raise WorkerRunnerError(f"cannot detect GPUs: {detail}")
    gpu_ids = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not gpu_ids:
        raise WorkerRunnerError("cannot detect GPUs: nvidia-smi reported no visible GPUs")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise WorkerRunnerError("cannot detect GPUs: nvidia-smi reported duplicate indices")
    return gpu_ids


def parse_gpu_ids(value: str) -> tuple[str, ...]:
    gpu_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise WorkerRunnerError("--gpu-ids must be a comma-separated list of unique GPU IDs")
    return gpu_ids


def _pid_is_alive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _common_ancestor(paths: Sequence[Path]) -> Path:
    if not paths:
        raise WorkerRunnerError("cannot find a mount ancestor without paths")
    common = Path(os.path.commonpath([str(path) for path in paths]))
    if not common.is_dir():
        raise WorkerRunnerError(f"resolved mount ancestor is not a directory: {common}")
    return common


def resolve_mount_ancestor(dataset_root: Path, jobs: Iterable[WorkerJob]) -> Path:
    """Resolve every image-link chain and return the safe read-only mount root."""
    root = dataset_root.resolve()
    paths = [root]
    for job in jobs:
        block = root / job.data_path
        if not block.is_dir():
            raise WorkerRunnerError(f"job {job.job_id!r} data path is missing: {block}")
        images = block / "images"
        if not images.exists():
            raise WorkerRunnerError(
                f"job {job.job_id!r} images link is missing or dangling: {images}"
            )
        resolved_images = images.resolve()
        if not resolved_images.is_dir():
            raise WorkerRunnerError(
                f"job {job.job_id!r} images target is not a directory: {images}"
            )
        paths.append(resolved_images)
    return _common_ancestor(paths)


def _rewrite_argv(
    job: WorkerJob, data_path: str, output_path: str, crop_callback: str | None
) -> list[str]:
    argv = list(job.argv)
    for flag, replacement in (("--data-path", data_path), ("--output-path", output_path)):
        position = argv.index(flag)
        argv[position + 1] = replacement
    for index, value in enumerate(argv):
        if value == "--centralize":
            if index + 1 == len(argv):
                raise WorkerRunnerError(f"job {job.job_id!r} has a dangling --centralize flag")
            argv[index + 1] = "off"
            break
        if value.startswith("--centralize="):
            argv[index] = "--centralize=off"
            break
    else:
        argv.append("--centralize=off")
    if crop_callback is not None:
        argv.extend(("--python-script", crop_callback))
    return argv


@dataclass(frozen=True)
class LaunchConfig:
    dataset_root: Path
    run_dir: Path
    trainer_mode: str = "auto"
    image: str | None = None
    lichtfeld_bin: str | None = None
    container_engine: str = "docker"
    container_bin: str = "LichtFeld-Studio"
    crop_callback: Path | None = None

    def __post_init__(self) -> None:
        if self.trainer_mode not in {"auto", "container", "native"}:
            raise WorkerRunnerError("trainer_mode must be one of auto, container, native")
        if not self.container_engine:
            raise WorkerRunnerError("container_engine must not be empty")
        if not self.container_bin:
            raise WorkerRunnerError("container_bin must not be empty")


def choose_trainer_mode(config: LaunchConfig) -> str:
    """Select an available runtime without probing or launching a trainer."""
    engine_available = shutil.which(config.container_engine) is not None
    native_available = bool(config.lichtfeld_bin and shutil.which(config.lichtfeld_bin))
    if config.trainer_mode == "container":
        if not config.image:
            raise WorkerRunnerError("container mode requires --image")
        if not engine_available:
            raise WorkerRunnerError(f"container engine is unavailable: {config.container_engine}")
        return "container"
    if config.trainer_mode == "native":
        if not native_available:
            raise WorkerRunnerError("native mode requires an executable --lichtfeld-bin")
        return "native"
    if config.image and engine_available:
        return "container"
    if native_available:
        return "native"
    raise WorkerRunnerError(
        "auto mode needs an available container engine with --image "
        "or an executable --lichtfeld-bin"
    )


def build_launch_command(
    job: WorkerJob, config: LaunchConfig, mode: str, mount_ancestor: Path
) -> tuple[list[str], dict[str, str]]:
    """Build a single trainer command after validating the block image link."""
    block = config.dataset_root / job.data_path
    images = block / "images"
    if not images.exists():
        raise WorkerRunnerError(f"job {job.job_id!r} images link is missing or dangling: {images}")
    output_dir = config.run_dir / "outputs" / job.job_id
    if mode == "native":
        if not config.lichtfeld_bin:
            raise WorkerRunnerError("native mode requires --lichtfeld-bin")
        argv = _rewrite_argv(
            job,
            str(block),
            str(output_dir),
            None if config.crop_callback is None else str(config.crop_callback),
        )
        return [config.lichtfeld_bin, *argv], {
            "CUDA_VISIBLE_DEVICES": "",
            "LST_DATASET_ROOT": str(config.dataset_root),
            "LST_RUN_DIR": str(config.run_dir),
            "LST_BLOCK_ID": str(job.block_id),
        }

    if mode != "container" or not config.image:
        raise WorkerRunnerError("container mode requires --image")
    try:
        container_block = Path("/input") / block.relative_to(mount_ancestor)
    except ValueError as exc:
        raise WorkerRunnerError(f"block is outside its resolved mount ancestor: {block}") from exc
    container_output = Path("/run") / "outputs" / job.job_id
    try:
        container_root = Path("/input") / config.dataset_root.relative_to(mount_ancestor)
    except ValueError as exc:
        raise WorkerRunnerError(
            f"dataset root is outside its resolved mount ancestor: {config.dataset_root}"
        ) from exc
    argv = _rewrite_argv(
        job,
        container_block.as_posix(),
        container_output.as_posix(),
        "/run/runtime/post_train_crop.py" if config.crop_callback is not None else None,
    )
    # The container must include a POSIX shell and expose LichtFeld on PATH (or
    # receive its path through --container-bin).  Shell entrypoint injection
    # validates the symlink *after* Docker's mounts are active.
    preflight = (
        'test -e "$BLOCK_IMAGES" || { echo "missing block images: $BLOCK_IMAGES" >&2; exit 64; }; '
        'exec "$CONTAINER_TRAINER" "$@"'
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "LST_DATASET_ROOT": container_root.as_posix(),
        "LST_RUN_DIR": "/run",
        "LST_BLOCK_ID": str(job.block_id),
    }
    command = [
        config.container_engine,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{mount_ancestor}:/input:ro",
        "-v",
        f"{config.run_dir}:/run:rw",
        "-e",
        f"BLOCK_IMAGES={container_block.as_posix()}/images",
        "-e",
        f"CONTAINER_TRAINER={config.container_bin}",
        "--entrypoint",
        "/bin/sh",
        config.image,
        "-c",
        preflight,
        "sh",
        *argv,
    ]
    entrypoint = command.index("--entrypoint")
    environment_flags: list[str] = []
    for name, value in environment.items():
        # Pass the worker-selected GPU through to Docker; assigning the empty
        # template value here would hide every GPU inside the container.
        rendered = name if name == "CUDA_VISIBLE_DEVICES" else f"{name}={value}"
        environment_flags.extend(("-e", rendered))
    command[entrypoint:entrypoint] = environment_flags
    return command, environment


class QueueStore:
    """Filesystem-backed per-job state machine for one training run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.queue_dir = self.run_dir / "queue"

    def _state_dir(self, state: str) -> Path:
        if state not in QUEUE_STATES:
            raise WorkerRunnerError(f"unknown queue state: {state}")
        return self.queue_dir / state

    def _state_path(self, state: str, job_id: str) -> Path:
        return self._state_dir(state) / f"{job_id}.json"

    def initialize(self, jobs: Sequence[WorkerJob], dataset_root: Path, retry_failed: bool) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(exist_ok=True)
        for state in QUEUE_STATES:
            self._state_dir(state).mkdir(exist_ok=True)
        meta = self.queue_dir / "meta.json"
        expected_ids = [job.job_id for job in jobs]
        if meta.exists():
            current = _read_json(meta, "queue metadata")
            if current.get("schema") != QUEUE_SCHEMA_VERSION:
                raise WorkerRunnerError("queue has an unsupported schema")
            if current.get("dataset_root") != str(dataset_root):
                raise WorkerRunnerError("run directory belongs to a different dataset root")
            if current.get("job_ids") != expected_ids:
                raise WorkerRunnerError(
                    "jobs.json differs from the queue already in this run directory"
                )
        else:
            # A shutdown may happen while a new queue is being populated.  A
            # partial set of pending files is safe to resume; create only the
            # missing states, validate the complete set, then publish metadata.
            existing: dict[str, list[str]] = {job_id: [] for job_id in expected_ids}
            for state_name in QUEUE_STATES:
                for path in self._state_dir(state_name).glob("*.json"):
                    if path.stem not in existing:
                        raise WorkerRunnerError(f"queue contains unexpected job state: {path}")
                    existing[path.stem].append(state_name)
            for job in jobs:
                if not existing[job.job_id]:
                    _write_json_atomically(
                        self._state_path("pending", job.job_id),
                        {"schema": QUEUE_SCHEMA_VERSION, "state": "pending", "job": job.raw},
                    )
                elif len(existing[job.job_id]) != 1:
                    raise WorkerRunnerError(
                        f"queue has invalid state files for job(s): {job.job_id}"
                    )
        self._validate_job_files(expected_ids)
        if not meta.exists():
            _write_json_atomically(
                meta,
                {
                    "schema": QUEUE_SCHEMA_VERSION,
                    "dataset_root": str(dataset_root),
                    "job_ids": expected_ids,
                    "created": _utc_now(),
                },
            )
        self._recover_dead_running()
        if retry_failed:
            for path in sorted(self._state_dir("failed").glob("*.json")):
                self._transition(path, "failed", "pending", {"retried_at": _utc_now()})

    def _validate_job_files(self, expected_ids: Sequence[str]) -> None:
        actual: dict[str, list[str]] = {job_id: [] for job_id in expected_ids}
        for state in QUEUE_STATES:
            for path in self._state_dir(state).glob("*.json"):
                job_id = path.stem
                if job_id not in actual:
                    raise WorkerRunnerError(f"queue contains unexpected job state: {path}")
                actual[job_id].append(state)
        invalid = [job_id for job_id, states in actual.items() if len(states) != 1]
        if invalid:
            raise WorkerRunnerError(
                f"queue has invalid state files for job(s): {', '.join(invalid)}"
            )

    def _recover_dead_running(self) -> None:
        for path in sorted(self._state_dir("running").glob("*.json")):
            state = _read_json(path, "running queue state")
            if not _pid_is_alive(state.get("process_id")):
                self._transition(path, "running", "pending", {"recovered_at": _utc_now()})

    def _transition(
        self, path: Path, old_state: str, new_state: str, updates: dict[str, Any]
    ) -> Path:
        state = _read_json(path, f"{old_state} queue state")
        if state.get("state") != old_state:
            raise WorkerRunnerError(f"queue state file disagrees with its directory: {path}")
        state.update(updates)
        state["state"] = new_state
        target = self._state_path(new_state, path.stem)
        _write_json_atomically(path, state)
        try:
            path.replace(target)
        except FileNotFoundError as exc:
            raise WorkerRunnerError(f"queue state disappeared during transition: {path}") from exc
        return target

    def claim(self, worker_id: str, gpu_id: str) -> tuple[WorkerJob, Path] | None:
        for path in sorted(self._state_dir("pending").glob("*.json")):
            target = self._state_path("running", path.stem)
            try:
                path.replace(target)
            except FileNotFoundError:
                continue
            state = _read_json(target, "claimed queue state")
            if state.get("state") != "pending":
                raise WorkerRunnerError(f"pending queue state is malformed: {target}")
            job = WorkerJob.from_dict(state.get("job"))
            state.update(
                {
                    "state": "running",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "claimed_at": _utc_now(),
                    "worker_process_id": os.getpid(),
                    "process_id": None,
                }
            )
            _write_json_atomically(target, state)
            return job, target
        return None

    def set_process_id(self, running_path: Path, process_id: int) -> None:
        """Associate an acquired job with the trainer process that owns it."""
        state = _read_json(running_path, "running queue state")
        if state.get("state") != "running":
            raise WorkerRunnerError(
                f"queue state file disagrees with its directory: {running_path}"
            )
        state["process_id"] = process_id
        _write_json_atomically(running_path, state)

    def complete(
        self,
        running_path: Path,
        *,
        command: Sequence[str],
        log_path: Path,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        final_state = "succeeded" if exit_code == 0 and error is None else "failed"
        updates: dict[str, Any] = {
            "finished_at": _utc_now(),
            "command": list(command),
            "log_path": str(log_path),
            "exit_code": exit_code,
        }
        if error is not None:
            updates["error"] = error
        self._transition(running_path, "running", final_state, updates)

    def counts(self) -> dict[str, int]:
        return {
            state: sum(1 for _ in self._state_dir(state).glob("*.json"))
            for state in QUEUE_STATES
        }


@dataclass(frozen=True)
class RunSummary:
    counts: dict[str, int]
    mode: str
    merge_state: str = "pending"
    merged_ply: Path | None = None

    @property
    def successful(self) -> bool:
        return (
            self.counts["failed"] == 0
            and self.counts["pending"] == 0
            and self.counts["running"] == 0
            and self.merge_state == "succeeded"
        )


@dataclass(frozen=True)
class JobStatus:
    """Panel-ready status of one generated job and its latest queue state."""

    job_id: str
    block_id: int
    camera_count: int | None
    state: str
    exit_code: int | None


@dataclass(frozen=True)
class MergeStatus:
    """Panel-ready state for the post-training crop/merge artifacts."""

    state: str
    completed_crops: int
    total_crops: int
    merged_ply: Path | None


def load_job_status(dataset_root: str | Path, run_dir: str | Path | None) -> tuple[JobStatus, ...]:
    """Read generated jobs plus optional run state without modifying either tree."""
    jobs = load_jobs(dataset_root)
    states: dict[str, dict[str, Any]] = {}
    if run_dir:
        queue = QueueStore(run_dir)
        if queue.queue_dir.is_dir():
            for state_name in QUEUE_STATES:
                state_dir = queue._state_dir(state_name)
                for path in state_dir.glob("*.json"):
                    state = _read_json(path, f"{state_name} queue state")
                    if state.get("state") != state_name:
                        raise WorkerRunnerError(
                            f"queue state file disagrees with its directory: {path}"
                        )
                    if path.stem in states:
                        raise WorkerRunnerError(f"queue has duplicate state for job {path.stem!r}")
                    states[path.stem] = state
    rows: list[JobStatus] = []
    for job in jobs:
        state = states.get(job.job_id, {})
        camera_count = job.raw.get("camera_count")
        rows.append(
            JobStatus(
                job_id=job.job_id,
                block_id=job.block_id,
                camera_count=camera_count if isinstance(camera_count, int) else None,
                state=str(state.get("state", "not started")),
                exit_code=(
                    state.get("exit_code") if isinstance(state.get("exit_code"), int) else None
                ),
            )
        )
    unexpected = sorted(set(states) - {job.job_id for job in jobs})
    if unexpected:
        raise WorkerRunnerError(f"queue contains state for unknown job {unexpected[0]!r}")
    return tuple(rows)


def load_merge_status(dataset_root: str | Path, run_dir: str | Path | None) -> MergeStatus:
    """Read lightweight crop/merge status without hashing multi-GB PLYs on the UI thread."""
    if not run_dir:
        return MergeStatus("not started", 0, 0, None)
    root = Path(dataset_root).expanduser().resolve()
    try:
        _, artifacts = load_block_artifacts(root)
    except MergeError as exc:
        raise WorkerRunnerError(f"crop/merge input validation failed: {exc}") from exc
    completed = sum(
        crop_receipt_matches(crop_paths(run_dir, artifact.block.block_id), artifact)
        for artifact in artifacts
    )
    output, _ = merged_paths(run_dir)
    if merge_receipt_matches(run_dir, artifacts):
        return MergeStatus("merged", completed, len(artifacts), output)
    if completed:
        return MergeStatus("awaiting merge", completed, len(artifacts), None)
    return MergeStatus("awaiting crops", 0, len(artifacts), None)


def _worker_loop(
    queue: QueueStore,
    config: LaunchConfig,
    mode: str,
    mount_ancestor: Path,
    artifacts: dict[int, BlockArtifact],
    gpu_id: str,
) -> None:
    worker_id = f"{os.getpid()}-gpu-{gpu_id}"
    while True:
        claimed = queue.claim(worker_id, gpu_id)
        if claimed is None:
            return
        job, running_path = claimed
        log_path = config.run_dir / "logs" / f"{job.job_id}.log"
        command: list[str] = []
        try:
            command, environment = build_launch_command(job, config, mode, mount_ancestor)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            (config.run_dir / "outputs" / job.job_id).mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(environment)
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                queue.set_process_id(running_path, process.pid)
                exit_code = process.wait()
                # Docker reserves 125 for failures before the container starts
                # (for example a missing daemon or NVIDIA runtime).  In auto
                # mode that is precisely when the native fallback is useful;
                # a nonzero trainer exit remains a normal failed job.
                if (
                    exit_code == 125
                    and config.trainer_mode == "auto"
                    and config.lichtfeld_bin
                    and shutil.which(config.lichtfeld_bin)
                ):
                    log.write("container launch failed; retrying with native trainer\n")
                    command, environment = build_launch_command(
                        job, config, "native", mount_ancestor
                    )
                    native_env = os.environ.copy()
                    native_env.update(environment)
                    native_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=native_env
                    )
                    queue.set_process_id(running_path, process.pid)
                    exit_code = process.wait()
            if exit_code == 0:
                artifact = artifacts[job.block_id]
                if not crop_is_complete(crop_paths(config.run_dir, job.block_id), artifact):
                    raise WorkerRunnerError(
                        f"job {job.job_id!r} exited successfully but did not produce "
                        "a valid cropped PLY and receipt"
                    )
            queue.complete(running_path, command=command, log_path=log_path, exit_code=exit_code)
        except (OSError, WorkerRunnerError) as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"worker launch error: {exc}\n")
            queue.complete(
                running_path,
                command=command,
                log_path=log_path,
                exit_code=None,
                error=str(exc),
            )


def run_workers(
    dataset_root: str | Path,
    run_dir: str | Path,
    *,
    gpu_ids: Sequence[str],
    trainer_mode: str = "auto",
    image: str | None = None,
    lichtfeld_bin: str | None = None,
    container_engine: str = "docker",
    container_bin: str = "LichtFeld-Studio",
    retry_failed: bool = False,
    dry_run: bool = False,
) -> RunSummary:
    """Run each pending job once, concurrently across the selected GPUs."""
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise WorkerRunnerError("at least one unique GPU ID is required")
    root = Path(dataset_root).expanduser().resolve()
    jobs = load_jobs(root)
    try:
        _, artifacts_tuple = load_block_artifacts(root)
        validate_source_hashes(root, artifacts_tuple)
    except MergeError as exc:
        raise WorkerRunnerError(f"crop/merge input validation failed: {exc}") from exc
    artifacts = {artifact.block.block_id: artifact for artifact in artifacts_tuple}
    if set(artifacts) != {job.block_id for job in jobs}:
        raise WorkerRunnerError("jobs.json does not match the exported block manifests")
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config = LaunchConfig(
        dataset_root=root,
        run_dir=resolved_run_dir,
        trainer_mode=trainer_mode,
        image=image,
        lichtfeld_bin=lichtfeld_bin,
        container_engine=container_engine,
        container_bin=container_bin,
        crop_callback=_write_crop_bootstrap(resolved_run_dir),
    )
    mode = choose_trainer_mode(config)
    mount_ancestor = resolve_mount_ancestor(root, jobs)
    queue = QueueStore(config.run_dir)
    queue.initialize(jobs, root, retry_failed)
    if dry_run:
        for job in jobs:
            build_launch_command(job, config, mode, mount_ancestor)
        return RunSummary(queue.counts(), mode, "dry-run")
    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(queue, config, mode, mount_ancestor, artifacts, gpu_id),
            name=f"gpu-worker-{gpu_id}",
        )
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    counts = queue.counts()
    if counts["failed"] or counts["pending"] or counts["running"]:
        return RunSummary(counts, mode, "pending")
    if merge_is_complete(config.run_dir, artifacts_tuple):
        return RunSummary(counts, mode, "succeeded", merged_paths(config.run_dir)[0])
    try:
        inputs = require_complete_crops(config.run_dir, artifacts_tuple)
        output, _ = merged_paths(config.run_dir)
        merge_cropped_plys(inputs, output)
        write_merge_receipt(config.run_dir, artifacts_tuple)
    except MergeError:
        return RunSummary(counts, mode, "failed")
    return RunSummary(counts, mode, "succeeded", merged_paths(config.run_dir)[0])
