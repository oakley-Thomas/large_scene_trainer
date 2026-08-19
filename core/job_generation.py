"""Portable per-block training job descriptions for the GPU worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid


BLOCKS_DIRNAME = "blocks"
JOBS_FILENAME = "jobs.json"
JOB_SCHEMA_VERSION = 1
TRAINING_STRATEGIES = ("mcmc", "mrnf", "igs+")


class JobGenerationError(ValueError):
    """Raised when an exported block set cannot safely produce training jobs."""


@dataclass(frozen=True)
class JobGenerationConfig:
    """Plugin-selected training settings shared by every generated job."""

    strategy: str = "mrnf"
    max_cap_per_camera: int = 3_000
    max_width: int = 3_840

    def __post_init__(self) -> None:
        if self.strategy not in TRAINING_STRATEGIES:
            raise JobGenerationError(
                f"unsupported training strategy {self.strategy!r}; "
                f"choose from {TRAINING_STRATEGIES}"
            )
        if (
            isinstance(self.max_cap_per_camera, bool)
            or not isinstance(self.max_cap_per_camera, int)
            or self.max_cap_per_camera < 1
        ):
            raise JobGenerationError("max_cap_per_camera must be a positive integer")
        if (
            isinstance(self.max_width, bool)
            or not isinstance(self.max_width, int)
            or self.max_width < 0
        ):
            raise JobGenerationError("max_width must be zero or a positive integer")


@dataclass(frozen=True)
class TrainingJob:
    """One worker-ready, runtime-neutral LichtFeld training command."""

    job_id: str
    block_id: int
    camera_count: int
    max_cap: int
    data_path: str
    output_path: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "block_id": self.block_id,
            "camera_count": self.camera_count,
            "max_cap": self.max_cap,
            "data_path": self.data_path,
            "output_path": self.output_path,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class JobGenerationSummary:
    """Result of atomically publishing a job manifest."""

    job_file: Path
    jobs: tuple[TrainingJob, ...]


def _read_json(path: Path, description: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JobGenerationError(f"cannot read {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JobGenerationError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise JobGenerationError(f"{description} must be a JSON object: {path}")
    return data


def _relative_block_directory(directory: object, block_id: int) -> Path:
    if not isinstance(directory, str) or not directory:
        raise JobGenerationError(f"block {block_id} index entry has no directory")
    relative = Path(directory)
    if relative.is_absolute() or ".." in relative.parts:
        raise JobGenerationError(f"block {block_id} directory must stay under {BLOCKS_DIRNAME}")
    return relative


def _camera_count(manifest_path: Path, block_id: int) -> int:
    manifest = _read_json(manifest_path, f"manifest for block {block_id}")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise JobGenerationError(f"manifest for block {block_id} has no counts object")
    camera_count = counts.get("cameras")
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count < 1:
        raise JobGenerationError(f"manifest for block {block_id} has invalid counts.cameras")
    return camera_count


def _load_jobs(dataset_root: Path, config: JobGenerationConfig) -> tuple[TrainingJob, ...]:
    blocks_dir = dataset_root / BLOCKS_DIRNAME
    index = _read_json(blocks_dir / "index.json", "block index")
    if index.get("schema") != 1:
        raise JobGenerationError("blocks/index.json has an unsupported schema")
    entries = index.get("blocks")
    if not isinstance(entries, list) or not entries:
        raise JobGenerationError("blocks/index.json must contain a non-empty blocks list")

    jobs: list[TrainingJob] = []
    seen_ids: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise JobGenerationError("blocks/index.json contains a non-object block entry")
        block_id = entry.get("block_id")
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0:
            raise JobGenerationError("block index entry has an invalid block_id")
        if block_id in seen_ids:
            raise JobGenerationError(f"blocks/index.json repeats block_id {block_id}")
        seen_ids.add(block_id)
        directory = _relative_block_directory(entry.get("directory"), block_id)
        block_dir = blocks_dir / directory
        if not block_dir.is_dir():
            raise JobGenerationError(f"block {block_id} directory is missing: {block_dir}")
        camera_count = _camera_count(block_dir / "manifest.json", block_id)
        data_path = (Path(BLOCKS_DIRNAME) / directory).as_posix()
        output_path = (Path("outputs") / f"block_{block_id:03d}").as_posix()
        max_cap = camera_count * config.max_cap_per_camera
        argv = (
            "--headless",
            "--train",
            "--data-path",
            data_path,
            "--output-path",
            output_path,
            "--strategy",
            config.strategy,
            "--max-cap",
            str(max_cap),
            "--max-width",
            str(config.max_width),
        )
        jobs.append(
            TrainingJob(
                job_id=f"block_{block_id:03d}",
                block_id=block_id,
                camera_count=camera_count,
                max_cap=max_cap,
                data_path=data_path,
                output_path=output_path,
                argv=argv,
            )
        )
    return tuple(sorted(jobs, key=lambda job: job.block_id))


def _write_jobs_atomically(target: Path, jobs: tuple[TrainingJob, ...]) -> None:
    payload = {"schema": JOB_SCHEMA_VERSION, "jobs": [job.to_dict() for job in jobs]}
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError as exc:
        raise JobGenerationError(f"cannot write jobs file: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def generate_jobs(
    dataset_root: str | Path, config: JobGenerationConfig = JobGenerationConfig()
) -> JobGenerationSummary:
    """Write ``jobs.json`` from exported manifests without starting training."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise JobGenerationError(f"dataset root does not exist: {root}")
    jobs = _load_jobs(root, config)
    target = root / JOBS_FILENAME
    _write_jobs_atomically(target, jobs)
    return JobGenerationSummary(target, jobs)
