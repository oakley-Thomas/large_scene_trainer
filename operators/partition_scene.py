"""Host adapter for the deterministic on-disk COLMAP block exporter."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import lichtfeld as lf
from lfs_plugins.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from lfs_plugins.types import Operator

from ..adapters.viewport_preview import (
    ViewportPreviewError,
    show_block_preview,
)
from ..adapters.splat_io import SplatIOError, load_cropped_blocks
from ..core.assignment_report import BlockAssignmentRow, block_assignment_rows
from ..core.camera_assign import CameraAssignmentConfig, frustum_config_from_diagnostics
from ..core.colmap_io import load_cameras
from ..core.export import ExportError, export_dataset
from ..core.job_generation import (
    JobGenerationConfig,
    JobGenerationError,
    TRAINING_STRATEGIES,
    generate_jobs,
)
from ..core.gpu_workers import WorkerRunnerError, load_job_status, load_merge_status
from ..core.merge import (
    MergeError,
    load_block_artifacts,
    require_complete_crops,
    validate_source_hashes,
)
from ..core.partition import config_from_diagnostics
from ..core.preview_layout import PreviewLayoutError, load_exported_layout
from ..core.trajectory_diagnostics import (
    load_trajectory_diagnostics,
    write_trajectory_diagnostics,
)
from ..core.trajectory_plane import fit_trajectory_plane
from ..core.types import SegmentationConfig


class PartitionScene(Operator):
    """Partition a parent COLMAP dataset and export its training blocks."""

    label = "Export Training Blocks"
    description = "Create per-block COLMAP datasets below the selected dataset root"
    options = {"BLOCKING"}

    dataset_root = StringProperty(
        default="", subtype="DIR_PATH", name="Dataset root", description="Parent COLMAP dataset"
    )
    target_blocks = IntProperty(default=4, min=1, name="Target blocks")
    core_extent_su = FloatProperty(default=1.0, min=1e-12, name="Core extent (scene units)")
    max_cameras_per_block = IntProperty(default=3, min=3, name="Maximum cameras per block")
    lateral_margin_su = FloatProperty(default=1.0, min=1e-12, name="Lateral margin (scene units)")
    context_margin_frac = FloatProperty(default=0.1, min=1e-12, name="Context margin fraction")
    z_pad_down_su = FloatProperty(default=1.0, min=1e-12, name="Downward z padding (scene units)")
    z_pad_up_su = FloatProperty(default=1.0, min=1e-12, name="Upward z padding (scene units)")
    merge_slack = FloatProperty(default=1.0, min=1.0, name="Merge slack")
    min_cameras_per_block = IntProperty(default=32, min=1, name="Minimum assigned cameras")
    camera_assignment_predicate = EnumProperty(
        items=[
            ("centre", "Centre inside context", "Conservative fallback assignment"),
            (
                "frustum",
                "Frustum intersects context",
                "Use only after tuning the far plane for this dataset",
            ),
        ],
        default="centre",
        name="Camera assignment",
    )
    frustum_far_su = FloatProperty(
        default=0.0,
        min=0.0,
        name="Frustum far plane (scene units)",
        description="Required for frustum assignment; select it with tune_frustum_far.py",
    )
    training_strategy = EnumProperty(
        items=[
            (strategy, strategy, f"Use LichtFeld's {strategy} optimization strategy")
            for strategy in TRAINING_STRATEGIES
        ],
        default="mrnf",
        name="Training strategy",
    )
    max_cap_per_camera = IntProperty(
        default=3_000,
        min=1,
        name="Maximum Gaussians per camera",
    )
    max_width = IntProperty(
        default=3_840,
        min=0,
        name="Training maximum image width",
    )
    training_run_dir = StringProperty(
        default="",
        subtype="DIR_PATH",
        name="Training run directory",
        description="Optional run directory created by scripts/run_gpu_workers.py",
    )

    _STATUS_REFRESH_SECONDS = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._validated_root: Path | None = None
        self._preview_blocks = ()
        self._preview_frame = None
        self._status_cache_key: tuple[str, str] | None = None
        self._status_cache_at = 0.0
        self._cached_job_status = ()
        self._cached_merge_status = None
        self.last_status = "Choose a parent dataset and validate it."

    @classmethod
    def poll(cls, context) -> bool:
        # Export operates entirely on the selected dataset, not the active scene.
        return True

    def _root(self) -> Path:
        if not self.dataset_root.strip():
            raise ExportError("choose a parent COLMAP dataset root first")
        return Path(self.dataset_root).expanduser().resolve()

    @property
    def has_dataset_root(self) -> bool:
        """Whether status polling has enough input to touch the filesystem."""
        return bool(self.dataset_root.strip())

    def _refresh_status_cache(self) -> None:
        """Rate-limit panel status reads; ``draw`` can run hundreds of times per second."""
        key = (self.dataset_root.strip(), self.training_run_dir.strip())
        now = monotonic()
        if key == self._status_cache_key and now - self._status_cache_at < self._STATUS_REFRESH_SECONDS:
            return
        self._status_cache_key = key
        self._status_cache_at = now
        self._cached_job_status = ()
        self._cached_merge_status = None
        if not key[0]:
            return
        try:
            root = self._root()
            self._cached_job_status = load_job_status(root, key[1] or None)
            self._cached_merge_status = load_merge_status(root, key[1] or None)
        except (OSError, ValueError, WorkerRunnerError) as exc:
            # Status reads are passive UI polling. Keep the error visible in
            # the panel but never emit it for every redraw.
            self.last_status = f"Training-status read failed: {exc}"

    def _segmentation_config(self) -> SegmentationConfig:
        return SegmentationConfig(
            core_extent_su=self.core_extent_su,
            max_cameras_per_block=self.max_cameras_per_block,
            lateral_margin_su=self.lateral_margin_su,
            context_margin_frac=self.context_margin_frac,
            z_pad_down_su=self.z_pad_down_su,
            z_pad_up_su=self.z_pad_up_su,
            merge_slack=self.merge_slack,
        )

    def _assignment_config(self, diagnostics) -> CameraAssignmentConfig:
        """Build the selected assignment mode from the current dataset diagnostics."""
        if self.camera_assignment_predicate == "centre":
            return CameraAssignmentConfig(
                predicate="centre", min_cameras_per_block=self.min_cameras_per_block
            )
        if self.frustum_far_su <= 0.0:
            raise ExportError(
                "set a positive Frustum far plane from tune_frustum_far.py before exporting"
            )
        return frustum_config_from_diagnostics(
            diagnostics,
            frustum_far_su=self.frustum_far_su,
            min_cameras_per_block=self.min_cameras_per_block,
        )

    def validate_dataset(self) -> bool:
        """Populate controls from diagnostics-derived defaults for this dataset."""
        try:
            root = self._root()
            cameras = load_cameras(root / "sparse" / "0")
            frame, diagnostics = fit_trajectory_plane(cameras)
            write_trajectory_diagnostics(root, frame, diagnostics)
            defaults = config_from_diagnostics(diagnostics, target_blocks=self.target_blocks)
            for field in (
                "core_extent_su",
                "max_cameras_per_block",
                "lateral_margin_su",
                "context_margin_frac",
                "z_pad_down_su",
                "z_pad_up_su",
                "merge_slack",
            ):
                setattr(self, field, getattr(defaults, field))
            self._validated_root = root
            self.last_status = (
                f"Validated {len(cameras):,} images; wrote canonical frame; arc length "
                f"{diagnostics.trajectory_arc_length:.6g} scene units."
            )
            lf.log.info(f"PartitionScene: {self.last_status}")
            return True
        except (OSError, ValueError, ExportError) as exc:
            self._validated_root = None
            self.last_status = f"Validation failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return False

    def execute(self, context) -> set:
        try:
            root = self._root()
            if self._validated_root != root:
                raise ExportError(
                    "validate this dataset before exporting so controls have scene-unit defaults"
                )
            diagnostics = load_trajectory_diagnostics(root).diagnostics
            summary = export_dataset(
                root,
                self._segmentation_config(),
                self._assignment_config(diagnostics),
            )
            self._preview_blocks = summary.blocks_definition
            self._preview_frame = summary.ground_frame
            self.last_status = (
                f"Exported {len(summary.blocks)} blocks to {summary.blocks_dir}; "
                f"assignment={summary.assignment.predicate}; "
                f"bind-mount {summary.dataset_root} for training."
            )
            lf.log.info(f"PartitionScene: {self.last_status}")
            return {"FINISHED"}
        except (OSError, ValueError, ExportError) as exc:
            self.last_status = f"Export failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return {"CANCELLED"}

    @property
    def has_preview_data(self) -> bool:
        """Whether this session has a successfully exported block layout."""
        return bool(self._preview_blocks and self._preview_frame is not None)

    def assignment_rows(self) -> tuple[BlockAssignmentRow, ...]:
        """Return a panel-ready summary from this or a prior successful export."""
        if not self.has_preview_data:
            try:
                self._preview_blocks, self._preview_frame = load_exported_layout(self._root())
            except (OSError, ValueError, PreviewLayoutError):
                return ()
        return block_assignment_rows(self._preview_blocks)

    def show_preview(self) -> bool:
        """Draw core and context wireframes in the currently loaded parent scene."""
        if not self.has_preview_data:
            try:
                self._preview_blocks, self._preview_frame = load_exported_layout(self._root())
            except (OSError, ValueError, PreviewLayoutError) as exc:
                self.last_status = f"Preview failed: {exc}"
                lf.log.error(f"PartitionScene: {self.last_status}")
                return False
        try:
            summary = show_block_preview(self._preview_blocks, self._preview_frame)
            if summary.already_visible:
                self.last_status = (
                    "Block preview is already visible; reload the parent dataset to refresh it."
                )
            else:
                self.last_status = (
                    f"Viewport preview: {summary.block_count} blocks; cyan=core, "
                    f"orange=context. Reload the parent dataset to clear it."
                )
            lf.log.info(f"PartitionScene: {self.last_status}")
            return True
        except ViewportPreviewError as exc:
            self.last_status = f"Preview failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return False

    def generate_training_jobs(self) -> bool:
        """Publish portable worker jobs from an existing block export."""
        try:
            summary = generate_jobs(
                self._root(),
                JobGenerationConfig(
                    strategy=self.training_strategy,
                    max_cap_per_camera=self.max_cap_per_camera,
                    max_width=self.max_width,
                ),
            )
            self.last_status = (
                f"Generated {len(summary.jobs)} portable training jobs: {summary.job_file}"
            )
            lf.log.info(f"PartitionScene: {self.last_status}")
            return True
        except (OSError, ValueError, JobGenerationError) as exc:
            self.last_status = f"Job generation failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return False

    def job_status_rows(self):
        """Return jobs plus their current queue state for the panel."""
        self._refresh_status_cache()
        return self._cached_job_status

    def merge_status(self):
        """Return crop/merge progress for the currently selected training run."""
        self._refresh_status_cache()
        return self._cached_merge_status

    def show_cropped_blocks(self) -> bool:
        """Load completed crop artifacts as separate nodes for seam inspection."""
        try:
            if not self.training_run_dir.strip():
                raise MergeError("choose a training run directory first")
            _, artifacts = load_block_artifacts(self._root())
            validate_source_hashes(self._root(), artifacts)
            crops = require_complete_crops(self.training_run_dir, artifacts)
            summary = load_cropped_blocks(crops)
            self.last_status = f"Loaded {len(summary)} cropped block nodes for seam inspection."
            lf.log.info(f"PartitionScene: {self.last_status}")
            return True
        except (OSError, ValueError, MergeError, SplatIOError) as exc:
            self.last_status = f"Load cropped blocks failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return False
