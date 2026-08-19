"""Host adapter for the deterministic on-disk COLMAP block exporter."""

from __future__ import annotations

from pathlib import Path

import lichtfeld as lf
from lfs_plugins.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from lfs_plugins.types import Operator

from ..adapters.viewport_preview import (
    ViewportPreviewError,
    show_block_preview,
)
from ..core.camera_assign import CameraAssignmentConfig, frustum_config_from_diagnostics
from ..core.colmap_io import load_cameras
from ..core.export import ExportError, export_dataset
from ..core.partition import config_from_diagnostics
from ..core.preview_layout import PreviewLayoutError, load_exported_layout
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

    def __init__(self) -> None:
        super().__init__()
        self._validated_root: Path | None = None
        self._preview_blocks = ()
        self._preview_frame = None
        self.last_status = "Choose a parent dataset and validate it."

    @classmethod
    def poll(cls, context) -> bool:
        # Export operates entirely on the selected dataset, not the active scene.
        return True

    def _root(self) -> Path:
        if not self.dataset_root.strip():
            raise ExportError("choose a parent COLMAP dataset root first")
        return Path(self.dataset_root).expanduser().resolve()

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
            _, diagnostics = fit_trajectory_plane(cameras)
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
                f"Validated {len(cameras):,} images; arc length "
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
            cameras = load_cameras(root / "sparse" / "0")
            _, diagnostics = fit_trajectory_plane(cameras)
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
