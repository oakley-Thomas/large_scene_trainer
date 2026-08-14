"""Host adapter for the deterministic on-disk COLMAP block exporter."""

from __future__ import annotations

from pathlib import Path

import lichtfeld as lf
from lfs_plugins.props import FloatProperty, IntProperty, StringProperty
from lfs_plugins.types import Operator

from ..core.camera_assign import CameraAssignmentConfig
from ..core.colmap_io import load_cameras
from ..core.export import ExportError, export_dataset
from ..core.partition import config_from_diagnostics
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

    def __init__(self) -> None:
        super().__init__()
        self._validated_root: Path | None = None
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
            summary = export_dataset(
                root,
                self._segmentation_config(),
                CameraAssignmentConfig(
                    predicate="centre", min_cameras_per_block=self.min_cameras_per_block
                ),
            )
            self.last_status = (
                f"Exported {len(summary.blocks)} blocks to {summary.blocks_dir}; "
                f"bind-mount {summary.dataset_root} for training."
            )
            lf.log.info(f"PartitionScene: {self.last_status}")
            return {"FINISHED"}
        except (OSError, ValueError, ExportError) as exc:
            self.last_status = f"Export failed: {exc}"
            lf.log.error(f"PartitionScene: {self.last_status}")
            return {"CANCELLED"}
