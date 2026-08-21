"""Controls for the disk-backed large-scene partition/export workflow."""

from __future__ import annotations

import lichtfeld as lf

from .. import __version__
from ..operators.partition_scene import PartitionScene


class MainPanel(lf.ui.Panel):
    id = "large_scene_trainer.main"
    label = "Large Scene"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200
    poll_dependencies = set()

    def __init__(self) -> None:
        self._operator = PartitionScene.get_instance()

    def draw(self, ui):
        ui.heading("Large Scene Trainer")
        ui.text_disabled(f"v{__version__} — COLMAP block export")
        changed, path = ui.path_input(
            "Dataset root",
            self._operator.dataset_root,
            folder_mode=True,
            dialog_title="COLMAP dataset",
        )
        if changed:
            self._operator.dataset_root = path
            self._operator._validated_root = None
        ui.prop(self._operator, "target_blocks")
        if ui.button("Validate dataset"):
            self._operator.validate_dataset()

        if ui.collapsing_header("Segmentation settings", default_open=True):
            for field in (
                "core_extent_su",
                "max_cameras_per_block",
                "lateral_margin_su",
                "context_margin_frac",
                "z_pad_down_su",
                "z_pad_up_su",
                "merge_slack",
                "min_cameras_per_block",
            ):
                ui.prop(self._operator, field)
        if ui.collapsing_header("Camera assignment", default_open=True):
            ui.prop(self._operator, "camera_assignment_predicate")
            if self._operator.camera_assignment_predicate == "frustum":
                ui.prop(self._operator, "frustum_far_su")
                ui.text_wrapped(
                    "Near plane is derived from the dataset's median frame spacing. "
                    "Run scripts/tune_frustum_far.py before setting the far plane."
                )
            else:
                ui.text_disabled("Centre assignment is the conservative default.")
        if ui.button_styled("Export blocks", "primary"):
            self._operator.execute(None)
        if ui.collapsing_header("Training jobs", default_open=True):
            ui.prop(self._operator, "training_strategy")
            ui.prop(self._operator, "max_cap_per_camera")
            ui.prop(self._operator, "max_width")
            ui.text_disabled("Writes portable jobs.json; it does not start training.")
            if ui.button("Generate jobs"):
                self._operator.generate_training_jobs()
        if ui.collapsing_header("Training status", default_open=True):
            changed, path = ui.path_input(
                "Training run directory",
                self._operator.training_run_dir,
                folder_mode=True,
                dialog_title="GPU worker run directory",
            )
            if changed:
                self._operator.training_run_dir = path
            if not self._operator.has_dataset_root:
                ui.text_disabled("Select a parent COLMAP Dataset root to view training status.")
            else:
                rows = self._operator.job_status_rows()
                if not rows:
                    ui.text_disabled("Generate jobs first, then select an optional training run directory.")
                elif ui.begin_table("block_training_status", 5):
                    ui.table_setup_column("Block")
                    ui.table_setup_column("Job")
                    ui.table_setup_column("Cameras")
                    ui.table_setup_column("State")
                    ui.table_setup_column("Exit")
                    ui.table_headers_row()
                    for row in rows:
                        ui.table_next_row()
                        ui.table_next_column()
                        ui.label(str(row.block_id))
                        ui.table_next_column()
                        ui.label(row.job_id)
                        ui.table_next_column()
                        ui.label("—" if row.camera_count is None else str(row.camera_count))
                        ui.table_next_column()
                        ui.label(row.state)
                        ui.table_next_column()
                        ui.label("—" if row.exit_code is None else str(row.exit_code))
                    ui.end_table()
                merge = self._operator.merge_status()
                if merge is not None:
                    ui.text_disabled(
                        "Crop/merge: "
                        f"{merge.state} ({merge.completed_crops}/{merge.total_crops} crops); "
                        f"RAD: {merge.rad_state}"
                    )
                    if merge.merged_ply is not None:
                        ui.text_disabled(f"Merged PLY: {merge.merged_ply}")
                    if merge.rad_path is not None:
                        ui.text_disabled(f"RAD: {merge.rad_path}")
                    if merge.state == "merged":
                        ui.prop(self._operator, "rad_converter_bin")
                        if ui.button_styled("Export merged RAD", "primary"):
                            self._operator.export_merged_rad()
            if ui.button("Load cropped blocks for seam inspection"):
                self._operator.show_cropped_blocks()
        if ui.collapsing_header("Viewport preview", default_open=True):
            ui.text_disabled("Cyan wireframes are cores; orange wireframes are contexts.")
            ui.text_disabled("Reload the parent dataset to clear or refresh the preview.")
            if ui.button("Show block boxes"):
                self._operator.show_preview()
        if ui.collapsing_header("Assignment table", default_open=True):
            rows = self._operator.assignment_rows()
            if not rows:
                ui.text_disabled("No existing block export selected.")
            elif ui.begin_table("block_assignment_summary", 4):
                ui.table_setup_column("Block")
                ui.table_setup_column("Cameras")
                ui.table_setup_column("Frame span")
                ui.table_setup_column("Core extent (u, v, h)")
                ui.table_headers_row()
                for row in rows:
                    ui.table_next_row()
                    ui.table_next_column()
                    ui.label(str(row.block_id))
                    ui.table_next_column()
                    ui.label(str(row.camera_count))
                    ui.table_next_column()
                    ui.label(f"{row.frame_start}–{row.frame_end}")
                    ui.table_next_column()
                    extent = tuple(
                        maximum - minimum for minimum, maximum in zip(row.core_min, row.core_max)
                    )
                    ui.label(" × ".join(f"{value:.3g}" for value in extent))
                ui.end_table()
        ui.text_wrapped(self._operator.last_status)
