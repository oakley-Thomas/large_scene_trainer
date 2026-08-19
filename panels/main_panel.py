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
        if ui.collapsing_header("Viewport preview", default_open=True):
            ui.text_disabled("Cyan wireframes are cores; orange wireframes are contexts.")
            ui.text_disabled("Reload the parent dataset to clear or refresh the preview.")
            if ui.button("Show block boxes"):
                self._operator.show_preview()
        ui.text_wrapped(self._operator.last_status)
