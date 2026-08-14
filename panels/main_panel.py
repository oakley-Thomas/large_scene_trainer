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
        ui.text_disabled("Camera assignment: centre inside context (frustum pending ticket 1.3).")
        if ui.button_styled("Export blocks", "primary"):
            self._operator.execute(None)
        ui.text_wrapped(self._operator.last_status)
