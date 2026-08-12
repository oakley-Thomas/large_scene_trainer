"""Main panel — deliberately minimal for ticket 0.6.

Exit criterion is "loads without error", not "does something". Phase 1 replaces
the body of `draw()` with the partition controls; the class shell stays as-is.
"""

import lichtfeld as lf

from .. import __version__


class MainPanel(lf.ui.Panel):
    id = "large_scene_trainer.main"
    label = "Large Scene"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200
    poll_dependencies = {lf.ui.PollDependency.SCENE}

    def draw(self, ui):
        ui.heading("Large Scene Trainer")
        ui.text_disabled(f"v{__version__} — skeleton")

        if not lf.has_scene():
            ui.text_disabled("Load a dataset to begin.")
            return

        scene = lf.get_scene()
        ui.label(f"Gaussians: {scene.total_gaussian_count:,}")

        with ui.row() as row:
            row.button("Partition Scene")  # wired to PartitionScene in Phase 1
