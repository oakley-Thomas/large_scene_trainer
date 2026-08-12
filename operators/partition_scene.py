"""Partition operator — a no-op for ticket 0.6.

Phase 1 fills `execute()` by calling into `core.partition`, translating the
scene through `adapters.scene_read` first. The operator itself must never
contain geometry logic: keep it a five-line adapter -> core -> adapter call so
the partitioner stays testable without the host.
"""

import lichtfeld as lf
from lfs_plugins.types import Operator


class PartitionScene(Operator):
    label = "Partition Scene"
    description = "Split the loaded scene into overlapping training blocks"
    options = set()  # no UNDO: this writes files, it does not mutate the scene

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        lf.log.info(
            f"PartitionScene: stub — scene has {scene.total_gaussian_count:,} gaussians"
        )
        return {"FINISHED"}
