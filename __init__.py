"""Large Scene Trainer — block-partitioned large-scene training for LichtFeld Studio.

Layering rule enforced throughout this package:

    panels/ operators/ adapters/   ->  may import `lichtfeld`
    core/                          ->  must NOT import `lichtfeld`

`core/` is pure Python + numpy so the partitioner is unit-testable without the
GUI and so nothing that runs server-side depends on the host application.

Note that this module itself imports the host *lazily*, inside `on_load()`.
Because the plugin directory is the package root, `import large_scene_trainer.core`
executes this file first — a module-scope `import lichtfeld` here would make
`core/` unimportable outside the application and quietly break the rule above.
"""

__version__ = "0.1.0"

# Populated by on_load(), drained by on_unload(). Module-level so hot reload,
# which re-executes this file, always starts from a known-empty registry.
_registered: list = []


def on_load() -> None:
    import lichtfeld as lf

    from .operators.partition_scene import PartitionScene
    from .panels.main_panel import MainPanel

    api = getattr(lf, "PLUGIN_API_VERSION", "unknown")
    features = getattr(getattr(lf, "plugins", None), "FEATURES", ())
    lf.log.info(f"large_scene_trainer {__version__}: plugin_api={api} features={list(features)}")

    # Operators before panels: a panel that polls an operator never sees it missing.
    for cls in (PartitionScene, MainPanel):
        lf.register_class(cls)
        _registered.append(cls)

    lf.log.info("large_scene_trainer loaded")


def on_unload() -> None:
    import lichtfeld as lf

    while _registered:
        lf.unregister_class(_registered.pop())

    lf.log.info("large_scene_trainer unloaded")
