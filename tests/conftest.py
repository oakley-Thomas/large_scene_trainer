"""Make the plugin importable as a package when running tests directly.

The plugin's own directory *is* the package (LichtFeld loads
`~/.lichtfeld/plugins/large_scene_trainer/`), so `large_scene_trainer` is only
importable when its *parent* directory is on `sys.path`. Inside the application
the host arranges that; under bare pytest we have to do it ourselves.

Consequence worth knowing: the checkout directory must be named exactly
`large_scene_trainer`. Cloning it as `large-scene-trainer` breaks both the tests
and the plugin loader, because a hyphen is not a legal module name.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT = REPO_ROOT.parent

if REPO_ROOT.name != "large_scene_trainer":
    raise RuntimeError(
        f"checkout directory is {REPO_ROOT.name!r}; it must be named "
        f"'large_scene_trainer' for the package to import"
    )

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
