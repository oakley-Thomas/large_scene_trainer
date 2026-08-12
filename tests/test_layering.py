"""Enforce the one architectural rule this repo has.

`core/` must never import `lichtfeld` or `lfs_plugins`. If it does, the
partitioner stops being testable outside the GUI and anything shipped to the
training server drags the desktop application with it.

This is checked statically with `ast` rather than by importing the modules,
so it fails on a bad import line even when the host happens to be installed.
"""

import ast
from pathlib import Path

import pytest

HOST_MODULES = {"lichtfeld", "lfs_plugins"}

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
CORE_FILES = sorted(CORE_DIR.rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which can never reach the host.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_directory_is_not_empty():
    """Guards against this whole suite passing vacuously after a bad refactor."""
    assert CORE_FILES, f"no python files found under {CORE_DIR}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_core_module_does_not_import_host(path: Path):
    offenders = imported_roots(path) & HOST_MODULES
    assert not offenders, (
        f"{path.relative_to(CORE_DIR.parent)} imports {sorted(offenders)}; "
        f"move that logic into adapters/ and pass plain values into core/"
    )
