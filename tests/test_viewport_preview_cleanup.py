"""Regression guard for a host crash when preview groups were deleted."""

import ast
from pathlib import Path


ADAPTER_PATH = Path(__file__).resolve().parents[1] / "adapters" / "viewport_preview.py"


def test_preview_never_mutates_or_removes_existing_scene_nodes():
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"), filename=str(ADAPTER_PATH))

    forbidden_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remove_node"
    ]
    visibility_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "visible"
            for target in node.targets
        )
    ]

    assert not forbidden_calls
    assert not visibility_writes
