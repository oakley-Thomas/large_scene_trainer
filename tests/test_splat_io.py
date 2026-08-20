"""Host-adapter unit tests using a minimal LichtFeld fake."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _adapter(monkeypatch):
    class FakeTensor:
        def __init__(self, value):
            self.value = value
            self.on_cuda = False

        def cuda(self):
            self.on_cuda = True
            return self

    class Tensor:
        @staticmethod
        def from_numpy(value):
            return FakeTensor(value)

    fake = types.SimpleNamespace(Tensor=Tensor)
    monkeypatch.setitem(sys.modules, "lichtfeld", fake)
    sys.modules.pop("large_scene_trainer.adapters.splat_io", None)
    module = importlib.import_module("large_scene_trainer.adapters.splat_io")
    return module, FakeTensor


def test_world_positions_respects_column_major_node_transform(monkeypatch):
    adapter, _ = _adapter(monkeypatch)
    transform = np.eye(4)
    transform[:3, 3] = [10, -2, 4]

    positions = adapter.world_positions([[1, 2, 3]], transform.reshape(-1, order="F"))

    assert np.allclose(positions, [[11, 0, 7]])


def test_soft_delete_uses_cuda_boolean_mask(monkeypatch):
    adapter, fake_tensor = _adapter(monkeypatch)

    class Splat:
        def __init__(self):
            self.mask = None

        def soft_delete(self, mask):
            self.mask = mask

    class Node:
        def __init__(self):
            self.splat = Splat()

        def splat_data(self):
            return self.splat

    node = Node()
    adapter._soft_delete(node, np.array([True, False]))

    assert isinstance(node.splat.mask, fake_tensor)
    assert node.splat.mask.on_cuda
    assert node.splat.mask.value.dtype == np.bool_


def test_load_cropped_blocks_uses_data_loader_and_adds_separate_splat_nodes(monkeypatch):
    adapter, _ = _adapter(monkeypatch)

    class Splat:
        means_raw = "means"
        sh0_raw = "sh0"
        shN_raw = "shN"
        scaling_raw = "scaling"
        rotation_raw = "rotation"
        opacity_raw = "opacity"
        active_sh_degree = 3
        scene_scale = 2.0

    class Scene:
        def __init__(self):
            self.added = []
            self.changed = False

        def add_group(self, name):
            assert name == "LST Cropped Blocks"
            return 17

        def add_splat(self, name, *values, **kwargs):
            self.added.append((name, values, kwargs))
            return len(self.added)

        def get_node_by_id(self, node_id):
            return f"node-{node_id}"

        def notify_changed(self):
            self.changed = True

    scene = Scene()
    adapter.lf.get_scene = lambda: scene
    adapter.lf.io = types.SimpleNamespace(
        load=lambda path: types.SimpleNamespace(splat_data=Splat())
    )

    loaded = adapter.load_cropped_blocks(["/run/block_000.ply", "/run/block_001.ply"])

    assert loaded == ("node-1", "node-2")
    assert [item[0] for item in scene.added] == ["block_000", "block_001"]
    assert all(item[2] == {"sh_degree": 3, "scene_scale": 2.0, "parent": 17} for item in scene.added)
    assert scene.changed
