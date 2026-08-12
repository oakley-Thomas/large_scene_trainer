# Large Scene Trainer

A [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) plugin for training
large scenes that do not fit in a single GPU's memory. It splits a scene into
overlapping blocks along the capture trajectory, exports each block as an ordinary
LichtFeld dataset, generates the training jobs, and merges the trained blocks back
into one scene for LOD delivery.


## Why blocks

A city-block-scale street capture exceeds single-GPU VRAM as one monolithic
optimisation. Partitioning bounds the working set per run, and because each block is
a plain LichtFeld dataset, every block trains with the stock trainer.

Each block carries two volumes:

- **core box** — tiles the scene without gaps; defines what survives the merge crop
- **context box** — the dilated region whose cameras and points feed training, so
  neighbouring blocks agree across their shared seams

## Install

Requires LichtFeld Studio ≥ 0.5.3 (the RAD/LOD export path).

Clone directly into the plugin directory:
```bash
git clone https://github.com/oakley-Thomas/large_scene_trainer \
    ~/.lichtfeld/plugins/large_scene_trainer
```


## Layout

```
large_scene_trainer/
├── pyproject.toml     manifest: [project] + [tool.lichtfeld]
├── __init__.py        on_load / on_unload
├── panels/            UI                  ─┐
├── operators/         actions              ├─ may import lichtfeld
├── adapters/          host <-> core       ─┘
├── core/              geometry, manifests, jobs 
├── runners/           local subprocess / remote job emission
└── tests/             plain pytest, no GUI, no CUDA
```

## Licence

GPL-3.0-or-later, matching LichtFeld Studio. See [LICENSE](LICENSE).
