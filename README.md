# Large Scene Trainer

A [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) plugin for training
large scenes that do not fit in a single GPU's memory. It splits a scene into
overlapping blocks along the capture trajectory, exports each block as an ordinary
LichtFeld dataset, generates the training jobs, and merges the trained blocks back
into one scene for LOD delivery.

**Status: skeleton.** The plugin loads and registers a panel and an operator. The
partitioner, job generator, and merge are not implemented yet.

## Why blocks

A city-block-scale street capture exceeds single-GPU VRAM as one monolithic
optimisation. Partitioning bounds the working set per run, and because each block is
a plain LichtFeld dataset, every block trains with the stock trainer — the plugin
never touches the rasteriser, the optimiser, or the Gaussian representation.

Each block carries two volumes:

- **core box** — tiles the scene without gaps; defines what survives the merge crop
- **context box** — the dilated region whose cameras and points feed training, so
  neighbouring blocks agree across their shared seams

## Install

Requires LichtFeld Studio ≥ 0.5.3 (the RAD/LOD export path).

```bash
# from inside the application
python -c "import lichtfeld as lf; lf.plugins.install('YOURNAME/large_scene_trainer')"
```

Or clone directly into the plugin directory:

```bash
git clone https://github.com/YOURNAME/large_scene_trainer \
    ~/.lichtfeld/plugins/large_scene_trainer
```

The directory name must be exactly `large_scene_trainer` — it *is* the Python package
name, and a hyphenated clone will not import.

## Layout

```
large_scene_trainer/
├── pyproject.toml     manifest: [project] + [tool.lichtfeld]
├── __init__.py        on_load / on_unload
├── panels/            UI                  ─┐
├── operators/         actions              ├─ may import lichtfeld
├── adapters/          host <-> core        ─┘
├── core/              geometry, manifests, jobs  ── must NOT import lichtfeld
├── runners/           local subprocess / remote job emission
└── tests/             plain pytest, no GUI, no CUDA
```

### The layering rule

Nothing under `core/` may import `lichtfeld` or `lfs_plugins`. `tests/test_layering.py`
enforces this statically. Two things depend on it:

1. The partitioner is unit-testable without launching the application.
2. Nothing that runs on the training server drags the desktop app along with it.

`adapters/` is the only module that speaks both vocabularies, and it converts rather
than computes — any geometry appearing there belongs in `core/`.

## Development

```bash
pytest                              # from inside the checkout
LichtFeld-Studio plugin check large_scene_trainer
```

`hot_reload` is on: saving any `.py` file reloads the plugin in place.

## Execution model

The plugin prepares a self-contained run bundle and hands it off; it does not connect
to the training server itself.

```
out/<run_id>/
├── jobs.json          the interface — all paths relative to the bundle root
├── run_blocks.sh      generated launcher, needs only bash and the binary
├── blocks/<id>/       manifest.json + filtered COLMAP model + images
└── results/<id>/      point_cloud.ply + status.json
```

Because every path is bundle-relative, the same bundle runs unchanged from a local
directory or from inside a container on a rented GPU. Status is derived from the
filesystem, so local and remote runs are collected by identical code.

> Block image directories are symlinked. Transfer with `rsync -aL` or `tar -ch`, or
> you will ship a bundle of dangling links.

## Licence

GPL-3.0-or-later, matching LichtFeld Studio. See [LICENSE](LICENSE).
