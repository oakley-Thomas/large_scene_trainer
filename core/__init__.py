"""Host-independent core: geometry, partitioning, manifests, job construction.

HARD RULE: nothing under `core/` may import `lichtfeld` or `lfs_plugins`.

This is what lets the partitioner run under plain pytest (ticket 1.6) and what
keeps anything shipped to the training server free of the desktop application.
`tests/test_layering.py` enforces it.
"""
