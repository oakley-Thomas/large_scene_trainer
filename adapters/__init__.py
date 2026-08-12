"""Translation between LichtFeld host objects and `core` value types.

This is the only place that speaks both vocabularies. Keep it dumb: convert,
do not compute. Any geometry or policy that appears here belongs in `core/`,
where it can be tested without the application running.

Phase 1 adds `scene_read.py` (cameras/bounds -> core.types); Phase 3 adds
`splat_io.py` (add_splat, DLPack tensors, crop -> export).
"""
