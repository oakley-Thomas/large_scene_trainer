"""COLMAP binary reader tests using a small writer local to this test module."""

from __future__ import annotations

import struct

import numpy as np

from large_scene_trainer.core.colmap_io import (
    quaternion_to_matrix,
    load_cameras,
    read_cameras_bin,
    read_images_bin,
)


def write_cameras(path, records) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(records)))
        for camera_id, model_id, width, height, params in records:
            handle.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            handle.write(np.asarray(params, dtype="<f8").tobytes())


def write_images(path, records) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(records)))
        for image_id, qvec, tvec, camera_id, name, points in records:
            handle.write(struct.pack("<i", image_id))
            handle.write(np.asarray(qvec, dtype="<f8").tobytes())
            handle.write(np.asarray(tvec, dtype="<f8").tobytes())
            handle.write(struct.pack("<i", camera_id))
            handle.write(name.encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(points)))
            for x, y, point_id in points:
                handle.write(struct.pack("<ddq", x, y, point_id))


def test_colmap_readers_recover_binary_values_and_sorted_cameras(tmp_path):
    write_cameras(
        tmp_path / "cameras.bin",
        [
            (7, 0, 1920, 1080, [900.0, 960.0, 540.0]),
            (8, 1, 800, 600, [700.0, 710.0, 400.0, 300.0]),
        ],
    )
    write_images(
        tmp_path / "images.bin",
        [
            (2, [2.0, 0.0, 0.0, 0.0], [-1.0, 2.0, -3.0], 7, "z/é.jpg", [(1.5, 2.5, 9)]),
            (1, [1.0, 0.0, 0.0, 0.0], [4.0, 5.0, 6.0], 8, "images/frame_0001.jpg", []),
        ],
    )

    cameras = read_cameras_bin(tmp_path / "cameras.bin")
    images = read_images_bin(tmp_path / "images.bin")
    assert cameras[7].width == 1920
    assert cameras[8].params.tolist() == [700.0, 710.0, 400.0, 300.0]
    assert images[2].name == "z/é.jpg"
    assert images[2].qvec.tolist() == [2.0, 0.0, 0.0, 0.0]
    assert images[2].tvec.tolist() == [-1.0, 2.0, -3.0]

    loaded = load_cameras(tmp_path)
    assert [camera.name for camera in loaded] == ["images/frame_0001.jpg", "z/é.jpg"]
    assert loaded[0].fx == 700.0 and loaded[0].fy == 710.0
    assert loaded[1].fx == 900.0 and loaded[1].fy == 900.0
    np.testing.assert_array_equal(loaded[1].C_world, [1.0, -2.0, 3.0])


def test_quaternion_conversion_normalises_before_constructing_rotation():
    rotation = quaternion_to_matrix(2.0, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(rotation, np.eye(3))
