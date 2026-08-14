"""COLMAP binary reader tests using a small writer local to this test module."""

from __future__ import annotations

import struct

import numpy as np

from large_scene_trainer.core.colmap_io import (
    CAMERA_PARAM_COUNTS,
    ColmapCamera,
    ColmapImage,
    quaternion_to_matrix,
    load_cameras,
    read_cameras_bin,
    read_images_bin,
    read_points3D_bin,
    write_cameras_bin,
    write_images_bin,
    write_points3D_bin,
)
from large_scene_trainer.core.types import Point3D


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


def test_points3d_roundtrip_is_byte_identical(tmp_path):
    points = {
        19: Point3D(19, [1.0, 2.0, 3.0], [1, 2, 3], 0.25, ((7, 4), (3, 2))),
        4: Point3D(4, [-1.0, 0.0, 5.0], [250, 0, 13], 1.75, ()),
    }
    first = tmp_path / "points-first.bin"
    second = tmp_path / "points-second.bin"
    write_points3D_bin(points, first)
    recovered = read_points3D_bin(first)
    write_points3D_bin(recovered, second)
    assert second.read_bytes() == first.read_bytes()
    assert recovered[19].track == ((7, 4), (3, 2))


def test_images_roundtrip_preserves_names_pose_ids_and_points2d(tmp_path):
    images = {
        9: ColmapImage(
            9,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([-1.0, 2.0, -3.0]),
            41,
            "é/last.png",
            ((4.5, 9.25, 102),),
        ),
        2: ColmapImage(2, np.array([4.0, 3.0, 2.0, 1.0]), np.zeros(3), 12, "first.jpg"),
    }
    first = tmp_path / "images-first.bin"
    second = tmp_path / "images-second.bin"
    write_images_bin(images, first)
    recovered = read_images_bin(first)
    write_images_bin(recovered, second)
    assert second.read_bytes() == first.read_bytes()
    assert recovered[9].qvec.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert recovered[9].points2D == ((4.5, 9.25, 102),)


def test_cameras_roundtrip_preserves_raw_params_for_all_models(tmp_path):
    cameras = {
        model_id + 50: ColmapCamera(
            model_id + 50,
            model_id,
            640,
            480,
            np.arange(CAMERA_PARAM_COUNTS[model_id], dtype=np.float64) + model_id,
        )
        for model_id in CAMERA_PARAM_COUNTS
    }
    first = tmp_path / "cameras-first.bin"
    second = tmp_path / "cameras-second.bin"
    write_cameras_bin(cameras, first)
    recovered = read_cameras_bin(first)
    write_cameras_bin(recovered, second)
    assert second.read_bytes() == first.read_bytes()
