"""Minimal readers for COLMAP sparse-model binary files.

Only camera data needed for trajectory fitting is read.  This implementation is
written from COLMAP's documented binary layout rather than vendoring its helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from .types import CameraRef


CAMERA_PARAM_COUNTS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
}
_SINGLE_FOCAL_MODELS = frozenset({0, 2, 8})


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of COLMAP binary file")
    return data


def _read_c_string(handle) -> str:
    chunks = bytearray()
    while True:
        byte = _read_exact(handle, 1)
        if byte == b"\x00":
            return chunks.decode("utf-8")
        chunks.extend(byte)


def read_cameras_bin(path: str | Path) -> dict[int, ColmapCamera]:
    """Parse ``cameras.bin`` into a mapping keyed by COLMAP camera id."""
    cameras: dict[int, ColmapCamera] = {}
    with Path(path).open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_exact(handle, 24)
            )
            try:
                n_params = CAMERA_PARAM_COUNTS[model_id]
            except KeyError as exc:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}") from exc
            params = np.frombuffer(_read_exact(handle, 8 * n_params), dtype="<f8").astype(
                np.float64
            )
            cameras[camera_id] = ColmapCamera(camera_id, model_id, width, height, params)
    return cameras


def read_images_bin(path: str | Path) -> dict[int, ColmapImage]:
    """Parse ``images.bin`` into a mapping keyed by COLMAP image id."""
    images: dict[int, ColmapImage] = {}
    with Path(path).open("rb") as handle:
        (count,) = struct.unpack("<Q", _read_exact(handle, 8))
        for _ in range(count):
            image_id = struct.unpack("<i", _read_exact(handle, 4))[0]
            qvec = np.frombuffer(_read_exact(handle, 32), dtype="<f8").astype(np.float64)
            tvec = np.frombuffer(_read_exact(handle, 24), dtype="<f8").astype(np.float64)
            camera_id = struct.unpack("<i", _read_exact(handle, 4))[0]
            name = _read_c_string(handle)
            (n_points,) = struct.unpack("<Q", _read_exact(handle, 8))
            _read_exact(handle, n_points * 24)  # x, y, point3D_id
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
    return images


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return COLMAP's world-to-camera matrix from a scalar-first quaternion."""
    q = np.asarray((qw, qx, qy, qz), dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("COLMAP image has a zero-length quaternion")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_cameras(sparse_dir: str | Path) -> list[CameraRef]:
    """Load joined COLMAP cameras, sorted ascending by image name."""
    sparse_dir = Path(sparse_dir)
    cameras = read_cameras_bin(sparse_dir / "cameras.bin")
    images = read_images_bin(sparse_dir / "images.bin")
    result: list[CameraRef] = []
    for image in images.values():
        try:
            camera = cameras[image.camera_id]
        except KeyError as exc:
            raise ValueError(
                f"image {image.image_id} references missing camera {image.camera_id}"
            ) from exc
        rotation = quaternion_to_matrix(*image.qvec)
        centre = -rotation.T @ image.tvec
        fx = float(camera.params[0])
        fy = fx if camera.model_id in _SINGLE_FOCAL_MODELS else float(camera.params[1])
        result.append(
            CameraRef(
                image_id=image.image_id,
                name=image.name,
                camera_id=camera.camera_id,
                C_world=centre,
                R_world_cam=rotation,
                fx=fx,
                fy=fy,
                width=camera.width,
                height=camera.height,
            )
        )
    return sorted(result, key=lambda camera: camera.name)
