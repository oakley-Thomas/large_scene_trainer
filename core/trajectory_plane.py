"""Deterministic trajectory-plane fitting and ground-frame construction."""

from __future__ import annotations

import math

import numpy as np

from .types import CameraRef, GroundFrame, PlaneDiagnostics


ORIENTATION_DISPERSION_OK_DEG = 3.0
ORIENTATION_DISPERSION_MAX_DEG = 10.0
PCA_DEGENERATE_RATIO = 10.0
AGREEMENT_OK_DEG = 5.0
_TINY = np.finfo(np.float64).tiny


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length == 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return vector / length


def _sorted_cameras(cams: list[CameraRef]) -> list[CameraRef]:
    return sorted(cams, key=lambda camera: camera.name)


def _acute_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def estimate_normal_from_orientations(cams: list[CameraRef]) -> tuple[np.ndarray, float]:
    """Return the orientation-consensus unit normal pointing up and its dispersion."""
    if not cams:
        raise ValueError("at least one camera is required")
    down = np.stack([_unit(camera.down_world) for camera in _sorted_cameras(cams)])
    _, _, vt = np.linalg.svd(down, full_matrices=False)
    dominant = vt[0]
    if float(np.dot(dominant, down.mean(axis=0))) < 0.0:
        dominant = -dominant
    angles = np.arccos(np.clip(np.abs(down @ dominant), -1.0, 1.0))
    dispersion = math.degrees(float(np.sqrt(np.mean(angles**2))))
    return _unit(-dominant), dispersion


def estimate_normal_pca(centres: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return PCA normal, planarity condition ratio, and ordered in-plane axes."""
    points = np.asarray(centres, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"centres must have shape (N, 3), got {points.shape}")
    if len(points) < 2:
        raise ValueError("at least two centres are required")
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / len(points)
    values, vectors = np.linalg.eigh(covariance)
    normal = _unit(vectors[:, 0])
    # A perfectly planar trajectory has lambda_0 == 0, for which the intended
    # condition ratio is infinite rather than a noisy runtime warning.
    with np.errstate(over="ignore", divide="ignore"):
        ratio = float(values[1] / max(values[0], _TINY))
    in_plane_axes = np.stack([_unit(vectors[:, 2]), _unit(vectors[:, 1])])
    return normal, ratio, in_plane_axes


def estimate_normal_nullspace(
    cams: list[CameraRef], normal_orientation: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Fit an up vector orthogonal to camera right and forward axes."""
    if not cams:
        raise ValueError("at least one camera is required")
    rows = []
    for camera in _sorted_cameras(cams):
        rows.extend((_unit(camera.right_world), _unit(camera.forward_world)))
    _, singular_values, vt = np.linalg.svd(np.stack(rows), full_matrices=False)
    normal = _unit(vt[-1])
    sign_reference = (
        normal_orientation
        if normal_orientation is not None
        else np.array([0.0, 0.0, 1.0])
    )
    if float(np.dot(normal, sign_reference)) < 0.0:
        normal = -normal
    ratio = float(singular_values[-1] / max(singular_values[-2], _TINY))
    return normal, ratio


def build_ground_frame(cams: list[CameraRef], normal: np.ndarray) -> GroundFrame:
    """Build a deterministic right-handed frame whose ``u`` axis follows capture."""
    ordered = _sorted_cameras(cams)
    if not ordered:
        raise ValueError("at least one camera is required")
    centres = np.stack([camera.C_world for camera in ordered])
    origin = centres.mean(axis=0)
    h = _unit(normal)
    offsets = centres - origin
    projected = offsets - np.outer(offsets @ h, h)
    _, _, vt = np.linalg.svd(projected, full_matrices=False)
    u = _unit(vt[0])
    # A capture can return near its start, making first-versus-last orientation
    # numerically meaningless.  This component rule is independent of capture
    # order and remains stable for closed loops.
    if u[np.argmax(np.abs(u))] < 0.0:
        u = -u
    v = _unit(np.cross(h, u))
    u = _unit(np.cross(v, h))
    frame_rotation = np.stack([u, v, h])
    if np.linalg.det(frame_rotation) <= 0.0:
        raise AssertionError("ground frame is not right-handed")
    if not np.allclose(frame_rotation @ frame_rotation.T, np.eye(3), atol=1e-12, rtol=0.0):
        raise AssertionError("ground frame is not orthonormal")
    return GroundFrame(R=frame_rotation, origin=origin)


def fit_trajectory_plane(
    cams: list[CameraRef], scene_units_per_metre: float | None = None
) -> tuple[GroundFrame, PlaneDiagnostics]:
    """Fit the trajectory frame and return it with observability diagnostics."""
    ordered = _sorted_cameras(cams)
    if len(ordered) < 3:
        raise ValueError("at least three cameras are required to fit a trajectory plane")
    if scene_units_per_metre is not None and scene_units_per_metre <= 0.0:
        raise ValueError("scene_units_per_metre must be positive")

    centres = np.stack([camera.C_world for camera in ordered])
    normal_orientation, dispersion = estimate_normal_from_orientations(ordered)
    normal_pca, pca_ratio, _ = estimate_normal_pca(centres)
    pca_degenerate = pca_ratio < PCA_DEGENERATE_RATIO
    agreement = None if pca_degenerate else _acute_angle_deg(normal_orientation, normal_pca)
    warnings: list[str] = []
    if pca_degenerate:
        warnings.append(
            f"PCA camera-centre fit is degenerate (condition ratio {pca_ratio:.6g}); "
            "its normal was discarded"
        )

    if dispersion <= ORIENTATION_DISPERSION_OK_DEG:
        estimator_used = "orientation"
        normal = normal_orientation
        if agreement is not None and agreement > AGREEMENT_OK_DEG:
            warnings.append(
                f"orientation and PCA normals disagree by {agreement:.3f} degrees; "
                "using orientation consensus"
            )
    elif dispersion <= ORIENTATION_DISPERSION_MAX_DEG:
        estimator_used = "orientation"
        normal = normal_orientation
        warnings.append(
            f"orientation dispersion is elevated at {dispersion:.3f} degrees; "
            "using orientation consensus"
        )
    else:
        estimator_used = "nullspace"
        normal, nullspace_ratio = estimate_normal_nullspace(ordered, normal_orientation)
        warnings.append(
            f"orientation dispersion is {dispersion:.3f} degrees; using nullspace "
            f"fallback (sigma_min/sigma_mid={nullspace_ratio:.6g})"
        )

    frame = build_ground_frame(ordered, normal)
    ground_centres = frame.to_ground(centres)
    height_residuals = ground_centres[:, 2] - ground_centres[:, 2].mean()
    spacings = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    arc_length = float(spacings.sum())
    median_spacing = float(np.median(spacings))
    if scene_units_per_metre is None:
        warnings.append(
            "scene scale is unspecified: trajectory arc length "
            f"{arc_length:.6g} and median inter-frame spacing {median_spacing:.6g} are "
            "in scene units; block extents must use those units"
        )
    if arc_length > 0.0 and float(np.sqrt(np.mean(height_residuals**2))) > 0.03 * arc_length:
        warnings.append(
            "a single global trajectory plane is a poor model of this scene; "
            "derive per-block z extents from local camera heights"
        )

    return frame, PlaneDiagnostics(
        estimator_used=estimator_used,
        normal_orientation=normal_orientation,
        orientation_dispersion_deg=float(dispersion),
        normal_pca=normal_pca,
        pca_condition_ratio=float(pca_ratio),
        pca_degenerate=pca_degenerate,
        agreement_deg=agreement,
        height_residual_rms=float(np.sqrt(np.mean(height_residuals**2))),
        height_residual_max=float(np.max(np.abs(height_residuals))),
        trajectory_arc_length=arc_length,
        median_inter_frame_spacing=median_spacing,
        extent_ground=tuple(float(value) for value in np.ptp(ground_centres, axis=0)),
        n_cameras=len(ordered),
        warnings=warnings,
    )
