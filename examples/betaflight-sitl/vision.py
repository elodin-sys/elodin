"""Camera calibration, corner order, and planar PnP for the FPV camera.

This module is the production vision contract. It does not know world gate
poses. ``inner_size`` is the public opening, supplied by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fpv_camera import FOV_DEG, HEIGHT, MOUNT, TILT_DEG, WIDTH

_MAX_REPROJ_PX = 5.0


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics. Pixel y increases downward."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_vertical_fov(cls, width: int, height: int, vfov_deg: float) -> CameraIntrinsics:
        """Square pixels. ``fy = (height / 2) / tan(vfov / 2)`` and ``fx = fy``."""

        if width <= 0 or height <= 0:
            raise ValueError("image size must be positive")
        if not math.isfinite(vfov_deg) or vfov_deg <= 0.0 or vfov_deg >= 180.0:
            raise ValueError("vertical FoV must be in (0, 180) degrees")
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.fy))


FPV_INTRINSICS = CameraIntrinsics.from_vertical_fov(WIDTH, HEIGHT, FOV_DEG)


def _ry(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


# Columns are the OpenCV camera axes expressed in body FLU:
# image-right → body −Y, image-down → body −Z, optical axis → body +X.
R_FLU_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class CameraMount:
    """The one camera-to-body transform. Positive tilt pitches the optical axis up."""

    position_body: np.ndarray
    tilt_deg: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_body, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("camera position must be three finite body coordinates")
        if not math.isfinite(self.tilt_deg):
            raise ValueError("camera tilt must be finite")
        object.__setattr__(self, "position_body", position)

    @property
    def R_body_cam(self) -> np.ndarray:
        # Ry(−tilt) turns body +X toward +Z when tilt is positive.
        return _ry(math.radians(-self.tilt_deg)) @ R_FLU_FROM_OPENCV

    def camera_to_body(self, vector: np.ndarray, *, point: bool) -> np.ndarray:
        """Map a camera vector into body FLU. Points include the mount translation."""

        mapped = self.R_body_cam @ np.asarray(vector, dtype=np.float64)
        if point:
            mapped = mapped + self.position_body
        return mapped

    def body_to_camera(self, vector: np.ndarray, *, point: bool) -> np.ndarray:
        """Inverse of ``camera_to_body``."""

        value = np.asarray(vector, dtype=np.float64)
        if point:
            value = value - self.position_body
        return self.R_body_cam.T @ value


FPV_MOUNT = CameraMount(position_body=np.asarray(MOUNT, dtype=np.float64), tilt_deg=TILT_DEG)


@dataclass(frozen=True, slots=True)
class GatePose:
    """Gate center and opening frame recovered from image corners."""

    rotation_cam_gate: np.ndarray
    translation_cam: np.ndarray
    position_body: np.ndarray
    normal_body: np.ndarray
    bearing_body: np.ndarray
    range_m: float


@dataclass(frozen=True, slots=True)
class GateCandidate:
    """One Section 7.7 detection. Corners are TL, TR, BR, BL in pixels."""

    sample_time: float
    corners_px: np.ndarray
    confidence: float
    pose: GatePose | None
    reprojection_error_px: float | None

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_px, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            raise ValueError("corners must be four finite pixel positions")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.reprojection_error_px is not None and (
            not math.isfinite(self.reprojection_error_px) or self.reprojection_error_px < 0.0
        ):
            raise ValueError("reprojection error must be finite and non-negative")
        object.__setattr__(self, "corners_px", corners)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL. Valid for image roll within ±45°."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        raise ValueError("corner ordering needs four finite pixel positions")
    center = pts.mean(axis=0)
    delta = pts - center
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    order = np.argsort(angles)
    ordered = pts[order]
    ordered_angles = angles[order]
    target = -3.0 * math.pi / 4.0
    delta_angle = np.angle(np.exp(1j * (ordered_angles - target)))
    shift = int(np.argmin(np.abs(delta_angle)))
    return np.roll(ordered, -shift, axis=0)


def gate_object_corners(inner_size: float) -> np.ndarray:
    """Inner-opening corners in the gate frame, seen from the approach side.

    Order is TL, TR, BR, BL. Gate +Y is left and gate +Z is up, so the corners
    sit on the local YZ plane at local X = 0.
    """

    if not math.isfinite(inner_size) or inner_size <= 0.0:
        raise ValueError("gate inner size must be positive and finite")
    half = inner_size * 0.5
    return np.array(
        [
            [0.0, half, half],
            [0.0, -half, half],
            [0.0, -half, -half],
            [0.0, half, -half],
        ],
        dtype=np.float64,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rodrigues(rotation: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotation))
    if theta < 1e-12:
        return np.eye(3) + _skew(rotation)
    axis = rotation / theta
    k = _skew(axis)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return axis * (theta / (2.0 * math.sin(theta)))


def _project(points_cam: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    z = points_cam[:, 2:3]
    return np.column_stack(
        (
            intrinsics.fx * points_cam[:, 0] / z[:, 0] + intrinsics.cx,
            intrinsics.fy * points_cam[:, 1] / z[:, 0] + intrinsics.cy,
        )
    )


def _homography(plane_yz: np.ndarray, normalized_xy: np.ndarray) -> np.ndarray:
    rows = []
    for (y, z), (xn, yn) in zip(plane_yz, normalized_xy, strict=True):
        rows.append([y, z, 1.0, 0.0, 0.0, 0.0, -xn * y, -xn * z, -xn])
        rows.append([0.0, 0.0, 0.0, y, z, 1.0, -yn * y, -yn * z, -yn])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    return vh[-1].reshape(3, 3)


def _pose_from_homography(
    homography: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]
    scale = 0.5 * (float(np.linalg.norm(h1)) + float(np.linalg.norm(h2)))
    if scale < 1e-12:
        return None
    rotation_y = h1 / scale
    rotation_z = h2 / scale
    translation = h3 / scale
    if translation[2] < 0.0:
        rotation_y = -rotation_y
        rotation_z = -rotation_z
        translation = -translation
    rotation_x = np.cross(rotation_y, rotation_z)
    rotation = np.column_stack((rotation_x, rotation_y, rotation_z))
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    return rotation, translation


def _residuals(
    rotation_vector: np.ndarray,
    translation: np.ndarray,
    object_points: np.ndarray,
    observed: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    camera_points = (_rodrigues(rotation_vector) @ object_points.T).T + translation
    if np.any(camera_points[:, 2] <= 1e-8):
        return None
    projected = _project(camera_points, intrinsics)
    return (projected - observed).reshape(-1)


def _refine(
    rotation: np.ndarray,
    translation: np.ndarray,
    object_points: np.ndarray,
    observed: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    rotation_vector = _rotation_vector(rotation)
    translation = translation.copy()
    residual = _residuals(rotation_vector, translation, object_points, observed, intrinsics)
    if residual is None:
        return None
    for _ in range(10):
        if float(residual @ residual) < 1e-18:
            break
        jacobian = np.zeros((residual.size, 6))
        step = 1e-6
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = step
            perturbed = _residuals(
                rotation_vector + delta[:3],
                translation + delta[3:],
                object_points,
                observed,
                intrinsics,
            )
            if perturbed is None:
                return None
            jacobian[:, index] = (perturbed - residual) / step
        update, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        rotation_vector = rotation_vector - update[:3]
        translation = translation - update[3:]
        residual = _residuals(rotation_vector, translation, object_points, observed, intrinsics)
        if residual is None:
            return None
    rotation = _rodrigues(rotation_vector)
    rms = float(math.sqrt(float(residual @ residual) / observed.shape[0]))
    return rotation, translation, rms


def solve_square_pnp(
    corners_px: np.ndarray,
    inner_size: float,
    intrinsics: CameraIntrinsics,
    *,
    max_reproj_px: float = _MAX_REPROJ_PX,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Planar PnP for the public square opening.

    Returns ``(R_cam_gate, t_cam, reprojection_rms)`` or ``None`` when the gate
    is behind the camera, the opening faces the camera, or the fit is poor.
    Corners must already be ordered TL, TR, BR, BL.
    """

    observed = np.asarray(corners_px, dtype=np.float64)
    if observed.shape != (4, 2) or not np.all(np.isfinite(observed)):
        return None
    object_points = gate_object_corners(inner_size)
    normalized = np.column_stack(
        (
            (observed[:, 0] - intrinsics.cx) / intrinsics.fx,
            (observed[:, 1] - intrinsics.cy) / intrinsics.fy,
        )
    )
    recovered = _pose_from_homography(_homography(object_points[:, 1:], normalized))
    if recovered is None:
        return None
    refined = _refine(recovered[0], recovered[1], object_points, observed, intrinsics)
    if refined is None:
        return None
    rotation, translation, rms = refined
    if translation[2] <= 0.0 or float(rotation[:, 0] @ translation) <= 0.0:
        return None
    if rms > max_reproj_px:
        return None
    return rotation, translation, rms


def _body_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    mount: CameraMount,
) -> GatePose:
    position_body = mount.camera_to_body(translation, point=True)
    normal_body = mount.camera_to_body(rotation[:, 0], point=False)
    range_m = float(np.linalg.norm(position_body))
    bearing_body = position_body / range_m if range_m > 0.0 else position_body
    return GatePose(
        rotation_cam_gate=rotation,
        translation_cam=translation,
        position_body=position_body,
        normal_body=normal_body,
        bearing_body=bearing_body,
        range_m=range_m,
    )


def candidate_from_corners(
    sample_time: float,
    corners_px: np.ndarray,
    confidence: float,
    inner_size: float,
    intrinsics: CameraIntrinsics,
    mount: CameraMount,
) -> GateCandidate:
    """Order corners, solve planar PnP, and express the pose in body FLU."""

    ordered = order_corners(corners_px)
    solved = solve_square_pnp(ordered, inner_size, intrinsics)
    if solved is None:
        return GateCandidate(sample_time, ordered, confidence, None, None)
    rotation, translation, rms = solved
    return GateCandidate(
        sample_time,
        ordered,
        confidence,
        _body_pose(rotation, translation, mount),
        rms,
    )
