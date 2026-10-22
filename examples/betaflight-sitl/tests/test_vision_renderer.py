"""Renderer pitch is independent of CameraMount.

The projection below copies ``sensor_camera_pose``: ``Rx(roll) · Ry(+pitch) ·
Rz(yaw)``, forward = R · +X, up = R · +Z. It does not call ``CameraMount``.
"""

import math

import numpy as np
import pytest
from fpv_camera import MOUNT, sensor_rot_offset
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    CameraMount,
    candidate_from_ordered_corners,
    gate_object_corners,
)


def _renderer_rotation(rot_offset: list[float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(angle) for angle in rot_offset)

    def rx(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def ry(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def rz(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    return rx(roll) @ ry(pitch) @ rz(yaw)


def _project(points_body: np.ndarray, rot_offset: list[float]) -> np.ndarray:
    rotation = _renderer_rotation(rot_offset)
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    up = rotation @ np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    body_from_camera = np.column_stack((right, -up, forward))
    mount = np.asarray(MOUNT, dtype=np.float64)
    pixels = []
    for point in points_body:
        camera = body_from_camera.T @ (point - mount)
        assert camera[2] > 1e-8
        pixels.append(
            (
                FPV_INTRINSICS.fx * camera[0] / camera[2] + FPV_INTRINSICS.cx,
                FPV_INTRINSICS.fy * camera[1] / camera[2] + FPV_INTRINSICS.cy,
            )
        )
    return np.asarray(pixels, dtype=np.float64)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@pytest.mark.parametrize("tilt_deg", [-20.0, -10.0, 10.0, 20.0])
def test_registered_pitch_matches_the_nose_up_mount(tilt_deg: float) -> None:
    mount = CameraMount(position_body=np.asarray(MOUNT, dtype=np.float64), tilt_deg=tilt_deg)
    rotation = _renderer_rotation(sensor_rot_offset(tilt_deg))
    optical = mount.R_body_cam @ np.array([0.0, 0.0, 1.0])
    up = mount.R_body_cam @ np.array([0.0, -1.0, 0.0])

    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx(optical)
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(up)

    center = np.array([10.0, 0.0, 0.02])
    corners = _project(center + gate_object_corners(2.5), sensor_rot_offset(tilt_deg))
    candidate = candidate_from_ordered_corners(0.0, corners, 1.0, 2.5, FPV_INTRINSICS, mount)
    assert candidate.pose is not None
    assert _angle_deg(candidate.pose.bearing_body, center) < 0.25

    wrong = _project(center + gate_object_corners(2.5), [0.0, tilt_deg, 0.0])
    mismatched = candidate_from_ordered_corners(0.0, wrong, 1.0, 2.5, FPV_INTRINSICS, mount)
    assert mismatched.pose is not None
    assert _angle_deg(mismatched.pose.bearing_body, center) == pytest.approx(
        abs(2.0 * tilt_deg), abs=1.0
    )


def test_nose_up_tilt_moves_a_level_point_below_center() -> None:
    mount = np.asarray(MOUNT, dtype=np.float64)
    ahead = mount + np.array([5.0, 0.0, 0.0])
    pixels = _project(ahead.reshape(1, 3), sensor_rot_offset(10.0))

    assert pixels[0, 1] == pytest.approx(180.0 + 320.0 * math.tan(math.radians(10.0)))
    assert pixels[0, 1] - FPV_INTRINSICS.cy == pytest.approx(56.42, abs=0.01)
    assert sensor_rot_offset(10.0) == [0.0, -10.0, 0.0]
    assert FPV_MOUNT.tilt_deg == 0.0
