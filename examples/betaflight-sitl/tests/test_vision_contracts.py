"""Camera calibration, mount, corner order, and candidate validation."""

import itertools
import math

import numpy as np
import pytest
from fpv_camera import FOCAL_PX, FOV_DEG, HEIGHT, WIDTH
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    R_FLU_FROM_OPENCV,
    CameraIntrinsics,
    CameraMount,
    GateCandidate,
    order_corners,
)


def test_fpv_intrinsics_match_the_focal_length_contract() -> None:
    assert FPV_INTRINSICS.fx == pytest.approx(FOCAL_PX, abs=1e-9)
    assert FPV_INTRINSICS.fy == pytest.approx(FOCAL_PX, abs=1e-9)
    assert FPV_INTRINSICS.cx == pytest.approx(WIDTH / 2.0)
    assert FPV_INTRINSICS.cy == pytest.approx(HEIGHT / 2.0)
    assert FPV_INTRINSICS.vfov_deg == pytest.approx(FOV_DEG, abs=1e-9)
    assert FPV_INTRINSICS.hfov_deg == pytest.approx(90.0, abs=1e-6)
    assert FPV_INTRINSICS.K[0, 0] == pytest.approx(FPV_INTRINSICS.fx)


def test_vertical_fov_constructor_uses_square_pixels() -> None:
    intrinsics = CameraIntrinsics.from_vertical_fov(640, 360, 60.0)

    expected_fy = 180.0 / math.tan(math.radians(30.0))
    assert intrinsics.fy == pytest.approx(expected_fy)
    assert intrinsics.fx == pytest.approx(intrinsics.fy)
    assert intrinsics.cx == 320.0
    assert intrinsics.cy == 180.0


def test_opencv_axes_map_onto_body_flu() -> None:
    optical = R_FLU_FROM_OPENCV @ np.array([0.0, 0.0, 1.0])
    image_right = R_FLU_FROM_OPENCV @ np.array([1.0, 0.0, 0.0])
    image_down = R_FLU_FROM_OPENCV @ np.array([0.0, 1.0, 0.0])

    assert optical == pytest.approx((1.0, 0.0, 0.0))
    assert image_right == pytest.approx((0.0, -1.0, 0.0))
    assert image_down == pytest.approx((0.0, 0.0, -1.0))


def test_mount_translates_points_and_round_trips() -> None:
    direction = np.array([0.0, 0.0, 5.0])
    point = np.array([0.2, -0.3, 4.0])

    assert FPV_MOUNT.camera_to_body(direction, point=False) == pytest.approx((5.0, 0.0, 0.0))
    assert FPV_MOUNT.camera_to_body(point, point=True)[0] == pytest.approx(
        FPV_MOUNT.camera_to_body(point, point=False)[0] + FPV_MOUNT.position_body[0]
    )
    assert FPV_MOUNT.body_to_camera(
        FPV_MOUNT.camera_to_body(point, point=True), point=True
    ) == pytest.approx(point)
    assert FPV_MOUNT.body_to_camera(
        FPV_MOUNT.camera_to_body(direction, point=False), point=False
    ) == pytest.approx(direction)


def test_positive_tilt_pitches_the_optical_axis_up() -> None:
    mount = CameraMount(position_body=FPV_MOUNT.position_body, tilt_deg=10.0)
    optical = mount.camera_to_body(np.array([0.0, 0.0, 1.0]), point=False)

    assert optical[2] > 0.0
    ahead = mount.position_body + np.array([8.0, 0.0, 0.0])
    camera = mount.body_to_camera(ahead, point=True)
    pixel_v = FPV_INTRINSICS.fy * camera[1] / camera[2] + FPV_INTRINSICS.cy
    assert pixel_v > FPV_INTRINSICS.cy


def test_order_corners_is_stable_for_permutations_and_limited_roll() -> None:
    corners = np.array([[100.0, 40.0], [400.0, 50.0], [390.0, 280.0], [110.0, 270.0]])
    ordered, ambiguous = order_corners(corners)
    assert not ambiguous

    for permutation in itertools.permutations(range(4)):
        recovered, _flag = order_corners(corners[list(permutation)])
        assert recovered == pytest.approx(ordered)

    center = ordered.mean(axis=0)
    rolled = []
    angle = math.radians(30.0)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for corner in ordered:
        rolled.append(center + rotation @ (corner - center))
    recovered, _flag = order_corners(np.asarray(rolled))
    unrolled = []
    for corner in recovered:
        unrolled.append(center + rotation.T @ (corner - center))
    assert np.asarray(unrolled) == pytest.approx(ordered, abs=1e-9)


# Reviewer fixture. The top edge is about −44°, inside the old ±45° claim.
_PERSPECTIVE_QUAD = np.array(
    [
        [467.5981, 139.7371],
        [495.3146, 112.7539],
        [533.4484, 147.0313],
        [500.1575, 170.8499],
    ]
)


def test_perspective_quad_keeps_gate_order_and_flags_ambiguity() -> None:
    ordered, ambiguous = order_corners(_PERSPECTIVE_QUAD)

    assert ordered == pytest.approx(_PERSPECTIVE_QUAD)
    assert ambiguous
    shuffled = _PERSPECTIVE_QUAD[[2, 0, 3, 1]]
    reshuffled, reshuffled_flag = order_corners(shuffled)
    assert reshuffled == pytest.approx(_PERSPECTIVE_QUAD)
    assert reshuffled_flag


def test_candidate_validation_rejects_bad_corners_and_confidence() -> None:
    corners = np.zeros((4, 2))
    GateCandidate(0.0, corners, 0.0, None, None, False)
    GateCandidate(0.0, corners, 1.0, None, 0.0, False)
    with pytest.raises(ValueError, match="corners"):
        GateCandidate(0.0, np.zeros((3, 2)), 1.0, None, None, False)
    with pytest.raises(ValueError, match="confidence"):
        GateCandidate(0.0, corners, 1.1, None, None, False)
    with pytest.raises(ValueError, match="reprojection"):
        GateCandidate(0.0, corners, 1.0, None, -0.1, False)
