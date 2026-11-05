"""Classical detector acceptance on synthetic orange rings."""

import math

import numpy as np
import pytest
from controls import GuidanceUpdate
from course import Gate
from gate_detector import DetectorConfig, GateDetector, detect_fresh
from gate_images import (
    CORE_BEARINGS_DEG,
    CORE_RANGES_M,
    camera_pose_for_range,
    render_gates,
)
from synthetic_vision import _world_rotation
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    _project,
    _rodrigues,
    _rotation_vector,
    gate_object_corners,
    solve_square_pnp,
)

_INNER = 2.5
_GATE = Gate(0, (10.0, 0.0, 1.8), 0.0, _INNER)


def _bearing(gate: Gate, pose: np.ndarray) -> np.ndarray:
    body = _world_rotation(pose[:4]).T @ (np.asarray(gate.center, dtype=np.float64) - pose[4:])
    return body / np.linalg.norm(body)


def _angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(
        np.clip(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)), -1.0, 1.0)
    )
    return math.degrees(math.acos(cosine))


def _corner_error(detected: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(detected - truth, axis=1).max())


def _one(gate: Gate, pose: np.ndarray, inner_size: float = _INNER):
    image, corners = render_gates([gate], pose)
    assert len(corners) == 1
    found = GateDetector().detect(image, 1.25, inner_size)
    return image, corners[0], found


@pytest.mark.parametrize(
    ("yaw", "bearing"),
    [(0.0, 0.0), (0.0, 8.0), (math.radians(30.0), 0.0), (math.radians(-30.0), 0.0)],
)
def test_clean_corners_stay_within_half_pixel(yaw: float, bearing: float) -> None:
    gate = Gate(0, (10.0, 0.0, 1.8), yaw, _INNER)
    _image, truth, found = _one(gate, camera_pose_for_range(gate, 12.0, bearing_deg=bearing))

    assert len(found) == 1
    assert found[0].pose is not None
    assert _corner_error(found[0].corners_px, truth) <= 0.5


@pytest.mark.parametrize("range_m", (4.0, 10.0, 20.0))
def test_ranged_corners_stay_within_half_pixel(range_m: float) -> None:
    _image, truth, found = _one(_GATE, camera_pose_for_range(_GATE, range_m))

    assert len(found) == 1
    assert _corner_error(found[0].corners_px, truth) <= 0.5


def test_nested_gates_return_the_larger_opening_first() -> None:
    near = Gate(0, (10.0, 0.0, 1.8), 0.0, _INNER)
    pose = camera_pose_for_range(near, 6.0)
    camera = pose[4:] + np.asarray(FPV_MOUNT.position_body)
    far = Gate(1, tuple(camera + np.array([12.0, 0.0, 0.0])), 0.0, _INNER)
    image, corners = render_gates([near, far], pose)
    found = GateDetector().detect(image, 0.5, _INNER)

    assert len(corners) == 2
    assert len(found) == 2
    near_area = _polygon_area(found[0].corners_px)
    far_area = _polygon_area(found[1].corners_px)
    assert near_area > far_area
    assert _corner_error(found[0].corners_px, corners[0]) <= 0.5
    assert _corner_error(found[1].corners_px, corners[1]) <= 0.5


def test_clipped_opening_is_rejected() -> None:
    pose = camera_pose_for_range(_GATE, 8.0, bearing_deg=42.0)
    image, corners = render_gates([_GATE], pose)

    assert len(corners) == 1
    assert np.any(corners[0][:, 0] < 0.0) or np.any(corners[0][:, 0] > 640.0)
    assert GateDetector().detect(image, 0.0, _INNER) == ()


def test_opening_touching_the_border_is_rejected() -> None:
    pose = None
    corners = None
    image = None
    for bearing in np.linspace(28.0, 40.0, 25):
        candidate = camera_pose_for_range(_GATE, 8.0, bearing_deg=float(bearing))
        rendered, projected = render_gates([_GATE], candidate)
        xs = projected[0][:, 0]
        ys = projected[0][:, 1]
        if xs.min() <= 2.0 or xs.max() >= 637.0 or ys.min() <= 2.0 or ys.max() >= 357.0:
            if xs.min() >= -5.0 and xs.max() <= 645.0 and ys.min() >= -5.0 and ys.max() <= 365.0:
                pose, image, corners = candidate, rendered, projected[0]
                break
    assert pose is not None and image is not None and corners is not None
    assert GateDetector().detect(image, 0.0, _INNER) == ()


def test_tiny_quad_and_out_of_range_gate_are_rejected() -> None:
    distant = camera_pose_for_range(_GATE, 80.0)
    image, _corners = render_gates([_GATE], distant)
    assert GateDetector().detect(image, 0.0, _INNER) == ()

    ranged = camera_pose_for_range(_GATE, 50.0)
    image, corners = render_gates([_GATE], ranged)
    assert _polygon_area(corners[0]) >= 200.0
    assert GateDetector().detect(image, 0.0, _INNER) == ()

    close = camera_pose_for_range(_GATE, 10.0)
    image, _corners = render_gates([_GATE], close)
    assert GateDetector().detect(image, 0.0, 0.15) == ()


def test_empty_and_non_orange_frames_return_nothing() -> None:
    detector = GateDetector()
    blank = np.zeros((360, 640, 4), dtype=np.uint8)
    blue = np.zeros((360, 640, 4), dtype=np.uint8)
    blue[..., 2] = 255
    blue[..., 3] = 255

    assert detector.detect(blank, 0.0, _INNER) == ()
    assert detector.detect(blue, 0.0, _INNER) == ()
    assert detector.detect(blank, math.nan, _INNER) == ()


def test_detect_fresh_ignores_held_frames() -> None:
    image, _truth, found = _one(_GATE, camera_pose_for_range(_GATE, 10.0))
    assert len(found) == 1
    update = GuidanceUpdate(
        sim_time=1.0,
        tick=1,
        gyro=np.zeros(3),
        accel=np.zeros(3),
        frame=image,
        frame_time=1.0,
        frame_fresh=False,
        gate_count=1,
        gate_inner_size=_INNER,
    )
    assert detect_fresh(GateDetector(), update) == ()
    fresh = GuidanceUpdate(
        sim_time=1.0,
        tick=1,
        gyro=np.zeros(3),
        accel=np.zeros(3),
        frame=image,
        frame_time=1.0,
        frame_fresh=True,
        gate_count=1,
        gate_inner_size=_INNER,
    )
    assert len(detect_fresh(GateDetector(), fresh)) == 1


def test_detect_does_not_modify_the_frame_and_is_deterministic() -> None:
    image, _truth, _found = _one(_GATE, camera_pose_for_range(_GATE, 10.0))
    original = image.copy()
    image.setflags(write=False)
    detector = GateDetector()
    first = detector.detect(image, 2.0, _INNER)
    second = detector.detect(image, 2.0, _INNER)

    assert np.array_equal(image, original)
    assert len(first) == 1 and len(second) == 1
    assert first[0].corners_px == pytest.approx(second[0].corners_px)
    assert 0.0 <= first[0].confidence <= 1.0


def test_core_course_detection_rate_and_bearing() -> None:
    detector = GateDetector()
    errors = []
    visible = 0
    detected = 0
    for range_m in CORE_RANGES_M:
        for bearing in CORE_BEARINGS_DEG:
            pose = camera_pose_for_range(_GATE, range_m, bearing_deg=bearing)
            image, corners = render_gates([_GATE], pose)
            if not corners or np.any(corners[0] < 0.0) or np.any(corners[0][:, 0] > 640.0):
                continue
            if np.any(corners[0][:, 1] > 360.0):
                continue
            visible += 1
            found = detector.detect(image, range_m, _INNER)
            if not found or found[0].pose is None:
                continue
            detected += 1
            errors.append(_angle_deg(found[0].pose.bearing_body, _bearing(_GATE, pose)))

    assert visible > 0
    assert detected / visible >= 0.90
    assert float(np.median(errors)) <= 2.0


def test_rotation_near_180_degrees_yields_no_pose() -> None:
    """Accepted Package G limit: the Rodrigues conversion is unsafe within 1e-6° of 180°."""

    angle = math.pi - math.radians(1e-6)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    recovered = _rodrigues(_rotation_vector(rotation))
    assert not np.allclose(recovered, rotation, atol=1e-2)

    objects = gate_object_corners(_INNER)
    translation = np.array([0.0, 0.0, 8.0])
    observed = _project((rotation @ objects.T).T + translation, FPV_INTRINSICS)
    assert solve_square_pnp(observed, _INNER, FPV_INTRINSICS) is None


def test_config_rejects_an_even_close_kernel() -> None:
    with pytest.raises(ValueError, match="close kernel"):
        DetectorConfig(close_kernel=2)


def _polygon_area(corners: np.ndarray) -> float:
    shifted = np.roll(corners, -1, axis=0)
    return 0.5 * abs(float(np.sum(corners[:, 0] * shifted[:, 1] - shifted[:, 0] * corners[:, 1])))
