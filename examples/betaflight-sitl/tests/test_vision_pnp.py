"""Projection and planar-PnP round trips for the public 2.5 m opening.

Noise-free fixtures have four exact correspondences in float64, so the only
error is solver conditioning. Bearing must stay inside the 0.25° contract and
translation inside 1e-6·range + 1e-6 m. Integer pixels are a separate case.
The first-order depth sensitivity is Z²·σ/(f·S) with σ = 0.5 px, f = 320 px
and S = 2.5 m. All four corners quantize together, so the test allows twice
that estimate. At 2 m the 2.5 m opening is taller than the 360 px image, so
that range is outside the fully-visible fixture set.
"""

import math

import numpy as np
import pytest
from course import Gate, course_from_name
from synthetic_vision import SyntheticNoise, project_gate_corners, synthetic_sequence
from vision import (
    FPV_INTRINSICS,
    FPV_MOUNT,
    CameraIntrinsics,
    CameraMount,
    solve_square_pnp,
)

_INNER = 2.5
_BEARING_TOL_DEG = 0.25


def _level_pose(position: tuple[float, float, float], yaw: float = 0.0) -> np.ndarray:
    half = yaw * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half), *position], dtype=np.float64)


def _centered_height(gate: Gate) -> float:
    return gate.center[2] - float(FPV_MOUNT.position_body[2])


def _gate_center_body(gate: Gate, pose: np.ndarray) -> np.ndarray:
    from synthetic_vision import _world_rotation

    rotation = _world_rotation(pose[:4])
    return rotation.T @ (np.asarray(gate.center, dtype=np.float64) - pose[4:])


def _assert_round_trip(
    gate: Gate, pose: np.ndarray, *, bearing_tol: float, translation_tol: float
) -> None:
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    solved = solve_square_pnp(corners, gate.inner_size, FPV_INTRINSICS)
    assert solved is not None
    rotation, translation, rms = solved
    position = FPV_MOUNT.camera_to_body(translation, point=True)
    truth = _gate_center_body(gate, pose)
    bearing = position / np.linalg.norm(position)
    truth_bearing = truth / np.linalg.norm(truth)
    cosine = float(np.clip(bearing @ truth_bearing, -1.0, 1.0))
    assert math.degrees(math.acos(cosine)) <= bearing_tol
    assert position == pytest.approx(truth, abs=translation_tol)
    assert rms < 1e-6


def test_centered_gate_matches_the_hand_derived_corners() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None

    distance = gate.center[0] - pose[4] - float(FPV_MOUNT.position_body[0])
    half = _INNER / 2.0
    offset = FPV_INTRINSICS.fx * half / distance
    expected = np.array(
        [
            [320.0 - offset, 180.0 - offset],
            [320.0 + offset, 180.0 - offset],
            [320.0 + offset, 180.0 + offset],
            [320.0 - offset, 180.0 + offset],
        ]
    )
    assert corners == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("y", "z_offset"),
    [(0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 0.5), (0.0, -0.5)],
)
def test_centered_and_translated_gates_round_trip(y: float, z_offset: float) -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, y, _centered_height(gate) + z_offset))
    truth = _gate_center_body(gate, pose)
    _assert_round_trip(
        gate, pose, bearing_tol=1e-6, translation_tol=1e-6 * np.linalg.norm(truth) + 1e-6
    )


@pytest.mark.parametrize("yaw_deg", [-20.0, 20.0])
def test_yawed_drone_round_trips(yaw_deg: float) -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)), math.radians(yaw_deg))
    truth = _gate_center_body(gate, pose)
    _assert_round_trip(
        gate,
        pose,
        bearing_tol=_BEARING_TOL_DEG,
        translation_tol=1e-6 * np.linalg.norm(truth) + 1e-6,
    )


@pytest.mark.parametrize("gate_yaw_deg", [-25.0, 25.0])
def test_yawed_gate_round_trips(gate_yaw_deg: float) -> None:
    gate = Gate(index=0, center=(10.0, 0.0, 1.8), yaw=math.radians(gate_yaw_deg), inner_size=_INNER)
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    truth = _gate_center_body(gate, pose)
    _assert_round_trip(
        gate,
        pose,
        bearing_tol=_BEARING_TOL_DEG,
        translation_tol=1e-6 * np.linalg.norm(truth) + 1e-6,
    )


@pytest.mark.parametrize("range_m", [3.0, 5.0, 10.0, 20.0, 40.0])
def test_ranged_gates_round_trip(range_m: float) -> None:
    gate = course_from_name("single").gates[0]
    x = gate.center[0] - range_m - float(FPV_MOUNT.position_body[0])
    pose = _level_pose((x, 0.0, _centered_height(gate)))
    _assert_round_trip(
        gate, pose, bearing_tol=_BEARING_TOL_DEG, translation_tol=1e-6 * range_m + 1e-6
    )


def test_two_metre_opening_does_not_fit_in_the_image() -> None:
    gate = course_from_name("single").gates[0]
    x = gate.center[0] - 2.0 - float(FPV_MOUNT.position_body[0])
    pose = _level_pose((x, 0.0, _centered_height(gate)))

    assert project_gate_corners(gate, pose) is None


def test_integer_pixels_stay_inside_the_monocular_depth_bound() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    quantized = np.rint(corners)
    solved = solve_square_pnp(quantized, gate.inner_size, FPV_INTRINSICS)
    assert solved is not None
    _, translation, _ = solved
    truth_z = float(FPV_MOUNT.body_to_camera(_gate_center_body(gate, pose), point=True)[2])
    depth_tol = 2.0 * truth_z**2 * 0.5 / (FPV_INTRINSICS.fx * _INNER)
    assert translation[2] == pytest.approx(truth_z, abs=depth_tol)


def test_swapped_corners_are_rejected() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    swapped = corners.copy()
    swapped[[0, 1]] = swapped[[1, 0]]

    assert solve_square_pnp(swapped, gate.inner_size, FPV_INTRINSICS) is None


def test_cyclic_corner_shift_does_not_recover_the_gate_frame() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    shifted = np.roll(corners, 1, axis=0)
    solved = solve_square_pnp(shifted, gate.inner_size, FPV_INTRINSICS)
    truth = solve_square_pnp(corners, gate.inner_size, FPV_INTRINSICS)
    assert solved is not None and truth is not None
    relative = truth[0].T @ solved[0]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    assert math.degrees(math.acos(cosine)) > 45.0


def test_wrong_intrinsics_fail_the_round_trip() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    wrong = CameraIntrinsics(
        fx=300.0,
        fy=FPV_INTRINSICS.fy,
        cx=FPV_INTRINSICS.cx,
        cy=FPV_INTRINSICS.cy,
        width=FPV_INTRINSICS.width,
        height=FPV_INTRINSICS.height,
    )
    solved = solve_square_pnp(corners, gate.inner_size, wrong)
    assert solved is not None
    position = FPV_MOUNT.camera_to_body(solved[1], point=True)
    truth = _gate_center_body(gate, pose)
    assert not np.allclose(position, truth, atol=1e-3)


def test_flipped_tilt_sign_fails_the_body_round_trip() -> None:
    gate = course_from_name("single").gates[0]
    mount = CameraMount(position_body=FPV_MOUNT.position_body, tilt_deg=10.0)
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose, mount=mount)
    assert corners is not None
    solved = solve_square_pnp(corners, gate.inner_size, FPV_INTRINSICS)
    assert solved is not None
    flipped = CameraMount(position_body=mount.position_body, tilt_deg=-mount.tilt_deg)
    position = flipped.camera_to_body(solved[1], point=True)
    truth = _gate_center_body(gate, pose)
    assert not np.allclose(position, truth, atol=1e-2)


def test_transposed_camera_rotation_fails_the_body_round_trip() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 1.0, _centered_height(gate)))
    corners = project_gate_corners(gate, pose)
    assert corners is not None
    solved = solve_square_pnp(corners, gate.inner_size, FPV_INTRINSICS)
    assert solved is not None
    transposed = FPV_MOUNT.R_body_cam.T @ solved[1] + FPV_MOUNT.position_body
    truth = _gate_center_body(gate, pose)
    assert not np.allclose(transposed, truth, atol=1e-2)
    assert FPV_MOUNT.camera_to_body(solved[1], point=True) == pytest.approx(truth, abs=1e-6)


def test_synthetic_noise_is_deterministic() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    poses = [pose, pose, pose, pose]
    times = [0.0, 0.1, 0.2, 0.3]
    quiet = SyntheticNoise()
    noisy = SyntheticNoise(pixel_sigma=0.25, seed=7)
    other = SyntheticNoise(pixel_sigma=0.25, seed=8)

    clean = synthetic_sequence(gate, poses, times, quiet)
    same = synthetic_sequence(gate, poses, times, noisy)
    repeated = synthetic_sequence(gate, poses, times, noisy)
    different = synthetic_sequence(gate, poses, times, other)

    assert all(sample is not None and sample.pose is not None for sample in clean)
    assert same[0].corners_px == pytest.approx(repeated[0].corners_px)
    assert not np.allclose(same[0].corners_px, different[0].corners_px)
    assert clean[0].corners_px == pytest.approx(project_gate_corners(gate, pose))


def test_dropout_extremes_and_rate() -> None:
    gate = course_from_name("single").gates[0]
    pose = _level_pose((0.0, 0.0, _centered_height(gate)))
    poses = [pose] * 400
    times = [index / 30.0 for index in range(400)]

    kept = synthetic_sequence(gate, poses, times, SyntheticNoise(dropout_probability=0.0, seed=1))
    dropped = synthetic_sequence(
        gate, poses, times, SyntheticNoise(dropout_probability=1.0, seed=1)
    )
    mixed = synthetic_sequence(gate, poses, times, SyntheticNoise(dropout_probability=0.3, seed=3))

    assert all(sample is not None for sample in kept)
    assert all(sample is None for sample in dropped)
    rate = sum(sample is None for sample in mixed) / len(mixed)
    assert rate == pytest.approx(0.3, abs=0.08)
