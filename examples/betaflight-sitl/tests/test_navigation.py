"""Deterministic acceptance checks for the offline gate navigator."""

import math

import numpy as np
import pytest
from nav_trajectory import straight_run
from navigation import GateNavigator, NavParams, TrackCode
from vision import GateCandidate, GatePose

_DT = 0.002


def _run(traj, *, on_pad_s=0.5, gate_index=0, arrival_time=False, extra_candidates=None):
    navigator = GateNavigator()
    estimates = []
    for index, stamp in enumerate(traj.time):
        candidate = traj.candidates[index]
        batch = []
        if candidate is not None:
            if arrival_time:
                candidate = _retimed(candidate, float(stamp))
            batch.append(candidate)
        if extra_candidates is not None:
            replacement = extra_candidates(index, stamp, navigator)
            if replacement:
                batch = replacement
        estimates.append(
            navigator.step(
                float(stamp),
                traj.gyro[index],
                traj.specific_force[index],
                barometer=float(traj.barometer[index]),
                magnetometer=traj.magnetometer[index],
                candidates=batch,
                on_pad=float(stamp) <= on_pad_s,
                next_gate_index=gate_index if isinstance(gate_index, int) else gate_index(stamp),
            )
        )
    return navigator, estimates


def _retimed(candidate: GateCandidate, stamp: float) -> GateCandidate:
    return GateCandidate(
        stamp,
        candidate.corners_px,
        candidate.confidence,
        candidate.pose,
        candidate.reprojection_error_px,
        candidate.corner_order_ambiguous,
    )


def _ready(estimates):
    return [item for item in estimates if item.ready and item.gate is not None]


def test_converges_from_the_pad():
    traj = straight_run(duration_s=4.0, speed_mps=0.0, rest_s=4.0, seed=2)
    _navigator, estimates = _run(traj, on_pad_s=4.0)
    final = estimates[-1]
    assert final.ready
    assert final.gate is not None
    assert final.position_sigma_m < 1.0
    assert np.linalg.norm(final.velocity) < 0.3
    assert np.linalg.norm(final.position) < 0.5


def test_nis_and_nees_stay_inside_chi_square():
    traj = straight_run(duration_s=8.0, speed_mps=0.4, pixel_sigma=0.7, seed=3)
    _navigator, estimates = _run(traj, on_pad_s=0.5)
    innovations = []
    seen = None
    for item in _ready(estimates):
        updated = item.gate.last_update_time
        if item.gate.last_nis > 0.0 and updated != seen:
            innovations.append(item.gate.last_nis)
            seen = updated
    assert len(innovations) >= 30
    # Sum of 3-dof NIS values is chi-square with 3N degrees of freedom.
    total = float(np.sum(innovations))
    degrees = 3 * len(innovations)
    scale = math.sqrt(2.0 * degrees)
    assert abs(total - degrees) < 2.0 * scale
    pad = traj.position[int(0.3 / _DT)]
    truth_gate = np.asarray(traj.gate.center) - pad
    errors = []
    seen_error = None
    for estimate, index in zip(estimates, range(len(estimates)), strict=True):
        if estimate.gate is None or estimate.gate.track is TrackCode.EMPTY:
            continue
        if estimate.time < 2.0 or estimate.gate.last_update_time == seen_error:
            continue
        seen_error = estimate.gate.last_update_time
        truth_position = traj.position[index] - pad
        truth_relative = truth_gate - truth_position
        estimated_relative = estimate.gate.center - estimate.position
        error = estimated_relative - truth_relative
        sigma = max(estimate.gate.relative_sigma_m, 1e-6)
        errors.append(float(error @ error) / (sigma**2))
    assert len(errors) >= 30
    # ||e||^2 / sigma^2 for a 3-vector averages near 3. The interval is chi-square.
    nees_sum = float(np.sum(errors))
    nees_degrees = 3 * len(errors)
    assert abs(nees_sum - nees_degrees) < 2.0 * math.sqrt(2.0 * nees_degrees)


def test_depth_errors_do_not_swing_the_bearing():
    traj = straight_run(duration_s=5.0, speed_mps=0.0, rest_s=5.0, seed=4, depth_bias_m=0.0)
    biased = straight_run(duration_s=5.0, speed_mps=0.0, rest_s=5.0, seed=4, depth_bias_m=3.0)
    _nav, plain = _run(traj, on_pad_s=5.0)
    _nav_b, shifted = _run(biased, on_pad_s=5.0)
    plain_rel = plain[-1].relative_body()
    shifted_rel = shifted[-1].relative_body()
    plain_bearing = plain_rel[:3] / np.linalg.norm(plain_rel[:3])
    shifted_bearing = shifted_rel[:3] / np.linalg.norm(shifted_rel[:3])
    cosine = float(np.clip(np.dot(plain_bearing, shifted_bearing), -1.0, 1.0))
    assert math.degrees(math.acos(cosine)) < 2.0
    assert abs(np.linalg.norm(shifted_rel[:3]) - np.linalg.norm(plain_rel[:3])) < 6.0


def test_outlier_candidate_is_rejected():
    traj = straight_run(duration_s=4.0, speed_mps=0.0, rest_s=4.0, seed=5)
    pose = GatePose(
        np.eye(3),
        np.zeros(3),
        np.array([30.0, 20.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        36.0,
    )
    navigator = GateNavigator()
    before = None
    during = None
    for index, stamp in enumerate(traj.time):
        candidate = traj.candidates[index]
        batch = []
        if candidate is not None and not 2.9 <= stamp < 3.1:
            batch.append(candidate)
        elif candidate is not None:
            batch.append(GateCandidate(stamp, candidate.corners_px, 0.2, pose, 0.4, False))
        estimate = navigator.step(
            float(stamp),
            traj.gyro[index],
            traj.specific_force[index],
            barometer=float(traj.barometer[index]),
            magnetometer=traj.magnetometer[index],
            candidates=batch,
            on_pad=True,
        )
        if abs(stamp - 2.9) < _DT and estimate.gate is not None:
            before = estimate.gate.center.copy()
        if abs(stamp - 3.05) < _DT and estimate.gate is not None:
            during = estimate
    assert before is not None and during is not None and during.gate is not None
    assert np.linalg.norm(during.gate.center - before) < 0.75
    assert during.gate.rejection_streak >= 1
    assert navigator.estimate(3.2).gate is not None
    assert navigator.estimate(3.2).gate.track is not TrackCode.EMPTY


def test_stale_sample_is_dropped_and_counted():
    traj = straight_run(duration_s=3.0, speed_mps=0.0, rest_s=3.0, seed=6)
    navigator = GateNavigator()
    dropped_before = 0
    for index, stamp in enumerate(traj.time):
        candidate = traj.candidates[index]
        batch = []
        if candidate is not None and stamp > 1.0:
            batch.append(_retimed(candidate, stamp - 1.0))
        elif candidate is not None:
            batch.append(candidate)
        estimate = navigator.step(
            float(stamp),
            traj.gyro[index],
            traj.specific_force[index],
            barometer=float(traj.barometer[index]),
            magnetometer=traj.magnetometer[index],
            candidates=batch,
            on_pad=True,
        )
        if stamp < 1.0:
            dropped_before = estimate.dropped_vision_samples
    assert estimate.dropped_vision_samples > dropped_before


def test_coast_is_bounded_then_lost():
    traj = straight_run(duration_s=6.0, speed_mps=0.3, seed=7)
    navigator = GateNavigator()
    last_tracked = None
    lost = False
    drift = None
    for index, stamp in enumerate(traj.time):
        candidate = traj.candidates[index] if stamp < 2.5 else None
        estimate = navigator.step(
            float(stamp),
            traj.gyro[index],
            traj.specific_force[index],
            barometer=float(traj.barometer[index]),
            magnetometer=traj.magnetometer[index],
            candidates=[] if candidate is None else [candidate],
            on_pad=stamp <= 0.5,
        )
        if estimate.gate is not None and estimate.gate.track is TrackCode.TRACKING:
            last_tracked = estimate.position.copy()
        if last_tracked is not None and abs(stamp - 5.0) < _DT and estimate.gate is not None:
            drift = np.linalg.norm(estimate.position - last_tracked)
        if estimate.gate is not None and estimate.gate.track is TrackCode.LOST:
            lost = True
    assert drift is not None and drift < 5.0
    assert lost


def test_rejection_streak_resets_only_when_the_gate_is_in_view():
    traj = straight_run(
        duration_s=4.0, speed_mps=0.0, rest_s=4.0, seed=8, gate_center=(6.0, 0.0, 1.5)
    )
    bad = GatePose(
        np.eye(3),
        np.zeros(3),
        np.array([15.0, 8.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        17.0,
    )

    def spoof(index, stamp, _navigator):
        if stamp < 2.0 or traj.candidates[index] is None:
            return []
        return [GateCandidate(stamp, traj.candidates[index].corners_px, 0.4, bad, 0.2, False)]

    _navigator, estimates = _run(traj, on_pad_s=4.0, extra_candidates=spoof)
    assert any(
        item.time > 2.3 and (item.gate is None or item.gate.track is TrackCode.EMPTY)
        for item in estimates
    )

    close = straight_run(
        duration_s=7.0, speed_mps=1.2, rest_s=0.5, seed=9, gate_center=(6.0, 0.0, 1.5)
    )

    def spoof_close(index, stamp, _navigator):
        if stamp < 4.8:
            return []
        # Keep rejecting after the real gate has left the image.
        if index % 15 != 0:
            return []
        corners = np.zeros((4, 2))
        if close.candidates[index] is not None:
            corners = close.candidates[index].corners_px
        return [GateCandidate(float(stamp), corners, 0.4, bad, 0.2, False)]

    _nav, close_estimates = _run(close, on_pad_s=0.5, extra_candidates=spoof_close)
    approaching = [
        item for item in close_estimates if item.gate is not None and 4.5 < item.time < 4.8
    ]
    assert approaching
    assert float(np.linalg.norm(approaching[-1].relative_body()[:3])) < 2.0
    # Inside the close-range blind zone, rejected frames must not clear the track.
    assert [
        item
        for item in close_estimates
        if item.time > 5.0 and (item.gate is None or item.gate.track is TrackCode.EMPTY)
    ] == []


def test_image_time_updates_are_less_biased_than_arrival_time():
    truth = straight_run(duration_s=5.0, speed_mps=4.0, seed=10, pixel_sigma=0.0)
    delay_s = 0.2

    def drive(stamp_at_arrival: bool):
        navigator = GateNavigator()
        held: list[tuple[float, GateCandidate]] = []
        last = None
        for index, stamp in enumerate(truth.time):
            candidate = truth.candidates[index]
            if candidate is not None:
                held.append((float(stamp), candidate))
            due = []
            remaining = []
            for born, sample in held:
                if stamp + 1e-9 < born + delay_s:
                    remaining.append((born, sample))
                    continue
                due.append(sample if not stamp_at_arrival else _retimed(sample, float(stamp)))
            held = remaining
            last = navigator.step(
                float(stamp),
                truth.gyro[index],
                truth.specific_force[index],
                barometer=float(truth.barometer[index]),
                magnetometer=truth.magnetometer[index],
                candidates=due,
                on_pad=stamp <= 0.5,
            )
        return last

    pad = truth.position[int(0.4 / _DT)]
    truth_gate = np.asarray(truth.gate.center) - pad

    def along(estimate):
        assert estimate is not None and estimate.gate is not None
        return abs(float(estimate.gate.center[0] - truth_gate[0]))

    assert along(drive(True)) > along(drive(False)) + 0.2


def test_progress_change_switches_the_target():
    traj = straight_run(duration_s=4.0, speed_mps=0.0, rest_s=4.0, seed=11)

    def gate_index(stamp: float) -> int:
        return 0 if stamp < 2.5 else 1

    _navigator, estimates = _run(traj, on_pad_s=4.0, gate_index=gate_index)
    before = estimates[int(2.4 / _DT)].gate
    after = estimates[int(2.7 / _DT)].gate
    assert before is not None and before.index == 0
    assert after is not None and after.index == 1
    assert not np.allclose(before.center, after.center)


def test_passed_flag_follows_the_gate_plane():
    traj = straight_run(
        duration_s=8.0, speed_mps=2.0, seed=12, gate_center=(4.0, 0.0, 1.5), rest_s=0.5
    )
    _navigator, estimates = _run(traj, on_pad_s=0.5)
    crossed = [item for item in estimates if item.gate is not None and item.gate.passed]
    before = [item for item in estimates if item.time < 2.0 and item.gate is not None]
    assert before and all(not item.gate.passed for item in before)
    assert crossed


def test_attitude_errors_decay_and_accel_correction_gates_off_in_a_turn():
    navigator = GateNavigator(NavParams(init_stationary_s=0.2))
    mag = np.array([0.0, 1.0, 0.0])
    force = np.array([0.0, 0.0, 9.81])
    for step in range(200):
        navigator.step(step * _DT, np.zeros(3), force, barometer=0.1, magnetometer=mag, on_pad=True)
    assert navigator.estimate(0.4).ready
    # 20° nose-up, then hold a gravity reading so tilt correction is allowed.
    navigator._q = _pitch(math.radians(20.0))
    for step in range(20000):
        navigator.step(0.4 + step * _DT, np.zeros(3), force, barometer=0.1, on_pad=True)
    corrected = _up_error(navigator._q)
    navigator._q = _pitch(math.radians(20.0))
    thrusting = np.array([0.0, 0.0, 20.0])
    for step in range(20000):
        navigator.step(50.0 + step * _DT, np.zeros(3), thrusting, barometer=0.1, on_pad=False)
    gated = _up_error(navigator._q)
    assert corrected < math.radians(8.0)
    assert gated > corrected

    yawed = GateNavigator(NavParams(init_stationary_s=0.2))
    for step in range(150):
        yawed.step(step * _DT, np.zeros(3), force, barometer=0.1, magnetometer=mag, on_pad=True)
    yawed._q = _yaw(math.radians(30.0))
    for step in range(8000):
        yawed.step(
            0.4 + step * _DT, np.zeros(3), force, barometer=0.1, magnetometer=mag, on_pad=True
        )
    assert abs(_yaw_of(yawed._q)) < math.radians(10.0)


def test_no_fixture_produces_a_nan():
    traj = straight_run(duration_s=3.0, speed_mps=1.0, pixel_sigma=0.7, dropout=0.1, seed=13)
    _navigator, estimates = _run(traj)
    for estimate in estimates:
        assert np.all(np.isfinite(estimate.position))
        assert np.all(np.isfinite(estimate.velocity))
        assert np.all(np.isfinite(estimate.q_lb))
        if estimate.gate is not None:
            assert np.all(np.isfinite(estimate.gate.center))
            assert math.isfinite(estimate.gate.yaw_sigma_rad)


def test_telemetry_batch_is_50_hz_with_sentinels_before_a_gate():
    navigator = GateNavigator()
    samples = []
    force = np.array([0.0, 0.0, 9.81])
    mag = np.array([0.0, 1.0, 0.0])
    for step in range(500):
        stamp = step * _DT
        navigator.step(stamp, np.zeros(3), force, barometer=0.1, magnetometer=mag, on_pad=True)
        batch = navigator.due_telemetry(stamp)
        if batch is not None:
            samples.append(batch)
    assert len(samples) == pytest.approx(500 * _DT * 50.0, abs=2)
    assert samples[0]["drone.nav_est"].shape == (10,)
    assert samples[0]["drone.nav_gate"].shape == (6,)
    assert samples[0]["drone.nav_diag"].shape == (5,)
    assert samples[0]["drone.gate_det"][2] == -1.0


def _pitch(angle: float) -> np.ndarray:
    half = angle * 0.5
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)])


def _yaw(angle: float) -> np.ndarray:
    half = angle * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def _up_error(quat: np.ndarray) -> float:
    from navigation import _rotation_body_from_local

    up = _rotation_body_from_local(quat) @ np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(up[2], -1.0, 1.0))
    return math.acos(cosine)


def _yaw_of(quat: np.ndarray) -> float:
    from navigation import _rotation_local_from_body

    rotation = _rotation_local_from_body(quat)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
