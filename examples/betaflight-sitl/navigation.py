"""Offline attitude filter and gate navigator.

Consumes detector candidates, IMU, barometer, magnetometer, time, the on-pad
flag, and ``next_gate_index``. It never sees pixels, a simulation context, or
world gate poses. Live flight and database writes belong to later packages.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from guidance import TrackState
from vision import GateCandidate

_G = 9.81
_GRAVITY_L = np.array([0.0, 0.0, -_G], dtype=np.float64)
_NORTH_L = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_STEP_S = 0.002
_STATE = 12
_I12 = np.eye(_STATE, dtype=np.float64)


class TrackCode(IntEnum):
    """Section 7.8 track codes. Codes 1–4 match ``guidance.TrackState``."""

    EMPTY = 0
    TENTATIVE = int(TrackState.TENTATIVE)
    TRACKING = int(TrackState.TRACKING)
    COASTING = int(TrackState.COASTING)
    LOST = int(TrackState.LOST)


@dataclass(frozen=True, slots=True)
class NavParams:
    """Starting Section 7.10 tuning. Tests may replace these; they are not the contract."""

    accel_gain: float = 1.0 / 30.0
    mag_gain: float = 1.0 / 5.0
    accel_gate_mps2: float = 0.25
    rate_gate_rad_s: float = 0.3
    init_stationary_s: float = 0.25
    step_s: float = _STEP_S
    accel_psd: float = 0.05**2
    bias_psd: float = 0.01**2
    # Bearing matches ~0.7 px on the FPV camera. Depth keeps the range-squared
    # shape from Section 7.10; the coefficient is fit to synthetic PnP residuals
    # (about 7 cm along the line of sight at 10 m, against a 6 m starting value).
    bearing_sigma_rad: float = math.radians(0.08)
    vision_noise_scale: float = 1.0
    depth_coefficient: float = 0.012
    depth_floor_m: float = 0.01
    baro_sigma_m: float = 0.1
    zero_velocity_sigma_mps: float = 0.02
    association_d2: float = 16.3
    confirm_hits: int = 3
    confirm_window_s: float = 0.5
    reset_streak: int = 10
    tracking_fresh_s: float = 0.25
    coast_limit_s: float = 2.5
    buffer_s: float = 0.5
    close_range_m: float = 2.0
    max_range_m: float = 40.0
    telemetry_hz: float = 50.0
    normal_vertical_max: float = 0.34


@dataclass(frozen=True, slots=True)
class GateEstimate:
    """Target gate in the local frame, plus track bookkeeping."""

    index: int
    center: np.ndarray
    yaw: float
    track: TrackCode
    last_update_time: float | None
    relative_sigma_m: float
    yaw_sigma_rad: float
    last_nis: float
    rejection_streak: int
    passed: bool


@dataclass(frozen=True, slots=True)
class NavEstimate:
    """Navigator output. ``ready`` is false until the pad interval initializes attitude."""

    time: float
    ready: bool
    position: np.ndarray
    velocity: np.ndarray
    q_lb: np.ndarray
    yaw_rate: float
    accel_bias: np.ndarray
    position_sigma_m: float
    gate: GateEstimate | None
    dropped_vision_samples: int

    def relative_body(self) -> np.ndarray:
        """Body-FLU ``[px, py, pz, vx, vy, vz]`` of Section 7.8, or zeros when invalid."""

        if self.gate is None or self.gate.track is TrackCode.EMPTY:
            return np.zeros(6, dtype=np.float64)
        rotation_bl = _rotation_body_from_local(self.q_lb)
        relative = rotation_bl @ (self.gate.center - self.position)
        relative_velocity = -(rotation_bl @ self.velocity)
        return np.concatenate((relative, relative_velocity))

    def telemetry(self) -> dict[str, np.ndarray]:
        """Arrays J will write. Invalid channels use the Section 7.8 sentinels."""

        quat = self.q_lb if self.ready else np.array([0.0, 0.0, 0.0, 1.0])
        nav_est = np.concatenate((self.position, self.velocity, quat))
        if not self.ready:
            nav_est[:] = 0.0
        if self.gate is None or self.gate.track is TrackCode.EMPTY:
            nav_gate = np.zeros(6, dtype=np.float64)
            gate_rel = np.zeros(6, dtype=np.float64)
            gate_det = np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float64)
            nis = 0.0
            streak = 0.0
            sigma = 0.0
        else:
            age = (
                0.0
                if self.gate.last_update_time is None
                else self.time - self.gate.last_update_time
            )
            nav_gate = np.array(
                [
                    *self.gate.center,
                    self.gate.yaw,
                    float(self.gate.track),
                    age,
                ],
                dtype=np.float64,
            )
            gate_rel = self.relative_body()
            bearing = gate_rel[:3]
            norm = float(np.linalg.norm(bearing))
            if norm < 1e-9:
                azimuth = elevation = 0.0
                range_m = -1.0
            else:
                azimuth = math.atan2(bearing[1], bearing[0])
                elevation = math.atan2(bearing[2], math.hypot(bearing[0], bearing[1]))
                range_m = norm
            gate_det = np.array([azimuth, elevation, range_m, 1.0], dtype=np.float64)
            nis = self.gate.last_nis
            streak = float(self.gate.rejection_streak)
            sigma = self.gate.relative_sigma_m
        nav_diag = np.array(
            [float(self.ready), nis, streak, sigma, 0.0],
            dtype=np.float64,
        )
        return {
            "drone.nav_est": nav_est,
            "drone.nav_gate": nav_gate,
            "drone.nav_diag": nav_diag,
            "drone.gate_rel_est": gate_rel if self.gate is not None else np.zeros(6),
            "drone.gate_det": gate_det,
        }


class GateNavigator:
    """Attitude filter plus the 12-state gate navigator."""

    def __init__(self, params: NavParams | None = None) -> None:
        self.params = NavParams() if params is None else params
        self._q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._x = np.zeros(_STATE, dtype=np.float64)
        self._P = np.eye(_STATE, dtype=np.float64)
        self._P[0:3, 0:3] *= 1.0
        self._P[3:6, 3:6] *= 1.0
        self._P[6:9, 6:9] *= 0.05
        self._P[9:12, 9:12] *= 25.0
        self._yaw = 0.0
        self._yaw_var = math.radians(20.0) ** 2
        self._ready = False
        self._init_force: list[np.ndarray] = []
        self._init_mag: list[np.ndarray] = []
        self._init_baro: list[float] = []
        self._init_started: float | None = None
        self._baro_pad = 0.0
        self._history: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self._track = TrackCode.EMPTY
        self._gate_index = 0
        self._last_update: float | None = None
        self._last_nis = 0.0
        self._rejection_streak = 0
        self._hit_times: list[float] = []
        self._dropped = 0
        self._passed = False
        self._last_telemetry: float | None = None
        self._yaw_rate = 0.0
        self._last_accel_l = np.zeros(3, dtype=np.float64)

    def estimate(self, sim_time: float) -> NavEstimate:
        gate = None
        if self._ready and self._track is not TrackCode.EMPTY:
            relative = (
                self._P[9:12, 9:12] - self._P[9:12, 0:3] - self._P[0:3, 9:12] + self._P[0:3, 0:3]
            )
            age_sigma = math.sqrt(max(0.0, float(np.trace(relative)) / 3.0))
            gate = GateEstimate(
                index=self._gate_index,
                center=self._x[9:12].copy(),
                yaw=self._yaw,
                track=self._track,
                last_update_time=self._last_update,
                relative_sigma_m=age_sigma,
                yaw_sigma_rad=math.sqrt(max(0.0, self._yaw_var)),
                last_nis=self._last_nis,
                rejection_streak=self._rejection_streak,
                passed=self._passed,
            )
        return NavEstimate(
            time=sim_time,
            ready=self._ready,
            position=self._x[0:3].copy(),
            velocity=self._x[3:6].copy(),
            q_lb=self._q.copy(),
            yaw_rate=self._yaw_rate,
            accel_bias=self._x[6:9].copy(),
            position_sigma_m=math.sqrt(max(0.0, float(np.trace(self._P[0:3, 0:3])) / 3.0)),
            gate=gate,
            dropped_vision_samples=self._dropped,
        )

    def due_telemetry(self, sim_time: float) -> dict[str, np.ndarray] | None:
        """50 Hz batch. ``None`` until the next sample is due."""

        period = 1.0 / self.params.telemetry_hz
        if self._last_telemetry is not None and sim_time + 1e-12 < self._last_telemetry + period:
            return None
        self._last_telemetry = sim_time
        return self.estimate(sim_time).telemetry()

    def _accumulate_init(
        self,
        sim_time: float,
        omega: np.ndarray,
        force: np.ndarray,
        barometer: float | None,
        mag: np.ndarray | None,
    ) -> None:
        if float(np.linalg.norm(omega)) > self.params.rate_gate_rad_s:
            self._init_force.clear()
            self._init_mag.clear()
            self._init_baro.clear()
            self._init_started = None
            return
        if self._init_started is None:
            self._init_started = sim_time
        self._init_force.append(force.copy())
        if mag is not None:
            self._init_mag.append(mag.copy())
        if barometer is not None and math.isfinite(barometer):
            self._init_baro.append(float(barometer))
        if sim_time - self._init_started < self.params.init_stationary_s or not self._init_mag:
            return
        mean_force = np.mean(self._init_force, axis=0)
        mean_mag = np.mean(self._init_mag, axis=0)
        self._q = _attitude_from_specific_force_and_mag(mean_force, mean_mag)
        self._baro_pad = float(np.mean(self._init_baro)) if self._init_baro else 0.0
        self._ready = True
        self._remember(sim_time)

    def _propagate_attitude(
        self, omega: np.ndarray, force: np.ndarray, mag: np.ndarray | None
    ) -> None:
        corrected = omega.copy()
        force_norm = float(np.linalg.norm(force))
        rate = float(np.linalg.norm(omega))
        if (
            abs(force_norm - _G) <= self.params.accel_gate_mps2
            and rate <= self.params.rate_gate_rad_s
            and force_norm > 1e-6
        ):
            predicted_up = _rotation_body_from_local(self._q) @ np.array([0.0, 0.0, 1.0])
            measured_up = force / force_norm
            corrected = corrected - self.params.accel_gain * np.cross(predicted_up, measured_up)
        if mag is not None and float(np.linalg.norm(mag)) > 1e-9:
            mag_l = _rotation_local_from_body(self._q) @ mag
            horizontal = np.array([mag_l[0], mag_l[1], 0.0])
            if float(np.linalg.norm(horizontal)) > 1e-9:
                yaw_error = math.atan2(-horizontal[0], horizontal[1])
                # The error is about local up. Express it in the body frame so a
                # banked vehicle is not pitched or rolled by the heading correction.
                yaw_rate_l = np.array([0.0, 0.0, -self.params.mag_gain * yaw_error])
                corrected = corrected + _rotation_body_from_local(self._q) @ yaw_rate_l
        self._q = _integrate_quaternion(self._q, corrected, self.params.step_s)

    def _predict(self) -> None:
        dt = self.params.step_s
        acceleration = self._last_accel_l
        self._x[0:3] += self._x[3:6] * dt + 0.5 * acceleration * dt * dt
        self._x[3:6] += acceleration * dt
        state_transition = _I12.copy()
        state_transition[0:3, 3:6] = dt * np.eye(3)
        state_transition[0:3, 6:9] = -0.5 * dt * dt * np.eye(3)
        state_transition[3:6, 6:9] = -dt * np.eye(3)
        q = self.params.accel_psd
        process = np.zeros((_STATE, _STATE), dtype=np.float64)
        process[0:3, 0:3] = (q * dt**3 / 3.0) * np.eye(3)
        process[0:3, 3:6] = (q * dt**2 / 2.0) * np.eye(3)
        process[3:6, 0:3] = process[0:3, 3:6]
        process[3:6, 3:6] = (q * dt) * np.eye(3)
        process[6:9, 6:9] = (self.params.bias_psd * dt) * np.eye(3)
        self._P = state_transition @ self._P @ state_transition.T + process
        self._P = 0.5 * (self._P + self._P.T)
        yaw_process = math.radians(0.05) ** 2 * dt
        self._yaw_var += yaw_process

    def _remember(self, sim_time: float) -> None:
        rotation = _rotation_local_from_body(self._q)
        self._history.append((sim_time, self._x[0:3].copy(), rotation.copy()))
        horizon = sim_time - self.params.buffer_s
        while self._history and self._history[0][0] < horizon:
            self._history.popleft()

    def _set_acceleration(self, specific_force: np.ndarray) -> None:
        rotation = _rotation_local_from_body(self._q)
        self._last_accel_l = rotation @ specific_force + _GRAVITY_L - self._x[6:9]

    def step(
        self,
        sim_time: float,
        gyro: np.ndarray,
        specific_force: np.ndarray,
        *,
        barometer: float | None = None,
        magnetometer: np.ndarray | None = None,
        candidates: Sequence[GateCandidate] = (),
        on_pad: bool = False,
        next_gate_index: int = 0,
    ) -> NavEstimate:
        """Advance one 2 ms step. ``specific_force`` is the accelerometer, m/s²."""

        if not math.isfinite(sim_time):
            raise ValueError("sim_time must be finite")
        omega = np.asarray(gyro, dtype=np.float64).reshape(3)
        force = np.asarray(specific_force, dtype=np.float64).reshape(3)
        mag = (
            None if magnetometer is None else np.asarray(magnetometer, dtype=np.float64).reshape(3)
        )
        self._yaw_rate = float(omega[2])
        if not self._ready:
            self._accumulate_init(sim_time, omega, force, barometer, mag)
            return self.estimate(sim_time)
        self._propagate_attitude(omega, force, mag)
        self._set_acceleration(force)
        self._predict()
        self._remember(sim_time)
        if barometer is not None and math.isfinite(barometer):
            self._update_barometer(float(barometer))
        if on_pad:
            self._update_zero_velocity()
        if int(next_gate_index) != self._gate_index and self._track is not TrackCode.EMPTY:
            self._clear_gate()
        self._gate_index = int(next_gate_index)
        self._ingest_candidates(sim_time, candidates)
        self._refresh_track(sim_time)
        return self.estimate(sim_time)

    def _update_barometer(self, altitude: float) -> None:
        residual = np.array([altitude - self._baro_pad - self._x[2]])
        measurement = np.zeros((1, _STATE))
        measurement[0, 2] = 1.0
        self._joseph(residual, measurement, np.array([[self.params.baro_sigma_m**2]]))

    def _update_zero_velocity(self) -> None:
        residual = -self._x[3:6]
        measurement = np.zeros((3, _STATE))
        measurement[0:3, 3:6] = np.eye(3)
        variance = self.params.zero_velocity_sigma_mps**2
        self._joseph(residual, measurement, variance * np.eye(3))

    def _ingest_candidates(self, sim_time: float, candidates: Sequence[GateCandidate]) -> None:
        if not candidates:
            return
        usable = []
        for candidate in candidates:
            located = self._locate(candidate)
            if located is None:
                continue
            usable.append(located)
        if not usable:
            if self._track is not TrackCode.EMPTY and self._predicted_fully_visible():
                self._rejection_streak += 1
                if self._rejection_streak >= self.params.reset_streak:
                    self._clear_gate()
            return
        if self._track is TrackCode.EMPTY:
            chosen = min(
                usable, key=lambda item: item[0].pose.range_m if item[0].pose else math.inf
            )
            self._initialize_gate(chosen, sim_time)
            return
        best: tuple[GateCandidate, np.ndarray, np.ndarray, np.ndarray, float] | None = None
        best_d2 = math.inf
        for candidate, _measured, covariance, _sample_position, relative in usable:
            residual, measurement = self._relative_measurement(candidate.sample_time, relative)
            innovation = measurement @ self._P @ measurement.T + covariance
            try:
                solved = np.linalg.solve(innovation, residual)
            except np.linalg.LinAlgError:
                continue
            distance = float(residual @ solved)
            if distance < best_d2:
                best_d2 = distance
                best = (candidate, residual, covariance, measurement, distance)
        if best is None or best_d2 > self.params.association_d2:
            if self._predicted_fully_visible():
                self._rejection_streak += 1
                if self._rejection_streak >= self.params.reset_streak:
                    self._clear_gate()
            return
        _candidate, residual, covariance, measurement, distance = best
        self._joseph(residual, measurement, covariance)
        self._last_nis = distance
        self._last_update = sim_time
        self._rejection_streak = 0
        self._hit_times.append(sim_time)
        self._hit_times = [
            t for t in self._hit_times if sim_time - t <= self.params.confirm_window_s
        ]
        self._update_yaw(best[0])
        self._passed = self._plane_behind()

    def _locate(
        self, candidate: GateCandidate
    ) -> tuple[GateCandidate, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        pose = candidate.pose
        if pose is None or not 1.0 <= float(pose.range_m) <= self.params.max_range_m:
            self._dropped += 1
            return None
        sample = self._pose_at(candidate.sample_time)
        if sample is None:
            self._dropped += 1
            return None
        _time, position, rotation = sample
        normal_l = rotation @ np.asarray(pose.normal_body, dtype=np.float64)
        if abs(float(normal_l[2])) > self.params.normal_vertical_max:
            return None
        relative = rotation @ np.asarray(pose.position_body, dtype=np.float64)
        measured = position + relative
        covariance = _anisotropic_covariance(relative, self.params)
        return candidate, measured, covariance, position, relative

    def _relative_measurement(
        self, sample_time: float, relative: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """PnP observes ``g - p`` at the image time.

        Position at exposure is the current state rewound by the delay. The
        residual uses that same rewind, so the position and velocity gains match
        the innovation.
        """

        delay = max(0.0, self._history[-1][0] - float(sample_time)) if self._history else 0.0
        measurement = np.zeros((3, _STATE))
        measurement[:, 0:3] = -np.eye(3)
        measurement[:, 3:6] = delay * np.eye(3)
        measurement[:, 9:12] = np.eye(3)
        return relative - measurement @ self._x, measurement

    def _pose_at(self, sample_time: float) -> tuple[float, np.ndarray, np.ndarray] | None:
        if not self._history or not math.isfinite(sample_time):
            return None
        newest = self._history[-1][0]
        oldest = self._history[0][0]
        if sample_time > newest + 1e-9 or sample_time < oldest - 1e-9:
            return None
        return min(self._history, key=lambda item: abs(item[0] - sample_time))

    def _initialize_gate(
        self,
        located: tuple[GateCandidate, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        sim_time: float,
    ) -> None:
        candidate, measured, covariance, _sample_position, _relative = located
        self._x[9:12] = measured
        self._P[9:12, :] = 0.0
        self._P[:, 9:12] = 0.0
        self._P[9:12, 9:12] = self._P[0:3, 0:3] + covariance
        self._P[9:12, 0:3] = self._P[0:3, 0:3]
        self._P[0:3, 9:12] = self._P[0:3, 0:3]
        self._track = TrackCode.TENTATIVE
        # Age from when the update is applied. The image time can be up to the
        # buffer length earlier, which would mark a new track as already coasting.
        self._last_update = sim_time
        self._last_nis = 0.0
        self._rejection_streak = 0
        self._hit_times = [sim_time]
        self._update_yaw(candidate)
        self._passed = False

    def _update_yaw(self, candidate: GateCandidate) -> None:
        pose = candidate.pose
        if pose is None:
            return
        sample = self._pose_at(candidate.sample_time)
        if sample is None:
            return
        normal_l = sample[2] @ np.asarray(pose.normal_body, dtype=np.float64)
        measured = math.atan2(float(normal_l[1]), float(normal_l[0]))
        innovation = _wrap(measured - self._yaw)
        range_m = max(float(pose.range_m), 0.5)
        sigma = math.radians(max(1.0, (range_m / 4.0) ** 2))
        variance = sigma**2
        kalman = self._yaw_var / (self._yaw_var + variance)
        self._yaw = _wrap(self._yaw + kalman * innovation)
        self._yaw_var = (1.0 - kalman) * self._yaw_var

    def _predicted_fully_visible(self) -> bool:
        relative = _rotation_body_from_local(self._q) @ (self._x[9:12] - self._x[0:3])
        distance = float(np.linalg.norm(relative))
        if distance < self.params.close_range_m or distance > self.params.max_range_m:
            return False
        azimuth = math.atan2(float(relative[1]), float(relative[0]))
        elevation = math.atan2(float(relative[2]), math.hypot(relative[0], relative[1]))
        return abs(azimuth) < math.radians(40.0) and abs(elevation) < math.radians(25.0)

    def _plane_behind(self) -> bool:
        normal = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        return float(np.dot(self._x[0:3] - self._x[9:12], normal)) > 0.0

    def _refresh_track(self, sim_time: float) -> None:
        if self._track is TrackCode.EMPTY or self._last_update is None:
            return
        age = sim_time - self._last_update
        hits = len(self._hit_times)
        self._passed = self._plane_behind()
        if age > self.params.coast_limit_s:
            self._track = TrackCode.LOST
        elif age > self.params.tracking_fresh_s:
            self._track = TrackCode.COASTING
        elif hits >= self.params.confirm_hits:
            self._track = TrackCode.TRACKING
        else:
            self._track = TrackCode.TENTATIVE

    def _clear_gate(self) -> None:
        self._x[9:12] = 0.0
        self._P[9:12, :] = 0.0
        self._P[:, 9:12] = 0.0
        self._P[9:12, 9:12] = 25.0 * np.eye(3)
        self._track = TrackCode.EMPTY
        self._last_update = None
        self._last_nis = 0.0
        self._rejection_streak = 0
        self._hit_times.clear()
        self._passed = False
        self._yaw = 0.0
        self._yaw_var = math.radians(20.0) ** 2

    def _joseph(
        self, residual: np.ndarray, measurement: np.ndarray, covariance: np.ndarray
    ) -> None:
        innovation = measurement @ self._P @ measurement.T + covariance
        try:
            gain = self._P @ measurement.T @ np.linalg.inv(innovation)
        except np.linalg.LinAlgError:
            return
        self._x = self._x + gain @ residual
        identity = _I12
        reduced = identity - gain @ measurement
        self._P = reduced @ self._P @ reduced.T + gain @ covariance @ gain.T
        self._P = 0.5 * (self._P + self._P.T)


def _anisotropic_covariance(relative_l: np.ndarray, params: NavParams) -> np.ndarray:
    distance = float(np.linalg.norm(relative_l))
    if distance < 1e-6:
        return np.eye(3)
    direction = relative_l / distance
    lateral = max(params.depth_floor_m, distance * params.bearing_sigma_rad) ** 2
    depth_sigma = min(
        8.0,
        max(params.depth_floor_m, params.depth_coefficient * (distance / 4.0) ** 2),
    )
    depth = depth_sigma**2
    outer = np.outer(direction, direction)
    return params.vision_noise_scale * (depth * outer + lateral * (np.eye(3) - outer))


def _attitude_from_specific_force_and_mag(force: np.ndarray, mag: np.ndarray) -> np.ndarray:
    up = force / max(float(np.linalg.norm(force)), 1e-9)
    north = mag - up * float(np.dot(up, mag))
    north_norm = float(np.linalg.norm(north))
    if north_norm < 1e-9:
        north = np.array([0.0, 1.0, 0.0])
    else:
        north = north / north_norm
    east = np.cross(north, up)
    east_norm = float(np.linalg.norm(east))
    if east_norm < 1e-9:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = east / east_norm
    north = np.cross(up, east)
    rotation = np.stack((east, north, up), axis=0)
    return _quaternion_from_rotation(rotation)


def _integrate_quaternion(quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    half = np.array([*(omega * (0.5 * dt)), 1.0], dtype=np.float64)
    return _normalize(_quat_mul(quat, half))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = (float(v) for v in left)
    x2, y2, z2, w2 = (float(v) for v in right)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quat / norm


def _rotation_local_from_body(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_body_from_local(quat: np.ndarray) -> np.ndarray:
    return _rotation_local_from_body(quat).T


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    return _normalize(quat.astype(np.float64))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
