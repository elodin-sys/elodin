"""Pure truth-guided standoff controller for the Betaflight SITL example.

The controller consumes only the immutable ``TruthGuidanceUpdate`` contract.
All positions/velocities are ENU, vehicle attitude is scalar-last, and attitude
sticks use the measured semantic pitch/roll signs from the Package D audit.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from types import MappingProxyType

from controls import GatePose, SemanticControl, TruthGuidanceUpdate, TruthState

Vec3 = tuple[float, float, float]


class GuidancePhase(enum.IntEnum):
    """Stable ``drone.guidance_mode`` numeric mapping (0 means disabled)."""

    DISABLED = 0
    ARM = 1
    TAKEOFF = 2
    HOVER = 3
    ACQUIRE = 4
    HOLD = 5
    FAULT = 6


GUIDANCE_PHASE_NAMES = MappingProxyType({phase.value: phase.name for phase in GuidancePhase})


@dataclass(frozen=True, slots=True)
class TruthGuidanceConfig:
    """Focused controller, mission timing, and RC-limit configuration."""

    simulation_duration_s: float = 45.0
    standoff_distance_m: float = 4.0
    boot_grace_s: float = 5.0
    arm_settle_s: float = 0.6
    arm_timeout_s: float = 12.0
    takeoff_timeout_s: float = 16.0
    hover_timeout_s: float = 6.0
    acquire_timeout_s: float = 22.0
    hold_timeout_s: float = 60.0
    takeoff_climb_rate_mps: float = 0.20
    takeoff_position_tolerance_m: float = 0.20
    takeoff_velocity_tolerance_mps: float = 0.20
    takeoff_dwell_s: float = 0.5
    hover_position_tolerance_m: float = 0.20
    hover_velocity_tolerance_mps: float = 0.18
    hover_dwell_s: float = 0.75
    acquire_position_tolerance_m: float = 0.25
    acquire_velocity_tolerance_mps: float = 0.20
    acquire_dwell_s: float = 1.0
    kp_xy: float = 0.7
    kv_xy: float = 1.5
    max_horizontal_speed_mps: float = 0.8
    max_horizontal_accel_mps2: float = 1.5
    max_tilt_rad: float = math.radians(9.5)
    stick_angle_rad_per_unit: float = math.radians(20.0)
    max_horizontal_stick: float = 0.48
    kp_altitude: float = 0.075
    kd_altitude: float = 0.12
    hover_rc_trim: float = 0.13
    throttle_min: float = 0.10
    throttle_max: float = 0.32
    yaw_kp: float = 0.5
    yaw_rate_damping: float = 0.10
    max_yaw_stick: float = 0.15
    gate_brake_accel_mps2: float = 0.8
    gate_brake_reaction_s: float = 0.5
    gate_brake_margin_m: float = 0.9
    gate_safety_slab_m: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            "simulation_duration_s",
            "standoff_distance_m",
            "arm_settle_s",
            "arm_timeout_s",
            "takeoff_timeout_s",
            "hover_timeout_s",
            "acquire_timeout_s",
            "hold_timeout_s",
            "takeoff_climb_rate_mps",
            "kp_xy",
            "kv_xy",
            "max_horizontal_speed_mps",
            "max_horizontal_accel_mps2",
            "max_tilt_rad",
            "stick_angle_rad_per_unit",
            "max_horizontal_stick",
            "kp_altitude",
            "kd_altitude",
            "hover_rc_trim",
            "throttle_max",
            "yaw_kp",
            "max_yaw_stick",
            "gate_brake_accel_mps2",
            "gate_brake_margin_m",
            "gate_safety_slab_m",
        )
        nonnegative = (
            "boot_grace_s",
            "takeoff_position_tolerance_m",
            "takeoff_velocity_tolerance_mps",
            "takeoff_dwell_s",
            "hover_position_tolerance_m",
            "hover_velocity_tolerance_mps",
            "hover_dwell_s",
            "acquire_position_tolerance_m",
            "acquire_velocity_tolerance_mps",
            "acquire_dwell_s",
            "gate_brake_reaction_s",
            "yaw_rate_damping",
            "throttle_min",
        )
        for name in positive + nonnegative:
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"{name} must be a finite number")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0 or (name in positive and value <= 0.0):
                polarity = "positive" if name in positive else "non-negative"
                raise ValueError(f"{name} must be finite and {polarity}")
            object.__setattr__(self, name, value)
        if self.throttle_min > self.hover_rc_trim or self.hover_rc_trim > self.throttle_max:
            raise ValueError("throttle limits must contain hover_rc_trim")
        if self.throttle_max > 1.0:
            raise ValueError("normalized throttle limits must not exceed one")
        if self.max_tilt_rad >= math.pi / 2.0:
            raise ValueError("max_tilt_rad must be less than 90 degrees")
        if self.max_horizontal_stick > 1.0 or self.max_yaw_stick > 1.0:
            raise ValueError("normalized stick limits must not exceed one")


@dataclass(frozen=True, slots=True)
class _ControlState:
    position: Vec3
    velocity: Vec3
    yaw: float
    yaw_rate: float


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add_scaled(a: Vec3, b: Vec3, scale: float) -> Vec3:
    return (a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2])


def _norm(vector: Vec3) -> float:
    return math.hypot(*vector)


def _clip_vector(vector: Vec3, maximum: float) -> Vec3:
    magnitude = _norm(vector)
    if magnitude <= maximum or magnitude <= 1.0e-15:
        return vector
    scale = maximum / magnitude
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` without a discontinuity near zero."""

    return math.remainder(angle, math.tau)


def _yaw_from_xyzw(quaternion: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TruthGuidance:
    """Deterministic ARM→TAKEOFF→HOVER→ACQUIRE→HOLD truth controller.

    The first gate pose is latched once and is never retargeted. Race progress
    can therefore signal an unexpected pass, but can never advance this E
    controller to a later gate or a commit phase.
    """

    def __init__(self, config: TruthGuidanceConfig | None = None):
        self.config = config or TruthGuidanceConfig()
        self.phase = GuidancePhase.ARM
        self.phase_started_at: float | None = None
        self.failure_reason: str | None = None
        self.braking_active = False
        self._gate: GatePose | None = None
        self._target: Vec3 | None = None
        self._launch_position: Vec3 | None = None
        self._target_altitude: float | None = None
        self._last_valid_state: TruthState | None = None
        self._last_time: float | None = None
        self._dwell_since: float | None = None
        self._arm_requested_at: float | None = None
        self._arm_requested = False
        self._previous_gate_local_x: float | None = None
        self._fault_target: Vec3 | None = None
        self._fault_altitude: float | None = None

    @property
    def phase_code(self) -> int:
        return int(self.phase)

    @property
    def target_enu(self) -> Vec3 | None:
        return self._target

    def update(self, update: TruthGuidanceUpdate) -> SemanticControl:
        """Advance from one immutable update and return bounded semantic RC intent."""

        if not math.isfinite(update.sim_time) or update.sim_time < 0.0:
            self._enter_fault("invalid_sim_time", self._last_time)
            return self._fault_control(update)
        if self._last_time is not None and update.sim_time < self._last_time:
            self._enter_fault("nonmonotonic_sim_time", self._last_time)
            return self._fault_control(update)
        self._last_time = update.sim_time

        truth = update.truth_state
        if truth is None or not self._valid_sensor_rate(update):
            self._enter_fault("invalid_truth", update.sim_time)
            return self._fault_control(update)
        self._last_valid_state = truth
        if self.phase_started_at is None:
            self.phase_started_at = update.sim_time
        if self._gate is None:
            if not update.gate_poses:
                self._enter_fault("missing_gate", update.sim_time)
                return self._fault_control(update)
            self._gate = update.gate_poses[0]
            self._target = self._standoff_target(self._gate, self.config.standoff_distance_m)
            self._launch_position = truth.position_enu
            self._target_altitude = self._gate.center_enu[2]

        local_x = self._gate.world_to_local(truth.position_enu)[0]
        crossed = self._crossed_forward_plane(self._previous_gate_local_x, local_x)
        self._previous_gate_local_x = local_x
        if crossed or update.last_gate_passed >= 0:
            self._enter_fault("gate_crossing", update.sim_time)
        if self.phase is GuidancePhase.FAULT:
            return self._fault_control(update)

        if self.phase is GuidancePhase.ARM:
            if update.sim_time < self.config.boot_grace_s:
                return SemanticControl.safe()
            self._arm_requested = True
            if self._arm_requested_at is None:
                self._arm_requested_at = update.sim_time
            # ARM sequencing is simulation-time deterministic. Actual exchanged
            # command/lockstep/motor evidence belongs only to the host grader.
            if update.sim_time - self._arm_requested_at + 1.0e-12 >= self.config.arm_settle_s:
                self._transition(GuidancePhase.TAKEOFF, update.sim_time)
            elif update.sim_time - (self.phase_started_at or 0.0) >= self.config.arm_timeout_s:
                self._arm_requested = False
                self._enter_fault("arm_timeout", update.sim_time)
                return self._fault_control(update)
            else:
                return SemanticControl(armed=True, angle_mode=True)

        state = self._control_state(truth, update.gyro)
        if self.phase is GuidancePhase.TAKEOFF:
            elapsed = max(0.0, update.sim_time - (self.phase_started_at or update.sim_time))
            altitude_setpoint = min(
                self._target_altitude,
                self._launch_position[2] + self.config.takeoff_climb_rate_mps * elapsed,
            )
            launch_xy_error = math.hypot(
                self._launch_position[0] - truth.position_enu[0],
                self._launch_position[1] - truth.position_enu[1],
            )
            if (
                abs(self._target_altitude - truth.position_enu[2])
                <= self.config.takeoff_position_tolerance_m
                and abs(truth.velocity_enu[2]) <= self.config.takeoff_velocity_tolerance_mps
                and launch_xy_error <= self.config.takeoff_position_tolerance_m
            ):
                if self._dwell(update.sim_time, self.config.takeoff_dwell_s):
                    self._transition(GuidancePhase.HOVER, update.sim_time)
            else:
                self._dwell_since = None
            if self.phase is GuidancePhase.TAKEOFF and elapsed >= self.config.takeoff_timeout_s:
                self._enter_fault("takeoff_timeout", update.sim_time)
            if self.phase is GuidancePhase.TAKEOFF:
                return self._position_control(
                    state,
                    self._launch_position,
                    altitude_setpoint,
                    self._gate.yaw_rad,
                )

        if self.phase is GuidancePhase.HOVER:
            launch_xy_error = math.hypot(
                self._launch_position[0] - truth.position_enu[0],
                self._launch_position[1] - truth.position_enu[1],
            )
            altitude_error = abs(self._target_altitude - truth.position_enu[2])
            if (
                launch_xy_error <= self.config.hover_position_tolerance_m
                and altitude_error <= self.config.hover_position_tolerance_m
                and _norm(truth.velocity_enu) <= self.config.hover_velocity_tolerance_mps
            ):
                if self._dwell(update.sim_time, self.config.hover_dwell_s):
                    self._transition(GuidancePhase.ACQUIRE, update.sim_time)
            else:
                self._dwell_since = None
            elapsed = update.sim_time - (self.phase_started_at or update.sim_time)
            if self.phase is GuidancePhase.HOVER and elapsed >= self.config.hover_timeout_s:
                self._enter_fault("hover_timeout", update.sim_time)
            if self.phase is GuidancePhase.HOVER:
                return self._position_control(
                    state,
                    self._launch_position,
                    self._target_altitude,
                    self._gate.yaw_rad,
                )

        if self.phase is GuidancePhase.ACQUIRE:
            target_error = _sub(self._target, truth.position_enu)
            if (
                _norm(target_error) <= self.config.acquire_position_tolerance_m
                and _norm(truth.velocity_enu) <= self.config.acquire_velocity_tolerance_mps
            ):
                if self._dwell(update.sim_time, self.config.acquire_dwell_s):
                    self._transition(GuidancePhase.HOLD, update.sim_time)
            else:
                self._dwell_since = None
            elapsed = update.sim_time - (self.phase_started_at or update.sim_time)
            if self.phase is GuidancePhase.ACQUIRE and elapsed >= self.config.acquire_timeout_s:
                self._enter_fault("acquire_timeout", update.sim_time)
            if self.phase is GuidancePhase.ACQUIRE:
                return self._position_control(
                    state,
                    self._target,
                    self._target_altitude,
                    self._gate.yaw_rad,
                )

        if self.phase is GuidancePhase.HOLD:
            elapsed = update.sim_time - (self.phase_started_at or update.sim_time)
            if elapsed >= self.config.hold_timeout_s:
                self._enter_fault("hold_timeout", update.sim_time)
            else:
                return self._position_control(
                    state,
                    self._target,
                    self._target_altitude,
                    self._gate.yaw_rad,
                )

        return self._fault_control(update)

    @staticmethod
    def _valid_sensor_rate(update: TruthGuidanceUpdate) -> bool:
        gyro = update.gyro
        return bool(
            getattr(gyro, "shape", None) == (3,)
            and all(math.isfinite(float(value)) for value in gyro)
        )

    @staticmethod
    def _standoff_target(gate: GatePose, distance: float = 4.0) -> Vec3:
        normal = gate.normal_enu
        return (
            gate.center_enu[0] - distance * normal[0],
            gate.center_enu[1] - distance * normal[1],
            gate.center_enu[2],
        )

    @staticmethod
    def _crossed_forward_plane(previous_x: float | None, current_x: float) -> bool:
        return previous_x is not None and previous_x < 0.0 <= current_x

    def _transition(self, phase: GuidancePhase, sim_time: float) -> None:
        self.phase = phase
        self.phase_started_at = sim_time
        self._dwell_since = None

    def _enter_fault(self, reason: str, sim_time: float | None) -> None:
        if self.phase is not GuidancePhase.FAULT:
            if self._last_valid_state is not None:
                self._fault_target = self._last_valid_state.position_enu
                self._fault_altitude = self._last_valid_state.position_enu[2]
            else:
                self._fault_target = self._launch_position
                self._fault_altitude = self._target_altitude
            self.phase = GuidancePhase.FAULT
            self.phase_started_at = sim_time
            self.failure_reason = reason
            self._dwell_since = None

    def _dwell(self, sim_time: float, duration: float) -> bool:
        if self._dwell_since is None:
            self._dwell_since = sim_time
        return sim_time - self._dwell_since + 1.0e-12 >= duration

    def _control_state(self, truth: TruthState, gyro) -> _ControlState:
        yaw_rate = float(gyro[2])
        if not math.isfinite(yaw_rate):
            yaw_rate = 0.0
        return _ControlState(
            position=truth.position_enu,
            velocity=truth.velocity_enu,
            yaw=_yaw_from_xyzw(truth.orientation_xyzw),
            yaw_rate=yaw_rate,
        )

    def _fault_control(self, update: TruthGuidanceUpdate) -> SemanticControl:
        truth = update.truth_state or self._last_valid_state
        if truth is None or not self._arm_requested:
            return SemanticControl.safe()
        if self._gate is None or self._target is None or self._launch_position is None:
            return SemanticControl(armed=True, angle_mode=True)
        gyro = update.gyro if self._valid_sensor_rate(update) else (0.0, 0.0, 0.0)
        state = self._control_state(truth, gyro)
        return self._position_control(
            state,
            self._fault_target if self._fault_target is not None else self._target,
            self._fault_altitude if self._fault_altitude is not None else self._target_altitude,
            self._gate.yaw_rad,
        )

    def _position_control(
        self,
        state: _ControlState,
        target: Vec3,
        altitude_setpoint: float,
        yaw_setpoint: float,
    ) -> SemanticControl:
        self.braking_active = False
        position_error = (target[0] - state.position[0], target[1] - state.position[1], 0.0)
        velocity_setpoint = _clip_vector(
            (self.config.kp_xy * position_error[0], self.config.kp_xy * position_error[1], 0.0),
            self.config.max_horizontal_speed_mps,
        )
        acceleration = (
            self.config.kv_xy * (velocity_setpoint[0] - state.velocity[0]),
            self.config.kv_xy * (velocity_setpoint[1] - state.velocity[1]),
            0.0,
        )

        gate = self._gate
        if gate is not None:
            local_x = gate.world_to_local(state.position)[0]
            forward_velocity = _dot(state.velocity, gate.normal_enu)
            distance_to_plane = -local_x
            stopping_distance = (
                max(0.0, forward_velocity) ** 2 / (2.0 * self.config.gate_brake_accel_mps2)
                + self.config.gate_brake_reaction_s * max(0.0, forward_velocity)
                + self.config.gate_brake_margin_m
            )
            if local_x >= -self.config.gate_safety_slab_m or (
                local_x < 0.0 and distance_to_plane <= stopping_distance
            ):
                forward_acceleration = _dot(acceleration, gate.normal_enu)
                if forward_velocity > 0.05:
                    forward_acceleration = min(
                        forward_acceleration,
                        -min(
                            self.config.gate_brake_accel_mps2,
                            max(0.2, self.config.kv_xy * forward_velocity),
                        ),
                    )
                else:
                    forward_acceleration = min(0.0, forward_acceleration)
                acceleration = _add_scaled(
                    acceleration,
                    gate.normal_enu,
                    forward_acceleration - _dot(acceleration, gate.normal_enu),
                )
                self.braking_active = True

        acceleration = _clip_vector(acceleration, self.config.max_horizontal_accel_mps2)
        heading_forward = (math.cos(state.yaw), math.sin(state.yaw), 0.0)
        heading_left = (-math.sin(state.yaw), math.cos(state.yaw), 0.0)
        accel_forward = _dot(acceleration, heading_forward)
        accel_left = _dot(acceleration, heading_left)
        pitch_angle = math.atan2(accel_forward, 9.81)
        roll_angle = -math.atan2(accel_left, 9.81)
        pitch_angle = max(-self.config.max_tilt_rad, min(self.config.max_tilt_rad, pitch_angle))
        roll_angle = max(-self.config.max_tilt_rad, min(self.config.max_tilt_rad, roll_angle))
        pitch = max(
            -self.config.max_horizontal_stick,
            min(
                self.config.max_horizontal_stick,
                pitch_angle / self.config.stick_angle_rad_per_unit,
            ),
        )
        roll = max(
            -self.config.max_horizontal_stick,
            min(
                self.config.max_horizontal_stick,
                roll_angle / self.config.stick_angle_rad_per_unit,
            ),
        )
        yaw_error = wrap_to_pi(yaw_setpoint - state.yaw)
        yaw = -self.config.yaw_kp * yaw_error + self.config.yaw_rate_damping * state.yaw_rate
        yaw = max(-self.config.max_yaw_stick, min(self.config.max_yaw_stick, yaw))
        throttle = (
            self.config.hover_rc_trim
            + self.config.kp_altitude * (altitude_setpoint - state.position[2])
            - self.config.kd_altitude * state.velocity[2]
        )
        throttle = max(self.config.throttle_min, min(self.config.throttle_max, throttle))
        return SemanticControl(
            roll=roll,
            pitch=pitch,
            throttle=throttle,
            yaw=yaw,
            armed=True,
            angle_mode=True,
        )


__all__ = [
    "GUIDANCE_PHASE_NAMES",
    "GuidancePhase",
    "TruthGuidance",
    "TruthGuidanceConfig",
    "wrap_to_pi",
]
