"""RC and guidance contracts for the Betaflight racing example.

This module deliberately has no Elodin or Betaflight process dependencies so the
command boundary and its failsafe behavior can be tested offline.
"""

from __future__ import annotations

import enum
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

RC_MIN = 1000
RC_CENTER = 1500
RC_MAX = 2000
RC_ARMED = 1800
RC_ANGLE = 1800
RC_CHANNEL_COUNT = 6
MANUAL_CONTROL_COUNT = 7
MANUAL_STALE_AFTER_S = 0.25


class GuidanceMode(enum.StrEnum):
    """Configured source of RC commands."""

    SCRIPTED = "scripted"
    MANUAL = "manual"
    TRUTH = "truth"
    VISION = "vision"


def guidance_mode_from_env(environ: Mapping[str, str] = os.environ) -> GuidanceMode:
    """Parse ``RACE_GUIDANCE``, rejecting unknown and not-yet-landed modes."""

    value = environ.get("RACE_GUIDANCE", GuidanceMode.SCRIPTED.value).strip().lower()
    try:
        mode = GuidanceMode(value)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in GuidanceMode)
        raise ValueError(f"unknown RACE_GUIDANCE={value!r}; expected one of: {choices}") from exc
    if mode in (GuidanceMode.TRUTH, GuidanceMode.VISION):
        raise ValueError(f"RACE_GUIDANCE={value!r} is reserved but not implemented yet")
    return mode


def _clamp_int(value: float, low: int = RC_MIN, high: int = RC_MAX) -> int:
    if not math.isfinite(value):
        return low
    return max(low, min(high, round(value)))


def _direct_axis_to_pwm(value: float) -> int:
    """Map a normalized axis directly to centered PWM with clamping.

    The measured semantic signs are applied by ``semantic_to_rc``: roll and
    pitch are direct, while yaw is inverted.
    """

    value = 0.0 if not math.isfinite(value) else max(-1.0, min(1.0, value))
    return _clamp_int(RC_CENTER + value * (RC_MAX - RC_CENTER))


@dataclass(frozen=True, slots=True)
class SemanticControl:
    """Vehicle-independent pilot intent consumed by the RC conversion seam."""

    roll: float = 0.0
    pitch: float = 0.0
    throttle: float = 0.0
    yaw: float = 0.0
    armed: bool = False
    angle_mode: bool = True

    @classmethod
    def safe(cls) -> "SemanticControl":
        """The manual failsafe: level sticks, minimum throttle, disarmed."""

        return cls()


@dataclass(frozen=True, slots=True)
class RcCommand:
    """The six stable AETR/AUX RC channels sent to Betaflight."""

    roll: int = RC_CENTER
    pitch: int = RC_CENTER
    throttle: int = RC_MIN
    yaw: int = RC_CENTER
    arm: int = RC_MIN
    mode: int = RC_ANGLE

    def __post_init__(self) -> None:
        for name in ("roll", "pitch", "throttle", "yaw", "arm", "mode"):
            object.__setattr__(self, name, _clamp_int(getattr(self, name)))

    def as_array(self) -> NDArray[np.uint16]:
        return np.array(
            [self.roll, self.pitch, self.throttle, self.yaw, self.arm, self.mode],
            dtype=np.uint16,
        )

    def fill_channels(self, channels: NDArray[np.uint16]) -> NDArray[np.uint16]:
        """Fill all RC channels deterministically and return ``channels``."""

        if channels.ndim != 1 or len(channels) < RC_CHANNEL_COUNT:
            raise ValueError(
                "RC channel buffer must be a one-dimensional array with at least 6 entries"
            )
        channels.fill(RC_CENTER)
        channels[:RC_CHANNEL_COUNT] = self.as_array()
        return channels


def semantic_to_rc(control: SemanticControl) -> RcCommand:
    """Convert semantic control to clamped AETR/AUX PWM in exactly one path."""

    throttle = control.throttle if math.isfinite(control.throttle) else 0.0
    throttle = max(0.0, min(1.0, throttle))
    return RcCommand(
        roll=_direct_axis_to_pwm(control.roll),
        pitch=_direct_axis_to_pwm(control.pitch),
        throttle=_clamp_int(RC_MIN + throttle * (RC_MAX - RC_MIN)),
        # Positive semantic yaw is nose-right/clockwise. The physical SITL
        # audit measured that direction below RC center.
        yaw=_direct_axis_to_pwm(-control.yaw),
        arm=RC_ARMED if control.armed else RC_MIN,
        mode=RC_ANGLE if control.angle_mode else RC_MIN,
    )


def _immutable_copy(value: NDArray[np.generic] | None) -> NDArray[np.generic] | None:
    if value is None:
        return None
    copy = np.array(value, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, slots=True)
class GuidanceUpdate:
    """I/O-free update supplied to a guidance source once per physics tick.

    Truth state and gate poses intentionally are not fields on this base type.
    Future vision guidance can therefore receive this type without having a
    nullable back door to referee truth.
    """

    sim_time: float
    tick: int
    gyro: NDArray[np.float64]
    accel: NDArray[np.float64]
    barometer: float | None = None
    barometer_fresh: bool = False
    magnetometer: NDArray[np.float64] | None = None
    magnetometer_fresh: bool = False
    frame: NDArray[np.uint8] | None = None
    frame_sample_time: float | None = None
    frame_fresh: bool = False
    last_gate_passed: int = -1
    next_gate_index: int | None = None
    gate_count: int = 0
    gate_inner_size: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gyro", _immutable_copy(self.gyro))
        object.__setattr__(self, "accel", _immutable_copy(self.accel))
        object.__setattr__(self, "magnetometer", _immutable_copy(self.magnetometer))
        object.__setattr__(self, "frame", _immutable_copy(self.frame))


@dataclass(frozen=True, slots=True)
class TruthGuidanceUpdate(GuidanceUpdate):
    """Guidance update extension reserved for truth-guided modes."""

    truth_state: object | None = None
    gate_poses: tuple[object, ...] = field(default_factory=tuple)


class GuidanceSource(Protocol):
    """Stateful command source that cannot reach simulation I/O directly."""

    def update(self, update: GuidanceUpdate) -> SemanticControl: ...


@dataclass(slots=True)
class ScriptedGuidance:
    """Package A's timed takeoff, now behind the guidance boundary."""

    boot_grace: float = 5.0
    arm_duration: float = 2.0
    throttle_duration: float = 10.0
    flight_throttle: float = 0.4
    phase: str = "boot"

    def update(self, update: GuidanceUpdate) -> SemanticControl:
        t = update.sim_time
        if t < self.boot_grace:
            self.phase = "boot"
            return SemanticControl(angle_mode=False)
        if t < self.boot_grace + self.arm_duration:
            self.phase = "arm"
            return SemanticControl(armed=True, angle_mode=False)
        if t < self.boot_grace + self.arm_duration + self.throttle_duration:
            self.phase = "throttle"
            return SemanticControl(throttle=self.flight_throttle, armed=True, angle_mode=False)
        self.phase = "disarm"
        return SemanticControl(angle_mode=False)


@dataclass(frozen=True, slots=True)
class ManualSample:
    """Decoded semantic input plus transport heartbeat."""

    control: SemanticControl
    heartbeat: float

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> "ManualSample":
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (MANUAL_CONTROL_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError(f"manual control must contain {MANUAL_CONTROL_COUNT} finite values")
        heartbeat = float(values[6])
        if heartbeat <= 0.0:
            raise ValueError("manual control heartbeat must be positive")
        return cls(
            control=SemanticControl(
                roll=float(values[0]),
                pitch=float(values[1]),
                throttle=float(values[2]),
                yaw=float(values[3]),
                armed=bool(values[4] >= 0.5),
                angle_mode=bool(values[5] >= 0.5),
            ),
            heartbeat=heartbeat,
        )


@dataclass(slots=True)
class ManualGuidance:
    """Freshness monitor for externally supplied semantic pilot input."""

    stale_after_s: float = MANUAL_STALE_AFTER_S
    _last_heartbeat: float | None = None
    _heartbeat_seen_at: float | None = None
    fresh: bool = False
    reason: str = "controller absent"
    _control: SemanticControl = field(default_factory=SemanticControl.safe)

    def accept_input(self, values: NDArray[np.float64] | None, now: float) -> None:
        """Accept the adapter's latest DB sample without gaining DB access.

        The adapter calls this at the 8 kHz physics rate while the controller
        publishes at about 100 Hz. A heartbeat identifies a new transport
        sample, so unchanged samples only need the inexpensive age check. This
        keeps manual mode inside the lockstep real-time budget without changing
        its freshness semantics.
        """

        if values is None:
            self._last_heartbeat = None
            self._heartbeat_seen_at = None
            self.fresh = False
            self.reason = "controller absent"
            self._control = SemanticControl.safe()
            return

        values = np.asarray(values, dtype=np.float64)
        if values.shape != (MANUAL_CONTROL_COUNT,) or not math.isfinite(float(values[-1])):
            self.fresh = False
            self.reason = "invalid controller input"
            self._control = SemanticControl.safe()
            return

        heartbeat = float(values[-1])
        if heartbeat <= 0.0:
            self.fresh = False
            self.reason = "invalid controller input"
            self._control = SemanticControl.safe()
            return
        if self._last_heartbeat is None or heartbeat != self._last_heartbeat:
            try:
                sample = ManualSample.from_array(values)
            except (TypeError, ValueError):
                self.fresh = False
                self.reason = "invalid controller input"
                self._control = SemanticControl.safe()
                return
            self._last_heartbeat = sample.heartbeat
            self._heartbeat_seen_at = now
            self._control = sample.control

        age = (
            math.inf if self._heartbeat_seen_at is None else max(0.0, now - self._heartbeat_seen_at)
        )
        self.fresh = age <= self.stale_after_s
        if not self.fresh:
            self.reason = f"controller heartbeat stale ({age:.3f}s)"
            self._control = SemanticControl.safe()
            return

        self.reason = "fresh"

    def update(self, update: GuidanceUpdate) -> SemanticControl:
        """Return pilot intent for this tick through the common guidance API."""

        del update
        return self._control


@dataclass(slots=True)
class DelayedRcCommand:
    """One-tick command latch used by the post-step lockstep adapter."""

    current: RcCommand = field(default_factory=lambda: semantic_to_rc(SemanticControl.safe()))

    def command_for_exchange(self) -> RcCommand:
        return self.current

    def retain_for_next_tick(self, command: RcCommand) -> None:
        self.current = command
