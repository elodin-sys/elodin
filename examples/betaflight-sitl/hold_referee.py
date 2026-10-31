"""Independent rolling truth hold grader for Betaflight Package E."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from course import Gate

Vec3 = tuple[float, float, float]
E_HOLD_WINDOW_S = 10.0
E_HOLD_RMS_LIMIT_M = 0.30
E_STANDOFF_DISTANCE_M = 4.0


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True, slots=True)
class HoldEligibility:
    """Independent evidence that one truth sample is eligible for E grading."""

    phase_hold: bool
    armed: bool
    angle_mode: bool
    lockstep_response: bool
    motor_response: bool

    @property
    def eligible(self) -> bool:
        return all(
            (
                self.phase_hold,
                self.armed,
                self.angle_mode,
                self.lockstep_response,
                self.motor_response,
            )
        )


@dataclass(frozen=True, slots=True)
class _HoldSample:
    sim_time: float
    error: Vec3
    cumulative_squared_error_integral: float

    @property
    def squared_error(self) -> float:
        return _dot(self.error, self.error)


@dataclass(frozen=True, slots=True)
class _HoldWindow:
    start_time: float
    end_time: float
    rms_error_m: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class HoldQualification:
    """Final immutable Package E hold grade, separate from the race schema."""

    target_enu: Vec3
    duration_s: float
    threshold_m: float
    window: _HoldWindow | None
    gate_crossed: bool
    guidance_fault: bool
    armed_seen: bool
    angle_seen: bool
    lockstep_seen: bool
    motor_seen: bool
    eligible_samples: int
    reason: str

    @property
    def passed(self) -> bool:
        return (
            self.window is not None
            and self.window.rms_error_m <= self.threshold_m + 1.0e-12
            and not self.gate_crossed
            and not self.guidance_fault
        )

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def format(self) -> str:
        if self.window is None:
            window_text = "na"
            duration_text = "na"
            rms_text = "na"
        else:
            window_text = f"[{self.window.start_time:.6f},{self.window.end_time:.6f}]"
            duration_text = f"{self.window.end_time - self.window.start_time:.6f}"
            rms_text = f"{self.window.rms_error_m:.6f}"
        target = ",".join(f"{value:.6f}" for value in self.target_enu)
        return (
            f"[E-HOLD] status={self.status} target=({target}) window={window_text} "
            f"duration={duration_text} rms={rms_text} threshold={self.threshold_m:.6f} "
            f"control=armed:{str(self.armed_seen).lower()},angle:{str(self.angle_seen).lower()},"
            f"lockstep:{str(self.lockstep_seen).lower()},motor:{str(self.motor_seen).lower()} "
            f"eligible_samples={self.eligible_samples} crossing={str(self.gate_crossed).lower()} "
            f"fault={str(self.guidance_fault).lower()} reason={self.reason}"
        )


class StandoffHoldReferee:
    """Independent rolling time-weighted 3D RMS grader for one truth gate.

    It retains only the current contiguous eligible interval, capped by
    ``max_samples``, plus scalar first-pass/best-window summaries. The metric is
    the trapezoidal integral of squared 3D position error with exact interpolation
    at the rolling window's start and end times.
    """

    def __init__(
        self,
        gate: Gate,
        *,
        standoff_distance_m: float = E_STANDOFF_DISTANCE_M,
        duration_s: float = E_HOLD_WINDOW_S,
        threshold_m: float = E_HOLD_RMS_LIMIT_M,
        max_sample_gap_s: float | None = None,
        max_samples: int = 100_000,
    ):
        values = (standoff_distance_m, duration_s, threshold_m)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("standoff distance, hold duration, and RMS threshold must be positive")
        if max_sample_gap_s is not None and (
            not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0.0
        ):
            raise ValueError("max_sample_gap_s must be positive and finite")
        if max_samples < 2:
            raise ValueError("max_samples must be at least two")
        self._gate = gate
        normal = gate.normal
        self.target_enu = (
            gate.center[0] - standoff_distance_m * normal[0],
            gate.center[1] - standoff_distance_m * normal[1],
            gate.center[2],
        )
        self.duration_s = float(duration_s)
        self.threshold_m = float(threshold_m)
        self._max_sample_gap_s = max_sample_gap_s
        self._max_samples = max_samples
        self._samples: deque[_HoldSample] = deque()
        self._previous_time: float | None = None
        self._previous_local_x: float | None = None
        self._gate_crossed = False
        self._guidance_fault = False
        self._passed_window: _HoldWindow | None = None
        self._best_window: _HoldWindow | None = None
        self._reason = "no_qualified_window"
        self._eligible_samples = 0
        self._armed_seen = False
        self._angle_seen = False
        self._lockstep_seen = False
        self._motor_seen = False

    @property
    def gate_crossed(self) -> bool:
        return self._gate_crossed

    def observe_truth(
        self,
        position_enu: Sequence[float] | None,
        sim_time: float,
        eligibility: HoldEligibility,
        *,
        gate_crossed: bool = False,
        phase_fault: bool = False,
    ) -> None:
        """Observe one sample and reset interrupted candidates deterministically."""

        if gate_crossed:
            self._gate_crossed = True
        if phase_fault:
            self._guidance_fault = True
        if not math.isfinite(sim_time) or sim_time < 0.0:
            self._reset_candidate("invalid_time")
            self._previous_time = None
            self._previous_local_x = None
            return
        if position_enu is None:
            self._reset_candidate("invalid_truth")
            if self._previous_time is not None and sim_time <= self._previous_time:
                self._previous_local_x = None
            self._previous_time = float(sim_time)
            return
        try:
            position = tuple(float(value) for value in position_enu)
        except (TypeError, ValueError):
            position = ()
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            self._reset_candidate("invalid_truth")
            if self._previous_time is not None and sim_time <= self._previous_time:
                self._previous_local_x = None
            self._previous_time = float(sim_time)
            return

        local_x = self._gate.world_to_local(position)[0]
        if self._previous_time is not None and sim_time <= self._previous_time:
            self._reset_candidate("nonmonotonic_time")
            self._previous_time = float(sim_time)
            self._previous_local_x = None
            return
        # Only forward passage through the plane violates E's no-crossing
        # requirement. A backward crossing (or motion parallel to the plane)
        # is not a scored pass and must not permanently fail the run.
        if self._previous_local_x is not None and self._previous_local_x < 0.0 <= local_x:
            self._gate_crossed = True
        self._previous_time = float(sim_time)
        self._previous_local_x = local_x

        if self._gate_crossed or self._guidance_fault:
            self._reset_candidate("gate_crossing" if self._gate_crossed else "guidance_fault")
            return
        self._armed_seen |= eligibility.armed
        self._angle_seen |= eligibility.angle_mode
        self._lockstep_seen |= eligibility.lockstep_response
        self._motor_seen |= eligibility.motor_response
        if not eligibility.eligible:
            self._reset_candidate("ineligible_sample")
            return

        self._eligible_samples += 1
        error = (
            position[0] - self.target_enu[0],
            position[1] - self.target_enu[1],
            position[2] - self.target_enu[2],
        )
        if not all(math.isfinite(value) for value in error) or not math.isfinite(
            _dot(error, error)
        ):
            self._reset_candidate("invalid_truth")
            return
        if self._samples:
            previous = self._samples[-1]
            delta_t = sim_time - previous.sim_time
            if self._max_sample_gap_s is not None and delta_t > self._max_sample_gap_s:
                self._reset_candidate("sample_gap")
            elif len(self._samples) >= self._max_samples:
                self._reset_candidate("sample_capacity")

        if self._samples:
            previous = self._samples[-1]
            delta_t = sim_time - previous.sim_time
            cumulative = (
                previous.cumulative_squared_error_integral
                + delta_t * (previous.squared_error + _dot(error, error)) * 0.5
            )
            if not math.isfinite(cumulative):
                self._reset_candidate("invalid_truth")
                return
        else:
            cumulative = 0.0
        self._samples.append(_HoldSample(float(sim_time), error, cumulative))
        self._evaluate_latest_window()

    def _reset_candidate(self, reason: str) -> None:
        self._samples.clear()
        self._reason = reason

    def _evaluate_latest_window(self) -> None:
        if len(self._samples) < 2:
            return
        end = self._samples[-1]
        start_time = end.sim_time - self.duration_s
        if start_time < self._samples[0].sim_time:
            return

        # Keep the sample bracketing the moving exact-duration start and newer
        # samples only, so storage stays bounded by the rolling interval.
        while len(self._samples) >= 2 and self._samples[1].sim_time <= start_time:
            self._samples.popleft()
        if len(self._samples) < 2:
            return
        first, second = self._samples[0], self._samples[1]
        if start_time < first.sim_time or start_time > second.sim_time:
            return
        if start_time == first.sim_time:
            start_integral = first.cumulative_squared_error_integral
        else:
            fraction = (start_time - first.sim_time) / (second.sim_time - first.sim_time)
            boundary_squared = first.squared_error + fraction * (
                second.squared_error - first.squared_error
            )
            start_integral = first.cumulative_squared_error_integral + (
                (start_time - first.sim_time) * (first.squared_error + boundary_squared) * 0.5
            )
        squared_error_integral = end.cumulative_squared_error_integral - start_integral
        rms = math.sqrt(max(0.0, squared_error_integral / self.duration_s))
        window = _HoldWindow(
            start_time=start_time,
            end_time=end.sim_time,
            rms_error_m=rms,
            sample_count=len(self._samples),
        )
        if self._best_window is None or rms < self._best_window.rms_error_m:
            self._best_window = window
        if rms <= self.threshold_m + 1.0e-12 and self._passed_window is None:
            self._passed_window = window
            self._reason = "qualified"

    def result(self) -> HoldQualification:
        """Return the first passing, otherwise the best observed rolling window."""

        window = self._passed_window or self._best_window
        reason = (
            "gate_crossing"
            if self._gate_crossed
            else "guidance_fault"
            if self._guidance_fault
            else self._reason
        )
        return HoldQualification(
            target_enu=self.target_enu,
            duration_s=self.duration_s,
            threshold_m=self.threshold_m,
            window=window,
            gate_crossed=self._gate_crossed,
            guidance_fault=self._guidance_fault,
            armed_seen=self._armed_seen,
            angle_seen=self._angle_seen,
            lockstep_seen=self._lockstep_seen,
            motor_seen=self._motor_seen,
            eligible_samples=self._eligible_samples,
            reason=reason,
        )
