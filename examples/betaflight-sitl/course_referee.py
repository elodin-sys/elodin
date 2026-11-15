"""Independent physical completion/safety grader for the truth C1 scenario.

COMPLETE alone is insufficient. The host measures a final continuous grounded,
upright, slow, exchanged-disarmed interval with stopped motors and healthy I/O.
No guidance target, phase-completion flag, or guidance-provided grade is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from controls import RC_ARMED, RcCommand
from referee import RaceResult


@dataclass(frozen=True, slots=True)
class CourseQualification:
    complete: bool
    safe_duration_s: float
    final_time_s: float | None
    fault: bool
    failure_reason: str | None
    flight_motor_seen: bool

    @property
    def passed(self) -> bool:
        return (
            self.complete
            and self.safe_duration_s >= 1.0 - 1e-9
            and self.final_time_s is not None
            and self.final_time_s <= 60.0
            and self.flight_motor_seen
            and not self.fault
            and self.failure_reason is None
        )

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def format(self) -> str:
        reason = self.failure_reason or (
            "guidance_fault"
            if self.fault
            else "incomplete"
            if not self.complete
            else "no_flight_motor_response"
            if not self.flight_motor_seen
            else "unsafe_finish"
            if self.safe_duration_s < 1.0 - 1e-9
            else "qualified"
        )
        time = "na" if self.final_time_s is None else f"{self.final_time_s:.6f}"
        return (
            f"[F-COURSE] status={'PASS' if self.passed else 'FAIL'} "
            f"complete={str(self.complete).lower()} safe_duration={self.safe_duration_s:.6f} "
            f"final_time={time} fault={str(self.fault).lower()} reason={reason}"
        )


class TruthCourseReferee:
    """Bounded scalar evidence only; final safety must persist to the last tick.

    Safety envelope: <=35 deg tilt, <=3 m/s total speed, 0..4 m altitude;
    downward speed <=0.65 m/s below 0.20 m. After boot (5 s) every tick must
    supply finite truth, motor and lockstep evidence with no skipped samples.
    Finish: >=1 continuous second after ordered completion at z<=0.15 m,
    speed<=0.20 m/s, tilt<=10 deg, actual RC disarm/minimum throttle/centered
    axes, all motors<=0.02. Failure is sticky, including after an earlier safe
    interval. This grader is independent of the controller's landing thresholds.
    """

    def __init__(self, *, max_sample_gap_s: float):
        if not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0:
            raise ValueError("max_sample_gap_s must be positive and finite")
        self.max_sample_gap_s = max_sample_gap_s
        self._last_time: float | None = None
        self._safe_since: float | None = None
        self._safe_duration_s = 0.0
        self._failure: str | None = None
        self._fault = False
        self._flight_motor_seen = False

    def _fail(self, reason: str) -> None:
        if self._failure is None:
            self._failure = reason
        self._safe_since = None
        self._safe_duration_s = 0.0

    def observe_truth(
        self,
        sim_time: float,
        world_pos: Sequence[float] | None,
        world_vel: Sequence[float] | None,
        command_sent: RcCommand,
        motors: Sequence[float],
        *,
        lockstep_response: bool,
        race_complete: bool,
        guidance_fault: bool,
    ) -> None:
        self._fault |= guidance_fault
        if not math.isfinite(sim_time) or sim_time < 0 or sim_time > 60.0:
            self._fail("invalid_time")
            return
        previous = self._last_time
        self._last_time = sim_time
        if previous is not None and (
            sim_time <= previous or (sim_time >= 5 and sim_time - previous > self.max_sample_gap_s)
        ):
            self._fail("sample_gap_or_time_reversal")
        try:
            pos = tuple(float(v) for v in world_pos) if world_pos is not None else ()
            vel = tuple(float(v) for v in world_vel) if world_vel is not None else ()
            motor = tuple(float(v) for v in motors)
            valid = (
                len(pos) == 7
                and len(vel) == 6
                and len(motor) == 4
                and all(math.isfinite(v) for v in pos + vel + motor)
                and abs(sum(q * q for q in pos[:4]) - 1.0) <= 1e-3
                and all(0.0 <= v <= 1.0 for v in motor)
            )
        except (ValueError, TypeError):
            valid = False
        if not valid or not lockstep_response:
            self._safe_since = None
            self._safe_duration_s = 0.0
            if sim_time >= 5.0:
                self._fail("missing_truth_or_lockstep_or_motors")
            return
        speed = math.hypot(*vel[3:])
        upright_cos = 1.0 - 2.0 * (pos[0] ** 2 + pos[1] ** 2)
        if (
            speed > 3.0
            or not -0.01 <= pos[6] <= 4.0
            or upright_cos < math.cos(math.radians(35.0))
            or (self._flight_motor_seen and pos[6] < 0.20 and vel[5] < -0.65)
        ):
            self._fail("unsafe_flight")
        self._flight_motor_seen |= (
            command_sent.arm >= RC_ARMED and pos[6] > 0.5 and max(motor) > 0.05
        )
        safe = (
            race_complete
            and not self._fault
            and self._failure is None
            and pos[6] <= 0.15
            and speed <= 0.20
            and upright_cos >= math.cos(math.radians(10.0))
            and command_sent.arm < RC_ARMED
            and command_sent.throttle == 1000
            and (command_sent.roll, command_sent.pitch, command_sent.yaw) == (1500, 1500, 1500)
            and max(motor) <= 0.02
        )
        if safe:
            if self._safe_since is None:
                self._safe_since = sim_time
            self._safe_duration_s = sim_time - self._safe_since
        else:
            self._safe_since = None
            self._safe_duration_s = 0.0

    def result(
        self, race_result: RaceResult, *, adapter_failure_reason: str | None = None
    ) -> CourseQualification:
        return CourseQualification(
            complete=(
                race_result.course == "c1_straight"
                and race_result.gates_passed == race_result.gate_count == 3
                and len(race_result.pass_times) == 3
                and all(math.isfinite(t) and 0.0 <= t <= 60.0 for t in race_result.pass_times)
                and all(a < b for a, b in zip(race_result.pass_times, race_result.pass_times[1:]))
                and race_result.lap_time == race_result.pass_times[-1]
            ),
            safe_duration_s=self._safe_duration_s,
            final_time_s=self._last_time,
            fault=self._fault,
            failure_reason=self._failure or adapter_failure_reason,
            flight_motor_seen=self._flight_motor_seen,
        )
