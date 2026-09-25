"""Deterministic live qualification for Package C's positive referee path.

This module is dependency-light so configuration and acceptance logic stay in the
fast pure pytest suite.  The live adapter in ``main.py`` supplies observations
from the production post-step/referee/telemetry path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from referee import (
    CORE_COURSE_GATE_CAPACITY,
    PASS_TIME_UNSET_S,
    GatePassEvent,
    RaceResult,
    Vec3,
)

AUDIT_ENV = "RACE_REFEREE_AUDIT"
AUDIT_INITIAL_POSITION = (5.0, 0.0, 4.9)
AUDIT_INITIAL_VELOCITY = (10.0, 0.0, 0.0)
AUDIT_DURATION_S = 2.5
AUDIT_EXPECTED_PASS_TIME_RANGE_S = (0.9, 1.3)
AUDIT_MIN_DEPARTURE_X_M = 10.5
AUDIT_PASS_TIME_TOLERANCE_S = 1.0e-9


@dataclass(frozen=True, slots=True)
class RefereeAuditConfig:
    """Validated opt-in configuration for the qualification fixture."""

    enabled: bool


def referee_audit_from_env(
    env: Mapping[str, str],
    *,
    course_name: str,
    guidance_mode: str,
    manual_audit_requested: bool,
) -> RefereeAuditConfig:
    """Parse and validate the narrow referee-audit environment contract.

    The fixture deliberately retains the normal scripted command source, whose
    first five simulated seconds are disarmed and safe.  Since this audit ends
    at 2.5 seconds, its motion comes only from the controlled initial condition
    and ordinary six-DOF physics, not an RC steering command.
    """

    value = env.get(AUDIT_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{AUDIT_ENV} must be '0' or '1'")
    enabled = value == "1"
    if not enabled:
        return RefereeAuditConfig(enabled=False)
    if course_name != "single":
        raise ValueError(f"{AUDIT_ENV}=1 requires RACE_COURSE=single")
    if guidance_mode != "scripted":
        raise ValueError(
            f"{AUDIT_ENV}=1 requires RACE_GUIDANCE=scripted; "
            "the qualification bypasses steering during its short safe boot phase"
        )
    if manual_audit_requested:
        raise ValueError(f"{AUDIT_ENV}=1 is incompatible with RACE_MANUAL_AUDIT=1")
    return RefereeAuditConfig(enabled=True)


def audit_initial_condition(
    enabled: bool,
    default_position: Sequence[float],
    default_velocity: Sequence[float],
    default_duration: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return fixture values when enabled and untouched defaults otherwise."""

    if enabled:
        return AUDIT_INITIAL_POSITION, AUDIT_INITIAL_VELOCITY, AUDIT_DURATION_S
    return (
        tuple(float(value) for value in default_position),
        tuple(float(value) for value in default_velocity),
        float(default_duration),
    )


@dataclass(frozen=True, slots=True)
class RefereeAuditEvidence:
    """End-of-run evidence collected exclusively from the live adapter path."""

    events: tuple[GatePassEvent, ...]
    telemetry_last_gate_passed: int | None
    telemetry_pass_times: tuple[float, ...] | None
    telemetry_verified_on_later_tick: bool
    race_result: RaceResult
    first_position: tuple[float, float, float] | None
    crossing_previous_position: tuple[float, float, float] | None
    crossing_current_position: tuple[float, float, float] | None
    final_position: tuple[float, float, float] | None
    race_line_count: int
    audit_line_count: int


@dataclass(slots=True)
class RefereeAuditCollector:
    """Collect live audit observations without changing post-step ordering."""

    previous_position: Vec3
    events: list[GatePassEvent] = field(default_factory=list)
    event_tick: int | None = None
    first_position: Vec3 | None = None
    crossing_previous_position: Vec3 | None = None
    crossing_current_position: Vec3 | None = None
    final_position: Vec3 | None = None
    telemetry_last_gate_passed: int | None = None
    telemetry_pass_times: tuple[float, ...] | None = None
    telemetry_verified_on_later_tick: bool = False

    def record_truth_read(self, position: Vec3) -> None:
        if self.first_position is None:
            self.first_position = position
        self.final_position = position

    def telemetry_due(self, tick: int) -> bool:
        return (
            self.event_tick is not None
            and tick > self.event_tick
            and not self.telemetry_verified_on_later_tick
        )

    def record_telemetry(self, last_gate_passed: int, pass_times: tuple[float, ...]) -> None:
        self.telemetry_last_gate_passed = last_gate_passed
        self.telemetry_pass_times = pass_times
        self.telemetry_verified_on_later_tick = True

    def record_scoring_tick(self, position: Vec3, tick: int, event: GatePassEvent | None) -> None:
        if event is not None:
            self.events.append(event)
            self.event_tick = tick
            self.crossing_previous_position = self.previous_position
            self.crossing_current_position = position
        self.previous_position = position

    def evidence(
        self, race_result: RaceResult, race_line_count: int, audit_line_count: int
    ) -> RefereeAuditEvidence:
        return RefereeAuditEvidence(
            events=tuple(self.events),
            telemetry_last_gate_passed=self.telemetry_last_gate_passed,
            telemetry_pass_times=self.telemetry_pass_times,
            telemetry_verified_on_later_tick=self.telemetry_verified_on_later_tick,
            race_result=race_result,
            first_position=self.first_position,
            crossing_previous_position=self.crossing_previous_position,
            crossing_current_position=self.crossing_current_position,
            final_position=self.final_position,
            race_line_count=race_line_count,
            audit_line_count=audit_line_count,
        )


@dataclass(frozen=True, slots=True)
class RefereeAuditResult:
    """Stable pass/fail result with an integration-friendly exit status."""

    passed: bool
    gate: int | None
    passes: int
    telemetry: bool
    race_status: str
    pass_time: float | None
    failed_checks: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def format(self) -> str:
        gate = "na" if self.gate is None else str(self.gate)
        pass_time = "na" if self.pass_time is None else f"{self.pass_time:.6f}"
        return (
            f"[C-REFEREE-AUDIT] gate={gate} passes={self.passes} "
            f"telemetry={str(self.telemetry).lower()} result={self.race_status} "
            f"pass_time={pass_time} status={'PASS' if self.passed else 'FAIL'}"
        )


def _finite_position(position: tuple[float, float, float] | None) -> bool:
    return position is not None and len(position) == 3 and all(math.isfinite(v) for v in position)


def evaluate_referee_audit(evidence: RefereeAuditEvidence) -> RefereeAuditResult:
    """Evaluate every Package C live positive-path acceptance criterion."""

    failures: list[str] = []
    events = evidence.events
    event = events[0] if len(events) == 1 else None
    event_gate = event.gate_index if event is not None else None
    event_time = event.pass_time if event is not None else None

    if len(events) != 1:
        failures.append("event_count")
    if event_gate != 0:
        failures.append("event_gate")
    if event_time is None or not (
        AUDIT_EXPECTED_PASS_TIME_RANGE_S[0] <= event_time <= AUDIT_EXPECTED_PASS_TIME_RANGE_S[1]
    ):
        failures.append("pass_time_window")

    telemetry_ok = True
    telemetry_times = evidence.telemetry_pass_times
    if not evidence.telemetry_verified_on_later_tick:
        failures.append("telemetry_later_tick")
        telemetry_ok = False
    if evidence.telemetry_last_gate_passed != 0:
        failures.append("telemetry_last_gate")
        telemetry_ok = False
    if telemetry_times is None or len(telemetry_times) != CORE_COURSE_GATE_CAPACITY:
        failures.append("telemetry_width")
        telemetry_ok = False
    elif event_time is None or not math.isclose(
        telemetry_times[0],
        event_time,
        rel_tol=0.0,
        abs_tol=AUDIT_PASS_TIME_TOLERANCE_S,
    ):
        failures.append("telemetry_pass_time")
        telemetry_ok = False
    if telemetry_times is not None and tuple(telemetry_times[1:]) != (
        PASS_TIME_UNSET_S,
        PASS_TIME_UNSET_S,
    ):
        failures.append("telemetry_unset_slots")
        telemetry_ok = False

    race = evidence.race_result
    if not (
        race.course == "single"
        and race.gates_passed == 1
        and race.gate_count == 1
        and race.status == "COMPLETE"
        and len(race.pass_times) == 1
        and event_time is not None
        and math.isclose(
            race.pass_times[0],
            event_time,
            rel_tol=0.0,
            abs_tol=AUDIT_PASS_TIME_TOLERANCE_S,
        )
    ):
        failures.append("race_result")

    first = evidence.first_position
    before = evidence.crossing_previous_position
    after = evidence.crossing_current_position
    final = evidence.final_position
    if not (
        _finite_position(first)
        and math.isclose(first[0], AUDIT_INITIAL_POSITION[0], rel_tol=0.0, abs_tol=0.01)
        and math.isclose(first[1], AUDIT_INITIAL_POSITION[1], rel_tol=0.0, abs_tol=0.01)
        and first[2] <= AUDIT_INITIAL_POSITION[2] + 0.01
    ):
        failures.append("initial_position")
    if not (
        _finite_position(before)
        and _finite_position(after)
        and before[0] < 10.0 <= after[0]
        and after[0] > before[0]
    ):
        failures.append("forward_crossing_segment")
    if not (_finite_position(final) and final[0] >= AUDIT_MIN_DEPARTURE_X_M):
        failures.append("visible_departure")

    if evidence.race_line_count != 1:
        failures.append("race_line_count")
    if evidence.audit_line_count != 1:
        failures.append("audit_line_count")

    return RefereeAuditResult(
        passed=not failures,
        gate=event_gate,
        passes=len(events),
        telemetry=telemetry_ok,
        race_status=race.status,
        pass_time=event_time,
        failed_checks=tuple(failures),
    )
