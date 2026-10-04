"""Pure ordered gate-crossing referee for Betaflight SITL racing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields

from course import Course, Gate

# Allows only floating-point roundoff at a mathematically inclusive inner edge;
# it is one billionth of a metre and is not a physical expansion of the gate.
GEOMETRY_EPSILON_M = 1.0e-9
LAST_GATE_UNSET = -1
PASS_TIME_UNSET_S = -1.0
CORE_COURSE_GATE_CAPACITY = 3


Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PlaneCrossing:
    """Interpolated gate-local plane intersection used only by the referee."""

    fraction: float
    local_position: Vec3


@dataclass(frozen=True, slots=True)
class GatePassEvent:
    """The only per-crossing information exposed outside the referee."""

    gate_index: int
    pass_time: float


@dataclass(frozen=True, slots=True)
class RaceProgress:
    """Narrow immutable race view safe to provide to any guidance mode.

    This intentionally contains ordered progress and public course rules only:
    no gate center, normal, yaw, crossing point, or drone truth state.
    """

    last_gate_passed: int
    next_gate_index: int | None
    gate_count: int
    gate_inner_size: float | None


@dataclass(frozen=True, slots=True)
class RaceResult:
    course: str
    gates_passed: int
    gate_count: int
    lap_time: float | None
    pass_times: tuple[float, ...]

    @property
    def status(self) -> str:
        return "COMPLETE" if self.gates_passed == self.gate_count else "INCOMPLETE"

    def format(self) -> str:
        lap_time = "na" if self.lap_time is None else f"{self.lap_time:.6f}"
        times = ",".join(f"{value:.6f}" for value in self.pass_times)
        return (
            f"[RACE] course={self.course} "
            f"gates_passed={self.gates_passed}/{self.gate_count} "
            f"lap_time={lap_time} status={self.status} pass_times=[{times}]"
        )


def interpolate_forward_plane_crossing(
    gate: Gate,
    previous_world: Sequence[float],
    current_world: Sequence[float],
) -> PlaneCrossing | None:
    """Interpolate a forward segment's intersection with gate-local ``x = 0``.

    The directional test is exactly ``previous_x < 0 <= current_x``. Looking at
    the whole segment, rather than only either endpoint, prevents a fast vehicle
    from tunnelling through the scoring plane between physics ticks.
    """

    previous = gate.world_to_local(previous_world)
    current = gate.world_to_local(current_world)
    if not (previous[0] < 0.0 <= current[0]):
        return None

    delta_x = current[0] - previous[0]
    # The directional predicate guarantees a strictly positive denominator.
    fraction = -previous[0] / delta_x
    local_y = previous[1] + fraction * (current[1] - previous[1])
    local_z = previous[2] + fraction * (current[2] - previous[2])
    return PlaneCrossing(fraction=fraction, local_position=(0.0, local_y, local_z))


def crossing_through_gate_opening(
    gate: Gate,
    previous_world: Sequence[float],
    current_world: Sequence[float],
) -> PlaneCrossing | None:
    """Return the interpolated forward crossing when it lies in the opening."""

    crossing = interpolate_forward_plane_crossing(gate, previous_world, current_world)
    if crossing is None:
        return None
    half_inner = gate.inner_size * 0.5
    _, local_y, local_z = crossing.local_position
    if (
        abs(local_y) <= half_inner + GEOMETRY_EPSILON_M
        and abs(local_z) <= half_inner + GEOMETRY_EPSILON_M
    ):
        return crossing
    return None


def crosses_gate_opening(
    gate: Gate,
    previous_world: Sequence[float],
    current_world: Sequence[float],
) -> bool:
    """Return whether a segment crosses forward through the square opening."""

    return crossing_through_gate_opening(gate, previous_world, current_world) is not None


def world_position_from_transform(world_pos: Sequence[float]) -> Vec3:
    """Extract ENU XYZ from Elodin ``[qx, qy, qz, qw, x, y, z]`` layout."""

    if len(world_pos) != 7:
        raise ValueError("world_pos must use [qx, qy, qz, qw, x, y, z] layout")
    position = (float(world_pos[4]), float(world_pos[5]), float(world_pos[6]))
    if not all(math.isfinite(value) for value in position):
        raise ValueError("world_pos position must be finite")
    return position


class Referee:
    """Stateful truth scorer that only counts the next ordered gate."""

    def __init__(self, course: Course):
        self._course = course
        self._previous_position: Vec3 | None = None
        self._previous_sim_time: float | None = None
        self._pass_times: list[float] = []

    @property
    def pass_times(self) -> tuple[float, ...]:
        return tuple(self._pass_times)

    def progress(self) -> RaceProgress:
        passed = len(self._pass_times)
        count = len(self._course.gates)
        return RaceProgress(
            last_gate_passed=passed - 1 if passed else LAST_GATE_UNSET,
            next_gate_index=passed if passed < count else None,
            gate_count=count,
            gate_inner_size=self._course.inner_size,
        )

    def observe_truth(
        self, current_position: Sequence[float], sim_time: float
    ) -> GatePassEvent | None:
        """Score one truth sample and return at most one public ordered event.

        A successful pass time is linearly interpolated between the previous and
        current *simulation* sample times using the geometric plane-crossing
        fraction. It never uses wall-clock time. This gives sub-tick timing for
        fast segments while retaining deterministic sampled-segment semantics.
        """

        position = tuple(float(value) for value in current_position)
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("current truth position must contain three finite coordinates")
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise ValueError("simulation time must be finite and non-negative")
        if self._previous_sim_time is not None and sim_time < self._previous_sim_time:
            raise ValueError("simulation time must be monotonic")

        event = None
        next_index = len(self._pass_times)
        if (
            self._previous_position is not None
            and self._previous_sim_time is not None
            and next_index < len(self._course.gates)
        ):
            gate = self._course.gates[next_index]
            crossing = crossing_through_gate_opening(gate, self._previous_position, position)
            if crossing is not None:
                pass_time = self._previous_sim_time + crossing.fraction * (
                    sim_time - self._previous_sim_time
                )
                self._pass_times.append(pass_time)
                event = GatePassEvent(gate_index=gate.index, pass_time=pass_time)

        self._previous_position = position
        self._previous_sim_time = float(sim_time)
        return event

    def telemetry_pass_times(self) -> tuple[float, ...]:
        """Return the fixed-width three-gate telemetry array.

        Unpassed slots use ``PASS_TIME_UNSET_S == -1.0``. Package C supports at
        most one gate, while the stable array already reserves all three slots
        needed by the core course introduced in Package F.
        """

        if len(self._pass_times) > CORE_COURSE_GATE_CAPACITY:
            raise ValueError("race telemetry supports at most three gates")
        return tuple(self._pass_times) + (PASS_TIME_UNSET_S,) * (
            CORE_COURSE_GATE_CAPACITY - len(self._pass_times)
        )

    def result(self) -> RaceResult:
        passed = len(self._pass_times)
        count = len(self._course.gates)
        complete = passed == count
        # Lap time runs from simulation time zero to the interpolated final
        # plane crossing. A nonempty course has no lap time until completion.
        lap_time = self._pass_times[-1] if complete and count > 0 else None
        return RaceResult(
            course=self._course.name,
            gates_passed=passed,
            gate_count=count,
            lap_time=lap_time,
            pass_times=tuple(self._pass_times),
        )


def public_progress_field_names() -> tuple[str, ...]:
    """Expose the stable adapter surface for focused truth-isolation tests."""

    return tuple(field.name for field in fields(RaceProgress))
