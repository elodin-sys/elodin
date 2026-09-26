import math
from dataclasses import FrozenInstanceError, fields

import pytest
from course import Course, Gate, course_from_name
from referee import (
    GEOMETRY_EPSILON_M,
    LAST_GATE_UNSET,
    PASS_TIME_UNSET_S,
    RaceProgress,
    Referee,
    crosses_gate_opening,
    interpolate_forward_plane_crossing,
    public_progress_field_names,
    world_position_from_transform,
)


def single_gate():
    return course_from_name("single").gates[0]


def local_segment(gate, previous, current):
    return gate.local_to_world(previous), gate.local_to_world(current)


def test_centered_forward_crossing_counts():
    gate = single_gate()
    previous, current = local_segment(gate, (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    crossing = interpolate_forward_plane_crossing(gate, previous, current)

    assert crossing is not None
    assert crossing.fraction == pytest.approx(0.5)
    assert crossing.local_position == pytest.approx((0.0, 0.0, 0.0))
    assert crosses_gate_opening(gate, previous, current)


@pytest.mark.parametrize(("axis", "sign"), [(1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)])
def test_each_inner_edge_is_inclusive(axis, sign):
    gate = single_gate()
    edge = gate.inner_size / 2.0
    local = [0.0, 0.0, 0.0]
    local[axis] = sign * edge
    previous = local.copy()
    current = local.copy()
    previous[0] = -1.0
    current[0] = 1.0
    previous_world, current_world = local_segment(gate, previous, current)

    assert crosses_gate_opening(gate, previous_world, current_world)


def test_inner_edge_allows_only_documented_roundoff_tolerance():
    gate = single_gate()
    edge = gate.inner_size / 2.0
    previous, current = local_segment(
        gate,
        (-1.0, edge + 0.5 * GEOMETRY_EPSILON_M, 0.0),
        (1.0, edge + 0.5 * GEOMETRY_EPSILON_M, 0.0),
    )

    assert crosses_gate_opening(gate, previous, current)


@pytest.mark.parametrize(("axis", "sign"), [(1, -1.0), (1, 1.0), (2, -1.0), (2, 1.0)])
def test_each_just_outside_inner_edge_is_rejected(axis, sign):
    gate = single_gate()
    outside = gate.inner_size / 2.0 + 2.0 * GEOMETRY_EPSILON_M
    local = [0.0, 0.0, 0.0]
    local[axis] = sign * outside
    previous = local.copy()
    current = local.copy()
    previous[0] = -1.0
    current[0] = 1.0
    previous_world, current_world = local_segment(gate, previous, current)

    assert not crosses_gate_opening(gate, previous_world, current_world)


def test_missed_crossing_is_rejected():
    gate = single_gate()
    previous, current = local_segment(gate, (-2.0, 2.0, 0.0), (3.0, 2.0, 0.0))

    assert interpolate_forward_plane_crossing(gate, previous, current) is not None
    assert not crosses_gate_opening(gate, previous, current)


def test_backward_crossing_is_rejected():
    gate = single_gate()
    previous, current = local_segment(gate, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))

    assert interpolate_forward_plane_crossing(gate, previous, current) is None
    assert not crosses_gate_opening(gate, previous, current)


def test_current_endpoint_on_plane_counts_with_exact_directional_predicate():
    gate = single_gate()
    previous, current = local_segment(gate, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    crossing = interpolate_forward_plane_crossing(gate, previous, current)

    assert crossing is not None
    assert crossing.fraction == pytest.approx(1.0)
    assert crosses_gate_opening(gate, previous, current)


def test_starting_on_plane_does_not_count_as_a_crossing():
    gate = single_gate()
    previous, current = local_segment(gate, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert not crosses_gate_opening(gate, previous, current)


def test_yawed_gate_crossing_uses_gate_local_coordinates():
    gate = Gate(index=0, center=(4.0, -3.0, 2.0), yaw=math.radians(37.0), inner_size=2.5)
    previous, current = local_segment(gate, (-2.0, 0.7, -0.4), (3.0, 0.7, -0.4))

    crossing = interpolate_forward_plane_crossing(gate, previous, current)

    assert crossing is not None
    assert crossing.fraction == pytest.approx(0.4)
    assert crossing.local_position == pytest.approx((0.0, 0.7, -0.4), abs=1e-12)
    assert crosses_gate_opening(gate, previous, current)


def test_fast_diagonal_segment_cannot_tunnel_through_plane():
    gate = single_gate()
    previous, current = local_segment(gate, (-100.0, -1.0, -1.0), (100.0, 1.0, 1.0))

    crossing = interpolate_forward_plane_crossing(gate, previous, current)

    assert crossing is not None
    assert crossing.fraction == pytest.approx(0.5)
    assert crossing.local_position == pytest.approx((0.0, 0.0, 0.0))
    assert crosses_gate_opening(gate, previous, current)


def test_referee_records_gate_once_and_updates_fixed_width_telemetry_once():
    referee = Referee(course_from_name("single"))

    assert referee.progress() == RaceProgress(LAST_GATE_UNSET, 0, 1, 2.5)
    assert referee.telemetry_pass_times() == (PASS_TIME_UNSET_S,) * 3
    assert referee.observe_truth((9.0, 0.0, 1.8), 1.0) is None

    event = referee.observe_truth((11.0, 0.0, 1.8), 2.0)
    assert event is not None
    assert event.gate_index == 0
    # Crossing is halfway through the sampled segment, so timing is
    # interpolated in simulation time rather than rounded to the current tick.
    assert event.pass_time == 1.5
    assert referee.progress() == RaceProgress(0, None, 1, 2.5)
    assert referee.telemetry_pass_times() == (1.5, PASS_TIME_UNSET_S, PASS_TIME_UNSET_S)

    # Further motion, including another forward crossing, cannot duplicate a
    # completed ordered gate or mutate its pass time.
    assert referee.observe_truth((9.0, 0.0, 1.8), 3.0) is None
    assert referee.observe_truth((11.0, 0.0, 1.8), 4.0) is None
    assert referee.pass_times == (1.5,)


def test_only_next_ordered_gate_can_count():
    course = Course(
        "synthetic_ordered",
        (
            Gate(0, (10.0, 0.0, 1.8), 0.0, 2.5),
            Gate(1, (20.0, 0.0, 1.8), 0.0, 2.5),
        ),
    )
    referee = Referee(course)

    referee.observe_truth((19.0, 0.0, 1.8), 0.0)
    assert referee.observe_truth((21.0, 0.0, 1.8), 1.0) is None
    assert referee.progress().next_gate_index == 0

    referee.observe_truth((9.0, 0.0, 1.8), 2.0)
    first = referee.observe_truth((11.0, 0.0, 1.8), 3.0)
    assert first is not None and first.gate_index == 0

    referee.observe_truth((19.0, 0.0, 1.8), 4.0)
    second = referee.observe_truth((21.0, 0.0, 1.8), 5.0)
    assert second is not None and second.gate_index == 1
    assert referee.pass_times == (2.5, 4.5)


def test_race_result_contract_for_incomplete_and_complete_course():
    referee = Referee(course_from_name("single"))
    referee.observe_truth((9.0, 0.0, 1.8), 2.0)

    assert referee.result().format() == (
        "[RACE] course=single gates_passed=0/1 lap_time=na status=INCOMPLETE pass_times=[]"
    )

    referee.observe_truth((11.0, 0.0, 1.8), 2.25)
    assert referee.result().format() == (
        "[RACE] course=single gates_passed=1/1 lap_time=2.125000 "
        "status=COMPLETE pass_times=[2.125000]"
    )


def test_referee_rejects_non_monotonic_simulation_time():
    referee = Referee(course_from_name("single"))
    referee.observe_truth((9.0, 0.0, 1.8), 2.0)

    with pytest.raises(ValueError, match="simulation time must be monotonic"):
        referee.observe_truth((11.0, 0.0, 1.8), 1.0)


def test_public_progress_is_immutable_and_contains_no_truth_geometry():
    referee = Referee(course_from_name("single"))
    progress = referee.progress()

    assert public_progress_field_names() == (
        "last_gate_passed",
        "next_gate_index",
        "gate_count",
        "gate_inner_size",
    )
    forbidden = {
        "world_pos",
        "world_vel",
        "center",
        "normal",
        "yaw",
        "gate_poses",
        "crossing_position",
    }
    assert forbidden.isdisjoint(field.name for field in fields(progress))
    with pytest.raises(FrozenInstanceError):
        progress.last_gate_passed = 99


def test_referee_truth_input_is_independent_of_any_guidance_source():
    referee = Referee(course_from_name("single"))

    # The scorer accepts only truth position/time and has no guidance-mode or
    # guidance-object input that could disable scoring in manual/vision modes.
    referee.observe_truth((9.0, 0.0, 1.8), sim_time=0.0)
    event = referee.observe_truth((11.0, 0.0, 1.8), sim_time=0.1)

    assert event is not None
    assert referee.progress().last_gate_passed == 0


def test_world_position_layout_extraction_is_explicit():
    assert world_position_from_transform((0.0, 0.0, 0.0, 1.0, 10.0, -2.0, 3.0)) == (
        10.0,
        -2.0,
        3.0,
    )
    with pytest.raises(ValueError, match="world_pos must use"):
        world_position_from_transform((10.0, -2.0, 3.0))
