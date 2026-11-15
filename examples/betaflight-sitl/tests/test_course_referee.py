"""Independent F host grading, including COMPLETE-but-unsafe failure cases."""

import math
from dataclasses import replace

import pytest
from controls import RcCommand
from course_referee import TruthCourseReferee
from exit_status import scenario_exit_code
from referee import RaceResult

COMPLETE = RaceResult("c1_straight", 3, 3, 40.0, (25.0, 32.0, 40.0))
INCOMPLETE = RaceResult("c1_straight", 2, 3, None, (25.0, 32.0))


def observe(grader, t, **changes):
    values = {
        "sim_time": t,
        "world_pos": (0, 0, 0, 1, 32, 0, 0),
        "world_vel": (0, 0, 0, 0, 0, 0),
        "command_sent": RcCommand(),
        "motors": (0, 0, 0, 0),
        "lockstep_response": True,
        "race_complete": True,
        "guidance_fault": False,
    }
    values.update(changes)
    grader.observe_truth(**values)


def flown():
    grader = TruthCourseReferee(max_sample_gap_s=1.01)
    observe(
        grader,
        40.0,
        world_pos=(0, 0, 0, 1, 30, 0, 1.8),
        command_sent=RcCommand(arm=1800, throttle=1130),
        motors=(0.13,) * 4,
    )
    return grader


def qualify(grader):
    for t in (41.0, 42.0, 42.1):
        observe(grader, t)
    return grader.result(COMPLETE)


def test_host_requires_physical_terminal_interval_and_actually_exchanged_disarm():
    grader = flown()
    assert grader.result(COMPLETE).exit_code == 1
    observe(grader, 41, command_sent=RcCommand(arm=1800, throttle=1000))
    assert grader.result(COMPLETE).safe_duration_s == 0
    observe(grader, 42)
    observe(grader, 43)
    result = grader.result(COMPLETE)
    assert result.passed and result.exit_code == 0
    assert result.safe_duration_s == pytest.approx(1)
    assert "status=PASS" in result.format() and "reason=qualified" in result.format()


def test_initial_disarmed_free_fall_is_not_scored_as_a_hard_landing():
    grader = TruthCourseReferee(max_sample_gap_s=1.01)
    observe(grader, 0.1, world_vel=(0, 0, 0, 0, 0, -1.0), race_complete=False)
    assert grader.result(INCOMPLETE).failure_reason is None


def test_incomplete_race_cannot_pass_even_with_all_finish_evidence():
    grader = flown()
    result = qualify(grader)
    assert result.passed
    assert not grader.result(INCOMPLETE).passed
    incomplete = replace(result, complete=False)
    assert incomplete.exit_code == 1
    assert "reason=incomplete" in incomplete.format()
    assert scenario_exit_code(course_exit_code=incomplete.exit_code) == 1


def test_completion_plus_finish_without_any_flight_motor_evidence_fails():
    grader = TruthCourseReferee(max_sample_gap_s=1.01)
    observe(grader, 40)
    observe(grader, 41)
    result = grader.result(COMPLETE)
    assert result.exit_code == 1
    assert "reason=no_flight_motor_response" in result.format()


@pytest.mark.parametrize(
    "changes",
    [
        {"world_pos": (0, 0, 0, 1, 32, 0, 0.16)},
        {"world_vel": (0, 0, 0, 0.21, 0, 0)},
        {"world_pos": (math.sin(0.1), 0, 0, math.cos(0.1), 32, 0, 0)},
        {"command_sent": RcCommand(throttle=1001)},
        {"command_sent": RcCommand(roll=1510)},
        {"command_sent": RcCommand(pitch=1510)},
        {"command_sent": RcCommand(yaw=1510)},
        {"command_sent": RcCommand(arm=1800)},
        {"motors": (0.021, 0, 0, 0)},
        {"race_complete": False},
    ],
)
def test_terminal_measurement_is_continuous_and_not_a_latched_earlier_pass(changes):
    grader = flown()
    assert qualify(grader).passed
    observe(grader, 43, **changes)
    result = grader.result(COMPLETE)
    assert not result.passed
    assert result.safe_duration_s == 0
    assert "reason=unsafe_finish" in result.format()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"world_pos": None}, "missing_truth_or_lockstep_or_motors"),
        ({"world_vel": None}, "missing_truth_or_lockstep_or_motors"),
        ({"world_pos": (0, 0, 0, 0, 32, 0, 0)}, "missing_truth_or_lockstep_or_motors"),
        ({"world_pos": (0, 0, 0, 1, 32, 0, float("nan"))}, "missing_truth_or_lockstep_or_motors"),
        ({"motors": (float("nan"),) * 4}, "missing_truth_or_lockstep_or_motors"),
        ({"motors": (1.1,) * 4}, "missing_truth_or_lockstep_or_motors"),
        ({"lockstep_response": False}, "missing_truth_or_lockstep_or_motors"),
        ({"world_vel": (0, 0, 0, 3.01, 0, 0)}, "unsafe_flight"),
        ({"world_vel": (0, 0, 0, 0, 0, -0.66)}, "unsafe_flight"),
        ({"world_pos": (0, 0, 0, 1, 32, 0, 4.1)}, "unsafe_flight"),
        ({"world_pos": (0, 0, 0, 1, 32, 0, -0.02)}, "unsafe_flight"),
        ({"world_pos": (math.sin(0.4), 0, 0, math.cos(0.4), 32, 0, 1)}, "unsafe_flight"),
    ],
)
def test_bad_evidence_or_unsafe_flight_is_sticky_even_after_a_later_safe_landing(changes, reason):
    grader = flown()
    observe(grader, 40.5, **changes)
    result = qualify(grader)
    assert not result.passed
    assert result.failure_reason == reason
    assert scenario_exit_code(course_exit_code=result.exit_code) == 1


def test_guidance_fault_is_a_veto_not_a_guidance_completion_grade():
    grader = flown()
    observe(grader, 40.5, guidance_fault=True)
    result = qualify(grader)
    assert not result.passed and result.fault
    assert "reason=guidance_fault" in result.format()


@pytest.mark.parametrize("time", (float("nan"), -1, 60.001, 39.9, 42))
def test_invalid_or_skipped_simulation_time_fails(time):
    grader = flown()
    observe(grader, time)
    result = grader.result(COMPLETE)
    assert not result.passed and result.failure_reason is not None


def test_missing_initial_boot_response_not_fatal_but_post_boot_response_is_required():
    grader = TruthCourseReferee(max_sample_gap_s=10)
    observe(grader, 0, lockstep_response=False)
    assert grader.result(INCOMPLETE).failure_reason is None
    observe(grader, 5.0, lockstep_response=False)
    assert grader.result(INCOMPLETE).failure_reason == "missing_truth_or_lockstep_or_motors"


def test_finish_must_be_observed_within_sixty_simulated_seconds():
    grader = flown()
    for t in range(41, 61):
        observe(grader, float(t))
    assert grader.result(COMPLETE).passed
    observe(grader, 60.000125)
    assert not grader.result(COMPLETE).passed


@pytest.mark.parametrize("gap", (0, -1, float("nan"), float("inf")))
def test_sample_gap_must_be_positive_finite(gap):
    with pytest.raises(ValueError):
        TruthCourseReferee(max_sample_gap_s=gap)
