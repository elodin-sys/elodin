"""Pure tests for the independent Package E standoff hold grader."""

import math

import pytest
from course import Gate, course_from_name
from hold_referee import HoldEligibility, StandoffHoldReferee

ELIGIBLE = HoldEligibility(True, True, True, True, True)
TARGET = (6.0, 0.0, 1.8)


def _observe(grader, times, positions, eligibility=ELIGIBLE):
    for time, position in zip(times, positions, strict=True):
        grader.observe_truth(position, time, eligibility)


def _constant_series(duration, offset=(0.0, 0.0, 0.0), step=0.1, start=0.0):
    count = round(duration / step)
    times = [start + index * step for index in range(count + 1)]
    position = tuple(TARGET[index] + offset[index] for index in range(3))
    return times, [position] * len(times)


def _single_gate():
    return course_from_name("single").gates[0]


def test_target_is_independently_derived_and_exact_ten_second_window_passes() -> None:
    grader = StandoffHoldReferee(_single_gate())
    assert grader.target_enu == TARGET
    times, positions = _constant_series(10.0)
    _observe(grader, times, positions)

    result = grader.result()
    assert result.passed
    assert result.status == "PASS"
    assert result.window.start_time == pytest.approx(0.0)
    assert result.window.end_time == pytest.approx(10.0)
    assert result.window.rms_error_m == pytest.approx(0.0)
    assert result.eligible_samples == 101
    assert result.armed_seen and result.angle_seen
    assert result.lockstep_seen and result.motor_seen


def test_duration_just_below_ten_seconds_does_not_qualify() -> None:
    grader = StandoffHoldReferee(_single_gate())
    times, positions = _constant_series(9.9)
    _observe(grader, times, positions)

    assert grader.result().window is None
    assert grader.result().passed is False


def test_three_dimensional_rms_accepts_exact_threshold_and_rejects_above() -> None:
    exact = StandoffHoldReferee(_single_gate())
    times, positions = _constant_series(10.0, offset=(0.1, 0.2, 0.2), step=0.1)
    _observe(exact, times, positions)
    assert exact.result().window.rms_error_m == pytest.approx(0.3, abs=1e-12)
    assert exact.result().passed

    over = StandoffHoldReferee(_single_gate())
    times, positions = _constant_series(10.0, offset=(0.30001, 0.0, 0.0), step=0.1)
    _observe(over, times, positions)
    assert over.result().window.rms_error_m == pytest.approx(0.30001, abs=1e-9)
    assert not over.result().passed


def test_time_weighted_trapezoidal_integral_interpolates_exact_start_endpoint() -> None:
    grader = StandoffHoldReferee(_single_gate())
    # A sparse linear 3D segment brackets the 10 s window start at t=0.5.
    points = [TARGET, (TARGET[0] + 1.05, TARGET[1] - 0.525, TARGET[2] + 0.21)]
    _observe(grader, [0.0, 10.5], points)

    # The trapezoid interpolates squared-error samples at the exact start; with
    # q(0)=0 and q(10.5)=|e_end|², q(.5)=q_end*(.5/10.5).
    end_error = (1.05, -0.525, 0.21)
    q_end = sum(value * value for value in end_error)
    fraction = 0.5 / 10.5
    q_start = q_end * fraction
    expected = math.sqrt((q_start + q_end) / 2.0)
    assert grader.result().window.start_time == pytest.approx(0.5)
    assert grader.result().window.end_time == pytest.approx(10.5)
    assert grader.result().window.rms_error_m == pytest.approx(expected)


def test_later_good_rolling_window_qualifies_after_an_early_bad_window() -> None:
    grader = StandoffHoldReferee(_single_gate())
    times = [index * 0.1 for index in range(201)]
    positions = [
        (TARGET[0] + (0.5 if time <= 10.0 else 0.0), TARGET[1], TARGET[2]) for time in times
    ]
    _observe(grader, times, positions)

    result = grader.result()
    assert result.passed
    assert result.window.start_time > 0.0
    assert result.window.end_time > 10.0
    assert result.window.rms_error_m <= 0.30


def test_eligibility_gap_resets_contiguous_window_and_control_loss_is_evidence() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=2.0)
    for index in range(21):
        grader.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    gap = HoldEligibility(True, False, True, True, True)
    grader.observe_truth(TARGET, 2.1, gap)
    assert grader.result().window is not None  # The earlier valid window is retained.
    assert grader.result().passed

    fresh = StandoffHoldReferee(_single_gate(), duration_s=2.0)
    for index in range(10):
        fresh.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    fresh.observe_truth(TARGET, 1.0, HoldEligibility(False, True, True, True, True))
    for index in range(11, 32):
        fresh.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    assert fresh.result().passed
    assert fresh.result().window.start_time == pytest.approx(1.1)


@pytest.mark.parametrize(
    "eligibility",
    [
        HoldEligibility(False, True, True, True, True),
        HoldEligibility(True, False, True, True, True),
        HoldEligibility(True, True, False, True, True),
        HoldEligibility(True, True, True, False, True),
        HoldEligibility(True, True, True, True, False),
    ],
)
def test_each_hold_control_criterion_is_required(eligibility) -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    for index in range(11):
        grader.observe_truth(TARGET, index * 0.1, ELIGIBLE if index != 5 else eligibility)

    assert not grader.result().passed


def test_nonfinite_nonmonotonic_and_explicit_missing_truth_samples_reset_candidate() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    for index in range(6):
        grader.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    grader.observe_truth((math.nan, 0.0, 1.8), 0.6, ELIGIBLE)
    for index in range(1, 12):
        grader.observe_truth(TARGET, 0.6 + index * 0.1, ELIGIBLE)
    assert grader.result().passed

    nonmonotonic = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    for index in range(6):
        nonmonotonic.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    nonmonotonic.observe_truth(TARGET, 0.2, ELIGIBLE)
    for index in range(1, 12):
        nonmonotonic.observe_truth(TARGET, 0.2 + index * 0.1, ELIGIBLE)
    assert nonmonotonic.result().passed

    missing = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    for index in range(6):
        missing.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    missing.observe_truth(None, 0.6, ELIGIBLE)
    assert not missing.result().passed


def test_nonmonotonic_or_nonfinite_times_are_handled_without_raising() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    grader.observe_truth(TARGET, 0.0, ELIGIBLE)
    grader.observe_truth(TARGET, float("nan"), ELIGIBLE)
    grader.observe_truth(TARGET, 0.5, ELIGIBLE)
    grader.observe_truth(TARGET, 0.4, ELIGIBLE)
    assert not grader.result().passed
    assert grader.result().reason == "nonmonotonic_time"


def test_sample_gap_resets_candidate_and_storage_capacity_is_bounded() -> None:
    gap = StandoffHoldReferee(_single_gate(), duration_s=1.0, max_sample_gap_s=0.2)
    gap.observe_truth(TARGET, 0.0, ELIGIBLE)
    gap.observe_truth(TARGET, 0.3, ELIGIBLE)
    assert gap.result().window is None
    assert gap.result().reason == "sample_gap"

    bounded = StandoffHoldReferee(_single_gate(), duration_s=1.0, max_samples=3)
    for index in range(20):
        bounded.observe_truth(TARGET, index * 0.1, ELIGIBLE)
    assert len(bounded._samples) <= 3
    assert not bounded.result().passed


def test_gate_crossing_permanently_fails_even_after_qualified_window() -> None:
    grader = StandoffHoldReferee(_single_gate())
    times, positions = _constant_series(10.0)
    _observe(grader, times, positions)
    assert grader.result().passed

    grader.observe_truth((10.0, 0.0, 1.8), 10.1, ELIGIBLE)
    result = grader.result()
    assert grader.gate_crossed
    assert not result.passed
    assert result.status == "FAIL"
    assert result.reason == "gate_crossing"
    assert result.window is not None


def test_backward_or_parallel_plane_motion_does_not_fail_hold() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    grader.observe_truth((11.0, 0.0, 1.8), 0.0, ELIGIBLE)
    grader.observe_truth((10.0, 0.0, 1.8), 0.1, ELIGIBLE)
    grader.observe_truth((9.0, 0.0, 1.8), 0.2, ELIGIBLE)
    grader.observe_truth((9.0, 2.0, 1.8), 0.3, ELIGIBLE)

    assert not grader.gate_crossed
    assert grader.result().reason != "gate_crossing"
    # A subsequent genuine forward passage still fails permanently.
    grader.observe_truth((10.0, 2.0, 1.8), 0.4, ELIGIBLE)
    assert grader.gate_crossed
    assert grader.result().reason == "gate_crossing"


def test_forward_crossing_still_fails_across_missing_truth_sample() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    grader.observe_truth((9.0, 2.0, 1.8), 0.0, ELIGIBLE)
    grader.observe_truth(None, 0.1, ELIGIBLE)
    grader.observe_truth((11.0, 2.0, 1.8), 0.2, ELIGIBLE)
    assert grader.gate_crossed
    assert grader.result().status == "FAIL"


def test_guidance_fault_permanently_fails_even_after_qualified_window() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    grader.observe_truth(TARGET, 0.0, ELIGIBLE)
    grader.observe_truth(TARGET, 1.0, ELIGIBLE)
    assert grader.result().passed
    grader.observe_truth(
        TARGET, 1.1, HoldEligibility(False, True, True, True, True), phase_fault=True
    )
    grader.observe_truth(TARGET, 2.1, ELIGIBLE)
    assert grader.result().status == "FAIL"
    assert grader.result().reason == "guidance_fault"


def test_explicit_referee_crossing_evidence_is_permanent() -> None:
    grader = StandoffHoldReferee(_single_gate(), duration_s=1.0)
    grader.observe_truth(TARGET, 0.0, ELIGIBLE)
    grader.observe_truth(TARGET, 1.0, ELIGIBLE, gate_crossed=True)
    grader.observe_truth(TARGET, 2.0, ELIGIBLE)
    assert grader.gate_crossed
    assert not grader.result().passed


def test_yawed_gate_target_and_result_line_contract() -> None:
    gate = Gate(0, (4.0, -3.0, 2.0), math.pi / 2, 2.5)
    grader = StandoffHoldReferee(gate, duration_s=1.0)
    assert grader.target_enu == pytest.approx((4.0, -7.0, 2.0))
    grader.observe_truth(grader.target_enu, 0.0, ELIGIBLE)
    grader.observe_truth(grader.target_enu, 1.0, ELIGIBLE)

    result = grader.result()
    assert result.format() == (
        "[E-HOLD] status=PASS target=(4.000000,-7.000000,2.000000) "
        "window=[0.000000,1.000000] duration=1.000000 rms=0.000000 threshold=0.300000 "
        "control=armed:true,angle:true,lockstep:true,motor:true eligible_samples=2 "
        "crossing=false fault=false reason=qualified"
    )
    assert result.exit_code == 0


def test_missing_qualification_is_explicit_failure_and_nonzero() -> None:
    grader = StandoffHoldReferee(_single_gate())
    result = grader.result()

    assert result.status == "FAIL"
    assert result.exit_code == 1
    assert "window=na" in result.format()
    assert "rms=na" in result.format()
    assert "control=armed:false,angle:false,lockstep:false,motor:false" in result.format()
