"""Pure exit-code regressions for independent B/C/D/E and baseline outcomes."""

import pytest

from exit_status import CallbackExecution, scenario_exit_code


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ({}, 0),
        ({"c0_passed": True, "camera_accepted": True}, 0),
        ({"hold_exit_code": 0, "camera_accepted": True}, 0),
        ({"hold_exit_code": 0, "camera_accepted": False}, 1),
        ({"hold_exit_code": 1, "camera_accepted": True}, 1),
        ({"hold_exit_code": 1, "camera_accepted": False}, 1),
        ({"hold_exit_code": 2, "camera_accepted": True}, 2),
        ({"c0_passed": False, "camera_accepted": True}, 1),
        ({"referee_exit_code": 1, "camera_accepted": True}, 1),
        ({"axis_passed": False, "camera_accepted": True}, 1),
        ({"course_exit_code": 0}, 0),
        ({"execution_exit_code": 1, "course_exit_code": 0}, 1),
        ({"execution_exit_code": 0, "course_exit_code": 0}, 0),
        ({"course_exit_code": 1}, 1),
        ({"course_exit_code": 1, "camera_accepted": True}, 1),
        ({"course_exit_code": 0, "camera_accepted": False}, 1),
        ({"hold_exit_code": 0}, 0),  # E deliberately does not require a complete race.
        ({"c0_passed": True}, 0),  # Scripted course incompletion is observational.
        ({"axis_passed": True, "camera_accepted": False}, 1),
        ({"recording_failed": True}, 1),
        ({"recording_failed": False}, 0),
        ({"recording_failed": None}, 0),
        ({"c0_passed": False, "recording_failed": True}, 1),
        ({"c0_passed": True, "recording_failed": True}, 1),
        ({"course_exit_code": 2, "recording_failed": True}, 2),
    ],
)
def test_scenario_exit_code_preserves_each_contract_and_combined_failures(
    results: dict, expected: int
) -> None:
    assert scenario_exit_code(**results) == expected


def test_callback_execution_requires_full_normal_sequence_and_final_success():
    status = CallbackExecution(3)
    for tick in range(3):
        assert status.exit_code == 1
        status.begin(tick)
        assert status.exit_code == 1
        status.completed(tick)
    assert status.normal_completion
    assert status.failure_reason is None
    assert status.exit_code == 0


@pytest.mark.parametrize("error", (RuntimeError("fatal"), KeyboardInterrupt(), SystemExit(1)))
def test_callback_error_is_sticky_even_after_more_callbacks_or_prior_normal_end(error):
    status = CallbackExecution(2)
    for tick in range(2):
        status.begin(tick)
        status.completed(tick)
    assert status.exit_code == 0
    status.record_error(error)
    assert not status.normal_completion
    assert status.failure_reason == f"callback_error_{type(error).__name__}"
    assert status.exit_code == 1
    status.completed(1)
    assert status.exit_code == 1


@pytest.mark.parametrize("ticks", ((1, 2), (0, 2), (0, 0, 1, 2), (0, 1)))
def test_skipped_duplicate_or_truncated_callbacks_cannot_pass(ticks):
    status = CallbackExecution(3)
    for tick in ticks:
        status.begin(tick)
        status.completed(tick)
    assert not status.normal_completion
    assert status.exit_code == 1


@pytest.mark.parametrize("expected", (0, -1, True, 1.5))
def test_callback_expected_tick_count_is_a_positive_integer(expected):
    with pytest.raises(ValueError):
        CallbackExecution(expected)
