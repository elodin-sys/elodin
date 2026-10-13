"""Pure exit-code regressions for independent B/C/D/E and baseline outcomes."""

import pytest

from exit_status import scenario_exit_code


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
        ({"axis_passed": True, "camera_accepted": False}, 1),
    ],
)
def test_scenario_exit_code_preserves_each_contract_and_combined_failures(
    results: dict, expected: int
) -> None:
    assert scenario_exit_code(**results) == expected
