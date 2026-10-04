"""Tests for the default scripted SITL scenario result."""

import pytest

from baseline import evaluate_c0


def test_c0_passes_when_all_acceptance_criteria_are_met() -> None:
    result = evaluate_c0(
        lockstep_steps=120_000,
        max_motor=0.4,
        initial_altitude=0.1,
        max_altitude=0.2,
    )

    assert result.motor_response is True
    assert result.takeoff_delta_m == pytest.approx(0.1)
    assert result.passed is True


@pytest.mark.parametrize(
    ("lockstep_steps", "max_motor", "max_altitude"),
    [
        pytest.param(0, 0.4, 1.0, id="no-lockstep-response"),
        pytest.param(120_000, 0.06, 1.0, id="no-meaningful-motor-response"),
        pytest.param(120_000, 0.4, 0.199, id="insufficient-takeoff"),
    ],
)
def test_c0_fails_when_an_acceptance_criterion_is_unmet(
    lockstep_steps: int, max_motor: float, max_altitude: float
) -> None:
    result = evaluate_c0(
        lockstep_steps=lockstep_steps,
        max_motor=max_motor,
        initial_altitude=0.1,
        max_altitude=max_altitude,
    )

    assert result.passed is False


def test_c0_result_format_is_stable_and_machine_readable() -> None:
    result = evaluate_c0(
        lockstep_steps=119_995,
        max_motor=0.5736,
        initial_altitude=0.1,
        max_altitude=56.94,
    )

    assert result.format() == (
        "[C0] lockstep_steps=119995 motor_response=true max_motor=0.574 "
        "takeoff_delta_m=56.840 status=PASS"
    )
