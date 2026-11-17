"""Pure integer scheduler/startup tests; no SITL processes or wall clocks."""

from fractions import Fraction
from itertools import pairwise

import pytest
from guidance_cadence import GuidanceCadence, guidance_rate_from_env


@pytest.mark.parametrize("rate,expected", [(60, 2700), (100, 4500), (120, 5400), (8000, 360000)])
def test_45_seconds_counts_and_bounded_phase(rate, expected):
    cadence = GuidanceCadence(8000, Fraction(rate))
    ticks = [tick for tick in range(360000) if cadence.due(tick)]
    assert len(ticks) == expected == cadence.scheduled_updates
    assert ticks[0] == 0
    for ordinal, tick in enumerate(ticks):
        # Integer proof of ceil quantization: 0 <= actual-ideal < 1 tick.
        assert 0 <= tick * rate - ordinal * 8000 < rate
    if rate != 8000:
        assert {b - a for a, b in pairwise(ticks)} == {60: {133, 134}, 100: {80}, 120: {66, 67}}[rate]
    assert cadence.due(360000)  # Closed endpoint adds one, outside 45s callbacks.


def test_decimal_nondivisor_is_exact_over_long_run():
    rate = guidance_rate_from_env("truth", 8000, {"RACE_GUIDANCE_HZ": "59.94"})
    assert rate == Fraction(2997, 50)
    cadence = GuidanceCadence(8000, rate)
    ticks = [tick for tick in range(360000) if cadence.due(tick)]
    assert len(ticks) == 2698
    assert all(0 <= tick * rate - i * 8000 < rate for i, tick in enumerate(ticks))


def test_missed_ticks_update_current_state_once_and_keep_original_phase():
    cadence = GuidanceCadence(8000, Fraction(60))
    assert cadence.due(0)
    assert not cadence.due(1)
    assert cadence.due(1000)  # Seven scheduled deadlines skipped, not replayed.
    assert not cadence.due(1000)
    assert not cadence.due(1066)
    assert cadence.due(1067)
    assert cadence.scheduled_updates == 3


@pytest.mark.parametrize("rate", [None, Fraction(8000)])
def test_unset_and_equal_physics_rate_are_full_rate(rate):
    cadence = GuidanceCadence(8000, rate)
    assert cadence.full_rate
    assert all(cadence.due(tick) for tick in range(1000))
    assert cadence.scheduled_updates == 1000


@pytest.mark.parametrize(
    "value", ["", " ", "nan", "inf", "-inf", "0", "-1", "8000.1", "wat", "1/0"]
)
def test_invalid_rate_fails_startup(value):
    with pytest.raises(ValueError, match="RACE_GUIDANCE_HZ"):
        guidance_rate_from_env("truth", 8000, {"RACE_GUIDANCE_HZ": value}, default_hz=100)


@pytest.mark.parametrize("mode", ["scripted", "manual"])
def test_rate_is_rejected_outside_truth(mode):
    assert guidance_rate_from_env(mode, 8000, {}) is None
    with pytest.raises(ValueError, match="requires RACE_GUIDANCE=truth"):
        guidance_rate_from_env(mode, 8000, {"RACE_GUIDANCE_HZ": "60"})


@pytest.mark.parametrize("value", ["60", "120", "8000", " 120 ", "1e2", "0.01"])
def test_positive_finite_rate_values(value):
    assert (
        guidance_rate_from_env("truth", 8000, {"RACE_GUIDANCE_HZ": value}, default_hz=100)
        == Fraction(value)
    )
