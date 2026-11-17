"""Truth outer-guidance cadence on integer simulation ticks (not camera timing)."""

import os
from collections.abc import Mapping
from fractions import Fraction


def guidance_rate_from_env(
    guidance: str,
    physics_hz: float,
    environ: Mapping[str, str] = os.environ,
    *,
    default_hz: int | None = None,
) -> Fraction | None:
    """An explicit truth-only rate overrides the caller's scenario default."""
    value = environ.get("RACE_GUIDANCE_HZ")
    if value is None:
        return None if default_hz is None else Fraction(default_hz)
    if guidance != "truth":
        raise ValueError("RACE_GUIDANCE_HZ requires RACE_GUIDANCE=truth")
    try:
        rate = Fraction(value.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("RACE_GUIDANCE_HZ must be finite, positive and <= physics rate") from exc
    if not 0 < rate <= Fraction(str(physics_hz)):
        raise ValueError("RACE_GUIDANCE_HZ must be finite, positive and <= physics rate")
    return rate


class GuidanceCadence:
    """Ceil ideal deadlines to physics ticks, with <1 tick delay and no drift.

    Tick zero is due. A skipped deadline produces just one current-state update;
    the next deadline stays on the original grid, strictly after the current tick.
    Counters count scheduled and *additional*, off-grid forced fault updates;
    a fault on a scheduled tick counts only as scheduled.
    """

    def __init__(self, physics_hz: float, guidance_hz: Fraction | None = None):
        physics = Fraction(str(physics_hz))
        rate = physics if guidance_hz is None else guidance_hz
        if not 0 < rate <= physics:
            raise ValueError("guidance rate must be positive and <= physics rate")
        self.hz = rate
        self.full_rate = rate == physics
        period = physics / rate
        self._numerator = period.numerator
        self._denominator = period.denominator
        self._next_tick = 0
        self.scheduled_updates = 0
        self.forced_updates = 0

    def due(self, tick: int) -> bool:
        if tick < self._next_tick:
            return False
        ordinal = tick * self._denominator // self._numerator + 1
        self._next_tick = (ordinal * self._numerator + self._denominator - 1) // self._denominator
        self.scheduled_updates += 1
        return True
