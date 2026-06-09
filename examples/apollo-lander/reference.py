"""Shared Apollo 11 descent reference profile (standard library only).

The raw LEM telemetry (``data/apollo11_lem_raw.csv``) reports IMU stable-member
gimbal angles and a noisy landing-radar slant range. This module turns that into
a clean, monotonically descending altitude/descent-rate reference plus a smoothed
attitude trend, so both the simulation truth display and the external guidance
controller can share one source of truth without depending on numpy/jax.

Range is treated as an altitude proxy (a documented teaching approximation; the
true geometry needs the mission state we do not have). Attitude is a smoothed
reconstruction from the inner gimbal angle and is likewise approximate.
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
DERIVED_PATH = DATA_DIR / "apollo11_descent.csv"
RAW_PATH = DATA_DIR / "apollo11_lem_raw.csv"

REFERENCE_DT_S = 1.0
ALTITUDE_SMOOTH_S = 5.0
RATE_SMOOTH_S = 25.0
PITCH_SMOOTH_S = 9.0


def _num(value: str) -> float | None:
    value = value.strip().replace(" ", "")
    if value in ("", ".", "-", "-."):
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class DescentData:
    range_time_s: list[float]
    range_m: list[float]
    gimbal_time_s: list[float]
    inner_deg: list[float]
    middle_deg: list[float]
    outer_deg: list[float]


def load_descent(path: Path = DERIVED_PATH) -> DescentData:
    range_time_s: list[float] = []
    range_m: list[float] = []
    gimbal_time_s: list[float] = []
    inner_deg: list[float] = []
    middle_deg: list[float] = []
    outer_deg: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            gimbal_time_s.append(t)
            inner_deg.append(float(row["inner_deg"]))
            middle_deg.append(float(row["middle_deg"]))
            outer_deg.append(float(row["outer_deg"]))
            value = _num(row.get("range_m", ""))
            if value is not None:
                range_time_s.append(t)
                range_m.append(value)
    return DescentData(range_time_s, range_m, gimbal_time_s, inner_deg, middle_deg, outer_deg)


def interp(t: float, xs: list[float], ys: list[float]) -> float:
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, t)
    lo, hi = i - 1, i
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    frac = (t - xs[lo]) / span
    return ys[lo] + (ys[hi] - ys[lo]) * frac


def _median_filter(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    half = window // 2
    n = len(values)
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sorted(values[lo:hi])[(hi - lo) // 2])
    return out


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    half = window // 2
    out: list[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = values[lo:hi]
        out.append(sum(window_vals) / len(window_vals))
    return out


@dataclass
class Reference:
    time_s: list[float]
    altitude_m: list[float]
    descent_rate_mps: list[float]
    pitch_deg: list[float]
    t_end: float

    def altitude(self, t: float) -> float:
        return interp(t, self.time_s, self.altitude_m)

    def descent_rate(self, t: float) -> float:
        return interp(t, self.time_s, self.descent_rate_mps)

    def pitch(self, t: float) -> float:
        return interp(t, self.time_s, self.pitch_deg)


def build_reference(data: DescentData | None = None) -> Reference:
    if data is None:
        data = load_descent()
    t_end = data.gimbal_time_s[-1]
    steps = int(round(t_end / REFERENCE_DT_S))
    grid = [i * REFERENCE_DT_S for i in range(steps + 1)]

    raw_alt = [interp(t, data.range_time_s, data.range_m) for t in grid]
    # Median filter rejects isolated radar-range spikes before averaging.
    despiked = _median_filter(raw_alt, 5)
    smoothed = _moving_average(despiked, max(1, round(ALTITUDE_SMOOTH_S / REFERENCE_DT_S)))

    # Enforce a physically sensible monotonic descent (radar range is noisy).
    monotonic: list[float] = []
    running = smoothed[0]
    for value in smoothed:
        running = min(running, value)
        monotonic.append(max(running, 0.0))

    # The landing radar still reads the antenna height above the footpads at
    # touchdown, so the profile bottoms out a few meters high. Shift it down by
    # its terminal value so the truth vehicle reaches altitude zero exactly at
    # the recorded touchdown time.
    terminal_offset = monotonic[-1]
    monotonic = [max(value - terminal_offset, 0.0) for value in monotonic]

    rate: list[float] = []
    for i in range(len(grid)):
        lo = max(0, i - 1)
        hi = min(len(grid) - 1, i + 1)
        dt = grid[hi] - grid[lo]
        rate.append((monotonic[hi] - monotonic[lo]) / dt if dt > 0 else 0.0)
    rate = _moving_average(rate, max(1, round(RATE_SMOOTH_S / REFERENCE_DT_S)))

    # Approximate pitch-from-vertical trend from the dominant inner gimbal angle.
    inner = [interp(t, data.gimbal_time_s, data.inner_deg) for t in grid]
    inner = _moving_average(inner, max(1, round(PITCH_SMOOTH_S / REFERENCE_DT_S)))
    pitch = [-value for value in inner]

    return Reference(grid, monotonic, rate, pitch, t_end)


def sanity_check(tolerance_m: float = 1e-3) -> dict[str, float]:
    """Confirm the derived measurements agree with the raw GitHub source."""

    derived = load_descent(DERIVED_PATH)
    raw_rows = 0
    raw_range = 0
    start = None
    max_range_err = 0.0
    derived_range = dict(zip(derived.range_time_s, derived.range_m))
    with RAW_PATH.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        from datetime import datetime, timezone

        for row in reader:
            if not row or len(row) < 8:
                continue
            try:
                dt = datetime.strptime(row[0].strip(), "%y%m%d%H%M%S.%f").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            if start is None:
                start = dt
            t = round((dt - start).total_seconds(), 6)
            raw_rows += 1
            cell = row[7].strip().replace(",", "")
            if cell:
                raw_range += 1
                expected = float(cell) * 0.3048
                nearest = min(derived_range, key=lambda k: abs(k - t), default=None)
                if nearest is not None and abs(nearest - t) < 1e-6:
                    max_range_err = max(max_range_err, abs(derived_range[nearest] - expected))
    return {
        "raw_rows": raw_rows,
        "raw_range_rows": raw_range,
        "derived_range_rows": len(derived.range_m),
        "max_range_error_m": max_range_err,
        "tolerance_m": tolerance_m,
        "ok": max_range_err <= tolerance_m,
    }


if __name__ == "__main__":
    ref = build_reference()
    print(f"reference samples: {len(ref.time_s)} over {ref.t_end:.1f} s")
    for t in range(0, int(ref.t_end) + 1, 30):
        print(
            f"  t={t:3d}s  alt={ref.altitude(t):8.1f} m"
            f"  rate={ref.descent_rate(t):7.2f} m/s  pitch={ref.pitch(t):7.2f} deg"
        )
    print("sanity:", sanity_check())
