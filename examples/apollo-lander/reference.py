"""Shared Apollo 11 descent reference profile (standard library only).

Two vendored datasets drive the reference:

- ``data/apollo11_descent.csv`` — derived SI measurements from the public LEM
  telemetry (PGNS stable-member gimbal angles plus landing-radar slant range,
  from jumpjack/Apollo11LEMdata ``data.csv``).
- ``data/apollo11_altitude_raw.csv`` — the verbatim digitized true-altitude
  profile of the powered descent (jumpjack/Apollo11LEMdata
  ``004-altitude-dot.csv``, digitized from the mission-report descent chart).
  The feet columns are authoritative; the source's ``Altitudem`` column has a
  unit bug (x0.3405 instead of x0.3048), so feet are converted here.

The module builds the cleaned altitude / descent-rate / pitch reference, keeps
the radar slant range as a secondary display series, and reconstructs the
braking-phase horizontal-speed / downrange profile by integrating the vehicle
dynamics along the recorded attitude, calibrated to documented mission events
(throttle-down, high gate, transcript velocity callouts, touchdown). Everything
is standard library so the simulation truth display and the external guidance
controller share one dependency-free source of truth.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
DERIVED_PATH = DATA_DIR / "apollo11_descent.csv"
RAW_PATH = DATA_DIR / "apollo11_lem_raw.csv"
ALTITUDE_PATH = DATA_DIR / "apollo11_altitude_raw.csv"

FT_TO_M = 0.3048

REFERENCE_DT_S = 1.0
ALTITUDE_SMOOTH_S = 7.0
RATE_SMOOTH_S = 25.0
PITCH_SMOOTH_S = 9.0
RANGE_SMOOTH_S = 5.0
# The first chart samples show the pre-update PGNS altitude (43,753 ft) before
# the landing-radar delta-H correction stepped the state vector down ~1.6 km.
# Skip them and back-extrapolate the radar-corrected track to t = 0, which
# matches Armstrong's debrief ("39,000 or 40,000 feet at the time we had
# radar lockup").
ALTITUDE_SKIP_HEAD_S = 6.0
ALTITUDE_TREND_WINDOW_S = 30.0

# --- Horizontal-profile reconstruction model (mirrors sim.py constants) ---
DPS_MAX_THRUST_N = 45_040.0
DPS_FTP_THROTTLE = 0.925
DPS_ISP_S = 311.0
G0 = 9.80665
G_MOON = 1.622
R_MOON_M = 1_737_400.0
MIN_VERTICAL_ACCEL_MPS2 = 0.05
# Vehicle mass at the start of the telemetry window: descent-stack dry mass
# plus the DPS propellant remaining at landing-radar lock plus RCS propellant.
WINDOW_START_MASS_KG = 6_853.0 + 3_950.0 + 240.0

# Documented mission events, in seconds after the window start
# (landing-radar lock-on, GET 102:37:53 = 1969-07-20T20:09:53Z).
THROTTLE_DOWN_S = 98.0  # GET 102:39:31 — end of the fixed-throttle (FTP) burn
HIGH_GATE_S = 219.0  # GET 102:41:32 — P63 -> P64 pitchover at ~7,400 ft
P66_START_S = 353.0  # ~GET 102:43:46 — manual landing phase (P66)

# Digitized chart ends ~4.6 m above the surface (entity / CG altitude). Extend
# the profile to the footpad contact height so guidance and the truth ghost
# continue the final let-down instead of hovering above the regolith.
# Must match sim.FOOTPAD_HEIGHT_M (entity z when pads meet z = 0).
FOOTPAD_CONTACT_ALT_M = 2.40
TERMINAL_CONTACT_RATE_MPS = 0.5  # matches controller MIN_DESCENT_RATE_MPS

# Horizontal-velocity anchors used to calibrate the reconstruction (m/s).
HIGH_GATE_HSPEED_MPS = 152.0  # ~500 ft/s nominal P63 exit velocity
ANCHOR_333S_MPS = 17.7  # GET 102:43:26 transcript: "58 (ft/s) forward"
ANCHOR_353S_MPS = 14.3  # GET 102:43:46 transcript: "47 (ft/s) forward"


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


def _parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp.strip(), "%y%m%d%H%M%S.%f").replace(tzinfo=timezone.utc)


def load_altitude(path: Path = ALTITUDE_PATH) -> tuple[list[float], list[float]]:
    """Load the digitized true-altitude profile as (time_s, altitude_m).

    Times are relative to the first sample (which coincides with the start of
    the gimbal/range telemetry window). The digitizer output contains
    out-of-order and duplicate timestamps; samples are sorted and duplicates
    averaged. Feet are converted at 0.3048 (the source's metric column has a
    documented unit bug and is ignored).
    """

    samples: dict[float, list[float]] = {}
    start: datetime | None = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                stamp = _parse_stamp(row["Raw timestamp"])
            except (KeyError, ValueError):
                continue
            if start is None:
                start = stamp
            t = round((stamp - start).total_seconds(), 3)
            feet = _num(row.get("Interpolated", "")) or _num(row.get("Rawaltitudeft", ""))
            if feet is None:
                continue
            samples.setdefault(t, []).append(feet * FT_TO_M)
    times = [t for t in sorted(samples) if t >= ALTITUDE_SKIP_HEAD_S]
    values = [sum(samples[t]) / len(samples[t]) for t in times]
    # Back-extrapolate the radar-corrected track over the skipped head.
    trend = [(t, v) for t, v in zip(times, values) if t <= ALTITUDE_TREND_WINDOW_S]
    if len(trend) >= 2 and times[0] > 0.0:
        slope = (trend[-1][1] - trend[0][1]) / (trend[-1][0] - trend[0][0])
        times.insert(0, 0.0)
        values.insert(0, values[0] - slope * trend[0][0])
    return times, values


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
    slant_range_m: list[float]
    horizontal_speed_mps: list[float]
    downrange_m: list[float]
    t_end: float

    def altitude(self, t: float) -> float:
        return interp(t, self.time_s, self.altitude_m)

    def descent_rate(self, t: float) -> float:
        return interp(t, self.time_s, self.descent_rate_mps)

    def pitch(self, t: float) -> float:
        return interp(t, self.time_s, self.pitch_deg)

    def slant_range(self, t: float) -> float:
        return interp(t, self.time_s, self.slant_range_m)

    def horizontal_speed(self, t: float) -> float:
        return interp(t, self.time_s, self.horizontal_speed_mps)

    def downrange(self, t: float) -> float:
        return interp(t, self.time_s, self.downrange_m)


def _reconstruction_throttle(t: float) -> float:
    """Documented DPS throttle history over the telemetry window.

    FTP until throttle-down, the LUMINARY post-throttle-down creep through the
    end of P63, then decreasing commands through P64 into the near-hover P66.
    The P64/P66 segment shapes are calibrated against the velocity anchors in
    ``_reconstruct_horizontal``.
    """

    if t < THROTTLE_DOWN_S:
        return DPS_FTP_THROTTLE
    if t < HIGH_GATE_S:
        f = (t - THROTTLE_DOWN_S) / (HIGH_GATE_S - THROTTLE_DOWN_S)
        return 0.57 + 0.05 * f
    if t < P66_START_S:
        f = (t - HIGH_GATE_S) / (P66_START_S - HIGH_GATE_S)
        return 0.55 - 0.20 * f
    return 0.32


def _calibrated_speed(grid: list[float], decel: list[float]) -> list[float]:
    """Integrate the deceleration profile through the velocity anchors.

    Segments between anchors are scaled so the profile passes through the
    anchors exactly; the pre-high-gate segment (the well-documented FTP burn)
    is integrated unscaled, which is what yields the window-start velocity.
    """

    n = len(grid)
    dt = REFERENCE_DT_S
    speed = [0.0] * n
    anchors = [
        (int(HIGH_GATE_S), HIGH_GATE_HSPEED_MPS),
        (333, ANCHOR_333S_MPS),
        (353, ANCHOR_353S_MPS),
        (n - 1, 0.0),
    ]
    for (i_a, v_a), (i_b, v_b) in zip(anchors, anchors[1:]):
        raw_delta = sum(decel[i] * dt for i in range(i_a, i_b))
        scale = (v_a - v_b) / raw_delta if raw_delta > 0.0 else 0.0
        speed[i_b] = v_b
        for i in range(i_b - 1, i_a - 1, -1):
            speed[i] = speed[i + 1] + scale * decel[i] * dt
    i_hg = anchors[0][0]
    for i in range(i_hg - 1, -1, -1):
        speed[i] = speed[i + 1] + decel[i] * dt
    return speed


def _reconstruct_horizontal(
    grid: list[float], rate_mps: list[float]
) -> tuple[list[float], list[float]]:
    """Reconstruct horizontal speed and downrange distance over the window.

    The documented DPS throttle history sets the total thrust acceleration;
    the recorded altitude profile sets the vertical share (Newton in the
    vertical axis, including the centrifugal relief of the residual orbital
    velocity); the horizontal deceleration is the remainder. This keeps the
    reconstruction flyable at the documented throttle, so a controller
    tracking it does not fight the DPS erosion band. The profile is then
    calibrated through documented velocity anchors (high gate, transcript
    callouts, touchdown), and downrange distance is integrated backward from
    touchdown (x = 0 at the landing site).
    """

    n = len(grid)
    dt = REFERENCE_DT_S
    speed = [0.0] * n
    # The vertical share depends on the centrifugal relief, which depends on
    # the speed being reconstructed; two fixed-point passes converge.
    for _ in range(2):
        mass = WINDOW_START_MASS_KG
        decel: list[float] = []
        for i in range(n):
            throttle = _reconstruction_throttle(grid[i])
            thrust = throttle * DPS_MAX_THRUST_N
            lo = max(0, i - 1)
            hi = min(n - 1, i + 1)
            rate_slope = (rate_mps[hi] - rate_mps[lo]) / ((hi - lo) * dt) if hi > lo else 0.0
            g_eff = max(G_MOON - speed[i] ** 2 / R_MOON_M, 0.0)
            vertical = max(g_eff + rate_slope, MIN_VERTICAL_ACCEL_MPS2)
            total = thrust / mass
            decel.append(math.sqrt(max(total * total - vertical * vertical, 0.0)))
            mass -= thrust / (DPS_ISP_S * G0) * dt
        speed = _calibrated_speed(grid, decel)

    downrange = [0.0] * n
    for i in range(n - 2, -1, -1):
        downrange[i] = downrange[i + 1] - 0.5 * (speed[i] + speed[i + 1]) * dt
    return speed, downrange


def build_reference(
    data: DescentData | None = None,
    altitude: tuple[list[float], list[float]] | None = None,
) -> Reference:
    if data is None:
        data = load_descent()
    if altitude is None:
        altitude = load_altitude()
    alt_time_s, alt_m = altitude
    t_end = data.gimbal_time_s[-1]
    steps = int(round(t_end / REFERENCE_DT_S))
    grid = [i * REFERENCE_DT_S for i in range(steps + 1)]

    # True altitude from the digitized mission-report chart. The digitizer
    # output is jumpy, so despike and smooth; no monotonic clamp — the real
    # profile briefly levels off during the final manual maneuvering.
    raw_alt = [interp(t, alt_time_s, alt_m) for t in grid]
    despiked = _median_filter(raw_alt, 5)
    smoothed = _moving_average(despiked, max(1, round(ALTITUDE_SMOOTH_S / REFERENCE_DT_S)))
    altitude_m = [max(value, 0.0) for value in smoothed]

    rate: list[float] = []
    for i in range(len(grid)):
        lo = max(0, i - 1)
        hi = min(len(grid) - 1, i + 1)
        dt = grid[hi] - grid[lo]
        rate.append((altitude_m[hi] - altitude_m[lo]) / dt if dt > 0 else 0.0)
    rate = _moving_average(rate, max(1, round(RATE_SMOOTH_S / REFERENCE_DT_S)))
    # The one-sided smoothing window plus the extrapolated head flatten the
    # first few seconds of rate; hold the first clean value across that zone.
    head = int(round((ALTITUDE_SKIP_HEAD_S + ALTITUDE_SMOOTH_S) / REFERENCE_DT_S))
    head = min(head, len(rate) - 1)
    for i in range(head):
        rate[i] = rate[head]

    # Approximate pitch-from-vertical trend from the dominant inner gimbal angle.
    inner = [interp(t, data.gimbal_time_s, data.inner_deg) for t in grid]
    inner = _moving_average(inner, max(1, round(PITCH_SMOOTH_S / REFERENCE_DT_S)))
    pitch = [-value for value in inner]

    # Landing-radar slant range, kept as a display series (it visibly exceeds
    # the true altitude during the pitched-back braking phase).
    raw_range = [interp(t, data.range_time_s, data.range_m) for t in grid]
    slant = _median_filter(raw_range, 5)
    slant = _moving_average(slant, max(1, round(RANGE_SMOOTH_S / REFERENCE_DT_S)))

    hspeed, downrange = _reconstruct_horizontal(grid, rate)

    # Chart digitization stops short of contact. Append a P66-style let-down
    # at the terminal contact rate so the vehicle reaches footpad height.
    grid, altitude_m, rate, pitch, slant, hspeed, downrange = _extend_to_contact(
        grid, altitude_m, rate, pitch, slant, hspeed, downrange
    )
    t_end = grid[-1] if grid else t_end

    return Reference(grid, altitude_m, rate, pitch, slant, hspeed, downrange, t_end)


def _extend_to_contact(
    grid: list[float],
    altitude_m: list[float],
    rate: list[float],
    pitch: list[float],
    slant: list[float],
    hspeed: list[float],
    downrange: list[float],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Append samples from the last chart altitude down to footpad contact."""
    if not altitude_m:
        return grid, altitude_m, rate, pitch, slant, hspeed, downrange
    alt = altitude_m[-1]
    if alt <= FOOTPAD_CONTACT_ALT_M + 1e-6:
        return grid, altitude_m, rate, pitch, slant, hspeed, downrange

    t = grid[-1]
    last_pitch = pitch[-1]
    last_slant = slant[-1]
    last_hspeed = hspeed[-1]
    last_downrange = downrange[-1]
    while alt > FOOTPAD_CONTACT_ALT_M + 1e-6:
        t += REFERENCE_DT_S
        next_alt = max(FOOTPAD_CONTACT_ALT_M, alt - TERMINAL_CONTACT_RATE_MPS * REFERENCE_DT_S)
        grid.append(t)
        altitude_m.append(next_alt)
        rate.append(-TERMINAL_CONTACT_RATE_MPS)
        pitch.append(last_pitch)
        # Slant ≈ altitude when upright over the site.
        slant.append(max(last_slant * (next_alt / alt) if alt > 1e-6 else next_alt, next_alt))
        # Null residual groundspeed over the site during the final let-down.
        last_hspeed = max(0.0, last_hspeed - 0.15 * REFERENCE_DT_S)
        hspeed.append(last_hspeed)
        downrange.append(last_downrange)
        alt = next_alt
    return grid, altitude_m, rate, pitch, slant, hspeed, downrange


def sanity_check(tolerance_m: float = 1e-3) -> dict[str, float]:
    """Confirm the derived measurements agree with the raw vendored sources."""

    derived = load_descent(DERIVED_PATH)
    raw_rows = 0
    raw_range = 0
    start = None
    max_range_err = 0.0
    derived_range = dict(zip(derived.range_time_s, derived.range_m))
    with RAW_PATH.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 8:
                continue
            try:
                dt = _parse_stamp(row[0])
            except ValueError:
                continue
            if start is None:
                start = dt
            t = round((dt - start).total_seconds(), 6)
            raw_rows += 1
            cell = row[7].strip().replace(",", "")
            if cell:
                raw_range += 1
                expected = float(cell) * FT_TO_M
                nearest = min(derived_range, key=lambda k: abs(k - t), default=None)
                if nearest is not None and abs(nearest - t) < 1e-6:
                    max_range_err = max(max_range_err, abs(derived_range[nearest] - expected))

    alt_time_s, alt_m = load_altitude()
    ref = build_reference()
    return {
        "raw_rows": raw_rows,
        "raw_range_rows": raw_range,
        "derived_range_rows": len(derived.range_m),
        "max_range_error_m": max_range_err,
        "tolerance_m": tolerance_m,
        "altitude_rows": len(alt_m),
        "altitude_first_m": alt_m[0],
        "altitude_last_m": alt_m[-1],
        "hspeed_init_mps": ref.horizontal_speed_mps[0],
        "hspeed_high_gate_mps": ref.horizontal_speed(HIGH_GATE_S),
        "hspeed_333s_mps": ref.horizontal_speed(333.0),
        "downrange_init_m": ref.downrange_m[0],
        "ok": (
            max_range_err <= tolerance_m
            # Radar-corrected altitude at lock-on: "39,000 or 40,000 feet".
            and 11_500.0 < alt_m[0] < 12_500.0
            and alt_m[-1] < 1.0
            and abs(ref.horizontal_speed(333.0) - ANCHOR_333S_MPS) < 0.5
            and 600.0 < ref.horizontal_speed_mps[0] < 1_100.0
        ),
    }


if __name__ == "__main__":
    ref = build_reference()
    print(f"reference samples: {len(ref.time_s)} over {ref.t_end:.1f} s")
    for t in range(0, int(ref.t_end) + 1, 30):
        print(
            f"  t={t:3d}s  alt={ref.altitude(t):8.1f} m  rate={ref.descent_rate(t):7.2f} m/s"
            f"  pitch={ref.pitch(t):7.2f} deg  vh={ref.horizontal_speed(t):6.1f} m/s"
            f"  x={ref.downrange(t) / 1000.0:8.2f} km  slant={ref.slant_range(t):8.1f} m"
        )
    print("sanity:", sanity_check())
