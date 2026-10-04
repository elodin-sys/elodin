"""CRS-12 / CRS-11 truth reference profiles from the vendored webcast data.

Stdlib-only (shared by the sim, tests, and tooling). Loads the raw stage-1
telemetry, cleans it per the monte-carlo skill recipe (uniform resample,
median despike, moving-average smooth), splits speed into vertical/horizontal
using the recorded altitude, and integrates a signed downrange profile for
the truth-ghost track. `python reference.py` prints the profile and runs
`sanity_check()`.

Data quality notes live in data/README.md. The vertical/horizontal split and
downrange are reconstructions (display-quantized inputs); speed/altitude are
the scoring channels, the reconstructions are visualization/guidance aids.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GRID_DT_S = 0.5


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


def _median_filter(values: list[float], window: int = 5) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sorted(values[lo:hi])[(hi - lo) // 2])
    return out


def _moving_average(values: list[float], window: int = 7) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _centered_rate(values: list[float], dt: float) -> list[float]:
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        out[i] = (values[hi] - values[lo]) / ((hi - lo) * dt)
    return out


@dataclass
class Reference:
    mission: str
    time_s: list[float]
    speed_mps: list[float]  # display-space ground-relative speed
    altitude_m: list[float]  # display-space geodetic altitude
    vspeed_mps: list[float]  # reconstruction: d(altitude)/dt
    hspeed_mps: list[float]  # reconstruction: sqrt(max(v^2 - vspeed^2, 0))
    downrange_m: list[float]  # reconstruction: signed along-track integral
    events: dict[str, float | None]
    t_end: float

    def speed(self, t: float) -> float:
        return _interp(t, self.time_s, self.speed_mps)

    def altitude(self, t: float) -> float:
        return _interp(t, self.time_s, self.altitude_m)

    def vspeed(self, t: float) -> float:
        return _interp(t, self.time_s, self.vspeed_mps)

    def hspeed(self, t: float) -> float:
        return _interp(t, self.time_s, self.hspeed_mps)

    def downrange(self, t: float) -> float:
        return _interp(t, self.time_s, self.downrange_m)


def build_reference(mission: str = "crs12") -> Reference:
    raw = json.loads((DATA_DIR / mission / "stage1_raw.json").read_text())
    events = json.loads((DATA_DIR / mission / "events.json").read_text())

    t_raw = [float(x) for x in raw["time"]]
    v_raw = [float(x) for x in raw["velocity"]]
    a_raw = [float(x) * 1000.0 for x in raw["altitude"]]

    t_end = t_raw[-1]
    n = int(t_end / GRID_DT_S) + 1
    time_s = [i * GRID_DT_S for i in range(n)]
    speed = [_interp(t, t_raw, v_raw) for t in time_s]
    alt = [_interp(t, t_raw, a_raw) for t in time_s]

    speed = _moving_average(_median_filter(speed), 7)
    alt = _moving_average(_median_filter(alt), 9)

    vspeed = _moving_average(_centered_rate(alt, GRID_DT_S), 9)
    hspeed = [math.sqrt(max(v * v - vz * vz, 0.0)) for v, vz in zip(speed, vspeed)]

    # Signed downrange: horizontal velocity reverses once, at the boostback
    # speed minimum (the recorded ~500 m/s dip near t ~= 203 s for CRS-12).
    bb_start = events.get("boostback_start") or 0.0
    bb_end = events.get("boostback_end") or 0.0
    i_lo = int(bb_start / GRID_DT_S)
    i_hi = max(i_lo + 1, int(bb_end / GRID_DT_S))
    i_rev = min(range(i_lo, min(i_hi, n)), key=lambda i: speed[i]) if i_hi > i_lo else n
    downrange = [0.0] * n
    for i in range(1, n):
        direction = 1.0 if i <= i_rev else -1.0
        downrange[i] = downrange[i - 1] + direction * hspeed[i] * GRID_DT_S

    return Reference(
        mission=mission,
        time_s=time_s,
        speed_mps=speed,
        altitude_m=alt,
        vspeed_mps=vspeed,
        hspeed_mps=hspeed,
        downrange_m=downrange,
        events={k: (float(v) if v is not None else None) for k, v in events.items()},
        t_end=t_end,
    )


def _window_min(ref: Reference, t0: float, t1: float) -> tuple[float, float]:
    lo = int(t0 / GRID_DT_S)
    hi = int(t1 / GRID_DT_S)
    i = min(range(lo, hi), key=lambda k: ref.speed_mps[k])
    return ref.time_s[i], ref.speed_mps[i]


def sanity_check(ref: Reference) -> None:
    """Assert the cleaned profile against documented anchors (data/README.md)."""
    assert ref.time_s[0] == 0.0 and ref.t_end > 400.0
    assert all(b >= a for a, b in zip(ref.time_s, ref.time_s[1:]))

    apogee = max(ref.altitude_m)
    assert 115_000.0 < apogee < 121_000.0, f"apogee {apogee}"

    meco_t = ref.events["meco"]
    v_meco = ref.speed(meco_t)
    assert 1_550.0 < v_meco < 1_750.0, f"MECO speed {v_meco}"

    if ref.mission == "crs12":
        t_dip, v_dip = _window_min(ref, ref.events["boostback_start"], ref.events["boostback_end"])
        assert 195.0 < t_dip < 210.0, f"boostback reversal at {t_dip}"
        assert 480.0 < v_dip < 540.0, f"reversal speed {v_dip}"
        # Entry-burn mean deceleration ~26 m/s^2 (WHITEPAPER 8.3).
        e0, e1 = ref.events["entry_start"], ref.events["entry_end"]
        decel = (ref.speed(e0) - ref.speed(e1)) / (e1 - e0)
        assert 20.0 < decel < 32.0, f"entry decel {decel}"
        # Landing-burn start state (recorded: 318 m/s at 4.6 km).
        t_land = ref.events["landing_start"]
        assert 280.0 < ref.speed(t_land) < 360.0
        assert 4_000.0 < ref.altitude(t_land) < 5_200.0

    # Terminal state: on the ground, slow.
    assert ref.altitude(ref.t_end) < 500.0
    assert ref.speed(ref.t_end) < 60.0
    # The split must reproduce the total speed within quantization noise
    # wherever the signal is clean (above 1 km, before touchdown).
    worst = 0.0
    for i, t in enumerate(ref.time_s):
        if ref.altitude_m[i] > 1_000.0:
            recon = math.hypot(ref.vspeed_mps[i], ref.hspeed_mps[i])
            worst = max(worst, abs(recon - ref.speed_mps[i]))
    assert worst < 25.0, f"split residual {worst}"


if __name__ == "__main__":
    for mission in ("crs12", "crs11"):
        ref = build_reference(mission)
        sanity_check(ref)
        apogee = max(ref.altitude_m)
        print(f"[{mission}] {len(ref.time_s)} samples, t_end={ref.t_end:.1f} s")
        print(f"  apogee {apogee / 1000.0:.1f} km, MECO v {ref.speed(ref.events['meco']):.0f} m/s")
        for name in ("maxq", "meco", "boostback_start", "entry_start", "landing_start"):
            t = ref.events.get(name)
            if t is not None:
                print(
                    f"  {name:16} t={t:5.0f}  v={ref.speed(t):7.1f} m/s  "
                    f"alt={ref.altitude(t) / 1000.0:6.2f} km  dr={ref.downrange(t) / 1000.0:7.2f} km"
                )
        print(f"  touchdown downrange {ref.downrange(ref.t_end) / 1000.0:.1f} km")
        print("  sanity_check: OK")
