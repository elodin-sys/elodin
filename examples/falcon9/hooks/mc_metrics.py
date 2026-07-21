"""Shared Monte Carlo metric helpers for the Falcon 9 campaign hooks.

Imported by score.py, report.py, ci_score.py, and ci_gate.py so the
soft-landing criteria, parity targets, and run-pass semantics live in exactly
one place. Standard library only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Soft-landing acceptance (mirrors main.py).
SOFT_VERTICAL_MPS = 2.0
SOFT_LATERAL_MPS = 1.0
SOFT_TILT_DEG = 10.0
SOFT_POS_ERR_M = 500.0

# Parity targets vs the recorded flight (plan Phase 9 / WHITEPAPER 13).
PARITY_SPEED_RMSE_MPS = 15.0
PARITY_ALT_RMSE_M = 150.0
PARITY_EVENT_ERR_S = 3.0

# Recorded CRS-12 webcast event times (data/crs12/events.json).
RECORDED_EVENTS_S = {
    "phase_4_t_s": 147.0,  # MECO
    "phase_8_t_s": 370.0,  # entry burn ignition
    "phase_10_t_s": 433.0,  # landing burn ignition
    "touchdown_t_s": 466.0,
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def to_float(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def soft_landing(result: dict) -> bool:
    if "soft_landing" in result:
        return bool(result["soft_landing"])
    vertical = to_float(result.get("touchdown_vertical_mps"))
    lateral = to_float(result.get("touchdown_lateral_mps"))
    tilt = to_float(result.get("touchdown_tilt_deg"))
    pos = to_float(result.get("touchdown_pos_err_m"))
    prop = to_float(result.get("prop_remaining_kg"), 0.0)
    if None in (vertical, lateral, tilt, pos):
        return False
    return (
        bool(result.get("landed", False))
        and vertical <= SOFT_VERTICAL_MPS
        and lateral <= SOFT_LATERAL_MPS
        and tilt <= SOFT_TILT_DEG
        and pos <= SOFT_POS_ERR_M
        and prop > 0.0
    )


def event_errors(result: dict) -> dict[str, float]:
    """Signed sim-minus-recorded event-time errors (s)."""
    errors = {}
    for key, recorded in RECORDED_EVENTS_S.items():
        t = to_float(result.get(key))
        if t is not None:
            errors[key] = t - recorded
    return errors


def fit_score(result: dict) -> float | None:
    """Combined parity-normalized fit (lower is better; ~3 = at target).

    Trajectory fit + event timing + landing quality. The landing term keeps
    crashed runs from out-ranking soft landings that fit slightly worse —
    parity means reproducing the SUCCESSFUL recorded flight.
    """
    speed = to_float(result.get("speed_rmse_mps"))
    alt = to_float(result.get("alt_rmse_m"))
    if speed is None or alt is None:
        return None
    ev = event_errors(result)
    ev_term = sum(abs(e) / PARITY_EVENT_ERR_S for e in ev.values()) / len(ev) if ev else 10.0
    vert = to_float(result.get("touchdown_vertical_mps"), 1e3)
    lat = to_float(result.get("touchdown_lateral_mps"), 1e3)
    tilt = to_float(result.get("touchdown_tilt_deg"), 180.0)
    pos = to_float(result.get("touchdown_pos_err_m"), 1e6)
    landing_term = (
        vert / SOFT_VERTICAL_MPS
        + lat / SOFT_LATERAL_MPS
        + tilt / SOFT_TILT_DEG
        + pos / SOFT_POS_ERR_M
    ) / 4.0
    crash_term = 0.0 if bool(result.get("landed", False)) else 50.0
    return (
        speed / PARITY_SPEED_RMSE_MPS
        + alt / PARITY_ALT_RMSE_M
        + ev_term
        + landing_term
        + crash_term
    )


def run_passed(row: dict) -> bool:
    passed = row.get("passed")
    if passed is not None and passed != "":
        return str(passed).strip().lower() in {"true", "1", "yes"}
    return row.get("status", "") == "ok"
