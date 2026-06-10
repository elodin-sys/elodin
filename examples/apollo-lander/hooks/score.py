from __future__ import annotations

import json
import math
from pathlib import Path

SOFT_HORIZONTAL_SPEED_MPS = 1.0
SOFT_VERTICAL_SPEED_MPS = 3.0
UPRIGHT_DOT_MIN = 0.94


def _number(value, default=float("inf")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def post_run(ctx):
    result_path = Path(ctx.run_dir) / "result.json"
    result = {}
    if result_path.exists():
        result = json.loads(result_path.read_text())

    touchdown_speed = _number(result.get("touchdown_speed"))
    horizontal_speed = _number(result.get("horizontal_speed"))
    fuel_remaining = _number(result.get("fuel_remaining"), default=0.0)
    rcs_fuel_remaining = _number(result.get("rcs_fuel_remaining"), default=0.0)
    traj_rmse = _number(result.get("traj_rmse"))
    pitch_rmse = _number(result.get("pitch_rmse"))
    downrange_miss = _number(result.get("downrange_miss"))
    upright_dot = _number(result.get("upright_dot"), default=-1.0)
    landed = bool(result.get("landed", False))
    computed_soft_landing = (
        landed
        and touchdown_speed <= SOFT_VERTICAL_SPEED_MPS
        and horizontal_speed <= SOFT_HORIZONTAL_SPEED_MPS
        and upright_dot >= UPRIGHT_DOT_MIN
        and fuel_remaining > 0.0
    )
    soft_landing = bool(result.get("soft_landing", computed_soft_landing))

    return {
        "landed": landed,
        "soft_landing": soft_landing,
        "pass": soft_landing,
        "touchdown_speed_mps": touchdown_speed,
        "horizontal_speed_mps": horizontal_speed,
        "fuel_remaining_kg": fuel_remaining,
        "rcs_fuel_remaining_kg": rcs_fuel_remaining,
        "traj_rmse_m": traj_rmse,
        "pitch_rmse_deg": pitch_rmse,
        "downrange_miss_m": downrange_miss,
    }
