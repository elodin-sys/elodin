from __future__ import annotations

from pathlib import Path

from mc_metrics import read_json, soft_landing, to_float


def post_run(ctx):
    result = read_json(Path(ctx.run_dir) / "result.json")

    landed = bool(result.get("landed", False))
    # Honor the sim's own `soft_landing` flag when present, else recompute from
    # the raw kinematics (mc_metrics checks the first arg before the second).
    passed = soft_landing(result, result)

    return {
        "landed": landed,
        "soft_landing": passed,
        "pass": passed,
        "touchdown_speed_mps": to_float(result.get("touchdown_speed"), float("inf")),
        "horizontal_speed_mps": to_float(result.get("horizontal_speed"), float("inf")),
        "fuel_remaining_kg": to_float(result.get("fuel_remaining"), 0.0),
        "rcs_fuel_remaining_kg": to_float(result.get("rcs_fuel_remaining"), 0.0),
        "traj_rmse_m": to_float(result.get("traj_rmse"), float("inf")),
        "pitch_rmse_deg": to_float(result.get("pitch_rmse"), float("inf")),
        "downrange_miss_m": to_float(result.get("downrange_miss"), float("inf")),
    }
