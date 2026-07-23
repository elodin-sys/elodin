"""CI smoke scoring: infrastructure check, not a landing validation.

The CI run truncates the mission with ELODIN_FALCON9_MAX_TICKS (~40 s of
flight), so it passes when the SITL stack produced a result with the booster
airborne under closed-loop control — plant, sensors, UDP bridge, FSW,
scoring, and hooks all exercised.
"""

from __future__ import annotations

from pathlib import Path

from mc_metrics import read_json, to_float


def post_run(ctx):
    result = read_json(Path(ctx.run_dir) / "result.json")
    lifted = to_float(result.get("phase_1_t_s")) is not None
    ascending = to_float(result.get("phase_3_t_s")) is not None
    return {
        "valid": bool(result),
        "pass": bool(result) and lifted and ascending,
        "speed_rmse_mps": to_float(result.get("speed_rmse_mps"), float("inf")),
        "final_phase": to_float(result.get("final_phase"), -1.0),
    }
