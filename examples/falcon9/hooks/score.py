from __future__ import annotations

from pathlib import Path

from mc_metrics import event_errors, fit_score, read_json, soft_landing, to_float


def post_run(ctx):
    result = read_json(Path(ctx.run_dir) / "result.json")
    landed = bool(result.get("landed", False))
    passed = soft_landing(result)
    errors = event_errors(result)

    out = {
        "landed": landed,
        "soft_landing": passed,
        "valid": bool(result),
        "pass": passed,
        "speed_rmse_mps": to_float(result.get("speed_rmse_mps"), float("inf")),
        "alt_rmse_m": to_float(result.get("alt_rmse_m"), float("inf")),
        "fit_score": to_float(fit_score(result), float("inf")),
        "touchdown_vertical_mps": to_float(result.get("touchdown_vertical_mps"), float("inf")),
        "touchdown_lateral_mps": to_float(result.get("touchdown_lateral_mps"), float("inf")),
        "touchdown_tilt_deg": to_float(result.get("touchdown_tilt_deg"), float("inf")),
        "touchdown_pos_err_m": to_float(result.get("touchdown_pos_err_m"), float("inf")),
        "prop_remaining_kg": to_float(result.get("prop_remaining_kg"), 0.0),
        "purge_events": to_float(result.get("purge_events"), 0.0),
        "qbar_peak_descent_kpa": to_float(result.get("qbar_peak_descent_kpa"), 0.0),
    }
    for key, err in errors.items():
        out[f"err_{key}"] = err
    return out
