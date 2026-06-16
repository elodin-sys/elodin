from __future__ import annotations

import csv
import math
from pathlib import Path

from mc_metrics import read_json, run_passed, to_float


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def post_campaign(ctx):
    out_dir = Path(ctx.out_dir)
    results_path = Path(ctx.results)
    rows = []
    if results_path.exists():
        with results_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    runs = []
    for row in rows:
        run_id = row.get("run_id", "")
        result_path = out_dir / row.get("result_json", "")
        result = read_json(result_path)
        post_result = read_json(result_path.with_name("post_run_result.json"))
        context = read_json(result_path.with_name("post_run_context.json"))
        runs.append(
            {
                "run_id": run_id,
                "status": row.get("status", ""),
                "passed": row.get("passed", ""),
                "wall_ms": to_float(row.get("wall_ms")),
                "result": result,
                "post": post_result,
                "params": context.get("params", {}),
            }
        )

    ok_runs = [run for run in runs if run_passed(run)]
    soft = [run for run in ok_runs if bool(run["post"].get("soft_landing", False))]
    touchdown_speeds = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("touchdown_speed_mps"))) is not None
    ]
    horizontal_speeds = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("horizontal_speed_mps"))) is not None
    ]
    fuel_remaining = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("fuel_remaining_kg"))) is not None
    ]
    rcs_fuel_remaining = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("rcs_fuel_remaining_kg"))) is not None
    ]
    rmse_values = [
        (run, value)
        for run in ok_runs
        if (value := to_float(run["post"].get("traj_rmse_m"))) is not None
    ]
    pitch_rmse = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("pitch_rmse_deg"))) is not None
    ]
    downrange_miss = [
        value
        for run in ok_runs
        if (value := to_float(run["post"].get("downrange_miss_m"))) is not None
    ]
    best = min(rmse_values, key=lambda item: item[1], default=(None, None))

    lines = [
        "Apollo 11 lander Monte Carlo report",
        "====================================",
        "",
        f"runs completed: {len(ok_runs)}/{len(runs)}",
        f"soft landings: {len(soft)}/{len(ok_runs)}",
        f"success rate: {_fmt((len(soft) / len(ok_runs) * 100.0) if ok_runs else None, '%')}",
        "",
        "Vertical touchdown speed",
        f"  mean: {_fmt((sum(touchdown_speeds) / len(touchdown_speeds)) if touchdown_speeds else None, ' m/s')}",
        f"  p95:  {_fmt(_percentile(touchdown_speeds, 0.95), ' m/s')}",
        f"  max:  {_fmt(max(touchdown_speeds) if touchdown_speeds else None, ' m/s')}",
        "",
        "Horizontal touchdown speed",
        f"  mean: {_fmt((sum(horizontal_speeds) / len(horizontal_speeds)) if horizontal_speeds else None, ' m/s')}",
        f"  p95:  {_fmt(_percentile(horizontal_speeds, 0.95), ' m/s')}",
        f"  max:  {_fmt(max(horizontal_speeds) if horizontal_speeds else None, ' m/s')}",
        "",
        "Fuel remaining",
        f"  mean: {_fmt((sum(fuel_remaining) / len(fuel_remaining)) if fuel_remaining else None, ' kg')}",
        f"  p05:  {_fmt(_percentile(fuel_remaining, 0.05), ' kg')}",
        f"  min:  {_fmt(min(fuel_remaining) if fuel_remaining else None, ' kg')}",
        "",
        "RCS fuel remaining",
        f"  mean: {_fmt((sum(rcs_fuel_remaining) / len(rcs_fuel_remaining)) if rcs_fuel_remaining else None, ' kg')}",
        f"  min:  {_fmt(min(rcs_fuel_remaining) if rcs_fuel_remaining else None, ' kg')}",
        "",
        "Landing dispersion (downrange miss from site)",
        f"  mean: {_fmt((sum(downrange_miss) / len(downrange_miss)) if downrange_miss else None, ' m')}",
        f"  p95:  {_fmt(_percentile(downrange_miss, 0.95), ' m')}",
        f"  max:  {_fmt(max(downrange_miss) if downrange_miss else None, ' m')}",
        "",
        "Apollo telemetry fit",
        f"  best run: {best[0]['run_id'] if best[0] else 'n/a'}",
        f"  best altitude RMSE: {_fmt(best[1], ' m')}",
        f"  mean pitch RMSE: {_fmt((sum(pitch_rmse) / len(pitch_rmse)) if pitch_rmse else None, ' deg')}",
    ]
    if best[0]:
        lines.append("  best-fit params:")
        for key, value in sorted(best[0]["params"].items()):
            lines.append(f"    {key}: {value}")
    lines.extend(
        [
            "",
            "Calibration hint",
            "  Narrow spec.toml ranges around the best-fit params and re-run, or use calibrate.py for the optional automated loop.",
        ]
    )

    report = out_dir / "post_campaign" / "apollo_lander_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    report.write_text(text)
    print(text)

    return {
        "completed": len(ok_runs),
        "soft_landings": len(soft),
        "success_rate": (len(soft) / len(ok_runs)) if ok_runs else None,
        "best_run": best[0]["run_id"] if best[0] else None,
        "best_traj_rmse_m": best[1],
        "report": str(report),
    }
