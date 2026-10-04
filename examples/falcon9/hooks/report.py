from __future__ import annotations

import csv
import math
from pathlib import Path

from mc_metrics import (
    PARITY_ALT_RMSE_M,
    PARITY_SPEED_RMSE_MPS,
    read_json,
    to_float,
)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def _stats(runs: list[dict], key: str) -> list[float]:
    return [v for run in runs if (v := to_float(run["post"].get(key))) is not None]


def post_campaign(ctx):
    out_dir = Path(ctx.out_dir)
    rows = []
    results_path = Path(ctx.results)
    if results_path.exists():
        with results_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

    runs = []
    for row in rows:
        result_path = out_dir / row.get("result_json", "")
        runs.append(
            {
                "run_id": row.get("run_id", ""),
                "result": read_json(result_path),
                "post": read_json(result_path.with_name("post_run_result.json")),
                "params": read_json(result_path.with_name("post_run_context.json")).get(
                    "params", {}
                ),
                "row": row,
            }
        )

    # Calibration semantics: aggregate every VALID run (the process ran and
    # produced a result); "pass" (soft landing at parity) is reported, not a
    # filter — otherwise the report is empty until the model is already good.
    ok = [r for r in runs if r["row"].get("status", "") == "ok" and r["post"]]
    soft = [r for r in ok if bool(r["post"].get("soft_landing", False))]
    landed = [r for r in ok if bool(r["post"].get("landed", False))]

    speed_rmse = _stats(ok, "speed_rmse_mps")
    alt_rmse = _stats(ok, "alt_rmse_m")
    pos_err = _stats(ok, "touchdown_pos_err_m")
    vert = _stats(ok, "touchdown_vertical_mps")
    lat = _stats(ok, "touchdown_lateral_mps")
    fits = [(r, v) for r in ok if (v := to_float(r["post"].get("fit_score"))) is not None]
    best = min(fits, key=lambda item: item[1], default=(None, None))

    lines = [
        "Falcon 9 CRS-12 parity campaign report",
        "=======================================",
        "",
        f"runs completed: {len(ok)}/{len(runs)}",
        f"landed: {len(landed)}/{len(ok)}   soft landings: {len(soft)}/{len(ok)}",
        "",
        f"Speed RMSE vs recorded (target <= {PARITY_SPEED_RMSE_MPS} m/s)",
        f"  mean: {_fmt(sum(speed_rmse) / len(speed_rmse) if speed_rmse else None, ' m/s')}"
        f"   min: {_fmt(min(speed_rmse) if speed_rmse else None, ' m/s')}",
        f"Altitude RMSE vs recorded (target <= {PARITY_ALT_RMSE_M} m)",
        f"  mean: {_fmt(sum(alt_rmse) / len(alt_rmse) if alt_rmse else None, ' m')}"
        f"   min: {_fmt(min(alt_rmse) if alt_rmse else None, ' m')}",
        "",
        "Touchdown",
        f"  pos err   mean: {_fmt(sum(pos_err) / len(pos_err) if pos_err else None, ' m')}"
        f"  p95: {_fmt(_percentile(pos_err, 0.95), ' m')}",
        f"  vertical  mean: {_fmt(sum(vert) / len(vert) if vert else None, ' m/s')}"
        f"  p95: {_fmt(_percentile(vert, 0.95), ' m/s')}",
        f"  lateral   mean: {_fmt(sum(lat) / len(lat) if lat else None, ' m/s')}"
        f"  p95: {_fmt(_percentile(lat, 0.95), ' m/s')}",
        "",
        "Event-time errors vs recorded (s)",
    ]
    for key in ("err_phase_4_t_s", "err_phase_8_t_s", "err_phase_10_t_s", "err_touchdown_t_s"):
        vals = _stats(ok, key)
        lines.append(
            f"  {key[4:]:16} mean {_fmt(sum(vals) / len(vals) if vals else None)}"
            f"  |max| {_fmt(max((abs(v) for v in vals), default=None))}"
        )
    lines += [
        "",
        "Best-fit run (parity-normalized fit score)",
        f"  run: {best[0]['run_id'] if best[0] else 'n/a'}   fit: {_fmt(best[1])}",
    ]
    if best[0]:
        lines.append("  params:")
        for key, value in sorted(best[0]["params"].items()):
            lines.append(f"    {key}: {value}")
    lines += [
        "",
        "Calibration: narrow spec.toml around the best-fit params and re-run",
        "(keep the LHS seed fixed while iterating).",
    ]

    report = out_dir / "post_campaign" / "falcon9_report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    report.write_text(text)
    print(text)

    return {
        "completed": len(ok),
        "landed": len(landed),
        "soft_landings": len(soft),
        "best_run": best[0]["run_id"] if best[0] else None,
        "best_fit_score": best[1],
        "best_speed_rmse": to_float(best[0]["post"].get("speed_rmse_mps")) if best[0] else None,
        "best_alt_rmse": to_float(best[0]["post"].get("alt_rmse_m")) if best[0] else None,
        "report": str(report),
    }
