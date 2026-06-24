from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _to_float(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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
    results = out_dir / "results.csv"
    rows = []
    if results.exists():
        with results.open(newline="") as f:
            rows = list(csv.DictReader(f))
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed") == "true")
    errors = [error for row in rows if (error := _to_float(row.get("error"))) is not None]
    capture_radius = next(
        (value for row in rows if (value := _to_float(row.get("capture_radius_m"))) is not None),
        None,
    )
    worst = []
    for row in rows:
        error = _to_float(row.get("error"))
        if error is None:
            continue
        context_path = out_dir / "runs" / row.get("run_id", "") / "post_run_context.json"
        try:
            context = json.loads(context_path.read_text())
        except OSError:
            context = {}
        worst.append((error, row.get("run_id", ""), context.get("params", {})))
    worst.sort(reverse=True, key=lambda item: item[0])

    summary = json.loads(Path(ctx.summary).read_text())
    report = Path(ctx.out_dir) / "post_campaign" / "report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Native Monte Carlo SITL report",
        "==============================",
        "",
        f"runs completed: {total}",
        f"passed: {passed}/{total}",
        f"pass rate: {_fmt((passed / total * 100.0) if total else None, '%')}",
        f"failed: {summary.get('failed', 0)}",
        f"invalid: {summary.get('invalid', 0)}",
        f"capture radius: {_fmt(capture_radius, ' m')}",
        "",
        "Final position error",
        f"  mean: {_fmt((sum(errors) / len(errors)) if errors else None, ' m')}",
        f"  p50:  {_fmt(_percentile(errors, 0.50), ' m')}",
        f"  p80:  {_fmt(_percentile(errors, 0.80), ' m')}",
        f"  p95:  {_fmt(_percentile(errors, 0.95), ' m')}",
        f"  max:  {_fmt(max(errors) if errors else None, ' m')}",
        "",
        "Worst misses",
    ]
    for error, run_id, params in worst[:5]:
        param_text = ", ".join(f"{key}={value:.3g}" for key, value in sorted(params.items()))
        lines.append(f"  {run_id}: error={error:.3f} m ({param_text})")
    lines.extend(
        [
            "",
            "Interpretation",
            "  This example intentionally uses a tiny point-mass plant and simple saturated PD controller. Failures are scored misses near the sampled envelope boundary, not infrastructure failures.",
        ]
    )
    text = "\n".join(lines) + "\n"
    report.write_text(text)
    print(text)
    return {
        "completed": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else None,
        "capture_radius_m": capture_radius,
    }
