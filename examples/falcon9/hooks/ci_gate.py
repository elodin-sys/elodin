"""CI gate: fail the pipeline when any smoke run failed."""

from __future__ import annotations

import csv
from pathlib import Path


def post_campaign(ctx):
    results_path = Path(ctx.results)
    rows = list(csv.DictReader(results_path.open())) if results_path.exists() else []
    failed = [
        row["run_id"]
        for row in rows
        if str(row.get("passed", "")).strip().lower() not in {"true", "1", "yes"}
    ]
    if not rows:
        raise RuntimeError("falcon9 CI campaign produced no runs")
    if failed:
        raise RuntimeError(f"falcon9 CI smoke failed for runs: {', '.join(failed)}")
    return {"runs": len(rows), "failed": 0}
