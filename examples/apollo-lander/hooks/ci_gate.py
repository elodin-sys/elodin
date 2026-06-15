"""CI post-campaign gate: fail the hook (and thus the campaign) on run failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _failed_run_ids(results_path: Path) -> list[str]:
    if not results_path.exists():
        return []
    with results_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    failed = []
    for row in rows:
        passed = row.get("passed")
        if passed is not None and passed != "":
            if passed.strip().lower() not in {"true", "1", "yes"}:
                failed.append(row.get("run_id", ""))
            continue
        if row.get("status", "") != "ok":
            failed.append(row.get("run_id", ""))
    return [run_id for run_id in failed if run_id]


def post_campaign(ctx):
    summary_path = Path(ctx.summary)
    summary = json.loads(summary_path.read_text())
    failed = int(summary.get("failed", 0))
    passed = int(summary.get("passed", 0))
    total = int(summary.get("total_runs", 0))
    if failed > 0:
        run_ids = _failed_run_ids(Path(ctx.results))
        detail = f" ({', '.join(run_ids)})" if run_ids else ""
        raise RuntimeError(
            f"apollo-lander monte-carlo CI gate: {failed}/{total} run(s) failed "
            f"({passed} passed){detail}"
        )
    return {
        "pass": True,
        "passed": passed,
        "failed": failed,
        "total_runs": total,
    }
