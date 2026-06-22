"""CI post-campaign gate: fail the hook on failed or invalid runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mc_metrics import run_passed


def _failed_run_ids(results_path: Path) -> list[str]:
    if not results_path.exists():
        return []
    with results_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [row.get("run_id", "") for row in rows if not run_passed(row) and row.get("run_id")]


def post_campaign(ctx):
    summary_path = Path(ctx.summary)
    summary = json.loads(summary_path.read_text())
    failed = int(summary.get("failed", 0))
    invalid = int(summary.get("invalid", 0))
    passed = int(summary.get("passed", 0))
    total = int(summary.get("total_runs", 0))
    if failed + invalid > 0:
        run_ids = _failed_run_ids(Path(ctx.results))
        detail = f" ({', '.join(run_ids)})" if run_ids else ""
        raise RuntimeError(
            f"apollo-lander monte-carlo CI gate: {failed} failed and {invalid} invalid "
            f"of {total} run(s) "
            f"({passed} passed){detail}"
        )
    return {
        "pass": True,
        "passed": passed,
        "failed": failed,
        "invalid": invalid,
        "total_runs": total,
    }
