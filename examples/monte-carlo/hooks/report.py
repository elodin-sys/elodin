from __future__ import annotations

import csv
import json
from pathlib import Path


def post_campaign(ctx):
    results = Path(ctx.out_dir) / "results.csv"
    passed = 0
    total = 0
    errors = []
    if results.exists():
        with results.open() as f:
            for row in csv.DictReader(f):
                total += 1
                if row.get("passed") == "true":
                    passed += 1
                if row.get("error"):
                    errors.append(float(row["error"]))
    summary = json.loads(Path(ctx.summary).read_text())
    report = Path(ctx.out_dir) / "post_campaign" / "report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"completed={total}\n"
        f"passed={passed}\n"
        f"failed={summary.get('failed', 0)}\n"
        f"invalid={summary.get('invalid', 0)}\n"
        f"mean_error={sum(errors) / len(errors) if errors else float('nan')}\n"
    )
    return {"completed": total, "passed": passed}
