from __future__ import annotations

import csv
from pathlib import Path


def post_campaign(ctx):
    results = Path(ctx.out_dir) / "results.csv"
    passed = 0
    total = 0
    if results.exists():
        with results.open() as f:
            for row in csv.DictReader(f):
                total += 1
                if row.get("status") == "ok":
                    passed += 1
    report = Path(ctx.out_dir) / "post_campaign" / "report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"completed={total}\nstatus_ok={passed}\n")
    return {"completed": total, "status_ok": passed}
