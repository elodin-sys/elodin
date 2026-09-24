#!/usr/bin/env python3
"""Compare two branch-regression capture directories and print a markdown report.

Usage:
    python compare_runs.py <base-dir> <head-dir> [--rmse-threshold 0.05]

Per example (union of *.exit files in both dirs) it compares:
- exit codes
- WARN/ERROR log lines new in head (timestamps stripped, deduped)
- screenshot RMSE (normalized 0..1) when both PNGs exist

Exit code 1 when any example is flagged, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TIMESTAMP = re.compile(r"^\S*\d{2}:\d{2}:\d{2}\S*\s+")
LEVEL = re.compile(r"\b(WARN|ERROR)\b")


def log_issues(path: Path) -> set[str]:
    if not path.exists():
        return set()
    issues = set()
    for line in path.read_text(errors="replace").splitlines():
        if LEVEL.search(line):
            issues.add(TIMESTAMP.sub("", line).strip())
    return issues


def screenshot_rmse(a: Path, b: Path) -> float | None:
    if not (a.exists() and b.exists() and a.stat().st_size and b.stat().st_size):
        return None
    import numpy as np
    from PIL import Image

    im_a = Image.open(a).convert("RGB")
    im_b = Image.open(b).convert("RGB")
    if im_a.size != im_b.size:
        im_b = im_b.resize(im_a.size)
    arr_a = np.asarray(im_a, dtype=np.float64) / 255.0
    arr_b = np.asarray(im_b, dtype=np.float64) / 255.0
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


def read_exit(path: Path) -> str:
    return path.read_text().strip() if path.exists() else "missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--rmse-threshold", type=float, default=0.05)
    args = parser.parse_args()

    examples = sorted(
        {p.stem for p in args.base.glob("*.exit")} | {p.stem for p in args.head.glob("*.exit")}
    )
    if not examples:
        print(f"no captures found under {args.base} / {args.head}", file=sys.stderr)
        return 2

    flagged = []
    print("| example | base exit | head exit | new WARN/ERROR | RMSE | verdict |")
    print("|---|---|---|---|---|---|")
    details = []
    for name in examples:
        base_exit = read_exit(args.base / f"{name}.exit")
        head_exit = read_exit(args.head / f"{name}.exit")
        new_issues = sorted(
            log_issues(args.head / f"{name}.log") - log_issues(args.base / f"{name}.log")
        )
        rmse = screenshot_rmse(args.base / f"{name}.png", args.head / f"{name}.png")

        problems = []
        if base_exit != head_exit:
            problems.append("exit code changed")
        if head_exit not in ("0", "missing") and base_exit == "0":
            problems.append("head failed")
        if new_issues:
            problems.append(f"{len(new_issues)} new log issue(s)")
        base_png = args.base / f"{name}.png"
        head_png = args.head / f"{name}.png"
        if (
            base_png.exists()
            and base_png.stat().st_size
            and (not head_png.exists() or not head_png.stat().st_size)
        ):
            problems.append("screenshot missing in head")
        if rmse is not None and rmse > args.rmse_threshold:
            problems.append(f"RMSE {rmse:.3f} > {args.rmse_threshold}")

        verdict = "FLAG: " + "; ".join(problems) if problems else "ok"
        if problems:
            flagged.append(name)
        rmse_str = f"{rmse:.4f}" if rmse is not None else "n/a"
        print(
            f"| {name} | {base_exit} | {head_exit} | {len(new_issues)} | {rmse_str} | {verdict} |"
        )
        if new_issues:
            details.append((name, new_issues))

    for name, issues in details:
        print(f"\n### New WARN/ERROR in head — {name}")
        for line in issues[:20]:
            print(f"- `{line}`")
        if len(issues) > 20:
            print(f"- … and {len(issues) - 20} more")

    if flagged:
        print(f"\nFlagged examples: {', '.join(flagged)}")
        return 1
    print("\nAll examples within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
