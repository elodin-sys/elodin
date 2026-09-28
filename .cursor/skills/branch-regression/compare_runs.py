#!/usr/bin/env python3
"""Compare two branch-regression capture directories and write reports.

Usage:
    python compare_runs.py <base-dir> <head-dir> --html report.html
        [--rmse-threshold 0.05]

Per example (union of *.exit files in both dirs) it compares:
- exit codes
- WARN/ERROR log lines new in head (timestamps stripped, deduped)
- screenshot RMSE (normalized 0..1) when both PNGs exist

Exit code 1 when any example is flagged, 0 otherwise.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

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


def screenshot_html(capture: Path, name: str, report: Path) -> str:
    image = capture / f"{name}.png"
    if not image.exists() or not image.stat().st_size:
        return '<span class="missing">missing</span>'
    relative = Path(os.path.relpath(image, report.parent)).as_posix()
    url = quote(relative)
    alt = html.escape(f"{capture.name} screenshot for {name}", quote=True)
    return f'<a href="{url}"><img src="{url}" alt="{alt}" loading="lazy"></a>'


def write_html_report(
    path: Path,
    base: Path,
    head: Path,
    results: list[dict[str, object]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_commit = html.escape(read_exit(base / "commit.txt"))
    head_commit = html.escape(read_exit(head / "commit.txt"))
    rows = []
    details = []
    for result in results:
        name = str(result["name"])
        issues = result["issues"]
        assert isinstance(issues, list)
        flagged = bool(result["problems"])
        row_class = ' class="flagged"' if flagged else ""
        rows.append(
            f"""<tr{row_class}>
<td>{html.escape(name)}</td>
<td>{html.escape(str(result["base_exit"]))}</td>
<td>{html.escape(str(result["head_exit"]))}</td>
<td>{len(issues)}</td>
<td>{html.escape(str(result["rmse"]))}</td>
<td>{html.escape(str(result["verdict"]))}</td>
<td>{screenshot_html(base, name, path)}</td>
<td>{screenshot_html(head, name, path)}</td>
</tr>"""
        )
        if issues:
            issue_items = "".join(f"<li><code>{html.escape(issue)}</code></li>" for issue in issues)
            details.append(
                f"<details><summary>{html.escape(name)} — {len(issues)} new WARN/ERROR</summary>"
                f"<ul>{issue_items}</ul></details>"
            )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Branch regression report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: .5rem; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; position: sticky; top: 0; }}
tr.flagged {{ background: #fff1f0; }}
img {{ display: block; width: 280px; max-height: 180px; object-fit: contain; }}
code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
.commits {{ display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; }}
.missing {{ color: #cf222e; }}
details {{ margin: .75rem 0; }}
</style>
</head>
<body>
<h1>Branch regression report</h1>
<div class="commits">
<strong>Base</strong><code>{base_commit}</code>
<strong>Head</strong><code>{head_commit}</code>
<strong>RMSE threshold</strong><span>{threshold}</span>
</div>
<p>Click any screenshot thumbnail to open the full-size image.</p>
<table>
<thead><tr><th>Example</th><th>Base exit</th><th>Head exit</th><th>New WARN/ERROR</th><th>RMSE</th><th>Verdict</th><th>Base screenshot</th><th>Head screenshot</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
{"".join(details)}
</body>
</html>
"""
    path.write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--rmse-threshold", type=float, default=0.05)
    parser.add_argument("--html", type=Path)
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
    results = []
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
        results.append(
            {
                "name": name,
                "base_exit": base_exit,
                "head_exit": head_exit,
                "issues": new_issues,
                "rmse": rmse_str,
                "verdict": verdict,
                "problems": problems,
            }
        )

    for name, issues in details:
        print(f"\n### New WARN/ERROR in head — {name}")
        for line in issues[:20]:
            print(f"- `{line}`")
        if len(issues) > 20:
            print(f"- … and {len(issues) - 20} more")

    if args.html is not None:
        write_html_report(args.html, args.base, args.head, results, args.rmse_threshold)

    if flagged:
        print(f"\nFlagged examples: {', '.join(flagged)}")
        return 1
    print("\nAll examples within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
