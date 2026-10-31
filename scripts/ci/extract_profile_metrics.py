#!/usr/bin/env python3
"""Extract profile metrics from a benchmark run log; missing metrics are written as null."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRIC_PATTERNS = {
    "build_time_ms": re.compile(r"^build time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", re.MULTILINE),
    "real_time_factor_median": re.compile(
        r"^real_time_factor_median:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", required=True, type=Path, help="Path to benchmark run log")
    parser.add_argument("--ticks", required=True, type=int, help="Ticks used for the benchmark run")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for extracted profile metrics",
    )
    return parser.parse_args()


def _extract_metric(text: str, name: str) -> float | None:
    matches = METRIC_PATTERNS[name].findall(text)
    return float(matches[-1]) if matches else None


def main() -> int:
    args = _parse_args()
    text = args.run_log.read_text(encoding="utf-8", errors="replace")

    metrics = {
        "ticks": args.ticks,
        "build_time_ms": _extract_metric(text, "build_time_ms"),
        "real_time_factor_median": _extract_metric(text, "real_time_factor_median"),
    }

    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
