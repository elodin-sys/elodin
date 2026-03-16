#!/usr/bin/env python3
"""
Analyze Tracy CSV from the sensor-camera render server and give a pass/fail verdict.

Reference thresholds are hardcoded — adjust here if the CI hardware changes.

Usage:
    python scripts/ci/sensor_camera_tracy_analysis.py trace-render.csv
"""

from __future__ import annotations

import csv
import sys

THRESHOLDS_MEAN_US = {
    "sensor_camera_poll_wait": 5_000,
    "sensor_camera_image_copy_driver": 10_000,
}


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <trace-render.csv>")
        return 1

    spans: dict[str, float] = {}
    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip('"')
            try:
                spans[name] = float(row.get("mean_ns", 0)) / 1000.0
            except (ValueError, TypeError):
                continue

    if not spans:
        print("FAIL: no spans in CSV")
        return 1

    fail = False
    for span, limit in THRESHOLDS_MEAN_US.items():
        mean = spans.get(span)
        if mean is None:
            print(f"  SKIP: {span} not found")
            continue
        ok = mean <= limit
        tag = "OK" if ok else "FAIL"
        print(f"  {tag}: {span} mean={mean:.0f}µs (limit {limit}µs)")
        if not ok:
            fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
