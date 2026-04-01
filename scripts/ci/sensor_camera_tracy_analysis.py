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

REPORT_SPANS = ("impeller2_table_sink",)


def _parse_mean_us(row: dict[str, str]) -> float | None:
    try:
        return float(row.get("mean_ns", 0)) / 1000.0
    except (ValueError, TypeError):
        return None


def _parse_int(row: dict[str, str], *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (ValueError, TypeError):
            continue
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <trace-render.csv>")
        return 1

    span_rows: dict[str, dict[str, str]] = {}
    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip('"')
            if not name:
                continue
            span_rows[name] = row

    if not span_rows:
        print("FAIL: no spans in CSV")
        return 1

    fail = False
    for span, limit in THRESHOLDS_MEAN_US.items():
        row = span_rows.get(span)
        mean = _parse_mean_us(row) if row else None
        if row is None or mean is None:
            print(f"  SKIP: {span} not found")
            continue
        ok = mean <= limit
        tag = "OK" if ok else "FAIL"
        print(f"  {tag}: {span} mean={mean:.0f}µs (limit {limit}µs)")
        if not ok:
            fail = True

    for span in REPORT_SPANS:
        row = span_rows.get(span)
        mean = _parse_mean_us(row) if row else None
        if row is None or mean is None:
            print(f"  INFO: {span} not found")
            continue

        calls = _parse_int(row, "count", "calls", "call_count")
        if calls is None:
            print(f"  INFO: {span} mean={mean:.0f}µs")
        else:
            print(f"  INFO: {span} mean={mean:.0f}µs calls={calls}")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
