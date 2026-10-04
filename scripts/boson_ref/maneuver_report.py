#!/usr/bin/env python3
"""Detectors and reporting for the LWIR maneuver harness.

Pure functions over grayscale captures so thresholds are unit-testable and
reusable. Failure modes covered:

- sky band: a bright horizontal stripe in sky-only frames (horizon assumed at
  image center instead of the world horizon)
- AGC latch: ground frames after sky exposure stay crushed near black and
  never recover
- liveness: the renderer silently stops producing new frames
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .extract_frames import frame_metrics, write_grayscale_png
except ImportError:
    from extract_frames import frame_metrics, write_grayscale_png

BAND_SCORE_MAX_DN = 40.0
RECOVERY_MEAN_TOLERANCE_DN = 25.0
RECOVERY_EDGE_FRACTION_MIN_RATIO = 0.5
RECOVERY_NEAR_BLACK_MAX_FRACTION = 0.60
RECOVERY_P99_MIN_DN = 120
NEAR_BLACK_DN = 2

MONTAGE_COLUMNS = 5
MONTAGE_SEPARATOR_PX = 4


def row_band_score(gray: np.ndarray, width: int, height: int) -> float:
    """Peak row brightness above the typical row: high for a horizontal band."""
    rows = gray.reshape(height, width).mean(axis=1)
    return float(rows.max() - np.median(rows))


def near_black_fraction(gray: np.ndarray) -> float:
    return float((gray <= NEAR_BLACK_DN).mean())


def capture_report(capture: dict, width: int, height: int) -> dict:
    gray = capture["gray"]
    metrics = frame_metrics(gray.tobytes(), width, height)
    metrics["band_score_dn"] = round(row_band_score(gray, width, height), 3)
    metrics["near_black_fraction"] = round(near_black_fraction(gray), 6)
    return {"label": capture["label"], "t": capture["t"], "path": capture["path"], **metrics}


def evaluate_captures(captures: list[dict], width: int, height: int) -> tuple[dict, list[str]]:
    reports = [capture_report(capture, width, height) for capture in captures]
    by_label = {report["label"]: report for report in reports}
    failures: list[str] = []

    for label in ("sky_hold_a", "sky_hold_b"):
        report = by_label.get(label)
        if report is None:
            continue
        if report["band_score_dn"] > BAND_SCORE_MAX_DN:
            failures.append(
                f"sky band: {label} row-band score {report['band_score_dn']:.1f} DN "
                f"exceeds {BAND_SCORE_MAX_DN:.0f} DN"
            )

    baseline = by_label.get("ground_initial_b") or by_label.get("ground_initial_a")
    final = by_label.get("ground_return_c")
    if baseline is not None and final is not None:
        mean_delta = abs(final["mean_dn"] - baseline["mean_dn"])
        if mean_delta > RECOVERY_MEAN_TOLERANCE_DN:
            failures.append(
                f"agc recovery: ground_return mean {final['mean_dn']:.1f} DN vs "
                f"initial {baseline['mean_dn']:.1f} DN (|delta| "
                f"{mean_delta:.1f} > {RECOVERY_MEAN_TOLERANCE_DN:.0f})"
            )
        edge_floor = RECOVERY_EDGE_FRACTION_MIN_RATIO * baseline["edge_fraction"]
        if final["edge_fraction"] < edge_floor:
            failures.append(
                f"agc recovery: ground_return edge_fraction {final['edge_fraction']:.4f} "
                f"below {RECOVERY_EDGE_FRACTION_MIN_RATIO:.0%} of initial "
                f"{baseline['edge_fraction']:.4f}"
            )
        if final["near_black_fraction"] > RECOVERY_NEAR_BLACK_MAX_FRACTION:
            failures.append(
                "agc recovery: ground_return near-black fraction "
                f"{final['near_black_fraction']:.2f} exceeds "
                f"{RECOVERY_NEAR_BLACK_MAX_FRACTION:.2f}"
            )
        if final["p99_dn"] < RECOVERY_P99_MIN_DN:
            failures.append(
                f"agc recovery: ground_return p99 {final['p99_dn']} DN below "
                f"{RECOVERY_P99_MIN_DN} DN"
            )

    for earlier, later in zip(captures, captures[1:], strict=False):
        if np.array_equal(earlier["gray"], later["gray"]):
            failures.append(f"liveness: {later['label']} is byte-identical to {earlier['label']}")

    return {"captures": reports, "failures": failures}, failures


def write_montage(captures: list[dict], width: int, height: int, path: Path) -> None:
    """Chronological grid of captures with white separators (legend in JSON)."""
    if not captures:
        return
    columns = min(MONTAGE_COLUMNS, len(captures))
    rows = (len(captures) + columns - 1) // columns
    sep = MONTAGE_SEPARATOR_PX
    total_w = columns * width + (columns - 1) * sep
    total_h = rows * height + (rows - 1) * sep
    canvas = np.full((total_h, total_w), 255, dtype=np.uint8)
    for index, capture in enumerate(captures):
        row, col = divmod(index, columns)
        y = row * (height + sep)
        x = col * (width + sep)
        canvas[y : y + height, x : x + width] = capture["gray"].reshape(height, width)
    write_grayscale_png(path, canvas.tobytes(), total_w, total_h)
