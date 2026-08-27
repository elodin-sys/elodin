#!/usr/bin/env python3
"""Compare a rendered white-hot RGBA frame with Boson flight-data ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .extract_frames import frame_metrics
except ImportError:
    from extract_frames import frame_metrics

REFERENCE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "p01_dn": (0.0, 25.0),
    # Reference flight frames are steep no-sky views (p50 86-91); rendered
    # validation frames keep the horizon in view, so cold sky drags the
    # whole-frame median lower. Latched/broken frames measure ~4 or ~250.
    "p50_dn": (40.0, 170.0),
    "p99_dn": (200.0, 255.0),
    "mean_gradient_dn": (6.0, 30.0),
    "edge_fraction": (0.03, 0.25),
    "laplacian_rms_dn": (8.0, 40.0),
}


def rgba_metrics(data: bytes, width: int, height: int) -> dict[str, float | int]:
    expected = width * height * 4
    if len(data) != expected:
        raise ValueError(f"expected {expected} RGBA bytes, got {len(data)}")
    white_hot = bytes(data[index] for index in range(0, len(data), 4))
    return frame_metrics(white_hot, width, height)


def evaluate_metrics(metrics: dict[str, float | int]) -> list[str]:
    failures = []
    for name, (minimum, maximum) in REFERENCE_THRESHOLDS.items():
        value = float(metrics[name])
        if not minimum <= value <= maximum:
            failures.append(f"{name}={value:.4g} outside [{minimum:.4g}, {maximum:.4g}]")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgba", type=Path, help="raw RGBA8 frame")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output", type=Path, help="optional metrics JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = rgba_metrics(args.rgba.read_bytes(), args.width, args.height)
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    failures = evaluate_metrics(metrics)
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print("PASS: rendered frame is within Boson reference ranges")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
