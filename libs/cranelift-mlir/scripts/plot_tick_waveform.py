#!/usr/bin/env python3
"""Plot the per-tick wall-time waveform from a cranelift-mlir profile.

Usage:
    plot_tick_waveform.py <profile.json> [--out trace.png]

The input JSON is `$ELODIN_CRANELIFT_DEBUG_DIR/profile.json` from a
debug-mode run. It must contain `main_tick_waveform` (always populated
when debug mode was on). Produces a PNG with the per-tick wall time
as a line chart and horizontal reference lines for mean / p50 / p95
/ p99 / p99.9.

This is a diagnostic helper: cheap way to see whether slow ticks are
periodic (GC pause), first-tick-only (cold cache), or scattered
outliers (OS preemption).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "matplotlib is required. Install with `uv pip install matplotlib`",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", type=Path, help="Profile JSON with waveform")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <profile>.png)",
    )
    ap.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every Nth sample (useful for >100k-tick captures)",
    )
    args = ap.parse_args()

    try:
        data = json.loads(args.profile.read_text())
    except Exception as e:
        print(f"failed to read {args.profile}: {e}", file=sys.stderr)
        return 1

    waveform = data.get("main_tick_waveform")
    if not waveform:
        print(
            "profile has no `main_tick_waveform`. "
            "Re-capture with `ELODIN_CRANELIFT_DEBUG_DIR=/path` set.",
            file=sys.stderr,
        )
        return 2

    if args.downsample > 1:
        waveform = waveform[:: args.downsample]

    ticks = list(range(len(waveform)))
    # Convert ns to µs for a friendlier y-axis.
    us = [ns / 1e3 for ns in waveform]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ticks, us, linewidth=0.5, alpha=0.8, label="per-tick us")

    # Overlay percentiles from `functions[]` if main is present.
    main_fn = next((f for f in data.get("functions", []) if f.get("name") == "main"), None)
    if main_fn:
        for label, key, color in (
            ("p50", "p50_ns", "green"),
            ("p95", "p95_ns", "orange"),
            ("p99", "p99_ns", "red"),
        ):
            v = main_fn.get(key)
            if v:
                ax.axhline(
                    v / 1e3,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"{label} = {v / 1e3:.1f} us",
                )

    mean_us = sum(us) / max(len(us), 1)
    ax.axhline(mean_us, color="blue", linestyle=":", linewidth=1, label=f"mean = {mean_us:.1f} us")

    ax.set_xlabel("tick")
    ax.set_ylabel("wall time (us)")
    ax.set_title(
        f"cranelift-mlir per-tick waveform ({len(waveform)} ticks"
        + (f", downsampled x{args.downsample}" if args.downsample > 1 else "")
        + ")"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out = args.out or args.profile.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
