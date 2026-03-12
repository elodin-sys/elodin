#!/usr/bin/env python3
"""
Summarize sensor-camera performance from an elodin run log.

Prints:
 - PERF interpretation against real-time tick budget
 - Distribution stats for total_request_ms and total_render_client_ms
 - Explicit 5-8ms goal checks with clear verdicts
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([0-9]+(?:\.[0-9]+)?)")
REQ_RE = re.compile(r"total_request_ms=([0-9]+(?:\.[0-9]+)?)")
CLI_RE = re.compile(r"total_render_client_ms=([0-9]+(?:\.[0-9]+)?)")
CAM_RE = re.compile(r"camera_count=(\d+)")
ENQ_RE = re.compile(r"db_enqueue_count=(\d+)")


@dataclass
class DistStats:
    count: int
    minimum: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float
    lt5_pct: float
    lt8_pct: float
    le8_pct: float


def pct_le(vals: list[float], threshold: float) -> float:
    if not vals:
        return 0.0
    return 100.0 * sum(1 for v in vals if v <= threshold) / len(vals)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def compute_dist(vals: list[float]) -> DistStats | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return DistStats(
        count=n,
        minimum=s[0],
        mean=sum(s) / n,
        p50=percentile(s, 0.50),
        p90=percentile(s, 0.90),
        p95=percentile(s, 0.95),
        p99=percentile(s, 0.99),
        maximum=s[-1],
        lt5_pct=100.0 * sum(1 for v in vals if v < 5.0) / n,
        lt8_pct=100.0 * sum(1 for v in vals if v < 8.0) / n,
        le8_pct=100.0 * sum(1 for v in vals if v <= 8.0) / n,
    )


def fmt_dist(title: str, vals: list[float]) -> None:
    stats = compute_dist(vals)
    if stats is None:
        print(f"{title}: n=0")
        return
    print(
        f"{title}: n={stats.count} mean={stats.mean:.3f} "
        f"p50={stats.p50:.3f} p90={stats.p90:.3f} p95={stats.p95:.3f} "
        f"p99={stats.p99:.3f} min={stats.minimum:.3f} max={stats.maximum:.3f}"
    )
    print(
        f"  <5ms={stats.lt5_pct:.2f}%  <8ms={stats.lt8_pct:.2f}%  <=8ms={stats.le8_pct:.2f}%"
    )


def parse_perf_line(line: str) -> dict[str, float] | None:
    if not line.startswith("PERF sensor_camera "):
        return None
    data: dict[str, float] = {}
    for key, value in KV_RE.findall(line):
        try:
            data[key] = float(value)
        except ValueError:
            continue
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    parser.add_argument(
        "--sim-time-step",
        type=float,
        default=1.0 / 120.0,
        help="Simulation time step in seconds used to compute real-time budget.",
    )
    parser.add_argument(
        "--goal-low-ms",
        type=float,
        default=5.0,
        help="Lower goal bound in ms (informational target).",
    )
    parser.add_argument(
        "--goal-ms",
        type=float,
        default=8.0,
        help="Critical budget in ms for <= checks.",
    )
    parser.add_argument(
        "--goal-p95-ms",
        type=float,
        default=8.0,
        help="Target p95 in ms for render-path checks.",
    )
    parser.add_argument(
        "--goal-under-pct",
        type=float,
        default=90.0,
        help="Target percentage of samples <= goal-ms for render-path checks.",
    )
    args = parser.parse_args()

    perf_entries: list[dict[str, float]] = []
    total_request: list[float] = []
    total_render_client: list[float] = []
    total_request_by_camera: dict[int, list[float]] = defaultdict(list)
    total_render_client_by_enqueue: dict[int, list[float]] = defaultdict(list)

    with open(args.log_file, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw).strip()
            perf = parse_perf_line(line)
            if perf is not None:
                perf_entries.append(perf)

            req_m = REQ_RE.search(line)
            if req_m:
                req = float(req_m.group(1))
                total_request.append(req)
                cam_m = CAM_RE.search(line)
                if cam_m:
                    total_request_by_camera[int(cam_m.group(1))].append(req)

            cli_m = CLI_RE.search(line)
            if cli_m:
                cli = float(cli_m.group(1))
                total_render_client.append(cli)
                enq_m = ENQ_RE.search(line)
                if enq_m:
                    total_render_client_by_enqueue[int(enq_m.group(1))].append(cli)

    if not perf_entries:
        print("PERF summary: missing PERF sensor_camera line")
        return 1

    perf = perf_entries[-1]
    sim_time_step = args.sim_time_step
    budget_ms = sim_time_step * 1000.0
    tick_hz = (1.0 / sim_time_step) if sim_time_step > 0 else math.nan

    max_ticks = perf.get("max_ticks")
    elapsed_s = perf.get("elapsed_s")
    rtf = perf.get("rtf")
    goal_low_ms = args.goal_low_ms
    goal_ms = args.goal_ms

    avg_tick_from_elapsed_ms = None
    if max_ticks and elapsed_s and max_ticks > 0:
        avg_tick_from_elapsed_ms = (elapsed_s * 1000.0) / max_ticks

    avg_tick_from_rtf_ms = None
    if rtf and rtf > 0:
        avg_tick_from_rtf_ms = budget_ms / rtf

    print("=== PERF Budget Interpretation ===")
    print(f"Tick rate: {tick_hz:.2f} Hz  -> real-time budget={budget_ms:.3f} ms/tick")
    if rtf is not None and avg_tick_from_rtf_ms is not None:
        print(
            f"From rtf={rtf:.3f}: estimated wall tick ~= {budget_ms:.3f}/{rtf:.3f} = "
            f"{avg_tick_from_rtf_ms:.3f} ms"
        )
    if avg_tick_from_elapsed_ms is not None:
        print(
            f"From elapsed/max_ticks: {elapsed_s:.3f}s/{int(max_ticks)} = "
            f"{avg_tick_from_elapsed_ms:.3f} ms/tick"
        )
        if avg_tick_from_elapsed_ms <= goal_low_ms:
            print(f"Budget status: within {goal_low_ms:.1f}ms target")
        elif avg_tick_from_elapsed_ms <= goal_ms:
            print(
                f"Budget status: above {goal_low_ms:.1f}ms target, "
                f"within {goal_ms:.1f}ms critical budget"
            )
        else:
            print(f"Budget status: above {goal_ms:.1f}ms critical budget")

    print("\n=== Render Path Distributions ===")
    fmt_dist("total_request_ms (all)", total_request)
    for cam_count in sorted(total_request_by_camera):
        fmt_dist(
            f"total_request_ms (camera_count={cam_count})",
            total_request_by_camera[cam_count],
        )

    fmt_dist("total_render_client_ms (all)", total_render_client)
    for enq_count in sorted(total_render_client_by_enqueue):
        fmt_dist(
            f"total_render_client_ms (db_enqueue_count={enq_count})",
            total_render_client_by_enqueue[enq_count],
        )

    print("\n=== 5-8ms Goal Check ===")
    if avg_tick_from_elapsed_ms is not None:
        tick_ok = avg_tick_from_elapsed_ms <= goal_ms
        tick_status = "OK" if tick_ok else "KO"
        print(
            f"system_tick_ms: {avg_tick_from_elapsed_ms:.3f} "
            f"(target <= {goal_ms:.3f}) -> {tick_status}"
        )
    else:
        print("system_tick_ms: unavailable -> UNKNOWN")

    def print_goal_line(name: str, vals: list[float]) -> str:
        stats = compute_dist(vals)
        if stats is None:
            print(f"{name}: no samples -> UNKNOWN")
            return "UNKNOWN"
        le_goal_pct = pct_le(vals, goal_ms)
        p95_ok = stats.p95 <= args.goal_p95_ms
        pct_ok = le_goal_pct >= args.goal_under_pct
        status = "OK" if p95_ok and pct_ok else "KO"
        print(
            f"{name}: p95={stats.p95:.3f}ms (<= {args.goal_p95_ms:.3f}), "
            f"<= {goal_ms:.3f}ms={le_goal_pct:.2f}% (>= {args.goal_under_pct:.2f}%) -> {status}"
        )
        return status

    req_status = print_goal_line("total_request_ms", total_request)
    cli_status = print_goal_line("total_render_client_ms", total_render_client)

    statuses = [s for s in [req_status, cli_status] if s != "UNKNOWN"]
    if not statuses:
        overall = "UNKNOWN"
    elif all(s == "OK" for s in statuses):
        overall = "OK"
    else:
        overall = "KO"
    print(f"overall_render_path: {overall}")

    if not total_request and not total_render_client:
        print(
            "\nnote: no total_*_ms probe lines found in log "
            "(set ELODIN_SENSOR_CAMERA_LOG_METRICS=1 when running)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
