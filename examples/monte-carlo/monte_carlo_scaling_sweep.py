#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Monte Carlo scaling sweep.")
    parser.add_argument("--sim", type=Path, default=Path("examples/monte-carlo/main.py"))
    parser.add_argument("--campaign", type=Path, default=Path("examples/monte-carlo/campaign.toml"))
    parser.add_argument("--plan", type=Path, default=Path("examples/monte-carlo/plan.csv"))
    parser.add_argument("--out", type=Path, default=Path("dbs/monte-carlo-scaling"))
    parser.add_argument("--workers", default="1,2,5,8,11,15,20,22,30")
    parser.add_argument("--grid-size", default="4096")
    parser.add_argument("--probe-rows", default="0")
    parser.add_argument("--controllers", default="0,1")
    parser.add_argument("--runtime-threads", default="0")
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def run_case(
    *,
    sim: Path,
    campaign: Path,
    plan: Path,
    out_dir: Path,
    workers: int,
    grid_size: str,
    probe_rows: int,
    controller: int,
    runtime_threads: int,
    memory_probe: bool,
) -> dict[str, object]:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ELODIN_MONTE_CARLO_GRID_SIZE"] = grid_size
    env["ELODIN_MONTE_CARLO_PROBE_ROWS"] = str(probe_rows)
    env["ELODIN_MONTE_CARLO_CONTROLLER"] = str(controller)
    cmd = [
        "elodin",
        "monte-carlo",
        "run",
        str(sim),
        "--campaign",
        str(campaign),
        "--plan",
        str(plan),
        "--workers",
        str(workers),
        "--out",
        str(out_dir),
    ]
    if runtime_threads > 1:
        cmd.extend(["--runtime-threads", str(runtime_threads)])
    if memory_probe:
        cmd.append("--memory-probe")
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    (out_dir / "sweep.stdout.log").write_text(completed.stdout)
    (out_dir / "sweep.stderr.log").write_text(completed.stderr)
    summary = (
        json.loads((out_dir / "summary.json").read_text()) if completed.returncode == 0 else {}
    )
    sim_phase = summary.get("sim_phase_summary") or {}
    attribution = summary.get("phase_attribution") or {}
    resource = summary.get("resource_summary") or {}
    concurrency = summary.get("concurrency_summary") or {}
    return {
        "workers": workers,
        "probe_rows": probe_rows,
        "controller": controller,
        "runtime_threads": runtime_threads,
        "memory_probe": int(memory_probe),
        "returncode": completed.returncode,
        "passed": summary.get("passed", ""),
        "failed": summary.get("failed", ""),
        "wall_ms": summary.get("wall_ms", ""),
        "average_run_wall_ms": summary.get("average_run_wall_ms", ""),
        "parallel_efficiency": summary.get("parallel_efficiency", ""),
        "mean_active_runs": concurrency.get("mean_active_runs", ""),
        "peak_active_runs": concurrency.get("peak_active_runs", ""),
        "tick_mean_ns": (sim_phase.get("world_run") or {}).get("sum_ns", 0)
        / max((sim_phase.get("world_run") or {}).get("count", 1), 1),
        "avg_python_import_ms": attribution.get("average_python_import_ms", ""),
        "avg_compile_ms": attribution.get("average_compile_ms", ""),
        "avg_loop_ms": attribution.get("average_loop_ms", ""),
        "avg_teardown_ms": attribution.get("average_teardown_ms", ""),
        "avg_process_shutdown_ms": attribution.get("average_process_shutdown_ms", ""),
        "avg_cpu_percent": resource.get("average_cpu_percent", ""),
        "peak_cpu_percent": resource.get("peak_cpu_percent", ""),
        "peak_cpu_core_percent": resource.get("peak_cpu_core_percent", ""),
        "peak_load_average_1m": resource.get("peak_load_average_1m", ""),
        "peak_context_switches_per_sec": resource.get("peak_context_switches_per_sec", ""),
    }


def main() -> None:
    args = parse_args()
    workers = parse_csv_ints(args.workers)
    probe_rows = parse_csv_ints(args.probe_rows)
    controllers = parse_csv_ints(args.controllers)
    runtime_threads = parse_csv_ints(args.runtime_threads)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.warmup:
        run_case(
            sim=args.sim,
            campaign=args.campaign,
            plan=args.plan,
            out_dir=args.out / "warmup",
            workers=1,
            grid_size=args.grid_size,
            probe_rows=0,
            controller=1,
            runtime_threads=0,
            memory_probe=args.memory_probe,
        )
    for worker in workers:
        for probe in probe_rows:
            for controller in controllers:
                for runtime_thread in runtime_threads:
                    name = (
                        f"workers-{worker}_probe-{probe}_controller-{controller}"
                        f"_runtime-{runtime_thread}"
                    )
                    row = run_case(
                        sim=args.sim,
                        campaign=args.campaign,
                        plan=args.plan,
                        out_dir=args.out / name,
                        workers=worker,
                        grid_size=args.grid_size,
                        probe_rows=probe,
                        controller=controller,
                        runtime_threads=runtime_thread,
                        memory_probe=args.memory_probe,
                    )
                    rows.append(row)
                    print(row)
    baselines: dict[tuple[int, int, int], float] = {}
    for row in rows:
        key = (
            int(row["probe_rows"]),
            int(row["controller"]),
            int(row["runtime_threads"]),
            int(row["memory_probe"]),
        )
        if int(row["workers"]) == 1 and row["wall_ms"] != "" and row.get("failed") == 0:
            baselines[key] = float(row["wall_ms"])
    for row in rows:
        key = (
            int(row["probe_rows"]),
            int(row["controller"]),
            int(row["runtime_threads"]),
            int(row["memory_probe"]),
        )
        baseline = baselines.get(key)
        if baseline and row["wall_ms"] != "" and row.get("failed") == 0:
            row["speedup_vs_1worker"] = baseline / float(row["wall_ms"])
        else:
            row["speedup_vs_1worker"] = ""
    output = args.out / "scaling.csv"
    if rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
