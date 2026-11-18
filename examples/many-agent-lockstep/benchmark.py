#!/usr/bin/env python3
"""Run the many-agent example over a grid of sizes and modes, and save the timing of each run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
CONTROLLER_BINARY = EXAMPLE / "controller/build/release/many-agent-controller"
HOOK_LIBRARY = EXAMPLE / "hook/build/release/libmany_agent_hook.so"
PHASES = ("copy_db_to_world", "world_run", "pre_commit", "commit", "total_cycle")


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1,2,4,8,10,12,16,20,24")
    parser.add_argument("--modes", default="runtime,local,lockstep")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-ticks", type=int, default=2000)
    parser.add_argument("--measured-ticks", type=int, default=10000)
    parser.add_argument("--aux-components", type=int, default=32)
    parser.add_argument("--compute-rounds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hook-workers", type=int, default=2)
    parser.add_argument("--hook", choices=("native", "python"), default="native")
    parser.add_argument("--paced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--elodin-bin", default="elodin")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_output/many-agent-lockstep"),
    )
    return parser.parse_args()


def metrics(summary: dict) -> dict[str, float | int]:
    cycles = int(summary["cycles"])
    values: dict[str, float | int] = {
        "cycles": cycles,
        "effective_hz": cycles * 1.0e9 / max(int(summary["wall_ns"]), 1),
        "late_fraction": int(summary["pacing_behind_events"]) / max(cycles, 1),
        "pacing_resets": int(summary["pacing_resets"]),
    }
    for phase in PHASES:
        stats = summary[phase]
        values[f"{phase}_mean_us"] = stats["sum_ns"] / max(stats["count"], 1) / 1000.0
        values[f"{phase}_p95_us"] = stats["p95_ns"] / 1000.0
        values[f"{phase}_max_us"] = stats["max_ns"] / 1000.0
    return values


def build_crate(artifact: Path) -> None:
    """Builds the example crate that `artifact` is the release output of."""
    crate = artifact.parents[2]
    env = os.environ.copy()
    env["CARGO_" + "TAR" + "GET_DIR"] = str(crate / "build")
    result = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(crate / "Cargo.toml"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0 or not artifact.is_file():
        raise RuntimeError(f"{crate.name} build failed")


def run_case(
    args: argparse.Namespace,
    *,
    size: int,
    mode: str,
    repeat: int,
) -> dict:
    case_dir = args.output / f"{mode}-{size:02d}-run-{repeat:02d}"
    shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True)
    summary_path = case_dir / "summary.json"
    result_path = case_dir / "result.json"
    log_path = case_dir / "run.log"
    env = os.environ.copy()
    env.update(
        {
            "ELODIN_BACKEND": "cranelift",
            "ELODIN_DB_PATH": str(case_dir / "db"),
            "ELODIN_MANY_AGENT_AGENTS": str(size),
            "ELODIN_MANY_AGENT_OBJECTS": str(size),
            "ELODIN_MANY_AGENT_MAX_TICKS": str(args.warmup_ticks + args.measured_ticks),
            "ELODIN_MANY_AGENT_MODE": mode,
            "ELODIN_MANY_AGENT_HOOK": args.hook,
            "ELODIN_MANY_AGENT_REALTIME": "1" if args.paced else "0",
            "ELODIN_MANY_AGENT_RESULT_JSON": str(result_path),
            "ELODIN_MANY_AGENT_CONTROLLER_BIN": str(CONTROLLER_BINARY),
            "ELODIN_MANY_AGENT_HOOK_LIB": str(HOOK_LIBRARY),
            "ELODIN_MANY_AGENT_PORT_BASE": str(13000 + repeat * 256),
            "ELODIN_SIM_WARMUP_TICKS": str(args.warmup_ticks),
            "ELODIN_SIM_SUMMARY_JSON": str(summary_path),
            "ELODIN_MANY_AGENT_AUX_COMPONENTS": str(args.aux_components),
            "ELODIN_MANY_AGENT_COMPUTE_ROUNDS": str(args.compute_rounds),
            "ELODIN_MANY_AGENT_WORKERS": str(args.workers),
            "ELODIN_MANY_AGENT_HOOK_WORKERS": str(args.hook_workers),
        }
    )
    if args.trace:
        env["ELODIN_DETAILED_TIMING"] = "1"
        env["ELODIN_PACING_TRACE"] = str(case_dir / "pacing.csv")
        env["ELODIN_PACING_TRACE_MAX"] = str(args.warmup_ticks + args.measured_ticks + 100)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [args.elodin_bin, "run", str(EXAMPLE / "main.py")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    wall_s = time.perf_counter() - started
    if process.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"case failed: {case_dir}; see {log_path}")
    summary = json.loads(summary_path.read_text())
    run = {
        "configuration": {
            "agents": size,
            "objects": size,
            "mode": mode,
            "hook": args.hook,
            "warmup_ticks": args.warmup_ticks,
            "measured_ticks": args.measured_ticks,
            "paced": args.paced,
            "repeat": repeat,
            "aux_components": args.aux_components,
            "compute_rounds": args.compute_rounds,
            "workers": args.workers,
            "hook_workers": args.hook_workers,
        },
        "metrics": metrics(summary),
        "sync": summary.get("sync"),
        "backend": summary.get("backend"),
        "callbacks": summary.get("callbacks"),
        "wall_s": wall_s,
        "output_digest": (
            json.loads(result_path.read_text())["sha256"] if result_path.is_file() else None
        ),
    }
    (case_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return run


def mismatched_sizes(runs: list[dict]) -> list[int]:
    """Returns the agent counts whose runs ended in different states.

    Every repeat of the `local` and `lockstep` modes must end in the same state, because the
    controllers compute the same commands as the Python hook.
    """
    digests: dict[int, set[str]] = {}
    for run in runs:
        if run["output_digest"]:
            digests.setdefault(run["configuration"]["agents"], set()).add(run["output_digest"])
    return sorted(size for size, found in digests.items() if len(found) > 1)


def main() -> int:
    args = parse_args()
    sizes = [int(item) for item in csv_values(args.sizes)]
    modes = csv_values(args.modes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("sizes must be positive")
    if not modes or any(mode not in {"runtime", "local", "lockstep"} for mode in modes):
        raise ValueError("invalid mode")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if "lockstep" in modes:
        build_crate(CONTROLLER_BINARY)
        if args.hook == "native":
            build_crate(HOOK_LIBRARY)
    runs = []
    for repeat in range(1, args.repeats + 1):
        ordered_sizes = sizes if repeat % 2 else list(reversed(sizes))
        for mode in modes:
            for size in ordered_sizes:
                run = run_case(args, size=size, mode=mode, repeat=repeat)
                runs.append(run)
                print(
                    mode,
                    size,
                    f"{run['metrics']['effective_hz']:.1f} Hz",
                    f"{run['metrics']['total_cycle_mean_us']:.1f} us/cycle",
                    flush=True,
                )
    mismatches = mismatched_sizes(runs)
    matrix = {
        "host": {
            "architecture": platform.machine(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
        },
        "mismatched_sizes": mismatches,
        "runs": runs,
    }
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for size in mismatches:
        print(f"runs with {size} agents ended in different states")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
