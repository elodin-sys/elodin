#!/usr/bin/env python3
"""Run the many-agent scaling matrix and write versioned receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
CONTROLLER_BINARY = EXAMPLE / "controller/build/release/many-agent-controller"
HOOK_LIBRARY = EXAMPLE / "hook/build/release/libmany_agent_hook.so"
PHASES = (
    "copy_db_to_world",
    "world_run",
    "pre_commit",
    "commit",
    "post_step",
    "callback",
    "total_cycle",
)
ACCEPTANCE = {
    "effective_hz_min": 995.0,
    "cycle_mean_ms_max": 1.1,
    "world_run_mean_us_max": 400.0,
    "world_run_p95_us_max": 400.0,
    "callback_mean_us_max": 600.0,
    "callback_p95_us_max": 1000.0,
    "late_fraction_max": 0.001,
    "pacing_resets_max": 0,
}


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1,2,4,8,10,12,16,20,24")
    parser.add_argument("--modes", default="runtime,local,lockstep")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-ticks", type=int, default=2000)
    parser.add_argument("--measured-ticks", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-components", type=int, default=32)
    parser.add_argument("--compute-rounds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hook-workers", type=int, default=2)
    parser.add_argument("--paced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--io-mode", choices=("native", "prepared", "batch", "scalar"), default="native"
    )
    parser.add_argument("--elodin-bin", default="elodin")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_output/many-agent-lockstep"),
    )
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--expect",
        type=Path,
        nargs="+",
        default=[],
        help="matrix.json files whose output digests every run with the same workload must reproduce",
    )
    return parser.parse_args()


def phase_mean_ns(summary: dict, name: str) -> float:
    phase = summary[name]
    return float(phase["sum_ns"]) / max(int(phase["count"]), 1)


def phase_p95_ns(summary: dict, name: str) -> float:
    phase = summary[name]
    if "p95_ns" in phase:
        return float(phase["p95_ns"])
    count = int(phase["count"])
    if count == 0:
        return 0.0
    rank = max(1, int(np.ceil(0.95 * count)))
    cumulative = 0
    for index, bucket_count in enumerate(phase["buckets"]):
        previous = cumulative
        cumulative += int(bucket_count)
        if cumulative >= rank:
            lower = 1 << index
            upper = 1 << (index + 1)
            fraction = (rank - previous) / max(int(bucket_count), 1)
            estimate = lower + fraction * (upper - lower)
            return min(
                max(estimate, float(phase["min_ns"])),
                float(phase["max_ns"]),
            )
    return float(phase["max_ns"])


def metrics(summary: dict) -> dict[str, float | int]:
    cycles = int(summary["cycles"])
    wall_ns = max(int(summary["wall_ns"]), 1)
    values: dict[str, float | int] = {
        "cycles": cycles,
        "steps": int(summary["steps"]),
        "effective_hz": cycles * 1.0e9 / wall_ns,
        "late_fraction": int(summary["pacing_behind_events"]) / max(cycles, 1),
        "pacing_resets": int(summary["pacing_resets"]),
    }
    for phase in PHASES:
        values[f"{phase}_mean_us"] = phase_mean_ns(summary, phase) / 1000.0
        values[f"{phase}_p95_us"] = phase_p95_ns(summary, phase) / 1000.0
        values[f"{phase}_max_us"] = float(summary[phase]["max_ns"]) / 1000.0
    values["cycle_mean_ms"] = values["total_cycle_mean_us"] / 1000.0
    return values


def acceptance_failures(values: dict[str, float | int]) -> list[str]:
    checks = (
        ("effective_hz", ">=", ACCEPTANCE["effective_hz_min"]),
        ("cycle_mean_ms", "<=", ACCEPTANCE["cycle_mean_ms_max"]),
        ("world_run_mean_us", "<=", ACCEPTANCE["world_run_mean_us_max"]),
        ("world_run_p95_us", "<=", ACCEPTANCE["world_run_p95_us_max"]),
        ("callback_mean_us", "<=", ACCEPTANCE["callback_mean_us_max"]),
        ("callback_p95_us", "<=", ACCEPTANCE["callback_p95_us_max"]),
        ("late_fraction", "<=", ACCEPTANCE["late_fraction_max"]),
        ("pacing_resets", "<=", ACCEPTANCE["pacing_resets_max"]),
    )
    failures = []
    for name, operator, limit in checks:
        value = values[name]
        passed = value >= limit if operator == ">=" else value <= limit
        if not passed:
            failures.append(f"{name}={value} {operator} {limit}")
    return failures


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def source_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


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
    trace_path = case_dir / "pacing.csv"
    db_path = case_dir / "db"
    env = os.environ.copy()
    env.update(
        {
            "ELODIN_BACKEND": "cranelift",
            "ELODIN_DB_PATH": str(db_path),
            "ELODIN_MANY_AGENT_AGENTS": str(size),
            "ELODIN_MANY_AGENT_OBJECTS": str(size),
            "ELODIN_MANY_AGENT_MAX_TICKS": str(args.warmup_ticks + args.measured_ticks),
            "ELODIN_MANY_AGENT_MODE": mode,
            "ELODIN_MANY_AGENT_IO": args.io_mode,
            "ELODIN_MANY_AGENT_REALTIME": "1" if args.paced else "0",
            "ELODIN_MANY_AGENT_RESULT_JSON": str(result_path),
            "ELODIN_MANY_AGENT_CONTROLLER_BIN": str(CONTROLLER_BINARY),
            "ELODIN_MANY_AGENT_HOOK_LIB": str(HOOK_LIBRARY),
            "ELODIN_MANY_AGENT_PORT_BASE": str(13000 + repeat * 256),
            "ELODIN_SIM_WARMUP_TICKS": str(args.warmup_ticks),
            "ELODIN_SIM_SUMMARY_JSON": str(summary_path),
            "ELODIN_MANY_AGENT_SEED": str(args.seed),
            "ELODIN_MANY_AGENT_AUX_COMPONENTS": str(args.aux_components),
            "ELODIN_MANY_AGENT_COMPUTE_ROUNDS": str(args.compute_rounds),
            "ELODIN_MANY_AGENT_WORKERS": str(args.workers),
            "ELODIN_MANY_AGENT_HOOK_WORKERS": str(args.hook_workers),
        }
    )
    if args.trace:
        env["ELODIN_DETAILED_TIMING"] = "1"
        env["ELODIN_PACING_TRACE"] = str(trace_path)
        env["ELODIN_PACING_TRACE_MAX"] = str(args.warmup_ticks + args.measured_ticks + 100)
    command = [
        args.elodin_bin,
        "run",
        str(EXAMPLE / "main.py"),
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
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
    measured = metrics(summary)
    acceptance_evaluated = size == 24 and args.paced
    failures = acceptance_failures(measured) if acceptance_evaluated else []
    result_data = json.loads(result_path.read_text()) if result_path.is_file() else None
    receipt = {
        "schema": "elodin.many_agent.receipt.v1",
        "source_revision": source_revision(),
        "source_dirty": source_dirty(),
        "configuration": {
            "agents": size,
            "objects": size,
            "mode": mode,
            "io_mode": args.io_mode,
            "rate_hz": 1000,
            "warmup_ticks": args.warmup_ticks,
            "measured_ticks": args.measured_ticks,
            "paced": args.paced,
            "seed": args.seed,
            "repeat": repeat,
            "aux_components": args.aux_components,
            "compute_rounds": args.compute_rounds,
            "workers": args.workers,
            "hook_workers": args.hook_workers,
        },
        "metrics": measured,
        "acceptance": {
            "limits": ACCEPTANCE,
            "failures": failures,
            "evaluated": acceptance_evaluated,
            "passed": not failures if acceptance_evaluated else None,
            "p95_source": summary.get("percentile_source", "histogram"),
        },
        "sync": summary.get("sync"),
        "backend": summary.get("backend"),
        "callbacks": summary.get("callbacks"),
        "wall_s": wall_s,
        "output_digest": result_data["sha256"] if result_data else None,
        "entity_digests": (
            {
                "agents": result_data["agent_sha256"],
                "objects": result_data["object_sha256"],
            }
            if result_data
            else None
        ),
    }
    receipt_path = case_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["artifacts"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (summary_path, result_path, log_path, trace_path)
        if path.is_file()
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def forecast(receipts: list[dict]) -> dict:
    rows = [row for row in receipts if row["configuration"]["mode"] == "lockstep"]
    result = {"schema": "elodin.many_agent.forecast.v1", "phases": {}}
    if len(rows) < 2:
        return result
    sizes = np.asarray([row["configuration"]["agents"] for row in rows], dtype=float)
    for phase in (
        "copy_db_to_world_mean_us",
        "world_run_mean_us",
        "commit_mean_us",
        "callback_mean_us",
        "total_cycle_mean_us",
    ):
        values = np.asarray([row["metrics"][phase] for row in rows], dtype=float)
        slope, intercept = np.polyfit(sizes, values, 1)
        result["phases"][phase] = {
            "slope_us_per_agent": slope,
            "intercept_us": intercept,
            "projected_24_us": slope * 24.0 + intercept,
        }
    result["note"] = (
        "Current-machine projection only; certification requires the configured x86 Linux VM."
    )
    return result


def workload(config: dict) -> tuple:
    return (
        config["agents"],
        config["objects"],
        config["warmup_ticks"] + config["measured_ticks"],
        config["aux_components"],
        config["compute_rounds"],
    )


def expected_digests(matrix_paths: list[Path]) -> dict[tuple, str]:
    return {
        workload(run["configuration"]): run["output_digest"]
        for path in matrix_paths
        for run in json.loads(path.read_text())["runs"]
        if run["output_digest"]
    }


def parity(receipts: list[dict], expected: dict[tuple, str] | None = None) -> dict:
    failures = []
    grouped: dict[tuple[str, int], set[str]] = {}
    by_run = {}
    compared = 0
    for receipt in receipts:
        config = receipt["configuration"]
        digest = receipt["output_digest"]
        if digest is None:
            continue
        grouped.setdefault((config["mode"], config["agents"]), set()).add(digest)
        by_run[(config["mode"], config["agents"], config["repeat"])] = receipt
        if expected and workload(config) in expected:
            compared += 1
            if digest != expected[workload(config)]:
                failures.append(
                    f"digest differs from expected for {config['mode']} {config['agents']}, "
                    f"run {config['repeat']}"
                )
    for key, digests in grouped.items():
        if len(digests) != 1:
            failures.append(f"repeat digest mismatch for {key}")
    repeats = {receipt["configuration"]["repeat"] for receipt in receipts}
    sizes = {receipt["configuration"]["agents"] for receipt in receipts}
    for repeat in repeats:
        for size in sizes:
            local = by_run.get(("local", size, repeat))
            lockstep = by_run.get(("lockstep", size, repeat))
            if local and lockstep and local["output_digest"] != lockstep["output_digest"]:
                failures.append(f"local/lockstep digest mismatch at {size}, run {repeat}")
        full = by_run.get(("lockstep", 24, repeat))
        if full and full["entity_digests"]:
            for size in sizes:
                smaller = by_run.get(("lockstep", size, repeat))
                if not smaller or not smaller["entity_digests"] or size > 24:
                    continue
                if (
                    smaller["entity_digests"]["agents"] != full["entity_digests"]["agents"][:size]
                    or smaller["entity_digests"]["objects"]
                    != full["entity_digests"]["objects"][:size]
                ):
                    failures.append(f"prefix digest mismatch at {size}, run {repeat}")
    return {
        "passed": not failures,
        "failures": failures,
        "policy": "exact SHA-256 equality",
        "expected_digests_compared": compared,
    }


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
        if args.io_mode == "native":
            build_crate(HOOK_LIBRARY)
    receipts = []
    for repeat in range(1, args.repeats + 1):
        ordered_sizes = sizes if repeat % 2 else list(reversed(sizes))
        for mode in modes:
            for size in ordered_sizes:
                receipt = run_case(args, size=size, mode=mode, repeat=repeat)
                receipts.append(receipt)
                print(
                    mode,
                    size,
                    f"{receipt['metrics']['effective_hz']:.1f} Hz",
                    f"{receipt['metrics']['total_cycle_mean_us']:.1f} us/cycle",
                    flush=True,
                )
    parity_result = parity(receipts, expected_digests(args.expect))
    matrix = {
        "schema": "elodin.many_agent.matrix.v1",
        "host": {
            "architecture": platform.machine(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
        },
        "acceptance": ACCEPTANCE,
        "parity": parity_result,
        "runs": receipts,
    }
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (args.output / "forecast.json").write_text(json.dumps(forecast(receipts), indent=2) + "\n")
    failed = not parity_result["passed"] or any(
        not receipt["acceptance"]["passed"]
        for receipt in receipts
        if receipt["configuration"]["agents"] == 24
        and receipt["configuration"]["mode"] == "lockstep"
        and receipt["configuration"]["paced"]
    )
    return 1 if args.require_pass and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
