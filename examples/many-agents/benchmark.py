#!/usr/bin/env python3
"""Run the many-agent example over a grid of sizes and modes, and save the timing of each run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
CONTROLLER_BINARY = EXAMPLE / "controller/build/release/many-agent-controller"
HOOK_LIBRARY = (
    EXAMPLE
    / "hook/build/release"
    / ("libmany_agent_hook.dylib" if sys.platform == "darwin" else "libmany_agent_hook.so")
)
PHASES = (
    "copy_db_to_world",
    "world_run",
    "pre_commit",
    "commit",
    "deferred_writes",
    "callback",
    "total_cycle",
)
MODES = ("runtime", "local", "lockstep", "lockstep-history")
# Receipt counters of the history mode's reads and deferred writes.
COUNTERS = ("history_queries", "history_misses", "deferred_rows")


def free_ports(count: int, kind: int = socket.SOCK_STREAM) -> int:
    """A port P with P up to P + count - 1 free on localhost, for TCP or `socket.SOCK_DGRAM`. A DB
    takes three TCP ports in a row, for itself and its asset and gRPC servers, and the agents two
    UDP ports each. The ports are below every OS's ephemeral range, which macOS hands out in
    order, starting next to the last one."""
    while True:
        port = random.randrange(20000, 32000)
        try:
            for offset in range(count):
                with socket.socket(type=kind) as probe:
                    probe.bind(("127.0.0.1", port + offset))
        except OSError:
            continue
        return port


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1,2,4,8,10,12,16,20,24")
    parser.add_argument("--modes", default=",".join(MODES))
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
    parser.add_argument(
        "--witness",
        nargs="?",
        const="digest",
        metavar="raw|START:STOP",
        help="record each run's witness; raw keeps every tick's exact bytes, START:STOP those ticks'",
    )
    parser.add_argument(
        "--compare-witness", type=Path, help="compare each run with this witness as it runs"
    )
    parser.add_argument(
        "--replay-witness",
        action="store_true",
        help="replay each run's hook from its witness, which needs exact bytes",
    )
    parser.add_argument("--fault-mode", choices=("bit-flip", "completion-race", "causal-token"))
    parser.add_argument(
        "--assert-bitwise",
        action="store_true",
        help="fail unless every repeat of a mode and size records the same witness; the first "
        "keeps every tick's exact bytes, and later repeats compare with it as they run",
    )
    parser.add_argument("--elodin-bin", default="elodin")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profile_output/many-agents"),
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
    for counter in COUNTERS:
        values[counter] = (summary.get("callbacks") or {}).get(counter, 0)
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


def replay(case_dir: Path, env: dict[str, str]) -> dict:
    """Replays the run's hooks from its witness, without the simulation or the controllers."""
    env = {
        **env,
        "ELODIN_MANY_AGENT_REPLAY": str(case_dir / "run.witness"),
        "ELODIN_MANY_AGENT_RESULT_JSON": str(case_dir / "replay.json"),
    }
    with (case_dir / "replay.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, str(EXAMPLE / "main.py")],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = case_dir / "replay.json"
    if process.returncode not in (0, 1) or not result.is_file():
        raise RuntimeError(f"replay failed: {case_dir}; see replay.log")
    return json.loads(result.read_text())


def run_case(
    args: argparse.Namespace,
    *,
    size: int,
    mode: str,
    repeat: int,
    reference: Path | None = None,
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
            "ELODIN_MANY_AGENT_PORT_BASE": str(free_ports(2 * size, socket.SOCK_DGRAM)),
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
    if args.witness:
        env["ELODIN_MANY_AGENT_WITNESS_PATH"] = str(case_dir / "run.witness")
        env["ELODIN_MANY_AGENT_WITNESS_RAW"] = {"digest": "0", "raw": "1"}.get(
            args.witness, args.witness
        )
        if reference:
            env["ELODIN_MANY_AGENT_WITNESS_REFERENCE"] = str(reference)
        elif args.assert_bitwise:
            # The reference of later repeats keeps exact bytes, so they can name the byte.
            env["ELODIN_MANY_AGENT_WITNESS_RAW"] = "1"
    if args.fault_mode:
        env["ELODIN_MANY_AGENT_FAULT_MODE"] = args.fault_mode
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
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
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
        "output_digest": result.get("sha256"),
        "frames": result.get("frames"),
        "witness": summary.get("witness"),
    }
    if args.replay_witness:
        run["replay"] = replay(case_dir, env)
    (case_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return run


def mismatched_sizes(runs: list[dict]) -> list[int]:
    """Returns the agent counts whose runs ended in different states.

    Every repeat of the `local`, `lockstep` and `lockstep-history` modes must end in the same
    state, because the controllers compute the same commands as the Python hook, and the camera
    only watches.
    """
    digests: dict[int, set[str]] = {}
    for run in runs:
        if run["output_digest"]:
            digests.setdefault(run["configuration"]["agents"], set()).add(run["output_digest"])
    return sorted(size for size, found in digests.items() if len(found) > 1)


def mismatched_witnesses(runs: list[dict]) -> list[tuple[str, int]]:
    """Returns the modes and agent counts whose repeats recorded different witnesses."""
    roots: dict[tuple[str, int], set[str | None]] = {}
    for run in runs:
        if run["witness"]:
            key = (run["configuration"]["mode"], run["configuration"]["agents"])
            roots.setdefault(key, set()).add(run["witness"]["root"])
    return sorted(key for key, found in roots.items() if len(found) > 1)


def main() -> int:
    args = parse_args()
    sizes = [int(item) for item in csv_values(args.sizes)]
    modes = csv_values(args.modes)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("sizes must be positive")
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError("invalid mode")
    if (args.assert_bitwise or args.replay_witness or args.compare_witness) and not args.witness:
        raise ValueError("--assert-bitwise, --replay-witness and --compare-witness need --witness")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(mode.startswith("lockstep") for mode in modes):
        build_crate(CONTROLLER_BINARY)
        if args.hook == "native":
            build_crate(HOOK_LIBRARY)
    runs = []
    # The first repeat's witness of each mode and size, which later repeats compare with.
    references: dict[tuple[str, int], Path] = {}
    for repeat in range(1, args.repeats + 1):
        ordered_sizes = sizes if repeat % 2 else list(reversed(sizes))
        for mode in modes:
            for size in ordered_sizes:
                reference = args.compare_witness or references.get((mode, size))
                run = run_case(args, size=size, mode=mode, repeat=repeat, reference=reference)
                runs.append(run)
                if args.assert_bitwise and run["witness"]:
                    references.setdefault((mode, size), Path(run["witness"]["path"]))
                witness = run["witness"] or {}
                print(
                    mode,
                    size,
                    f"{run['metrics']['effective_hz']:.1f} Hz",
                    f"{run['metrics']['total_cycle_mean_us']:.1f} us/cycle",
                    *([f"witness {witness['root'][:16]}"] if witness.get("root") else []),
                    *(
                        [f"first mismatch at tick {witness['first_mismatch']['tick']}"]
                        if witness.get("first_mismatch")
                        else []
                    ),
                    flush=True,
                )
    mismatches = mismatched_sizes(runs)
    witness_mismatches = mismatched_witnesses(runs) if args.assert_bitwise else []
    replay_mismatches = [run for run in runs if (run.get("replay") or {}).get("first_mismatch")]
    matrix = {
        "host": {
            "architecture": platform.machine(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
        },
        "mismatched_sizes": mismatches,
        "witness_mismatches": witness_mismatches,
        "runs": runs,
    }
    (args.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for size in mismatches:
        print(f"runs with {size} agents ended in different states")
    if witness_mismatches or replay_mismatches:
        from elodin.determinism import describe

        for mode, size in witness_mismatches:
            print(f"{mode} runs with {size} agents recorded different witnesses")
            group = [
                run
                for run in runs
                if (run["configuration"]["mode"], run["configuration"]["agents"]) == (mode, size)
            ]
            for run in group:
                if run["witness"]["first_mismatch"]:
                    print(f"repeat {run['configuration']['repeat']}:")
                    print(describe(run["witness"]["first_mismatch"]))
        for run in replay_mismatches:
            print(f"the replay of {run['witness']['path']} differs from the run:")
            print(describe(run["replay"]["first_mismatch"]))
    return 1 if mismatches or witness_mismatches or replay_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
