#!/usr/bin/env python3
"""Measure the real, paced C0 flight; compare non-pacing work, not paced duration."""

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path

STEPS = 120_000
RATE_HZ = 8000
WARMUP = 100
PHASES = (
    "pre_step",
    "copy_db_to_world",
    "world_run",
    "commit",
    "wait_for_write",
    "post_step",
    "real_time_pacing",
    "total_cycle",
)


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"expected positive integer, got {value!r}")
    return value


def measurement(summary, log, db):
    """Reject incomplete runs even when the SDK emitted a shutdown summary."""
    cycles = positive_int(summary["cycles"])
    steps = positive_int(summary["steps"])
    if (
        summary["warmup_excluded_ticks"] != WARMUP
        or steps + WARMUP != STEPS
        or cycles != steps
        or summary["simulation_rate_hz"] != RATE_HZ
        or summary["telemetry_rate_hz"] != RATE_HZ
    ):
        raise ValueError("expected a complete 15s / 8kHz C0 flight with one step per cycle")
    for phase in PHASES:
        stats = summary[phase]
        count = positive_int(stats["count"])
        # Current warmup reset occurs just after pre_step; tolerate either placement.
        minimum = cycles - 1 if phase == "pre_step" else cycles
        if not minimum <= count <= cycles:
            raise ValueError(f"incomplete phase {phase}: {count} samples for {cycles} cycles")
        positive_int(stats["sum_ns"])
    work_ns = summary["total_cycle"]["sum_ns"] - summary["real_time_pacing"]["sum_ns"]
    positive_int(work_ns)
    wall_ns = positive_int(summary["wall_ns"])
    if wall_ns < summary["total_cycle"]["sum_ns"]:
        raise ValueError("wall time shorter than total measured cycle time")

    c0 = re.findall(
        r"^\[C0\] lockstep_steps=(\d+) motor_response=true "
        r"max_motor=([\d.]+) takeoff_delta_m=([\d.]+) status=PASS$",
        log,
        re.MULTILINE,
    )
    if len(c0) != 1:
        raise ValueError("missing or duplicate successful C0 contract")
    lockstep, motor, takeoff = c0[0]
    if not (
        STEPS * 0.99 <= int(lockstep) <= STEPS and float(motor) > 0.06 and float(takeoff) >= 0.1
    ):
        raise ValueError("C0 motor/takeoff/lockstep evidence is incomplete")
    ticks = re.findall(r"^\s*Total ticks: (\d+)$", log, re.MULTILINE)
    sync = re.findall(r"^\s*Sync steps: (\d+)$", log, re.MULTILINE)
    if ticks != [str(STEPS - 1)] or len(sync) != 1 or int(sync[0]) < int(lockstep):
        raise ValueError("missing final tick or inconsistent firmware sync steps")
    if re.search(
        r"post_step error|pre_step error|error ticking world|error committing head|Traceback|"
        r"\[SITL\] Warning: Could not read sensor data:",
        log,
    ):
        raise ValueError("runtime/callback error in flight log")
    if not (db / "db_state").is_file() or (db / "db_state").stat().st_size == 0:
        raise ValueError(f"missing or empty DB state: {db}")
    return {
        "work_ns_per_cycle": work_ns / cycles,
        "achieved_rtf": (steps / RATE_HZ) / (wall_ns / 1e9),
        "cycles": cycles,
        "steps": steps,
        "warmup_excluded_ticks": WARMUP,
        "lockstep_steps": int(lockstep),
    }


def compare(base, candidate, threshold):
    base_work = statistics.median(run["work_ns_per_cycle"] for run in base)
    candidate_work = statistics.median(run["work_ns_per_cycle"] for run in candidate)
    pair_ratios = [
        candidate_run["work_ns_per_cycle"] / base_run["work_ns_per_cycle"]
        for base_run, candidate_run in zip(base, candidate, strict=True)
    ]
    paired_ratio = statistics.median(pair_ratios)
    return {
        "metric": "paired median non-pacing work-time ratio (candidate/base; higher is worse)",
        "base_work_ns_per_cycle": base_work,
        "candidate_work_ns_per_cycle": candidate_work,
        "pair_work_time_increases": [ratio - 1 for ratio in pair_ratios],
        "work_time_increase": paired_ratio - 1,
        "allowed_increase": threshold,
        "base_achieved_rtf": statistics.median(run["achieved_rtf"] for run in base),
        "candidate_achieved_rtf": statistics.median(run["achieved_rtf"] for run in candidate),
        "regression": paired_ratio > 1 + threshold,
    }


def confirmation(run_attempt):
    """Only a measured regression gets one fresh confirmation; exceptions propagate."""
    if not run_attempt(1):
        return "PASS"
    if not run_attempt(2):
        return "PASS on confirmation (first comparison regressed)"
    return "FAIL: performance regression confirmed"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_flight(args, side, attempt, index):
    name = f"attempt-{attempt}-{side}-{index}"
    run_dir = args.work / name
    run_dir.mkdir()
    source = getattr(args, f"{side}_source")
    # Each flight starts from exactly the same verified EEPROM, outside measured cycles.
    shutil.copyfile(args.eeprom, source / "eeprom.bin")
    env = {key: value for key, value in os.environ.items() if not key.startswith("RACE_")}
    # Use production pacing defaults, without inherited diagnostic overhead.
    for key in (
        "ELODIN_PACING_LEAD_US",
        "ELODIN_PACING_GRACE_US",
        "ELODIN_PACING_TRACE",
        "ELODIN_PACING_TRACE_MAX",
        "ELODIN_DETAILED_TIMING",
    ):
        env.pop(key, None)
    env.update(
        RACE_GUIDANCE="scripted",
        RACE_COURSE="none",
        RACE_CAMERA="0",
        ELODIN_SIM_SUMMARY_JSON=str(args.output / f"{name}.summary.json"),
        ELODIN_SIM_WARMUP_TICKS=str(WARMUP),
        ELODIN_BACKEND="cranelift",
        JAX_PLATFORMS="cpu",
        PYTHONUNBUFFERED="1",
    )
    if side == "base":
        env["PYTHONPATH"] = args.base_site + os.pathsep + env.get("PYTHONPATH", "")
    log_path = args.output / f"{name}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(source / "main.py"), "run", args.addr],
            cwd=run_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=args.timeout)
        finally:
            # Also reap firmware if shutdown/timeout interrupted s10 cleanup.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    if code != 0:
        raise ValueError(f"{name} exited {code}; see {log_path}")
    dbs = list(run_dir.glob("betaflight_db*"))
    if len(dbs) != 1:
        raise ValueError(f"{name}: expected exactly one recorded DB, found {dbs}")
    result = measurement(
        json.loads((args.output / f"{name}.summary.json").read_text()),
        log_path.read_text(errors="replace"),
        dbs[0],
    )
    write_json(args.output / f"{name}.measurement.json", result)
    print(
        f"{name}: {result['work_ns_per_cycle']:.0f} work ns/cycle, {result['achieved_rtf']:.3f}x achieved RTF",
        flush=True,
    )
    shutil.rmtree(run_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--base-site")
    parser.add_argument("--eeprom", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--addr", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (
        args.runs < 1
        or not math.isfinite(args.threshold)
        or args.threshold < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or bool(args.base_source) != bool(args.base_site)
    ):
        parser.error("invalid runs/threshold/timeout or incomplete base configuration")

    def attempt(number):
        results = {"base": [], "candidate": []}
        for index in range(1, args.runs + 1):
            # Alternate pair order to reduce a consistent first-run advantage.
            sides = ("base", "candidate") if index % 2 else ("candidate", "base")
            for side in sides:
                results[side].append(run_flight(args, side, number, index))
        report = compare(results["base"], results["candidate"], args.threshold)
        report["runs"] = results
        write_json(args.output / f"comparison-{number}.json", report)
        print(
            f"Paired comparison {number}: median work-time increase {report['work_time_increase']:+.2%} "
            f"(allowed {args.threshold:.2%}); "
            f"diagnostic side medians base={report['base_work_ns_per_cycle']:.0f}ns "
            f"candidate={report['candidate_work_ns_per_cycle']:.0f}ns; "
            f"achieved RTF base={report['base_achieved_rtf']:.3f}x "
            f"candidate={report['candidate_achieved_rtf']:.3f}x",
            flush=True,
        )
        return report["regression"]

    if args.base_source:
        verdict = confirmation(attempt)
    else:
        run_flight(args, "candidate", 1, 1)
        verdict = "SKIP performance A/B: baseline unavailable; candidate C0 validated"
    print(verdict, flush=True)
    (args.output / "verdict.txt").write_text(verdict + "\n")
    return int(verdict.startswith("FAIL"))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: Betaflight performance setup/measurement: {error}", file=sys.stderr)
        sys.exit(1)
