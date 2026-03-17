#!/usr/bin/env uv run

from __future__ import annotations

import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

METRIC_RE = {
    "tick_ms": re.compile(r"tick time:\s+([0-9.]+)\s+ms"),
    "kernel_ms": re.compile(r"kernel_invoke time:\s+([0-9.]+)\s+ms"),
    "h2d_ms": re.compile(r"h2d_upload time:\s+([0-9.]+)\s+ms"),
    "d2h_ms": re.compile(r"d2h_download time:\s+([0-9.]+)\s+ms"),
    "build_ms": re.compile(r"build time:\s+([0-9.]+)\s+ms"),
    "compile_ms": re.compile(r"compile time:\s+([0-9.]+)\s+ms"),
    "rtf": re.compile(r"real_time_factor:\s+([0-9.]+)"),
}
DB_PATH_RE = re.compile(r'created db path="([^"]+)"')


@dataclass(frozen=True)
class RunMetrics:
    tick_ms: float
    kernel_ms: float
    h2d_ms: float
    d2h_ms: float
    build_ms: float
    compile_ms: float
    rtf: float
    db_path: str


def parse_metrics(output: str) -> RunMetrics:
    parsed: dict[str, float] = {}
    for key, regex in METRIC_RE.items():
        match = regex.search(output)
        if not match:
            raise RuntimeError(f"missing metric {key} in benchmark output")
        parsed[key] = float(match.group(1))
    db_match = DB_PATH_RE.search(output)
    db_path = db_match.group(1) if db_match else ""
    return RunMetrics(
        tick_ms=parsed["tick_ms"],
        kernel_ms=parsed["kernel_ms"],
        h2d_ms=parsed["h2d_ms"],
        d2h_ms=parsed["d2h_ms"],
        build_ms=parsed["build_ms"],
        compile_ms=parsed["compile_ms"],
        rtf=parsed["rtf"],
        db_path=db_path,
    )


def run_once(backend: str, ticks: int, timeout_s: int) -> RunMetrics:
    env = os.environ.copy()
    env["ELODIN_BACKEND"] = backend
    env["DBNAME"] = f"dbs/n-body-{backend}-bench"
    cmd = [
        "uv",
        "run",
        "examples/n-body/main.py",
        "bench",
        "--profile",
        "--detail",
        "--ticks",
        str(ticks),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    full_output = proc.stdout + "\n" + proc.stderr
    log_path = Path("/tmp") / f"n-body-bench-{backend}.log"
    log_path.write_text(full_output, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} benchmark failed\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return parse_metrics(full_output)


def format_ms(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ticks = int(os.environ.get("ELODIN_NBODY_BENCH_TICKS", "50000"))
    repeats = int(os.environ.get("ELODIN_NBODY_BENCH_REPEATS", "3"))
    timeout_s = int(os.environ.get("ELODIN_NBODY_BENCH_TIMEOUT_S", "1800"))
    backends = ("iree-cpu", "iree-gpu", "jax-cpu") #, "jax-gpu"), # enable 'cuda' in nox-py pyproject.toml

    print(f"Running strict-realism n-body backend benchmark: ticks={ticks}, repeats={repeats}")
    print("Compile/build is reported but steady-state comparison uses tick/kernel/rtf.")

    summary: dict[str, RunMetrics] = {}
    for backend in backends:
        runs: list[RunMetrics] = []
        for i in range(repeats):
            started = time.time()
            m = run_once(backend, ticks=ticks, timeout_s=timeout_s)
            elapsed = time.time() - started
            runs.append(m)
            print(
                f"{backend} run {i + 1}/{repeats}: "
                f"tick={m.tick_ms:.3f}ms kernel={m.kernel_ms:.3f}ms rtf={m.rtf:.3f} "
                f"(wall={elapsed:.1f}s)"
            )

        summary[backend] = RunMetrics(
            tick_ms=statistics.median([r.tick_ms for r in runs]),
            kernel_ms=statistics.median([r.kernel_ms for r in runs]),
            h2d_ms=statistics.median([r.h2d_ms for r in runs]),
            d2h_ms=statistics.median([r.d2h_ms for r in runs]),
            build_ms=statistics.median([r.build_ms for r in runs]),
            compile_ms=statistics.median([r.compile_ms for r in runs]),
            rtf=statistics.median([r.rtf for r in runs]),
            db_path=runs[-1].db_path,
        )

    print("\nMedian metrics by backend")
    print("| backend | tick_ms | kernel_ms | h2d_ms | d2h_ms | rtf | build_ms | compile_ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for backend in backends:
        m = summary[backend]
        print(
            f"| {backend} | {format_ms(m.tick_ms)} | {format_ms(m.kernel_ms)} | "
            f"{format_ms(m.h2d_ms)} | {format_ms(m.d2h_ms)} | {m.rtf:.3f} | "
            f"{format_ms(m.build_ms)} | {format_ms(m.compile_ms)} |"
        )
        if m.db_path:
            print(f"  db_path[{backend}]: {m.db_path}")


if __name__ == "__main__":
    main()
