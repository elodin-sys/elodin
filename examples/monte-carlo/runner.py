#!/usr/bin/env uv run

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = ROOT / "examples/monte-carlo/main.py"


@dataclass(frozen=True)
class RunResult:
    seed: int
    db_path: Path
    returncode: int


@dataclass(frozen=True)
class MappingSample:
    pid: int
    virtual_kib: int
    rss_kib: int
    pss_kib: int
    rollup_rss_kib: int
    rollup_pss_kib: int
    paths: tuple[str, ...]
    cmd: str


def run_case(
    seed: int,
    out_dir: Path,
    grid_size: int,
    backend: str,
    cache_dir: Path,
    ticks: int,
    hold_after_run: float,
) -> RunResult:
    db_path = out_dir / f"run-{seed:04d}.db"
    if db_path.exists():
        shutil.rmtree(db_path)
    env = os.environ.copy()
    env["ELODIN_BACKEND"] = backend
    env["ELODIN_DB_PATH"] = str(db_path)
    env["ELODIN_MONTE_CARLO_SEED"] = str(seed)
    env["ELODIN_MONTE_CARLO_GRID_SIZE"] = str(grid_size)
    env["ELODIN_MONTE_CARLO_HOLD_AFTER_RUN_SEC"] = str(hold_after_run)
    env["ELODIN_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(
        ["uv", "run", str(ENTRYPOINT), "bench", "--ticks", str(ticks)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    log_path = out_dir / f"run-{seed:04d}.log"
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"seed {seed} failed; see {log_path}")
    return RunResult(seed=seed, db_path=db_path, returncode=proc.returncode)


def read_rollup_kib(path: Path) -> dict[str, int]:
    rollup = {}
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            parts = raw_value.split()
            if parts and parts[0].isdigit():
                rollup[key] = int(parts[0])
    except OSError:
        pass
    return rollup


def cache_mapping_samples(cache_dir: Path) -> list[MappingSample]:
    cache_prefix = str(cache_dir)
    samples = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            smaps = (proc_dir / "smaps").read_text(errors="ignore").splitlines()
        except OSError:
            continue

        paths: set[str] = set()
        virtual_kib = 0
        rss_kib = 0
        pss_kib = 0
        in_cache_mapping = False
        for line in smaps:
            parts = line.split()
            if parts and "-" in parts[0] and parts[0].count("-") == 1:
                in_cache_mapping = len(parts) >= 6 and parts[-1].startswith(cache_prefix)
                if in_cache_mapping:
                    start, end = (int(addr, 16) for addr in parts[0].split("-", 1))
                    virtual_kib += (end - start) // 1024
                    paths.add(parts[-1])
                continue
            if in_cache_mapping and line.startswith("Rss:"):
                rss_kib += int(line.split()[1])
            elif in_cache_mapping and line.startswith("Pss:"):
                pss_kib += int(line.split()[1])

        if not paths:
            continue

        try:
            cmd = (
                (proc_dir / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="ignore")
                .strip()
            )
        except OSError:
            cmd = ""
        rollup = read_rollup_kib(proc_dir / "smaps_rollup")
        samples.append(
            MappingSample(
                pid=int(proc_dir.name),
                virtual_kib=virtual_kib,
                rss_kib=rss_kib,
                pss_kib=pss_kib,
                rollup_rss_kib=rollup.get("Rss", 0),
                rollup_pss_kib=rollup.get("Pss", 0),
                paths=tuple(sorted(paths)),
                cmd=cmd[:180],
            )
        )
    return samples


def sample_memory(cache_dir: Path, stop: threading.Event, interval_s: float) -> list[MappingSample]:
    peak: list[MappingSample] = []
    peak_score = (-1, -1, -1)
    while not stop.wait(interval_s):
        samples = cache_mapping_samples(cache_dir)
        score = (
            len(samples),
            sum(sample.virtual_kib for sample in samples),
            sum(sample.rss_kib for sample in samples),
        )
        if score > peak_score:
            peak = samples
            peak_score = score
    final_samples = cache_mapping_samples(cache_dir)
    final_score = (
        len(final_samples),
        sum(sample.virtual_kib for sample in final_samples),
        sum(sample.rss_kib for sample in final_samples),
    )
    if final_score > peak_score:
        peak = final_samples
    return peak


def fmt_mib(kib: float) -> str:
    return f"{kib / 1024.0:.1f} MiB"


def print_memory_report(
    out_dir: Path,
    cache_dir: Path,
    grid_size: int,
    jobs: int,
    peak: list[MappingSample],
) -> None:
    table_bytes = grid_size * 2 * 8
    cache_files = sorted(cache_dir.glob("*.bin"))
    cache_bytes = sum(path.stat().st_size for path in cache_files)

    print("\n=== Shared constant memory report ===")
    print(f"cache_dir={cache_dir}")
    print(f"runs_out_dir={out_dir}")
    print(f"grid_size={grid_size} table_bytes={table_bytes} ({fmt_mib(table_bytes / 1024)})")
    print(
        f"cache_files={len(cache_files)} cache_bytes={cache_bytes} ({fmt_mib(cache_bytes / 1024)})"
    )
    print(f"peak_mapped_processes={len(peak)} jobs={jobs}")
    if cache_files:
        for path in cache_files:
            print(f"cache_file={path.name} size={fmt_mib(path.stat().st_size / 1024)}")

    for sample in sorted(peak, key=lambda item: item.pid):
        print(
            "mapping "
            f"pid={sample.pid} "
            f"virtual={fmt_mib(sample.virtual_kib)} "
            f"rss={fmt_mib(sample.rss_kib)} "
            f"pss={fmt_mib(sample.pss_kib)} "
            f"process_rss={fmt_mib(sample.rollup_rss_kib)} "
            f"process_pss={fmt_mib(sample.rollup_pss_kib)} "
            f"files={len(sample.paths)}"
        )
        if sample.paths:
            print(f"  path={sample.paths[0]}")
        print(f"  cmd={sample.cmd}")

    naive_bytes = table_bytes * max(1, len(peak))
    print(f"naive_constant_bytes_at_peak={naive_bytes} ({fmt_mib(naive_bytes / 1024)})")
    print(f"shared_cache_file_bytes={cache_bytes} ({fmt_mib(cache_bytes / 1024)})")
    if peak:
        print(
            "observed_mapping_pss_at_peak="
            f"{sum(sample.pss_kib for sample in peak)} KiB "
            f"({fmt_mib(sum(sample.pss_kib for sample in peak))})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Elodin implicit-constant Monte Carlo example"
    )
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("dbs/monte-carlo"))
    parser.add_argument("--grid-size", type=int, default=16_777_216)
    parser.add_argument("--backend", default="cranelift")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--hold-after-run", type=float, default=8.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.out_dir / "const-cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    stop_sampling = threading.Event()
    peak_samples: list[MappingSample] = []
    sampler = threading.Thread(
        target=lambda: peak_samples.extend(
            sample_memory(cache_dir, stop_sampling, args.sample_interval)
        ),
        name="monte-carlo-memory-sampler",
        daemon=True,
    )
    sampler.start()

    seeds = list(range(args.runs))
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    run_case,
                    seed,
                    args.out_dir,
                    args.grid_size,
                    args.backend,
                    cache_dir,
                    args.ticks,
                    args.hold_after_run,
                )
                for seed in seeds
            ]
            for future in as_completed(futures):
                result = future.result()
                print(f"seed={result.seed} db_path={result.db_path}")
    finally:
        stop_sampling.set()
        sampler.join()

    print_memory_report(args.out_dir, cache_dir, args.grid_size, args.jobs, peak_samples)


if __name__ == "__main__":
    main()
