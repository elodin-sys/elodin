#!/usr/bin/env uv run

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BodyErrorStats:
    body: str
    samples: int
    rms_au: float
    median_au: float
    max_au: float
    median_truth_radius_au: float


def parse_vec(s: str) -> np.ndarray:
    return np.fromstring(s.strip().strip("[]"), sep=",")


def read_timeseries(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ts = row[0]
            arr = parse_vec(row[1])
            if arr.shape[0] < 7:
                continue
            # Keep only position xyz and use the last write for duplicate timestamps.
            out[ts] = arr[4:7]
    return out


def body_error_stats(export_dir: Path, body: str) -> BodyErrorStats | None:
    sim_path = export_dir / f"{body}.world_pos.csv"
    truth_path = export_dir / f"truth_{body}.truth_world_pos.csv"
    if not sim_path.exists() or not truth_path.exists():
        return None

    sim = read_timeseries(sim_path)
    truth = read_timeseries(truth_path)
    common_ts = sorted(set(sim).intersection(truth))
    if not common_ts:
        return None

    sim_pos = np.stack([sim[ts] for ts in common_ts], axis=0)
    truth_pos = np.stack([truth[ts] for ts in common_ts], axis=0)
    errors = np.linalg.norm(sim_pos - truth_pos, axis=1)
    truth_radius = np.linalg.norm(truth_pos, axis=1)

    return BodyErrorStats(
        body=body,
        samples=len(common_ts),
        rms_au=float(np.sqrt(np.mean(errors * errors))),
        median_au=float(np.median(errors)),
        max_au=float(np.max(errors)),
        median_truth_radius_au=float(np.median(truth_radius)),
    )


def main() -> None:
    export_dir = Path(os.environ.get("ELODIN_NBODY_EXPORT_DIR", "/tmp/n-body-export.csv"))
    if not export_dir.exists():
        raise FileNotFoundError(
            f"export directory not found: {export_dir}. "
            "Run `elodin-db export <db_path> --format csv --output <dir>` first."
        )

    sim_files = sorted(export_dir.glob("*.world_pos.csv"))
    bodies = sorted(
        path.name.removesuffix(".world_pos.csv")
        for path in sim_files
        if not path.name.startswith("truth_")
    )
    body_stats = [s for body in bodies if (s := body_error_stats(export_dir, body)) is not None]
    if not body_stats:
        raise RuntimeError("no matching sim/truth world_pos series found in export directory")

    all_rms = np.array([s.rms_au for s in body_stats], dtype=np.float64)
    all_median = np.array([s.median_au for s in body_stats], dtype=np.float64)
    all_max = np.array([s.max_au for s in body_stats], dtype=np.float64)
    radii = np.array([s.median_truth_radius_au for s in body_stats], dtype=np.float64)
    # Dimensionless aggregate coefficient: 1 is perfect, lower is worse.
    global_rms_au = float(np.sqrt(np.mean(all_rms * all_rms)))
    median_truth_radius_au = float(np.median(radii))
    relative_rms = global_rms_au / max(median_truth_radius_au, 1e-12)
    accuracy_coefficient = 1.0 / (1.0 + relative_rms)

    print(f"Export dir: {export_dir}")
    print(f"Bodies compared: {len(body_stats)}")
    print(f"Global RMS error (AU): {global_rms_au:.9e}")
    print(f"Global median error (AU): {float(np.median(all_median)):.9e}")
    print(f"Global max error (AU): {float(np.max(all_max)):.9e}")
    print(f"Median truth radius (AU): {median_truth_radius_au:.9e}")
    print(f"Relative RMS error: {relative_rms:.9e}")
    print(f"Accuracy coefficient: {accuracy_coefficient:.9f}")

    print("\nTop 10 bodies by RMS error (AU):")
    print("| body | samples | rms_au | median_au | max_au |")
    print("|---|---:|---:|---:|---:|")
    for s in sorted(body_stats, key=lambda x: x.rms_au, reverse=True)[:10]:
        print(f"| {s.body} | {s.samples} | {s.rms_au:.9e} | {s.median_au:.9e} | {s.max_au:.9e} |")


if __name__ == "__main__":
    main()
