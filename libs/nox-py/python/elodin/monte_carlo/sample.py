"""Materialize a Monte Carlo sampling spec into a plan CSV."""

from __future__ import annotations

import csv
import itertools
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _sample_dist(spec: dict, u: float):
    dist = str(spec.get("dist", "fixed")).lower()
    if dist == "fixed":
        return spec.get("value")
    if dist == "choice":
        values = spec["values"]
        return values[min(int(u * len(values)), len(values) - 1)]
    if dist == "uniform":
        lo = float(spec.get("min", spec.get("lo")))
        hi = float(spec.get("max", spec.get("hi")))
        return lo + (hi - lo) * u
    if dist == "loguniform":
        lo = math.log(float(spec.get("min", spec.get("lo"))))
        hi = math.log(float(spec.get("max", spec.get("hi"))))
        return math.exp(lo + (hi - lo) * u)
    if dist == "normal":
        mean = float(spec["mean"])
        std = float(spec["std"])
        clipped = min(max(u, 1e-12), 1.0 - 1e-12)
        return mean + std * NormalDist().inv_cdf(clipped)
    raise ValueError(f"unsupported distribution: {dist}")


def _lhs(n: int, d: int, rng: random.Random) -> list[list[float]]:
    rows = [[0.0] * d for _ in range(n)]
    for col in range(d):
        values = [(row + rng.random()) / n for row in range(n)]
        rng.shuffle(values)
        for row, value in enumerate(values):
            rows[row][col] = value
    return rows


def _mc_rows(root: dict) -> list[dict[str, object]]:
    mc = root.get("monte_carlo")
    if not mc:
        return [{}]
    n = int(mc.get("n_samples", 1))
    method = str(mc.get("method", "lhs")).lower()
    rng = random.Random(mc.get("seed"))
    variables = dict(mc.get("variables", {}))
    keys = sorted(variables)
    units = (
        _lhs(n, len(keys), rng)
        if method == "lhs"
        else [[rng.random() for _ in keys] for _ in range(n)]
    )
    rows = []
    for unit_row in units:
        row = {}
        for key, u in zip(keys, unit_row):
            row[f"param.{key}"] = _sample_dist(variables[key], u)
        rows.append(row)
    return rows


def _grid_rows(table: dict | None, prefix: str) -> list[dict[str, object]]:
    if not table:
        return [{}]
    keys = sorted(table)
    combos = itertools.product(*(table[key] for key in keys))
    return [{f"{prefix}.{key}": value for key, value in zip(keys, combo)} for combo in combos]


def materialize(spec_path: Path, output_path: Path) -> None:
    root = tomllib.loads(spec_path.read_text())
    sim_rows = _grid_rows(root.get("sim_sweep"), "param")
    meta_rows = _grid_rows(root.get("meta_sweep"), "meta")
    mc_rows = _mc_rows(root)
    rows = []
    for sim_row, meta_row, mc_row in itertools.product(sim_rows, meta_rows, mc_rows):
        merged = {}
        merged.update(sim_row)
        merged.update(mc_row)
        merged.update(meta_row)
        rows.append(merged)
    headers = ["run_id", "seed"] + sorted({key for row in rows for key in row})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for idx, row in enumerate(rows):
            payload = {"run_id": f"run_{idx:07}", "seed": idx}
            payload.update(row)
            writer.writerow(payload)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m elodin.monte_carlo.sample SPEC.toml PLAN.csv")
    materialize(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
