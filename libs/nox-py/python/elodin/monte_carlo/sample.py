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


KNOWN_DISTS = ("fixed", "choice", "uniform", "loguniform", "normal")
# Accepted spellings for range bounds, in precedence order.
_MIN_ALIASES = ("min", "lo", "low")
_MAX_ALIASES = ("max", "hi", "high")


def _bound(spec: dict, aliases: tuple[str, ...]):
    for alias in aliases:
        if alias in spec:
            return spec[alias]
    return None


def _spec_error(name: str, dist: str, needs: str, spec: dict) -> ValueError:
    got = ", ".join(sorted(key for key in spec if key != "dist")) or "nothing"
    return ValueError(f'{dist} for "{name}" needs {needs} (got: {got})')


def _validate_variable(name: str, spec) -> None:
    """Fail up front with the variable name and what's missing, instead of a
    bare TypeError from deep inside sampling."""
    if not isinstance(spec, dict):
        raise ValueError(f'variable "{name}" must be a table like {{ dist = "normal", ... }}')
    dist = str(spec.get("dist", "fixed")).lower()
    if dist not in KNOWN_DISTS:
        known = ", ".join(KNOWN_DISTS)
        raise ValueError(f'unknown dist "{dist}" for "{name}" (known: {known})')
    if dist == "fixed" and "value" not in spec:
        raise _spec_error(name, dist, "value", spec)
    if dist == "choice" and not spec.get("values"):
        raise _spec_error(name, dist, "a non-empty values list", spec)
    if dist in ("uniform", "loguniform"):
        if _bound(spec, _MIN_ALIASES) is None or _bound(spec, _MAX_ALIASES) is None:
            raise _spec_error(name, dist, "min/max", spec)
        if dist == "loguniform" and (
            float(_bound(spec, _MIN_ALIASES)) <= 0 or float(_bound(spec, _MAX_ALIASES)) <= 0
        ):
            raise ValueError(f'loguniform for "{name}" needs positive min/max')
    if dist == "normal" and ("mean" not in spec or "std" not in spec):
        raise _spec_error(name, dist, "mean/std", spec)


def _sample_dist(spec: dict, u: float):
    dist = str(spec.get("dist", "fixed")).lower()
    if dist == "fixed":
        return spec.get("value")
    if dist == "choice":
        values = spec["values"]
        return values[min(int(u * len(values)), len(values) - 1)]
    if dist == "uniform":
        lo = float(_bound(spec, _MIN_ALIASES))
        hi = float(_bound(spec, _MAX_ALIASES))
        return lo + (hi - lo) * u
    if dist == "loguniform":
        lo = math.log(float(_bound(spec, _MIN_ALIASES)))
        hi = math.log(float(_bound(spec, _MAX_ALIASES)))
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
    if n < 1:
        raise ValueError(f"n_samples must be >= 1 (got {n})")
    method = str(mc.get("method", "lhs")).lower()
    if method not in ("lhs", "random"):
        raise ValueError(f'unknown method "{method}" (known: lhs, random)')
    rng = random.Random(mc.get("seed"))
    variables = dict(mc.get("variables", {}))
    for name, spec in variables.items():
        _validate_variable(name, spec)
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
            payload = {"run_id": f"run_{idx:07}", "seed": idx + 1}
            payload.update(row)
            writer.writerow(payload)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m elodin.monte_carlo.sample SPEC.toml PLAN.csv")
    materialize(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
