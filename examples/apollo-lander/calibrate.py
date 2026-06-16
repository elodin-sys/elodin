#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# Share the soft-landing / RMSE / float helpers with the hooks.
sys.path.insert(0, str(Path(__file__).parent / "hooks"))
from mc_metrics import read_json, soft_landing, to_float, traj_rmse  # noqa: E402


def _plan_params(out_dir: Path) -> dict[str, dict[str, float]]:
    """Map run_id -> sampled params, read from the campaign's `plan.csv`.

    The plan is the materialized source of truth for each run's inputs (one
    `param.<name>` column per variable), so we read it directly instead of the
    per-run `post_run_context.json`.
    """
    plan_path = out_dir / "plan.csv"
    params_by_run: dict[str, dict[str, float]] = {}
    if not plan_path.exists():
        return params_by_run
    with plan_path.open(newline="") as f:
        for row in csv.DictReader(f):
            run_id = row.get("run_id", "")
            if not run_id:
                continue
            params = {}
            for key, value in row.items():
                if key.startswith("param."):
                    parsed = to_float(value)
                    if parsed is not None:
                        params[key[len("param.") :]] = parsed
            params_by_run[run_id] = params
    return params_by_run


def best_fit(out_dir: Path) -> tuple[str, dict[str, float], float]:
    rows_path = out_dir / "results.csv"
    with rows_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    plan_params = _plan_params(out_dir)
    best_run = ""
    best_params: dict[str, float] = {}
    best_rmse = float("inf")
    for row in rows:
        result_path = out_dir / row.get("result_json", "")
        result = read_json(result_path)
        post = read_json(result_path.with_name("post_run_result.json"))
        if not soft_landing(post, result):
            continue
        rmse = traj_rmse(post, result)
        if rmse is None or rmse >= best_rmse:
            continue
        run_id = row.get("run_id", "")
        best_run = run_id
        best_params = plan_params.get(run_id, {})
        best_rmse = rmse
    if not best_run:
        raise RuntimeError(f"no soft-landing runs with a trajectory RMSE in {out_dir}")
    return best_run, best_params, best_rmse


def narrowed_spec(
    base_spec: Path, best_params: dict[str, float], shrink: float, n_samples: int
) -> str:
    root = tomllib.loads(base_spec.read_text())
    mc = root["monte_carlo"]
    variables = dict(mc["variables"])
    lines = [
        "[monte_carlo]",
        f"n_samples = {n_samples}",
        f"seed = {int(mc.get('seed', 19690720)) + 1}",
        f"method = {mc.get('method', 'lhs')!r}",
        "",
        "[monte_carlo.variables]",
    ]
    for key in sorted(variables):
        spec = dict(variables[key])
        center = best_params.get(key)
        if center is not None and spec.get("dist", "uniform") == "uniform":
            lo = float(spec["min"])
            hi = float(spec["max"])
            half_width = (hi - lo) * shrink * 0.5
            spec["min"] = max(lo, center - half_width)
            spec["max"] = min(hi, center + half_width)
        rendered = ", ".join(
            f"{item_key} = {item_value!r}"
            if isinstance(item_value, str)
            else f"{item_key} = {item_value}"
            for item_key, item_value in spec.items()
        )
        lines.append(f"{key} = {{ {rendered} }}")
    return "\n".join(lines) + "\n"


def run_campaign(spec: Path, out_dir: Path) -> None:
    cmd = [
        "elodin",
        "monte-carlo",
        "run",
        "examples/apollo-lander/main.py",
        "--campaign",
        "examples/apollo-lander/campaign.toml",
        "--spec",
        str(spec),
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iteratively narrow Apollo lander Monte Carlo ranges."
    )
    parser.add_argument(
        "--initial-out", type=Path, required=True, help="Existing campaign output to start from."
    )
    parser.add_argument("--base-spec", type=Path, default=Path("examples/apollo-lander/spec.toml"))
    parser.add_argument("--work-dir", type=Path, default=Path("dbs/apollo-lander-calibration"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--shrink", type=float, default=0.5, help="Fraction of prior range width to keep."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write narrowed specs without launching campaigns."
    )
    args = parser.parse_args()

    previous_out = args.initial_out
    spec = args.base_spec
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for round_idx in range(args.rounds):
        run_id, params, rmse = best_fit(previous_out)
        next_spec = args.work_dir / f"round_{round_idx + 1:02d}.toml"
        next_spec.write_text(narrowed_spec(spec, params, args.shrink, args.samples))
        next_out = args.work_dir / f"round_{round_idx + 1:02d}"
        print(f"round {round_idx + 1}: best={run_id} rmse={rmse:.3f}m -> {next_spec}")
        spec = next_spec
        if not args.dry_run:
            run_campaign(next_spec, next_out)
            previous_out = next_out


if __name__ == "__main__":
    main()
