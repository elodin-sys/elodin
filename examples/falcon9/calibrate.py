#!/usr/bin/env python3
"""Calibration-loop tooling for the Falcon 9 parity campaign.

Subcommands:
  rank  <out_dir>                     - rank runs by fit_score, print best params
  narrow <out_dir> <spec> <new_spec>  - write a spec narrowed around the best run

The loop (plan Phase 9 / monte-carlo skill): run campaign -> rank -> narrow ->
run again, keeping the LHS seed fixed while iterating.
"""

from __future__ import annotations

import csv
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "hooks"))
from mc_metrics import fit_score  # noqa: E402

KEEP_FRACTION = 0.4  # each round keeps this fraction of every range


def load_runs(out_dir: Path) -> list[tuple[str, dict, dict]]:
    """Rank by fit recomputed from raw result.json with the CURRENT metric,
    so metric improvements re-rank old campaigns instead of being ignored."""
    rows = list(csv.DictReader((out_dir / "results.csv").open()))
    runs = []
    for row in rows:
        result_json = out_dir / row.get("result_json", "")
        ctx_path = result_json.with_name("post_run_context.json")
        if not result_json.exists():
            continue
        result = json.loads(result_json.read_text())
        post = dict(result)
        post["fit_score"] = fit_score(result)
        post.setdefault("touchdown_pos_err_m", post.get("touchdown_pos_err_m"))
        params = {}
        if ctx_path.exists():
            params = json.loads(ctx_path.read_text()).get("params", {})
        runs.append((row.get("run_id", "?"), post, params))
    runs.sort(key=lambda r: r[1]["fit_score"] if r[1]["fit_score"] is not None else float("inf"))
    return runs


def rank(out_dir: Path, top: int = 5) -> None:
    runs = load_runs(out_dir)
    print(f"{len(runs)} scored runs; best {min(top, len(runs))} by fit_score:")
    for run_id, post, _params in runs[:top]:
        print(
            f"  {run_id}  fit={post.get('fit_score', float('nan')):8.2f}"
            f"  speed_rmse={post.get('speed_rmse_mps', float('nan')):7.1f}"
            f"  alt_rmse={post.get('alt_rmse_m', float('nan')):8.0f}"
            f"  pos={post.get('touchdown_pos_err_m', float('nan')):7.0f}"
            f"  vert={post.get('touchdown_vertical_mps', float('nan')):6.2f}"
            f"  lat={post.get('touchdown_lateral_mps', float('nan')):6.2f}"
        )
    if runs:
        print("\nbest-run params:")
        for key, value in sorted(runs[0][2].items()):
            print(f"  {key} = {value}")


def narrow(out_dir: Path, spec_path: Path, new_spec_path: Path) -> None:
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit("no scored runs to narrow around")
    best = runs[0][2]
    spec = tomllib.loads(spec_path.read_text())
    variables = spec.get("monte_carlo", {}).get("variables", {})

    lines = [
        "# Auto-narrowed by calibrate.py around the best-fit run.",
        "",
        "[monte_carlo]",
        f"n_samples = {spec['monte_carlo'].get('n_samples', 24)}",
        f"seed = {spec['monte_carlo'].get('seed', 20170814)}",
        f'method = "{spec["monte_carlo"].get("method", "lhs")}"',
        "",
        "[monte_carlo.variables]",
    ]
    for key, cfg in variables.items():
        lo, hi = float(cfg["min"]), float(cfg["max"])
        center = float(best.get(key, (lo + hi) / 2.0))
        half = (hi - lo) * KEEP_FRACTION / 2.0
        new_lo = max(lo, center - half)
        new_hi = min(hi, center + half)
        lines.append(f'{key} = {{ dist = "uniform", min = {new_lo}, max = {new_hi} }}')
    new_spec_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {new_spec_path} (kept {KEEP_FRACTION:.0%} of each range around best run)")


def best_params_json(out_dir: Path) -> None:
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit("no scored runs")
    print(json.dumps(runs[0][2]))


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    cmd = sys.argv[1]
    if cmd == "rank":
        rank(Path(sys.argv[2]))
    elif cmd == "narrow":
        narrow(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif cmd == "best-json":
        best_params_json(Path(sys.argv[2]))
    else:
        print(__doc__)
        raise SystemExit(2)


main()
