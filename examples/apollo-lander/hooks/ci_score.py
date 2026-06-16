"""CI post-run smoke: pass when the truncated sim produced a result artifact."""

from __future__ import annotations

from pathlib import Path

from mc_metrics import read_json, traj_rmse


def post_run(ctx):
    result_path = Path(ctx.run_dir) / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"CI smoke: missing {result_path}")
    result = read_json(result_path)
    rmse = traj_rmse({}, result)
    if rmse is None:
        raise RuntimeError(f"CI smoke: {result_path} has no traj_rmse")
    return {
        "pass": True,
        "smoke": True,
        "landed": bool(result.get("landed", False)),
        "traj_rmse_m": rmse,
    }
