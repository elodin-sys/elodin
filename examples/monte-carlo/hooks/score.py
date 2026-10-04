from __future__ import annotations

import json
import math
import os

CAPTURE_RADIUS_ENV = "ELODIN_MONTE_CARLO_CAPTURE_RADIUS_M"
DEFAULT_CAPTURE_RADIUS_M = 8.5


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default


def _finite_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def post_run(ctx):
    result = {}
    result_path = ctx.run_dir + "/result.json"
    try:
        with open(result_path) as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    error = _finite_float(result.get("error"), float("inf"))
    capture_radius = _float_env(CAPTURE_RADIUS_ENV, DEFAULT_CAPTURE_RADIUS_M)
    valid = bool(result) and math.isfinite(error)
    return {
        "error": error,
        "capture_radius_m": capture_radius,
        "valid": valid,
        "pass": valid and error < capture_radius,
    }
