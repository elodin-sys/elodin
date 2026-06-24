from __future__ import annotations

import math
import os

CAPTURE_RADIUS_ENV = "ELODIN_MONTE_CARLO_CAPTURE_RADIUS_M"
DEFAULT_CAPTURE_RADIUS_M = 8.5


def post_run(ctx):
    result = {}
    result_path = ctx.run_dir + "/result.json"
    try:
        import json

        with open(result_path) as f:
            result = json.load(f)
    except OSError:
        pass
    error = float(result.get("error", float("inf")))
    capture_radius = float(os.environ.get(CAPTURE_RADIUS_ENV, DEFAULT_CAPTURE_RADIUS_M))
    valid = bool(result) and math.isfinite(error)
    return {
        "error": error,
        "capture_radius_m": capture_radius,
        "valid": valid,
        "pass": valid and error < capture_radius,
    }
