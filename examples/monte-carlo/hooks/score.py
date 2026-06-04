from __future__ import annotations


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
    return {
        "error": error,
        "pass": error < 2.0,
    }
