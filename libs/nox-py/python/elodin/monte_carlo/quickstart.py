"""Scaffold a runnable Monte Carlo campaign from a simulation's declared params.

`elodin monte-carlo quickstart SIM.py OUTDIR` reads the sim's `params_spec`
(via `python SIM.py params`) and writes a ready-to-run campaign skeleton:

    OUTDIR/
      spec.toml        # one variable per param: uniform[min,max] or fixed default
      campaign.toml    # worker pool + lifecycle hooks
      hooks/score.py   # post_run: copies result.json metrics, marks pass
      hooks/gate.py    # post_campaign: fails the run when any run failed

The generated files are intentionally minimal and meant to be edited. The goal
is to get a new simulation into `elodin monte-carlo run` in a couple of minutes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCORE_HOOK = '''"""post_run hook: record metrics emitted via el.monte_carlo.result(...)."""

from __future__ import annotations

import json
from pathlib import Path


def post_run(ctx):
    result_path = Path(ctx.run_dir) / "result.json"
    result = {}
    if result_path.exists():
        result = json.loads(result_path.read_text())

    # Edit this to express your own pass/fail criterion. By default a run passes
    # when it produced a result.json (i.e. the sim ran to completion).
    passed = bool(result)

    return {"pass": passed, **result}
'''

GATE_HOOK = '''"""post_campaign hook: fail the campaign process when any run failed.

Use this in CI. The campaign itself exits 0 by default (partial failures are
normal for exploratory Monte Carlo); this hook is what turns a failure count
into a non-zero exit for a gated pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path


def post_campaign(ctx):
    summary = json.loads(Path(ctx.summary).read_text())
    failed = int(summary.get("failed", 0))
    total = int(summary.get("total_runs", 0))
    if failed > 0:
        raise RuntimeError(f"monte-carlo gate: {failed}/{total} run(s) failed")
    return {"pass": True, "failed": failed, "total_runs": total}
'''


def _spec(sim_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(sim_path), "params"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout or "{}")


def _variable_line(name: str, param: dict) -> str:
    minimum = param.get("min")
    maximum = param.get("max")
    if minimum is not None and maximum is not None:
        return f'"{name}" = {{ dist = "uniform", min = {json.dumps(minimum)}, max = {json.dumps(maximum)} }}'
    return f'"{name}" = {{ dist = "fixed", value = {json.dumps(param.get("default"))} }}'


def write_quickstart(sim_path: Path, out_dir: Path) -> list[Path]:
    params = _spec(sim_path).get("params", {})
    hooks_dir = out_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    spec_lines = [
        "[monte_carlo]",
        "n_samples = 16",
        "seed = 0",
        'method = "lhs"',
        "",
        "[monte_carlo.variables]",
    ]
    spec_lines += [_variable_line(name, params[name]) for name in sorted(params)]
    spec_path = out_dir / "spec.toml"
    spec_path.write_text("\n".join(spec_lines) + "\n")

    campaign_path = out_dir / "campaign.toml"
    campaign_path.write_text(
        "\n".join(
            [
                'timeout = "120s"',
                "retries = 0",
                "continue_on_error = true",
                "",
                "[hooks]",
                'post_run = "hooks/score.py"',
                'post_campaign = "hooks/gate.py"',
            ]
        )
        + "\n"
    )

    score_path = hooks_dir / "score.py"
    score_path.write_text(SCORE_HOOK)
    gate_path = hooks_dir / "gate.py"
    gate_path.write_text(GATE_HOOK)

    written = [spec_path, campaign_path, score_path, gate_path]

    rel_out = out_dir.as_posix()
    print("Monte Carlo quickstart scaffolded:")
    for path in written:
        print(f"  {path.as_posix()}")
    print("")
    print("Next steps:")
    print("  1. Edit spec.toml ranges and hooks/score.py pass criterion.")
    print("  2. Run the campaign:")
    print(f"     elodin monte-carlo run {sim_path.as_posix()} \\")
    print(f"       --campaign {rel_out}/campaign.toml \\")
    print(f"       --spec {rel_out}/spec.toml \\")
    print(f"       --out dbs/{out_dir.name}")
    return written


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m elodin.monte_carlo.quickstart SIM.py OUTDIR")
    write_quickstart(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
