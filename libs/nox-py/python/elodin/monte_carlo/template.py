"""Generate starter Monte Carlo campaign files from a simulation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _spec(sim_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(sim_path), "params"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout or "{}")


def write_template(sim_path: Path, output_path: Path) -> None:
    spec = _spec(sim_path)
    params = spec.get("params", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        headers = ["run_id", "seed"] + [f"param.{name}" for name in sorted(params)]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": "run_0000000",
                    "seed": 1,
                    **{
                        f"param.{name}": param.get("default")
                        for name, param in sorted(params.items())
                    },
                }
            )
        return

    lines = [
        "[monte_carlo]",
        "n_samples = 4",
        "seed = 0",
        'method = "lhs"',
        "",
        "[monte_carlo.variables]",
    ]
    for name, param in sorted(params.items()):
        default = json.dumps(param.get("default"))
        lines.append(f'"{name}" = {{ dist = "fixed", value = {default} }}')
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m elodin.monte_carlo.template SIM.py OUT.toml|OUT.csv")
    write_template(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
