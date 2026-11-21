"""Export the ball's step for a native host.

    python3 examples/ball-cpp/export.py [model_dir]

Writes the step as StableHLO and as a precompiled object, the initial state and the schema to
`model_dir`, by default `examples/ball-cpp/build/model`.
"""

import argparse
from pathlib import Path

from sim import SIM_TIME_STEP, system, world

parser = argparse.ArgumentParser()
parser.add_argument(
    "model",
    nargs="?",
    type=Path,
    default=Path(__file__).parent / "build" / "model",
)
parser.add_argument("--cpu", choices=["native", "generic"], default="native")
args = parser.parse_args()

sim = world().to_jax(system(), simulation_rate=1.0 / SIM_TIME_STEP)
sim.export_model(args.model, cpu=args.cpu)
print(f"exported the ball model to {args.model} for {args.cpu} CPU features")
