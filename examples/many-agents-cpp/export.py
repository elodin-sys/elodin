"""Export the many-agents simulation's step for the C++ host.

    python3 examples/many-agents-cpp/export.py [model_dir] [--agents N] [--workers W]

Builds the world of `examples/many-agents/sim.py`, unchanged, and writes its step, compiled ahead
of time for this machine, with the initial state and schema, to `model_dir` (default
`examples/many-agents-cpp/build/model`). `--workers` must match the Elodin run being reproduced:
with more than one, Elodin traces each `@el.map` system in chunks of two entities.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "many-agents"))
from sim import RATE_HZ, build  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("model", nargs="?", type=Path, default=HERE / "build" / "model")
parser.add_argument("--agents", type=int, default=24)
parser.add_argument("--objects", type=int, help="moving objects (default: as many as agents)")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

world, system, *_ = build(args.agents, args.objects or args.agents)
sim = world.to_jax(system, simulation_rate=RATE_HZ, workers=args.workers)
sim.export_model(args.model)
print(f"exported {args.agents} agents for {args.workers} workers to {args.model}")
