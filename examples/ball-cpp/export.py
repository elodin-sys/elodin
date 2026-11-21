"""Export the ball's step for a native host.

    python3 examples/ball-cpp/export.py [model_dir]

Writes the step as StableHLO and as a precompiled object, the initial state and the schema to
`model_dir`, by default `examples/ball-cpp/build/model`.
"""

import sys
from pathlib import Path

from sim import SIM_TIME_STEP, system, world

model = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "build" / "model"
sim = world().to_jax(system(), simulation_rate=1.0 / SIM_TIME_STEP)
sim.export_model(model)
print(f"exported the ball model to {model}")
