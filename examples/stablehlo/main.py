"""Entry point for the StableHLO coverage validation example.

Runs a simulation that exercises every implemented StableHLO/CHLO op
through the full JAX -> StableHLO -> Cranelift JIT pipeline.

Usage:
    # Default (Cranelift backend):
    python examples/stablehlo/main.py

    # Bench mode for CI:
    python examples/stablehlo/main.py bench --ticks 100

    # Explicit backend override:
    ELODIN_BACKEND=jax-cpu python examples/stablehlo/main.py
"""

import os

from sim import SIMULATION_RATE, system, world

backend = os.environ.get("ELODIN_BACKEND", "cranelift")

w = world()
w.run(
    system(),
    simulation_rate=SIMULATION_RATE,
    max_ticks=100,
    backend=backend,
)
