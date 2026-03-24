"""Entry point for the linalg-IREE validation example.

Runs a Kalman filter simulation that exercises every IREE-safe linalg
shim (cholesky, solve, inv, qr, svd, det, slogdet, eigh) on the IREE
backend.

Usage:
    # Default (IREE backend):
    python examples/linalg-iree/main.py

    # Bench mode for CI:
    python examples/linalg-iree/main.py bench --ticks 200

    # Explicit backend override:
    ELODIN_BACKEND=jax-cpu python examples/linalg-iree/main.py
"""

import os

from sim import SIMULATION_RATE, system, world

backend = os.environ.get("ELODIN_BACKEND", "iree-cpu")

w = world()
w.run(
    system(),
    simulation_rate=SIMULATION_RATE,
    max_ticks=600,
    backend=backend,
)
