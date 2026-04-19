"""Entry point for the linalg validation example.

Runs a Kalman filter simulation that exercises every LAPACK-backed linalg
operation (cholesky, solve, inv, qr, svd, det, slogdet, eigh) on the
Cranelift backend.

Usage:
    # Default (Cranelift backend):
    python examples/linalg/main.py

    # Bench mode for CI:
    python examples/linalg/main.py bench --ticks 200

    # Explicit backend override:
    ELODIN_BACKEND=jax-cpu python examples/linalg/main.py
"""

import os

from sim import SIMULATION_RATE, system, world

backend = os.environ.get("ELODIN_BACKEND", "cranelift")

w = world()
w.run(
    system(),
    simulation_rate=SIMULATION_RATE,
    max_ticks=600,
    backend=backend,
)
