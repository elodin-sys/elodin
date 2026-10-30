#!/usr/bin/env uv run
"""N thrown drones flying a drone-policy controller in graph (policy_system.py).

    DRONE_N=16 python examples/drone-policy/main.py bench --ticks 1500
    DRONE_N=4 elodin editor examples/drone-policy/main.py

Each drone starts from its own DR draw and throw (env.reset of split(PRNGKey(DRONE_SEED), N)).
Settings come from the environment, since elodin owns the command line:
- DRONE_POLICY: a policy directory (policy.npz + config.json), a policy.npz or a baseline id
  (b0-nominal, b0-dr, b1-dr); default the shipped policy in policy/
- DRONE_N (16), DRONE_SEED (0), DRONE_DR (1), DRONE_THROW (1)
- DRONE_ESTIMATOR: default the policy's training estimator
- DRONE_BACKEND: default cranelift for one drone, jax-cpu for more (multi-entity worlds hit a
  known Cranelift miscompile in the stock chain)
"""

import os

import policy_system
from config import Config

env = os.environ
n = int(env.get("DRONE_N", "16"))
world, system = policy_system.build_world(
    env.get("DRONE_POLICY", policy_system.SHIPPED),
    n,
    env.get("DRONE_ESTIMATOR") or None,
    seed=int(env.get("DRONE_SEED", "0")),
    dr_scale=float(env.get("DRONE_DR", "1")),
    throw_scale=float(env.get("DRONE_THROW", "1")),
    schematic=True,
)
world.run(
    system,
    simulation_rate=Config.GLOBAL.simulation_rate,
    telemetry_rate=Config.GLOBAL.simulation_rate / 3.0,
    generate_real_time=True,
    max_ticks=1500,
    interactive=False,
    backend=env.get("DRONE_BACKEND") or ("cranelift" if n == 1 else "jax-cpu"),
)
