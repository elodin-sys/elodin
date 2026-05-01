#!/usr/bin/env uv run
"""Run the ellipsoid frustum-intersection example."""

from sim import SIM_RATE, post_step, pre_step, system, world

world_instance, _ = world()
world_instance.run(
    system(),
    simulation_rate=SIM_RATE,
    generate_real_time=True,
    pre_step=pre_step,
    post_step=post_step,
)
