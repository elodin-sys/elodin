#!/usr/bin/env uv run
"""Run the ellipsoid frustum-intersection example."""

from sim import post_step, system, world

world_instance, _ = world()
world_instance.run(
    system(),
    simulation_rate=120.0,
    generate_real_time=True,
    max_ticks=1200,
    post_step=post_step,
)
