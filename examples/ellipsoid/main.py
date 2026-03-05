#!/usr/bin/env uv run
"""Run the ellipsoid frustum-intersection example."""

from sim import system, world

world_instance, _ = world()
world_instance.run(
    system(),
    sim_time_step=1.0 / 120.0,
    run_time_step=1.0 / 120.0,
    max_ticks=1200,
)
