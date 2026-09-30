"""Minimal editor smoke test for real terrain rendering.

Prepare the real Brienz elevation and imagery atlas once from the repository
root before launching the example:

    ./scripts/prepare_editor_terrain_region.sh brienz
    elodin editor examples/terrain/main.py

The viewport intentionally uses HDR while SDR terrain pipeline specialization
remains a separate follow-up.
"""

import elodin as el
import jax.numpy as jnp
from pathlib import Path

SIM_RATE = 30.0


def world() -> el.World:
    world = el.World()
    world.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 3_000.0])),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        name="reference",
    )
    schematic_path = Path(__file__).with_name("terrain.kdl")
    world.schematic(schematic_path.read_text(), "terrain.kdl")
    return world


@el.map
def no_force(force: el.Force) -> el.Force:
    return force


world().run(
    el.six_dof(sys=no_force),
    simulation_rate=SIM_RATE,
    generate_real_time=True,
    max_ticks=int(SIM_RATE * 120.0),
)
