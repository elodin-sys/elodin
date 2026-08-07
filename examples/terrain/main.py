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
    world.schematic(
        """
        coordinate frame=ENU
        world_mesh "brienz"

        viewport frame=ENU name="Brienz Terrain" hdr=#true active=#true pos="(0,0,0,1, 4560,-4560,2640)" look_at="(0,0,0,1, 0,0,-120)" up="(0,0,1)" near=1.0 far=50000.0 fov=60.0 show_grid=#false show_view_cube=#true
        """,
        "terrain.kdl",
    )
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
