"""Minimal example demonstrating frustum∩ellipsoid intersection.

Two viewports: one creates the frustum (create_frustum=#true), the other
displays it with show_frustums=#true and the intersection overlay (COVERAGE, PROJ. 2D).
"""

import elodin as el
import jax.numpy as jnp


def world() -> tuple[el.World, el.EntityId]:
    world = el.World()
    body = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.SpatialInertia(mass=1.0),
            ),
        ],
        name="ellipsoid",
    )

    object_mesh = """
    object_3d ellipsoid.world_pos {
        ellipsoid scale="(1.2, 1.2, 0.5)" show_grid=#true {
            color 0 188 212 40
            grid_color 255 255 255 120
        }
    }
    """

    world.schematic(
        """
        theme mode="dark" scheme="default"

        tabs {
            hsplit name="Viewport" {
                viewport name="Frustum Source" pos="(0,0,0,1, 3,0.5,2)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true show_grid=#true active=#true near=0.05 far=6.0
                viewport name="Frustum View" pos="(0,0,0,1, 2,2,1.5)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true show_grid=#true active=#true
            }
        }
    """
        + object_mesh,
        "ellipsoid.kdl",
    )
    return world, body


@el.map
def no_force(f: el.Force) -> el.Force:
    return f


def system() -> el.System:
    return el.six_dof(sys=no_force)
