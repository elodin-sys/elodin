"""Minimal example demonstrating sensor camera frustum∩ellipsoid intersection.

One sensor camera creates the frustum. A 3D viewport displays it with
show_frustums=#true and the intersection overlay (COVERAGE, PROJ. 2D).
"""

import elodin as el
import jax.numpy as jnp

SENSOR_CAMERA_NAME = "ellipsoid.frustum_cam"


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

    world.sensor_camera(
        entity=body,
        name="frustum_cam",
        width=640,
        height=480,
        fov=45.0,
        near=0.05,
        far=6.0,
        pos_offset=[3.0, 0.5, 2.0],
        look_at_offset=[0.0, 0.0, 0.0],
        format="rgba",
        create_frustum=True,
        frustums_color=[1.0, 1.0, 0.0, 1.0],
        projection_color=[1.0, 1.0, 1.0, 1.0],
        frustums_thickness=0.006,
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
                viewport name="Frustum View" pos="(0,0,0,1, 2,2,1.5)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true show_grid=#true active=#true
                sensor_view "ellipsoid.frustum_cam" name="Sensor Camera"
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


def post_step(tick, ctx):
    if tick % 4 == 0:
        ctx.render_camera(SENSOR_CAMERA_NAME)
