"""Minimal example demonstrating sensor camera frustum∩ellipsoid intersection.

One sensor camera creates the frustum. A 3D viewport displays it with
show_frustums=#true and the intersection overlay (COVERAGE, PROJ. 2D).
"""

import elodin as el
import jax.numpy as jnp
import numpy as np

SIM_RATE = 120.0
SENSOR_CAMERA_NAME = "frustum_camera_rig.frustum_cam"
CAMERA_RIG_NAME = "frustum_camera_rig"
CAMERA_ORBIT_RADIUS = 3.2
CAMERA_ORBIT_HEIGHT = 1.6
CAMERA_ORBIT_RATE = 0.45


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
    camera_rig = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    linear=jnp.array([CAMERA_ORBIT_RADIUS, 0.0, CAMERA_ORBIT_HEIGHT])
                ),
                inertia=el.SpatialInertia(mass=1.0),
            ),
        ],
        name=CAMERA_RIG_NAME,
    )

    world.sensor_camera(
        entity=camera_rig,
        name="frustum_cam",
        width=640,
        height=480,
        fov=45.0,
        near=0.05,
        far=6.0,
        pos_offset=[0.0, 0.0, 0.0],
        look_at_offset=[0.0, 0.0, -1.0],
        format="rgba",
        create_frustum=True,
        frustums_color=[0.0, 1.0, 1.0, 1.0],
        projection_color=[0.0, 1.0, 1.0, 1.0],
        frustums_thickness=0.008,
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
            hsplit name="Frustums" {
                viewport name="Viewport Source" pos="(0,0,0,1, -3,-0.5,2)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true frustums_color="yalk" projection_color="mint" frustums_thickness=0.006 show_grid=#true active=#true near=0.05 far=6.0
                viewport name="Target View" pos="(0,0,0,1, 2,2,1.5)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true show_grid=#true active=#true
                sensor_view "frustum_camera_rig.frustum_cam" name="Sensor Camera"
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


def _quat_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    dot = np.dot(source, target)
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    xyz = np.cross(source, target)
    w = np.sqrt(np.dot(source, source) * np.dot(target, target)) + dot
    quat = np.array([xyz[0], xyz[1], xyz[2], w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def pre_step(tick, ctx):
    t = tick / SIM_RATE
    angle = t * CAMERA_ORBIT_RATE
    pos = np.array(
        [
            CAMERA_ORBIT_RADIUS * np.cos(angle),
            CAMERA_ORBIT_RADIUS * np.sin(angle),
            CAMERA_ORBIT_HEIGHT + 0.35 * np.sin(angle * 1.7),
        ],
        dtype=np.float64,
    )
    look_dir = -pos
    quat = _quat_from_to(np.array([0.0, 0.0, -1.0], dtype=np.float64), look_dir)

    ctx.write_component(
        f"{CAMERA_RIG_NAME}.world_pos",
        np.array(
            [quat[0], quat[1], quat[2], quat[3], pos[0], pos[1], pos[2]],
            dtype=np.float64,
        ),
    )


def post_step(tick, ctx):
    if tick % 4 == 0:
        ctx.render_camera(SENSOR_CAMERA_NAME)
