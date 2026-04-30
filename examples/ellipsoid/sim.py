"""Minimal example demonstrating sensor camera frustum∩ellipsoid intersection.

A drone GLB moves inside the ellipsoid. The drone-mounted sensor camera creates
the frustum, while the two 3D viewports keep the ellipsoid/debug view.
"""

import elodin as el
import jax.numpy as jnp
import numpy as np

SIM_RATE = 120.0
SENSOR_CAMERA_NAME = "drone.scene_cam"
DRONE_NAME = "drone"
ELLIPSOID_SCALE = np.array([0.9, 0.9, 0.38], dtype=np.float64)
DRONE_PATH_RADIUS = np.array([0.28, 0.2, 0.04], dtype=np.float64)
DRONE_PATH_RATE = 0.35


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
    drone = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.SpatialInertia(mass=1.0),
            ),
        ],
        name=DRONE_NAME,
    )

    world.sensor_camera(
        entity=drone,
        name="scene_cam",
        width=128,
        height=128,
        fov=110.0,
        near=0.01,
        far=0.35,
        pos_offset=[0.0, 0.08, 0.1],
        look_at_offset=[0.0, -0.42, 0.02],
        format="rgba",
        create_frustum=True,
        frustums_color=[1.0, 0.0, 1.0, 1.0],
        projection_color=[1.0, 0.0, 1.0, 1.0],
        frustums_thickness=0.008,
    )

    object_mesh = f"""
    object_3d ellipsoid.world_pos {{
        ellipsoid scale="({ELLIPSOID_SCALE[0]}, {ELLIPSOID_SCALE[1]}, {ELLIPSOID_SCALE[2]})" show_grid=#true {{
            color 0 188 212 28
            grid_color 255 255 255 120
        }}
    }}
    object_3d drone.world_pos {{
        glb path="crazyflie.glb" rotate="(0.0, 0.0, 0.0)" translate="(0.0, 0.0, 0.0)" scale=0.9
    }}
    """

    world.schematic(
        """
        theme mode="dark" scheme="default"

        tabs {
            hsplit name="Frustums" {
                viewport name="Viewport Source" pos="(0,0,0,1, -3,-0.5,2)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true frustums_color="yalk" projection_color="mint" frustums_thickness=0.006 show_grid=#true active=#true near=0.05 far=6.0
                viewport name="Target View" pos="(0,0,0,1, 2,2,1.5)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true show_grid=#true active=#true
                sensor_view "drone.scene_cam" name="Sensor Camera"
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


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def pre_step(tick, ctx):
    t = tick / SIM_RATE
    angle = t * DRONE_PATH_RATE
    pos = np.array(
        [
            DRONE_PATH_RADIUS[0] * np.sin(angle),
            DRONE_PATH_RADIUS[1] * np.sin(angle * 0.7 + 0.8),
            DRONE_PATH_RADIUS[2] * np.sin(angle * 1.3),
        ],
        dtype=np.float64,
    )
    roll = 0.18 * np.sin(angle * 1.9)
    pitch = 0.12 * np.sin(angle * 1.4 + 0.4)
    yaw = angle + 0.35 * np.sin(angle * 0.5)
    quat = _quat_from_euler(roll, pitch, yaw)

    ctx.write_component(
        f"{DRONE_NAME}.world_pos",
        np.array(
            [quat[0], quat[1], quat[2], quat[3], pos[0], pos[1], pos[2]],
            dtype=np.float64,
        ),
    )


def post_step(tick, ctx):
    if tick % 4 == 0:
        ctx.render_camera(SENSOR_CAMERA_NAME)
