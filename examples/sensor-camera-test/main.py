#!/usr/bin/env python3
"""
Sensor Camera Test — Bouncing Balls Room

Multiple balls bounce around inside a walled room under gravity.
Two special balls carry sensor cameras:
  - Cyan ball:    RGB camera  (60 fps)
  - Magenta ball: Thermal camera (30 fps)

The cameras look downward from above their host ball, capturing
the room floor and other balls as they move.

Usage:
    elodin editor examples/sensor-camera-test/main.py
    elodin run examples/sensor-camera-test/main.py
"""

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

SIM_TIME_STEP = 1.0 / 120.0
BALL_RADIUS = 0.3
BOUNDARY = 5.0
BOUNCINESS = 0.95
FRICTION = 0.05

# ── Systems ──────────────────────────────────────────────────────────────────


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, 0.0, -9.81]))


@el.map
def ground_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    pos = p.linear()
    vel = v.linear()
    new_vz = jax.lax.cond(
        jnp.logical_and(pos[2] < BALL_RADIUS, vel[2] < 0.0),
        lambda _: -vel[2] * BOUNCINESS,
        lambda _: vel[2],
        operand=None,
    )
    return el.SpatialMotion(linear=jnp.array([vel[0], vel[1], new_vz]))


@el.map
def wall_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    pos = p.linear()
    vel = v.linear()

    vel_x = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[0] > BOUNDARY, vel[0] > 0),
            jnp.logical_and(pos[0] < -BOUNDARY, vel[0] < 0),
        ),
        lambda _: -vel[0] * BOUNCINESS,
        lambda _: vel[0],
        operand=None,
    )
    vel_y = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[1] > BOUNDARY, vel[1] > 0),
            jnp.logical_and(pos[1] < -BOUNDARY, vel[1] < 0),
        ),
        lambda _: -vel[1] * BOUNCINESS,
        lambda _: vel[1],
        operand=None,
    )

    return el.SpatialMotion(linear=jnp.array([vel_x, vel_y, vel[2]]))


@el.map
def damping(v: el.WorldVel, f: el.Force) -> el.Force:
    return el.SpatialForce(linear=f.force() - FRICTION * v.linear())


# ── World ────────────────────────────────────────────────────────────────────

world = el.World()

BALL_DEFS = [
    # (name, position, velocity)
    ("cam_ball_a", [-3.0, -3.0, 4.0], [2.0, 3.0, 1.0]),
    ("cam_ball_b", [3.0, 3.0, 5.0], [-1.5, -2.0, 0.5]),
    ("ball_1", [0.0, 0.0, 6.0], [1.0, -1.0, 0.0]),
    ("ball_2", [-2.0, 2.0, 3.0], [3.0, 1.0, 2.0]),
    ("ball_3", [1.0, -3.0, 7.0], [-2.0, 2.0, -1.0]),
]

entities = {}
for name, pos, vel in BALL_DEFS:
    e = world.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array(pos)),
            world_vel=el.SpatialMotion(linear=jnp.array(vel)),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        name=name,
    )
    entities[name] = e

cam_ball_a = entities["cam_ball_a"]
cam_ball_b = entities["cam_ball_b"]

# ── Sensor Cameras ───────────────────────────────────────────────────────────

world.sensor_camera(
    entity=cam_ball_a,
    name="scene_cam",
    width=640,
    height=480,
    fov=90.0,
    pos_offset=[0.0, 0.0, 0.5],
    look_at_offset=[6.0, 6.0, 0.0],
    format="rgba",
    fps=60.0,
)

world.sensor_camera(
    entity=cam_ball_b,
    name="thermal_cam",
    width=128,
    height=128,
    fov=90.0,
    pos_offset=[0.0, 0.0, 0.5],
    look_at_offset=[-6.0, -6.0, 0.0],
    format="rgba",
    fps=30.0,
    effect="thermal",
    effect_params={"contrast": 1.5, "noise_sigma": 0.02},
)

# ── Schematic ────────────────────────────────────────────────────────────────

BALL_COLORS = {
    "cam_ball_a": "0 220 220",      # cyan
    "cam_ball_b": "220 0 220",      # magenta
    "ball_1": "255 140 0",          # orange
    "ball_2": "255 255 100",        # yellow
    "ball_3": "100 255 100",        # green
}

object_3d_lines = []
for name, color in BALL_COLORS.items():
    object_3d_lines.append(
        f'    object_3d {name}.world_pos {{ sphere radius={BALL_RADIUS} {{ color {color} }} }}'
    )

schematic = """
    hsplit {{
        viewport name=Main pos="(0,0,0,0, 14,14,10)" look_at="(0,0,0,0, 0,0,1)" show_grid=#true
        vsplit {{
            sensor_view "cam_ball_a.scene_cam" name="RGB Camera (Cyan Ball)"
            sensor_view "cam_ball_b.thermal_cam" name="Thermal (Magenta Ball)"
        }}
    }}
{objects}
    object_3d "(0,0,0,1, 0,0,0)" {{
        plane width=12 depth=12 {{ color 60 120 60 }}
    }}
""".format(objects="\n".join(object_3d_lines))

world.schematic(schematic, "sensor-camera-test.kdl")

# ── System composition ───────────────────────────────────────────────────────

constraints = ground_bounce | wall_bounce
effectors = gravity | damping
system = constraints | el.six_dof(sys=effectors)

# ── Post-step frame counter ──────────────────────────────────────────────────

rgb_frames = [0]
thermal_frames = [0]


def post_step(tick, ctx):
    try:
        frame = ctx.read_msg("cam_ball_a.scene_cam")
        if frame is not None and len(frame) > 0:
            rgb_frames[0] += 1
            if rgb_frames[0] == 1:
                arr = np.array(frame)
                print(
                    f"[RGB] First frame at tick {tick}: {len(frame)} bytes, "
                    f"nonzero={np.count_nonzero(arr)}"
                )
    except Exception:
        pass

    try:
        frame = ctx.read_msg("cam_ball_b.thermal_cam")
        if frame is not None and len(frame) > 0:
            thermal_frames[0] += 1
            if thermal_frames[0] == 1:
                arr = np.array(frame)
                print(
                    f"[THERMAL] First frame at tick {tick}: {len(frame)} bytes, "
                    f"nonzero={np.count_nonzero(arr)}"
                )
    except Exception:
        pass

    if tick > 0 and tick % 600 == 0:
        print(f"  tick {tick}: rgb={rgb_frames[0]} thermal={thermal_frames[0]} frames")


# ── Run ──────────────────────────────────────────────────────────────────────

world.run(
    system,
    sim_time_step=SIM_TIME_STEP,
    run_time_step=SIM_TIME_STEP,
    max_ticks=36000,
    post_step=post_step,
    interactive=False,
)

print(f"\nRGB frames: {rgb_frames[0]}, Thermal frames: {thermal_frames[0]}")
