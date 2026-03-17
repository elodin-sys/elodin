#!/usr/bin/env python3
"""
Log Streaming Example

Demonstrates text log ingestion into Elodin DB and live display in the
Editor's log viewer panel.

A C++ log client (libs/db/examples/log-client.cpp) is compiled and launched
automatically via S10.  It sends structured LogEntry messages that simulate
flight software output — boot sequence, flight telemetry, warnings, and
errors — which appear in real time in the "Flight Software Log" panel.

Usage:
    elodin editor examples/logstream/main.py
"""

import typing
from dataclasses import field
from pathlib import Path

import elodin as el
import jax
from jax import numpy as jnp

SIM_TIME_STEP = 1.0 / 120.0
BALL_RADIUS = 0.3
BOUNDARY = 4.0
BOUNCINESS = 0.95
FRICTION = 0.4

Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@el.dataclass
class WindData(el.Archetype):
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


WIND_ROTATION_PERIOD = 360
WIND_SPEED = 8.0


@el.system
def sample_wind(
    tick: el.Query[el.SimulationTick],
    w: el.Query[Wind],
) -> el.Query[Wind]:
    angle = (tick[0] / WIND_ROTATION_PERIOD) * 2.0 * jnp.pi
    wind_vec = jnp.array([jnp.cos(angle) * WIND_SPEED, jnp.sin(angle) * WIND_SPEED, 0.0])
    return w.map(Wind, lambda _: wind_vec)


@el.map
def wall_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    pos = p.linear()
    vel = v.linear()
    ang = v.angular()

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

    return el.SpatialMotion(angular=ang, linear=jnp.array([vel_x, vel_y, 0.0]))


@el.map
def rolling_motion(v: el.WorldVel) -> el.WorldVel:
    vel = v.linear()
    n = jnp.array([0.0, 0.0, 1.0])
    omega = jnp.cross(n, vel) / BALL_RADIUS
    return el.SpatialMotion(angular=omega, linear=vel)


WIND_FORCE_COEFFICIENT = 3.0


@el.map
def apply_wind(w: Wind, f: el.Force) -> el.Force:
    wind_force = w * WIND_FORCE_COEFFICIENT
    return el.SpatialForce(linear=f.force() + wind_force)


@el.map
def friction(v: el.WorldVel, f: el.Force) -> el.Force:
    vel = v.linear()
    friction_force = -FRICTION * vel
    return el.SpatialForce(linear=f.force() + friction_force)


world = el.World()

ball = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, BALL_RADIUS])),
            world_vel=el.SpatialMotion(
                angular=jnp.array([0.0, 0.0, 0.0]),
                linear=jnp.array([3.0, 2.0, 0.0]),
            ),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        WindData(),
    ],
    name="ball",
)

# Launch the C++ log client via S10
log_script = Path(__file__).parent / "build-and-run.sh"
log_client = el.s10.PyRecipe.process(
    name="log-client",
    cmd="bash",
    args=[str(log_script)],
    cwd=str(Path(__file__).parent),
)
world.recipe(log_client)

world.schematic("""
    hsplit {
        tabs share=0.5 {
            log_stream "fsw.log" name="Flight Software Log"
        }
        vsplit share=0.5 {
            tabs {
                viewport name=Viewport pos="(0,0,0,0, 0,0,12)" look_at="(0,0,0,0, 0,0,0)" show_grid=#true
            }
            graph "ball.wind" name="Wind (m/s)"
        }
    }
    object_3d ball.world_pos {
        sphere radius=0.3 {
            color orange
        }
    }
    object_3d "(0,0,0,1, 0,0,0)" {
        plane width=10 depth=10 {
            color 32 128 32 200
        }
    }
    vector_arrow "ball.wind" origin="(0,0,0,1, 0,0,0.5)" scale=0.3 name="Wind" show_name=#true {
        color cyan 200
    }
""")

print("Log Streaming Example")
print("=====================")
print()
print("A ball rolls around pushed by rotating wind, bouncing off walls.")
print("The log panel shows simulated flight software messages from a C++ client.")
print()

constraints = sample_wind | wall_bounce | rolling_motion
effectors = apply_wind | friction
sys = constraints | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)

sim = world.run(
    sys,
    simulation_rate=1.0 / SIM_TIME_STEP,
    generate_real_time=True,
    start_timestamp=0,
    db_path="./logstream-db",
)
