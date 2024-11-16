import typing
from dataclasses import field

import elodin as el
import jax
from jax import numpy as jnp
from jax import random
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0

BALL_RADIUS = 0.2


def world(seed: int = 0) -> el.World:
    world = el.World()
    # world.spawn(WindData(seed=jnp.int64(seed)), name="WindData")
    ball = world.spawn(
        [
            el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 6.0]))),
            world.shape(el.Mesh.sphere(BALL_RADIUS), el.Material.color(12.7, 9.2, 0.5)),
            WindData(seed=jnp.int64(seed)),
        ],
        name="Ball",
    )
    world.spawn(
        el.Panel.viewport(
            track_rotation=False,
            active=True,
            pos=[8.0, 2.0, 4.0],
            looking_at=[0.0, 0.0, 3.0],
            show_grid=True,
            hdr=True,
        ),
        name="Viewport",
    )
    world.spawn(el.Line3d(ball, "world_pos", index=[4, 5, 6], line_width=2.0))
    return world


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


BOUNCINESS = 0.85


@el.map
def bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[2], v.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(linear=v.linear() * jnp.array([1.0, 1.0, -1.0]) * BOUNCINESS),
        lambda _: v,
        operand=None,
    )


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
    seed: el.Seed = field(default_factory=lambda: jnp.int64(0))
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


@el.map
def sample_wind(s: el.Seed, _w: Wind) -> Wind:
    return random.normal(random.key(s), shape=(3,))


def calculate_drag(Cd, r, V, A):
    return 0.5 * (Cd * r * V**2 * A)


@el.map
def apply_drag(w: Wind, v: el.WorldVel, f: el.Force) -> el.Force:
    fluid_movement_vector = w
    fluid_movement_vector -= v.linear()

    ball_drag_coefficient = 0.5
    fluid_density = 1.225
    fluid_velocity = la.norm(fluid_movement_vector)
    ball_surface_area = 2 * 3.1415 * BALL_RADIUS**2

    drag_force = calculate_drag(
        ball_drag_coefficient, fluid_density, fluid_velocity, ball_surface_area
    )

    fluid_vector_direction = fluid_movement_vector / fluid_velocity
    return el.SpatialForce(linear=f.force() + drag_force * fluid_vector_direction)


def system() -> el.System:
    effectors = gravity | apply_drag
    sys = sample_wind | bounce | el.six_dof(sys=effectors)
    return sys
