import jax
import typing
import elodin as el
from jax import numpy as jnp
from jax import random
from jax.numpy import linalg as la
from dataclasses import field

TIME_STEP = 1.0 / 120.0

Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@el.dataclass
class Globals(el.Archetype):
    seed: el.Seed = field(default_factory=lambda: jnp.int64(0))
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


@el.system
def sample_wind(s: el.Query[el.Seed], q: el.Query[Wind]) -> el.Query[Wind]:
    return q.map(
        Wind,
        lambda _w: random.normal(random.key(s[0]), shape=(3,)),
    )


@el.system
def apply_wind(
    w: el.Query[Wind], q: el.Query[el.Force, el.WorldVel]
) -> el.Query[el.Force]:
    def apply_wind_inner(f, v):
        v_diff = w[0] - v.linear()
        v_diff_dir = v_diff / la.norm(v_diff)
        return el.SpatialForce.from_linear(
            f.force() + 0.2 * 0.5 * v_diff**2 * v_diff_dir
        )

    return q.map(
        el.Force,
        apply_wind_inner,
    )


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce.from_linear(
        jnp.array([0.0, inertia.mass() * -9.81, 0.0])
    )


@el.map
def bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[1], v.linear()[1]) < 0.0,
        lambda _: el.SpatialMotion.from_linear(
            v.linear() * jnp.array([1.0, -1.0, 1.0]) * 0.85
        ),
        lambda _: v,
        operand=None,
    )


def world(seed: int = 0) -> el.World:
    world = el.World()
    world.spawn(Globals(seed=jnp.int64(seed)), name="Globals")
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 6.0, 0.0]))
            ),
            world.shape(el.Mesh.sphere(0.4), el.Material.color(12.7, 9.2, 0.5)),
        ],
        name="Ball",
    )
    world.spawn(
        el.Panel.viewport(
            track_rotation=False,
            active=True,
            pos=[6.0, 3.0, 6.0],
            looking_at=[0.0, 1.0, 0.0],
            show_grid=True,
            hdr=True,
        ),
        name="Viewport",
    )
    return world


def system() -> el.System:
    effectors = gravity.pipe(apply_wind)
    sys = sample_wind.pipe(bounce.pipe(el.six_dof(TIME_STEP, effectors)))
    return sys
