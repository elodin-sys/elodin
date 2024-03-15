import jax
import typing
import elodin as el
from jax import numpy as jnp
from jax import random

TIME_STEP = 1.0 / 120.0
G = 6.6743e-11
R = 6.378e6
M = 5.972e24

Wind = typing.Annotated[
    el.SpatialForce, el.Component("wind", el.ComponentType.SpatialMotionF64)
]


@el.dataclass
class Globals(el.Archetype):
    seed: el.Seed
    wind: Wind = el.SpatialForce.zero()


@el.system
def sample_wind(s: el.Query[el.Seed], q: el.Query[Wind]) -> el.Query[Wind]:
    return q.map(
        Wind,
        lambda _w: el.Force.from_linear(
            0.2 * random.normal(random.key(s[0]), shape=(3,))
        ),
    )


@el.system
def apply_wind(w: el.Query[Wind], q: el.Query[el.Force]) -> el.Query[el.Force]:
    return q.map(el.Force, lambda f: el.Force.from_linear(f.force() + w[0].force()))


@el.system
def gravity(q: el.Query[el.Force, el.Inertia]) -> el.Query[el.Force]:
    def gravity_inner(force, inertia):
        m = inertia.mass()
        f = G * M * m / R**2
        return el.Force.from_linear(force.force() + jnp.array([0.0, -f, 0.0]))

    return q.map(el.Force, gravity_inner)


@el.system
def bounce(q: el.Query[el.WorldPos, el.WorldVel]) -> el.Query[el.WorldVel]:
    return q.map(
        el.WorldVel,
        lambda p, v: jax.lax.cond(
            jax.lax.max(p.linear()[1], v.linear()[1]) < 0.0,
            lambda _: el.WorldVel.from_linear(
                v.linear() * jnp.array([1.0, -1.0, 1.0]) * 0.85
            ),
            lambda _: v,
            operand=None,
        ),
    )


w = el.WorldBuilder()
w.spawn(Globals(seed=jnp.int64(1))).metadata(el.EntityMetadata("Globals"))
w.spawn(
    el.Body(
        world_pos=el.WorldPos.from_linear(jnp.array([0.0, 6.0, 0.0])),
        pbr=w.insert_asset(
            el.Pbr(el.Mesh.sphere(0.4), el.Material.color(12.7, 9.2, 0.5))
        ),
    )
).metadata(el.EntityMetadata("Ball"))
effectors = gravity.pipe(apply_wind)
sys = sample_wind.pipe(bounce.pipe(el.six_dof(TIME_STEP, effectors)))
w.run(sys, TIME_STEP)
