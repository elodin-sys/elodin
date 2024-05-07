import jax
import typing
import elodin as el
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from jax.numpy import linalg as la

TIME_STEP = 1.0 / 120.0
G = 6.6743e-11
R = 6.378e6
M = 5.972e24

Wind = typing.Annotated[
    el.SpatialMotion, el.Component("wind", el.ComponentType.SpatialMotionF64)
]


@el.dataclass
class Globals(el.Archetype):
    seed: el.Seed
    wind: Wind = el.SpatialMotion.zero()


@el.system
def sample_wind(s: el.Query[el.Seed], q: el.Query[Wind]) -> el.Query[Wind]:
    return q.map(
        Wind,
        lambda _w: el.Force.from_linear(random.normal(random.key(s[0]), shape=(3,))),
    )


@el.system
def apply_wind(
    w: el.Query[Wind], q: el.Query[el.Force, el.WorldVel]
) -> el.Query[el.Force]:
    def apply_wind_inner(f, v):
        v_diff = w[0].linear() - v.linear()
        v_diff_dir = v_diff / la.norm(v_diff)
        return el.Force.from_linear(f.force() + 0.2 * 0.5 * v_diff**2 * v_diff_dir)

    return q.map(
        el.Force,
        apply_wind_inner,
    )


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


def run(seed: int) -> pl.DataFrame:
    w = el.WorldBuilder()
    w.spawn(Globals(seed=jnp.int64(seed)), name="Globals")
    w.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 6.0, 0.0])),
            ),
            w.shape(el.Mesh.sphere(0.4), el.Material.color(12.7, 9.2, 0.5)),
        ],
        name="Ball",
    )
    effectors = gravity.pipe(apply_wind)
    sys = sample_wind.pipe(bounce.pipe(el.six_dof(TIME_STEP, effectors)))
    exec = w.build(sys)

    client = el.Client.cpu()
    for _ in range(1200):
        exec.run(client)
    return exec.history()

fig, ax = plt.subplots()

for i in range(0, 20):
    df = run(i)
    df = df.sort("tick").select(["tick", "world_pos"]).drop_nulls()
    df = df.with_columns(
        pl.col("world_pos").arr.get(4).alias("x"),
        pl.col("world_pos").arr.get(5).alias("y"),
        pl.col("world_pos").arr.get(6).alias("z"),
    )
    distance = np.linalg.norm(df.select(["x", "y", "z"]).to_numpy(), axis=1)
    df = df.with_columns(pl.Series(distance).alias("distance"))
    ax.plot(df["tick"], df["distance"])

plt.show()
