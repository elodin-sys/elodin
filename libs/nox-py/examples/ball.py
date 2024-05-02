import jax
import typing
import elodin as el
from jax import numpy as jnp
from jax import random
import polars as pl
import numpy as np
from typing import cast
from jax.numpy import linalg as la

TIME_STEP = 1.0 / 120.0

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


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force.from_linear(jnp.array([0.0, inertia.mass() * -9.81, 0.0]))


@el.map
def bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[1], v.linear()[1]) < 0.0,
        lambda _: el.WorldVel.from_linear(
            v.linear() * jnp.array([1.0, -1.0, 1.0]) * 0.85
        ),
        lambda _: v,
        operand=None,
    )


w = el.World()
w.spawn(Globals(seed=jnp.int64(0)), name="Globals")
w.spawn(
    el.Body(
        world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 6.0, 0.0])),
        pbr=w.insert_asset(
            el.Pbr(el.Mesh.sphere(0.4), el.Material.color(12.7, 9.2, 0.5))
        ),
    ),
    name="Ball",
)
w.spawn(
    el.Panel.viewport(
        track_rotation=False,
        active=True,
        pos=[6.0, 3.0, 6.0],
        looking_at=[0.0, 1.0, 0.0],
        show_grid=True,
    ),
    name="Viewport",
)
effectors = gravity.pipe(apply_wind)
sys = sample_wind.pipe(bounce.pipe(el.six_dof(TIME_STEP, effectors)))
w.run(sys)


def test_origin_drift(df: pl.DataFrame):
    world_pos_id = str(el.Component.id(el.WorldPos))
    df = df.sort("time").select(["time", world_pos_id]).drop_nulls()
    df = df.with_columns(
        pl.col(world_pos_id).arr.get(4).alias("x"),
        pl.col(world_pos_id).arr.get(5).alias("y"),
        pl.col(world_pos_id).arr.get(6).alias("z"),
    )
    distance = np.linalg.norm(df.select(["x", "z"]).to_numpy(), axis=1)
    df = df.with_columns(pl.Series(distance).alias("distance"))
    max_dist = cast(int, df["distance"].max())
    assert max_dist < 2
