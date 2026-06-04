#!/usr/bin/env uv run

import os
import time
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

SIMULATION_RATE_HZ = 120.0
DEFAULT_MAX_TICKS = 600
PROBE_ROWS = 4096
GRID_SIZE_ENV = "ELODIN_MONTE_CARLO_GRID_SIZE"
SEED_ENV = "ELODIN_MONTE_CARLO_SEED"
DB_PATH_ENV = "ELODIN_DB_PATH"
HOLD_AFTER_RUN_ENV = "ELODIN_MONTE_CARLO_HOLD_AFTER_RUN_SEC"
DEFAULT_GRID_SIZE = 16_777_216


def aero_grid(size: int) -> np.ndarray:
    mach = np.linspace(0.0, 5.0, size, dtype=np.float64)
    cd = 0.35 + 0.08 * np.sin(mach * 2.0)
    area = 0.015 + 0.002 * np.cos(mach * 3.0)
    return np.stack([cd, area], axis=1)


Wind = ty.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


def build(seed: int, grid_size: int) -> tuple[el.World, el.System]:
    world = el.World()
    aero_table = jnp.asarray(aero_grid(grid_size))
    probe_base = jnp.asarray(
        np.linspace(0, grid_size - 1, min(PROBE_ROWS, grid_size), dtype=np.int32)
    )
    rng = np.random.default_rng(seed)
    wind = jnp.array(rng.normal(loc=0.0, scale=1.0, size=3), dtype=jnp.float64)

    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 100.0])),
                world_vel=el.SpatialMotion(linear=jnp.array([80.0, 0.0, 0.0])),
            ),
            el.C(Wind, wind),
        ],
        name="vehicle",
    )

    @el.map
    def gravity(force: el.Force, inertia: el.Inertia) -> el.Force:
        return force + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())

    @el.map
    def aero_drag(wind: Wind, vel: el.WorldVel, force: el.Force) -> el.Force:
        table = aero_table
        rel_vel = wind - vel.linear()
        speed = jnp.linalg.norm(rel_vel)
        mach = speed / 340.0
        idx = jnp.clip(
            (mach / 5.0 * (table.shape[0] - 1)).astype(jnp.int32),
            0,
            table.shape[0] - 1,
        )
        cd = table[idx, 0]
        area = table[idx, 1]
        probe_rows = (probe_base + idx) % table.shape[0]
        probe_sum = jnp.sum(table[probe_rows, 0])
        drag_mag = 0.5 * 1.225 * cd * area * speed * speed
        drag_mag = drag_mag + probe_sum * 1e-300
        direction = rel_vel / jnp.maximum(speed, 1e-9)
        return force + el.SpatialForce(linear=drag_mag * direction)

    return world, el.six_dof(sys=gravity | aero_drag)


seed = int(os.environ.get(SEED_ENV, "0"))
grid_size = int(os.environ.get(GRID_SIZE_ENV, str(DEFAULT_GRID_SIZE)))
world, system = build(seed=seed, grid_size=grid_size)
world.run(
    system,
    simulation_rate=SIMULATION_RATE_HZ,
    max_ticks=DEFAULT_MAX_TICKS,
    db_path=os.environ.get(DB_PATH_ENV),
)
hold_after_run = float(os.environ.get(HOLD_AFTER_RUN_ENV, "0"))
if hold_after_run > 0:
    time.sleep(hold_after_run)
