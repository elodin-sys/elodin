import os
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

SIMULATION_RATE_HZ = 120.0
DEFAULT_MAX_TICKS = 360
DEFAULT_GRID_SIZE = 262_144
DEFAULT_PROBE_ROWS = 65_536
GRID_SIZE_ENV = "ELODIN_MONTE_CARLO_GRID_SIZE"
PROBE_ROWS_ENV = "ELODIN_MONTE_CARLO_PROBE_ROWS"

PARAMS = el.monte_carlo.params_spec(
    mass=el.monte_carlo.Param(float, default=1.5, min=0.5, max=5.0),
    target_x=el.monte_carlo.Param(float, default=30.0, min=5.0, max=100.0),
    thrust_gain=el.monte_carlo.Param(float, default=1.0, min=0.1, max=4.0),
    wind=el.monte_carlo.Param(float, default=0.0, min=-5.0, max=5.0),
)

Position = ty.Annotated[
    jax.Array,
    el.Component("position", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Velocity = ty.Annotated[
    jax.Array,
    el.Component("velocity", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Command = ty.Annotated[
    jax.Array,
    el.Component(
        "command",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
Target = ty.Annotated[
    jax.Array,
    el.Component("target", el.ComponentType(el.PrimitiveType.F64, (1,))),
]


def lookup_table(size: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack([1.0 + 0.05 * np.sin(x * 20.0), 0.1 + x * 0.01], axis=1)


def build(params: el.monte_carlo.Params) -> tuple[el.World, el.System]:
    world = el.World()
    grid_size = int(os.environ.get(GRID_SIZE_ENV, str(DEFAULT_GRID_SIZE)))
    probe_rows_count = int(os.environ.get(PROBE_ROWS_ENV, str(DEFAULT_PROBE_ROWS)))
    table = jnp.asarray(lookup_table(grid_size))
    probe_base = jnp.asarray(
        np.linspace(0, grid_size - 1, min(probe_rows_count, grid_size), dtype=np.int32)
    )

    mass = float(params.get("mass", 1.5))
    target_x = float(params.get("target_x", 30.0))
    wind = float(params.get("wind", 0.0))
    thrust_gain = float(params.get("thrust_gain", 1.0))

    world.spawn(
        [
            el.C(Position, jnp.array([0.0], dtype=jnp.float64)),
            el.C(Velocity, jnp.array([wind], dtype=jnp.float64)),
            el.C(Command, jnp.array([0.0], dtype=jnp.float64)),
            el.C(Target, jnp.array([target_x], dtype=jnp.float64)),
        ],
        name="vehicle",
    )

    dt = 1.0 / SIMULATION_RATE_HZ

    @el.map
    def point_mass(pos: Position, vel: Velocity, command: Command) -> tuple[Position, Velocity]:
        idx = jnp.clip(
            jnp.abs(vel[0] * 1000.0).astype(jnp.int32),
            0,
            table.shape[0] - 1,
        )
        drag_coeff = table[idx, 0]
        probe_rows = (probe_base + idx) % table.shape[0]
        probe_sum = jnp.sum(table[probe_rows, 0])
        drag = drag_coeff * vel[0] * jnp.abs(vel[0]) * 0.02
        acc = (command[0] * thrust_gain - drag) / mass
        acc = acc + probe_sum * 1e-300
        new_vel = vel + jnp.array([acc * dt])
        new_pos = pos + new_vel * dt
        return new_pos, new_vel

    return world, point_mass
