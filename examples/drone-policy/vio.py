"""Emulated external position/velocity source (stand-in for a VIO stream).

Truth position and velocity plus a per-episode position bias and white noise, sampled every
VIO_PERIOD ticks (30 Hz) with zero-order hold into a tick-resolution ring buffer, read back with
a per-entity delay. Yaw is truth yaw plus 0.5 deg of noise. Delayed reads use a float one-hot
select instead of an integer gather so the system compiles the same on Cranelift.
"""

import math
import typing as ty
from dataclasses import dataclass, field

import _drone_path  # noqa: F401  (examples/drone on sys.path)
import elodin as el
import jax
import jax.numpy as jnp
import jax.random as rng
import sensors
from dr import yaw_of

VIO_PERIOD = 10  # ticks (30 Hz at 300 Hz)
BUF_LEN = 8  # ticks
YAW_SIGMA = math.radians(0.5)
VIO_KEY = rng.fold_in(rng.key(0), 7)

# vio_params layout: pos bias [3] m, pos sigma m, vel sigma m/s, delay ticks
POS_BIAS = slice(0, 3)
POS_SIGMA = 3
VEL_SIGMA = 4
DELAY = 5

# vio_meas / vio_buf row layout: pos [3], vel [3], yaw
MEAS_DIM = 7

VioParams = ty.Annotated[
    jax.Array,
    el.Component(
        "vio_params",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"transient": "true", "element_names": "bx,by,bz,sp,sv,delay"},
    ),
]
VioSeed = ty.Annotated[
    jax.Array, el.Component("vio_seed", el.ComponentType.U64, metadata={"transient": "true"})
]
VioClock = ty.Annotated[
    jax.Array, el.Component("vio_clock", el.ComponentType.F64, metadata={"transient": "true"})
]
VioBuf = ty.Annotated[
    jax.Array,
    el.Component(
        "vio_buf",
        el.ComponentType(el.PrimitiveType.F64, (BUF_LEN, MEAS_DIM)),
        metadata={"transient": "true"},
    ),
]
VioMeas = ty.Annotated[
    jax.Array,
    el.Component(
        "vio_meas",
        el.ComponentType(el.PrimitiveType.F64, (MEAS_DIM,)),
        metadata={"element_names": "x,y,z,vx,vy,vz,yaw"},
    ),
]


def initial_meas() -> jax.Array:
    return jnp.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class VIO(el.Archetype):
    vio_params: VioParams = field(default_factory=lambda: jnp.zeros(6))
    vio_seed: VioSeed = field(default_factory=lambda: jnp.array(0, dtype=jnp.uint64))
    vio_clock: VioClock = field(default_factory=lambda: jnp.float64(VIO_PERIOD - 1))
    vio_buf: VioBuf = field(default_factory=lambda: jnp.tile(initial_meas(), (BUF_LEN, 1)))
    vio_meas: VioMeas = field(default_factory=initial_meas)


def one_hot(index: jax.Array, n: int) -> jax.Array:
    return jnp.where(jnp.arange(n, dtype=jnp.float64) == index, 1.0, 0.0)


def sample(pos: el.SpatialTransform, vel: el.SpatialMotion, params, key) -> jax.Array:
    n = rng.normal(key, (MEAS_DIM,), dtype=jnp.float64)
    p = pos.linear() + params[POS_BIAS] + params[POS_SIGMA] * n[0:3]
    v = vel.linear() + params[VEL_SIGMA] * n[3:6]
    yaw = yaw_of(pos.angular().vector()) + YAW_SIGMA * n[6]
    return jnp.concatenate([p, v, yaw[None]])


@el.map
def vio_update(
    pos: el.WorldPos,
    vel: el.WorldVel,
    tick: sensors.SensorTick,
    params: VioParams,
    seed: VioSeed,
    clock: VioClock,
    buf: VioBuf,
) -> tuple[VioClock, VioBuf, VioMeas]:
    clock = clock + 1.0
    due = clock >= VIO_PERIOD
    clock = jnp.where(due, 0.0, clock)
    key = rng.fold_in(rng.fold_in(VIO_KEY, seed), tick)
    row = jnp.where(due, sample(pos, vel, params, key), buf[0])
    buf = jnp.concatenate([row[None], buf[:-1]])
    meas = one_hot(params[DELAY], BUF_LEN) @ buf
    return clock, buf, meas
