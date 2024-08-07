import elodin as el
from dataclasses import field, dataclass
import jax
import typing as ty
import jax.numpy as jnp
import jax.random as rng

import filter
import params
from config import Config

enable_sensor_noise = True

# sensors
SensorTick = ty.Annotated[jax.Array, el.Component("sensor_tick", el.ComponentType.U64)]
Gyro = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 90, "element_names": "x,y,z"},
    ),
]
GyroBias = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
GyroLPFDelay = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_lpf_delay",
        el.ComponentType(el.PrimitiveType.F64, (4, 3)),
    ),
]
Accel = ty.Annotated[
    jax.Array,
    el.Component(
        "accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 89, "element_names": "x,y,z"},
    ),
]
AccelHealth = ty.Annotated[
    jax.Array,
    el.Component("accel_health", el.ComponentType.F64, metadata={"priority": 88}),
]
AccelBias = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
# idealized accelerometer that only considers gravity
GravityAccel = ty.Annotated[
    jax.Array,
    el.Component(
        "gravity_accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
AccelLPFDelay = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_lpf_delay",
        el.ComponentType(el.PrimitiveType.F64, (4, 3)),
    ),
]
Magnetometer = ty.Annotated[
    jax.Array,
    el.Component(
        "magnetometer",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 87, "element_names": "x,y,z"},
    ),
]
MagnetometerBias = ty.Annotated[
    jax.Array,
    el.Component(
        "magnetometer_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


class Noise:
    def __init__(
        self,
        seed: int,
        device: int,
        noise_covariance: float,
        bias_drift_covariance: float,
    ):
        self.noise_covariance = noise_covariance
        self.bias_drift_covariance = bias_drift_covariance
        self.key = rng.fold_in(rng.key(seed), device)

    def drift_bias(self, bias: jax.Array, tick: SensorTick, dt: float) -> jax.Array:
        key = rng.fold_in(self.key, tick)
        std_dev = jnp.sqrt(self.bias_drift_covariance)
        drift = std_dev * rng.normal(key, shape=bias.shape, dtype=bias.dtype) * dt
        return bias + drift

    def sample(self, m: jax.Array, bias: jax.Array, tick: SensorTick) -> jax.Array:
        key = rng.fold_in(self.key, tick)
        std_dev = jnp.sqrt(self.noise_covariance)
        noise = std_dev * rng.normal(key, shape=m.shape, dtype=m.dtype)
        return m + noise + bias


gyro_noise = Noise(0, 0, 0.001, 0.001)
init_gyro_bias = jnp.array([0.0025, 0.0001, 0.0005])
accel_noise = Noise(0, 1, 0.001, 0.0)
mag_noise = Noise(0, 2, 0.0001, 0.0)


@dataclass
class IMU(el.Archetype):
    sensor_tick: SensorTick = field(default_factory=lambda: jnp.array(0))
    gyro: Gyro = field(default_factory=lambda: jnp.zeros(3))
    gyro_bias: GyroBias = field(default_factory=lambda: jnp.array(init_gyro_bias))
    accel: Accel = field(default_factory=lambda: jnp.zeros(3))
    accel_health: AccelHealth = field(default_factory=lambda: jnp.float64(1.0))
    accel_bias: AccelBias = field(default_factory=lambda: jnp.zeros(3))
    mag: Magnetometer = field(default_factory=lambda: jnp.array([0.0, 1.0, 0.0]))
    mag_bias: MagnetometerBias = field(default_factory=lambda: jnp.zeros(3))
    grav_accel: GravityAccel = field(default_factory=lambda: jnp.zeros(3))
    gyro_lpf_delay: GyroLPFDelay = field(default_factory=lambda: jnp.zeros((4, 3)))
    accel_lpf_delay: AccelLPFDelay = field(default_factory=lambda: jnp.zeros((4, 3)))


@el.map
def advance_sensor_tick(tick: SensorTick) -> SensorTick:
    return tick + 1


@el.map
def update_gyro_noise(tick: SensorTick, bias: GyroBias) -> GyroBias:
    dt = Config.GLOBAL.fast_loop_time_step
    return gyro_noise.drift_bias(bias, tick, dt)


@el.map
def gyro(
    tick: SensorTick,
    p: el.WorldPos,
    v: el.WorldVel,
    delay: GyroLPFDelay,
    bias: GyroBias,
) -> tuple[GyroLPFDelay, Gyro]:
    dt = Config.GLOBAL.fast_loop_time_step
    body_v = p.angular().inverse() @ v.angular()
    if enable_sensor_noise:
        body_v = gyro_noise.sample(body_v, bias, tick)
    lpf = filter.BiquadLPF(params.INS_GYRO_FILTER, 1.0 / dt)
    new_delay = lpf.apply(delay, body_v)
    return (new_delay, new_delay[2])


@el.map
def accel(
    tick: SensorTick,
    p: el.WorldPos,
    a: el.WorldAccel,
    delay: AccelLPFDelay,
    bias: AccelBias,
) -> tuple[AccelLPFDelay, Accel]:
    dt = Config.GLOBAL.fast_loop_time_step
    body_a = p.angular().inverse() @ (a.linear() / 9.81 + jnp.array([0, 0, 1]))
    if enable_sensor_noise:
        body_a = accel_noise.sample(body_a, bias, tick)
    lpf = filter.BiquadLPF(params.INS_ACCEL_FILTER, 1.0 / dt)
    new_delay = lpf.apply(delay, body_a)
    return (new_delay, new_delay[2])


@el.map
def mag(
    tick: SensorTick,
    p: el.WorldPos,
    bias: MagnetometerBias,
    prev_mag: Magnetometer,
) -> Magnetometer:
    dt = Config.GLOBAL.fast_loop_time_step
    data_rate = 1.0 / 100.0
    tick_rate = round(data_rate / dt)
    assert tick_rate == 9
    body_mag_ref = p.angular().inverse() @ jnp.array([0.0, 1.0, 0.0])
    if enable_sensor_noise:
        body_mag_ref = mag_noise.sample(body_mag_ref, bias, tick)
    return jax.lax.cond(
        tick % tick_rate == 0,
        lambda _: body_mag_ref,
        lambda _: prev_mag,
        None,
    )


@el.map
def accel_health(accel: Accel, gyro: Gyro) -> AccelHealth:
    health = 1.0

    # 0.5 g deviation is considered maximally unhealthy
    accel_norm = jnp.linalg.norm(accel)
    accel_deviation = jnp.abs(accel_norm - 1)
    health *= 1.0 - jnp.clip(accel_deviation / 0.5, 0.0, 1.0)

    # 0.5 rad/s angular acceleration is considered maximally unhealthy
    gyro_norm = jnp.linalg.norm(gyro)
    health *= 1.0 - jnp.clip(gyro_norm / 0.5, 0.0, 1.0)

    return health


@el.map
def gravity_accel(p: el.WorldPos, a: Accel) -> GravityAccel:
    return p.angular().inverse() @ jnp.array([0.0, 0.0, 1.0])


imu = (
    advance_sensor_tick
    | update_gyro_noise
    | gyro
    | accel
    | accel_health
    | mag
    | gravity_accel
)
