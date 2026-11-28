"""Domain randomization and randomized throws (pure jax.numpy, vmap-able over keys).

Every range is written as nominal n and a full-scale draw u; a sample at curriculum scale s is
n + s * (u - n), so s = 0 gives the stock vehicle (dr_sample) or a start at rest at the target
in the hover trim attitude with random yaw (throw_sample), s = 1 the full ranges. Integer delays
use floor(s * U(0, max + 1)).

Plant DR is kept feasible: the stock Talon's full-throttle thrust-to-weight is only ~1.33
(29.2k rpm max vs 25.4k at hover). Mass x U(0.8, 1.2) with per-motor thrust x U(0.9, 1.1)
reached T/W ~1.0, where B0 recovered 0.4% of throws below T/W 1.25; mass x U(0.9, 1.1) with
thrust x U(0.95, 1.05) keeps the worst case at ~1.15 (~1.11 against the full -1 N z
disturbance force). Whether the stock motor model has the real vehicle's thrust margin is a
fidelity question for the Elodin model.
"""

import math
import typing as ty

import jax
import jax.numpy as jnp
import plant_dr

TARGET = (0.0, 0.0, 2.0)
STOCK_GYRO_BIAS = (0.0025, 0.0001, 0.0005)
G = 9.81
MAX_ACT_DELAY = 2
MAX_VIO_DELAY = 6
MAX_TILT = math.radians(30.0)
MAX_EST_ERR = math.radians(5.0)


class DRParams(ty.NamedTuple):
    mass_scale: jax.Array  # x stock mass
    inertia_scale: jax.Array  # [3] x stock inertia diagonal
    thrust_scale: jax.Array  # [4] per motor
    kq_scale: jax.Array  # reaction torque, 1 = stock
    motor_tau: jax.Array  # s
    drag_coeff: jax.Array  # F = c (w - v)|w - v|
    wind: jax.Array  # [3] m/s world
    force: jax.Array  # [3] N world
    torque: jax.Array  # [3] N m world
    rate_gain_scale: jax.Array  # [3] scales rate PID P, I, D per axis
    gyro_bias: jax.Array  # [3] rad/s true initial gyro bias
    act_delay: jax.Array  # int ticks of extra action latency
    vio_delay: jax.Array  # int ticks
    vio_pos_bias: jax.Array  # [3] m
    vio_pos_sigma: jax.Array  # m
    vio_vel_sigma: jax.Array  # m/s


class Throw(ty.NamedTuple):
    quat: jax.Array  # [4] xyzw initial attitude
    pos_offset: jax.Array  # [3] m from the target
    vel: jax.Array  # [3] m/s world
    body_rate: jax.Array  # [3] rad/s body FLU
    rpm_scale: jax.Array  # [4] x trim hover rpm (plant_dr.hover_trim)
    est_err: jax.Array  # [3] rotation vector, attitude_estimate = quat * exp(est_err)


def trim_quat() -> jax.Array:
    """Zero-yaw hover attitude: the net motor thrust is vertical (plant_dr.hover_trim)."""
    return jnp.asarray(plant_dr.hover_trim().quat)


def nominal_dr() -> DRParams:
    f = jnp.float64
    return DRParams(
        mass_scale=f(1.0),
        inertia_scale=jnp.ones(3),
        thrust_scale=jnp.ones(4),
        kq_scale=f(1.0),
        motor_tau=f(0.1),
        drag_coeff=f(0.1),
        wind=jnp.zeros(3),
        force=jnp.zeros(3),
        torque=jnp.zeros(3),
        rate_gain_scale=jnp.ones(3),
        gyro_bias=jnp.array(STOCK_GYRO_BIAS),
        act_delay=jnp.int32(0),
        vio_delay=jnp.int32(0),
        vio_pos_bias=jnp.zeros(3),
        vio_pos_sigma=f(0.0),
        vio_vel_sigma=f(0.0),
    )


def nominal_throw() -> Throw:
    return Throw(
        quat=trim_quat(),
        pos_offset=jnp.zeros(3),
        vel=jnp.zeros(3),
        body_rate=jnp.zeros(3),
        rpm_scale=jnp.ones(4),
        est_err=jnp.zeros(3),
    )


def _lerp(n, u, s):
    return n + s * (u - n)


def _delay(key, s, max_delay):
    d = jnp.floor(s * jax.random.uniform(key, (), maxval=max_delay + 1.0))
    return jnp.clip(d, 0, max_delay).astype(jnp.int32)


def dr_sample(key: jax.Array, dr_scale) -> DRParams:
    s = jnp.asarray(dr_scale, jnp.float64)
    k = jax.random.split(key, 18)
    u = jax.random.uniform
    wind_speed = u(k[6], (), maxval=2.0)
    wind_dir = u(k[7], (), maxval=2 * jnp.pi)
    wind = jnp.array(
        [
            wind_speed * jnp.cos(wind_dir),
            wind_speed * jnp.sin(wind_dir),
            u(k[8], (), minval=-0.3, maxval=0.3),
        ]
    )
    log_tau = u(k[4], (), minval=jnp.log(0.04), maxval=jnp.log(0.15))
    return DRParams(
        mass_scale=_lerp(1.0, u(k[0], (), minval=0.9, maxval=1.1), s),
        inertia_scale=_lerp(1.0, u(k[1], (3,), minval=0.8, maxval=1.2), s),
        thrust_scale=_lerp(1.0, u(k[2], (4,), minval=0.95, maxval=1.05), s),
        kq_scale=_lerp(1.0, u(k[3], (), minval=0.5, maxval=2.0), s),
        motor_tau=_lerp(0.1, jnp.exp(log_tau), s),
        drag_coeff=_lerp(0.1, u(k[5], (), minval=0.05, maxval=0.2), s),
        wind=s * wind,
        force=s * u(k[9], (3,), minval=-1.0, maxval=1.0),
        torque=s * u(k[10], (3,), minval=-0.05, maxval=0.05),
        rate_gain_scale=_lerp(1.0, u(k[11], (3,), minval=0.8, maxval=1.25), s),
        gyro_bias=_lerp(jnp.array(STOCK_GYRO_BIAS), u(k[12], (3,), minval=-0.01, maxval=0.01), s),
        act_delay=_delay(k[13], s, MAX_ACT_DELAY),
        vio_delay=_delay(k[14], s, MAX_VIO_DELAY),
        vio_pos_bias=s * u(k[15], (3,), minval=-0.02, maxval=0.02),
        vio_pos_sigma=s * u(k[16], (), maxval=0.02),
        vio_vel_sigma=s * u(k[17], (), maxval=0.05),
    )


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product of xyzw quaternions (same convention as el.Quaternion.__mul__)."""
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    v = aw * bv + bw * av + jnp.cross(av, bv)
    return jnp.concatenate([v, (aw * bw - jnp.dot(av, bv))[None]])


def quat_from_rotvec(r: jax.Array) -> jax.Array:
    angle = jnp.linalg.norm(r)
    safe = jnp.where(angle > 1e-12, angle, 1.0)
    axis = jnp.where(angle > 1e-12, r / safe, jnp.array([1.0, 0.0, 0.0]))
    return jnp.concatenate([axis * jnp.sin(angle / 2), jnp.cos(angle / 2)[None]])


def quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate v by xyzw quaternion q (body to world for an attitude quaternion)."""
    qv, qw = q[:3], q[3]
    t = 2.0 * jnp.cross(qv, v)
    return v + qw * t + jnp.cross(qv, t)


def _unit_vector(key: jax.Array) -> jax.Array:
    v = jax.random.normal(key, (3,))
    return v / jnp.maximum(jnp.linalg.norm(v), 1e-12)


def min_start_z(vz: jax.Array, horizon: float = 0.3, floor: float = 0.6) -> jax.Array:
    """Lowest start height whose ballistic drop over `horizon` s stays above `floor`."""
    return floor + jnp.maximum(0.0, 0.5 * G * horizon**2 - vz * horizon)


def throw_sample(key: jax.Array, throw_scale) -> Throw:
    """Randomized throw at scale s = throw_scale: tilt, body rates, position offset, velocity,
    rpm spread and initial estimator error scale with s; yaw is U(-pi, pi) at every s (the actor
    obs is yaw-free). The attitude is q_yaw * q_tilt * q_trim, so the tilt is that of the net
    thrust from vertical and s = 0 starts in the hover trim. The z offset is drawn from the part
    of U(-s, s) that passes the 0.3 s ballistic-drop check given vz (rejection sampling of z
    given vz, in closed form)."""
    s = jnp.asarray(throw_scale, jnp.float64)
    k = jax.random.split(key, 9)
    u = jax.random.uniform
    tilt = u(k[0], (), maxval=1.0) * MAX_TILT * s
    phi = u(k[1], (), maxval=2 * jnp.pi)
    yaw = u(k[2], (), minval=-jnp.pi, maxval=jnp.pi)
    q_tilt = quat_from_rotvec(tilt * jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0]))
    q_yaw = quat_from_rotvec(jnp.array([0.0, 0.0, yaw]))
    vel = s * u(k[3], (3,), minval=-1.0, maxval=1.0)
    z_lo = jnp.maximum(-s, min_start_z(vel[2]) - TARGET[2])
    z_off = z_lo + u(k[4], ()) * (s - z_lo)
    xy = s * u(k[5], (2,), minval=-1.0, maxval=1.0)
    est_angle = u(k[8], (), maxval=1.0) * MAX_EST_ERR * s
    return Throw(
        quat=quat_mul(quat_mul(q_yaw, q_tilt), trim_quat()),
        pos_offset=jnp.concatenate([xy, z_off[None]]),
        vel=vel,
        body_rate=s * u(k[6], (3,), minval=-2.0, maxval=2.0),
        rpm_scale=1.0 + s * u(k[7], (4,), minval=-0.2, maxval=0.2),
        est_err=est_angle * _unit_vector(jax.random.fold_in(k[8], 1)),
    )


def tilt_of(quat: jax.Array) -> jax.Array:
    """Angle between body z and world z (rad)."""
    c = 1.0 - 2.0 * (quat[0] ** 2 + quat[1] ** 2)
    return jnp.arccos(jnp.clip(c, -1.0, 1.0))


def quat_conj(q: jax.Array) -> jax.Array:
    return jnp.concatenate([-q[:3], q[3:]])


def yaw_of(quat: jax.Array) -> jax.Array:
    """Heading of the body x axis projected on the world xy plane (ZYX yaw)."""
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
