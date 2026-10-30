import typing as ty
from dataclasses import dataclass, field

import _drone_path  # noqa: F401  (examples/drone on sys.path)
import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import sensors
from config import Config
from mekf import AttEst
from sensors import Accel, Gyro, Magnetometer

# 6-state (attitude error, gyro bias) MEKF ported from fsw/roci/adcs/src/mekf.rs, the filter
# math the fsw/mekf service is built on. The noise model, the adaptive accel sigma and the accel
# gate below are tuned for this sim (300 Hz on the filtered sim IMU); the fsw/mekf service ships
# a different configuration. Quaternions are [x, y, z, w] Hamilton like nox, q @ v rotates
# body to world, and the attitude error is applied on the body side (q_hat * dq).
#
# Cranelift-safe by construction: matrices are built with stack/concatenate (no scatter),
# branches are jnp.where on float compares, and finiteness is tested with float arithmetic.

ACCEL_REF = np.array([0.0, 0.0, 1.0])  # ENU up: specific force at rest, in g
MAG_REF = np.array([0.0, 1.0, 0.0])  # sensors.mag measures R^T [0, 1, 0]
OMEGA_EPS = 1e-5  # mekf.rs small-rate cutoff
DET_EPS = 1e-12  # 3x3 inverse falls back to identity when det <= DET_EPS (trace / 3)^3
NORM_EPS = 1e-6  # vector measurements shorter than this are skipped
ACCEL_GATE = 0.5  # g; skip accel when | |a| - 1 | exceeds this (spin-up, free fall, hard pulls)
MAG_RATE = 100.0  # Hz, sensors.mag sample-and-hold rate

# Accel white noise alone would give sqrt(0.001 h / dt) = 0.018, but a multirotor accelerometer
# sees thrust and drag, not gravity, whenever it accelerates or turns. The accel sigma is
# therefore adaptive: ACCEL_SIGMA in steady hover, inflated with the specific-force deviation
# and the body rate,
#   sigma_a = ACCEL_SIGMA (1 + ACCEL_SIGMA_GAIN min(| |a| - 1 | / 0.1 g + |w| / 1 rad/s, 10)).
# Tuned on full-DR throw recoveries (best hover tilt p95, then recovery p95) without regressing
# the stock attitude flight; a fixed sigma is either slow to recover (2.0) or follows the
# specific force during recoveries (0.1-0.5).
ACCEL_SIGMA = 0.7
ACCEL_SIGMA_GAIN = 1.0
ACCEL_DEV_SCALE = 0.1  # g
ACCEL_RATE_SCALE = 1.0  # rad/s
ACCEL_INFLATE_MAX = 10.0
# Initial covariance: 0.1 rad attitude, 0.01 rad/s bias (roci starts from identity, which
# lets the first accel samples throw the bias estimate by ~0.1 rad/s).
P0 = np.diag([0.1**2] * 3 + [0.01**2] * 3)


class Sigmas(ty.NamedTuple):
    gyro: float  # angle random walk, rad/sqrt(s)
    gyro_bias: float  # bias random walk, rad/s/sqrt(s)
    accel: float  # unit-vector measurement sigma (steady value, see accel_sigma)
    mag: float  # unit-vector measurement sigma
    accel_gain: float = 0.0  # accel sigma inflation gain, 0 = fixed sigma


def sim_sigmas() -> Sigmas:
    # From the sim's own noise model (sensors.py), fast step h = 1/900 s:
    # gyro white noise var 0.001 per sample -> ARW sqrt(0.001 h) (the biquad keeps unity DC gain);
    # bias drift sqrt(0.001) N h per sample -> random walk sqrt(0.001) sqrt(h);
    # mag var 1e-4 per sample, held at 100 Hz, so each sample is reused (1/100)/dt times.
    h = Config.GLOBAL.fast_loop_time_step
    dt = Config.GLOBAL.dt
    return Sigmas(
        gyro=float(np.sqrt(sensors.gyro_noise.noise_covariance * h)),
        gyro_bias=float(np.sqrt(sensors.gyro_noise.bias_drift_covariance * h)),
        accel=ACCEL_SIGMA,
        mag=float(np.sqrt(sensors.mag_noise.noise_covariance / (MAG_RATE * dt))),
        accel_gain=ACCEL_SIGMA_GAIN,
    )


def accel_sigma(accel: jax.Array, gyro: jax.Array, sigma: float, gain: float) -> jax.Array:
    # float abs/min/sqrt only (no integer clamp or bool logic, Cranelift-safe)
    dev = jnp.abs(jnp.sqrt(jnp.dot(accel, accel)) - 1.0) / ACCEL_DEV_SCALE
    rate = jnp.sqrt(jnp.dot(gyro, gyro)) / ACCEL_RATE_SCALE
    return sigma * (1.0 + gain * jnp.minimum(dev + rate, ACCEL_INFLATE_MAX))


def reset_values(q_hat: jax.Array) -> dict[str, jax.Array]:
    """Estimator component values for an episode start with attitude estimate q_hat."""
    return {
        "attitude_estimate": jnp.asarray(q_hat),
        "mekf6_bias": jnp.zeros(3),
        "mekf6_cov": jnp.asarray(P0),
    }


Mekf6Bias = ty.Annotated[
    jax.Array,
    el.Component(
        "mekf6_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "transient": "true"},
    ),
]
Mekf6Cov = ty.Annotated[
    jax.Array,
    el.Component(
        "mekf6_cov",
        el.ComponentType(el.PrimitiveType.F64, (6, 6)),
        metadata={"transient": "true"},
    ),
]


@dataclass
class MEKF6(el.Archetype):
    # attitude_estimate comes from policy_io.AttitudeEstimate, spawned next to this archetype
    mekf6_bias: Mekf6Bias = field(default_factory=lambda: jnp.zeros(3))
    mekf6_cov: Mekf6Cov = field(default_factory=lambda: jnp.array(P0))


def mm(a: jax.Array, b: jax.Array) -> jax.Array:
    # Small matmul as row-wise multiply + last-axis sum: fuses under vmap (3x faster than
    # dot_general on CPU). Rank-3 broadcast/reduce forms of this miscompile on Cranelift.
    if b.ndim == 1:
        return jnp.sum(a * b, axis=-1)
    b_t = jnp.stack([b[:, j] for j in range(b.shape[1])])
    return jnp.stack([jnp.sum(a[i] * b_t, axis=-1) for i in range(a.shape[0])])


def skew(v: jax.Array) -> jax.Array:
    x, y, z = v[0], v[1], v[2]
    o = jnp.zeros_like(x)
    return jnp.stack([jnp.stack([o, -z, y]), jnp.stack([z, o, -x]), jnp.stack([-y, x, o])])


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return jnp.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(q: jax.Array) -> jax.Array:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_inverse(q: jax.Array) -> jax.Array:
    return quat_conj(q) / jnp.dot(q, q)


def quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    # nox `q * v`: q (v, 0) q^-1, exact for non-unit q
    return quat_mul(quat_mul(q, jnp.concatenate([v, jnp.zeros(1)])), quat_inverse(q))[0:3]


def integrate_body(q: jax.Array, delta: jax.Array) -> jax.Array:
    q = q + quat_mul(q, jnp.concatenate([0.5 * delta, jnp.zeros(1)]))
    return q / jnp.linalg.norm(q)


def calculate_covariance(sigma_g: jax.Array, sigma_b: jax.Array, dt: float) -> jax.Array:
    eye = np.identity(3)
    variance_g = eye * (sigma_g * sigma_g * dt)
    variance_b = eye * (sigma_b * sigma_b * dt)
    q_00 = variance_g + variance_b * dt * (dt / 3.0)
    q_01 = variance_b * (dt / 2.0)
    q_0 = jnp.concatenate([q_00, q_01], axis=1)
    q_1 = jnp.concatenate([q_01, variance_b], axis=1)
    return jnp.concatenate([q_0, q_1], axis=0)


def process_noise(sigma_g: jax.Array, sigma_b: jax.Array, dt: float) -> jax.Array:
    # State::new: Y Q Y^T with Y = diag(-1, -1, -1, 1, 1, 1)
    y = np.diag([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    return mm(mm(y, calculate_covariance(sigma_g, sigma_b, dt)), y.T)


def propagate_quaternion(q: jax.Array, omega: jax.Array, dt: float) -> jax.Array:
    omega_norm_squared = jnp.dot(omega, omega)
    moving = jnp.sqrt(omega_norm_squared) > OMEGA_EPS
    omega_norm = jnp.sqrt(jnp.where(moving, omega_norm_squared, 1.0))
    c = jnp.cos(0.5 * omega_norm * dt)
    s = jnp.sin(0.5 * omega_norm * dt) / omega_norm
    x, y, z = s * omega[0], s * omega[1], s * omega[2]
    big_omega = jnp.stack(
        [
            jnp.stack([c, z, -y, x]),
            jnp.stack([-z, c, x, y]),
            jnp.stack([y, -x, c, z]),
            jnp.stack([-x, -y, -z, c]),
        ]
    )
    return jnp.where(moving, mm(big_omega, q), q)


def propagate_state_covariance(
    big_p: jax.Array, omega: jax.Array, yqy: jax.Array, dt: float
) -> jax.Array:
    omega_norm_squared = jnp.dot(omega, omega)
    moving = jnp.sqrt(omega_norm_squared) > OMEGA_EPS
    omega_norm_squared = jnp.where(moving, omega_norm_squared, 1.0)
    omega_norm = jnp.sqrt(omega_norm_squared)
    s = jnp.sin(omega_norm * dt)
    c = jnp.cos(omega_norm * dt)
    p = s / omega_norm
    q = (1.0 - c) / omega_norm_squared
    r = (omega_norm * dt - s) / (omega_norm_squared * omega_norm)
    omega_cross = skew(omega)
    omega_cross_square = mm(omega_cross, omega_cross)
    eye = np.identity(3)
    phi_00 = jnp.where(moving, eye - omega_cross * p + omega_cross_square * q, eye)
    phi_01 = jnp.where(moving, omega_cross * q - eye * dt - omega_cross_square * r, eye * -dt)
    phi_1 = np.concatenate([np.zeros((3, 3)), eye], axis=1)
    phi = jnp.concatenate([jnp.concatenate([phi_00, phi_01], axis=1), phi_1], axis=0)
    return mm(mm(phi, big_p), phi.T) + yqy


def inverse3(m: jax.Array) -> jax.Array:
    # nox Matrix::try_inverse closed form for 3x3. nox never errs here (it divides by det),
    # so mekf.rs's unwrap_or(eye) is unreachable. The innovation covariance is SPD, so this
    # falls back to eye only on a non-positive, non-finite or relatively tiny det.
    m11, m12, m13 = m[0, 0], m[0, 1], m[0, 2]
    m21, m22, m23 = m[1, 0], m[1, 1], m[1, 2]
    m31, m32, m33 = m[2, 0], m[2, 1], m[2, 2]
    minor_m12_m23 = m22 * m33 - m32 * m23
    minor_m11_m23 = m21 * m33 - m31 * m23
    minor_m11_m22 = m21 * m32 - m31 * m22
    det = m11 * minor_m12_m23 - m12 * minor_m11_m23 + m13 * minor_m11_m22
    scale = jnp.abs(m11 + m22 + m33) / 3.0
    ok = det > DET_EPS * scale * scale * scale
    det = jnp.where(ok, det, 1.0)
    adj = jnp.stack(
        [
            jnp.stack([minor_m12_m23, m13 * m32 - m33 * m12, m12 * m23 - m22 * m13]),
            jnp.stack([-minor_m11_m23, m11 * m33 - m31 * m13, m13 * m21 - m23 * m11]),
            jnp.stack([minor_m11_m22, m12 * m31 - m32 * m11, m11 * m22 - m21 * m12]),
        ]
    )
    return jnp.where(ok, adj / det, np.identity(3))


def safe_normalize(v: jax.Array, gate: float | None = None) -> tuple[jax.Array, jax.Array]:
    # valid if |v| > NORM_EPS (and | |v| - 1 | < gate); masks are combined as floats
    norm_squared = jnp.dot(v, v)
    norm = jnp.sqrt(norm_squared)
    ok = jnp.where(norm > NORM_EPS, 1.0, 0.0)
    if gate is not None:
        ok = ok * jnp.where(jnp.abs(norm - 1.0) < gate, 1.0, 0.0)
    valid = ok > 0.5
    return v / jnp.sqrt(jnp.where(valid, norm_squared, 1.0)), valid


def vector_update(
    q_hat: jax.Array,
    p: jax.Array,
    delta_x_hat: jax.Array,
    reference: jax.Array,
    measured_body: jax.Array,
    sigma: float,
) -> tuple[jax.Array, jax.Array]:
    # one pass of the sequential loop in State::estimate_attitude
    var_r = np.identity(3) * sigma**2
    body_r = quat_rotate(quat_inverse(q_hat), reference)
    e = measured_body - body_r
    h = jnp.concatenate([skew(body_r), np.zeros((3, 3))], axis=1)
    h_trans = h.T
    s = inverse3(mm(mm(h, p), h_trans) + var_r)
    k = mm(p, mm(h_trans, s))
    p = mm(np.identity(6) - mm(k, h), p)
    d = mm(h, delta_x_hat)
    return p, delta_x_hat + mm(k, e - d)


def is_finite(*xs: jax.Array) -> jax.Array:
    # inf - inf and nan - nan are nan; avoids bool reductions (Cranelift-safe)
    return sum(jnp.sum(x - x) for x in xs) == 0.0


def reset_if_invalid(
    q_hat: jax.Array,
    b_hat: jax.Array,
    p: jax.Array,
    omega: jax.Array,
    p_reset: np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # roci resets P to identity; estimate() passes P0
    ok = is_finite(q_hat, b_hat, p, omega)
    return (
        jnp.where(ok, q_hat, np.array([0.0, 0.0, 0.0, 1.0])),
        jnp.where(ok, b_hat, np.zeros(3)),
        jnp.where(ok, p, np.identity(6) if p_reset is None else p_reset),
        jnp.where(ok, omega, np.zeros(3)),
    )


def estimate_attitude(
    q_hat: jax.Array,
    b_hat: jax.Array,
    p: jax.Array,
    omega: jax.Array,
    yqy: jax.Array,
    dt: float,
    measured_bodys: ty.Sequence[jax.Array],
    references: ty.Sequence[jax.Array],
    sigma_r: ty.Sequence[float],
    valid: ty.Sequence[jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # State::estimate_attitude; returns omega - b_hat (prior) like the stored State.omega.
    # A measurement whose valid flag is false leaves p and delta_x_hat untouched.
    omega = omega - b_hat
    q_hat = propagate_quaternion(q_hat, omega, dt)
    p = propagate_state_covariance(p, omega, yqy, dt)
    delta_x_hat = jnp.zeros(6)
    valid = valid if valid is not None else [None] * len(references)
    for reference, measured_body, sigma, ok in zip(references, measured_bodys, sigma_r, valid):
        p_new, dx_new = vector_update(q_hat, p, delta_x_hat, reference, measured_body, sigma)
        p = p_new if ok is None else jnp.where(ok, p_new, p)
        delta_x_hat = dx_new if ok is None else jnp.where(ok, dx_new, delta_x_hat)
    q_hat = integrate_body(q_hat, delta_x_hat[0:3])
    b_hat = b_hat + delta_x_hat[3:6]
    return q_hat, b_hat, p, omega


def estimate(
    q: jax.Array,
    b: jax.Array,
    p: jax.Array,
    gyro: jax.Array,
    accel: jax.Array,
    mag: jax.Array,
    sigmas: Sigmas,
    dt: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One MEKF step. q [x,y,z,w] body->world, b gyro bias (rad/s), p 6x6 covariance,
    gyro (rad/s), accel (g, skipped outside 1 +- ACCEL_GATE), mag (any scale, skipped if ~0).
    Returns (q, b, p, gyro_est) with gyro_est = gyro - b (posterior bias). A non-finite
    state resets to identity attitude, zero bias and P0."""
    ones = np.ones(3)
    yqy = process_noise(ones * sigmas.gyro, ones * sigmas.gyro_bias, dt)
    sigma_a = sigmas.accel
    if sigmas.accel_gain:
        sigma_a = accel_sigma(accel, gyro, sigmas.accel, sigmas.accel_gain)
    accel, accel_ok = safe_normalize(accel, ACCEL_GATE)
    mag, mag_ok = safe_normalize(mag)
    q, b, p, _ = estimate_attitude(
        q,
        b,
        p,
        gyro,
        yqy,
        dt,
        (accel, mag),
        (ACCEL_REF, MAG_REF),
        (sigma_a, sigmas.mag),
        (accel_ok, mag_ok),
    )
    return reset_if_invalid(q, b, p, gyro - b, P0)


# 300 Hz, at the head of the chain: mekf6_update | sim.system()
@el.map
def mekf6_update(
    gyro: Gyro,
    accel: Accel,
    mag: Magnetometer,
    q: AttEst,
    b: Mekf6Bias,
    p: Mekf6Cov,
) -> tuple[AttEst, Mekf6Bias, Mekf6Cov]:
    q, b, p, _ = estimate(q.vector(), b, p, gyro, accel, mag, sim_sigmas(), Config.GLOBAL.dt)
    return el.Quaternion(q), b, p
