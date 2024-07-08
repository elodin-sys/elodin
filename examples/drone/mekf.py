from numpy.typing import NDArray
import elodin as el
from dataclasses import field, dataclass
import typing as ty
import jax
import jax.numpy as jnp
import numpy as np

import params
import util
from sensors import Gyro, Accel, Magnetometer, AccelHealth

# TODO: incorporate GNSS, barometer, etc.

estimate_covariance = 0.1
gyro_cov = 0.1
gyro_bias_cov = 0.01
accel_proc_cov = 0.1
accel_bias_cov = 0.001
mag_proc_cov = 0.1
mag_bias_cov = 0.001
mag_obs_cov = 0.1

EstCov = ty.Annotated[
    jax.Array,
    el.Component(
        "estimate_covariance",
        el.ComponentType(el.PrimitiveType.F64, (18, 18)),
        # metadata={"priority": 400},
    ),
]
AttEst = ty.Annotated[
    el.Quaternion,
    el.Component(
        "attitude_estimate",
        metadata={"priority": 399, "element_names": "q0,q1,q2,q3"},
    ),
]
GyroBiasEst = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_bias_estimate",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 397, "element_names": "x,y,z"},
    ),
]
AccelBiasEst = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_bias_estimate",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 396, "element_names": "x,y,z"},
    ),
]
MagBiasEst = ty.Annotated[
    jax.Array,
    el.Component(
        "magnetometer_bias_estimate",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": 395, "element_names": "x,y,z"},
    ),
]
AttEstError = ty.Annotated[
    jax.Array,
    el.Component(
        "attitude_estimate_error",
        el.ComponentType.F64,
    ),
]


def observation_covariance(accel_obs_cov: jax.Array, mag_obs_cov: float) -> jax.Array:
    return (
        jnp.identity(6)
        .at[0:3, 0:3]
        .set(accel_obs_cov * jnp.identity(3))
        .at[3:6, 3:6]
        .set(mag_obs_cov * jnp.identity(3))
    )


def process_covariance(dt: float) -> NDArray[np.float64]:
    gyro_cov_mat = gyro_cov * np.identity(3, dtype=float)
    gyro_bias_cov_mat = gyro_bias_cov * np.identity(3, dtype=float)
    accel_cov_mat = accel_proc_cov * np.identity(3, dtype=float)
    accel_bias_cov_mat = accel_bias_cov * np.identity(3, dtype=float)
    # mag_cov_mat = mag_proc_cov * np.identity(3, dtype=float)
    mag_bias_cov_mat = mag_bias_cov * np.identity(3, dtype=float)

    Q = np.zeros(shape=(18, 18), dtype=float)
    Q[0:3, 0:3] = gyro_cov_mat * dt + gyro_bias_cov_mat * (dt**3) / 3.0
    Q[0:3, 9:12] = -gyro_bias_cov_mat * (dt**2) / 2.0
    Q[3:6, 3:6] = accel_cov_mat * dt + accel_bias_cov_mat * (dt**3) / 3.0
    Q[3:6, 6:9] = accel_bias_cov_mat * (dt**4) / 8.0 + accel_cov_mat * (dt**2) / 2.0
    Q[3:6, 12:15] = -accel_bias_cov_mat * (dt**2) / 2.0
    Q[6:9, 3:6] = accel_cov_mat * (dt**2) / 2.0 + accel_bias_cov_mat * (dt**4) / 8.0
    Q[6:9, 6:9] = accel_cov_mat * (dt**3) / 3.0 + accel_bias_cov_mat * (dt**5) / 20.0
    Q[6:9, 12:15] = -accel_bias_cov_mat * (dt**3) / 6.0
    Q[9:12, 0:3] = -gyro_bias_cov_mat * (dt**2) / 2.0
    Q[9:12, 9:12] = gyro_bias_cov_mat * dt
    Q[12:15, 3:6] = -accel_bias_cov_mat * (dt**2) / 2.0
    Q[12:15, 6:9] = -accel_bias_cov_mat * (dt**3) / 6.0
    Q[12:15, 12:15] = accel_bias_cov_mat * dt
    Q[15:18, 15:18] = mag_bias_cov_mat * dt
    return Q


@dataclass
class MEKF(el.Archetype):
    est_cov: EstCov = field(
        default_factory=lambda: jnp.identity(18) * estimate_covariance
    )
    att_est: AttEst = field(default_factory=lambda: el.Quaternion.identity())
    gyro_bias_est: GyroBiasEst = field(default_factory=lambda: jnp.zeros(3))
    accel_bias_est: AccelBiasEst = field(default_factory=lambda: jnp.zeros(3))
    mag_bias_est: MagBiasEst = field(default_factory=lambda: jnp.zeros(3))
    att_est_error: AttEstError = field(default_factory=lambda: jnp.float64(0.0))


def integrate_angular_velocity(
    q: el.Quaternion, w: jax.Array, dt: float
) -> el.Quaternion:
    return (q + q * el.Quaternion(jnp.array([*(0.5 * w * dt), 0.0]))).normalize()


@el.map
def update_filter(
    gyro: Gyro,
    accel: Accel,
    mag: Magnetometer,
    est_cov: EstCov,
    q: AttEst,
    gyro_bias_est: GyroBiasEst,
    accel_bias_est: AccelBiasEst,
    mag_bias_est: MagBiasEst,
    accel_health: AccelHealth,
) -> tuple[EstCov, AttEst, GyroBiasEst, AccelBiasEst, MagBiasEst]:
    dt = 1.0 / params.SCHED_LOOP_RATE
    # Normalize accel:
    accel = jax.lax.cond(
        jnp.linalg.norm(accel) > 1e-6,
        lambda _: accel / jnp.linalg.norm(accel),
        lambda _: jnp.array([0.0, 0.0, 1.0]),
        operand=None,
    )
    gyro = gyro - gyro_bias_est
    accel = accel - accel_bias_est
    mag = mag - mag_bias_est

    # Integrate angular velocity into quaternion
    q = integrate_angular_velocity(q, gyro, dt)

    # Process model
    G = (
        jnp.zeros(shape=(18, 18))
        .at[0:3, 9:12]
        .set(-jnp.identity(3))
        .at[6:9, 3:6]
        .set(jnp.identity(3))
        .at[0:3, 0:3]
        .set(-el.skew(gyro))
        .at[3:6, 0:3]
        .set(-util.quat_to_matrix(q).dot(el.skew(accel)))
        .at[3:6, 12:15]
        .set(-util.quat_to_matrix(q))
    )
    F = jnp.identity(18) + G * dt

    # A priori covariance
    est_cov = F.dot(est_cov).dot(F.T) + process_covariance(dt)

    # Kalman gain
    H = (
        jnp.zeros(shape=(6, 18))
        .at[0:3, 0:3]
        .set(el.skew(q.inverse() @ jnp.array([0.0, 0.0, 1.0])))
        .at[0:3, 12:15]
        .set(jnp.identity(3))
        .at[3:6, 0:3]
        .set(el.skew(q.inverse() @ jnp.array([0.0, 1.0, 0.0])))
        .at[3:6, 15:18]
        .set(jnp.identity(3))
    )
    PH_T = est_cov.dot(H.T)
    accel_cov = 1.0 + (1 - accel_health) * 1000.0
    inn_cov = H.dot(PH_T) + observation_covariance(accel_cov, mag_obs_cov)
    K = PH_T.dot(jnp.linalg.inv(inn_cov))

    # Update with a posteriori estimate
    est_cov = (jnp.identity(18) - K.dot(H)).dot(est_cov)

    # Observation model
    observation = jnp.concat([accel, mag])
    pred_observation = jnp.concat(
        [
            q.inverse() @ jnp.array([0.0, 0.0, 1.0]),
            q.inverse() @ jnp.array([0.0, 1.0, 0.0]),
        ]
    )

    aposteriori_state = K.dot((observation - pred_observation).transpose())

    q = q * el.Quaternion(jnp.array([*(0.5 * aposteriori_state[:3]), 1.0])).normalize()
    gyro_bias_est += aposteriori_state[9:12]
    accel_bias_est += aposteriori_state[12:15]
    mag_bias_est += aposteriori_state[15:18]

    return (
        est_cov,
        q,
        gyro_bias_est,
        accel_bias_est,
        mag_bias_est,
    )


@el.map
def att_est_error(att_est: AttEst, pos: el.WorldPos) -> AttEstError:
    return util.quat_dist(att_est, pos.angular())
