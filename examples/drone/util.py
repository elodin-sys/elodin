import elodin as el
import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike
from numpy._typing import NDArray


def motor_positions(angles: NDArray[np.float64], distance: float) -> NDArray[np.float64]:
    # for each angle, calculate the x and y position of the motor
    x = jnp.cos(angles)
    y = jnp.sin(angles)
    z = jnp.zeros_like(angles)
    return np.stack([x, y, z], axis=-1) * distance


# Closeness of two quaternions in terms of the angle of rotation
# required to go from one orientation to the other.
#
# Distance function is "Inner Product of Unit Quaternions"
# from "Metrics for 3D Rotations: Comparison and Analysis"" by Du Q. Huynh
# Also: https://math.stackexchange.com/a/90098
def quat_dist(q1: el.Quaternion, q2: el.Quaternion) -> jax.Array:
    return 2 * jnp.arccos(jnp.abs(jnp.dot(q1.vector(), q2.vector())))


def quat_to_matrix(q: el.Quaternion) -> jax.Array:
    q0, q1, q2, s = q.vector()
    v = jnp.array([q0, q1, q2])
    return 2.0 * jnp.outer(v, v) + jnp.identity(3) * (s**2 - jnp.dot(v, v)) + 2.0 * s * el.skew(v)


# Convert a quaternion to Euler angles in 3-2-1 sequence (roll, pitch, yaw).
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
def quat_to_euler(q: el.Quaternion) -> jax.Array:
    q0, q1, q2, s = q.vector()
    # roll
    sinr_cosp = 2.0 * (s * q0 + q1 * q2)
    cosr_cosp = 1.0 - 2.0 * (q0**2 + q1**2)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = jnp.sqrt(1.0 + 2.0 * (s * q1 - q0 * q2))
    cosp = jnp.sqrt(1.0 - 2.0 * (s * q1 - q0 * q2))
    pitch = 2 * jnp.arctan2(sinp, cosp) - jnp.pi / 2
    # yaw
    siny_cosp = 2.0 * (s * q2 + q0 * q1)
    cosy_cosp = 1.0 - 2.0 * (q1**2 + q2**2)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    return jnp.array([roll, pitch, yaw])


# Convert a quaternion to axis-angle representation.
# The resulting vector is the axis of rotation and its magnitude is the angle of rotation.
def quat_to_axis_angle(q: el.Quaternion) -> jax.Array:
    q0, q1, q2, s = q.vector()
    len = jnp.sqrt(q0**2 + q1**2 + q2**2)
    axis = jnp.array([q0, q1, q2])
    axis = jax.lax.cond(
        len < 1e-6,
        lambda _: axis,
        lambda _: (axis / len) * normalize_angle(2.0 * jnp.atan2(len, s)),
        operand=None,
    )
    return axis


def quat_from_axis_angle(v: jax.Array) -> el.Quaternion:
    theta = jnp.linalg.norm(v)
    return jax.lax.cond(
        theta < 1e-6,
        lambda _: el.Quaternion.identity(),
        lambda _: el.Quaternion.from_axis_angle(v / theta, theta),
        operand=None,
    )


# Convert Euler angles in 3-2-1 sequence (roll, pitch, yaw) to a quaternion.
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion
def euler_to_quat(euler: ArrayLike) -> el.Quaternion:
    roll, pitch, yaw = jnp.array(euler)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return el.Quaternion(jnp.array([x, y, z, w]))


# convert angular rate to euler rate
# resulting euler rate will diverge as the pitch angle approaches +/- 90 degrees (gimbal lock)
# call site should handle the NaNs and infinities
def angular_to_euler_rate(att: el.Quaternion, ang_rate: jax.Array) -> jax.Array:
    phi, theta, _ = quat_to_euler(att)
    return jnp.dot(
        jnp.array(
            [
                [1.0, jnp.sin(phi) * jnp.tan(theta), jnp.cos(phi) * jnp.tan(theta)],
                [0.0, jnp.cos(phi), -jnp.sin(phi)],
                [0.0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)],
            ]
        ),
        ang_rate,
    )


# convert euler rate to angular rate
def euler_to_angular_rate(att: el.Quaternion, euler_rate: jax.Array) -> jax.Array:
    phi, theta, _ = quat_to_euler(att)
    return jnp.dot(
        jnp.array(
            [
                [1.0, 0.0, -jnp.sin(theta)],
                [0.0, jnp.cos(phi), jnp.sin(phi) * jnp.cos(theta)],
                [0.0, -jnp.sin(phi), jnp.cos(phi) * jnp.cos(theta)],
            ]
        ),
        euler_rate,
    )


# normalize angle to be within [-pi, pi]
def normalize_angle(angle: jax.typing.ArrayLike) -> jax.Array:
    angle = jnp.mod(angle, 2.0 * jnp.pi)
    angle = jnp.where(angle < 0.0, angle + 2.0 * jnp.pi, angle)
    return jnp.where(angle > jnp.pi, angle - 2.0 * jnp.pi, angle)
