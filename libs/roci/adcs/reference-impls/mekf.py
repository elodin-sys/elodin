#!/usr/bin/env python3
#

import elodin as el
import jax
import jax.numpy as np
from elodin.elodin import Quaternion
from jax.numpy import linalg as la


def calculate_covariance(sigma_g: jax.Array, sigma_b: jax.Array, dt: float) -> jax.Array:
    variance_g = np.diag(sigma_g * sigma_g * dt)
    variance_b = np.diag(sigma_b * sigma_b * dt)
    Q_00 = variance_g + variance_b * dt**2 / 3
    Q_01 = variance_b * dt / 2
    Q_10 = Q_01
    Q_11 = variance_b
    return np.block([[Q_00, Q_01], [Q_10, Q_11]])


Q = calculate_covariance(np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), 1 / 120.0)
print(f"Q = {repr(Q)}")
Y = np.diag(np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))

YQY = Y @ Q @ Y.T

print(repr(YQY))


def propogate_quaternion(q: Quaternion, omega: jax.Array, dt: float) -> Quaternion:
    omega_norm = la.norm(omega)
    c = np.cos(0.5 * omega_norm * dt)
    s = np.sin(0.5 * omega_norm * dt) / omega_norm
    omega_s = s * omega
    [x, y, z] = omega_s
    big_omega = np.array([[c, z, -y, x], [-z, c, x, y], [y, -x, c, z], [-x, -y, -z, c]])
    o = big_omega @ q.vector()
    return Quaternion(jax.lax.select(omega_norm > 1e-5, o, q.vector()))


# returns q_hat
def update_quaternion(q: Quaternion, delta_alpha: jax.Array) -> Quaternion:
    a = 0.5 * delta_alpha
    qa = Quaternion(np.array([a[0], a[1], a[2], 0.0]))
    q_hat = q + q * qa
    return q_hat.normalize()


def propogate_state_covariance(big_p: jax.Array, omega: jax.Array, dt: float) -> jax.Array:
    omega_norm = la.norm(omega)
    s = np.sin(omega_norm * dt)
    c = np.cos(omega_norm * dt)
    p = s / omega_norm
    q = (1 - c) / (omega_norm**2)
    r = (omega_norm * dt - s) / (omega_norm**3)
    omega_cross = el.skew(omega)
    omega_cross_square = omega_cross @ omega_cross
    phi_00 = jax.lax.select(
        omega_norm > 1e-5,
        np.eye(3) - omega_cross * p + omega_cross_square * q,
        np.eye(3),
    )
    phi_01 = jax.lax.select(
        omega_norm > 1e-5,
        omega_cross * q - np.eye(3) * dt - omega_cross_square * r,
        np.eye(3) * -1.0 / 120.0,
    )
    phi_10 = np.zeros((3, 3))
    phi_11 = np.eye(3)
    phi = np.block([[phi_00, phi_01], [phi_10, phi_11]])
    return (phi @ big_p @ phi.T) + YQY


def estimate_attitude(
    q_hat: Quaternion,
    b_hat: jax.Array,
    omega: jax.Array,
    p: jax.Array,
    measured_bodys: jax.Array,
    measured_references: jax.Array,
    dt: float,
) -> tuple[Quaternion, jax.Array, jax.Array, jax.Array]:
    omega = omega - b_hat
    q_hat = propogate_quaternion(q_hat, omega, dt)
    p = propogate_state_covariance(p, omega, dt)
    delta_x_hat = np.zeros(6)
    var_r = np.eye(3) * 0.001
    for i in range(0, 2):
        measured_reference = measured_references[i]
        measured_body = measured_bodys[i]
        body_r = q_hat.inverse() @ measured_reference
        print(f"body_r = {repr(body_r)}")
        e = measured_body - body_r
        skew_sym = el.skew(body_r)
        print(f"skew_sym = {repr(skew_sym)}")
        H = np.block([skew_sym, np.zeros((3, 3))])
        print(f"H = {repr(H)}")
        H_trans = H.T
        print(f"H_trans = {repr(H_trans)}")
        hphtrans = H @ p @ H_trans
        print(f"hphtrans = {repr(hphtrans)}")
        gain = la.inv(hphtrans + var_r)
        print(f"gain = {repr(gain)}")
        K = p @ H_trans @ gain
        print(f"k = {repr(K)}")
        p = (np.eye(6) - K @ H) @ p
        delta_x_hat = delta_x_hat + K @ (e - H @ delta_x_hat)
    print("dxhat = ", delta_x_hat)
    delta_alpha = delta_x_hat[0:3]
    delta_beta = delta_x_hat[3:6]
    q_hat = update_quaternion(q_hat, delta_alpha)
    b_hat = b_hat + delta_beta
    return (q_hat, b_hat, p, omega)


ref_a = np.array([0.0, 1.0, 0.0])
ref_b = np.array([1.0, 0.0, 0.0])
q = Quaternion.from_axis_angle(np.array([0.0, 0.0, -1.0]), 3.14 / 4.0)
body_a = q.inverse() @ ref_a
body_b = q.inverse() @ ref_b
q_hat = Quaternion.identity()
b_hat = np.array([0.0, 0.0, 0.0])
p = np.eye(6)

for i in range(0, 180):
    omega = np.array([0.0, 0.0, 0.0])
    new_q_hat, new_b_hat, new_p, _omega = estimate_attitude(
        q_hat,
        b_hat,
        omega,
        p,
        np.array([body_a, body_b]),
        np.array([ref_a, ref_b]),
        1 / 120.0,
    )
    q_hat = new_q_hat
    b_hat = new_b_hat
    p = new_p
print(q_hat.vector())
print(q.vector())
