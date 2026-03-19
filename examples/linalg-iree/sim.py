"""Linalg-IREE example: validates IREE-safe linalg shims compile and run.

Uses a 3-state Kalman filter with cholesky, inv, det, slogdet, and QR
decomposition to prove these operations work on the IREE backend
without any user-side workarounds.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax.numpy as jnp

SIMULATION_RATE = 120.0

# --- Components ---

State = ty.Annotated[
    jnp.ndarray,
    el.Component("kf_state", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
Covariance = ty.Annotated[
    jnp.ndarray,
    el.Component("kf_cov", el.ComponentType(el.PrimitiveType.F64, (3, 3))),
]
Info = ty.Annotated[
    jnp.ndarray,
    el.Component("kf_info", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@el.dataclass
class KFArchetype(el.Archetype):
    kf_state: State = field(default_factory=lambda: jnp.zeros(3))
    kf_cov: Covariance = field(default_factory=lambda: jnp.eye(3))
    kf_info: Info = field(default_factory=lambda: jnp.zeros(3))


DT = 1.0 / SIMULATION_RATE

F = jnp.array(
    [
        [1.0, DT, 0.0],
        [0.0, 1.0, DT],
        [0.0, 0.0, 1.0],
    ]
)
Q = 0.01 * jnp.eye(3)
H = jnp.eye(3)
R = 0.1 * jnp.eye(3)


@el.map
def kf_step(state: State, cov: Covariance, info: Info) -> tuple[State, Covariance, Info]:
    x_pred = F @ state
    P_pred = F @ cov @ F.T + Q

    z = x_pred + 0.01 * jnp.ones(3)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R

    # Exercise: cholesky
    L_S = jnp.linalg.cholesky(S)
    _chol_check = L_S @ L_S.T  # noqa: F841

    # Exercise: inv  (Kalman gain via K = P H^T S^{-1})
    S_inv = jnp.linalg.inv(S)
    K = P_pred @ H.T @ S_inv

    # State update
    x_upd = x_pred + K @ y
    I3 = jnp.eye(3)
    IKH = I3 - K @ H
    P_upd = IKH @ P_pred @ IKH.T + K @ R @ K.T

    # Exercise: QR factorisation on P_upd
    Q_f, R_f = jnp.linalg.qr(P_upd)
    P_upd = Q_f @ R_f

    # Exercise: det / slogdet for log-likelihood
    d = jnp.linalg.det(S)
    sign, logdet = jnp.linalg.slogdet(S)
    log_lik = -0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + logdet + y @ S_inv @ y)

    info_out = jnp.array([log_lik, d, sign])
    return x_upd, P_upd, info_out


def world() -> el.World:
    w = el.World()
    w.spawn(
        KFArchetype(
            kf_state=jnp.array([0.0, 1.0, 0.0]),
            kf_cov=jnp.eye(3) * 10.0,
        ),
        name="tracker",
    )
    return w


def system() -> el.System:
    return kf_step
