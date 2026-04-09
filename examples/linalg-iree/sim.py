"""Linalg-IREE example: validates all LAPACK-backed linalg operations compile
and run correctly on the IREE backend.

Exercises every JAX linalg operation used in aerospace simulations:
  - solve, inv (2x2 small-matrix LAPACK dispatch)
  - cholesky, det, slogdet, qr (3x3 Kalman filter)
  - svd, pseudoinverse via SVD (6x6 navigation EKF)
  - solve (direct linear system)
  - eigh (symmetric eigendecomposition)
  - norm (vector/matrix norms)
  - .at[idx].set() scatter-into-constant (mode selector)
  - U64-typed components with uint64 transition tables and indexed access

Uses both @el.map and @el.map_seq to exercise both compilation paths.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la

SIMULATION_RATE = 120.0

# --- Components (3-state KF) ---

State3 = ty.Annotated[
    jnp.ndarray,
    el.Component("kf3_state", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
Cov3 = ty.Annotated[
    jnp.ndarray,
    el.Component("kf3_cov", el.ComponentType(el.PrimitiveType.F64, (3, 3))),
]
Info3 = ty.Annotated[
    jnp.ndarray,
    el.Component("kf3_info", el.ComponentType(el.PrimitiveType.F64, (5,))),
]

# --- Components (6-state EKF) ---

State6 = ty.Annotated[
    jnp.ndarray,
    el.Component("ekf6_state", el.ComponentType(el.PrimitiveType.F64, (6,))),
]
Cov6 = ty.Annotated[
    jnp.ndarray,
    el.Component("ekf6_cov", el.ComponentType(el.PrimitiveType.F64, (6, 6))),
]
Info6 = ty.Annotated[
    jnp.ndarray,
    el.Component("ekf6_info", el.ComponentType(el.PrimitiveType.F64, (4,))),
]

# --- Components (mode selector -- exercises .at[idx].set() scatter) ---

ModeState = ty.Annotated[
    jnp.ndarray,
    el.Component("mode_state", el.ComponentType(el.PrimitiveType.I64, (4,))),
]

# --- Components (2-state small system -- exercises small-matrix LAPACK dispatch) ---

State2 = ty.Annotated[
    jnp.ndarray,
    el.Component("sm2_state", el.ComponentType(el.PrimitiveType.F64, (2,))),
]
Cov2 = ty.Annotated[
    jnp.ndarray,
    el.Component("sm2_cov", el.ComponentType(el.PrimitiveType.F64, (2, 2))),
]

# --- Components (matrix-RHS solve -- exercises transposed B in dgetrs) ---

MatRhsState = ty.Annotated[
    jnp.ndarray,
    el.Component("mrhs_state", el.ComponentType(el.PrimitiveType.F64, (3, 2))),
]

# --- Components (uint64 ops -- exercises U64 components with scatter/index) ---
# Mirrors the customer's OpTransition / OpState pattern from sim_FT19.py where
# a uint64 transition table is indexed by a state value and combined with
# scatter-into-constant in a fused system pipeline.

OpTransition = ty.Annotated[
    jnp.ndarray,
    el.Component("op_transition", el.ComponentType(el.PrimitiveType.U64, (4,))),
]
OpState = ty.Annotated[
    jnp.ndarray,
    el.Component("op_state", el.ComponentType(el.PrimitiveType.U64, (4,))),
]

# --- Archetypes ---


@el.dataclass
class KF3Archetype(el.Archetype):
    kf3_state: State3 = field(default_factory=lambda: jnp.zeros(3))
    kf3_cov: Cov3 = field(default_factory=lambda: jnp.eye(3))
    kf3_info: Info3 = field(default_factory=lambda: jnp.zeros(5))


@el.dataclass
class EKF6Archetype(el.Archetype):
    ekf6_state: State6 = field(default_factory=lambda: jnp.zeros(6))
    ekf6_cov: Cov6 = field(default_factory=lambda: jnp.eye(6))
    ekf6_info: Info6 = field(default_factory=lambda: jnp.zeros(4))


@el.dataclass
class Small2Archetype(el.Archetype):
    sm2_state: State2 = field(default_factory=lambda: jnp.zeros(2))
    sm2_cov: Cov2 = field(default_factory=lambda: jnp.eye(2))


@el.dataclass
class MatRhsArchetype(el.Archetype):
    mrhs_state: MatRhsState = field(default_factory=lambda: jnp.zeros((3, 2)))


@el.dataclass
class ModeArchetype(el.Archetype):
    mode_state: ModeState = field(default_factory=lambda: jnp.zeros(4, dtype=jnp.int64))


@el.dataclass
class UintOpsArchetype(el.Archetype):
    op_transition: OpTransition = field(
        default_factory=lambda: jnp.array([1, 2, 3, 0], dtype=jnp.uint64)
    )
    op_state: OpState = field(
        default_factory=lambda: jnp.array([0, 0, 0, 0], dtype=jnp.uint64)
    )


DT = 1.0 / SIMULATION_RATE

# 3-state dynamics (position, velocity, acceleration)
F3 = jnp.array([[1.0, DT, 0.0], [0.0, 1.0, DT], [0.0, 0.0, 1.0]])
Q3 = 0.01 * jnp.eye(3)
H3 = jnp.eye(3)
R3 = 0.1 * jnp.eye(3)

# 2-state dynamics (exercises 2x2 solve/inv via LAPACK dispatch)
F2 = jnp.array([[1.0, DT], [0.0, 1.0]])
Q2 = 0.01 * jnp.eye(2)
H2 = jnp.eye(2)
R2 = 0.1 * jnp.eye(2)

# 6-state dynamics (3 position + 3 velocity)
F6 = jnp.block([[jnp.eye(3), DT * jnp.eye(3)], [jnp.zeros((3, 3)), jnp.eye(3)]])
Q6 = 0.01 * jnp.eye(6)
H6 = jnp.eye(6)
R6 = 0.1 * jnp.eye(6)


def _safe_matrix_inverse(matrix, tolerance=1e-12):
    """SVD-based pseudoinverse matching the customer's util.safe_matrix_inverse."""
    u, s, vh = la.svd(matrix)
    s_inv = jnp.where(s > tolerance, 1.0 / s, 0.0)
    return jnp.transpose(vh) @ jnp.diag(s_inv) @ jnp.transpose(u)


# --- Matrix-RHS solve (exercises solve(A[3,3], B[3,2]) transposed dgetrs path) ---


@el.map
def mat_rhs_step(state: MatRhsState) -> MatRhsState:
    A = F3 + 0.01 * jnp.eye(3)
    return jnp.linalg.solve(A, state)


# --- 2-state filter (exercises 2x2 solve + inv -- the customer's failing case) ---


@el.map
def small2_step(state: State2, cov: Cov2) -> tuple[State2, Cov2]:
    x_pred = F2 @ state
    P_pred = F2 @ cov @ F2.T + Q2

    z = x_pred + 0.01 * jnp.ones(2)
    y = z - H2 @ x_pred
    S = H2 @ P_pred @ H2.T + R2

    K = jnp.linalg.solve(S.T, (P_pred @ H2.T).T).T
    x_upd = x_pred + K @ y
    IKH = jnp.eye(2) - K @ H2
    P_upd = IKH @ P_pred @ IKH.T + K @ R2 @ K.T

    P_inv = jnp.linalg.inv(P_upd)
    _ = P_inv @ P_upd  # noqa: F841

    should_refine = jnp.logical_and(la.norm(y) < 50.0, state[0] > -1e6)
    x_upd = jax.lax.cond(
        should_refine,
        lambda _: x_upd + 1e-12 * jnp.linalg.solve(S + 1e-3 * jnp.eye(2), y),
        lambda _: x_upd,
        operand=None,
    )

    return x_upd, P_upd


# --- 3-state Kalman filter (exercises cholesky, solve, qr, det, slogdet) ---


@el.map
def kf3_step(state: State3, cov: Cov3, info: Info3) -> tuple[State3, Cov3, Info3]:
    x_pred = F3 @ state
    P_pred = F3 @ cov @ F3.T + Q3

    z = x_pred + 0.01 * jnp.ones(3)
    y = z - H3 @ x_pred
    S = H3 @ P_pred @ H3.T + R3

    # cholesky
    L_S = jnp.linalg.cholesky(S)
    _chol_check = L_S @ L_S.T  # noqa: F841

    # solve-based Kalman gain (uses dgetrf + dgetrs)
    K = jnp.linalg.solve(S.T, (P_pred @ H3.T).T).T

    x_upd = x_pred + K @ y
    IKH = jnp.eye(3) - K @ H3
    P_upd = IKH @ P_pred @ IKH.T + K @ R3 @ K.T

    # qr
    Q_f, R_f = jnp.linalg.qr(P_upd)
    P_upd = Q_f @ R_f

    # det / slogdet
    d = jnp.linalg.det(S)
    sign, logdet = jnp.linalg.slogdet(S)
    S_inv_y = jnp.linalg.solve(S, y)
    log_lik = -0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + logdet + y @ S_inv_y)

    def _heavy_cond_branch(_):
        solve_vec = jnp.linalg.solve(S + 1e-3 * jnp.eye(3), y + 1e-3 * jnp.ones(3))
        v = solve_vec + x_upd
        for _ in range(12):
            yaw = jnp.arctan2(v[1], v[0] + 1e-9)
            pitch = jnp.arctan2(v[2], jnp.sqrt(v[0] * v[0] + v[1] * v[1]) + 1e-9)
            c0 = jnp.cos(yaw)
            s0 = jnp.sin(yaw)
            c1 = jnp.cos(pitch)
            s1 = jnp.sin(pitch)
            v = jnp.array(
                [
                    v[0] * c0 - v[1] * s0 + 0.01 * s1,
                    v[0] * s0 + v[1] * c0 + 0.01 * c1,
                    v[2] * c1 + 0.01 * (s0 * c0),
                ],
                dtype=jnp.float64,
            )
        return x_upd + 1e-12 * v

    armed = jnp.logical_and(state[0] > 0.5, state[1] > -1e3)
    trigger = jnp.logical_and(armed, la.norm(x_upd) < 1e8)
    x_upd = jax.lax.cond(trigger, _heavy_cond_branch, lambda _: x_upd, operand=None)

    state_norm = la.norm(x_upd)
    info_out = jnp.array([log_lik, d, sign, state_norm, la.norm(K[:, 0])])
    return x_upd, P_upd, info_out


# --- 6-state EKF (exercises SVD, pseudoinverse, eigh, norm on larger matrices) ---


@el.map_seq
def ekf6_step(state: State6, cov: Cov6, info: Info6) -> tuple[State6, Cov6, Info6]:
    x_pred = F6 @ state
    P_pred = F6 @ cov @ F6.T + Q6

    z = x_pred + 0.001 * jnp.ones(6)
    y = z - H6 @ x_pred
    S = H6 @ P_pred @ H6.T + R6

    # SVD-based pseudoinverse (matching customer's safe_matrix_inverse pattern)
    S_pinv = _safe_matrix_inverse(S)
    K = P_pred @ H6.T @ S_pinv

    x_upd = x_pred + K @ y
    IKH = jnp.eye(6) - K @ H6
    P_upd = IKH @ P_pred @ IKH.T + K @ R6 @ K.T

    # eigh (symmetric eigendecomposition of covariance)
    eigvals, _eigvecs = jnp.linalg.eigh(P_upd)

    should_correct = jnp.logical_and(la.norm(y) < 100.0, eigvals[0] > 0.0)
    x_upd = jax.lax.cond(
        should_correct,
        lambda _: x_upd + 1e-12 * jnp.linalg.solve(P_upd + 1e-3 * jnp.eye(6), y),
        lambda _: x_upd,
        operand=None,
    )

    pos_norm = la.norm(x_upd[:3])
    info_out = jnp.array(
        [
            la.norm(y),
            jnp.max(eigvals),
            jnp.min(eigvals),
            pos_norm,
        ]
    )
    return x_upd, P_upd, info_out


# --- Mode selector (exercises .at[idx].set() scatter-into-constant) ---


@el.map
def mode_step(mode_state: ModeState) -> ModeState:
    active = jnp.logical_and(mode_state[0] > 1, mode_state[1] == 0)
    seed = jax.lax.cond(
        active,
        lambda _: mode_state + jnp.array([1, 0, 0, 0], dtype=jnp.int64),
        lambda _: mode_state,
        operand=None,
    )
    idx = seed[0] % 4
    result = jnp.zeros(4, dtype=jnp.int64)
    return result.at[idx].set(jnp.int64(1))


# --- Uint64 transition steps (exercises U64 components across fused-system boundary) ---
# Mirrors the customer's sim_FT19.py two-function pattern:
#   operation_state_override: writes a uint64 transition table (returns uint64)
#   transition_states: reads the table, mixes with int64 via jnp.where, uses as index
# The JaxTracer bitcast fix normalizes the uint64 Call output to int64 at the
# fused-system boundary so the downstream jnp.where doesn't promote to float64.


@el.map
def uint_override_step(transition: OpTransition) -> OpTransition:
    return jnp.array([1, 2, 3, 0], dtype=jnp.uint64)


@el.map
def uint_use_step(
    transition: OpTransition, op_state: OpState
) -> OpState:
    prev_state = op_state[0]
    looked_up = transition[prev_state % 4]

    switch = prev_state < 3
    new_state = jnp.where(switch, looked_up, prev_state)

    result = jnp.zeros(4, dtype=jnp.int64)
    result = result.at[new_state % 4].set(1)

    return jnp.array(
        [new_state, op_state[1] + 1, op_state[2], op_state[3]],
        dtype=jnp.int64,
    )


def world() -> el.World:
    w = el.World()
    w.spawn(
        KF3Archetype(
            kf3_state=jnp.array([0.0, 1.0, 0.0]),
            kf3_cov=jnp.eye(3) * 10.0,
        ),
        name="tracker3",
    )
    w.spawn(
        EKF6Archetype(
            ekf6_state=jnp.array([0.0, 0.0, 100.0, 10.0, 0.0, -5.0]),
            ekf6_cov=jnp.eye(6) * 100.0,
        ),
        name="tracker6",
    )
    w.spawn(
        MatRhsArchetype(
            mrhs_state=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ),
        name="mat_rhs",
    )
    w.spawn(
        Small2Archetype(
            sm2_state=jnp.array([1.0, 0.5]),
            sm2_cov=jnp.eye(2) * 5.0,
        ),
        name="small2",
    )
    w.spawn(
        ModeArchetype(
            mode_state=jnp.array([0, 0, 0, 0], dtype=jnp.int64),
        ),
        name="mode_sel",
    )
    w.spawn(
        UintOpsArchetype(
            op_transition=jnp.array([1, 2, 3, 0], dtype=jnp.uint64),
            op_state=jnp.array([0, 0, 0, 0], dtype=jnp.uint64),
        ),
        name="uint_ops",
    )
    return w


def system() -> el.System:
    return mat_rhs_step | small2_step | kf3_step | ekf6_step | mode_step | uint_override_step | uint_use_step
