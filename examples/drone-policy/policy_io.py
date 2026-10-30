"""Policy action path (action FIFO, CTBR scaling around the calibrated hover, rate-PID reset,
the derivative-on-measurement rate PID used by default, collective rate controller) and the
in-graph actor/critic observation systems."""

import typing as ty
from dataclasses import dataclass, field

import _drone_path  # noqa: F401  (examples/drone on sys.path)
import control
import elodin as el
import filter
import jax
import jax.numpy as jnp
import mekf
import motors
import numpy as np
import params
import plant_dr
import sensors
import vio
from config import Config
from dr import TARGET, quat_conj, quat_rotate, yaw_of

OBS_DIM = 19
ACT_DIM = 4
DR_DIM = 30
CRITIC_DIM = OBS_DIM + 12 + DR_DIM
FIFO_LEN = 4
ACT_SCALE = np.array([3.0, 3.0, 1.5])  # rad/s at |a| = 1, body FLU
# bisection on the JAX pipeline: zero mean a_z at rest in the hover trim attitude (net thrust
# vertical, dr.trim_quat), torque-free motor rpm, noise on (0.6898828 when level)
HOVER_COLLECTIVE = 0.689509
COLLECTIVE_SCALE = 0.311  # collective per unit a[3]
# rate-PID integrator settled in that hover (pitch trim for the tilted motors), seeded at reset
HOVER_RATE_I = np.array([0.0, 0.1067, 0.0])
P_INT_LIMIT = 1.0  # m s, per world axis
P_INT_GATE = 1.0  # m, p_int integrates only while |VIO position error| is below this
Z_UP = np.array([0.0, 0.0, 1.0])

PolicyAction = ty.Annotated[
    jax.Array,
    el.Component(
        "policy_action",
        el.ComponentType(el.PrimitiveType.F64, (ACT_DIM,)),
        metadata={"priority": 301, "element_names": "p,q,r,c"},
    ),
]
ActFifo = ty.Annotated[
    jax.Array,
    el.Component(
        "act_fifo",
        el.ComponentType(el.PrimitiveType.F64, (FIFO_LEN, ACT_DIM)),
        metadata={"transient": "true"},
    ),
]
ActDelay = ty.Annotated[
    jax.Array, el.Component("act_delay", el.ComponentType.F64, metadata={"transient": "true"})
]
CollectiveCmd = ty.Annotated[
    jax.Array, el.Component("collective_cmd", el.ComponentType.F64, metadata={"priority": 298})
]
RateGainScale = ty.Annotated[
    jax.Array,
    el.Component(
        "rate_gain_scale",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"transient": "true", "element_names": "r,p,y"},
    ),
]
TargetPos = ty.Annotated[
    jax.Array,
    el.Component(
        "target_pos",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"transient": "true", "element_names": "x,y,z"},
    ),
]
PosInt = ty.Annotated[
    jax.Array,
    el.Component(
        "p_int",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"transient": "true", "element_names": "x,y,z"},
    ),
]
PolicyObs = ty.Annotated[
    jax.Array,
    el.Component("policy_obs", el.ComponentType(el.PrimitiveType.F64, (OBS_DIM,))),
]
CriticObs = ty.Annotated[
    jax.Array,
    el.Component(
        "critic_obs",
        el.ComponentType(el.PrimitiveType.F64, (CRITIC_DIM,)),
        metadata={"transient": "true"},
    ),
]
RateSpPrev = ty.Annotated[
    jax.Array,
    el.Component(
        "rate_sp_prev",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"transient": "true", "element_names": "r,p,y"},
    ),
]


@dataclass
class PolicyIO(el.Archetype):
    policy_action: PolicyAction = field(default_factory=lambda: jnp.zeros(ACT_DIM))
    act_fifo: ActFifo = field(default_factory=lambda: jnp.zeros((FIFO_LEN, ACT_DIM)))
    act_delay: ActDelay = field(default_factory=lambda: jnp.float64(0.0))
    collective_cmd: CollectiveCmd = field(default_factory=lambda: jnp.float64(0.0))
    rate_gain_scale: RateGainScale = field(default_factory=lambda: jnp.ones(3))
    target_pos: TargetPos = field(default_factory=lambda: jnp.array(TARGET))
    p_int: PosInt = field(default_factory=lambda: jnp.zeros(3))
    policy_obs: PolicyObs = field(default_factory=lambda: jnp.zeros(OBS_DIM))
    critic_obs: CriticObs = field(default_factory=lambda: jnp.zeros(CRITIC_DIM))
    rate_sp_prev: RateSpPrev = field(default_factory=lambda: jnp.zeros(3))


@dataclass
class AttitudeEstimate(el.Archetype):
    # attitude_estimate alone, for estimators that do not use the stock mekf.MEKF archetype
    att_est: mekf.AttEst = field(default_factory=lambda: el.Quaternion.identity())


@el.map
def truth_estimate(pos: el.WorldPos) -> mekf.AttEst:
    """Oracle estimate. It runs at the head of the tick and actor_obs at the end, so the actor
    sees the attitude one tick (1/300 s) late."""
    return pos.angular()


def rate_pid_reset(gyro: jax.Array) -> jax.Array:
    """rate_pid_state for an episode start: e seeded with the first rate error at a = 0 (zero
    setpoint) so tick 1 has no error jump and zero derivative, i at the hover trim, d = 0."""
    return jnp.stack([-gyro, jnp.asarray(HOVER_RATE_I), jnp.zeros(3)])


@el.map
def action_apply(
    a: PolicyAction, fifo: ActFifo, delay: ActDelay
) -> tuple[ActFifo, control.AngVelSetpoint, CollectiveCmd]:
    fifo = jnp.concatenate([jnp.clip(a, -1.0, 1.0)[None], fifo[:-1]])
    a = vio.one_hot(delay, FIFO_LEN) @ fifo
    collective = jnp.clip(HOVER_COLLECTIVE + COLLECTIVE_SCALE * a[3], 0.1, 1.0)
    return fifo, ACT_SCALE * a[:3], collective


@el.map
def rate_pid_dmeas(
    state: control.RatePIDState,
    target: control.AngVelSetpoint,
    prev: RateSpPrev,
    gyro: sensors.Gyro,
) -> tuple[control.RatePIDState, RateSpPrev]:
    """control.rate_pid_state with the derivative on the measurement: the D term sees the error
    change at the previous setpoint, so a setpoint step gives no derivative kick. Identical to
    the stock PID while the setpoint is constant."""
    dt = Config.GLOBAL.dt
    e_filter = filter.LPF(
        jnp.array([params.ATC_RAT_RLL_FLTE, params.ATC_RAT_PIT_FLTE, params.ATC_RAT_YAW_FLTE]),
        1.0 / dt,
    )
    d_filter = filter.LPF(
        jnp.array([params.ATC_RAT_RLL_FLTD, params.ATC_RAT_PIT_FLTD, params.ATC_RAT_YAW_FLTD]),
        1.0 / dt,
    )
    e_prev, i_prev, d_prev = state
    e = e_filter.apply(e_prev, target - gyro)
    e_meas = e_filter.apply(e_prev, prev - gyro)
    i = i_prev + (e * dt)
    d = d_filter.apply(d_prev, (e_meas - e_prev) / dt)
    return jnp.array([e, i, d]), target


@el.map
def rate_control_collective(
    state: control.RatePIDState, collective: CollectiveCmd, gain_scale: RateGainScale
) -> motors.MotorInput:
    # control.rate_control with explicit row adds (see plant_dr.row_sum) and per-env gain scale
    g = Config.GLOBAL.control.rate_pid_gains
    mv = (state[0] * g[0] + state[1] * g[1] + state[2] * g[2]) * gain_scale
    return jnp.concatenate([mv, collective[None]])


@el.map
def setpoint_to_action(target: control.AngVelSetpoint) -> PolicyAction:
    # inverse of action_apply for the stock rate_flight_plan at hover + 0.05 collective
    stock = Config.GLOBAL.control.motor_thrust_hover + 0.05
    c = (stock - HOVER_COLLECTIVE) / COLLECTIVE_SCALE
    return jnp.concatenate([target / ACT_SCALE, jnp.array([c])])


def to_heading(yaw: jax.Array, v: jax.Array) -> jax.Array:
    """World vector into the heading frame (z up, x along yaw)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def actor_features(q_hat, gyro, meas, target, p_int, action) -> jax.Array:
    yaw = meas[6]
    return jnp.concatenate(
        [
            quat_rotate(quat_conj(q_hat), Z_UP),
            gyro,
            to_heading(yaw, meas[0:3] - target),
            to_heading(yaw, meas[3:6]),
            to_heading(yaw, p_int),
            action,
        ]
    )


def dr_features(mass, inertia_diag, params, gain_scale, gyro_bias, act_delay, vio_params):
    cfg = Config.GLOBAL
    p = plant_dr
    return jnp.concatenate(
        [
            ((mass / cfg.mass - 1.0) / 0.1)[None],
            (inertia_diag / cfg.inertia_diagonal - 1.0) / 0.2,
            (params[p.THRUST_SCALE] - 1.0) / 0.05,
            (params[p.KQ_SCALE] - 1.0)[None],
            (jnp.log(params[p.MOTOR_TAU] / 0.1) / np.log(2.5))[None],
            ((params[p.DRAG_COEFF] - 0.1) / 0.1)[None],
            params[p.WIND] / 2.0,
            params[p.FORCE],
            params[p.TORQUE] / 0.05,
            (gain_scale - 1.0) / 0.25,
            gyro_bias / 0.01,
            (act_delay / 2.0)[None],
            (vio_params[vio.DELAY] / 6.0)[None],
            (vio_params[vio.POS_SIGMA] / 0.02)[None],
            (vio_params[vio.VEL_SIGMA] / 0.05)[None],
        ]
    )


def critic_features(actor, q, p, w, v, target, dr) -> jax.Array:
    yaw = yaw_of(q)
    q_inv = quat_conj(q)
    return jnp.concatenate(
        [
            actor,
            quat_rotate(q_inv, Z_UP),
            quat_rotate(q_inv, w),
            to_heading(yaw, p - target),
            to_heading(yaw, v),
            dr,
        ]
    )


def update_p_int(p_int, meas, target) -> jax.Array:
    """World-frame integral of the VIO position error, held while |error| >= P_INT_GATE (no
    windup over a throw transient) and clipped to +-P_INT_LIMIT per world axis (the heading-frame
    obs is rotated after the clip, so its x, y can reach sqrt(2) P_INT_LIMIT)."""
    e = meas[0:3] - target
    de = jnp.where(jnp.dot(e, e) < P_INT_GATE**2, e * Config.GLOBAL.dt, 0.0)
    return jnp.clip(p_int + de, -P_INT_LIMIT, P_INT_LIMIT)


@el.map
def actor_obs(
    att: mekf.AttEst,
    gyro: sensors.Gyro,
    meas: vio.VioMeas,
    target: TargetPos,
    p_int: PosInt,
    a: PolicyAction,
) -> tuple[PosInt, PolicyObs]:
    p_int = update_p_int(p_int, meas, target)
    return p_int, actor_features(att.vector(), gyro, meas, target, p_int, a)


@el.map
def critic_obs(
    obs: PolicyObs,
    pos: el.WorldPos,
    vel: el.WorldVel,
    target: TargetPos,
    inertia: el.Inertia,
    params: plant_dr.DroneParams,
    gain_scale: RateGainScale,
    gyro_bias: sensors.GyroBias,
    delay: ActDelay,
    vio_params: vio.VioParams,
) -> CriticObs:
    dr = dr_features(
        inertia.mass(), inertia.inertia_diag(), params, gain_scale, gyro_bias, delay, vio_params
    )
    q = pos.angular().vector()
    return critic_features(obs, q, pos.linear(), vel.angular(), vel.linear(), target, dr)
