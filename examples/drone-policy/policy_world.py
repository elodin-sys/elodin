"""The drone-policy world: stock examples/drone plant, sensors and rate loop, plus the DR
plant, action path, VIO emulation, an attitude estimator and the observation systems.

Estimators (attitude_estimate is what the actor observes):
- "mekf6" (default): mekf6.mekf6_update at the head of the 300 Hz chain, with the
  MEKF6 bias and covariance state; reset to mekf6.reset_values(q_hat).
- "truth": attitude_estimate copied from world_pos at the head of the chain (oracle). The
  observation systems run at the end of the tick, so the actor sees the attitude one tick
  (1/300 s) late, the same staleness as the MEKFs' sensor inputs.
- "mekf18": the stock 18-state mekf.update_filter | mekf.att_est_error (equivalence tests).
"""

import dataclasses
import typing as ty

import _drone_path  # noqa: F401  (examples/drone on sys.path)
import control
import elodin as el
import jax
import jax.numpy as jnp
import mekf
import mekf6
import motors
import numpy as np
import plant_dr
import policy_io
import sensors
import sim
import telemetry
import vehicles
import vio
from config import Config
from dr import (
    TARGET,
    DRParams,
    G,
    Throw,
    quat_conj,
    quat_from_rotvec,
    quat_mul,
    quat_rotate,
    yaw_of,
)

vehicles.TALON_QUAD_CONFIG.set_as_global()

ESTIMATORS = ("mekf18", "truth", "mekf6")
_DTYPES = {"F64": np.float64, "F32": np.float32, "U64": np.uint64, "I64": np.int64}


def estimator_archetypes(estimator: str) -> list[el.Archetype]:
    if estimator == "mekf18":
        return [mekf.MEKF()]
    if estimator == "truth":
        return [policy_io.AttitudeEstimate()]
    if estimator == "mekf6":
        return [policy_io.AttitudeEstimate(), mekf6.MEKF6()]
    raise ValueError(f"unknown estimator {estimator!r}, expected one of {ESTIMATORS}")


def estimator_head(estimator: str) -> el.System:
    """The estimator's system at the head of the 300 Hz chain."""
    if estimator == "mekf18":
        return mekf.update_filter | mekf.att_est_error
    if estimator == "truth":
        return policy_io.truth_estimate
    if estimator == "mekf6":
        return mekf6.mekf6_update
    raise ValueError(f"unknown estimator {estimator!r}, expected one of {ESTIMATORS}")


def drone_archetypes(estimator: str = "mekf6") -> list[el.Archetype]:
    """Default (nominal, stock-equivalent) archetypes of one drone."""
    cfg = Config.GLOBAL
    return [
        el.Body(world_pos=cfg.spatial_transform, inertia=cfg.spatial_inertia),
        sim.Drone(),
        motors.Motors(),
        sensors.IMU(),
        control.AttitudeController(),
        *estimator_archetypes(estimator),
        telemetry.Telemetry(),
        plant_dr.Plant(),
        policy_io.PolicyIO(),
        vio.VIO(),
    ]


def _with_values(archetype: el.Archetype, init: dict, used: set) -> el.Archetype:
    hints = ty.get_type_hints(type(archetype), include_extras=True)
    updates = {}
    for f in dataclasses.fields(archetype):
        name = el.Component.name(hints[f.name])
        if name in init:
            dtype = _DTYPES[str(el.Component.of(hints[f.name]).ty.ty).split(".")[-1]]
            updates[f.name] = np.asarray(init[name], dtype=dtype)
            used.add(name)
    return dataclasses.replace(archetype, **updates)


def spawn_drone(
    world: el.World,
    name: str,
    estimator: str = "mekf6",
    init: dict | None = None,
    extra: ty.Sequence[el.Archetype] = (),
) -> el.EntityId:
    """Spawn one drone; `init` maps component names to initial values (see drone_init) and
    `extra` adds archetypes (for example the state of an in-graph policy system)."""
    init, used = init or {}, set()
    archetypes = [_with_values(a, init, used) for a in [*drone_archetypes(estimator), *extra]]
    if unknown := set(init) - used:
        raise ValueError(f"init values for components no archetype has: {sorted(unknown)}")
    return world.spawn(archetypes, name=name)


def world(
    n: int = 1,
    estimator: str = "mekf6",
    inits: ty.Sequence[dict | None] | None = None,
    extra: ty.Sequence[el.Archetype] = (),
) -> tuple[el.World, list[el.EntityId]]:
    """World of n drones named "drone" (n = 1) or "drone_0".."drone_{n-1}", no schematic."""
    w = el.World()
    names = ["drone"] if n == 1 else [f"drone_{i}" for i in range(n)]
    inits = inits or [None] * n
    pairs = zip(names, inits, strict=True)
    return w, [spawn_drone(w, nm, estimator, i, extra) for nm, i in pairs]


RATE_PIDS = {"stock": control.rate_pid_state, "dmeas": policy_io.rate_pid_dmeas}
# drone_init values that no system of the pipeline reads with this rate PID
UNUSED_INIT = {"stock": frozenset({"rate_sp_prev"}), "dmeas": frozenset()}

# Eval-only variants, all off by default: "gyroscopic" adds plant_dr.gyroscopic to the effectors,
# "vio_long" reads the VIO through a VIO_LONG_LEN-tick buffer (delays beyond vio.BUF_LEN - 1).
EFFECTS = ("gyroscopic", "vio_long")
VIO_LONG_LEN = 16

VioBufLong = ty.Annotated[
    jax.Array,
    el.Component(
        "vio_buf_long",
        el.ComponentType(el.PrimitiveType.F64, (VIO_LONG_LEN, vio.MEAS_DIM)),
        metadata={"transient": "true"},
    ),
]


@dataclasses.dataclass
class VioLong(el.Archetype):
    vio_buf_long: VioBufLong = dataclasses.field(
        default_factory=lambda: jnp.tile(vio.initial_meas(), (VIO_LONG_LEN, 1))
    )


@el.map
def vio_update_long(
    pos: el.WorldPos,
    vel: el.WorldVel,
    tick: sensors.SensorTick,
    params: vio.VioParams,
    seed: vio.VioSeed,
    clock: vio.VioClock,
    buf: VioBufLong,
) -> tuple[vio.VioClock, VioBufLong, vio.VioMeas]:
    # vio.vio_update over the long buffer
    clock = clock + 1.0
    due = clock >= vio.VIO_PERIOD
    clock = jnp.where(due, 0.0, clock)
    key = jax.random.fold_in(jax.random.fold_in(vio.VIO_KEY, seed), tick)
    row = jnp.where(due, vio.sample(pos, vel, params, key), buf[0])
    buf = jnp.concatenate([row[None], buf[:-1]])
    return clock, buf, vio.one_hot(params[vio.DELAY], VIO_LONG_LEN) @ buf


def effect_archetypes(effects: ty.Sequence[str]) -> list[el.Archetype]:
    out = []
    if "gyroscopic" in effects:
        out.append(plant_dr.Gyroscopic())
    if "vio_long" in effects:
        out.append(VioLong())
    return out


def effect_init(init: dict, effects: ty.Sequence[str]) -> dict:
    """drone_init values plus the start state of the effects' components."""
    if "vio_long" in effects:
        init = {**init, "vio_buf_long": jnp.tile(init["vio_meas"], (VIO_LONG_LEN, 1))}
    return init


def system(
    estimator: str = "mekf6",
    policy: el.System | None = None,
    setpoints: str = "policy",
    critic: bool = True,
    rate_pid: str = "dmeas",
    effects: ty.Sequence[str] = (),
) -> el.System:
    """300 Hz chain: [policy] | estimator head | action_apply | rate PID | collective rate
    control | mixer, then 3 x (six_dof(1/900, DR effectors) | imu | telemetry), then VIO and
    the observation systems at the end of the tick.

    setpoints="rate_flight_plan" replaces the policy with the stock rate flight plan at
    hover + 0.05 collective (the nominal-equivalence configuration, used with rate_pid="stock").
    rate_pid="dmeas" (default) is policy_io.rate_pid_dmeas (derivative on measurement, no
    kick on setpoint steps), "stock" control.rate_pid_state (derivative on error). effects
    (EFFECTS, eval only) need the components of effect_archetypes in the world."""
    if unknown := set(effects) - set(EFFECTS):
        raise ValueError(f"unknown effects {sorted(unknown)}, expected some of {EFFECTS}")
    cfg = Config.GLOBAL
    pre = []
    if policy is not None:
        pre.append(policy)
    if setpoints == "rate_flight_plan":
        pre.append(control.rate_flight_plan | policy_io.setpoint_to_action)
    elif setpoints != "policy":
        raise ValueError(f"unknown setpoints {setpoints!r}")
    pre.append(estimator_head(estimator))
    chain = (
        policy_io.action_apply
        | RATE_PIDS[rate_pid]
        | policy_io.rate_control_collective
        | motors.output
    )
    for s in reversed(pre):
        chain = s | chain

    effectors = plant_dr.effectors
    if "gyroscopic" in effects:
        effectors = effectors | plant_dr.gyroscopic
    substep = el.six_dof(cfg.fast_loop_time_step, effectors, integrator=el.Integrator.SemiImplicit)
    substep = substep | sensors.imu | telemetry.telemetry
    n_inner = round(cfg.dt / cfg.fast_loop_time_step)
    assert n_inner == 3

    tail = (vio_update_long if "vio_long" in effects else vio.vio_update) | policy_io.actor_obs
    if critic:
        tail = tail | policy_io.critic_obs
    return chain | sim.inner_loop(n_inner, substep) | tail


def drone_init(
    dr: DRParams, throw: Throw, key: jax.Array, estimator: str = "mekf6", target=TARGET
) -> dict[str, jax.Array]:
    """Initial component values for one drone from DR params, a throw and a noise key.

    Pure jax.numpy (vmap-able); env.reset_with writes these into the state tuple and
    spawn_drone writes them into a world for Cranelift/editor runs."""
    cfg = Config.GLOBAL
    kt, _ = plant_dr.motor_constants()
    target = jnp.asarray(target, jnp.float64)
    q = throw.quat / jnp.linalg.norm(throw.quat)
    p = target + throw.pos_offset
    world_pos = jnp.concatenate([q, p])
    world_vel = jnp.concatenate([quat_rotate(q, throw.body_rate), throw.vel])
    mass = cfg.mass * dr.mass_scale
    inertia = jnp.concatenate([cfg.inertia_diagonal * dr.inertia_scale, jnp.zeros(3), mass[None]])

    rpm = jnp.asarray(plant_dr.hover_trim().rpm) * throw.rpm_scale
    thrust = kt * rpm**2 * dr.thrust_scale
    accel = jnp.sum(cfg.motor_thrust_directions * thrust[:, None], axis=0) / (mass * G)
    gyro = throw.body_rate + dr.gyro_bias

    k_tick, k_seed, k_clock = jax.random.split(key, 3)
    sensor_tick = jax.random.randint(k_tick, (), 0, 2**28).astype(jnp.uint64) * jnp.uint64(9)
    vio_seed = jax.random.randint(k_seed, (), 0, 2**31 - 1).astype(jnp.uint64)
    vio_clock = jax.random.randint(k_clock, (), 0, vio.VIO_PERIOD).astype(jnp.float64)

    q_hat = q if estimator == "truth" else quat_mul(q, quat_from_rotvec(throw.est_err))
    meas = jnp.concatenate([p + dr.vio_pos_bias, throw.vel, yaw_of(q)[None]])
    act_delay = dr.act_delay.astype(jnp.float64)
    vio_params = jnp.concatenate(
        [
            dr.vio_pos_bias,
            jnp.stack([dr.vio_pos_sigma, dr.vio_vel_sigma, dr.vio_delay.astype(jnp.float64)]),
        ]
    )
    params = plant_dr.pack(
        dr.thrust_scale, dr.kq_scale, dr.motor_tau, dr.drag_coeff, dr.wind, dr.force, dr.torque
    )
    action = jnp.zeros(policy_io.ACT_DIM)
    p_int = jnp.zeros(3)
    obs = policy_io.actor_features(q_hat, gyro, meas, target, p_int, action)
    dr_feat = policy_io.dr_features(
        mass, inertia[:3], params, dr.rate_gain_scale, dr.gyro_bias, act_delay, vio_params
    )
    critic = policy_io.critic_features(obs, q, p, world_vel[:3], throw.vel, target, dr_feat)

    init = {
        "world_pos": world_pos,
        "world_vel": world_vel,
        "inertia": inertia,
        "motor_rpm": rpm,
        "rate_pid_state": policy_io.rate_pid_reset(gyro),
        "rate_sp_prev": jnp.zeros(3),
        "sensor_tick": sensor_tick,
        "gyro_bias": dr.gyro_bias,
        "gyro": gyro,
        "gyro_lpf_delay": jnp.tile(gyro, (4, 1)),
        "accel": accel,
        "accel_lpf_delay": jnp.tile(accel, (4, 1)),
        "magnetometer": quat_rotate(quat_conj(q), jnp.array([0.0, 1.0, 0.0])),
        "attitude_estimate": q_hat,
        "drone_params": params,
        "policy_action": action,
        "act_fifo": jnp.zeros((policy_io.FIFO_LEN, policy_io.ACT_DIM)),
        "act_delay": act_delay,
        "rate_gain_scale": dr.rate_gain_scale,
        "target_pos": target,
        "p_int": p_int,
        "policy_obs": obs,
        "critic_obs": critic,
        "vio_params": vio_params,
        "vio_seed": vio_seed,
        "vio_clock": vio_clock,
        "vio_buf": jnp.tile(meas, (vio.BUF_LEN, 1)),
        "vio_meas": meas,
    }
    if estimator == "mekf6":
        init.update(mekf6.reset_values(q_hat))
    return init
