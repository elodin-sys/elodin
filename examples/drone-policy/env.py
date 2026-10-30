"""Hover-recovery environment over the drone-policy world (pure, jit/vmap-able).

One policy step = TICKS_PER_STEP ticks of 1/300 s with the action held (100 Hz). The action is
written into policy_action before the first tick; observations are read from the in-graph
policy_obs / critic_obs after the last tick. No auto-reset inside step.
"""

import typing as ty

import jax
import jax.numpy as jnp
import numpy as np
import policy_io
import policy_world
from dr import (
    TARGET,
    DRParams,
    Throw,
    dr_sample,
    nominal_dr,
    nominal_throw,
    quat_conj,
    quat_rotate,
    throw_sample,
)
from jaxstep import JaxStep

__all__ = [
    "ACT_DIM",
    "CRITIC_DIM",
    "IN_TOL_BONUS",
    "OBS_DIM",
    "DRParams",
    "DroneEnv",
    "EnvState",
    "Throw",
    "dr_sample",
    "nominal_dr",
    "nominal_throw",
    "throw_sample",
]

OBS_DIM = policy_io.OBS_DIM
ACT_DIM = policy_io.ACT_DIM
CRITIC_DIM = policy_io.CRITIC_DIM
TICKS_PER_STEP = 3
STEP_DT = TICKS_PER_STEP / 300.0
SUCCESS_WINDOW = 100  # policy steps (the last 1 s)

TERM_TILT = np.deg2rad(80.0)
TERM_POS = 4.0
TERM_Z = 0.3
TERM_OMEGA = 15.0
TERM_REWARD = -10.0
OK_POS, OK_VEL, OK_TILT, OK_OMEGA = 0.15, 0.25, np.deg2rad(10.0), 0.5
IN_TOL_BONUS = 0.25  # paid on every step that meets all four success criteria


class EnvState(ty.NamedTuple):
    sim: tuple  # component slots, see DroneEnv.slot
    t: jax.Array  # int32 policy steps taken
    target: jax.Array  # f64[3]
    rng: jax.Array  # uint32[2] key data the episode was reset with
    succ_ok: jax.Array  # bool, success criteria held on every step of the window so far
    dr: DRParams
    throw: Throw


def _key_data(key: jax.Array) -> jax.Array:
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(key)
    return key


def _clean(x: jax.Array) -> jax.Array:
    return jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(jnp.float32)


class Metrics(ty.NamedTuple):
    reward: jax.Array
    terminated: jax.Array
    in_window_ok: jax.Array


def reward(e_p, speed, cos_tilt, w_xy, w_z, da, in_tol) -> jax.Array:
    """Reward while alive: > 0.0647 for any non-terminal state (tilt <= 80 deg, |e_p| <= 4 m),
    up to 1.1 + IN_TOL_BONUS inside the success tolerances. w_xy, w_z are the body roll/pitch
    rate norm and yaw rate. The linear |e_p| term (0.08/m) outweighs the low-speed slope of the
    speed term over the gamma-0.995 horizon, so approaching the target pays from anywhere; the
    yaw-rate term is steep inside the 0.5 rad/s success bound."""
    return (
        1.1
        - 0.25 * jnp.tanh(e_p / 0.3)
        - 0.08 * jnp.minimum(e_p, TERM_POS)
        - 0.10 * jnp.tanh(speed)
        - 0.20 * (1.0 - cos_tilt)
        - 0.05 * jnp.tanh(w_xy / 2.0)
        - 0.10 * jnp.tanh(jnp.abs(w_z) / 0.5)
        - 0.05 * jnp.minimum(jnp.dot(da, da), 1.0)
        + IN_TOL_BONUS * in_tol
    )


def evaluate(world_pos, world_vel, target, action, a_prev, finite) -> Metrics:
    """Reward, termination and the per-step success test from truth (one env); tilt is the body
    z axis from world vertical, the reward's rate split is in the body frame."""
    q, p = world_pos[:4], world_pos[4:]
    omega, v = world_vel[:3], world_vel[3:]
    e_p = jnp.linalg.norm(p - target)
    speed = jnp.linalg.norm(v)
    cos_tilt = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
    w = jnp.linalg.norm(omega)
    w_body = quat_rotate(quat_conj(q), omega)
    w_xy = jnp.sqrt(w_body[0] ** 2 + w_body[1] ** 2)
    ok = (e_p < OK_POS) & (speed < OK_VEL) & (cos_tilt > np.cos(OK_TILT)) & (w < OK_OMEGA)
    r = reward(e_p, speed, cos_tilt, w_xy, w_body[2], action - a_prev, ok.astype(e_p.dtype))
    terminated = (
        ~finite
        | (cos_tilt < np.cos(TERM_TILT))
        | (e_p > TERM_POS)
        | (p[2] < TERM_Z)
        | (w > TERM_OMEGA)
    )
    return Metrics(jnp.where(terminated, TERM_REWARD, r), terminated, ok & ~terminated)


class DroneEnv:
    obs_dim = OBS_DIM
    critic_dim = CRITIC_DIM
    act_dim = ACT_DIM

    def __init__(
        self,
        estimator: str = "mekf6",
        episode_steps: int = 500,
        rate_pid: str = "dmeas",
        effects: ty.Sequence[str] = (),
        trace: ty.Sequence[str] = (),
    ):
        """estimator="mekf6" (default) is the roci-adcs 6-state MEKF that fsw/mekf is built on,
        with sim-tuned noise, an adaptive accel sigma and an accel gate (mekf6.py); it is not
        configured like the fsw/mekf service.
        rate_pid="dmeas" (default) takes the rate-PID derivative on the measurement, so a
        100 Hz action step gives no D kick; with "stock" (derivative on error) the Talon D gains
        turn exploration noise on a[0:3] into mixer saturation and z crashes within ~1 s.
        effects (policy_world.EFFECTS) and trace are eval-only and off by default; step() returns
        the traced slots after every tick in info["trace"]. With "gyroscopic" built in, the
        gyro_gain component starts at 1 (the effect on) unless the caller sets it."""
        self.estimator = estimator
        self.episode_steps = episode_steps
        self.rate_pid = rate_pid
        self.effects, self.trace = tuple(effects), tuple(trace)
        self._unused_init = policy_world.UNUSED_INIT[rate_pid]
        w, _ = policy_world.world(1, estimator, extra=policy_world.effect_archetypes(self.effects))
        system = policy_world.system(estimator, rate_pid=rate_pid, effects=self.effects)
        self.js = JaxStep(w, system, 300.0)
        if "vio_long" in self.effects and not self.js.has("vio_buf"):
            self._unused_init = self._unused_init | {"vio_buf"}
        self.template = self.js.init
        self._float_slots = [
            d for d, a in enumerate(self.template) if jnp.issubdtype(a.dtype, jnp.floating)
        ]
        self.i_action = self.js.slot("policy_action")
        self.i_obs = self.js.slot("policy_obs")
        self.i_critic = self.js.slot("critic_obs")
        self.i_pos = self.js.slot("world_pos")
        self.i_vel = self.js.slot("world_vel")

    def slot(self, name: str) -> int:
        return self.js.slot(name)

    def reset(
        self, key: jax.Array, dr_scale, throw_scale=None
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """dr_scale scales plant, sensor and latency DR; throw_scale (default dr_scale) scales
        the start throw."""
        throw_scale = dr_scale if throw_scale is None else throw_scale
        k_dr, k_throw, k_noise = jax.random.split(key, 3)
        dr = dr_sample(k_dr, dr_scale)
        throw = throw_sample(k_throw, throw_scale)
        return self.reset_with(k_noise, dr, throw)

    def reset_with(
        self, key: jax.Array, dr: DRParams, throw: Throw
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        target = jnp.array(TARGET)
        init = policy_world.drone_init(dr, throw, key, self.estimator, target)
        if self.effects:
            init = policy_world.effect_init(init, self.effects)
        state = list(self.template)
        for name, value in init.items():
            if name in self._unused_init:
                continue
            if not self.js.has(name):
                raise KeyError(f"drone_init value {name!r} has no state slot in this pipeline")
            d = self.js.slot(name)
            state[d] = (
                jnp.asarray(value).astype(self.template[d].dtype).reshape(self.template[d].shape)
            )
        state = EnvState(
            sim=tuple(state),
            t=jnp.int32(0),
            target=target,
            rng=_key_data(key),
            succ_ok=jnp.bool_(True),
            dr=dr,
            throw=throw,
        )
        return state, *self.observe(state)

    def observe(self, state: EnvState) -> tuple[jax.Array, jax.Array]:
        """(obs f32, critic_obs f32); non-finite entries (a diverged, terminated env) -> 0."""
        return _clean(state.sim[self.i_obs]), _clean(state.sim[self.i_critic])

    def finite(self, sim: tuple) -> jax.Array:
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(sim[d])) for d in self._float_slots]))

    def step(self, state: EnvState, action: jax.Array):
        a = jnp.clip(jnp.asarray(action, jnp.float64), -1.0, 1.0)
        sim = list(state.sim)
        a_prev = sim[self.i_action]
        sim[self.i_action] = a
        if self.trace:
            sim, trace = self._run_traced(tuple(sim))
        else:
            sim = self.js.run(tuple(sim), TICKS_PER_STEP)
        t = state.t + 1
        m = evaluate(sim[self.i_pos], sim[self.i_vel], state.target, a, a_prev, self.finite(sim))
        in_window = t >= self.episode_steps - SUCCESS_WINDOW
        succ_ok = state.succ_ok & (m.in_window_ok | ~in_window)
        truncated = (t >= self.episode_steps) & ~m.terminated
        done = m.terminated | truncated
        new_state = state._replace(sim=sim, t=t, succ_ok=succ_ok)
        obs, critic = self.observe(new_state)
        info = {
            "terminated": m.terminated,
            "truncated": truncated,
            "success": truncated & succ_ok,
            "t": t,
        }
        if self.trace:
            info["trace"] = trace
        return new_state, obs, critic, m.reward.astype(jnp.float32), done, info

    def _run_traced(self, sim: tuple) -> tuple[tuple, dict]:
        slots = [self.js.slot(n) for n in self.trace]

        def tick(s, _):
            s = self.js.step(s)
            return s, {n: s[d] for n, d in zip(self.trace, slots)}

        return jax.lax.scan(tick, sim, None, length=TICKS_PER_STEP)
