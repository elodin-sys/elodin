"""Evaluation and deploy suite for drone-policy controllers.

    python examples/drone-policy/evaluate.py all --runs DIR [DIR ...] --out results/ [--quick]
    python examples/drone-policy/evaluate.py e2 e4 plots --runs DIR --out results/

Controllers are every RUN/policy.npz under the --runs directories (scratch or residual; the done
marker must read 0, or final_eval.json must exist when there is none) plus --baselines (B0
nominal, B0 DR, B1 DR). Policies fly the estimator they were trained with, baselines
--baseline-estimator. Every rollout is deterministic and kept per episode. The episodes are the
held-out protocol's (ppo.make_evaluate's reset keys from --seed 20011); a section with fewer
episodes takes the first ones, so all rows of a section fly the same episodes.

- e1   held-out: 4096 episodes at (1,1), (0.5,1), (0,1); success with a Wilson 95% interval,
       return, terminations by cause, paired diff vs B0 DR; checked against final_eval.json.
- e3   nominal hover at (0,0): position and tilt RMS over the last 2 s of the survivors.
- e9   smoothness at (1,1): std of the per-tick motor_pwm first difference and mean
       |a_t - a_t-1|^2 of the executed action (t >= 2); also the in-distribution reference
       of e2/e4 and the observations e8 replays.
- e2   OOD slices, each alone at full DR, 1024 episodes per point: action latency, VIO delay,
       motor tau and per-motor thrust spread beyond the training ranges.
- e4   unseen effects at full DR, 1024 episodes each and all together: the gyroscopic torque,
       motor tau 0.2 s, thrust spread +-15%, kq scale 2.5, wind 4 m/s.
- e5   estimator gap: every policy on the held-out protocol with the estimators it was not
       trained with (rows of --track-a summary.json files are reused when they have them).
- e8   numpy export: max abs diff vs the jnp policy on 10k recorded observations (float64) and
       the median single-sample latency over 10k calls.
- e6   deploy in-graph (policy_system.py): (a) 1-drone Cranelift vs the JAX env for one episode,
       300 ticks; (b) 256 held-out episodes, 1-drone Cranelift vs JAX; (c) 16 drones in one
       jax-cpu world vs 16 vmapped env episodes; the main.py bench. (b), (c) and the bench fly
       --e6, by default the headline rule: the best seed (by e1 success at (1,1)) of the config
       with the best mean (1,1) success among configs with at least 2 seeds.
- e10  throughput of the training config: env steps/s, simulated flight hours per wall minute.
- plots  heldout_bars.png, e2_ood.png, e4_unseen.png, learning_curves.png.

Writes OUT/results.json (every number with its protocol, updated after each section; run paths
are written relative to their --runs parent) and OUT/summary.md, and e1 writes the per-episode
arrays to OUT/heldout_episodes.npz (about 1 MB; regenerable, so not meant for the repo). The
module imports numpy only; jax and elodin load inside the sections, so NumpyPolicy (the e8
export) runs without them.
"""

import argparse
import dataclasses
import fnmatch
import functools
import json
import math
import os
import re
import subprocess
import sys
import time
import typing as ty

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SECTIONS = ("e1", "e3", "e9", "e2", "e4", "e5", "e8", "e6", "e10", "plots")
SCALES = ((1.0, 1.0), (0.5, 1.0), (0.0, 1.0))
BASELINES = {
    "b0-nominal": "B0 nominal",
    "b0-dr": "B0 DR",
    "b1-nominal": "B1 nominal",
    "b1-dr": "B1 DR",
}
REFERENCE = "b0-dr"
ESTIMATORS = ("mekf6", "truth")
CAUSES = ("nonfinite", "tilt", "omega", "z_low", "pos_far")  # cascade_b0.CAUSES
Z95 = 1.959964
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060}
THRUST_HALF_WIDTH = 0.05  # dr.dr_sample: per-motor thrust x U(0.95, 1.05)
MAX_DELAY_STEPS = 2  # whole policy steps of extra latency DelayEnv can add
MAX_ACT_DELAY = 3 * MAX_DELAY_STEPS + 2  # ticks: DelayEnv steps plus the graph FIFO's 2
VIO_LONG_LEN = 16  # policy_world.VIO_LONG_LEN: delays up to 15 ticks
ACT_OBS = slice(15, 19)  # the policy's own last action in the actor obs
E2_SLICES = {
    "act_delay": {"values": (0, 1, 2, 3, 4, 6), "unit": "ticks", "train": (0, 2)},
    "vio_delay": {"values": (0, 3, 6, 9, 12), "unit": "ticks", "train": (0, 6)},
    "motor_tau": {"values": (0.04, 0.1, 0.15, 0.2, 0.3), "unit": "s", "train": (0.04, 0.15)},
    "thrust_spread": {"values": (0.0, 0.05, 0.1, 0.15), "unit": "half-width", "train": (0, 0.05)},
}
E4_EFFECTS = {
    "gyroscopic": {"gyro": 1.0},
    "motor_tau_0.2": {"motor_tau": 0.2},
    "thrust_spread_0.15": {"thrust_spread": 0.15},
    "kq_scale_2.5": {"kq_scale": 2.5},
    "wind_4": {"wind_speed": 4.0},
}
E4_EFFECTS["all"] = {k: v for e in E4_EFFECTS.values() for k, v in e.items()}
# categorical slots of the reference palette, in order; baselines recede in greys
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREYS = {"b0-nominal": "#9a9893", "b0-dr": "#52514e", "b1-nominal": "#c3c2b7", "b1-dr": "#0b0b0b"}


@dataclasses.dataclass(frozen=True)
class Sizes:
    heldout: int = 4096
    batch: int = 1024
    slice: int = 1024
    e2_values: int = 0  # 0: every value; 2: the first and last of each slice
    e6_parity_ticks: int = 300
    e6_episodes: int = 256
    e6_ticks: int = 1500
    e6_drones: int = 16
    e8_obs: int = 10000
    e8_calls: int = 10000
    e10_envs: int = 1024
    e10_steps: int = 32
    e10_repeats: int = 5
    bench_ticks: int = 1500


FULL = Sizes()
QUICK = Sizes(
    heldout=64,
    batch=64,
    slice=64,
    e2_values=2,
    e6_parity_ticks=60,
    e6_episodes=2,
    e6_ticks=300,
    e6_drones=2,
    e8_obs=500,
    e8_calls=500,
    e10_envs=64,
    e10_steps=4,
    e10_repeats=2,
    bench_ticks=30,
)


class Override(ty.NamedTuple):
    """Per-episode DR overrides of the eval env; a negative value keeps the DR draw. The
    overrides assume full DR (dr_scale 1): thrust_spread rescales a full-scale draw."""

    act_delay: ty.Any = -1  # ticks of extra action latency (DelayEnv beyond 2)
    vio_delay: ty.Any = -1  # ticks (the long VIO buffer beyond 7)
    motor_tau: ty.Any = -1.0  # s
    thrust_spread: ty.Any = -1.0  # per-motor half-width; rescales the full-DR draw
    kq_scale: ty.Any = -1.0
    wind_speed: ty.Any = -1.0  # m/s horizontal, in the drawn direction (vertical kept)
    gyro: ty.Any = 0.0  # gain of the gyroscopic effector


class DelayState(ty.NamedTuple):
    env: ty.Any
    buf: ty.Any  # f64[MAX_DELAY_STEPS, 4] previous commanded actions, newest first
    k: ty.Any  # int32 whole policy steps of extra latency


def _6g(x) -> float | None:
    x = float(x)
    return float(f"{x:.6g}") if math.isfinite(x) else None


def wilson(k: int, n: int, z: float = Z95) -> list[float]:
    p, d = k / n, 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 6), round(centre + half, 6)]


def paired(a, b) -> dict:
    """Success of a minus b on the same episodes, with the standard error of the mean diff."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    d = a.astype(np.float64) - b.astype(np.float64)
    se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else 0.0
    return {
        "diff": _6g(d.mean()),
        "se": _6g(se),
        "only": int(np.sum(a & ~b)),
        "ref_only": int(np.sum(b & ~a)),
    }


def stats(ep: dict) -> dict:
    n, k = int(ep["success"].size), int(np.sum(ep["success"]))
    term = np.asarray(ep["terminated"], bool)
    return {
        "episodes": n,
        "successes": k,
        "success_rate": _6g(k / n),
        "wilson95": wilson(k, n),
        "terminated_rate": _6g(term.mean()),
        "mean_return": _6g(np.mean(ep["ret"], dtype=np.float64)),
        "mean_length": _6g(np.mean(ep["length"], dtype=np.float64)),
        "causes": {c: int(np.sum(term & (ep["cause"] == i))) for i, c in enumerate(CAUSES)},
    }


def hover_stats(ep: dict) -> dict:
    """cascade_b0.summarize's hover metrics: RMS over the last 2 s of the episodes that did not
    terminate (body tilt includes the 1.89 deg trim; axis tilt is the hover thrust axis)."""
    surv = ~np.asarray(ep["terminated"], bool)
    n = max(int(ep["n_win"][surv].sum()), 1)

    def rms(k):
        return float(np.sqrt(ep[k][surv].sum() / n))

    return {
        "survivors": int(surv.sum()),
        "pos_rms_m": _6g(rms("se_p")),
        "tilt_rms_deg": _6g(np.degrees(rms("se_tilt"))),
        "axis_tilt_rms_deg": _6g(np.degrees(rms("se_axis"))),
    }


def smooth_stats(ep: dict) -> dict:
    n_pwm, n_da = int(ep["n_pwm"].sum()), int(ep["n_da"].sum())
    mean = ep["pwm_d"].sum() / max(n_pwm, 1)
    var = max(ep["pwm_d2"].sum() / max(n_pwm, 1) - mean**2, 0.0)
    return {
        "pwm_diff_std": _6g(math.sqrt(var)),
        "pwm_diffs": n_pwm,
        "mean_sq_action_diff": _6g(ep["da2"].sum() / max(n_da, 1)),
        "action_diffs": n_da,
    }


def seed_mean_ci(values) -> tuple[float, float | None, list[float] | None]:
    """(mean, SD over seeds, t 95% interval of the mean); SD and interval need 2 seeds."""
    v = np.asarray(values, np.float64)
    if v.size < 2:
        return float(v.mean()), None, None
    sd = float(v.std(ddof=1))
    half = T95.get(v.size - 1, Z95) * sd / math.sqrt(v.size)
    return float(v.mean()), sd, [float(v.mean() - half), float(v.mean() + half)]


def _rel(path: str) -> str:
    """A run directory relative to its --runs parent (results files carry no home paths)."""
    path = os.path.normpath(os.path.abspath(path))
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))


def _load_avg() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def _finished(run: str) -> bool:
    if not all(os.path.exists(os.path.join(run, f)) for f in ("policy.npz", "config.json")):
        return False
    done = os.path.join(run, "done")
    if os.path.exists(done):
        with open(done) as f:
            return f.read().strip() == "0"
    return os.path.exists(os.path.join(run, "final_eval.json"))


def discover(runs: ty.Sequence[str], baselines, baseline_estimator: str, only=None) -> list:
    out = []
    for b in baselines:
        kind, gain_set = b.split("-", 1)
        out.append(
            {
                "id": b,
                "label": BASELINES[b],
                "kind": kind,
                "gain_set": gain_set,
                "estimator": baseline_estimator,
                "group": b,
            }
        )
    for d in runs:
        for name in sorted(os.listdir(d)):
            run = os.path.join(d, name)
            if not os.path.isdir(run) or not _finished(run):
                continue
            with open(os.path.join(run, "config.json")) as f:
                config = json.load(f)
            out.append(
                {
                    "id": name,
                    "label": name,
                    "kind": "policy",
                    "path": run,
                    "estimator": config["estimator"],
                    "residual": config.get("residual"),
                    "obs_clip": float(config["ppo"]["obs_clip"]),
                    "group": re.sub(r"_s\d+$", "", name),
                }
            )
    if only:
        out = [c for c in out if any(fnmatch.fnmatchcase(c["id"], pat) for pat in only)]
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate controller ids: {ids}")
    return out


# ------------------------------------------------------------------ numpy export (e8)

B0_KEYS = (
    "b0_gains",
    "b0_thrust_axis",
    "b0_a_z_min",
    "b0_g",
    "b0_hover_collective",
    "b0_collective_scale",
    "b0_act_scale",
)


def b0_numpy(p: ty.Mapping, obs: np.ndarray) -> np.ndarray:
    """residual.b0_from_arrays in numpy alone, float64."""
    kp_xy, kd_xy, ki_xy, kp_z, kd_z, ki_z, k_att, k_rp, k_yaw, yaw_max, max_tilt, a_max, h_fl = p[
        "b0_gains"
    ]
    g_acc = p["b0_g"]
    obs = np.asarray(obs, np.float64)
    g_body, gyro, p_err, v, p_int = (obs[..., i : i + 3] for i in range(0, 15, 3))
    kp = np.stack([kp_xy, kp_xy, kp_z])
    kd = np.stack([kd_xy, kd_xy, kd_z])
    ki = np.stack([ki_xy, ki_xy, ki_z])
    a = -kp * p_err - kd * v - ki * p_int
    az = np.clip(a[..., 2] + g_acc, p["b0_a_z_min"], a_max)
    h = np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)
    room = np.sqrt(np.maximum((a_max - az) * (a_max + az), 0.0))
    h_tilt = az * np.sin(max_tilt) / np.cos(max_tilt)
    h_max = np.maximum(np.minimum(h_tilt, room), h_fl * h_tilt)
    s = np.minimum(1.0, h_max / np.maximum(h, 1e-9))
    ax, ay = a[..., 0] * s, a[..., 1] * s

    g = g_body / np.maximum(np.sqrt(np.sum(g_body**2, axis=-1, keepdims=True)), 1e-9)
    c_th = np.sqrt(g[..., 1] ** 2 + g[..., 2] ** 2)
    s_th = -g[..., 0]
    c_ph = g[..., 2] / np.maximum(c_th, 1e-9)
    s_ph = g[..., 1] / np.maximum(c_th, 1e-9)
    ux = c_th * ax - s_th * az
    uz = s_th * ax + c_th * az
    b = np.stack([ux, c_ph * ay + s_ph * uz, -s_ph * ay + c_ph * uz], axis=-1)

    t = p["b0_thrust_axis"]
    thrust = np.sum(b * t, axis=-1)
    n = b / np.maximum(np.sqrt(np.sum(b**2, axis=-1, keepdims=True)), 1e-9)
    axis = np.cross(t, n)
    sin_e = np.sqrt(np.sum(axis**2, axis=-1))
    err = np.arctan2(sin_e, np.sum(t * n, axis=-1))
    w_xy = (k_att * err / np.maximum(sin_e, 1e-9))[..., None] * axis[..., :2]
    w_xy = w_xy + k_rp * (w_xy - gyro[..., :2])
    w_z = gyro[..., 2] - np.clip(k_yaw * gyro[..., 2], -yaw_max, yaw_max)
    w = np.concatenate([w_xy, w_z[..., None]], axis=-1)

    hover = p["b0_hover_collective"]
    col = (hover * thrust / g_acc - hover) / p["b0_collective_scale"]
    act = np.concatenate([w / p["b0_act_scale"], col[..., None]], axis=-1)
    return np.clip(act, -1.0, 1.0)


class NumpyPolicy:
    """Deterministic forward of a policy.npz with numpy alone (no jax, no elodin): obs
    normalization, tanh MLP and clip, and the B0 cascade under a residual policy. The same
    operations in the same order as nn.policy_action / residual.residual_action."""

    def __init__(self, policy: ty.Mapping, dtype=np.float64):
        self.dtype = dtype

        def f(k):
            return np.asarray(policy[k], dtype)

        n = sum(1 for k in policy if k.startswith("actor_w"))
        self.layers = [(f(f"actor_w{i}"), f(f"actor_b{i}")) for i in range(n)]
        self.mean, self.std, self.clip = f("obs_mean"), f("obs_std"), f("obs_clip")
        self.residual = "residual_base" in policy
        if self.residual:
            self.b0 = {k: np.asarray(policy[k], np.float64) for k in B0_KEYS}
            self.scale = f("residual_scale")

    def mlp(self, x: np.ndarray) -> np.ndarray:
        x = np.minimum(np.maximum((x - self.mean) / self.std, -self.clip), self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.minimum(np.maximum(x, -1.0), 1.0)

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, self.dtype)
        if not self.residual:
            return self.mlp(obs)
        base = b0_numpy(self.b0, obs)
        res = self.mlp(np.concatenate([obs, base.astype(self.dtype)], axis=-1))
        return np.clip(base + self.scale * res, -1.0, 1.0)


# ------------------------------------------------------------------ eval envs (jax)


def override_dr(dr, ov: Override):
    import jax.numpy as jnp

    def pick(v, x):
        return jnp.where(v >= 0, jnp.asarray(v).astype(x.dtype), x)

    spread = 1.0 + ov.thrust_spread / THRUST_HALF_WIDTH * (dr.thrust_scale - 1.0)
    heading = jnp.arctan2(dr.wind[1], dr.wind[0])
    wind = jnp.stack(
        [ov.wind_speed * jnp.cos(heading), ov.wind_speed * jnp.sin(heading), dr.wind[2]]
    )
    return dr._replace(
        act_delay=pick(ov.act_delay, dr.act_delay),
        vio_delay=pick(ov.vio_delay, dr.vio_delay),
        motor_tau=pick(ov.motor_tau, dr.motor_tau),
        kq_scale=pick(ov.kq_scale, dr.kq_scale),
        thrust_scale=jnp.where(ov.thrust_spread >= 0, spread, dr.thrust_scale),
        wind=jnp.where(ov.wind_speed >= 0, wind, dr.wind),
    )


class DelayEnv:
    """Action latency past the graph's 4-slot FIFO: dr.act_delay = d ticks runs as k = d // 3
    whole policy steps here plus d % 3 ticks in the graph. The policy still sees its own latest
    action in obs[15:19], as with the in-graph FIFO alone, so d <= 2 is the wrapped env bit for
    bit and d = 3 flies the graph's 3-tick delay. The reward's action-change term and the
    critic's delay feature see the executed action and d % 3."""

    def __init__(self, env, max_steps: int = MAX_DELAY_STEPS):
        self.env, self.max_steps = env, max_steps

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, key, dr_scale, throw_scale=None):
        import jax
        from dr import dr_sample, throw_sample

        throw_scale = dr_scale if throw_scale is None else throw_scale
        k_dr, k_throw, k_noise = jax.random.split(key, 3)
        return self.reset_with(
            k_noise, dr_sample(k_dr, dr_scale), throw_sample(k_throw, throw_scale)
        )

    def reset_with(self, key, dr, throw):
        import jax.numpy as jnp
        from env import ACT_DIM, TICKS_PER_STEP

        d = jnp.asarray(dr.act_delay, jnp.int32)
        k = d // TICKS_PER_STEP
        dr = dr._replace(act_delay=d - k * TICKS_PER_STEP)
        state, obs, critic = self.env.reset_with(key, dr, throw)
        return DelayState(state, jnp.zeros((self.max_steps, ACT_DIM)), k), obs, critic

    def step(self, state: DelayState, action):
        import jax.numpy as jnp

        a = jnp.clip(jnp.asarray(action, jnp.float64), -1.0, 1.0)
        rows = jnp.where(jnp.arange(1, self.max_steps + 1) == state.k, 1.0, 0.0)
        executed = jnp.where(state.k == 0, a, rows @ state.buf)
        s, obs, critic, reward, done, info = self.env.step(state.env, executed)
        obs = obs.at[ACT_OBS].set(a.astype(obs.dtype))
        critic = critic.at[ACT_OBS].set(a.astype(critic.dtype))
        buf = jnp.concatenate([a[None], state.buf[:-1]])
        return DelayState(s, buf, state.k), obs, critic, reward, done, info


def _drone_env(env):
    from env import DroneEnv

    while not isinstance(env, DroneEnv):
        env = env.env
    return env


def make_act(kind: str, obs_clip: float):
    """Batched act(params, obs, critic) of make_evaluate (policy) or cascade_b0 (b0, b1)."""
    import cascade_b0
    import jax
    import jax.numpy as jnp
    import nn
    import ppo

    def act(params, obs, critic):
        if kind == "policy":
            actor, obs_rms = params
            mean, _ = ppo.actor_dist(actor, nn.normalize(obs_rms, obs, obs_clip))
            return jnp.clip(mean, -1.0, 1.0)
        if kind == "b0":
            return jax.vmap(lambda o: cascade_b0.b0_action(o, params))(obs)
        return jax.vmap(lambda o, c: cascade_b0.b1_action(o, c, params))(obs, critic)

    return act


def make_episodes(env, kind: str, obs_clip: float, variant: str, devices: int, record: int):
    """-> run(params, keys, dr, throw, ov) -> (per-episode numpy dict, recorded obs or None).

    "plain" resets with env.reset (make_evaluate's episodes); "eval" resets through the
    Override (DR overrides, the gyroscopic gain) and also returns the obs the policy acted on
    and the alive mask for the first `record` episodes of every device shard."""
    import cascade_b0
    import jax
    import jax.numpy as jnp
    import ppo
    from dr import dr_sample, quat_rotate, throw_sample
    from env import TERM_OMEGA, TERM_POS, TERM_TILT, TERM_Z, EnvState
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    f32, f64, i32 = jnp.float32, jnp.float64, jnp.int32
    plant = _drone_env(env)
    act = make_act(kind, obs_clip)
    hover_from = plant.episode_steps - cascade_b0.HOVER_WINDOW
    axis = jnp.asarray(cascade_b0.THRUST_AXIS)
    i_pwm = plant.slot("motor_pwm")
    i_gyro = plant.slot("gyro_gain") if plant.js.has("gyro_gain") else None
    traced = "motor_pwm" in plant.trace

    def inner(state):
        while not isinstance(state, EnvState):
            state = state.env
        return state

    def set_slot(state, d, value):
        if isinstance(state, EnvState):
            sim = list(state.sim)
            sim[d] = jnp.asarray(value).astype(sim[d].dtype)
            return state._replace(sim=tuple(sim))
        return state._replace(env=set_slot(state.env, d, value))

    def reset(key, dr_scale, throw_scale, ov):
        if variant == "plain":
            return env.reset(key, dr_scale, throw_scale)
        k_dr, k_throw, k_noise = jax.random.split(key, 3)
        dr = override_dr(dr_sample(k_dr, dr_scale), ov)
        state, obs, critic = env.reset_with(k_noise, dr, throw_sample(k_throw, throw_scale))
        if i_gyro is not None:
            state = set_slot(state, i_gyro, ov.gyro)
        return state, obs, critic

    def per_env(sim, target):  # cascade_b0.CAUSES order, then the hover metrics
        pos, vel = sim[plant.i_pos], sim[plant.i_vel]
        q, p = pos[:4], pos[4:]
        cos_tilt = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
        e_p = jnp.linalg.norm(p - target)
        flags = jnp.stack(
            [
                ~plant.finite(sim),
                cos_tilt < np.cos(TERM_TILT),
                jnp.linalg.norm(vel[:3]) > TERM_OMEGA,
                p[2] < TERM_Z,
                e_p > TERM_POS,
            ]
        )
        tilt = jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0))
        axis_tilt = jnp.arccos(jnp.clip(quat_rotate(q, axis)[2], -1.0, 1.0))
        return jnp.argmax(flags).astype(i32), e_p, tilt, axis_tilt

    v_reset = jax.vmap(reset, in_axes=(0, None, None, None))
    v_step = jax.vmap(env.step)
    v_env = jax.vmap(per_env)

    def episodes(params, keys, dr_scale, throw_scale, ov):
        state, obs, critic = v_reset(keys, dr_scale, throw_scale, ov)
        n = keys.shape[0]
        es = inner(state)
        zf, zi, zb = jnp.zeros(n, f64), jnp.zeros(n, i32), jnp.zeros(n, bool)
        acc = {"ret": jnp.zeros(n, f32), "length": zi, "success": zb, "terminated": zb}
        acc.update(cause=jnp.full(n, -1, i32), se_p=zf, se_tilt=zf, se_axis=zf, n_win=zi)
        acc.update(da2=zf, n_da=zi, pwm_d=zf, pwm_d2=zf, n_pwm=zi)
        a0, pwm0 = es.sim[plant.i_action], es.sim[i_pwm]
        carry = (state, obs.astype(f32), critic.astype(f32), ~zb, a0, pwm0, acc)

        def step(carry, _):
            state, obs, critic, alive, a_prev, pwm_prev, acc = carry
            state, obs_n, critic_n, reward, done, info = v_step(state, act(params, obs, critic))
            es = inner(state)
            done, term_now = done.astype(bool), info["terminated"].astype(bool)
            ended = alive & done
            term = ended & term_now
            first, e_p, tilt, axis_tilt = v_env(es.sim, es.target)
            t = info["t"]
            win = alive & ~term_now & (t > hover_from)
            a_now = es.sim[plant.i_action]
            moved = alive & (t >= 2)
            new = {
                "ret": acc["ret"] + jnp.where(alive, ppo._finite(reward), 0.0),
                "length": acc["length"] + alive.astype(i32),
                "success": acc["success"] | (ended & info["success"].astype(bool)),
                "terminated": acc["terminated"] | term,
                "cause": jnp.where(term, first, acc["cause"]),
                "se_p": acc["se_p"] + jnp.where(win, e_p**2, 0.0),
                "se_tilt": acc["se_tilt"] + jnp.where(win, tilt**2, 0.0),
                "se_axis": acc["se_axis"] + jnp.where(win, axis_tilt**2, 0.0),
                "n_win": acc["n_win"] + win.astype(i32),
                "da2": acc["da2"] + jnp.where(moved, jnp.sum((a_now - a_prev) ** 2, -1), 0.0),
                "n_da": acc["n_da"] + moved.astype(i32),
                "pwm_d": acc["pwm_d"],
                "pwm_d2": acc["pwm_d2"],
                "n_pwm": acc["n_pwm"],
            }
            if traced:
                pwm = info["trace"]["motor_pwm"]  # [n, ticks, 4]
                d = jnp.diff(jnp.concatenate([pwm_prev[:, None], pwm], axis=1), axis=1)
                first_tick = jnp.arange(pwm.shape[1]) == 0
                m = alive[:, None] & ~(first_tick[None] & (t == 1)[:, None])  # [n, ticks]
                mf = m[..., None].astype(f64)
                new["pwm_d"] = acc["pwm_d"] + jnp.sum(mf * d, axis=(1, 2))
                new["pwm_d2"] = acc["pwm_d2"] + jnp.sum(mf * d**2, axis=(1, 2))
                new["n_pwm"] = acc["n_pwm"] + jnp.sum(m, axis=1).astype(i32) * pwm.shape[2]
                pwm_prev = pwm[:, -1]
            carry = (state, obs_n.astype(f32), critic_n.astype(f32), alive & ~ended)
            carry = (*carry, a_now, pwm_prev, new)
            out = (obs[:record, : plant.obs_dim], alive[:record]) if record else ()
            return carry, out

        carry, rec = jax.lax.scan(step, carry, None, plant.episode_steps)
        return carry[-1], rec

    out_specs = (P("b"), P(None, "b"))
    sharded = ppo._sharded(episodes, devices, (P(), P("b"), P(), P(), P()), out_specs)
    fn = jax.jit(sharded)
    to_mesh, _ = ppo._movers(devices)
    batch = NamedSharding(ppo.mesh(devices), P("b")) if devices > 1 else None

    def run(params, keys, dr, throw, ov: Override):
        ov = Override(*(jnp.asarray(v, i32 if i < 2 else f64) for i, v in enumerate(ov)))
        keys = jnp.asarray(keys)
        if batch is not None:
            keys = jax.device_put(keys, batch)
        params, ov = to_mesh((params, ov))
        acc, rec = jax.device_get(fn(params, keys, f32(dr), f32(throw), ov))
        return {k: np.asarray(v) for k, v in acc.items()}, rec

    return run


def make_traj(env, act, steps: int | None = None):
    """-> jitted traj(params, keys, dr, throw) -> per-step pos [T, n, 7] and vel [T, n, 6]
    (the episode keeps stepping after it ends), plus the env's success and terminated labels.
    act(params, obs, critic, sim) -> actions [n, 4]."""
    import jax
    import jax.numpy as jnp

    plant = _drone_env(env)
    steps = steps or plant.episode_steps
    v_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    v_step = jax.vmap(env.step)

    def inner(state):
        while hasattr(state, "env"):
            state = state.env
        return state

    @jax.jit
    def traj(params, keys, dr, throw):
        state, obs, critic = v_reset(keys, dr, throw)
        n = keys.shape[0]
        false = jnp.zeros(n, bool)

        def step(carry, _):
            state, obs, critic, alive, success, terminated = carry
            a = act(params, obs.astype(jnp.float32), critic.astype(jnp.float32), inner(state).sim)
            state, obs, critic, _, done, info = v_step(state, a)
            ended = alive & done.astype(bool)
            success = success | (ended & info["success"].astype(bool))
            terminated = terminated | (ended & info["terminated"].astype(bool))
            sim = inner(state).sim
            carry = (state, obs, critic, alive & ~ended, success, terminated)
            return carry, (sim[plant.i_pos], sim[plant.i_vel])

        carry = (state, obs, critic, ~false, false, false)
        carry, (pos, vel) = jax.lax.scan(step, carry, None, steps)
        return pos, vel, carry[4], carry[5]

    return traj


def labels(pos: np.ndarray, vel: np.ndarray, episode_steps: int = 500) -> dict:
    """Success / termination of trajectories [n, T, 7], [n, T, 6] sampled at the policy steps,
    by env.evaluate's tests and env.step's window (the episode ends at its first termination;
    success needs all episode_steps)."""
    import jax
    import jax.numpy as jnp
    from dr import TARGET
    from env import SUCCESS_WINDOW, evaluate

    z, target = jnp.zeros(4), jnp.asarray(TARGET)

    def one(p, v):
        finite = jnp.all(jnp.isfinite(p)) & jnp.all(jnp.isfinite(v))
        m = evaluate(p, v, target, z, z, finite)
        return m.terminated, m.in_window_ok

    term, ok = (np.asarray(x) for x in jax.jit(jax.vmap(jax.vmap(one)))(pos, vel))
    steps = pos.shape[1]
    window = np.arange(1, steps + 1) >= episode_steps - SUCCESS_WINDOW
    terminated = term.any(axis=1)
    end = np.where(terminated, np.argmax(term, axis=1) + 1, steps)
    success = ~terminated & np.all(ok[:, window], axis=1) & (steps == episode_steps)
    return {"success": success, "terminated": terminated, "end": end}


# ------------------------------------------------------------------ suite


class Suite:
    def __init__(self, args):
        self.args = args
        self.sizes = QUICK if args.quick else FULL
        self.devices = args.devices
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        self.path = os.path.join(self.out, "results.json")
        self.results = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.results = json.load(f)
        self.ctrls = discover(
            args.runs, args.baselines, args.baseline_estimator, args.controllers or None
        )
        self.policies = [c for c in self.ctrls if c["kind"] == "policy"]
        self.cache: dict = {}
        self._plants: dict = {}
        self._fns: dict = {}
        self._params: dict = {}
        self.rollout_time = [0.0, 0]  # wall s, env steps of the sections' rollouts
        self.results["protocol"] = {
            "seed": args.seed,
            "heldout_episodes": self.sizes.heldout,
            "slice_episodes": self.sizes.slice,
            "scales": [list(s) for s in SCALES],
            "deterministic": True,
            "episode_steps": 500,
            "policy_step_s": 0.01,
            "quick": bool(args.quick),
            "devices": self.devices,
            "runs": [os.path.basename(os.path.normpath(r)) for r in args.runs],
            "baseline_estimator": args.baseline_estimator,
        }
        keep = ("id", "label", "kind", "estimator", "group", "path", "residual")
        self.results["controllers"] = [
            {k: _rel(c[k]) if k == "path" and c.get(k) else c.get(k) for k in keep}
            for c in self.ctrls
        ]

    def save(self):
        with open(self.path + ".tmp", "w") as f:
            json.dump(self.results, f, indent=1)
        os.replace(self.path + ".tmp", self.path)

    def keys(self, n: int) -> np.ndarray:
        import jax

        if "keys" not in self.cache:
            k_reset, _ = jax.random.split(jax.random.PRNGKey(self.args.seed))
            self.cache["keys"] = np.asarray(jax.random.split(k_reset, FULL.heldout))
        return self.cache["keys"][:n]

    def plant(self, estimator: str, variant: str):
        key = (estimator, variant)
        if key not in self._plants:
            from env import DroneEnv
            from policy_world import EFFECTS

            if variant == "plain":
                env = DroneEnv(estimator)
            else:
                env = DelayEnv(DroneEnv(estimator, effects=EFFECTS, trace=("motor_pwm",)))
            self._plants[key] = env
        return self._plants[key]

    def env(self, c: dict, estimator: str, variant: str):
        env = self.plant(estimator, variant)
        if c.get("residual"):
            from residual import ResidualEnv

            env = ResidualEnv(env, c["residual"]["base"], c["residual"]["scale"])
        return env

    def params(self, c: dict):
        if c["id"] not in self._params:
            import jax.numpy as jnp

            if c["kind"] == "policy":
                from eval_runs import load_policy

                actor, stat, clip, _ = load_policy(os.path.join(c["path"], "policy.npz"))
                assert clip == c["obs_clip"], c["id"]
                self._params[c["id"]] = (actor, stat)
            else:
                import cascade_b0

                gains = cascade_b0.GAIN_SETS[c["gain_set"]]
                self._params[c["id"]] = cascade_b0.B0Gains(
                    *[jnp.asarray(v, jnp.float64) for v in gains]
                )
        return self._params[c["id"]]

    def fn(self, c: dict, estimator: str, variant: str):
        res = c.get("residual")
        key = (estimator, variant, c["kind"], res and (res["base"], res["scale"]))
        key += (c.get("obs_clip", 5.0),)
        if key not in self._fns:
            record = -(-32 // self.devices) if variant == "eval" else 0
            env = self.env(c, estimator, variant)
            self._fns[key] = make_episodes(
                env, c["kind"], c.get("obs_clip", 5.0), variant, self.devices, record
            )
        return self._fns[key]

    def run(self, c, variant, keys, dr, throw, ov=None, estimator=None):
        """Per-episode outcomes over keys in batches; the first batch's recording."""
        estimator = estimator or c["estimator"]
        fn = self.fn(c, estimator, variant)
        ov = ov or Override()
        # beyond these the DelayEnv one-hot or the long VIO buffer would silently read zeros
        assert ov.act_delay <= MAX_ACT_DELAY and ov.vio_delay < VIO_LONG_LEN, ov
        b = self.sizes.batch
        assert len(keys) % b == 0, (len(keys), b)
        parts, rec = [], None
        start = time.perf_counter()
        for lo in range(0, len(keys), b):
            ep, r = fn(self.params(c), keys[lo : lo + b], dr, throw, ov)
            parts.append(ep)
            rec = r if rec is None else rec
        self.rollout_time[0] += time.perf_counter() - start
        self.rollout_time[1] += len(keys) * _drone_env(self.plant(estimator, variant)).episode_steps
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, rec


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def e1(s: Suite):
    keys = s.keys(s.sizes.heldout)
    per = {}
    for c in s.ctrls:
        per[c["id"]] = [s.run(c, "plain", keys, dr, th)[0] for dr, th in SCALES]
        _log(f"e1 {c['id']}: " + " ".join(f"{np.mean(e['success']):.4f}" for e in per[c["id"]]))
    rows, checks = {}, {}
    for c in s.ctrls:
        evals = []
        for i, (dr, th) in enumerate(SCALES):
            ev = {"dr": dr, "throw": th, **stats(per[c["id"]][i])}
            if REFERENCE in per and c["id"] != REFERENCE:
                ev[f"vs_{REFERENCE}"] = paired(
                    per[c["id"]][i]["success"], per[REFERENCE][i]["success"]
                )
            evals.append(ev)
        rows[c["id"]] = evals
        if c["kind"] == "policy":
            checks[c["id"]] = final_eval_check(c, evals, s.args.seed)
    s.cache["e1"] = per
    fields = ("success", "terminated", "cause", "length", "ret")
    arrays = {
        f"{cid}|{dr:g},{th:g}|{k}": ep[k]
        for cid, eps in per.items()
        for (dr, th), ep in zip(SCALES, eps)
        for k in fields
    }
    np.savez_compressed(os.path.join(s.out, "heldout_episodes.npz"), **arrays)
    s.results["e1"] = {
        "protocol": {
            "episodes": len(keys),
            "seed": s.args.seed,
            "scales": [list(x) for x in SCALES],
            "estimator": "training estimator (baselines: --baseline-estimator)",
            "reference": REFERENCE,
            "keys": "ppo.make_evaluate reset keys: split(split(PRNGKey(seed))[0], 4096)",
            "per_episode": "heldout_episodes.npz, keys '<controller>|<dr>,<throw>|<field>'",
        },
        "rows": rows,
        "final_eval_check": checks,
    }


def final_eval_check(c, evals, seed) -> dict | None:
    """Counts of train.py's final_eval.json against the re-evaluation (same episodes only)."""
    path = os.path.join(c["path"], "final_eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        fe = json.load(f)
    n = evals[0]["episodes"]
    if fe["episodes"] != n or fe["seed"] != seed:
        return {"comparable": False}
    out = {"comparable": True, "success_count_diff": [], "terminated_count_diff": []}
    out["mean_return_abs_diff"] = []
    for ev in evals:
        w = next((x for x in fe["evals"] if (x["dr"], x["throw"]) == (ev["dr"], ev["throw"])), None)
        if w is None:
            continue
        out["success_count_diff"].append(ev["successes"] - round(w["success_rate"] * n))
        term = round(ev["terminated_rate"] * n) - round(w["terminated_rate"] * n)
        out["terminated_count_diff"].append(term)
        out["mean_return_abs_diff"].append(_6g(abs(ev["mean_return"] - w["mean_return"])))
    return out


def e3(s: Suite):
    keys = s.keys(s.sizes.slice)
    rows = {}
    for c in s.ctrls:
        ep, _ = s.run(c, "plain", keys, 0.0, 0.0)
        rows[c["id"]] = {**stats(ep), **hover_stats(ep)}
        _log(f"e3 {c['id']}: {rows[c['id']]['pos_rms_m']} m {rows[c['id']]['tilt_rms_deg']} deg")
    s.results["e3"] = {
        "protocol": {
            "episodes": len(keys),
            "seed": s.args.seed,
            "dr": 0.0,
            "throw": 0.0,
            "window": "last 2 s (200 policy steps) of the episodes that did not terminate",
        },
        "rows": rows,
    }


def in_distribution(s: Suite, c: dict) -> dict:
    """Eval-env episodes at (1,1) with nothing overridden (cached), and the recording."""
    cache = s.cache.setdefault("e9", {})
    if c["id"] not in cache:
        cache[c["id"]] = s.run(c, "eval", s.keys(s.sizes.slice), 1.0, 1.0)
    return cache[c["id"]]


def e9(s: Suite):
    rows = {}
    for c in s.ctrls:
        ep, _ = in_distribution(s, c)
        rows[c["id"]] = {**stats(ep), **smooth_stats(ep)}
        r = rows[c["id"]]
        _log(f"e9 {c['id']}: pwm diff std {r['pwm_diff_std']} da2 {r['mean_sq_action_diff']}")
    s.results["e9"] = {
        "protocol": {
            "episodes": s.sizes.slice,
            "seed": s.args.seed,
            "dr": 1.0,
            "throw": 1.0,
            "pwm": "per-tick (300 Hz) first difference over alive ticks, all motors pooled; "
            "the first tick of each episode excluded",
            "action": "executed policy_action (100 Hz), squared norm of the change, steps t >= 2",
            "env": "eval env (long VIO buffer, gyroscopic gain 0, motor_pwm traced): bitwise "
            "the training env",
        },
        "rows": rows,
    }


def _values(s: Suite, values):
    if s.sizes.e2_values and len(values) > s.sizes.e2_values:
        return (values[0], values[-1])
    return tuple(values)


def e2(s: Suite):
    keys = s.keys(s.sizes.slice)
    slices = {}
    for name, spec in E2_SLICES.items():
        values = _values(s, spec["values"])
        rows = {}
        for c in s.ctrls:
            base, _ = in_distribution(s, c)
            rows[c["id"]] = []
            for v in values:
                ep, _ = s.run(c, "eval", keys, 1.0, 1.0, Override(**{name: v}))
                row = {
                    "value": v,
                    **stats(ep),
                    "vs_in_distribution": paired(ep["success"], base["success"]),
                }
                rows[c["id"]].append(row)
            _log(
                f"e2 {name} {c['id']}: "
                + " ".join(f"{r['success_rate']:.3f}" for r in rows[c["id"]])
            )
        slices[name] = {
            "values": list(values),
            "unit": spec["unit"],
            "train_range": list(spec["train"]),
            "rows": rows,
        }
    s.results["e2"] = {
        "protocol": {
            "episodes_per_point": len(keys),
            "seed": s.args.seed,
            "dr": 1.0,
            "throw": 1.0,
            "override": "one DR field set to the value in every episode, the rest the full-DR "
            "draw; thrust spread rescales the drawn per-motor deviation to the half-width",
            "act_delay": "ticks of extra action latency; 3k + r runs as k whole policy steps "
            "(DelayEnv) plus r ticks in the graph FIFO",
            "vio_delay": "VIO read through a 16-tick buffer",
        },
        "slices": slices,
    }


def e4(s: Suite):
    keys = s.keys(s.sizes.slice)
    rows = {}
    for c in s.ctrls:
        base, _ = in_distribution(s, c)
        rows[c["id"]] = {"in_distribution": stats(base)}
        for name, ov in E4_EFFECTS.items():
            ep, _ = s.run(c, "eval", keys, 1.0, 1.0, Override(**ov))
            st = stats(ep)
            st["drop"] = _6g(rows[c["id"]]["in_distribution"]["success_rate"] - st["success_rate"])
            st["vs_in_distribution"] = paired(ep["success"], base["success"])
            rows[c["id"]][name] = st
        _log(
            f"e4 {c['id']}: "
            + " ".join(f"{k} {v['success_rate']:.3f}" for k, v in rows[c["id"]].items())
        )
    s.results["e4"] = {
        "protocol": {
            "episodes": len(keys),
            "seed": s.args.seed,
            "dr": 1.0,
            "throw": 1.0,
            "effects": E4_EFFECTS,
            "gyroscopic": "plant_dr.gyroscopic: -w x (I w) body torque, absent from six_dof",
            "wind": "horizontal speed set in the drawn direction, the drawn vertical kept",
            "drop": "in-distribution success minus effect success on the same episodes",
        },
        "rows": rows,
    }


def track_a_rows(paths, episodes: int, seed: int) -> dict:
    """{(checkpoint, eval estimator): final_eval} from eval_runs.py summary.json files (its
    --estimator / --cross rows: eval_estimator differs from the training estimator), kept only
    when the summary used this protocol (episodes, seed and SCALES)."""
    out = {}
    for path in paths or ():
        with open(path) as f:
            summary = json.load(f)
        pairs = [tuple(float(x) for x in p) for p in summary.get("pairs", [])]
        if summary.get("episodes") != episodes or summary.get("seed") != seed:
            continue
        if pairs != list(SCALES):
            continue
        for row in summary.get("runs", []):
            ev, train = row.get("eval_estimator"), row.get("estimator")
            if ev and ev != train and "final_eval" in row:
                out[(row.get("checkpoint", row["run"]), ev)] = row["final_eval"]
    return out


def e5(s: Suite):
    keys = s.keys(s.sizes.heldout)
    reused = track_a_rows(s.args.track_a, len(keys), s.args.seed)
    rows = []
    e1_rows = s.results.get("e1", {}).get("rows", {})
    for c in s.policies:
        own = s.cache.get("e1", {}).get(c["id"])
        for est in ESTIMATORS:
            if est == c["estimator"]:
                continue
            row = {"run": c["id"], "train_estimator": c["estimator"], "eval_estimator": est}
            if (c["id"], est) in reused:
                evals = reused[(c["id"], est)]
                for i, ev in enumerate(evals):
                    if c["id"] in e1_rows:
                        ev["training_estimator_success"] = e1_rows[c["id"]][i]["success_rate"]
                row.update(source="track-a summary.json", final_eval=evals)
                rows.append(row)
                _log(f"e5 {c['id']} on {est}: reused from --track-a")
                continue
            evals = []
            for i, (dr, th) in enumerate(SCALES):
                ep, _ = s.run(c, "plain", keys, dr, th, estimator=est)
                ev = {"dr": dr, "throw": th, **stats(ep)}
                if own is not None:
                    ev["vs_training_estimator"] = paired(ep["success"], own[i]["success"])
                elif c["id"] in e1_rows:
                    ev["training_estimator_success"] = e1_rows[c["id"]][i]["success_rate"]
                evals.append(ev)
            row.update(source="suite", final_eval=evals)
            rows.append(row)
            _log(f"e5 {c['id']} on {est}: " + " ".join(f"{e['success_rate']:.4f}" for e in evals))
    s.results["e5"] = {
        "protocol": {
            "episodes": len(keys),
            "seed": s.args.seed,
            "scales": [list(x) for x in SCALES],
            "paired": "vs the same policy on its training estimator (e1), same episodes",
        },
        "rows": rows,
    }


def e8(s: Suite):
    import jax
    import jax.numpy as jnp
    import nn
    import residual

    rows = {}
    load_start = _load_avg()
    for c in s.policies:
        policy = dict(np.load(os.path.join(c["path"], "policy.npz")))
        _, (obs, alive) = in_distribution(s, c)
        obs = np.asarray(obs, np.float64).transpose(1, 0, 2)[np.asarray(alive).T]
        obs = obs[: s.sizes.e8_obs]
        exp = NumpyPolicy(policy)
        forward = residual.residual_action if exp.residual else nn.policy_action
        ref = jax.jit(functools.partial(forward, policy, xp=jnp))
        got = exp(obs)
        want = np.asarray(ref(jnp.asarray(obs)))
        got32 = NumpyPolicy(policy, np.float32)(obs.astype(np.float32))
        want32 = np.asarray(ref(jnp.asarray(obs, jnp.float32)))
        latency = []
        for o in obs[: s.sizes.e8_calls] if len(obs) >= s.sizes.e8_calls else _cycle(obs, s):
            t0 = time.perf_counter_ns()
            exp(o)
            latency.append(time.perf_counter_ns() - t0)
        ref(jnp.asarray(obs[0])).block_until_ready()
        jlat = []
        for o in obs[: min(len(obs), 2000)]:
            x = jnp.asarray(o)
            t0 = time.perf_counter_ns()
            ref(x).block_until_ready()
            jlat.append(time.perf_counter_ns() - t0)
        rows[c["id"]] = {
            "observations": len(obs),
            "max_abs_diff_f64": float(np.max(np.abs(got - want))),
            "max_abs_diff_f32": float(np.max(np.abs(got32 - want32))),
            "max_abs_diff_f64_vs_f32_path": float(np.max(np.abs(got - want32))),
            "numpy_latency_median_us": _6g(np.median(latency) / 1e3),
            "numpy_latency_p99_us": _6g(np.percentile(latency, 99) / 1e3),
            "calls": len(latency),
            "jnp_jit_latency_median_us": _6g(np.median(jlat) / 1e3),
            "residual": exp.residual,
        }
        _log(f"e8 {c['id']}: {rows[c['id']]}")
    s.results["e8"] = {
        "protocol": {
            "observations": "obs the policy acted on in e9's eval-env episodes at (1,1), "
            "alive steps of the recorded episodes, first N",
            "reference": "nn.policy_action / residual.residual_action under jax.jit, float64",
            "export": "evaluate.NumpyPolicy (numpy only)",
            "latency": "single-sample perf_counter_ns per call on this machine, "
            + _conditions(s, load_start),
        },
        "rows": rows,
    }


def _conditions(s: Suite, load_start: float | None) -> str:
    """Run conditions for timing protocols: --note, then the 1-min load average."""
    load = f"1-min load average {load_start} at the start, {_load_avg()} at the end"
    return f"{s.args.note}; {load}" if s.args.note else load


def _cycle(obs, s):
    reps = -(-s.sizes.e8_calls // len(obs))
    return np.concatenate([obs] * reps)[: s.sizes.e8_calls]


def make_deploy_act(env, kind: str, is_residual: bool):
    """act for make_traj: the in-graph float64 forward of policy_system on the raw sim obs, with
    the policy arrays (or gains) as runtime parameters."""
    import cascade_b0
    import jax
    import jax.numpy as jnp
    import nn
    import residual

    def act(params, obs, critic, sim):
        o, c = sim[env.i_obs], sim[env.i_critic]
        if kind == "b0":
            return jax.vmap(lambda x: cascade_b0.b0_action(x, params))(o)
        if kind == "b1":
            return jax.vmap(lambda x, y: cascade_b0.b1_action(x, y, params))(o, c)
        if is_residual:
            return residual.residual_action(params, o, jnp)
        return nn.policy_action(params, o, jnp)

    return act


def _numeric(policy: ty.Mapping) -> dict:
    import jax.numpy as jnp

    return {
        k: jnp.asarray(v)
        for k, v in policy.items()
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiu"
    }


def e6(s: Suite):
    import policy_system as PS

    sz = s.sizes
    keys = s.keys(max(sz.e6_episodes, sz.e6_drones))
    chosen = s.args.e6 or _best_policy(s)
    out = {"protocol": {}, "a": {}, "chosen": chosen}
    trajs = {}
    for c in [x for x in s.ctrls if x["id"] == chosen] if s.args.quick else s.ctrls:
        policy = PS.load(c["path"]) if c["kind"] == "policy" else PS.load(c["id"])
        est = c["estimator"]
        n = sz.e6_episodes if c["id"] == chosen else sz.e6_drones
        train = _traj(s, c, est, keys[:n], "train")
        deploy = _traj(s, c, est, keys[:n], "deploy", policy)
        trajs[c["id"]] = (train, deploy, policy)
        ticks = sz.e6_parity_ticks
        steps = ticks // 3
        pos, _, timing = cranelift_episode(policy, est, keys[0], ticks)
        cl = pos[3::3][:steps, 4:]
        out["a"][c["id"]] = {
            "ticks": ticks,
            "max_dpos_vs_jax_env_m": float(np.max(np.abs(cl - train["pos"][0, :steps, 4:]))),
            "max_dpos_vs_jax_f64_forward_m": float(
                np.max(np.abs(cl - deploy["pos"][0, :steps, 4:]))
            ),
            **timing,
        }
        _log(f"e6a {c['id']}: {out['a'][c['id']]}")
    if chosen in trajs:
        train, deploy, policy = trajs[chosen]
        c = next(x for x in s.ctrls if x["id"] == chosen)
        out["b"] = e6_b(s, c, policy, keys, train, deploy)
        out["c"] = e6_c(s, c, policy, keys, train, deploy)
        out["bench"] = main_bench(s, c)
    out["protocol"] = {
        "seed": s.args.seed,
        "dr": 1.0,
        "throw": 1.0,
        "a": f"held-out episode 0, {sz.e6_parity_ticks} ticks, 1 drone on Cranelift vs the JAX env "
        "(env.step with the training-path float32 forward); also vs the env stepped with the "
        "in-graph float64 forward",
        "b": f"held-out episodes 0..{sz.e6_episodes - 1}, {sz.e6_ticks} ticks each, one 1-drone "
        "Cranelift world per episode vs the JAX env; labels from the trajectories by the env's "
        "tests; max |dpos| over the steps until the JAX episode ends",
        "c": f"held-out episodes 0..{sz.e6_drones - 1} as {sz.e6_drones} drones in one jax-cpu "
        f"world, {sz.e6_ticks} ticks, vs {sz.e6_drones} vmapped env episodes",
    }
    s.results["e6"] = out


def _best_policy(s: Suite) -> str | None:
    """The headline rule: the best seed (e1 success at (1,1)) of the config with the best mean
    (1,1) success among configs with at least 2 seeds; the best run when no config has 2."""
    rows = s.results.get("e1", {}).get("rows", {})
    rate = {c["id"]: rows[c["id"]][0]["success_rate"] for c in s.policies if c["id"] in rows}
    if not rate:
        return s.policies[0]["id"] if s.policies else REFERENCE
    groups: dict[str, list[str]] = {}
    for c in s.policies:
        if c["id"] in rate:
            groups.setdefault(c["group"], []).append(c["id"])
    multi = {g: ids for g, ids in groups.items() if len(ids) >= 2}
    cands = list(rate)
    if multi:
        cands = multi[max(multi, key=lambda g: (np.mean([rate[i] for i in multi[g]]), g))]
    return max(cands, key=lambda i: (rate[i], i))


def _traj(s: Suite, c, est, keys, mode, policy=None) -> dict:
    """Per-step JAX env trajectories [n, T, ...] of the controller's episodes: "train" flies the
    e1 action path, "deploy" the in-graph float64 forward."""
    import jax.numpy as jnp

    res = c.get("residual")
    key = (est, mode, c["kind"], res and (res["base"], res["scale"]), c.get("obs_clip", 5.0))
    cache = s.cache.setdefault("traj", {})
    if mode == "train":
        env = s.env(c, est, "plain")
        params = s.params(c)
        if key not in cache:
            act0 = make_act(c["kind"], c.get("obs_clip", 5.0))
            cache[key] = make_traj(env, lambda p, o, cr, sim: act0(p, o, cr), s.sizes.e6_ticks // 3)
    else:
        env = s.plant(est, "plain")
        params = s.params(c) if c["kind"] != "policy" else _numeric(policy)
        if key not in cache:
            act = make_deploy_act(env, c["kind"], bool(res))
            cache[key] = make_traj(env, act, s.sizes.e6_ticks // 3)
    pos, vel, success, terminated = cache[key](params, jnp.asarray(keys), 1.0, 1.0)
    return {
        "pos": np.asarray(pos).transpose(1, 0, 2),
        "vel": np.asarray(vel).transpose(1, 0, 2),
        "success": np.asarray(success),
        "terminated": np.asarray(terminated),
    }


def cranelift_episode(policy, estimator, key, ticks, backend="cranelift"):
    import policy_system as PS

    t0 = time.perf_counter()
    w, system = PS.build_world(policy, 1, estimator, keys=[key])
    ex = w.build(system, simulation_rate=300.0, backend=backend)
    t1 = time.perf_counter()
    ex.run(ticks, show_progress=False)
    t2 = time.perf_counter()
    df = ex.history(["drone.world_pos", "drone.world_vel"])
    pos = np.stack(df["drone.world_pos"].to_numpy())
    vel = np.stack(df["drone.world_vel"].to_numpy())
    assert len(pos) == ticks + 1, (len(pos), ticks)
    return pos, vel, {"build_s": round(t1 - t0, 3), "run_s": round(t2 - t1, 4)}


def _deviation(a: np.ndarray, b: np.ndarray, end: np.ndarray) -> dict:
    """max |dpos| per episode over steps 1..end and over the whole trajectory."""
    d = np.linalg.norm(a - b, axis=-1)  # [n, T]
    upto = np.array([d[i, : end[i]].max() for i in range(len(d))])
    whole = d.max(axis=1)

    def q(x):
        return {
            "median": _6g(np.median(x)),
            "p90": _6g(np.percentile(x, 90)),
            "max": _6g(np.max(x)),
        }

    return {"until_end_m": q(upto), "whole_m": q(whole)}


def _agreement(x: dict, y: dict) -> dict:
    n = len(x["success"])
    same = np.asarray(x["success"]) == np.asarray(y["success"])
    same_t = np.asarray(x["terminated"]) == np.asarray(y["terminated"])
    return {
        "episodes": n,
        "success_agree": _6g(same.mean()),
        "terminated_agree": _6g(same_t.mean()),
        "success_a": int(np.sum(x["success"])),
        "success_b": int(np.sum(y["success"])),
    }


def e6_b(s, c, policy, keys, train, deploy) -> dict:
    sz = s.sizes
    n, steps = sz.e6_episodes, sz.e6_ticks // 3
    pos, vel, builds, runs = [], [], [], []
    for i in range(n):
        p, v, timing = cranelift_episode(policy, c["estimator"], keys[i], sz.e6_ticks)
        pos.append(p[3::3][:steps])
        vel.append(v[3::3][:steps])
        builds.append(timing["build_s"])
        runs.append(timing["run_s"])
        if (i + 1) % 32 == 0:
            _log(f"e6b {i + 1}/{n}")
    pos, vel = np.stack(pos), np.stack(vel)
    cl = labels(pos, vel)
    jx = labels(train["pos"][:n], train["vel"][:n])
    env_labels = {"success": train["success"][:n], "terminated": train["terminated"][:n]}
    out = {
        "episodes": n,
        "ticks": sz.e6_ticks,
        "cranelift_vs_jax": _agreement(cl, jx),
        "jax_labels_vs_env_info": _agreement(jx, env_labels),
        "deviation_vs_jax_env": _deviation(pos[..., 4:], train["pos"][:n, :, 4:], jx["end"]),
        "deviation_vs_jax_f64_forward": _deviation(
            pos[..., 4:], deploy["pos"][:n, :, 4:], jx["end"]
        ),
        "cranelift_success_rate": _6g(cl["success"].mean()),
        "jax_success_rate": _6g(jx["success"].mean()),
        "build_s_median": _6g(np.median(builds)),
        "run_s_median": _6g(np.median(runs)),
        "cranelift_ticks_per_s": _6g(sz.e6_ticks / np.median(runs)),
    }
    _log(f"e6b: {out}")
    return out


def e6_c(s, c, policy, keys, train, deploy) -> dict:
    import policy_system as PS

    sz = s.sizes
    n, steps = sz.e6_drones, sz.e6_ticks // 3
    t0 = time.perf_counter()
    w, system = PS.build_world(policy, n, c["estimator"], keys=keys[:n])
    ex = w.build(system, simulation_rate=300.0, backend="jax-cpu")
    ex.run(1, show_progress=False)  # compiles
    t1 = time.perf_counter()
    ex.run(sz.e6_ticks - 1, show_progress=False)
    t2 = time.perf_counter()
    names = PS.entity_names(n)
    df = ex.history([f"{e}.{k}" for e in names for k in ("world_pos", "world_vel")])
    pos = np.stack([np.stack(df[f"{e}.world_pos"].to_numpy())[3::3][:steps] for e in names])
    vel = np.stack([np.stack(df[f"{e}.world_vel"].to_numpy())[3::3][:steps] for e in names])
    multi = labels(pos, vel)
    jx = labels(train["pos"][:n], train["vel"][:n])
    jd = labels(deploy["pos"][:n], deploy["vel"][:n])
    out = {
        "drones": n,
        "ticks": sz.e6_ticks,
        "backend": "jax-cpu",
        "world_vs_vmapped_env": _agreement(multi, jx),
        "world_vs_vmapped_env_f64_forward": _agreement(multi, jd),
        "deviation_vs_jax_env": _deviation(pos[..., 4:], train["pos"][:n, :, 4:], jx["end"]),
        "deviation_vs_jax_f64_forward": _deviation(
            pos[..., 4:], deploy["pos"][:n, :, 4:], jd["end"]
        ),
        "build_and_compile_s": round(t1 - t0, 3),
        "run_s": round(t2 - t1, 3),
        "drone_ticks_per_s": _6g(n * (sz.e6_ticks - 1) / (t2 - t1)),
    }
    _log(f"e6c: {out}")
    return out


def main_bench(s: Suite, c) -> dict:
    """python main.py bench --ticks N with the controller in graph (its own process)."""
    sz = s.sizes
    env = {**os.environ, "DRONE_POLICY": c.get("path") or c["id"], "DRONE_N": str(sz.e6_drones)}
    env.pop("XLA_FLAGS", None)
    cmd = [sys.executable, os.path.join(HERE, "main.py"), "bench", "--ticks", str(sz.bench_ticks)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True, cwd=HERE)
    wall = time.perf_counter() - t0
    text = res.stdout + res.stderr

    def num(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    out = {
        "command": " ".join(["DRONE_N=" + str(sz.e6_drones), "python main.py", *cmd[2:]]),
        "exit": res.returncode,
        "drones": sz.e6_drones,
        "ticks": sz.bench_ticks,
        "tick_ms": num(r"tick time:\s+([\d.]+) ms"),
        "build_ms": num(r"build time:\s+([\d.]+) ms"),
        "real_time_factor": num(r"real_time_factor:\s+([\d.]+)"),
        "process_wall_s": round(wall, 2),
    }
    if res.returncode:
        out["tail"] = text[-2000:]
    _log(f"bench: {out}")
    return out


def e10(s: Suite):
    import bench
    from env import STEP_DT

    sz = s.sizes
    load_start = _load_avg()
    r = bench.worker("mekf6", sz.e10_envs, sz.e10_steps, sz.e10_repeats, s.devices)
    sps = r["env_steps_per_s"]

    def hours_per_min(x):
        return _6g(x * STEP_DT / 3600.0 * 60.0) if x else None

    train = {}
    for c in s.policies:
        try:
            with open(os.path.join(c["path"], "log.jsonl")) as f:
                log = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            continue
        v = sorted(x["steps_per_s"] for x in log[1:])
        if v:
            med = v[len(v) // 2]
            train[c["id"]] = {"median_steps_per_s": med, "flight_hours_per_min": hours_per_min(med)}
    wall, steps = s.rollout_time
    suite = steps / wall if wall else None
    s.results["e10"] = {
        "protocol": {
            "bench": f"bench.worker: DroneEnv mekf6, {sz.e10_envs} envs x {sz.e10_steps} steps "
            f"scan, {s.devices} XLA host devices, random actions, median of {sz.e10_repeats}",
            "training_loop": "median per-iteration steps_per_s of train.py (rollout + PPO "
            "update) from each run's log.jsonl",
            "suite": "this run's own deterministic eval rollouts, compile included",
            "note": _conditions(s, load_start)
            + "; training-loop figures are from the training runs and their conditions",
        },
        "rollout": {**r, "flight_hours_per_min": hours_per_min(sps)},
        "training_loop": train,
        "suite_rollouts": {
            "env_steps_per_s": _6g(suite) if suite else None,
            "flight_hours_per_min": hours_per_min(suite),
        },
    }
    _log(f"e10: {s.results['e10']['rollout']}")


# ------------------------------------------------------------------ plots and summary


def _colors(ctrls) -> dict:
    groups = [c["group"] for c in ctrls if c["kind"] == "policy"]
    groups = list(dict.fromkeys(groups))
    out = {}
    for c in ctrls:
        if c["kind"] != "policy":
            out[c["id"]] = GREYS.get(c["id"], "#52514e")
        else:
            out[c["id"]] = PALETTE[groups.index(c["group"]) % len(PALETTE)]
    return out


def _seed_style(cid: str):
    m = re.search(r"_s(\d+)$", cid)
    i = int(m.group(1)) if m else 0
    return ("-", "--", ":", "-.")[i % 4], ("o", "s", "^", "D")[i % 4]


def plots(s: Suite):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#9a9893",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    ctrls = s.ctrls
    colors = _colors(ctrls)
    made = []
    r = s.results
    if "e1" in r:
        rows = r["e1"]["rows"]
        ids = [c["id"] for c in ctrls if c["id"] in rows]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(ids) + 2), 4.2))
        width = 0.26
        for j, (dr, th) in enumerate(SCALES):
            x = np.arange(len(ids)) + (j - 1) * width
            y = [rows[i][j]["success_rate"] for i in ids]
            lo = [y[k] - rows[i][j]["wilson95"][0] for k, i in enumerate(ids)]
            hi = [rows[i][j]["wilson95"][1] - y[k] for k, i in enumerate(ids)]
            ax.bar(x, y, width * 0.92, color=PALETTE[j], label=f"(dr, throw) = ({dr:g}, {th:g})")
            ax.errorbar(
                x, y, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", elinewidth=0.8, capsize=2
            )
        ax.set_xticks(
            np.arange(len(ids)), [BASELINES.get(i, i) for i in ids], rotation=30, ha="right"
        )
        ax.set_ylabel("held-out success rate")
        ax.set_ylim(0, 1.0)
        n = r["e1"]["protocol"]["episodes"]
        ax.set_title(
            f"Held-out success, {n} episodes per scale (Wilson 95%, episode noise only)", pad=24
        )
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(SCALES),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout()
        made.append(_save(fig, s, "heldout_bars.png"))
    if "e2" in r:
        sl = r["e2"]["slices"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        for ax, (name, spec) in zip(axes.flat, sl.items()):
            lo, hi = spec["train_range"]
            ax.axvspan(lo, hi, color="#e4e3df", alpha=0.7, lw=0, label="training range")
            for c in ctrls:
                rows = spec["rows"].get(c["id"])
                if not rows:
                    continue
                ls, mk = _seed_style(c["id"])
                ax.plot(
                    [x["value"] for x in rows],
                    [x["success_rate"] for x in rows],
                    ls=ls if c["kind"] == "policy" else "-",
                    marker=mk,
                    ms=4,
                    lw=1.5,
                    color=colors[c["id"]],
                    label=BASELINES.get(c["id"], c["id"]),
                )
            ax.set_xlabel(f"{name} ({spec['unit']})")
            ax.set_ylabel("success rate at full DR")
            ax.set_ylim(0, 1.0)
        handles, labels_ = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_,
            loc="lower center",
            ncol=min(len(labels_), 6),
            fontsize=8,
            frameon=False,
        )
        fig.suptitle(
            f"E2 out-of-distribution slices ({r['e2']['protocol']['episodes_per_point']} episodes per point)"
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        made.append(_save(fig, s, "e2_ood.png"))
    if "e4" in r:
        rows = r["e4"]["rows"]
        ids = [c["id"] for c in ctrls if c["id"] in rows]
        effects = ["in_distribution", *E4_EFFECTS]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        width = 0.8 / max(len(ids), 1)
        for j, cid in enumerate(ids):
            x = np.arange(len(effects)) + (j - (len(ids) - 1) / 2) * width
            ax.bar(
                x,
                [rows[cid][e]["success_rate"] for e in effects],
                width * 0.9,
                color=colors[cid],
                label=BASELINES.get(cid, cid),
                hatch="//" if cid.endswith(("_s1", "_s3")) else None,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(np.arange(len(effects)), [e.replace("_", " ") for e in effects])
        ax.set_ylabel("success rate at full DR")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"E4 unseen effects ({r['e4']['protocol']['episodes']} episodes each)")
        ax.legend(
            fontsize=7,
            ncol=min(len(ids), 6),
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
        )
        fig.tight_layout()
        made.append(_save(fig, s, "e4_unseen.png"))
    groups = list(dict.fromkeys(c["group"] for c in s.policies))
    if groups:
        fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.8), squeeze=False)
        for ax, g in zip(axes[0], groups):
            # in-training evals run at the config's own DR scale: dr 0 for the no-DR ablation
            no_dr = _trained_without_dr(next(c for c in s.policies if c["group"] == g))
            for c in [c for c in s.policies if c["group"] == g]:
                try:
                    with open(os.path.join(c["path"], "log.jsonl")) as f:
                        log = [json.loads(line) for line in f if line.strip()]
                except FileNotFoundError:
                    continue
                pts = [
                    (x["iter"] + 1, x["eval_success_rate"]) for x in log if "eval_success_rate" in x
                ]
                if not pts:
                    continue
                ls, mk = _seed_style(c["id"])
                ax.plot(*zip(*pts), ls=ls, lw=1.5, color=colors[c["id"]], label=c["id"])
            if "e1" in r and REFERENCE in r["e1"]["rows"]:
                j = 2 if no_dr else 0  # SCALES (0,1) or (1,1)
                ref = r["e1"]["rows"][REFERENCE][j]["success_rate"]
                scale = "({:g},{:g})".format(*SCALES[j])
                ax.axhline(
                    ref, color=GREYS[REFERENCE], lw=1, ls="--", label=f"B0 DR held-out {scale}"
                )
            ax.set_title(f"{g} (in-training eval at dr 0)" if no_dr else g)
            ax.set_xlabel("iteration")
            ax.set_ylabel("in-training eval success (1024 episodes)")
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=7, frameon=False, loc="center left" if no_dr else "upper left")
        fig.tight_layout()
        made.append(_save(fig, s, "learning_curves.png"))
    plt.close("all")
    s.results["plots"] = made


def _trained_without_dr(c: dict) -> bool:
    try:
        with open(os.path.join(c["path"], "config.json")) as f:
            config = json.load(f)
    except (FileNotFoundError, KeyError):
        return False
    return bool(config.get("nominal")) or config.get("dr") == "0"


def _save(fig, s, name) -> str:
    path = os.path.join(s.out, name)
    fig.savefig(path, dpi=130)
    return name


def _config_table(r: dict) -> list[str]:
    """E1 per config with 2 or more seeds: the seeds' success, mean +- SD over seeds, the t 95%
    interval of the mean, and the paired diff vs B0 DR as mean [t 95%] over seeds."""
    rows = r["e1"]["rows"]
    groups: dict[str, list[str]] = {}
    for c in r.get("controllers", []):
        if c["kind"] == "policy" and c["id"] in rows:
            groups.setdefault(c["group"], []).append(c["id"])
    groups = {g: ids for g, ids in groups.items() if len(ids) >= 2}
    if not groups:
        return []
    lines = [
        "",
        "### E1 per config (training seeds)",
        "",
        (
            "+- is the SD over seeds, not an interval; the t 95% intervals are over seeds and "
            "are the uncertainty of the config mean."
        ),
        "",
        "| config | (dr, throw) | per seed | mean +- SD | t 95% of the mean "
        + "| vs B0 DR mean [t 95%] |",
        "| --- " * 6 + "|",
    ]

    def ci(x):
        return "-" if x is None else f"[{x[0]:+.4f}, {x[1]:+.4f}]"

    for g, ids in groups.items():
        for j, (dr, th) in enumerate(SCALES):
            rates = [rows[i][j]["success_rate"] for i in ids]
            mean, sd, rate_ci = seed_mean_ci(rates)
            diffs = [rows[i][j].get(f"vs_{REFERENCE}") for i in ids]
            vs = "-"
            if all(diffs):
                d_mean, _, d_ci = seed_mean_ci([d["diff"] for d in diffs])
                vs = f"{d_mean:+.4f} {ci(d_ci)}"
            cells = [g, f"({dr:g},{th:g})", " / ".join(f"{x:.4f}" for x in rates)]
            cells += [f"{mean:.4f} +- {sd:.4f}", f"[{rate_ci[0]:.4f}, {rate_ci[1]:.4f}]", vs]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def summary_md(r: dict) -> str:
    lines = ["# drone-policy evaluation suite", ""]
    p = r.get("protocol", {})
    lines.append(f"seed {p.get('seed')}, quick={p.get('quick')}, runs {p.get('runs')}")
    if "e1" in r:
        lines += ["", f"## E1 held-out ({r['e1']['protocol']['episodes']} episodes per scale)", ""]
        lines += [
            (
                "Per run: success [Wilson 95%]. The Wilson interval counts episode noise only and "
                "says nothing about training-seed variance; the paired diff vs B0 DR is +- its "
                "standard error over the same episodes."
            ),
            "",
        ]
        lines += [
            "| controller | "
            + " | ".join(f"({a:g},{b:g})" for a, b in SCALES)
            + " | vs B0 DR at (1,1) |"
        ]
        lines += ["| --- " * (len(SCALES) + 2) + "|"]
        for cid, evs in r["e1"]["rows"].items():
            cells = [
                f"{e['success_rate']:.4f} [{e['wilson95'][0]:.3f}, {e['wilson95'][1]:.3f}]"
                for e in evs
            ]
            vs = evs[0].get(f"vs_{REFERENCE}")
            cells.append(f"{vs['diff']:+.4f} +- {vs['se']:.4f}" if vs else "reference")
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
        lines += _config_table(r)
    if "e2" in r:
        lines += ["", "## E2 OOD slices (success at full DR)", ""]
        for name, spec in r["e2"]["slices"].items():
            lines += [f"{name} ({spec['unit']}, training {spec['train_range']}):", ""]
            lines += ["| controller | " + " | ".join(f"{v:g}" for v in spec["values"]) + " |"]
            lines += ["| --- " * (len(spec["values"]) + 1) + "|"]
            for cid, rows in spec["rows"].items():
                lines.append(
                    f"| {cid} | " + " | ".join(f"{x['success_rate']:.3f}" for x in rows) + " |"
                )
            lines.append("")
    if "e3" in r:
        lines += [
            "## E3 nominal hover (0,0)",
            "",
            "| controller | success | pos RMS m | tilt RMS deg | axis tilt RMS deg |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cid, x in r["e3"]["rows"].items():
            lines.append(
                f"| {cid} | {x['success_rate']:.4f} | {x['pos_rms_m']:.4f} | {x['tilt_rms_deg']:.3f} | {x['axis_tilt_rms_deg']:.3f} |"
            )
    if "e4" in r:
        eff = ["in_distribution", *E4_EFFECTS]
        lines += ["", "## E4 unseen effects (success; drop vs in-distribution)", ""]
        lines += ["| controller | " + " | ".join(eff) + " |", "| --- " * (len(eff) + 1) + "|"]
        for cid, x in r["e4"]["rows"].items():
            cells = [
                f"{x[e]['success_rate']:.3f}"
                + (f" ({x[e]['drop']:+.3f})" if "drop" in x[e] else "")
                for e in eff
            ]
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    if "e5" in r:
        lines += [
            "",
            "## E5 estimator gap",
            "",
            "| run | trained | evaluated | "
            + " | ".join(f"({a:g},{b:g})" for a, b in SCALES)
            + " |",
            "| --- " * (3 + len(SCALES)) + "|",
        ]
        for x in r["e5"]["rows"]:
            cells = []
            for e in x["final_eval"]:
                d = e.get("vs_training_estimator")
                cells.append(f"{e['success_rate']:.4f}" + (f" ({d['diff']:+.4f})" if d else ""))
            lines.append(
                f"| {x['run']} | {x['train_estimator']} | {x['eval_estimator']} | "
                + " | ".join(cells)
                + " |"
            )
    if "e9" in r:
        lines += [
            "",
            "## E9 smoothness at (1,1)",
            "",
            "| controller | success | pwm diff std | mean sq action diff |",
            "| --- | --- | --- | --- |",
        ]
        for cid, x in r["e9"]["rows"].items():
            lines.append(
                f"| {cid} | {x['success_rate']:.4f} | {x['pwm_diff_std']:.5f} | {x['mean_sq_action_diff']:.5f} |"
            )
    if "e8" in r:
        lines += [
            "",
            "## E8 numpy export",
            "",
            "| policy | obs | max abs diff f64 | numpy median us | jnp jit median us |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cid, x in r["e8"]["rows"].items():
            lines.append(
                f"| {cid} | {x['observations']} | {x['max_abs_diff_f64']:.3e} | {x['numpy_latency_median_us']} | {x['jnp_jit_latency_median_us']} |"
            )
    if "e6" in r:
        e = r["e6"]
        lines += ["", f"## E6 deploy in-graph (chosen {e.get('chosen')})", ""]
        for cid, x in e.get("a", {}).items():
            lines.append(
                f"- (a) {cid}: max |dpos| {x['max_dpos_vs_jax_env_m']:.3e} m vs JAX env, {x['max_dpos_vs_jax_f64_forward_m']:.3e} m vs the f64 forward ({x['ticks']} ticks)"
            )
        if "b" in e:
            b = e["b"]
            lines.append(
                f"- (b) {b['episodes']} episodes: success labels agree {b['cranelift_vs_jax']['success_agree']}, median max |dpos| {b['deviation_vs_jax_env']['until_end_m']['median']} m"
            )
        if "c" in e:
            c = e["c"]
            lines.append(
                f"- (c) {c['drones']} drones jax-cpu: labels agree {c['world_vs_vmapped_env']['success_agree']}, median max |dpos| {c['deviation_vs_jax_env']['until_end_m']['median']} m"
            )
        if "bench" in e:
            lines.append(f"- main.py bench: {e['bench']}")
    if "e10" in r:
        x = r["e10"]["rollout"]
        lines += [
            "",
            "## E10 throughput",
            "",
            f"- rollout: {x['env_steps_per_s']} env steps/s ({x['flight_hours_per_min']} flight hours per wall minute)",
        ]
        for cid, t in r["e10"]["training_loop"].items():
            lines.append(
                f"- training loop {cid}: {t['median_steps_per_s']} steps/s ({t['flight_hours_per_min']} h/min)"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sections", nargs="+", help="all or any of " + ", ".join(SECTIONS))
    p.add_argument("--runs", nargs="+", required=True, help="directories of RUN/policy.npz")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--quick", action="store_true", help="small smoke version (tests)")
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--seed", type=int, default=20011)
    p.add_argument("--baselines", default="b0-nominal,b0-dr,b1-dr")
    p.add_argument("--baseline-estimator", default="mekf6")
    p.add_argument("--controllers", help="comma list of controller ids (or globs) to keep")
    p.add_argument(
        "--e6",
        help="controller for e6 (b), (c) and the bench (default: the best seed of the multi-seed "
        "config with the best mean e1 success at (1,1))",
    )
    p.add_argument("--track-a", nargs="*", help="summary.json files with cross-estimator rows")
    p.add_argument("--note", help="run conditions recorded with the e8 and e10 timings")
    args = p.parse_args(argv)
    sections = list(SECTIONS) if "all" in args.sections else args.sections
    for x in sections:
        if x not in SECTIONS:
            p.error(f"unknown section {x}")
    args.sections = sections
    args.baselines = [b for b in args.baselines.split(",") if b]
    for b in args.baselines:
        if b not in BASELINES:
            p.error(f"unknown baseline {b}")
    args.controllers = args.controllers.split(",") if args.controllers else None
    args.runs = [os.path.expanduser(r) for r in args.runs]
    args.out = os.path.expanduser(args.out)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.devices > 1:
        flag = f"--xla_force_host_platform_device_count={args.devices}"
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {flag}".strip()
    sys.path.insert(0, HERE)
    s = Suite(args)
    _log(f"controllers: {[c['id'] for c in s.ctrls]}")
    timing = s.results.setdefault("section_wall_s", {})
    for name in args.sections:
        t0 = time.perf_counter()
        globals()[name](s)
        timing[name] = round(time.perf_counter() - t0, 1)
        s.save()
        _log(f"{name} done in {timing[name]} s")
    with open(os.path.join(s.out, "summary.md"), "w") as f:
        f.write(summary_md(s.results))
    s.save()


if __name__ == "__main__":
    main()
