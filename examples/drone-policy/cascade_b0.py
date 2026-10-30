"""B0: classical cascade at the CTBR seam, driven by the 19-dim actor obs only.

Heading-frame PID on the VIO position error gives a desired acceleration (tilt clamped, norm
capped with vertical priority and a floor on the horizontal share); its direction is mapped into
the body frame through the roll and pitch recovered from g_body_est (yaw-free), a reduced-attitude
P loop with a gyro rate-error boost turns the thrust-axis error into body-rate setpoints, yaw is
damped with a bounded rate error, and the collective is the hover collective scaled by the
acceleration along the thrust axis. The result goes back through the env action scaling. Pure
jnp with float ops only (no scatter or integer ops), so it also runs in a Cranelift world. B1 is
the same controller fed the critic's truth attitude, rate, position and velocity (plus the actor
p_int_h), with the same gain sets (selected on mekf6 observations, not retuned for truth
inputs): a privileged reference, not an upper bound, and not deployable.

Two gain sets, because no single set is best on both distributions: B0_GAINS is tuned on the
nominal plant, B0_DR_GAINS under full DR, where the mass/thrust mismatch leaves a z offset that
a stiffer z loop and the integral remove. The nominal set leads by ~0.02 success on the nominal
plant, the DR set from dr 0.5 up.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 python cascade_b0.py tune --dr 0 --log A.jsonl
    python cascade_b0.py select A.jsonl --dr 0
    python cascade_b0.py eval --estimator mekf6,truth --set nominal,dr --dr 0,0,0.5,1 --throw 0,1,1,1
"""

import argparse
import json
import os
import time
import typing as ty

import jax
import jax.numpy as jnp
import numpy as np
import plant_dr
import policy_io
from dr import G, quat_rotate
from env import TERM_OMEGA, TERM_POS, TERM_TILT, TERM_Z, DroneEnv

A_Z_MIN = 0.3 * G  # floor on the vertical part of the commanded acceleration
# body direction of the torque-free hover thrust (~1.9 deg off z for the Talon); the env starts
# throws from the trim that makes it vertical, where g_body = THRUST_AXIS and B0 commands a = 0
THRUST_AXIS = plant_dr.hover_trim().axis
CAUSES = ("nonfinite", "tilt", "omega", "z_low", "pos_far")  # termination, in priority order
HOVER_WINDOW = 200  # policy steps (the last 2 s) for the hover metrics


class B0Gains(ty.NamedTuple):
    kp_xy: jax.Array  # 1/s^2 on p_err_h
    kd_xy: jax.Array  # 1/s on v_h
    ki_xy: jax.Array  # 1/s^3 on p_int_h
    kp_z: jax.Array
    kd_z: jax.Array
    ki_z: jax.Array
    k_att: jax.Array  # rad/s per rad of thrust-axis error
    k_rp: jax.Array  # roll/pitch rate-error boost: sp = w + k_rp (w - gyro)
    k_yaw: jax.Array  # yaw-rate damping: yaw-rate error = -k_yaw gyro_z ...
    yaw_err_max: jax.Array  # ... clipped to this (rad/s) so yaw does not starve the thrust
    max_tilt: jax.Array  # rad
    a_max: jax.Array  # m/s^2 cap on |a_cmd| (full throttle gives ~1.35 g); vertical gets it first
    h_floor: jax.Array  # share of the tilt-limited horizontal authority kept when az hits a_max


# hand-set starting point of the search
B0_DEFAULT = B0Gains(2.5, 2.5, 0.2, 3.0, 3.0, 0.5, 6.0, 2.0, 1.0, 0.1, 0.5, 1.3 * G, 0.0)
# the env v1 sets (B0Gains field order): start sets and reference candidates of the env v2 tune
_T1 = (
    (3.739, 2.909, 0.113, 8.333, 5.336, 0.0, 6.477, 2.485, 1.856, 0.1279, 0.4008, 14.35, 0.4569),
    (3.786, 2.536, 0.1763, 6.623, 3.981, 5.215, 5.424, 2.407, 1.064, 0.1353, 0.4739, 13.63, 0.3774),
)
T1_SETS = {"t1-nominal": B0Gains(*_T1[0]), "t1-dr": B0Gains(*_T1[1])}
# Both sets come from one pipeline on env v2 with full throws and 8 XLA host devices; D = 0 for
# B0_GAINS, 1 for B0_DR_GAINS:
#   tune --estimator mekf6 --dr D --seed 11+D --gains t1-nominal --gains t1-dr --log A_mekf6
#   tune --estimator truth --dr D --seed 13+D --gains t1-nominal --gains t1-dr --log A_truth
#   tune --dr D --seed 21+D --sets 0 --from A_mekf6 --from A_truth --refine 64 --refine-rounds 6
#        --top 6 --episodes 512 --perturb-scale 0.15 --val-episodes 1024 --log B
#   select A_mekf6 A_truth B --dr D --gains t1-nominal --gains t1-dr
# select re-scores the 24 validated leaders and the T1 sets on 4096 fresh mekf6 episodes (seed
# 5000) and prints the winner as below.
# Nominal plant: held-out success 0.975 (t1-nominal 0.958); ~0.60 at dr 1. a_max sits at the top
# of SPACE; raising it to 22-1000 does not help (0.971-0.972).
B0_GAINS = B0Gains(
    kp_xy=3.679,
    kd_xy=3.341,
    ki_xy=0.3397,
    kp_z=7.625,
    kd_z=4.562,
    ki_z=0.0,
    k_att=6.212,
    k_rp=1.977,
    k_yaw=2.094,
    yaw_err_max=0.1221,
    max_tilt=0.4075,
    a_max=19.55,
    h_floor=0.6319,
)
# Full DR: held-out success 0.741 (t1-nominal 0.582, t1-dr 0.474); 0.94-0.96 on the nominal plant
B0_DR_GAINS = B0Gains(
    kp_xy=4.254,
    kd_xy=3.075,
    ki_xy=0.3278,
    kp_z=11.77,
    kd_z=7.888,
    ki_z=1.247,
    k_att=6.477,
    k_rp=2.149,
    k_yaw=2.217,
    yaw_err_max=0.1664,
    max_tilt=0.355,
    a_max=18.07,
    h_floor=0.5185,
)
GAIN_SETS = {"nominal": B0_GAINS, "dr": B0_DR_GAINS}


def cascade(g_body, gyro, p_err, v, p_int, gains: B0Gains) -> jax.Array:
    k = gains
    kp = jnp.stack([k.kp_xy, k.kp_xy, k.kp_z])
    kd = jnp.stack([k.kd_xy, k.kd_xy, k.kd_z])
    ki = jnp.stack([k.ki_xy, k.ki_xy, k.ki_z])
    a = -kp * p_err - kd * v - ki * p_int + jnp.array([0.0, 0.0, G])

    # vertical first: the horizontal part gets the tilt limit and what is left of a_max, but at
    # least h_floor of the tilt limit (with none, a low drone holds az at a_max with no
    # horizontal authority and drifts at its throw velocity until |e_p| > 4 m)
    az = jnp.clip(a[2], A_Z_MIN, k.a_max)
    h = jnp.sqrt(a[0] ** 2 + a[1] ** 2)
    # factored so a clipped az gives exactly 0 (a_max^2 - az^2 fuses to +-4e-15, and its sqrt
    # then differs by 1e-7 between backends or is NaN)
    room = jnp.sqrt(jnp.maximum((k.a_max - az) * (k.a_max + az), 0.0))
    h_tilt = az * jnp.sin(k.max_tilt) / jnp.cos(k.max_tilt)
    h_max = jnp.maximum(jnp.minimum(h_tilt, room), k.h_floor * h_tilt)
    s = jnp.minimum(1.0, h_max / jnp.maximum(h, 1e-9))
    ax, ay = a[0] * s, a[1] * s

    # heading-to-body rotation R_hb = Ry(theta) Rx(phi) from g_body = R_hb^T z:
    # theta = -asin(g_x), phi = atan2(g_y, g_z)
    g = g_body / jnp.maximum(jnp.sqrt(jnp.sum(g_body**2)), 1e-9)
    c_th = jnp.sqrt(g[1] ** 2 + g[2] ** 2)
    s_th = -g[0]
    c_ph = g[2] / jnp.maximum(c_th, 1e-9)
    s_ph = g[1] / jnp.maximum(c_th, 1e-9)
    ux = c_th * ax - s_th * az
    uz = s_th * ax + c_th * az
    b = jnp.stack([ux, c_ph * ay + s_ph * uz, -s_ph * ay + c_ph * uz])

    t = jnp.asarray(THRUST_AXIS)
    thrust = jnp.dot(b, t)
    n = b / jnp.maximum(jnp.sqrt(jnp.sum(b**2)), 1e-9)
    axis = jnp.cross(t, n)
    sin_e = jnp.sqrt(jnp.sum(axis**2))
    err = jnp.arctan2(sin_e, jnp.dot(t, n))
    # the stock rate PID's D term (on measured angular acceleration) makes the rate loop lag by
    # ~0.4 s; boosting the rate error with the gyro speeds it up
    w_xy = k.k_att * err / jnp.maximum(sin_e, 1e-9) * axis[:2]
    w_xy = w_xy + k.k_rp * (w_xy - gyro[:2])
    # a large yaw-rate error saturates the mixer's yaw headroom, which caps the average throttle;
    # above |gyro_z| = 1.5 + yaw_err_max the action range clips the setpoint to 1.5 rad/s, so the
    # error there is |gyro_z| - 1.5, the smallest the action range allows
    w_z = gyro[2] - jnp.clip(k.k_yaw * gyro[2], -k.yaw_err_max, k.yaw_err_max)
    w = jnp.concatenate([w_xy, w_z[None]])

    collective = policy_io.HOVER_COLLECTIVE * thrust / G
    c = (collective - policy_io.HOVER_COLLECTIVE) / policy_io.COLLECTIVE_SCALE
    return jnp.clip(jnp.concatenate([w / policy_io.ACT_SCALE, c[None]]), -1.0, 1.0)


def b0_action(obs: jax.Array, gains: B0Gains = B0_GAINS) -> jax.Array:
    """Normalized action f64[4] in [-1, 1] from the actor obs (pure, jit/vmap-safe)."""
    return cascade(obs[0:3], obs[3:6], obs[6:9], obs[9:12], obs[12:15], gains)


def b1_action(obs: jax.Array, critic_obs: jax.Array, gains: B0Gains = B0_GAINS) -> jax.Array:
    """B0 on the truth g_body, omega, p_err_h and v_h of critic_obs, with the actor p_int_h."""
    c = critic_obs
    return cascade(c[19:22], c[22:25], c[25:28], c[28:31], obs[12:15], gains)


CONTROLLERS = {
    "b0": lambda obs, critic, gains: b0_action(obs, gains),
    "b1": b1_action,
}


def make_episode(env: DroneEnv, controller: str = "b0"):
    """-> episode(key, gains, dr_scale, throw_scale) -> per-episode stats (one env)."""
    act = CONTROLLERS[controller]
    steps = env.episode_steps

    def episode(key, gains, dr_scale, throw_scale):
        state, obs, critic = env.reset(key, dr_scale, throw_scale)
        zero, false = jnp.float64(0.0), jnp.bool_(False)
        acc = {
            "ret": jnp.float32(0.0),
            "length": jnp.int32(0),
            "success": false,
            "terminated": false,
            "cause": jnp.zeros(len(CAUSES), bool),
            "e_p_max": zero,  # whole episode, throw transient included
            "e_p_max_win": zero,  # last HOVER_WINDOW steps
            "se_p": zero,
            "se_tilt": zero,
            "se_axis": zero,
            "n_win": jnp.int32(0),
        }

        def body(c, _):
            state, obs, critic, alive, acc = c
            state, obs, critic, r, done, info = env.step(state, act(obs, critic, gains))
            pos, vel = state.sim[env.i_pos], state.sim[env.i_vel]
            q, p = pos[:4], pos[4:]
            e_p = jnp.linalg.norm(p - state.target)
            cos_tilt = 1.0 - 2.0 * (q[0] ** 2 + q[1] ** 2)
            tilt = jnp.arccos(jnp.clip(cos_tilt, -1.0, 1.0))
            axis_up = quat_rotate(q, jnp.asarray(THRUST_AXIS))[2]
            axis_tilt = jnp.arccos(jnp.clip(axis_up, -1.0, 1.0))
            term = alive & info["terminated"]
            flags = jnp.stack(
                [
                    ~env.finite(state.sim),
                    cos_tilt < np.cos(TERM_TILT),
                    jnp.linalg.norm(vel[:3]) > TERM_OMEGA,
                    p[2] < TERM_Z,
                    e_p > TERM_POS,
                ]
            )
            ok = alive & ~info["terminated"]
            win = ok & (info["t"] > steps - HOVER_WINDOW)
            acc = {
                "ret": acc["ret"] + jnp.where(alive, r, 0.0),
                "length": acc["length"] + alive.astype(jnp.int32),
                "success": acc["success"] | (alive & done & info["success"]),
                "terminated": acc["terminated"] | term,
                "cause": jnp.where(term, flags, acc["cause"]),
                "e_p_max": jnp.where(ok, jnp.maximum(acc["e_p_max"], e_p), acc["e_p_max"]),
                "e_p_max_win": jnp.where(
                    win, jnp.maximum(acc["e_p_max_win"], e_p), acc["e_p_max_win"]
                ),
                "se_p": acc["se_p"] + jnp.where(win, e_p**2, 0.0),
                "se_tilt": acc["se_tilt"] + jnp.where(win, tilt**2, 0.0),
                "se_axis": acc["se_axis"] + jnp.where(win, axis_tilt**2, 0.0),
                "n_win": acc["n_win"] + win.astype(jnp.int32),
            }
            return (state, obs, critic, alive & ~done, acc), None

        carry = (state, obs, critic, jnp.bool_(True), acc)
        (*_, acc), _ = jax.lax.scan(body, carry, None, length=steps)
        return acc

    return episode


class Runner:
    """Batched episodes over (key, gains) pairs, split over the local XLA devices."""

    def __init__(self, env: DroneEnv, controller: str = "b0", chunk: int = 8192):
        self.devices = jax.local_device_count()
        self.chunk = chunk - chunk % self.devices
        batched = jax.vmap(make_episode(env, controller), in_axes=(0, 0, None, None))
        if self.devices > 1:
            self.fn = jax.pmap(batched, in_axes=(0, 0, None, None))
        else:
            self.fn = jax.jit(batched)

    def __call__(self, keys, gains: B0Gains, dr_scale, throw_scale) -> dict:
        n, d = keys.shape[0], self.devices
        dr_scale, throw_scale = jnp.float64(dr_scale), jnp.float64(throw_scale)
        outs = []
        for lo in range(0, n, self.chunk):
            m = min(self.chunk, n - lo)
            idx = lo + np.arange(m + (-m % d)) % m  # pad to a multiple of the device count
            k, g = keys[idx], B0Gains(*[jnp.asarray(x)[idx] for x in gains])
            if d > 1:
                k, g = k.reshape(d, -1), B0Gains(*[x.reshape(d, -1) for x in g])
            o = self.fn(k, g, dr_scale, throw_scale)
            skip = 2 if d > 1 else 1
            outs.append(
                {c: np.asarray(v).reshape((-1,) + v.shape[skip:])[:m] for c, v in o.items()}
            )
        return {c: np.concatenate([o[c] for o in outs]) for c in outs[0]}


def summarize(o: dict) -> dict:
    """Episode stats. The hover RMS and max |e_p| are over the episodes that did not terminate;
    *_last2s use the last HOVER_WINDOW steps, e_p_max_episode includes the throw transient. The
    tilt is body z from vertical (as in the env; the hover trim alone is 1.89 deg), axis_tilt the
    hover thrust axis from vertical."""
    term = o["terminated"]
    first = np.where(o["cause"].any(-1), np.argmax(o["cause"], axis=-1), -1)
    survived = ~term
    n_win = np.maximum(o["n_win"][survived].sum(), 1)

    def rms_deg(k):
        return float(np.degrees(np.sqrt(o[k][survived].sum() / n_win)))

    return {
        "episodes": int(term.size),
        "success": float(o["success"].mean()),
        "mean_return": float(o["ret"].mean()),
        "mean_length": float(o["length"].mean()),
        "terminated": float(term.mean()),
        "causes": {c: int(np.sum(term & (first == i))) for i, c in enumerate(CAUSES)},
        "pos_rms_last2s": float(np.sqrt(o["se_p"][survived].sum() / n_win)),
        "tilt_rms_last2s_deg": rms_deg("se_tilt"),
        "axis_tilt_rms_last2s_deg": rms_deg("se_axis"),
        "e_p_max_last2s": float(o["e_p_max_win"][survived].max(initial=0.0)),
        "e_p_max_episode": float(o["e_p_max"][survived].max(initial=0.0)),
    }


def as_batch(gains: ty.Sequence[B0Gains] | B0Gains, n: int | None = None) -> B0Gains:
    """Stack gain sets (or repeat one set n times) into a B0Gains of f64 arrays."""
    if isinstance(gains, B0Gains):
        return B0Gains(*[jnp.full(n, v, jnp.float64) for v in gains])
    return B0Gains(*[jnp.asarray(v, jnp.float64) for v in zip(*gains)])


def parse_gains(s: str) -> B0Gains:
    """A GAIN_SETS or T1_SETS name, or a json dict of B0Gains fields (missing ones from
    B0_GAINS)."""
    named = {**GAIN_SETS, **T1_SETS}
    if s in named:
        return named[s]
    return B0_GAINS._replace(**json.loads(s))


def leaders(path: str) -> list[tuple[str, B0Gains]]:
    """The validated leaders of a tune log as (log:set, gains)."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "validate" in r:
                out.append((f"{os.path.basename(path)}:{r['set']}", B0Gains(**r["gains"])))
    return out


def round_gains(g: B0Gains, digits: int = 4) -> B0Gains:
    return B0Gains(*[float(f"{float(v):.{digits}g}") for v in g])


# search space: (low, high, log); h_floor = 1 turns the vertical priority off
SPACE = B0Gains(
    kp_xy=(0.5, 10.0, True),
    kd_xy=(0.7, 6.0, True),
    ki_xy=(0.0, 1.5, False),
    kp_z=(1.0, 25.0, True),
    kd_z=(1.0, 12.0, True),
    ki_z=(0.0, 8.0, False),
    k_att=(2.0, 12.0, True),
    k_rp=(0.0, 4.0, False),
    k_yaw=(0.2, 4.0, True),
    yaw_err_max=(0.05, 1.0, True),
    max_tilt=(0.2, 0.7, False),
    a_max=(1.15 * G, 2.0 * G, False),
    h_floor=(0.0, 1.0, False),
)


def sample_gains(rng: np.random.Generator, n: int) -> list[B0Gains]:
    cols = []
    for lo, hi, log in SPACE:
        u = rng.uniform(size=n)
        cols.append(np.exp(np.log(lo) + u * np.log(hi / lo)) if log else lo + u * (hi - lo))
    return [B0Gains(*[float(c[i]) for c in cols]) for i in range(n)]


def perturb(rng: np.random.Generator, g: B0Gains, scale: float = 0.25, dims: int = 3) -> B0Gains:
    """Move `dims` random gains of g (log-normal, or by scale/2 of the range)."""
    out = [float(v) for v in g]
    for i in rng.choice(len(SPACE), size=min(dims, len(SPACE)), replace=False):
        lo, hi, log = SPACE[i]
        if log:
            v = out[i] * np.exp(scale * rng.normal())
        else:
            v = out[i] + scale * (hi - lo) * 0.5 * rng.normal()
        out[i] = float(np.clip(v, lo, hi))
    return B0Gains(*out)


def score_sets(run: Runner, sets: list[B0Gains], keys, dr_scale, throw_scale) -> list[dict]:
    """Every set on the same episodes (common random numbers)."""
    e = keys.shape[0]
    g = as_batch(sets)
    g = jax.tree.map(lambda x: jnp.repeat(x, e), g)
    o = run(jnp.concatenate([keys] * len(sets)), g, dr_scale, throw_scale)
    o = {k: v.reshape((len(sets), e) + v.shape[1:]) for k, v in o.items()}
    return [summarize({k: v[i] for k, v in o.items()}) for i in range(len(sets))]


def rank_key(s: dict):
    return (s["success"], s["mean_return"])


def tune(args):
    env = DroneEnv(args.estimator)
    run = Runner(env, "b0", args.chunk)
    rng = np.random.default_rng(args.seed)
    keys = jax.random.split(jax.random.key(args.seed), args.episodes)
    t0 = time.perf_counter()
    # the shipped sets are not start sets, so a re-run does not depend on them
    extra = [parse_gains(g) for g in args.gains]
    extra += [g for path in args.start for _, g in leaders(path)]
    sets = list(dict.fromkeys([B0_DEFAULT, *extra]))
    sets += sample_gains(rng, max(args.sets - len(sets), 0))
    scores = score_sets(run, sets, keys, args.dr, args.throw)
    log(
        {"stage": "random", "sets": len(sets), "s": round(time.perf_counter() - t0, 1)},
        args.log,
    )
    for _ in range(args.refine_rounds):
        order = sorted(range(len(sets)), key=lambda i: rank_key(scores[i]), reverse=True)
        top = [sets[i] for i in order[: args.top]]
        new = [
            perturb(rng, top[i % len(top)], args.perturb_scale, args.perturb_dims)
            for i in range(args.refine)
        ]
        scores += score_sets(run, new, keys, args.dr, args.throw)
        sets += new
        best = scores[max(range(len(sets)), key=lambda i: rank_key(scores[i]))]
        log({"stage": "refine", "sets": len(sets), "best": best}, args.log)
    order = sorted(range(len(sets)), key=lambda i: rank_key(scores[i]), reverse=True)
    for i in order[:10]:
        log({"set": i, "gains": sets[i]._asdict(), **scores[i]}, args.log)

    # re-score the leaders on fresh episodes to avoid picking a lucky draw
    val_keys = jax.random.split(jax.random.key(args.seed + 1000), args.val_episodes)
    top = [sets[i] for i in order[: args.val_top]]
    val = score_sets(run, top, val_keys, args.dr, args.throw)
    best = max(range(len(top)), key=lambda i: rank_key(val[i]))
    for i, (g, s) in enumerate(zip(top, val)):
        log({"validate": i, "set": order[i], "gains": g._asdict(), **s}, args.log)
    log(
        {
            "winner": order[best],
            "estimator": args.estimator,
            "dr": args.dr,
            "throw": args.throw,
            "gains": top[best]._asdict(),
            "validation": val[best],
            "search": scores[order[best]],
            "total_sets": len(sets),
            "s": round(time.perf_counter() - t0, 1),
        },
        args.log,
    )


def select(args):
    """Re-score the validated leaders of tune logs, rounded as shipped, on fresh episodes."""
    cands = {}
    for path in args.logs:
        for name, g in leaders(path):
            cands.setdefault(round_gains(g), name)
    for s in args.gains:
        cands.setdefault(round_gains(parse_gains(s)), s)
    sets = list(cands)
    run = Runner(DroneEnv(args.estimator), "b0", args.chunk)
    keys = jax.random.split(jax.random.key(args.seed), args.episodes)
    scores = score_sets(run, sets, keys, args.dr, args.throw)
    order = sorted(range(len(sets)), key=lambda i: rank_key(scores[i]), reverse=True)
    for rank, i in enumerate(order):
        rec = {"rank": rank, "source": cands[sets[i]], "gains": sets[i]._asdict()}
        log({**rec, **scores[i]}, args.log)
    best = sets[order[0]]
    log({"selected": cands[best], "candidates": len(sets), "gains": best._asdict()}, args.log)
    print(f"B0Gains({', '.join(f'{k}={v!r}' for k, v in best._asdict().items())})")


def evaluate(args):
    pairs = list(zip(_floats(args.dr), _floats(args.throw)))
    for est in args.estimator.split(","):
        env = DroneEnv(est)
        keys = jax.random.split(jax.random.key(args.seed), args.episodes)
        for ctrl in args.controller.split(","):
            run = Runner(env, ctrl, args.chunk)
            names = [args.set] if args.set.lstrip().startswith("{") else args.set.split(",")
            for name in names:
                gains = parse_gains(name)
                for d, t in pairs:
                    t0 = time.perf_counter()
                    s = summarize(run(keys, as_batch(gains, args.episodes), d, t))
                    rec = {"controller": ctrl, "gains": name, "estimator": est, "dr": d, "throw": t}
                    rec.update(s, s=round(time.perf_counter() - t0, 1))
                    log(rec, args.log)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def log(rec: dict, path: str | None):
    line = json.dumps(rec, default=float)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--estimator", default="mekf6")
    t.add_argument("--sets", type=int, default=512)
    t.add_argument("--refine", type=int, default=128)
    t.add_argument("--refine-rounds", type=int, default=6)
    t.add_argument("--top", type=int, default=8)
    t.add_argument("--perturb-scale", type=float, default=0.25)
    t.add_argument("--perturb-dims", type=int, default=3)
    t.add_argument("--episodes", type=int, default=256)
    t.add_argument("--val-top", type=int, default=8)
    t.add_argument("--val-episodes", type=int, default=512)
    t.add_argument("--dr", type=float, default=0.0)
    t.add_argument("--throw", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--chunk", type=int, default=8192)
    t.add_argument("--gains", action="append", default=[], help="extra start set (see --set)")
    t.add_argument(
        "--from", dest="start", action="append", default=[], help="start from a tune log's leaders"
    )
    t.add_argument("--log", default=None)
    s = sub.add_parser("select")
    s.add_argument("logs", nargs="+", help="tune logs whose validated leaders are re-scored")
    s.add_argument("--estimator", default="mekf6")
    s.add_argument("--dr", type=float, default=0.0)
    s.add_argument("--throw", type=float, default=1.0)
    s.add_argument("--episodes", type=int, default=4096)
    s.add_argument("--seed", type=int, default=5000)
    s.add_argument("--chunk", type=int, default=8192)
    s.add_argument("--gains", action="append", default=[], help="extra candidate (see --set)")
    s.add_argument("--log", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--estimator", default="mekf6,truth")
    e.add_argument("--controller", default="b0,b1")
    e.add_argument(
        "--set",
        "--gains",
        dest="set",
        default="nominal,dr",
        help="comma list of gain sets: nominal (B0_GAINS), dr (B0_DR_GAINS), or one json dict",
    )
    e.add_argument("--dr", default="0,0,0.5,1")
    e.add_argument("--throw", default="0,1,1,1")
    e.add_argument("--episodes", type=int, default=512)
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--chunk", type=int, default=8192)
    e.add_argument("--log", default=None)
    args = p.parse_args()
    if args.cmd in ("tune", "select"):
        (tune if args.cmd == "tune" else select)(args)
        return
    if len(_floats(args.dr)) != len(_floats(args.throw)):
        p.error("--dr and --throw need the same number of values")
    evaluate(args)


if __name__ == "__main__":
    main()
