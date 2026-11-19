"""PPO smoke tests on a pure-JAX point-mass env that follows the DroneEnv API.

Run as a script for a longer learning demo that prints return and success before and after:

    python examples/drone-policy/tests/test_ppo_smoke.py --iters 400 --num-envs 256
"""

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
import typing as ty

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import nn
import numpy as np
import ppo
import pytest
import train

# elodin enables x64 process-wide; the networks must stay float32 regardless
jax.config.update("jax_enable_x64", True)

F32 = jnp.float32
TARGET = jnp.array([0.0, 0.0, 2.0])
# toy actions are 4 m/s^2 accelerations and need more exploration than the drone's rate setpoints
TOY_LOG_STD = math.log(0.3)


class PointState(ty.NamedTuple):
    pos: jax.Array
    vel: jax.Array
    p_int: jax.Array
    a_prev: jax.Array
    t: jax.Array
    hold_ok: jax.Array
    key: jax.Array


class PointMassEnv:
    """3D point mass that must reach and hold [0, 0, 2] m. Same API and obs layout as DroneEnv:
    obs = g_body, omega, p_err, v, p_int, a_prev (19); critic adds noise-free p_err and v."""

    obs_dim, critic_dim, act_dim = 19, 25, 4

    def __init__(self, episode_steps: int = 200, dt: float = 0.05):
        self.episode_steps = episode_steps
        self.dt = dt
        self.hold_from = int(0.8 * episode_steps)

    def _observe(self, state: PointState, key):
        noise = jax.random.normal(key, (12,))
        err = state.pos - TARGET
        obs = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 1.0]) + 0.02 * noise[0:3],
                0.05 * noise[3:6],
                err + 0.03 * noise[6:9],
                state.vel + 0.05 * noise[9:12],
                state.p_int,
                state.a_prev,
            ]
        )
        critic_obs = jnp.concatenate([obs, err, state.vel])
        return obs.astype(F32), critic_obs.astype(F32)

    def reset(self, key, dr_scale, throw_scale=None):
        s = dr_scale if throw_scale is None else throw_scale
        k_pos, k_vel, k_obs, key = jax.random.split(key, 4)
        state = PointState(
            pos=TARGET + s * jax.random.uniform(k_pos, (3,), minval=-1.0, maxval=1.0),
            vel=s * jax.random.uniform(k_vel, (3,), minval=-0.5, maxval=0.5),
            p_int=jnp.zeros(3),
            a_prev=jnp.zeros(4, F32),
            t=jnp.zeros((), jnp.int32),
            hold_ok=jnp.array(True),
            key=key,
        )
        return (state, *self._observe(state, k_obs))

    def step(self, state: PointState, action):
        a = jnp.clip(action, -1.0, 1.0).astype(F32)
        key, k_obs = jax.random.split(state.key)
        acc = 4.0 * jnp.stack([a[0], a[1], a[3]]) - 0.3 * state.vel
        vel = state.vel + self.dt * acc
        pos = state.pos + self.dt * vel
        err = pos - TARGET
        dist, speed = jnp.linalg.norm(err), jnp.linalg.norm(vel)
        t = state.t + 1
        terminated = (dist > 4.0) | ~jnp.all(jnp.isfinite(pos))
        truncated = (t >= self.episode_steps) & ~terminated
        holding = (dist < 0.15) & (speed < 0.25)
        hold_ok = state.hold_ok & (holding | (t <= self.hold_from))
        reward = 1.0 - 0.5 * dist - 0.1 * speed - 0.05 * jnp.sum((a - state.a_prev) ** 2)
        reward = jnp.where(terminated, reward - 10.0, reward)
        p_int = jnp.clip(state.p_int + self.dt * err, -1.0, 1.0)
        state = PointState(pos, vel, p_int, a, t, hold_ok, key)
        obs, critic_obs = self._observe(state, k_obs)
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "success": truncated & hold_ok,
            "t": t,
        }
        return state, obs, critic_obs, reward.astype(F32), terminated | truncated, info


class BlowUpEnv(PointMassEnv):
    """Ignores the action, flies out of the box, and returns NaN state, obs and reward there."""

    def step(self, state, action):
        state, obs, critic_obs, reward, done, info = super().step(state, jnp.ones(4, F32))
        bad = info["terminated"]
        state = state._replace(pos=jnp.where(bad, jnp.nan, state.pos))
        obs, critic_obs = jnp.where(bad, jnp.nan, obs), jnp.where(bad, jnp.nan, critic_obs)
        return state, obs, critic_obs, jnp.where(bad, jnp.nan, reward), done, info


def evaluate(env, ts, num_envs=256, seed=123):
    fn = ppo.make_evaluate(env, num_envs)
    return jax.device_get(fn(ts.params["actor"], ts.obs_rms, jax.random.PRNGKey(seed), 1.0))


def test_gae_truncation_bootstrap():
    gamma, lam = 0.9, 0.8
    rewards = np.array([[1.0], [2.0], [3.0], [4.0]], np.float32)
    values = np.array([[0.5], [0.6], [0.7], [0.8]], np.float32)
    finished = np.array([[False], [True], [True], [False]])
    bootstrap = np.array([[0.0], [5.0], [0.0], [0.0]], np.float32)  # t=1 truncated, t=2 terminated
    last_value = np.array([0.9], np.float32)
    adv, ret = ppo.gae(rewards, values, finished, bootstrap, last_value, gamma, lam)
    d3 = 4.0 + gamma * 0.9 - 0.8
    d2 = 3.0 - 0.7
    d1 = 2.0 + gamma * 5.0 - 0.6
    d0 = 1.0 + gamma * 0.6 - 0.5
    expected = np.array([d0 + gamma * lam * d1, d1, d2, d3], np.float32)[:, None]
    np.testing.assert_allclose(np.asarray(adv), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ret), expected + values, rtol=1e-6)


def test_two_iterations_finite(tmp_path):
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=8, num_steps=32, total_iters=2)
    ts, records = train.train(env, cfg, seed=0, out=str(tmp_path), log=None)
    assert len(records) == 2
    for record in records:
        for key in ("approx_kl", "entropy", "value_loss", "policy_loss", "lr", "explained_var"):
            assert record[key] is not None and math.isfinite(record[key]), (key, record)
    assert math.isclose(records[0]["lr"], 3e-4, rel_tol=1e-5)
    assert math.isclose(records[1]["lr"], 3e-5, rel_tol=1e-4)
    assert all(leaf.dtype == F32 for leaf in jax.tree.leaves(ts.params))
    assert all(leaf.dtype == F32 for leaf in jax.tree.leaves(ts.opt.m))

    policy = dict(np.load(tmp_path / "policy.npz"))
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["iter"] == 2 and config["obs_dim"] == 19 and config["ppo"]["num_envs"] == 8
    assert policy["actor_w0"].shape == (19, 64) and policy["actor_w2"].shape == (64, 4)
    assert all(v.dtype == np.float32 for v in policy.values())
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert [json.loads(line)["iter"] for line in lines] == [0, 1]

    obs = ts.obs
    mean, _ = ppo.actor_dist(ts.params["actor"], nn.normalize(ts.obs_rms, obs))
    expected = np.clip(np.asarray(mean), -1.0, 1.0)
    np.testing.assert_allclose(
        nn.policy_action(policy, np.asarray(obs), xp=np), expected, atol=1e-5
    )
    np.testing.assert_allclose(
        nn.policy_action(policy, obs.astype(jnp.float64)), expected, atol=1e-5
    )


def test_nonfinite_terminal_steps_do_not_poison_training():
    cfg = ppo.PPOConfig(num_envs=8, num_steps=32, total_iters=2)
    ts, records = train.train(BlowUpEnv(), cfg, seed=0, log=None)
    assert sum(r["nonfinite_steps"] for r in records) > 0
    assert all(r["value_loss"] is not None and r["approx_kl"] is not None for r in records)
    leaves = jax.tree.leaves((ts.params, ts.obs_rms, ts.critic_rms, ts.ret_rms))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_per_channel_log_std():
    env = PointMassEnv()
    ts = ppo.init(env, ppo.PPOConfig(num_envs=8), jax.random.PRNGKey(0))
    expected = np.log([0.1, 0.1, 0.1, 0.2]).astype(np.float32)
    np.testing.assert_allclose(ts.params["actor"]["log_std"], expected, rtol=1e-6)
    ts = ppo.init(env, ppo.PPOConfig(num_envs=8, init_log_std=-1.0), jax.random.PRNGKey(0))
    np.testing.assert_array_equal(ts.params["actor"]["log_std"], np.full(4, -1.0, np.float32))

    # lr 0 and a zero actor head: executed actions are pure noise with the per-channel std, and
    # the update's log-prob must use that same std (ratio 1)
    std = np.array([0.05, 0.1, 0.2, 0.3], np.float32)
    log_std = tuple(float(x) for x in np.log(std))
    cfg = ppo.PPOConfig(num_envs=4096, num_steps=2, lr=0.0, lr_end=0.0, init_log_std=log_std)
    ts = ppo.init(env, cfg, jax.random.PRNGKey(1))
    mlp = [dict(layer) for layer in ts.params["actor"]["mlp"]]
    mlp[-1] = jax.tree.map(jnp.zeros_like, mlp[-1])
    ts = ts._replace(params={**ts.params, "actor": {**ts.params["actor"], "mlp": mlp}})
    ts, m = ppo.make_train_iter(env, cfg)(ts, 1.0)
    live = np.asarray(ts.ep_len) > 0  # envs reset in the last step hold a_prev = 0
    np.testing.assert_allclose(np.asarray(ts.env_state.a_prev)[live].std(0), std, rtol=0.05)
    assert abs(float(m["approx_kl"])) < 1e-6 and float(m["clip_frac"]) == 0.0
    entropy = np.sum(np.log(std) + 0.5 * (math.log(2.0 * math.pi) + 1.0))
    assert math.isclose(float(m["entropy"]), entropy, rel_tol=1e-5)


def test_throw_scale_and_schedules():
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=16, num_steps=8)
    ts = ppo.init(env, cfg, jax.random.PRNGKey(0), 1.0, 0.0)
    np.testing.assert_array_equal(ts.env_state.pos, np.broadcast_to(TARGET, (16, 3)))
    ts = ppo.init(env, cfg, jax.random.PRNGKey(0), 0.0)
    np.testing.assert_array_equal(ts.env_state.pos, np.broadcast_to(TARGET, (16, 3)))
    ramp, zero = train.schedule("curriculum", 100), train.schedule("0", 100)
    assert (ramp(0), ramp(50), ramp(400), zero(50)) == (0.0, 0.5, 1.0, 0.0)


def test_nonfinite_gradients_skip_the_step():
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=8, num_steps=16, minibatches=2)
    ts = ppo.init(env, cfg, jax.random.PRNGKey(0))
    critic = [dict(layer) for layer in ts.params["critic"]]
    critic[-1]["b"] = jnp.full_like(critic[-1]["b"], jnp.nan)
    ts = ts._replace(params={**ts.params, "critic": critic})
    ts2, m = ppo.make_train_iter(env, cfg)(ts, 1.0)
    # every step of the first epoch is skipped, so its KL is undefined and the update stops there
    assert int(m["skipped_updates"]) == cfg.minibatches and int(m["epochs"]) == 1
    assert math.isnan(float(m["value_loss"])) and int(ts2.opt.count) == 0
    for a, b in zip(jax.tree.leaves(ts.params["actor"]), jax.tree.leaves(ts2.params["actor"])):
        np.testing.assert_array_equal(a, b)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(ts2.opt))


def test_skipped_minibatches_stay_out_of_metrics(monkeypatch):
    log_prob = nn.gaussian_log_prob

    def flaky(x, mean, log_std):
        # NaN for rare samples, so only the minibatches holding one get a non-finite gradient
        z = (x - mean)[..., 0] * jnp.exp(-log_std[0])
        return log_prob(x, mean, log_std) * jnp.where(jnp.abs(z) > 3.2, jnp.nan, 1.0)

    monkeypatch.setattr(nn, "gaussian_log_prob", flaky)
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=64, num_steps=32, minibatches=8, target_kl=1e9)
    ts = ppo.init(env, cfg, jax.random.PRNGKey(0))
    ts2, m = ppo.make_train_iter(env, cfg)(ts, 1.0)
    steps = cfg.epochs * cfg.minibatches
    skipped = int(m["skipped_updates"])
    assert 0 < skipped < steps and int(ts2.opt.count) == steps - skipped
    assert all(math.isfinite(float(m[k])) for k in (*ppo.METRIC_KEYS, "grad_norm"))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(ts2.params))


def test_replay_metric_counts_repeat_resets():
    cfg = ppo.PPOConfig(num_envs=8, num_steps=32, total_iters=1)
    env = PointMassEnv(episode_steps=10)
    _, records = train.train(env, cfg, seed=0, log=None, eval_every=1, eval_envs=8)
    assert records[0]["replay_envs"] == 8 and records[0]["mean_length"] == 10.0
    assert records[0]["grad_norm"] > 0.0 and records[0]["skipped_updates"] == 0
    assert records[0]["eval_mean_length"] == 10.0


def test_final_eval_records():
    env = PointMassEnv(episode_steps=20)
    cfg = ppo.PPOConfig(num_envs=8)
    ts = ppo.init(env, cfg, jax.random.PRNGKey(0))
    rows = train.final_eval(env, ts, cfg, [(0.0, 1.0), (0.0, 0.0)], num_envs=16)
    assert [(r["dr"], r["throw"]) for r in rows] == [(0.0, 1.0), (0.0, 0.0)]
    assert rows[1]["mean_length"] == 20.0 and rows[1]["terminated_rate"] == 0.0
    assert rows == train.final_eval(env, ts, cfg, [(0.0, 1.0), (0.0, 0.0)], num_envs=16)


class FakeDroneState(ty.NamedTuple):
    sim: tuple  # (world_pos, world_vel)
    target: jax.Array


class FakeDrone:
    """Echoes a fixed state with reward 1, so InTolWeight's bonus is the only change."""

    act_dim, i_pos, i_vel = 4, 0, 1

    def step(self, state, action):
        info = {"terminated": jnp.asarray(action[0] > 0.5)}
        return state, None, None, jnp.float32(1.0), info["terminated"], info


def test_in_tol_weight_bonus():
    import env as E

    env = train.InTolWeight(FakeDrone(), 1.0)
    level = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0])
    target = jnp.array([0.0, 0.0, 2.0])

    def reward(pos, vel, terminated=False):
        state = FakeDroneState((pos, vel), target)
        return float(env.step(state, jnp.full(4, float(terminated)))[3])

    still = jnp.zeros(6)
    assert reward(level, still) == 2.0 - E.IN_TOL_BONUS  # pays 1.0 instead of the env's bonus
    assert reward(level, still, terminated=True) == 1.0
    assert reward(level.at[6].set(2.5), still) == 1.0
    assert reward(level, still.at[3].set(0.5)) == 1.0
    assert (
        float(
            train.InTolWeight(FakeDrone(), E.IN_TOL_BONUS).step(
                FakeDroneState((level, still), target), jnp.zeros(4)
            )[3]
        )
        == 1.0
    )


def head(layers: list[dict]) -> np.ndarray:
    return np.asarray(layers[-1]["w"])


def test_head_gain_options():
    """actor_head_gain / critic_head_gain scale only the orthogonal output layers; the defaults
    (0.01, 1.0) are the previous init."""
    cfg = ppo.PPOConfig()
    assert (cfg.actor_head_gain, cfg.critic_head_gain, cfg.actor_warmup) == (0.01, 1.0, 0)
    env, key = PointMassEnv(), jax.random.PRNGKey(0)
    base = ppo.init(env, ppo.PPOConfig(num_envs=8), key).params
    alt = ppo.PPOConfig(num_envs=8, actor_head_gain=0.5, critic_head_gain=0.01)
    alt = ppo.init(env, alt, key).params
    a0, c0 = head(base["actor"]["mlp"]), head(base["critic"])
    np.testing.assert_allclose(a0.T @ a0, 0.01**2 * np.eye(4), atol=1e-9)
    assert c0.shape == (128, 1) and np.linalg.norm(c0) == pytest.approx(1.0, rel=1e-5)
    np.testing.assert_allclose(head(alt["actor"]["mlp"]), 50.0 * a0, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(head(alt["critic"]), 0.01 * c0, rtol=1e-5, atol=1e-10)
    for net in ("actor", "critic"):
        a, b = base[net], alt[net]
        hidden_a = (a["mlp"] if net == "actor" else a)[:-1]
        hidden_b = (b["mlp"] if net == "actor" else b)[:-1]
        for x, y in zip(jax.tree.leaves(hidden_a), jax.tree.leaves(hidden_b)):
            np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(base["actor"]["log_std"], alt["actor"]["log_std"])
    # a smaller critic head starts the value estimate near 0
    obs_n = nn.normalize(nn.running_stat((env.critic_dim,)), jnp.ones((16, env.critic_dim)))
    assert np.abs(np.asarray(ppo.critic_value(alt["critic"], obs_n))).max() < 0.1


def test_actor_warmup_freezes_actor_only():
    """With actor_warmup N the actor (mlp and log_std) is bit-identical for the first N
    iterations while the critic trains; iteration N + 1 moves the actor."""
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=32, num_steps=16, total_iters=4, actor_warmup=2)
    ts0 = ppo.init(env, cfg, jax.random.PRNGKey(0))
    train_iter = ppo.make_train_iter(env, cfg)
    states, metrics = [ts0], []
    for _ in range(3):
        ts, m = train_iter(states[-1], 1.0)
        states.append(ts)
        metrics.append(jax.device_get(m))

    def leaves(ts, net):
        return [np.asarray(x) for x in jax.tree.leaves(ts.params[net])]

    def same(a, b, net):
        return all(np.array_equal(x, y) for x, y in zip(leaves(a, net), leaves(b, net)))

    assert same(ts0, states[1], "actor") and same(ts0, states[2], "actor")
    assert not same(ts0, states[1], "critic") and not same(states[1], states[2], "critic")
    assert not same(states[2], states[3], "actor")
    lrs = [float(m["lr"]) for m in metrics]
    assert [float(m["actor_lr"]) for m in metrics] == [0.0, 0.0, lrs[2]] and lrs[2] > 0
    for m in metrics[:2]:  # frozen actor: ratio 1, so the KL never stops the critic's epochs
        assert abs(float(m["approx_kl"])) < 1e-6 and int(m["epochs"]) == cfg.epochs
    assert float(metrics[2]["approx_kl"]) > 1e-7
    # without warmup the actor moves in the first iteration
    ts_no, _ = ppo.make_train_iter(env, dataclasses.replace(cfg, actor_warmup=0))(ts0, 1.0)
    assert not same(ts0, ts_no, "actor")
    records = train.train(env, dataclasses.replace(cfg, total_iters=3), seed=0, log=None)[1]
    assert [r["actor_lr"] for r in records][:2] == [0.0, 0.0] and records[2]["actor_lr"] > 0


def test_cli_ppo_options():
    import env as E

    args = train.parse_args([])
    cfg, meta, train_env, eval_env = train.setup(args, PointMassEnv())
    defaults = ppo.PPOConfig()
    for f in ("actor_head_gain", "critic_head_gain", "actor_warmup", "init_log_std", "lr"):
        assert getattr(cfg, f) == getattr(defaults, f), f
    assert meta["in_tol_weight"] == E.IN_TOL_BONUS and meta["residual"] is None
    assert isinstance(train_env, PointMassEnv) and eval_env is train_env
    argv = ["--critic-head-gain", "0.01", "--actor-head-gain", "0.02", "--actor-warmup", "20"]
    argv += ["--init-std", "0.05", "--in-tol-weight", "1.0"]
    cfg, meta, train_env, eval_env = train.setup(train.parse_args(argv), PointMassEnv())
    assert (cfg.critic_head_gain, cfg.actor_head_gain, cfg.actor_warmup) == (0.01, 0.02, 20)
    assert cfg.init_log_std == pytest.approx(math.log(0.05))
    assert isinstance(train_env, train.InTolWeight) and train_env.weight == 1.0
    assert isinstance(eval_env, PointMassEnv)


def check_devices(devices: int):
    """Train and evaluate on 1 and `devices` XLA devices; every result must match bit for bit."""
    assert len(jax.devices()) >= devices
    env = PointMassEnv(episode_steps=40)
    results = []
    for d in (1, devices):
        cfg = ppo.PPOConfig(num_envs=32, num_steps=16, total_iters=3, devices=d)
        ts, records = train.train(env, cfg, seed=3, log=None)
        records = [{k: v for k, v in r.items() if k != "steps_per_s"} for r in records]
        ev = ppo.make_evaluate(env, 32, deterministic=False, devices=d)
        ev = jax.device_get(ev(ts.params["actor"], ts.obs_rms, jax.random.PRNGKey(5), 1.0))
        results.append((jax.device_get(ts), records, {k: float(v) for k, v in ev.items()}))
    (ts1, rec1, ev1), (tsn, recn, evn) = results
    assert rec1 == recn, (rec1, recn)
    assert ev1 == evn, (ev1, evn)
    for a, b in zip(jax.tree.leaves(ts1), jax.tree.leaves(tsn)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_multi_device_matches_single_device():
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4")
    code = "import test_ppo_smoke as t; t.check_devices(4); print('devices ok')"
    here = os.path.dirname(os.path.abspath(__file__))
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=here, env=env, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0 and "devices ok" in res.stdout, res.stderr[-3000:]


@pytest.mark.slow
def test_learns_point_mass():
    env = PointMassEnv()
    cfg = ppo.PPOConfig(num_envs=128, num_steps=32, total_iters=80, init_log_std=TOY_LOG_STD)
    ts0 = ppo.init(env, cfg, jax.random.PRNGKey(0))
    before = evaluate(env, ts0)
    ts, _ = train.train(env, cfg, seed=0, log=None)
    after = evaluate(env, ts)
    print("before", before, "after", after)
    assert after["mean_return"] > before["mean_return"] + 50.0
    assert after["terminated_rate"] <= 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-envs", type=int, default=1024)
    args = parser.parse_args()
    env = PointMassEnv()
    cfg = ppo.PPOConfig(
        num_envs=args.num_envs,
        num_steps=args.steps,
        total_iters=args.iters,
        init_log_std=TOY_LOG_STD,
    )
    ts0 = ppo.init(env, cfg, jax.random.PRNGKey(args.seed))
    before = evaluate(env, ts0, args.eval_envs)
    start = time.perf_counter()
    ts, _ = train.train(env, cfg, seed=args.seed, log=print)
    wall = time.perf_counter() - start
    after = evaluate(env, ts, args.eval_envs)
    summary = {
        "wall_s": round(wall, 1),
        "env_steps": args.iters * args.num_envs * args.steps,
        "before": {k: round(float(v), 3) for k, v in before.items()},
        "after": {k: round(float(v), 3) for k, v in after.items()},
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
