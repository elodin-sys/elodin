import os

import cascade_b0 as B
import env as E
import jax
import jax.numpy as jnp
import nn
import numpy as np
import policy_system as PS
import ppo
import pytest
import train
from conftest import history, references, write_policy
from residual import BASES, ResidualEnv, ResidualState, b0_from_arrays, residual_action

# the residual track is a negative result kept in the fork; its tests run with --runslow
pytestmark = pytest.mark.slow

STEPS = 500
KEYS = np.asarray(jax.random.split(jax.random.PRNGKey(41), 8))


def f64_gains(gains) -> B.B0Gains:
    return B.B0Gains(*[jnp.asarray(v, jnp.float64) for v in gains])


def test_zero_residual_reproduces_b0_rollout(env_mekf6):
    """A zero residual flies B0 bit for bit: every sim slot, reward, done and obs over full
    episodes at (dr 1, throw 1), and the episode stats of cascade_b0.make_episode."""
    env = env_mekf6
    res = ResidualEnv(env, "dr", 0.3)
    gains = f64_gains(B.B0_DR_GAINS)

    def b0_run(key):
        s, o, _ = env.reset(key, 1.0, 1.0)

        def body(carry, _):
            s, o = carry
            s, o, c, r, d, _ = env.step(s, B.b0_action(o, gains))
            return (s, o), (s.sim, o, c, r, d)

        return jax.lax.scan(body, (s, o), None, length=STEPS)[1]

    def res_run(key):
        rs, ro, rc = res.reset(key, 1.0, 1.0)

        def body(carry, _):
            rs, ret, alive, success = carry
            rs, ro, rc, r, d, info = res.step(rs, jnp.zeros(4, jnp.float32))
            ret = ret + jnp.where(alive, r, 0.0)
            success = success | (alive & d & info["success"])
            return (rs, ret, alive & ~d, success), (rs.env.sim, rs.obs_prev, ro, rc, r, d)

        carry = (rs, jnp.float32(0.0), jnp.bool_(True), jnp.bool_(False))
        (_, ret, _, success), out = jax.lax.scan(body, carry, None, length=STEPS)
        return out, ret, success, ro, rc

    b0_run, res_run = jax.jit(b0_run), jax.jit(res_run)
    episode = jax.jit(lambda k: B.make_episode(env, "b0")(k, gains, 1.0, 1.0))
    for key in jax.random.split(jax.random.key(7), 2):
        sim_a, obs_a, crit_a, r_a, d_a = jax.device_get(b0_run(key))
        (sim_b, prev_b, obs_b, crit_b, r_b, d_b), ret, success, obs0, crit0 = jax.device_get(
            res_run(key)
        )
        for x, y in zip(sim_a, sim_b):
            np.testing.assert_array_equal(x, y)
        for x, y in ((r_a, r_b), (d_a, d_b), (obs_a, prev_b), (obs_a, obs_b[:, :19])):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(crit_a, crit_b[:, : E.CRITIC_DIM])
        # B0 normalizes an f32 g_body in f32, so a separately compiled copy differs by f32 ulps
        base = np.asarray(jax.vmap(lambda o: B.b0_action(o, gains))(obs_a), np.float32)
        np.testing.assert_allclose(obs_b[:, 19:], base, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(crit_b[:, E.CRITIC_DIM :], obs_b[:, 19:])
        np.testing.assert_array_equal(obs0[19:], crit0[E.CRITIC_DIM :])
        ep = jax.device_get(episode(key))
        assert float(ep["ret"]) == float(ret) and bool(ep["success"]) == bool(success)
        assert int(ep["length"]) == int(np.argmax(d_b) + 1 if d_b.any() else STEPS)


def test_residual_env_dims_jit_vmap(env_mekf6):
    env = env_mekf6
    with pytest.raises(ValueError):
        ResidualEnv(env, "b1")
    assert BASES == ("dr", "nominal")
    res = ResidualEnv(env, "nominal", 0.5)
    assert (res.obs_dim, res.critic_dim, res.act_dim) == (23, E.CRITIC_DIM + 4, 4)
    assert res.episode_steps == env.episode_steps and res.gains == tuple(B.B0_GAINS)
    assert ResidualEnv(env).gains == tuple(B.B0_DR_GAINS) and ResidualEnv(env).scale == 0.3
    n = 4
    keys = jax.random.split(jax.random.key(11), n)
    a_res = jnp.linspace(-1.0, 1.0, n * 4, dtype=jnp.float32).reshape(n, 4)
    a_res = a_res.at[0].set(4.0)  # saturates every channel

    def one(key, a):
        s, o, c = res.reset(key, 1.0, 1.0)
        executed = res.action(s, a)
        s2, o2, c2, r, d, _ = res.step(s, a)
        return s, o, c, executed, s2, o2, c2, r, d

    batched = jax.device_get(jax.jit(jax.vmap(one))(keys, a_res))
    s, o, c, executed, s2, o2, c2, r, _ = batched
    assert isinstance(s, ResidualState)
    assert o.shape == (n, 23) and c.shape == (n, E.CRITIC_DIM + 4) and o2.shape == (n, 23)
    assert o.dtype == c.dtype == o2.dtype == c2.dtype == np.float32 and r.dtype == np.float32
    np.testing.assert_array_equal(s.obs_prev, o[:, :19])
    np.testing.assert_array_equal(s2.obs_prev, o2[:, :19])
    base = np.asarray(jax.vmap(lambda x: B.b0_action(x, f64_gains(B.B0_GAINS)))(s.obs_prev))
    want = np.clip(base + 0.5 * np.asarray(a_res, np.float64), -1.0, 1.0)
    np.testing.assert_allclose(executed, want, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.abs(executed[0]), 1.0)
    # the executed action is what the env wrote into the action path
    np.testing.assert_allclose(s2.env.sim[env.i_action], executed, rtol=0, atol=1e-12)
    np.testing.assert_allclose(o[:, 19:], base, rtol=0, atol=1e-6)
    # one key unbatched reproduces its row of the batch
    single = jax.device_get(jax.jit(one)(keys[2], a_res[2]))
    for x, y in zip(jax.tree.leaves(single), jax.tree.leaves(batched)):
        np.testing.assert_allclose(
            np.asarray(x, np.float64), np.asarray(y[2], np.float64), rtol=1e-12, atol=1e-9
        )


def test_train_setup_builds_residual(env_mekf6):
    argv = ["--residual", "nominal", "--residual-scale", "0.5", "--in-tol-weight", "1.0"]
    argv += ["--actor-warmup", "20", "--init-std", "0.05"]
    cfg, meta, train_env, eval_env = train.setup(train.parse_args(argv), env_mekf6)
    assert isinstance(train_env, ResidualEnv) and isinstance(eval_env, ResidualEnv)
    assert isinstance(train_env.env, train.InTolWeight) and eval_env.env is env_mekf6
    assert (train_env.base, train_env.scale) == ("nominal", 0.5) and train_env.obs_dim == 23
    assert meta["residual"] == {"base": "nominal", "scale": 0.5, "gains": list(B.B0_GAINS)}
    assert cfg.actor_warmup == 20 and cfg.actor_head_gain == 0.01
    _, meta, train_env, eval_env = train.setup(train.parse_args([]), env_mekf6)
    assert train_env is env_mekf6 and eval_env is env_mekf6 and meta["residual"] is None


def random_obs(key, n: int) -> np.ndarray:
    """Actor obs spanning hover to hard throws, so the cascade's clips and caps engage."""
    k = jax.random.split(key, 6)
    g = jax.random.normal(k[0], (n, 3)) * 0.3 + jnp.array([0.0, 0.0, 1.0])
    g = g / jnp.linalg.norm(g, axis=1, keepdims=True)
    parts = [
        g,
        jax.random.normal(k[1], (n, 3)) * 2.0,
        jax.random.normal(k[2], (n, 3)) * 1.5,
        jax.random.normal(k[3], (n, 3)) * 1.5,
        jax.random.uniform(k[4], (n, 3), minval=-1.4, maxval=1.4),
        jax.random.uniform(k[5], (n, 4), minval=-1.0, maxval=1.0),
    ]
    return np.asarray(jnp.concatenate(parts, 1), np.float64)


def test_residual_action_numpy_matches_jnp_and_env(env_mekf6, tmp_path):
    """residual_action from a saved policy.npz: numpy == jnp, its base controller == cascade_b0,
    and it reproduces the action ResidualEnv executes for the actor's mean."""
    res = ResidualEnv(env_mekf6, "dr", 0.3)
    cfg = ppo.PPOConfig(num_envs=16, actor_head_gain=1.0)  # a residual large enough to see
    ts = ppo.init(res, cfg, jax.random.PRNGKey(3), 1.0, 1.0)
    ts = ts._replace(obs_rms=nn.update_stat(ts.obs_rms, ts.obs))
    ppo.save_policy(str(tmp_path), ts, cfg, {"residual": res.config()}, res.policy_arrays())
    policy = dict(np.load(tmp_path / "policy.npz"))
    assert str(policy["residual_base"]) == "dr" and float(policy["residual_scale"]) == 0.3
    np.testing.assert_array_equal(policy["b0_gains"], np.asarray(B.B0_DR_GAINS, np.float64))
    np.testing.assert_array_equal(policy["b0_thrust_axis"], B.THRUST_AXIS)
    assert policy["actor_w0"].shape == (23, 64)

    obs = random_obs(jax.random.key(5), 4096)
    want = np.asarray(jax.vmap(B.b0_action, in_axes=(0, None))(obs, f64_gains(B.B0_DR_GAINS)))
    got = b0_from_arrays(policy, obs, np)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(b0_from_arrays(policy, jnp.asarray(obs))), got, atol=1e-12
    )
    assert (np.abs(want) == 1.0).mean() > 0.05  # the saturating branches are covered
    a_np = residual_action(policy, obs, np)
    np.testing.assert_allclose(
        np.asarray(residual_action(policy, jnp.asarray(obs))), a_np, atol=1e-10
    )
    obs32 = obs.astype(np.float32)
    a32 = residual_action(policy, obs32, np)
    np.testing.assert_allclose(
        np.asarray(residual_action(policy, jnp.asarray(obs32))), a32, atol=1e-6
    )
    assert a_np.shape == (4096, 4) and np.abs(a_np).max() <= 1.0

    # deploy parity with the training env: the executed action for the clipped actor mean
    mean, _ = ppo.actor_dist(ts.params["actor"], nn.normalize(ts.obs_rms, ts.obs, cfg.obs_clip))
    executed = np.asarray(jax.vmap(res.action)(ts.env_state, jnp.clip(mean, -1.0, 1.0)))
    obs_env = np.asarray(ts.obs[:, :19])
    np.testing.assert_array_equal(np.asarray(ts.env_state.obs_prev), obs_env)
    deployed = residual_action(policy, obs_env, np)
    np.testing.assert_allclose(deployed, executed, rtol=0, atol=2e-6)
    base = np.asarray(ts.obs[:, 19:], np.float64)
    assert np.abs(executed - base).max() > 0.05  # the residual moves the action


@pytest.fixture(scope="module")
def residual_run(tmp_path_factory, env_mekf6):
    res = ResidualEnv(env_mekf6, "dr", 0.3)
    root = tmp_path_factory.mktemp("residual_run")
    key = jax.random.PRNGKey(2)
    return write_policy(str(root / "residual_s0"), key, 23, res.policy_arrays(), res.config())


def test_residual_forward_in_graph(residual_run):
    """policy_system's in-graph forward of a residual policy is residual.residual_action."""
    obs = np.array(jax.random.normal(jax.random.PRNGKey(3), (64, 19)), np.float64)
    obs[:, 2] += 1.0
    fn, needs_critic = PS.forward(PS.load(residual_run))
    assert not needs_critic
    arrays = dict(np.load(os.path.join(residual_run, "policy.npz")))
    want = residual_action(arrays, jnp.asarray(obs), jnp)
    got = jax.vmap(fn, in_axes=(0, None))(jnp.asarray(obs), None)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert np.ptp(np.asarray(want)) > 0.1


def test_residual_cranelift_matches_env(residual_run, env_mekf6):
    """E6 (a) for a residual policy: Cranelift flies the JAX env's episode for 300 ticks."""
    train_pos, deploy_pos = references(residual_run, env_mekf6, 100, KEYS[:1])
    w, system = PS.build_world(residual_run, 1, "mekf6", keys=KEYS[:1])
    pos = history(w, system, "cranelift", 300, ["drone.world_pos"])["drone.world_pos"][3::3]
    assert np.ptp(pos[:, 4:], axis=0).max() > 0.05
    assert np.max(np.abs(pos[:, 4:] - train_pos[:, 0, 4:])) <= 1e-4
    assert np.max(np.abs(pos[:, 4:] - deploy_pos[:, 0, 4:])) <= 1e-9
