import json

import cascade_b0 as B
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from env import CRITIC_DIM, OBS_DIM

# success on EVAL_EPISODES full throws (mekf6, key 7), measured: B0_GAINS on the nominal plant,
# B0_DR_GAINS under full DR
TUNED_SUCCESS = {"nominal": (0.0, 0.984375), "dr": (1.0, 0.6875)}
SUCCESS_MARGIN = 0.1
EVAL_EPISODES = 64


@pytest.fixture(scope="module")
def run_mekf6(env_mekf6):
    return B.Runner(env_mekf6, "b0")


def random_obs(key, n, dim):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n, dim))
    return x * 10.0 ** jax.random.uniform(k2, (n, 1), minval=-3.0, maxval=4.0)


def test_actions_finite_and_bounded():
    k_obs, k_crit = jax.random.split(jax.random.key(0))
    obs = random_obs(k_obs, 4096, OBS_DIM)
    critic = random_obs(k_crit, 4096, CRITIC_DIM)
    edge = np.zeros((6, OBS_DIM))
    edge[1, 2] = -1.0  # upside down
    edge[2, 0] = 1.0  # pitched 90 deg
    edge[3, :3] = B.THRUST_AXIS  # hover attitude (thrust axis vertical) at the target
    edge[4, :] = 1e9
    edge[5, 6:9] = [0.0, 0.0, -1e3]  # far below the target: thrust saturates
    obs = jnp.concatenate([obs, jnp.asarray(edge, obs.dtype)])
    critic = jnp.concatenate([critic, jnp.zeros((6, CRITIC_DIM))])
    gains = B.sample_gains(np.random.default_rng(0), obs.shape[0])
    gains = B.as_batch(gains[:-2] + [B.B0_GAINS, B.B0_DR_GAINS])
    outs = [jax.jit(jax.vmap(B.b0_action))(obs, gains)]
    for g in B.GAIN_SETS.values():
        outs.append(jax.jit(jax.vmap(B.b0_action, in_axes=(0, None)))(obs, g))
        outs.append(jax.jit(jax.vmap(B.b1_action, in_axes=(0, 0, None)))(obs, critic, g))
        # hover attitude at the target: hover collective, no rate command
        np.testing.assert_allclose(B.b0_action(obs[-3], g), 0.0, atol=1e-6)
    for a in outs:
        a = np.asarray(a)
        assert a.shape == (obs.shape[0], 4)
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
    # B1 is B0 on the critic's truth fields
    mixed = jnp.concatenate([critic[0, 19:31], obs[0, 12:19]])
    np.testing.assert_array_equal(B.b1_action(obs[0], critic[0]), B.b0_action(mixed))


def test_trim_start_is_hover(env_mekf6):
    """At throw 0 the env starts in the hover trim: B0 reads g_body = THRUST_AXIS and, without
    the gyro bias, commands a = 0."""
    _, obs, _ = env_mekf6.reset(jax.random.key(0), 0.0, 0.0)
    np.testing.assert_allclose(obs[0:3], B.THRUST_AXIS, atol=1e-6)
    obs = obs.at[3:6].set(0.0)
    for g in B.GAIN_SETS.values():
        np.testing.assert_allclose(B.b0_action(obs, g), 0.0, atol=1e-5)


@pytest.mark.parametrize("name", list(B.GAIN_SETS))
def test_nominal_hover_holds(run_mekf6, name):
    keys = jax.random.split(jax.random.key(3), 16)
    o = run_mekf6(keys, B.as_batch(B.GAIN_SETS[name], 16), 0.0, 0.0)
    assert not o["terminated"].any() and o["success"].all()
    assert o["e_p_max"].max() < 0.05, o["e_p_max"]


@pytest.mark.parametrize("name", list(TUNED_SUCCESS))
def test_tuned_success_on_fixed_set(run_mekf6, name):
    dr, tuned = TUNED_SUCCESS[name]
    keys = jax.random.split(jax.random.key(7), EVAL_EPISODES)
    s = B.summarize(run_mekf6(keys, B.as_batch(B.GAIN_SETS[name], EVAL_EPISODES), dr, 1.0))
    assert s["success"] >= tuned - SUCCESS_MARGIN, s


def test_select_reads_validated_leaders(tmp_path):
    """select re-scores only the validated leaders of a tune log, rounded to 4 significant
    figures; the shipped sets are in that form."""
    g = B.T1_SETS["t1-nominal"]._replace(kp_xy=3.14159265)
    recs = [
        {"stage": "random", "sets": 4},
        {"set": 3, "gains": g._asdict()},
        {"validate": 0, "set": 3, "gains": g._asdict()},
        {"winner": 3, "gains": g._asdict()},
    ]
    path = tmp_path / "tune.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert B.leaders(str(path)) == [("tune.jsonl:3", g)]
    assert B.round_gains(g) == g._replace(kp_xy=3.142)
    for s in [*B.GAIN_SETS.values(), *B.T1_SETS.values()]:
        assert B.round_gains(s) == s
    assert B.parse_gains("t1-dr") == B.T1_SETS["t1-dr"]
