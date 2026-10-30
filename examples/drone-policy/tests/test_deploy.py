import json
import os
import shutil
import subprocess
import sys

import cascade_b0 as B
import jax
import jax.numpy as jnp
import nn
import numpy as np
import policy_system as PS
import pytest
from conftest import history, references, write_policy
from jaxstep import JaxStep

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = np.asarray(jax.random.split(jax.random.PRNGKey(41), 8))


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("deploy_runs")
    return {"scratch": write_policy(str(root / "scratch_s0"), jax.random.PRNGKey(1))}


def _obs(n=64):
    obs = np.array(jax.random.normal(jax.random.PRNGKey(3), (n, 19)), np.float64)
    obs[:, 2] += 1.0  # g_body near +z
    return obs


def test_forward_matches_deploy_paths(runs):
    """policy_system's in-graph forward is nn.policy_action, or the cascade, in float64."""
    obs = _obs()
    critic = np.asarray(jax.random.normal(jax.random.PRNGKey(4), (64, 61)), np.float64)
    policy = PS.load(runs["scratch"])
    assert policy["estimator"] == "mekf6"
    fn, needs_critic = PS.forward(policy)
    assert not needs_critic
    arrays = dict(np.load(os.path.join(runs["scratch"], "policy.npz")))
    want = nn.policy_action(arrays, jnp.asarray(obs), jnp)
    got = jax.vmap(fn, in_axes=(0, None))(jnp.asarray(obs), None)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert np.ptp(np.asarray(want)) > 0.1
    fn, needs_critic = PS.forward(PS.load("b1-dr"))
    assert needs_critic
    want = jax.vmap(lambda o, c: B.b1_action(o, c, B.B0_DR_GAINS))(obs, critic)
    np.testing.assert_array_equal(jax.vmap(fn)(jnp.asarray(obs), jnp.asarray(critic)), want)


def test_load_reads_config_next_to_policy(runs, tmp_path):
    """load() takes the training estimator from config.json, or policy_config.json, next to
    policy.npz, for a directory or the file itself; without either it has no estimator."""
    for name in ("config.json", "policy_config.json"):
        d = tmp_path / name.split(".")[0]
        d.mkdir()
        shutil.copy(os.path.join(runs["scratch"], "policy.npz"), d / "policy.npz")
        with open(d / name, "w") as f:
            json.dump({"estimator": "truth"}, f)
        assert PS.load(str(d))["estimator"] == "truth"
        assert PS.load(str(d / "policy.npz"))["estimator"] == "truth"
    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(os.path.join(runs["scratch"], "policy.npz"), bare / "policy.npz")
    assert "estimator" not in PS.load(str(bare))


def test_shipped_policy():
    """policy/ is the shipped policy (8000 iterations on mekf6): load() reads its estimator, its
    actions are finite and in [-1, 1], and its world flies the same 300 ticks on Cranelift and
    jax-cpu."""
    policy = PS.load(PS.SHIPPED)
    assert policy["estimator"] == "mekf6"
    with open(os.path.join(PS.SHIPPED, "config.json")) as f:
        config = json.load(f)
    assert config["ppo"]["total_iters"] == 8000 and config["obs_dim"] == 19
    fn, _ = PS.forward(policy)
    a = np.asarray(jax.vmap(fn, in_axes=(0, None))(jnp.asarray(_obs()), None))
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0) and np.ptp(a) > 0.01
    names = ["drone.world_pos", "drone.policy_action"]
    h = {}
    for backend in ("cranelift", "jax-cpu"):
        w, system = PS.build_world(PS.SHIPPED, 1, keys=KEYS[:1])
        h[backend] = history(w, system, backend, 300, names)
    pos = h["cranelift"]["drone.world_pos"]
    assert np.ptp(pos[:, 4:], axis=0).max() > 0.05  # the throw moves the drone
    for name in names:
        np.testing.assert_allclose(h["cranelift"][name], h["jax-cpu"][name], rtol=0, atol=1e-9)


def test_world_starts_as_env_reset(runs, env_mekf6):
    """build_world spawns every drone in the state env.reset gives the same key."""
    w, system = PS.build_world(runs["scratch"], 1, keys=KEYS[:1])
    js = JaxStep(w, system, 300.0)
    state, _, _ = env_mekf6.reset(jnp.asarray(KEYS[0]), 1.0, 1.0)
    assert set(env_mekf6.js.names) < set(js.names) and js.has("policy_phase")
    for name in env_mekf6.js.names:
        np.testing.assert_array_equal(
            np.asarray(js.get(js.init, name)), np.asarray(state.sim[env_mekf6.slot(name)]), name
        )


@pytest.mark.parametrize(
    "which,estimator", [("scratch", "mekf6"), ("scratch", "truth"), ("b1-dr", "mekf6")]
)
def test_cranelift_matches_env(which, estimator, runs, request):
    """E6 (a): the controller in graph on Cranelift flies the JAX env's episode for 300 ticks
    (max |dpos| <= 1e-4 m against the training-path float32 forward, 1e-9 m against the same
    float64 forward), acting on the first tick of every 3 and holding the action in between.
    The truth case guards the policy hold against the Cranelift rank-0 select miscompile
    (policy_system docstring), which moved truth worlds by up to 1.6 m in 300 ticks."""
    controller = runs.get(which, which)
    env = request.getfixturevalue(f"env_{estimator}")
    train, deploy = references(controller, env, 100, KEYS[:1])
    w, system = PS.build_world(controller, 1, estimator, keys=KEYS[:1])
    names = ["drone.world_pos", "drone.policy_action", "drone.policy_phase"]
    h = history(w, system, "cranelift", 300, names)
    pos = h["drone.world_pos"][3::3]
    assert np.ptp(pos[:, 4:], axis=0).max() > 0.05  # the throw moves the drone
    assert np.max(np.abs(pos[:, 4:] - train[:, 0, 4:])) <= 1e-4
    assert np.max(np.abs(pos[:, 4:] - deploy[:, 0, 4:])) <= 1e-9
    np.testing.assert_array_equal(h["drone.policy_phase"][:7], [0, 1, 2, 0, 1, 2, 0])
    a = h["drone.policy_action"]
    changed = np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1
    assert len(changed) > 50 and np.all(changed % 3 == 1)
    assert np.all(np.any(a[1:] != a[:-1], axis=0))  # every action lane updates


def test_multi_drone_world_matches_vmapped_env(runs, env_mekf6):
    """E6 (c) in small: 4 drones in one jax-cpu world fly the 4 vmapped env episodes."""
    train, deploy = references(runs["scratch"], env_mekf6, 30, KEYS[:4])
    w, system = PS.build_world(runs["scratch"], 4, "mekf6", keys=KEYS[:4])
    names = [f"drone_{i}.world_pos" for i in range(4)]
    h = history(w, system, "jax-cpu", 90, names)
    for i, name in enumerate(names):
        pos = h[name][3::3, 4:]
        assert np.max(np.abs(pos - train[:, i, 4:])) <= 1e-4
        assert np.max(np.abs(pos - deploy[:, i, 4:])) <= 1e-9


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Cranelift multi-entity miscompile in the stock chain (main.py uses jax-cpu for N > 1)",
    strict=False,
)
def test_multi_drone_world_on_cranelift(runs, env_mekf6):
    """2 drones in one Cranelift world fly the 2 vmapped env episodes (shows when the backend
    multi-entity issue is fixed)."""
    _, deploy = references(runs["scratch"], env_mekf6, 30, KEYS[:2])
    w, system = PS.build_world(runs["scratch"], 2, "mekf6", keys=KEYS[:2])
    names = [f"drone_{i}.world_pos" for i in range(2)]
    h = history(w, system, "cranelift", 90, names)
    for i, name in enumerate(names):
        assert np.max(np.abs(h[name][3::3, 4:] - deploy[:, i, 4:])) <= 1e-9


def test_main_bench_runs():
    """main.py builds N thrown drones flying the shipped policy (its default) and runs elodin's
    bench."""
    env = {**os.environ, "DRONE_N": "2"}
    env.pop("XLA_FLAGS", None)
    env.pop("DRONE_POLICY", None)
    out = subprocess.run(
        [sys.executable, "main.py", "bench", "--ticks", "9"],
        check=False,
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "tick time" in out.stdout + out.stderr


def test_schematic_has_every_drone():
    kdl = PS.build_schematic(3).emit_kdl()
    assert all(f"drone_{i}.world_pos" in kdl for i in range(3))
