import json
import os
import subprocess
import sys
from types import SimpleNamespace

import cascade_b0 as B
import dr as D
import elodin as el
import env as E
import evaluate as EV
import jax
import jax.numpy as jnp
import nn
import numpy as np
import plant_dr
import ppo
import pytest
import residual
from conftest import write_policy
from eval_runs import load_policy
from jaxstep import JaxStep
from policy_world import EFFECTS, VIO_LONG_LEN

# the evaluation-suite harness stays in the fork; its tests run with --runslow
pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 16


def heldout(n, seed=5):
    k_reset, _ = jax.random.split(jax.random.PRNGKey(seed))
    return np.asarray(jax.random.split(k_reset, n))


@pytest.fixture(scope="session")
def eval_env_truth():
    return EV.DelayEnv(E.DroneEnv("truth", effects=EFFECTS, trace=("motor_pwm",)))


@pytest.fixture(scope="session")
def policies(tmp_path_factory, env_truth):
    root = tmp_path_factory.mktemp("eval_runs")
    res = residual.ResidualEnv(env_truth, "dr", 0.3)
    return {
        "scratch": write_policy(str(root / "p_s0"), jax.random.PRNGKey(11), estimator="truth"),
        "residual": write_policy(
            str(root / "r_s0"), jax.random.PRNGKey(12), 23, res.policy_arrays(), res.config()
        ),
    }


@pytest.mark.parametrize("which", ["scratch", "residual"])
def test_rollout_matches_make_evaluate(which, policies, env_truth):
    """The suite's per-episode rollout is ppo.make_evaluate's: same episodes and actions."""
    actor, stat, _, _ = load_policy(os.path.join(policies[which], "policy.npz"))
    env = residual.ResidualEnv(env_truth, "dr", 0.3) if which == "residual" else env_truth
    run = EV.make_episodes(env, "policy", 5.0, "plain", 1, 0)
    ep, _ = run((actor, stat), heldout(N), 1.0, 1.0, EV.Override())
    want = jax.device_get(
        ppo.make_evaluate(env, N, obs_clip=5.0)(actor, stat, jax.random.PRNGKey(5), 1.0, 1.0)
    )
    assert ep["success"].mean() == pytest.approx(float(want["success_rate"]), abs=1e-7)
    assert ep["terminated"].mean() == pytest.approx(float(want["terminated_rate"]), abs=1e-7)
    assert ep["length"].mean() == pytest.approx(float(want["mean_length"]), rel=1e-6)
    assert ep["ret"].mean() == pytest.approx(float(want["mean_return"]), rel=1e-5)
    assert 0 < ep["length"].min() and ep["terminated"].any()
    assert np.all((ep["cause"] >= 0) == ep["terminated"])


def test_baseline_rows_match_cascade_episode(env_truth):
    """B0 rows (outcomes, causes, hover sums) are cascade_b0.make_episode's."""
    gains = B.B0Gains(*[jnp.asarray(v, jnp.float64) for v in B.B0_DR_GAINS])
    run = EV.make_episodes(env_truth, "b0", 5.0, "plain", 1, 0)
    keys = heldout(N)
    ep, _ = run(gains, keys, 1.0, 1.0, EV.Override())
    want = jax.device_get(
        jax.jit(jax.vmap(B.make_episode(env_truth, "b0"), in_axes=(0, None, None, None)))(
            jnp.asarray(keys), gains, 1.0, 1.0
        )
    )
    for k in ("success", "terminated", "length", "n_win"):
        np.testing.assert_array_equal(ep[k], want[k], err_msg=k)
    first = np.where(want["cause"].any(-1), np.argmax(want["cause"], -1), -1)
    np.testing.assert_array_equal(ep["cause"], np.where(want["terminated"], first, -1))
    for k in ("se_p", "se_tilt", "se_axis"):
        np.testing.assert_allclose(ep[k], want[k], rtol=1e-9, atol=1e-12, err_msg=k)


def test_eval_env_is_the_training_env(policies, env_truth, eval_env_truth):
    """With nothing overridden the eval env (long VIO buffer, gyroscopic gain 0, DelayEnv,
    motor_pwm trace) flies the training env bit for bit; it counts every pwm difference."""
    actor, stat, _, _ = load_policy(os.path.join(policies["scratch"], "policy.npz"))
    keys = heldout(N)
    a, _ = EV.make_episodes(env_truth, "policy", 5.0, "plain", 1, 0)(
        (actor, stat), keys, 1.0, 1.0, EV.Override()
    )
    b, (obs, alive) = EV.make_episodes(eval_env_truth, "policy", 5.0, "eval", 1, 4)(
        (actor, stat), keys, 1.0, 1.0, EV.Override()
    )
    for k in ("ret", "length", "success", "terminated", "cause", "se_p", "da2", "n_da"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    np.testing.assert_array_equal(b["n_pwm"], (3 * b["length"] - 1) * 4)
    assert np.all(b["pwm_d2"] > 0) and np.all(a["n_pwm"] == 0)
    assert obs.shape == (500, 4, 19) and alive.shape == (500, 4)
    np.testing.assert_array_equal(alive.sum(0), b["length"][:4])


def _open_loop(env, state, steps=40):
    step = jax.jit(env.step)
    out = []
    for i in range(steps):
        a = 0.3 * jnp.sin(jnp.arange(4.0) + 0.37 * i)
        state, obs, _, _, _, info = step(state, a)
        out.append((state, np.asarray(obs), info))
    return out


def test_delay_env_extends_the_fifo(env_truth):
    """DelayEnv: d <= 2 ticks is the plain env, d = 3 flies the graph's own 3-tick delay bit for
    bit (positions and the policy's obs), d = 6 executes the action of 2 policy steps ago."""
    delay = EV.DelayEnv(env_truth)
    key = jax.random.PRNGKey(31)
    dr0 = D.dr_sample(jax.random.PRNGKey(32), 1.0)
    throw = D.throw_sample(jax.random.PRNGKey(33), 1.0)
    for d, same in ((2, 2), (3, 3)):
        dr = dr0._replace(act_delay=jnp.int32(d))
        a = _open_loop(
            env_truth, env_truth.reset_with(key, dr._replace(act_delay=jnp.int32(same)), throw)[0]
        )
        b = _open_loop(delay, delay.reset_with(key, dr, throw)[0])
        for (sa, oa, _), (sb, ob, _) in zip(a, b):
            np.testing.assert_array_equal(sa.sim[env_truth.i_pos], sb.env.sim[env_truth.i_pos])
            np.testing.assert_array_equal(oa, ob)
    state, _, _ = delay.reset_with(key, dr0._replace(act_delay=jnp.int32(6)), throw)
    assert int(state.k) == 2 and int(state.env.sim[env_truth.slot("act_delay")]) == 0
    out = _open_loop(delay, state, 6)
    for i, (s, obs, _) in enumerate(out):
        want = 0.3 * np.sin(np.arange(4.0) + 0.37 * (i - 2)) if i >= 2 else np.zeros(4)
        np.testing.assert_allclose(s.env.sim[env_truth.i_action], want, atol=1e-15)
        np.testing.assert_allclose(obs[15:19], 0.3 * np.sin(np.arange(4.0) + 0.37 * i), atol=1e-7)


def test_override_dr():
    dr = D.dr_sample(jax.random.PRNGKey(7), 1.0)
    same = EV.override_dr(dr, EV.Override(*(jnp.asarray(v) for v in EV.Override())))
    for a, b in zip(dr, same):
        np.testing.assert_array_equal(a, b)
    ov = EV.Override(4, 12, 0.3, 0.15, 2.5, 4.0, 1.0)
    got = EV.override_dr(dr, EV.Override(*(jnp.asarray(v) for v in ov)))
    assert int(got.act_delay) == 4 and int(got.vio_delay) == 12
    assert float(got.motor_tau) == 0.3 and float(got.kq_scale) == 2.5
    np.testing.assert_allclose(got.thrust_scale - 1.0, 3.0 * (dr.thrust_scale - 1.0), atol=1e-15)
    assert np.all(np.abs(got.thrust_scale - 1.0) <= 0.15)
    assert float(jnp.hypot(got.wind[0], got.wind[1])) == pytest.approx(4.0, abs=1e-12)
    np.testing.assert_allclose(
        jnp.arctan2(got.wind[1], got.wind[0]), jnp.arctan2(dr.wind[1], dr.wind[0]), atol=1e-12
    )
    assert float(got.wind[2]) == float(dr.wind[2]) and float(got.mass_scale) == float(dr.mass_scale)


def test_long_vio_buffer_delays():
    """effects=("vio_long",) reads the VIO 12 ticks late: the same open-loop flight's meas at
    tick t is the 0-delay meas of tick t - 12."""
    env = E.DroneEnv("truth", effects=("vio_long",), trace=("vio_meas",))
    key = jax.random.PRNGKey(5)
    dr = D.dr_sample(jax.random.PRNGKey(6), 1.0)
    throw = D.throw_sample(jax.random.PRNGKey(7), 1.0)
    meas = {}
    for d in (0, 12):
        state, _, _ = env.reset_with(key, dr._replace(vio_delay=jnp.int32(d)), throw)
        out = _open_loop(env, state, 20)
        meas[d] = np.concatenate([np.asarray(info["trace"]["vio_meas"]) for _, _, info in out])
    np.testing.assert_array_equal(meas[12][12:], meas[0][:-12])
    assert np.ptp(meas[0][:, :3], axis=0).max() > 0.01


def test_gyroscopic_effector():
    """plant_dr.gyroscopic adds gain x -w x (I w) (body frame): the angular acceleration of a
    torque-free asymmetric body follows Euler's equations and its world angular momentum is
    conserved; with gain 0 six_dof keeps the world rate fixed and the momentum turns."""
    q = np.asarray(D.quat_from_rotvec(jnp.array([0.3, -0.2, 0.5])))
    w_body = np.array([2.0, -1.0, 3.0])
    w_world = np.asarray(D.quat_rotate(jnp.asarray(q), jnp.asarray(w_body)))
    inertia = np.array([0.02, 0.03, 0.05])
    world = el.World()
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(angular=el.Quaternion(jnp.asarray(q))),
                world_vel=el.SpatialMotion(angular=jnp.asarray(w_world)),
                inertia=el.SpatialInertia(mass=1.0, inertia=jnp.asarray(inertia)),
            ),
            plant_dr.Gyroscopic(),
        ],
        name="b",
    )
    system = el.six_dof(1.0 / 900.0, plant_dr.gyroscopic, integrator=el.Integrator.SemiImplicit)
    js = JaxStep(world, system, 900.0)
    alpha = -np.cross(w_body, inertia * w_body) / inertia
    s = js.step(js.init)
    want = np.asarray(D.quat_rotate(jnp.asarray(q), jnp.asarray(alpha)))
    np.testing.assert_allclose(np.asarray(js.get(s, "world_accel"))[:3], want, rtol=1e-12)

    def momentum(state):
        qq = jnp.asarray(js.get(state, "world_pos")[:4])
        wb = D.quat_rotate(D.quat_conj(qq), jnp.asarray(js.get(state, "world_vel")[:3]))
        return np.asarray(D.quat_rotate(qq, inertia * wb))

    run = jax.jit(lambda st: js.run(st, 900))
    l0 = momentum(js.init)
    on, off = run(js.init), run(js.set(js.init, "gyro_gain", 0.0))
    assert np.linalg.norm(momentum(on) - l0) / np.linalg.norm(l0) < 1e-3
    assert np.linalg.norm(js.get(on, "world_vel")[:3] - w_world) > 0.05
    assert np.linalg.norm(momentum(off) - l0) / np.linalg.norm(l0) > 0.1
    np.testing.assert_array_equal(js.get(off, "world_vel")[:3], w_world)


def test_discover_and_track_a_rows(tmp_path):
    """Controllers: finished runs only (done marker 0, else a final_eval.json), then baselines;
    Track A's cross-estimator rows are reused only on the same protocol."""
    for name, done in (("a_s0", "0"), ("b_s0", "1"), ("c_s0", None), ("d_s0", None)):
        write_policy(str(tmp_path / name), jax.random.PRNGKey(0))
        if done is None:
            os.remove(tmp_path / name / "done")
        else:
            (tmp_path / name / "done").write_text(done)
    (tmp_path / "d_s0" / "final_eval.json").write_text("{}")
    ids = [c["id"] for c in EV.discover([str(tmp_path)], ["b0-dr"], "mekf6")]
    assert ids == ["b0-dr", "a_s0", "d_s0"]
    only = EV.discover([str(tmp_path)], ["b0-dr", "b1-dr"], "mekf6", ["b1-*", "d_*"])
    assert [c["id"] for c in only] == ["b1-dr", "d_s0"]
    row = {"run": "S-A2_s0@mekf6", "checkpoint": "S-A2_s0", "estimator": "truth"}
    row.update(eval_estimator="mekf6", final_eval=[{"dr": 1.0}])
    own = {"run": "S-A2_s0", "estimator": "truth", "eval_estimator": "truth", "final_eval": []}
    pairs = [list(p) for p in EV.SCALES]
    for name, seed in (("a.json", 20011), ("b.json", 1)):
        summary = {"episodes": 4096, "seed": seed, "pairs": pairs, "runs": [row, own]}
        (tmp_path / name).write_text(json.dumps(summary))
    got = EV.track_a_rows([str(tmp_path / "a.json")], 4096, 20011)
    assert got == {("S-A2_s0", "mekf6"): [{"dr": 1.0}]}
    assert EV.track_a_rows([str(tmp_path / "b.json")], 4096, 20011) == {}


def test_quick_suite_end_to_end(tmp_path):
    """evaluate.py all --quick over one policy and B0 DR writes every section and plot."""
    runs = tmp_path / "runs"
    write_policy(str(runs / "q_s0"), jax.random.PRNGKey(21))
    out = tmp_path / "results"
    cmd = [sys.executable, "evaluate.py", "all", "--quick", "--devices", "1"]
    cmd += ["--runs", str(runs), "--out", str(out), "--baselines", "b0-dr"]
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    res = subprocess.run(cmd, check=False, cwd=HERE, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    with open(out / "results.json") as f:
        r = json.load(f)
    for section in EV.SECTIONS:
        assert section in r, section
    assert set(r["e1"]["rows"]) == {"q_s0", "b0-dr"}
    assert r["e1"]["rows"]["q_s0"][0]["episodes"] == EV.QUICK.heldout
    assert [len(v["values"]) for v in r["e2"]["slices"].values()] == [2, 2, 2, 2]
    assert set(r["e4"]["rows"]["q_s0"]) == {"in_distribution", *EV.E4_EFFECTS}
    assert r["e5"]["rows"][0]["eval_estimator"] == "truth"
    assert r["e8"]["rows"]["q_s0"]["max_abs_diff_f64"] <= 1e-9
    for row in r["e6"]["a"].values():
        assert row["max_dpos_vs_jax_env_m"] <= 1e-4
    assert r["e6"]["chosen"] == "q_s0"
    assert r["e6"]["b"]["cranelift_vs_jax"]["success_agree"] == 1.0
    assert r["e6"]["c"]["world_vs_vmapped_env"]["terminated_agree"] == 1.0
    assert r["e6"]["bench"]["exit"] == 0
    assert r["e10"]["rollout"]["env_steps_per_s"] > 0
    for name in ("heldout_bars.png", "e2_ood.png", "e4_unseen.png", "learning_curves.png"):
        assert (out / name).stat().st_size > 1000, name
    assert (out / "summary.md").exists()
    with np.load(out / "heldout_episodes.npz") as ep:
        assert ep["q_s0|1,1|success"].shape == (EV.QUICK.heldout,)


def test_numpy_export_matches_forward(policies):
    """evaluate.NumpyPolicy reproduces nn.policy_action / residual.residual_action, and its
    numpy B0 is residual.b0_from_arrays."""
    obs = np.array(jax.random.normal(jax.random.PRNGKey(3), (64, 19)), np.float64)
    obs[:, 2] += 1.0  # g_body near +z
    for kind, path in policies.items():
        arrays = dict(np.load(os.path.join(path, "policy.npz")))
        forward = residual.residual_action if kind == "residual" else nn.policy_action
        want = np.asarray(forward(arrays, jnp.asarray(obs), jnp))
        np.testing.assert_allclose(EV.NumpyPolicy(arrays)(obs), want, rtol=0, atol=1e-12)
        assert np.ptp(want) > 0.1
        if kind == "residual":
            np.testing.assert_array_equal(
                EV.b0_numpy(arrays, obs), residual.b0_from_arrays(arrays, obs, np)
            )


def test_numpy_export_imports_numpy_only(policies):
    """evaluate.NumpyPolicy runs from policy.npz with neither jax nor elodin imported."""
    code = (
        "import sys, numpy as np, evaluate\n"
        f"p = evaluate.NumpyPolicy(np.load({os.path.join(policies['residual'], 'policy.npz')!r}))\n"
        "a = p(np.r_[0, 0, 1.0, np.zeros(16)])\n"
        "assert a.shape == (4,) and np.all(np.abs(a) <= 1)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib', 'elodin')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_best_policy_is_best_seed_of_best_multi_seed_config():
    """The e6 default (headline rule): the config with the best mean among those with 2 or more
    seeds, then its best seed; the best run when no config has 2 seeds."""
    rates = {"A_s0": 0.9, "B_s0": 0.8, "B_s1": 0.85, "C_s0": 0.7, "C_s1": 0.9}
    rows = {i: [{"success_rate": r}] for i, r in rates.items()}
    pols = [{"id": i, "group": i.split("_")[0]} for i in rates]
    s = SimpleNamespace(results={"e1": {"rows": rows}}, policies=pols)
    assert EV._best_policy(s) == "B_s1"  # B mean 0.825 beats C 0.8; A has one seed
    s.policies = pols[:1]
    assert EV._best_policy(s) == "A_s0"


def test_seed_statistics_and_constants():
    """seed_mean_ci is the mean, the SD over seeds and the t 95% interval of the mean;
    the long VIO buffer length matches policy_world."""
    mean, sd, ci = EV.seed_mean_ci([3446 / 4096, 3285 / 4096, 3452 / 4096])
    assert round(mean, 4) == 0.8287 and round(sd, 4) == 0.0231
    assert [round(x, 4) for x in ci] == [0.7712, 0.8861]
    assert EV.seed_mean_ci([0.5])[1:] == (None, None)
    assert EV.VIO_LONG_LEN == VIO_LONG_LEN
