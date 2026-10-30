import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "drone"))
sys.path.insert(0, os.path.join(HERE, ".."))

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        help="also run the tests marked slow (eval-suite harness, long training and parity runs)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running; skipped unless --runslow is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def env_truth():
    import env

    return env.DroneEnv("truth")


@pytest.fixture(scope="session")
def env_truth_stock():
    import env

    return env.DroneEnv("truth", rate_pid="stock")


@pytest.fixture(scope="session")
def env_mekf18():
    import env

    return env.DroneEnv("mekf18")


@pytest.fixture(scope="session")
def env_mekf6():
    import env

    return env.DroneEnv("mekf6")


# ------------------------------------------------------------------ shared helpers


def write_policy(path, key, obs_dim=19, extra=None, res=None, estimator="mekf6") -> str:
    """A random run directory: policy.npz (head gain 0.3, so the action moves), config.json,
    done and a short log.jsonl."""
    import jax
    import jax.numpy as jnp
    import nn
    import numpy as np

    actor = nn.init_actor(key, obs_dim, 4, (64, 64), head_gain=0.3)
    k1, k2 = jax.random.split(jax.random.fold_in(key, 1))
    mean = (0.1 * jax.random.normal(k1, (obs_dim,))).astype(jnp.float32)
    var = jnp.exp(0.3 * jax.random.normal(k2, (obs_dim,))).astype(jnp.float32)
    arrays = nn.policy_arrays(actor, nn.RunningStat(mean, var, jnp.float32(1.0)), 5.0)
    arrays.update(extra or {})
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "policy.npz"), **arrays)
    config = {"ppo": {"obs_clip": 5.0, "total_iters": 2, "epochs": 4}, "estimator": estimator}
    config["residual"] = res
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(path, "done"), "w") as f:
        f.write("0\n")
    with open(os.path.join(path, "log.jsonl"), "w") as f:
        rec = {"steps_per_s": 1000.0, "eval_success_rate": 0.1}
        f.writelines(json.dumps({"iter": it, **rec}) + "\n" for it in (0, 1))
    return path


def _innermost(x):
    while hasattr(x, "env"):
        x = x.env
    return x


def env_positions(env, act, steps: int):
    """-> jitted pos(params, keys): world_pos [steps, n, 7] after every policy step of the env's
    episodes at dr = throw = 1, with act(params, obs, critic, sim) -> actions [n, 4]."""
    import jax

    plant = _innermost(env)
    v_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    v_step = jax.vmap(env.step)

    @jax.jit
    def pos(params, keys):
        state, obs, critic = v_reset(keys, 1.0, 1.0)

        def step(carry, _):
            state, obs, critic = carry
            a = act(params, obs, critic, _innermost(state).sim)
            state, obs, critic, *_ = v_step(state, a)
            return (state, obs, critic), _innermost(state).sim[plant.i_pos]

        return jax.lax.scan(step, (state, obs, critic), None, steps)[1]

    return pos


def references(controller, env, steps: int, keys):
    """JAX env trajectories (world_pos [steps, n, 7]) of the episodes of `keys` for a run
    directory or baseline id: (training-path float32 forward, in-graph float64 forward)."""
    import cascade_b0 as B
    import jax
    import jax.numpy as jnp
    import nn
    import numpy as np
    import policy_system as PS
    import ppo

    policy = PS.load(controller)
    if controller in PS.BASELINES:
        kind = PS.BASELINES[controller][0]
        gains = B.GAIN_SETS[PS.BASELINES[controller][1]]
        params = deploy_params = B.B0Gains(*[jnp.asarray(v, jnp.float64) for v in gains])
        train_env = env
    else:
        kind = "residual" if "residual_base" in policy else "policy"
        actor, stat, _, _ = nn.load_policy(os.path.join(controller, "policy.npz"))
        params = (actor, stat)
        deploy_params = {
            k: jnp.asarray(v)
            for k, v in policy.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "fiu"
        }
        train_env = env
        if kind == "residual":
            import residual

            train_env = residual.ResidualEnv(env, "dr", 0.3)

    def train_act(p, obs, critic, sim):
        if kind in ("policy", "residual"):
            mean, _ = ppo.actor_dist(p[0], nn.normalize(p[1], obs, 5.0))
            return jnp.clip(mean, -1.0, 1.0)
        if kind == "b0":
            return jax.vmap(lambda o: B.b0_action(o, p))(obs)
        return jax.vmap(lambda o, c: B.b1_action(o, c, p))(obs, critic)

    def deploy_act(p, obs, critic, sim):
        o, c = sim[env.i_obs], sim[env.i_critic]
        if kind == "b0":
            return jax.vmap(lambda x: B.b0_action(x, p))(o)
        if kind == "b1":
            return jax.vmap(lambda x, y: B.b1_action(x, y, p))(o, c)
        if kind == "residual":
            import residual

            return residual.residual_action(p, o, jnp)
        return nn.policy_action(p, o, jnp)

    keys = jnp.asarray(keys)
    train = env_positions(train_env, train_act, steps)(params, keys)
    deploy = env_positions(env, deploy_act, steps)(deploy_params, keys)
    return np.asarray(train), np.asarray(deploy)


def history(world, system, backend, ticks, names) -> dict:
    """Per-tick values of the named components over `ticks` ticks of an Exec."""
    import numpy as np

    ex = world.build(system, simulation_rate=300.0, backend=backend)
    ex.run(ticks, show_progress=False)
    df = ex.history(names)
    return {n: np.stack(df[n].to_numpy()) for n in names}
