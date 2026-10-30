"""PPO with an asymmetric critic, in pure jax.numpy.

Each iteration is two jitted calls: collect (rollout as lax.scan over T of vmap over B, with
auto-reset) and learn (reward scaling, GAE and the clipped-objective update). With
cfg.devices > 1, collect runs under shard_map with the env batch split over a 1-D mesh of XLA
host devices; the trajectory is then gathered on the first device and learn runs there once, so a
run is bitwise identical to a single-device run with the same seed (evaluation reduces on the first
device too). Networks stay float32 even though elodin enables x64.
Env contract (env.py): reset(key, dr_scale, throw_scale) -> (state, obs, critic_obs) and
step(state, action) -> (state, obs, critic_obs, reward, done, info).
"""

import dataclasses
import json
import os
import typing as ty

import jax
import jax.numpy as jnp
import nn
import numpy as np
from jax.sharding import NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P

F32 = jnp.float32
METRIC_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")
BATCH_FIELDS = ("env_state", "obs", "critic_obs", "ret_acc", "ep_return", "ep_len", "trunc_limit")


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 1024
    num_steps: int = 32
    total_iters: int = 1500
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    lr_end: float = 3e-5
    max_grad_norm: float = 0.5
    clip: float = 0.2
    gamma: float = 0.995
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    target_kl: float = 0.02
    obs_clip: float = 5.0
    init_log_std: float | tuple[float, ...] = nn.LOG_STD_INIT  # scalar or per action channel
    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (128, 128)
    actor_head_gain: float = 0.01  # orthogonal init gain of the output layers
    critic_head_gain: float = 1.0
    actor_warmup: int = 0  # iterations with the actor frozen (lr 0) while the critic trains
    scale_rewards: bool = True
    devices: int = 1  # XLA devices the rollout's env batch is split over


class TrainState(ty.NamedTuple):
    params: dict
    opt: nn.Adam
    obs_rms: nn.RunningStat
    critic_rms: nn.RunningStat
    ret_rms: nn.RunningStat
    env_state: ty.Any
    obs: jax.Array
    critic_obs: jax.Array
    ret_acc: jax.Array  # discounted raw return per env, for reward scaling
    ep_return: jax.Array
    ep_len: jax.Array
    trunc_limit: jax.Array  # staggers the first episode of each env
    key: jax.Array
    iteration: jax.Array


def tree_select(mask: jax.Array, on_true, on_false):
    def select(a, b):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            data = select(jax.random.key_data(a), jax.random.key_data(b))
            return jax.random.wrap_key_data(data, impl=jax.random.key_impl(a))
        m = mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim))
        return jnp.where(m, a, b)

    return jax.tree.map(select, on_true, on_false)


def gae(rewards, values, finished, bootstrap, last_value, gamma, lam):
    """[T, B] inputs. finished ends the episode after step t; bootstrap holds gamma-free V(s_final)
    for truncated steps and 0 for terminated or ongoing ones."""

    def step(carry, x):
        adv_next, v_next = carry
        r, v, fin, boot = x
        live = 1.0 - fin.astype(F32)
        delta = r + gamma * (live * v_next + boot) - v
        adv = delta + gamma * lam * live * adv_next
        return (adv, v), adv

    init = (jnp.zeros_like(last_value), last_value)
    _, adv = jax.lax.scan(step, init, (rewards, values, finished, bootstrap), reverse=True)
    return adv, adv + values


def _finite(x: jax.Array) -> jax.Array:
    x = x.astype(F32)
    return jnp.where(jnp.isfinite(x), x, 0.0)


def _all_finite(x: jax.Array) -> jax.Array:
    return jnp.all(jnp.isfinite(x), axis=-1)


def actor_dist(actor: dict, obs_n: jax.Array):
    return nn.mlp(actor["mlp"], obs_n), actor["log_std"]


def critic_value(critic: list, critic_obs_n: jax.Array) -> jax.Array:
    return nn.mlp(critic, critic_obs_n)[..., 0]


def mesh(devices: int) -> jax.sharding.Mesh:
    """1-D mesh over the first `devices` XLA devices, axis "b" (the env batch)."""
    available = jax.devices()
    if devices > len(available):
        raise ValueError(
            f"{devices} devices requested, {len(available)} available; set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={devices} before importing jax"
        )
    return jax.sharding.Mesh(np.asarray(available[:devices]), ("b",))


def _sharded(fn, devices: int, in_specs, out_specs):
    """fn unchanged on one device, else shard_map over the batch mesh."""
    if devices <= 1:
        return fn
    # check_vma off: the stock drone graph has lax.cond branches mixing per-env and constant outputs
    m = mesh(devices)
    return jax.shard_map(fn, mesh=m, in_specs=in_specs, out_specs=out_specs, check_vma=False)


def _movers(devices: int):
    """(to_mesh, to_first): replicate a pytree over the batch mesh / gather it on device 0."""
    m = mesh(max(devices, 1))
    first = SingleDeviceSharding(m.devices.flat[0])
    rep = NamedSharding(m, P()) if devices > 1 else first
    return (lambda x: jax.device_put(x, rep)), (lambda x: jax.device_put(x, first))


def place(ts: TrainState, devices: int) -> TrainState:
    """Put the per-env fields of ts on the batch mesh and replicate the rest."""
    if devices <= 1:
        return ts
    m = mesh(devices)
    batch, rep = NamedSharding(m, P("b")), NamedSharding(m, P())
    fields = (
        jax.device_put(v, batch if f in BATCH_FIELDS else rep) for f, v in zip(ts._fields, ts)
    )
    return TrainState(*fields)


def init(env, cfg: PPOConfig, key: jax.Array, dr_scale=0.0, throw_scale=None) -> TrainState:
    assert cfg.num_envs % cfg.devices == 0, "num_envs must divide over devices"
    throw_scale = dr_scale if throw_scale is None else throw_scale
    k_actor, k_critic, k_reset, k_stagger, key = jax.random.split(key, 5)
    params = {
        "actor": nn.init_actor(
            k_actor,
            env.obs_dim,
            env.act_dim,
            cfg.actor_hidden,
            cfg.init_log_std,
            cfg.actor_head_gain,
        ),
        "critic": nn.init_critic(k_critic, env.critic_dim, cfg.critic_hidden, cfg.critic_head_gain),
    }
    reset = jax.vmap(env.reset, in_axes=(0, None, None))
    reset = jax.jit(_sharded(reset, cfg.devices, (P("b"), P(), P()), P("b")))
    keys = jax.random.split(k_reset, cfg.num_envs)
    if cfg.devices > 1:
        keys = jax.device_put(keys, NamedSharding(mesh(cfg.devices), P("b")))
    env_state, obs, critic_obs = reset(keys, dr_scale, throw_scale)
    obs, critic_obs = obs.astype(F32), critic_obs.astype(F32)
    b = cfg.num_envs
    ts = TrainState(
        params=params,
        opt=nn.adam_init(params),
        obs_rms=nn.running_stat((env.obs_dim,)),
        critic_rms=nn.running_stat((env.critic_dim,)),
        ret_rms=nn.running_stat(),
        env_state=env_state,
        obs=obs,
        critic_obs=critic_obs,
        ret_acc=jnp.zeros(b, F32),
        ep_return=jnp.zeros(b, F32),
        ep_len=jnp.zeros(b, jnp.int32),
        trunc_limit=jax.random.randint(k_stagger, (b,), 1, env.episode_steps + 1, jnp.int32),
        key=key,
        iteration=jnp.zeros((), jnp.int32),
    )
    return place(ts, cfg.devices)


def make_train_iter(env, cfg: PPOConfig):
    """-> (TrainState, dr_scale, throw_scale=None) -> (TrainState, metrics); throw_scale
    defaults to dr_scale. Pass the scales as jnp.float32 to avoid retracing."""
    b, t = cfg.num_envs, cfg.num_steps
    n = b * t
    assert n % cfg.minibatches == 0, "num_envs * num_steps must divide into minibatches"
    assert b % cfg.devices == 0, "num_envs must divide over devices"
    mb_size = n // cfg.minibatches
    v_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    v_step = jax.vmap(env.step)
    never = env.episode_steps + 1

    def rollout(params, obs_rms, critic_rms, carry, pool_keys, noise, dr_scale, throw_scale):
        """Per shard: carry fields, pool_keys [b] and noise [T, b, A] cover this shard's envs."""
        # one fresh start per env per iteration; an env that finishes twice in one rollout reuses it
        pool_state, pool_obs, pool_cobs = v_reset(pool_keys, dr_scale, throw_scale)
        pool_obs, pool_cobs = pool_obs.astype(F32), pool_cobs.astype(F32)
        actor, critic = params["actor"], params["critic"]

        def value_of(cobs):
            return critic_value(critic, nn.normalize(critic_rms, cobs, cfg.obs_clip))

        pool_value = value_of(pool_cobs)

        def step(carry, eps):
            env_state, obs, cobs, value, ret_acc, ep_ret, ep_len, limit = carry
            mean, log_std = actor_dist(actor, nn.normalize(obs_rms, obs, cfg.obs_clip))
            raw = mean + jnp.exp(log_std) * eps
            logp = nn.gaussian_log_prob(raw, mean, log_std)
            env_state, nobs, ncobs, reward, done, info = v_step(env_state, jnp.clip(raw, -1.0, 1.0))
            finite = jnp.isfinite(reward) & _all_finite(nobs) & _all_finite(ncobs)
            nobs, ncobs, reward = _finite(nobs), _finite(ncobs), _finite(reward)
            done = done.astype(bool)
            terminated = done & info["terminated"].astype(bool)
            ep_len = ep_len + 1
            finished = done | (ep_len >= limit)
            v_final = value_of(ncobs)
            bootstrap = jnp.where(finished & ~terminated, v_final, 0.0)
            ep_ret = ep_ret + reward
            ret_acc = ret_acc * cfg.gamma + reward
            out = {
                "obs": obs,
                "cobs": cobs,
                "raw": raw,
                "logp": logp,
                "value": value,
                "reward": reward,
                "finished": finished,
                "bootstrap": bootstrap,
                "ret": ret_acc,
                "done": done,
                "done_return": jnp.where(done, ep_ret, 0.0),
                "done_len": jnp.where(done, ep_len, 0),
                "success": done & info["success"].astype(bool),
                "terminated": terminated,
                "nonfinite": ~finite,
            }
            env_state = tree_select(finished, pool_state, env_state)
            nobs = jnp.where(finished[:, None], pool_obs, nobs)
            ncobs = jnp.where(finished[:, None], pool_cobs, ncobs)
            value = jnp.where(finished, pool_value, v_final)
            ret_acc = jnp.where(finished, 0.0, ret_acc)
            ep_ret = jnp.where(finished, 0.0, ep_ret)
            ep_len = jnp.where(finished, 0, ep_len)
            limit = jnp.where(finished, never, limit)
            return (env_state, nobs, ncobs, value, ret_acc, ep_ret, ep_len, limit), out

        env_state, obs, cobs, ret_acc, ep_ret, ep_len, limit = carry
        carry = (env_state, obs, cobs, value_of(cobs), ret_acc, ep_ret, ep_len, limit)
        carry, traj = jax.lax.scan(step, carry, noise)
        env_state, obs, cobs, last_value, ret_acc, ep_ret, ep_len, limit = carry
        return (env_state, obs, cobs, ret_acc, ep_ret, ep_len, limit), last_value, traj

    rollout = _sharded(
        rollout,
        cfg.devices,
        (P(), P(), P(), P("b"), P("b"), P(None, "b"), P(), P()),
        (P("b"), P("b"), P(None, "b")),
    )

    def collect(params, obs_rms, critic_rms, carry, key, dr_scale, throw_scale):
        key, k_roll, k_update = jax.random.split(key, 3)
        k_pool, k_steps = jax.random.split(k_roll)
        # drawn for the whole batch so every env sees the same numbers on any device count
        pool_keys = jax.random.split(k_pool, b)
        noise = jax.random.normal(k_steps, (t, b, env.act_dim), F32)
        carry, last_value, traj = rollout(
            params, obs_rms, critic_rms, carry, pool_keys, noise, dr_scale, throw_scale
        )
        return carry, last_value, traj, key, k_update

    def loss_fn(params, mb):
        mean, log_std = actor_dist(params["actor"], mb["obs_n"])
        logp = nn.gaussian_log_prob(mb["raw"], mean, log_std)
        log_ratio = logp - mb["logp"]
        ratio = jnp.exp(log_ratio)
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        clipped = jnp.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
        policy_loss = jnp.maximum(-adv * ratio, -adv * clipped).mean()
        value = critic_value(params["critic"], mb["cobs_n"])
        value_loss = 0.5 * jnp.mean((value - mb["ret"]) ** 2)
        entropy = nn.gaussian_entropy(log_std)
        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
            "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip).astype(F32)),
        }
        return loss, metrics

    def update(params, opt, batch, key, lr, actor_frozen):
        def minibatch(carry, idx):
            params, opt = carry
            mb = jax.tree.map(lambda x: x[idx], batch)
            (_, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
            grads, grad_norm = nn.clip_by_global_norm(grads, cfg.max_grad_norm)
            ok = jnp.isfinite(grad_norm)
            stepped, stepped_opt = nn.adam_step(grads, opt, params, lr)
            # warmup: the actor keeps its params (Adam moments still see its gradients)
            actor = tree_select(actor_frozen, params["actor"], stepped["actor"])
            stepped = {**stepped, "actor": actor}
            # a non-finite gradient would poison params and Adam moments for good: skip the step
            params, opt = tree_select(ok, (stepped, stepped_opt), (params, opt))
            return (params, opt), {**m, "grad_norm": grad_norm, "ok": ok}

        # metrics are summed over the minibatches whose step ran, so skipped ones never enter them
        names = (*METRIC_KEYS, "grad_norm")

        def run(params, opt, k):
            idx = jax.random.permutation(k, n).reshape(cfg.minibatches, mb_size)
            (params, opt), m = jax.lax.scan(minibatch, (params, opt), idx)
            sums = {name: jnp.where(m["ok"], m[name], 0.0).sum().astype(F32) for name in names}
            return params, opt, sums, m["ok"].sum().astype(F32)

        def skip(params, opt, k):
            zero = jnp.zeros((), F32)
            return params, opt, {name: zero for name in names}, zero

        def epoch(carry, k):
            params, opt, stop = carry
            params, opt, sums, count = jax.lax.cond(stop, skip, run, params, opt, k)
            ran = ~stop
            # NaN when every step of the epoch was skipped, which also stops the update
            kl = sums["approx_kl"] / count
            stop = stop | ~(kl <= cfg.target_kl)
            return (params, opt, stop), (sums, count, ran)

        init = (params, opt, jnp.zeros((), bool))
        (params, opt, _), (sums, count, ran) = jax.lax.scan(
            epoch, init, jax.random.split(key, cfg.epochs)
        )
        steps = count.sum()
        metrics = {name: v.sum() / steps for name, v in sums.items()}
        metrics["epochs"] = ran.astype(F32).sum()
        metrics["skipped_updates"] = metrics["epochs"] * cfg.minibatches - steps
        return params, opt, metrics

    def learn(params, opt, obs_rms, critic_rms, ret_rms, iteration, traj, last_value, key):
        ret_rms = nn.update_stat(ret_rms, traj["ret"].reshape(-1))
        scale = jax.lax.rsqrt(ret_rms.var + 1e-8) if cfg.scale_rewards else 1.0
        rewards = traj["reward"] * scale
        adv, returns = gae(
            rewards,
            traj["value"],
            traj["finished"],
            traj["bootstrap"],
            last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        batch = {
            "obs_n": nn.normalize(obs_rms, traj["obs"], cfg.obs_clip),
            "cobs_n": nn.normalize(critic_rms, traj["cobs"], cfg.obs_clip),
            "raw": traj["raw"],
            "logp": traj["logp"],
            "adv": adv,
            "ret": returns,
        }
        batch = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), batch)
        frac = jnp.minimum(iteration.astype(F32) / max(cfg.total_iters - 1, 1), 1.0)
        lr = cfg.lr + frac * (cfg.lr_end - cfg.lr)
        actor_frozen = iteration < cfg.actor_warmup
        params, opt, metrics = update(params, opt, batch, key, lr, actor_frozen)

        finished_eps = traj["done"].sum()
        denom = jnp.maximum(finished_eps, 1).astype(F32)
        resid = (returns - traj["value"]).var()

        def per_episode(x):
            return jnp.where(finished_eps > 0, x.sum() / denom, jnp.nan)

        metrics.update(
            lr=lr,
            actor_lr=jnp.where(actor_frozen, 0.0, lr),
            finished_episodes=finished_eps,
            mean_return=per_episode(traj["done_return"]),
            mean_length=per_episode(traj["done_len"].astype(F32)),
            success_rate=per_episode(traj["success"].astype(F32)),
            terminated_rate=per_episode(traj["terminated"].astype(F32)),
            mean_reward=traj["reward"].mean(),
            ret_std=jnp.sqrt(ret_rms.var),
            explained_var=1.0 - resid / (returns.var() + 1e-8),
            nonfinite_steps=traj["nonfinite"].sum(),
            # envs reset more than once in this rollout replay the same pooled start
            replay_envs=(traj["finished"].sum(0) >= 2).sum(),
        )
        obs_rms = nn.update_stat(obs_rms, traj["obs"].reshape(n, -1))
        critic_rms = nn.update_stat(critic_rms, traj["cobs"].reshape(n, -1))
        return (params, opt, obs_rms, critic_rms, ret_rms, iteration + 1), metrics

    collect, learn = jax.jit(collect), jax.jit(learn)
    to_mesh, to_first = _movers(cfg.devices)

    def train_iter(ts: TrainState, dr_scale, throw_scale=None):
        throw_scale = dr_scale if throw_scale is None else throw_scale
        carry = tuple(getattr(ts, f) for f in BATCH_FIELDS)
        carry, last_value, traj, key, k_update = collect(
            ts.params, ts.obs_rms, ts.critic_rms, carry, ts.key, dr_scale, throw_scale
        )
        learner = (ts.params, ts.opt, ts.obs_rms, ts.critic_rms, ts.ret_rms, ts.iteration)
        learner, metrics = learn(*to_first((*learner, traj, last_value, k_update)))
        params, opt, obs_rms, critic_rms, ret_rms, iteration = to_mesh(learner)
        ts = TrainState(
            params, opt, obs_rms, critic_rms, ret_rms, *carry, key=key, iteration=iteration
        )
        return ts, metrics

    return train_iter


def make_evaluate(
    env, num_envs: int, deterministic: bool = True, obs_clip: float = 5.0, devices: int = 1
):
    """-> (actor, obs_rms, key, dr_scale, throw_scale=None) -> episode metrics over num_envs
    fresh episodes of env.episode_steps (no auto-reset), split over `devices` like the rollout."""
    assert num_envs % devices == 0, "num_envs must divide over devices"
    v_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    v_step = jax.vmap(env.step)

    def episodes(actor, obs_rms, keys, noise, dr_scale, throw_scale):
        state, obs, _ = v_reset(keys, dr_scale, throw_scale)
        false = jnp.zeros(keys.shape[0], bool)

        def step(carry, eps):
            state, obs, alive, ret, length, success, terminated = carry
            mean, log_std = actor_dist(actor, nn.normalize(obs_rms, obs, obs_clip))
            if not deterministic:
                mean = mean + jnp.exp(log_std) * eps
            state, obs, _, reward, done, info = v_step(state, jnp.clip(mean, -1.0, 1.0))
            ended = alive & done.astype(bool)
            ret = ret + jnp.where(alive, _finite(reward), 0.0)
            length = length + alive.astype(jnp.int32)
            success = success | (ended & info["success"].astype(bool))
            terminated = terminated | (ended & info["terminated"].astype(bool))
            return (state, obs.astype(F32), alive & ~ended, ret, length, success, terminated), None

        zeros = jnp.zeros(keys.shape[0], F32)
        carry = (state, obs.astype(F32), ~false, zeros, zeros.astype(jnp.int32), false, false)
        carry, _ = jax.lax.scan(step, carry, noise)
        return carry[3:]

    episodes = _sharded(episodes, devices, (P(), P(), P("b"), P(None, "b"), P(), P()), P("b"))

    @jax.jit
    def run(actor, obs_rms, key, dr_scale, throw_scale):
        k_reset, k_act = jax.random.split(key)
        noise = jax.random.normal(k_act, (env.episode_steps, num_envs, env.act_dim), F32)
        keys = jax.random.split(k_reset, num_envs)
        return episodes(actor, obs_rms, keys, noise, dr_scale, throw_scale)

    @jax.jit
    def summarize(ret, length, success, terminated):
        return {
            "mean_return": ret.mean(),
            "success_rate": success.astype(F32).mean(),
            "terminated_rate": terminated.astype(F32).mean(),
            "mean_length": length.astype(F32).mean(),
        }

    to_mesh, to_first = _movers(devices)

    def evaluate(actor, obs_rms, key, dr_scale, throw_scale=None):
        throw_scale = dr_scale if throw_scale is None else throw_scale
        per_env = run(*to_mesh((actor, obs_rms, key)), dr_scale, throw_scale)
        # reduced on one device so the means do not depend on the device count
        return summarize(*to_first(per_env))

    return evaluate


def save_policy(
    out_dir: str,
    ts: TrainState,
    cfg: PPOConfig,
    meta: dict | None = None,
    extra_arrays: dict | None = None,
):
    """Write out_dir/policy.npz (actor, log_std, frozen obs mean/std, plus extra_arrays such as
    a residual base controller's config) and out_dir/config.json."""
    os.makedirs(out_dir, exist_ok=True)
    arrays = {
        **nn.policy_arrays(ts.params["actor"], ts.obs_rms, cfg.obs_clip),
        **(extra_arrays or {}),
    }
    path = os.path.join(out_dir, "policy.npz")
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrays)
    os.replace(path + ".tmp", path)
    config = {"ppo": dataclasses.asdict(cfg), **(meta or {})}
    path = os.path.join(out_dir, "config.json")
    with open(path + ".tmp", "w") as f:
        json.dump(config, f, indent=2)
    os.replace(path + ".tmp", path)
