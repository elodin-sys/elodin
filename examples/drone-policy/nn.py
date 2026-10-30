"""Float32 MLPs, Gaussian policy helpers, Adam and running normalization in pure jax.numpy."""

import typing as ty

import jax
import jax.numpy as jnp
import numpy as np

F32 = jnp.float32
LOG_2PI = float(np.log(2.0 * np.pi))
LOG_STD_INIT = tuple(float(np.log(s)) for s in (0.1, 0.1, 0.1, 0.2))  # rates, collective


def init_mlp(key: jax.Array, sizes: ty.Sequence[int], head_gain: float) -> list[dict]:
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        gain = head_gain if i == len(sizes) - 2 else float(np.sqrt(2.0))
        w = jax.nn.initializers.orthogonal(gain)(k, (n_in, n_out), F32)
        layers.append({"w": w, "b": jnp.zeros(n_out, F32)})
    return layers


def mlp(layers: list[dict], x: jax.Array) -> jax.Array:
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_actor(
    key, obs_dim, act_dim, hidden=(64, 64), log_std=LOG_STD_INIT, head_gain=0.01
) -> dict:
    """log_std is a scalar or one value per action channel."""
    return {
        "mlp": init_mlp(key, (obs_dim, *hidden, act_dim), head_gain=head_gain),
        "log_std": jnp.broadcast_to(jnp.asarray(log_std, F32), (act_dim,)),
    }


def init_critic(key, critic_dim, hidden=(128, 128), head_gain=1.0) -> list[dict]:
    return init_mlp(key, (critic_dim, *hidden, 1), head_gain=head_gain)


def gaussian_log_prob(x: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    z = (x - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * z**2 - log_std - 0.5 * LOG_2PI, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (LOG_2PI + 1.0))


class Adam(ty.NamedTuple):
    m: ty.Any
    v: ty.Any
    count: jax.Array


def adam_init(params) -> Adam:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return Adam(zeros, zeros, jnp.zeros((), jnp.int32))


def adam_step(grads, opt: Adam, params, lr, b1=0.9, b2=0.999, eps=1e-8):
    count = opt.count + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, opt.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, opt.v, grads)
    c = count.astype(F32)
    m_scale = 1.0 / (1.0 - b1**c)
    v_scale = 1.0 / (1.0 - b2**c)
    params = jax.tree.map(
        lambda p, m, v: p - lr * m * m_scale / (jnp.sqrt(v * v_scale) + eps), params, m, v
    )
    return params, Adam(m, v, count)


def clip_by_global_norm(grads, max_norm: float):
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads), norm


class RunningStat(ty.NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def running_stat(shape: tuple[int, ...] = ()) -> RunningStat:
    return RunningStat(jnp.zeros(shape, F32), jnp.ones(shape, F32), jnp.asarray(1e-4, F32))


def update_stat(stat: RunningStat, x: jax.Array) -> RunningStat:
    # batched Welford merge (Chan et al.), x is [N, *shape]
    x = x.astype(F32)
    n = x.shape[0]
    batch_mean, batch_var = x.mean(0), x.var(0)
    total = stat.count + n
    delta = batch_mean - stat.mean
    mean = stat.mean + delta * (n / total)
    m2 = stat.var * stat.count + batch_var * n + delta**2 * (stat.count * n / total)
    return RunningStat(mean, m2 / total, total)


def normalize(stat: RunningStat, x: jax.Array, clip: float = 5.0) -> jax.Array:
    return jnp.clip((x.astype(F32) - stat.mean) / jnp.sqrt(stat.var + 1e-8), -clip, clip)


def policy_arrays(actor: dict, obs_stat: RunningStat, obs_clip: float) -> dict[str, np.ndarray]:
    out = {}
    for i, layer in enumerate(actor["mlp"]):
        out[f"actor_w{i}"] = np.asarray(layer["w"], np.float32)
        out[f"actor_b{i}"] = np.asarray(layer["b"], np.float32)
    out["log_std"] = np.asarray(actor["log_std"], np.float32)
    out["obs_mean"] = np.asarray(obs_stat.mean, np.float32)
    out["obs_std"] = np.asarray(jnp.sqrt(obs_stat.var + 1e-8), np.float32)
    out["obs_clip"] = np.asarray(obs_clip, np.float32)
    return out


def policy_action(policy: ty.Mapping[str, ty.Any], obs, xp=jnp):
    """Deterministic action from policy.npz arrays; computes in obs.dtype with xp = jnp or numpy."""
    dtype = obs.dtype
    clip = xp.asarray(policy["obs_clip"], dtype)
    x = (obs - xp.asarray(policy["obs_mean"], dtype)) / xp.asarray(policy["obs_std"], dtype)
    x = xp.minimum(xp.maximum(x, -clip), clip)
    n_layers = sum(1 for k in policy if k.startswith("actor_w"))
    for i in range(n_layers):
        x = x @ xp.asarray(policy[f"actor_w{i}"], dtype) + xp.asarray(policy[f"actor_b{i}"], dtype)
        if i < n_layers - 1:
            x = xp.tanh(x)
    return xp.minimum(xp.maximum(x, -1.0), 1.0)


def load_policy(path: str):
    """policy.npz -> (actor, obs_rms, obs_clip, exact) for ppo.make_evaluate. The variance is
    chosen so that normalize reproduces the saved obs_std bit for bit (exact)."""
    p = dict(np.load(path))
    n_layers = sum(1 for k in p if k.startswith("actor_w"))
    actor = {
        "mlp": [
            {"w": jnp.asarray(p[f"actor_w{i}"]), "b": jnp.asarray(p[f"actor_b{i}"])}
            for i in range(n_layers)
        ],
        "log_std": jnp.asarray(p["log_std"]),
    }
    std = p["obs_std"].astype(np.float32)
    var = np.maximum(std.astype(np.float64) ** 2 - 1e-8, 0.0).astype(np.float32)
    for _ in range(8):
        got = np.asarray(jnp.sqrt(jnp.asarray(var) + 1e-8))
        if np.array_equal(got, std):
            break
        var = np.where(got < std, np.nextafter(var, np.float32(np.inf)), var)
        var = np.where(got > std, np.nextafter(var, np.float32(0.0)), var)
    exact = bool(np.array_equal(np.asarray(jnp.sqrt(jnp.asarray(var) + 1e-8)), std))
    stat = RunningStat(jnp.asarray(p["obs_mean"]), jnp.asarray(var), jnp.asarray(1.0, F32))
    return actor, stat, float(p["obs_clip"]), exact
