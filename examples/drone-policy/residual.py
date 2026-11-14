"""Residual policy over the B0 cascade (pure jnp, jit/vmap-safe).

ResidualEnv wraps a DroneEnv: step(state, a_res) executes clip(b0_action(obs_prev) + scale *
a_res, -1, 1), where obs_prev is the actor obs the policy acted on. The actor obs is [obs (19),
base action (4)] and the critic obs gets the base action appended too. residual_action is the
deployable numpy/jnp forward of B0 + NN from a policy.npz that carries residual_arrays().

Module-level imports stay numpy/jax only, so residual_action runs without the elodin world;
ResidualEnv imports cascade_b0 when it is built.
"""

import typing as ty

import jax
import jax.numpy as jnp
import nn
import numpy as np

BASES = ("dr", "nominal")  # cascade_b0.GAIN_SETS names
F32 = jnp.float32


class ResidualState(ty.NamedTuple):
    env: ty.Any  # the wrapped env's state
    obs_prev: jax.Array  # f32[19] actor obs the next base action is computed from


class ResidualEnv:
    def __init__(self, env, base: str = "dr", scale: float = 0.3):
        import cascade_b0

        if base not in BASES:
            raise ValueError(f"base must be one of {BASES}, got {base!r}")
        self.env, self.base, self.scale = env, base, float(scale)
        self.gains = tuple(float(v) for v in cascade_b0.GAIN_SETS[base])
        self._b0 = cascade_b0
        self.obs_dim = env.obs_dim + env.act_dim
        self.critic_dim = env.critic_dim + env.act_dim
        self.act_dim = env.act_dim
        self.episode_steps = env.episode_steps

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def base_action(self, obs: jax.Array) -> jax.Array:
        # f64 gains, as cascade_b0.Runner evaluates them
        gains = self._b0.B0Gains(*[jnp.asarray(v, jnp.float64) for v in self.gains])
        return self._b0.b0_action(obs, gains)

    def _wrap(self, state, obs, critic):
        base = self.base_action(obs).astype(F32)
        return (
            ResidualState(state, obs),
            jnp.concatenate([obs, base]).astype(F32),
            jnp.concatenate([critic, base]).astype(F32),
        )

    def reset(self, key, dr_scale, throw_scale=None):
        return self._wrap(*self.env.reset(key, dr_scale, throw_scale))

    def reset_with(self, key, dr, throw):
        return self._wrap(*self.env.reset_with(key, dr, throw))

    def action(self, state: ResidualState, a_res: jax.Array) -> jax.Array:
        """The executed action for residual a_res."""
        return jnp.clip(self.base_action(state.obs_prev) + self.scale * a_res, -1.0, 1.0)

    def step(self, state: ResidualState, a_res: jax.Array):
        s, obs, critic, r, done, info = self.env.step(state.env, self.action(state, a_res))
        return (*self._wrap(s, obs, critic), r, done, info)

    def policy_arrays(self) -> dict[str, np.ndarray]:
        """policy.npz entries that rebuild the base controller for residual_action."""
        import policy_io
        from dr import G

        return {
            "residual_base": np.asarray(self.base),
            "residual_scale": np.asarray(self.scale, np.float64),
            "b0_gains": np.asarray(self.gains, np.float64),  # B0Gains field order
            "b0_thrust_axis": np.asarray(self._b0.THRUST_AXIS, np.float64),
            "b0_a_z_min": np.asarray(self._b0.A_Z_MIN, np.float64),
            "b0_g": np.asarray(G, np.float64),
            "b0_hover_collective": np.asarray(policy_io.HOVER_COLLECTIVE, np.float64),
            "b0_collective_scale": np.asarray(policy_io.COLLECTIVE_SCALE, np.float64),
            "b0_act_scale": np.asarray(policy_io.ACT_SCALE, np.float64),
        }

    def config(self) -> dict:
        return {"base": self.base, "scale": self.scale, "gains": list(self.gains)}


def b0_from_arrays(policy: ty.Mapping[str, ty.Any], obs, xp=jnp):
    """cascade_b0.b0_action in xp (numpy or jax.numpy) from the residual_arrays constants;
    obs is [..., >= 15], computed in float64."""
    f64 = xp.float64

    def c(name):
        return xp.asarray(policy[name], f64)

    gains = c("b0_gains")
    kp_xy, kd_xy, ki_xy, kp_z, kd_z, ki_z, k_att, k_rp, k_yaw, yaw_max, max_tilt, a_max, h_fl = (
        gains[i] for i in range(13)
    )
    g_acc = c("b0_g")
    obs = xp.asarray(obs, f64)
    g_body, gyro, p_err, v, p_int = (obs[..., i : i + 3] for i in range(0, 15, 3))
    kp = xp.stack([kp_xy, kp_xy, kp_z])
    kd = xp.stack([kd_xy, kd_xy, kd_z])
    ki = xp.stack([ki_xy, ki_xy, ki_z])
    a = -kp * p_err - kd * v - ki * p_int
    az = xp.clip(a[..., 2] + g_acc, c("b0_a_z_min"), a_max)
    h = xp.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)
    room = xp.sqrt(xp.maximum((a_max - az) * (a_max + az), 0.0))
    h_tilt = az * xp.sin(max_tilt) / xp.cos(max_tilt)
    h_max = xp.maximum(xp.minimum(h_tilt, room), h_fl * h_tilt)
    s = xp.minimum(1.0, h_max / xp.maximum(h, 1e-9))
    ax, ay = a[..., 0] * s, a[..., 1] * s

    g = g_body / xp.maximum(xp.sqrt(xp.sum(g_body**2, axis=-1, keepdims=True)), 1e-9)
    c_th = xp.sqrt(g[..., 1] ** 2 + g[..., 2] ** 2)
    s_th = -g[..., 0]
    c_ph = g[..., 2] / xp.maximum(c_th, 1e-9)
    s_ph = g[..., 1] / xp.maximum(c_th, 1e-9)
    ux = c_th * ax - s_th * az
    uz = s_th * ax + c_th * az
    b = xp.stack([ux, c_ph * ay + s_ph * uz, -s_ph * ay + c_ph * uz], axis=-1)

    t = c("b0_thrust_axis")
    thrust = xp.sum(b * t, axis=-1)
    n = b / xp.maximum(xp.sqrt(xp.sum(b**2, axis=-1, keepdims=True)), 1e-9)
    axis = xp.cross(t, n)
    sin_e = xp.sqrt(xp.sum(axis**2, axis=-1))
    err = xp.arctan2(sin_e, xp.sum(t * n, axis=-1))
    w_xy = (k_att * err / xp.maximum(sin_e, 1e-9))[..., None] * axis[..., :2]
    w_xy = w_xy + k_rp * (w_xy - gyro[..., :2])
    w_z = gyro[..., 2] - xp.clip(k_yaw * gyro[..., 2], -yaw_max, yaw_max)
    w = xp.concatenate([w_xy, w_z[..., None]], axis=-1)

    hover = c("b0_hover_collective")
    col = (hover * thrust / g_acc - hover) / c("b0_collective_scale")
    act = xp.concatenate([w / c("b0_act_scale"), col[..., None]], axis=-1)
    return xp.clip(act, -1.0, 1.0)


def residual_action(policy: ty.Mapping[str, ty.Any], obs, xp=jnp):
    """Deterministic B0 + NN action from policy.npz arrays for the 19-dim actor obs [..., 19]:
    clip(b0 + scale * NN([obs, b0]), -1, 1). The NN runs in obs.dtype like nn.policy_action, the
    base controller in float64."""
    dtype = obs.dtype
    base = b0_from_arrays(policy, obs, xp)
    res = nn.policy_action(policy, xp.concatenate([obs, base.astype(dtype)], axis=-1), xp)
    return xp.clip(base + xp.asarray(policy["residual_scale"], dtype) * res, -1.0, 1.0)
