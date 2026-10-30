import inspect

import dr as D
import env as E
import jax
import jax.numpy as jnp
import numpy as np
import plant_dr
import policy_io
import policy_world
import pytest
import vio
from config import Config
from jaxstep import JaxStep


def rollout(environment, keys, dr_scale, steps):
    """reset + `steps` policy steps with key-derived actions, vmapped over keys."""

    def one(key):
        state, obs, critic = environment.reset(key, dr_scale)

        def body(s, i):
            a = 0.3 * jax.random.normal(jax.random.fold_in(key, i), (4,))
            s, o, c, r, d, info = environment.step(s, a)
            return s, (o, c, r, d)

        state, out = jax.lax.scan(body, state, jnp.arange(steps))
        return state, obs, out

    return jax.jit(jax.vmap(one))(keys)


def test_jit_vmap_deterministic_per_key(env_truth):
    keys = jax.random.split(jax.random.key(0), 8)
    perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
    s1, o1, (obs1, crit1, r1, d1) = rollout(env_truth, keys, 1.0, 5)
    s2, o2, (obs2, crit2, r2, d2) = rollout(env_truth, keys, 1.0, 5)
    s3, o3, (obs3, crit3, r3, d3) = rollout(env_truth, keys[perm], 1.0, 5)
    for a, b, c in [(o1, o2, o3), (obs1, obs2, obs3), (crit1, crit2, crit3), (r1, r2, r3)]:
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.asarray(a)[perm], c)
    for a, b in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
        np.testing.assert_array_equal(a, b)
    # an unbatched run of one key reproduces its row of the batch
    single = jax.jit(lambda k: env_truth.reset(k, 1.0))(keys[5])
    np.testing.assert_allclose(single[1], o1[5], rtol=0, atol=1e-6)
    assert obs1.shape == (8, 5, E.OBS_DIM) and obs1.dtype == jnp.float32
    assert crit1.shape == (8, 5, E.CRITIC_DIM) and crit1.dtype == jnp.float32
    assert r1.dtype == jnp.float32 and d1.dtype == jnp.bool_
    # different keys give different episodes
    assert not np.allclose(o1[0], o1[1])


def check_dr(dr: D.DRParams, scale: float):
    if scale == 0.0:  # the stock vehicle
        nominal = D.nominal_dr()
        for field, value in dr._asdict().items():
            np.testing.assert_array_equal(
                value, np.broadcast_to(getattr(nominal, field), value.shape)
            )
        return

    def within(x, lo, hi, nominal_value):
        lo_s = nominal_value + scale * (lo - nominal_value)
        hi_s = nominal_value + scale * (hi - nominal_value)
        x = np.asarray(x)
        assert x.min() >= lo_s - 1e-9 and x.max() <= hi_s + 1e-9, (x.min(), x.max(), lo_s, hi_s)

    within(dr.mass_scale, 0.9, 1.1, 1.0)
    within(dr.inertia_scale, 0.8, 1.2, 1.0)
    within(dr.thrust_scale, 0.95, 1.05, 1.0)
    within(dr.kq_scale, 0.5, 2.0, 1.0)
    within(dr.motor_tau, 0.04, 0.15, 0.1)
    within(dr.drag_coeff, 0.05, 0.2, 0.1)
    within(dr.force, -1.0, 1.0, 0.0)
    within(dr.torque, -0.05, 0.05, 0.0)
    within(dr.rate_gain_scale, 0.8, 1.25, 1.0)
    within(dr.vio_pos_bias, -0.02, 0.02, 0.0)
    within(dr.vio_pos_sigma, 0.0, 0.02, 0.0)
    within(dr.vio_vel_sigma, 0.0, 0.05, 0.0)
    assert np.linalg.norm(np.asarray(dr.wind[:, :2]), axis=-1).max() <= 2.0 * scale + 1e-9
    assert np.abs(np.asarray(dr.wind[:, 2])).max() <= 0.3 * scale + 1e-9
    stock_bias = np.array(D.STOCK_GYRO_BIAS)
    for k in range(3):
        within(dr.gyro_bias[:, k], -0.01, 0.01, stock_bias[k])
    act, vd = np.asarray(dr.act_delay), np.asarray(dr.vio_delay)
    assert set(np.unique(act)) <= set(range(3)) and set(np.unique(vd)) <= set(range(7))
    if scale == 1.0:
        assert set(np.unique(act)) == {0, 1, 2} and set(np.unique(vd)) == set(range(7))
        assert np.ptp(np.asarray(dr.mass_scale)) > 0.15  # the range is covered


def thrust_tilt(quat: jax.Array) -> jax.Array:
    """Angle of the trim net-thrust axis from world vertical (rad)."""
    up = D.quat_rotate(quat, jnp.asarray(plant_dr.hover_trim().axis))[2]
    return jnp.arccos(jnp.clip(up, -1.0, 1.0))


def check_throw(environment, s: E.EnvState, scale: float):
    th, n = s.throw, s.t.shape[0]
    pos, vel = np.asarray(s.sim[environment.i_pos]), np.asarray(s.sim[environment.i_vel])
    tilt = np.asarray(jax.vmap(thrust_tilt)(th.quat))
    yaw = np.asarray(jax.vmap(D.yaw_of)(th.quat))
    assert np.ptp(yaw) > 5.0  # yaw is U(-pi, pi) at every throw scale
    if scale == 0.0:  # at rest at the target in the hover trim (net thrust vertical), trim rpm
        trim = plant_dr.hover_trim()
        np.testing.assert_allclose(tilt, 0.0, atol=1e-7)
        body_tilt = np.degrees(np.asarray(jax.vmap(D.tilt_of)(th.quat)))
        np.testing.assert_allclose(body_tilt, np.degrees(D.tilt_of(D.trim_quat())), atol=1e-9)
        nose = np.asarray(jax.vmap(D.quat_rotate, in_axes=(0, None))(th.quat, jnp.eye(3)[0]))
        np.testing.assert_allclose(nose[:, 2], -np.sin(np.deg2rad(body_tilt)), atol=1e-12)
        for x in (th.pos_offset, th.vel, th.body_rate, th.est_err):
            np.testing.assert_array_equal(x, 0.0)
        np.testing.assert_array_equal(th.rpm_scale, 1.0)
        np.testing.assert_allclose(pos[:, 4:], np.tile(D.TARGET, (n, 1)), atol=1e-12)
        rpm = np.broadcast_to(trim.rpm, (n, 4))
        np.testing.assert_allclose(s.sim[environment.slot("motor_rpm")], rpm, rtol=1e-12)
        np.testing.assert_array_equal(vel, 0.0)
        return
    assert tilt.max() <= np.deg2rad(30.0) * scale + 1e-9
    assert tilt.max() >= np.deg2rad(30.0) * scale * 0.8  # the range is covered
    assert np.abs(th.pos_offset).max() <= scale + 1e-12
    assert np.abs(th.vel).max() <= scale + 1e-12
    assert np.abs(th.body_rate).max() <= 2.0 * scale + 1e-12
    assert np.abs(th.body_rate).max() >= 1.8 * scale
    assert np.abs(th.rpm_scale - 1.0).max() <= 0.2 * scale + 1e-12
    est = np.linalg.norm(np.asarray(th.est_err), axis=-1)
    assert est.max() <= np.deg2rad(5.0) * scale + 1e-12
    # every start passes the 0.3 s ballistic-drop check
    assert np.all(pos[:, 6] + 0.3 * np.asarray(th.vel[:, 2]) - 0.5 * 9.81 * 0.09 >= 0.6 - 1e-9)
    np.testing.assert_allclose(vel[:, 3:], th.vel, atol=1e-12)


@pytest.mark.parametrize("dr_scale,throw_scale", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_reset_ranges(env_truth, dr_scale, throw_scale):
    """dr_scale scales plant/sensor/latency DR only, throw_scale the start throw only."""
    n = 256
    keys = jax.random.split(jax.random.key(1), n)
    reset = jax.jit(jax.vmap(env_truth.reset, in_axes=(0, None, None)))
    s, obs, critic = reset(keys, dr_scale, throw_scale)
    assert np.all(np.isfinite(obs)) and np.all(np.isfinite(critic))
    check_dr(s.dr, dr_scale)
    check_throw(env_truth, s, throw_scale)
    # throw_scale=None means throw_scale = dr_scale
    s_default, _, _ = jax.jit(jax.vmap(env_truth.reset, in_axes=(0, None)))(keys, dr_scale)
    s_same, _, _ = reset(keys, dr_scale, dr_scale)
    for a, b in zip(jax.tree.leaves(s_default), jax.tree.leaves(s_same)):
        np.testing.assert_array_equal(a, b)
    # the sampled values land in the world state
    sim, dr = s.sim, s.dr
    inertia = np.asarray(sim[env_truth.slot("inertia")])
    np.testing.assert_allclose(inertia[:, 6], 2.586 * np.asarray(dr.mass_scale), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(sim[env_truth.slot("act_delay")]), dr.act_delay)
    np.testing.assert_array_equal(np.asarray(sim[env_truth.slot("vio_params")])[:, 5], dr.vio_delay)
    assert np.all(np.asarray(sim[env_truth.slot("sensor_tick")]) % 9 == 0)
    np.testing.assert_array_equal(sim[env_truth.slot("act_fifo")], 0.0)
    np.testing.assert_array_equal(sim[env_truth.slot("p_int")], 0.0)
    buf = np.asarray(sim[env_truth.slot("vio_buf")])
    np.testing.assert_array_equal(buf, np.repeat(buf[:, :1], vio.BUF_LEN, axis=1))
    # rate PID seeded with the first error (zero setpoint) and the hover trim
    pid = np.asarray(sim[env_truth.slot("rate_pid_state")])
    np.testing.assert_array_equal(pid[:, 0], -np.asarray(sim[env_truth.slot("gyro")]))
    np.testing.assert_array_equal(pid[:, 1], np.broadcast_to(policy_io.HOVER_RATE_I, (n, 3)))
    np.testing.assert_array_equal(pid[:, 2], 0.0)


def test_sensor_noise_independent_per_env(env_truth):
    """Same params and throw, different key: different gyro and VIO noise; same key: same."""
    dr = D.nominal_dr()._replace(vio_pos_sigma=jnp.float64(0.02), vio_vel_sigma=jnp.float64(0.05))
    throw = D.nominal_throw()

    def run(key):
        s, _, _ = env_truth.reset_with(key, dr, throw)
        gyro, meas = [], []
        for _ in range(5):
            s, *_ = env_truth.step(s, jnp.zeros(4))
            gyro.append(s.sim[env_truth.slot("gyro")])
            meas.append(s.sim[env_truth.slot("vio_meas")])
        return jnp.stack(gyro), jnp.stack(meas)

    run = jax.jit(jax.vmap(run))
    k = jax.random.split(jax.random.key(2), 2)
    gyro, meas = run(jnp.stack([k[0], k[1], k[0]]))
    gyro, meas = np.asarray(gyro), np.asarray(meas)
    assert np.abs(gyro[0] - gyro[1]).max() > 1e-4  # gyro noise sigma is 0.03 rad/s pre-filter
    assert np.abs(meas[0] - meas[1]).max() > 1e-3
    np.testing.assert_array_equal(gyro[0], gyro[2])
    np.testing.assert_array_equal(meas[0], meas[2])


def body_rates(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """World angular velocity [N, 3] into the body frame of xyzw quaternions [N, 4] (R^T w)."""
    x, y, z, s = q.T
    r = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * s), 2 * (x * z + y * s)], -1),
            np.stack([2 * (x * y + z * s), 1 - 2 * (x * x + z * z), 2 * (y * z - x * s)], -1),
            np.stack([2 * (x * z - y * s), 2 * (y * z + x * s), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    return np.einsum("nji,nj->ni", r, w)


def reward_v3(e_p, speed, cos_tilt, w_xy, w_z, da2, in_tol):
    return (
        1.1
        - 0.25 * np.tanh(e_p / 0.3)
        - 0.08 * np.minimum(e_p, 4.0)
        - 0.10 * np.tanh(speed)
        - 0.20 * (1.0 - cos_tilt)
        - 0.05 * np.tanh(w_xy / 2.0)
        - 0.10 * np.tanh(np.abs(w_z) / 0.5)
        - 0.05 * np.minimum(da2, 1.0)
        + 0.25 * in_tol
    )


# alive infimum of reward v3: every penalty at its bound (|e_p| = 4 m, tilt = 80 deg)
REWARD_INFIMUM = (
    1.1 - 0.25 - 0.08 * 4.0 - 0.10 - 0.20 * (1.0 - np.cos(E.TERM_TILT)) - 0.05 - 0.10 - 0.05
)


def test_termination_and_reward_cases():
    target = jnp.array(D.TARGET)
    level = jnp.array([0.0, 0.0, 0.0, 1.0])
    zero4 = jnp.zeros(4)

    def case(
        tilt_deg=0.0,
        offset=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        omega=(0.0, 0.0, 0.0),
        a=zero4,
        a_prev=zero4,
        finite=True,
    ):
        q = D.quat_from_rotvec(jnp.array([np.deg2rad(tilt_deg), 0.0, 0.0])) if tilt_deg else level
        pos = jnp.concatenate([q, target + jnp.array(offset)])
        wv = jnp.concatenate([jnp.array(omega), jnp.array(vel)])
        return E.evaluate(pos, wv, target, a, a_prev, jnp.bool_(finite))

    m = case()
    assert not m.terminated and m.in_window_ok and float(m.reward) == pytest.approx(1.35)
    m = case(offset=(0.3, 0.4, 0.0), vel=(0.0, 0.0, 1.0), omega=(0.0, 0.0, 2.0), a=jnp.ones(4))
    expected = 1.1 - 0.25 * np.tanh(0.5 / 0.3) - 0.08 * 0.5 - 0.10 * np.tanh(1.0)
    expected -= 0.10 * np.tanh(2.0 / 0.5) + 0.05
    assert float(m.reward) == pytest.approx(expected)
    assert not m.terminated and not m.in_window_ok
    m = case(a=jnp.full(4, 0.1), a_prev=jnp.full(4, -0.1))
    assert float(m.reward) == pytest.approx(1.35 - 0.05 * 0.16)
    assert float(case(tilt_deg=30.0).reward) == pytest.approx(1.1 - 0.2 * (1 - np.cos(np.pi / 6)))
    # the rate split is in the body frame: a yaw about the tilted body z is all w_z
    c30, s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
    m = case(tilt_deg=30.0, omega=(0.0, -s30, c30))  # body z of a 30 deg roll, 1 rad/s
    assert float(m.reward) == pytest.approx(1.1 - 0.2 * (1 - c30) - 0.10 * np.tanh(2.0))
    m = case(tilt_deg=30.0, omega=(1.0, 0.0, 0.0))  # roll rate, body x = world x
    assert float(m.reward) == pytest.approx(1.1 - 0.2 * (1 - c30) - 0.05 * np.tanh(0.5))
    # yaw spin at hover: the 0.5 rad/s tanh scale makes 1.3 rad/s cost 0.099 of the 0.10
    spin = float(case(omega=(0.0, 0.0, 0.6)).reward) - float(case(omega=(0.0, 0.0, 1.3)).reward)
    assert spin == pytest.approx(0.10 * (np.tanh(2.6) - np.tanh(1.2)), rel=1e-9)
    # the linear term keeps a slope where the tanh is flat: 0.08/m out to the 4 m radius
    far = [float(case(offset=(d, 0.0, 0.0)).reward) for d in (2.0, 3.0, 3.9)]
    assert 0.0799 < far[0] - far[1] < 0.081 and 0.0719 < far[1] - far[2] < 0.0721
    # in_tol bonus: +IN_TOL_BONUS exactly when every success criterion holds
    for name, inside, outside in [
        ("offset", (0.1499, 0.0, 0.0), (0.1501, 0.0, 0.0)),
        ("vel", (0.2499, 0.0, 0.0), (0.2501, 0.0, 0.0)),
        ("tilt_deg", 9.99, 10.01),
        ("omega", (0.0, 0.4999, 0.0), (0.0, 0.5001, 0.0)),
        ("omega", (0.0, 0.0, 0.4999), (0.0, 0.0, 0.5001)),
    ]:
        a, b = case(**{name: inside}), case(**{name: outside})
        assert a.in_window_ok and not b.in_window_ok
        assert E.IN_TOL_BONUS < float(a.reward) - float(b.reward) < E.IN_TOL_BONUS + 5e-4, name
    # worst live corner: tilt 79.9 deg, |e_p| 3.99 m, fast, rolling and spinning, full swing
    t = np.deg2rad(79.9)
    m = case(
        tilt_deg=79.9,
        offset=(3.99, 0.0, 0.0),
        vel=(30.0, 0.0, 0.0),
        omega=(10.0, -11.0 * float(np.sin(t)), 11.0 * float(np.cos(t))),  # body (10, 0, 11)
        a=jnp.ones(4),
        a_prev=-jnp.ones(4),
    )
    assert not m.terminated and REWARD_INFIMUM < float(m.reward) < REWARD_INFIMUM + 0.003

    assert not case(tilt_deg=75.0).terminated and case(tilt_deg=85.0).terminated
    assert not case(offset=(3.9, 0.0, 0.0)).terminated and case(offset=(4.1, 0.0, 0.0)).terminated
    assert not case(offset=(0.0, 0.0, -1.6)).terminated  # z = 0.4
    assert case(offset=(0.0, 0.0, -1.8)).terminated  # z = 0.2
    assert not case(omega=(0.0, 14.0, 0.0)).terminated and case(omega=(0.0, 16.0, 0.0)).terminated
    m = case(finite=False)
    assert m.terminated and float(m.reward) == E.TERM_REWARD
    assert float(case(tilt_deg=85.0).reward) == E.TERM_REWARD
    # per-step success thresholds
    assert (
        case(offset=(0.1, 0.0, 0.0)).in_window_ok and not case(offset=(0.2, 0.0, 0.0)).in_window_ok
    )
    assert not case(vel=(0.3, 0.0, 0.0)).in_window_ok
    assert case(tilt_deg=9.0).in_window_ok and not case(tilt_deg=11.0).in_window_ok
    assert not case(omega=(0.6, 0.0, 0.0)).in_window_ok


def test_reward_alive_minimum_analytic():
    """Every reward v3 penalty is bounded (tanh < 1, |e_p| <= TERM_POS, tilt <= TERM_TILT,
    |da|^2 capped at 1), so a live step pays > 0.0647 > 0.03 while terminating pays -10."""
    assert REWARD_INFIMUM == pytest.approx(0.0647296, abs=1e-7)
    assert REWARD_INFIMUM > 0.03
    assert E.TERM_POS == 4.0 and E.TERM_TILT == pytest.approx(np.deg2rad(80.0))
    big = jnp.float64(1e9)
    r = E.reward(E.TERM_POS, big, np.cos(E.TERM_TILT), big, -big, 10.0 * jnp.ones(4), 0.0)
    assert 0.0 < float(r) - REWARD_INFIMUM < 1e-11  # tanh(4 / 0.3) is 1 - 5.2e-12
    zero = jnp.zeros(4)
    assert float(E.reward(0.0, 0.0, 1.0, 0.0, 0.0, zero, 1.0)) == pytest.approx(1.1 + 0.25)


def test_in_tol_bonus_single_source():
    """IN_TOL_BONUS lives in env.py only: env.reward pays it and train.py reads it from there."""
    import train

    zero = jnp.zeros(4)
    on, off = (float(E.reward(0.1, 0.1, 0.99, 0.1, 0.1, zero, b)) for b in (1.0, 0.0))
    assert on - off == pytest.approx(E.IN_TOL_BONUS, abs=1e-15)
    assert E.IN_TOL_BONUS == 0.25
    assert not hasattr(train, "ENV_IN_TOL_BONUS")
    assert "drone_env.IN_TOL_BONUS" in inspect.getsource(train.InTolWeight.step)
    assert "drone_env.IN_TOL_BONUS" in inspect.getsource(train.setup)
    src = inspect.getsource(E.reward)
    assert "IN_TOL_BONUS * in_tol" in src and "0.25 * in_tol" not in src
    assert train.parse_args([]).in_tol_weight is None  # setup resolves it to env.IN_TOL_BONUS


def test_reward_bounded_while_alive():
    """Reward v3 matches an independent formula on 250k random states, stays > REWARD_INFIMUM
    (> 0.03) for every non-terminal state and <= 1.35, and pays the in_tol bonus exactly on the
    success test."""
    n, n_near = 200_000, 50_000
    k = jax.random.split(jax.random.key(13), 8)
    u = jax.random.uniform

    def scales(broad, near):  # broad states, then near-hover states inside the tolerances
        return jnp.concatenate([jnp.full(n, broad), jnp.full(n_near, near)])[:, None]

    m_all = n + n_near
    axis = jax.random.normal(k[0], (m_all, 3)).at[:, 2].set(0.0)
    axis = axis / jnp.linalg.norm(axis, axis=1, keepdims=True)
    tilt = u(k[1], (m_all, 1)) * scales(np.deg2rad(85.0), np.deg2rad(12.0))
    yaw = u(k[7], (m_all, 1), minval=-np.pi, maxval=np.pi) * jnp.array([0.0, 0.0, 1.0])
    q_rp = jax.vmap(D.quat_from_rotvec)(axis * tilt)
    q = jax.vmap(D.quat_mul)(jax.vmap(D.quat_from_rotvec)(yaw), q_rp)
    target = jnp.array(D.TARGET)
    p = target + jax.random.normal(k[2], (m_all, 3)) * u(k[3], (m_all, 1)) * scales(3.0, 0.15)
    v = jax.random.normal(k[4], (m_all, 3)) * scales(20.0, 0.12)
    w = jax.random.normal(k[5], (m_all, 3)) * scales(6.0, 0.25)
    a = u(k[6], (m_all, 2, 4), minval=-1.0, maxval=1.0)
    m = jax.vmap(E.evaluate, in_axes=(0, 0, None, 0, 0, None))(
        jnp.concatenate([q, p], 1),
        jnp.concatenate([w, v], 1),
        target,
        a[:, 0],
        a[:, 1],
        jnp.bool_(True),
    )
    r, term, ok = np.asarray(m.reward), np.asarray(m.terminated), np.asarray(m.in_window_ok)
    assert 0.2 < term[:n].mean() < 0.9  # both sides are sampled
    assert 0.1 < ok[n:].mean() < 0.9
    q, p, v, w, a = (np.asarray(x) for x in (q, p, v, w, a))
    e_p = np.linalg.norm(p - np.asarray(target), axis=1)
    speed, rate = np.linalg.norm(v, axis=1), np.linalg.norm(w, axis=1)
    cos_tilt = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    w_b = body_rates(q, w)
    w_xy = np.hypot(w_b[:, 0], w_b[:, 1])
    in_tol = (e_p < 0.15) & (speed < 0.25) & (cos_tilt > np.cos(np.deg2rad(10.0))) & (rate < 0.5)
    da2 = np.sum((a[:, 0] - a[:, 1]) ** 2, axis=1)
    expected = reward_v3(e_p, speed, cos_tilt, w_xy, w_b[:, 2], da2, in_tol)
    np.testing.assert_allclose(r[~term], expected[~term], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ok, in_tol & ~term)
    alive = r[~term]
    assert alive.min() > REWARD_INFIMUM > 0.03 and alive.max() <= 1.35
    # tilted states: the body yaw rate is not the world z rate
    assert np.abs(w_b[:, 2] - w[:, 2]).max() > 1.0
    np.testing.assert_array_equal(r[term], E.TERM_REWARD)


def test_truncation_and_success_bookkeeping(env_truth):
    steps = env_truth.episode_steps
    hover, _, _ = env_truth.reset_with(jax.random.key(4), D.nominal_dr(), D.nominal_throw())
    step = jax.jit(env_truth.step)
    a0 = jnp.zeros(4)

    # last step of a clean episode: truncated and successful
    s, _, _, r, done, info = step(hover._replace(t=jnp.int32(steps - 1)), a0)
    assert bool(done) and bool(info["truncated"]) and bool(info["success"])
    assert not bool(info["terminated"]) and int(info["t"]) == steps and float(r) > 0.9
    # one earlier failure inside the window spoils success, outside it does not matter
    s, *_, info = step(hover._replace(t=jnp.int32(steps - 1), succ_ok=jnp.bool_(False)), a0)
    assert bool(info["truncated"]) and not bool(info["success"])
    off = env_truth.reset_with(
        jax.random.key(4),
        D.nominal_dr(),
        D.nominal_throw()._replace(pos_offset=jnp.array([0.5, 0, 0])),
    )[0]
    s, *_, info = step(off._replace(t=jnp.int32(10)), a0)
    assert bool(s.succ_ok) and not bool(info["truncated"]) and not bool(info["success"])
    s, *_, info = step(off._replace(t=jnp.int32(steps - E.SUCCESS_WINDOW)), a0)
    assert not bool(s.succ_ok)
    # termination on the last step: terminated, not truncated, not successful, -10
    tumble = D.nominal_throw()._replace(
        quat=D.quat_from_rotvec(jnp.array([np.deg2rad(88.0), 0, 0]))
    )
    bad = env_truth.reset_with(jax.random.key(4), D.nominal_dr(), tumble)[0]
    s, _, _, r, done, info = step(bad._replace(t=jnp.int32(steps - 1)), a0)
    assert bool(done) and bool(info["terminated"])
    assert not bool(info["truncated"]) and not bool(info["success"]) and float(r) == E.TERM_REWARD


@pytest.mark.parametrize("estimator", ["truth", "mekf18", "mekf6"])
def test_smoke_30_ticks_every_component(estimator, request):
    environment = request.getfixturevalue(f"env_{estimator}")
    js = environment.js
    get = js.get
    dr = D.dr_sample(jax.random.key(5), 1.0)._replace(
        act_delay=jnp.int32(2),
        vio_delay=jnp.int32(4),
        vio_pos_sigma=jnp.float64(0.01),
        thrust_scale=jnp.array([1.05, 0.95, 1.0, 1.1]),
    )
    throw = D.throw_sample(jax.random.key(6), 0.7)
    state, obs0, _ = environment.reset_with(jax.random.key(7), dr, throw)
    tick0 = int(get(state.sim, "sensor_tick"))

    # tick-level checks of the action FIFO with a 2-tick delay
    tick = jax.jit(js.step)
    sim = state.sim
    actions = [jnp.array([0.2, -0.1, 0.3, 0.1]), jnp.array([-0.4, 0.5, -0.2, -0.3])]
    applied = []
    clocks, rows = [], []
    for k in range(10):
        a = actions[k % 2]
        sim = js.set(sim, "policy_action", a)
        for _ in range(3):
            sim = tick(sim)
            applied.append(np.asarray(get(sim, "ang_vel_setpoint")))
            clocks.append(float(get(sim, "vio_clock")))
            rows.append(np.asarray(get(sim, "vio_buf"))[0])
        np.testing.assert_array_equal(get(sim, "act_fifo")[0], a)
        np.testing.assert_allclose(get(sim, "policy_obs")[15:19], a)
    applied = np.array(applied)
    scale = policy_io.ACT_SCALE
    # tick 3k + j uses the action written at step k when j >= 2, else the previous one
    for k in range(1, 10):
        np.testing.assert_allclose(applied[3 * k], scale * np.asarray(actions[(k - 1) % 2])[:3])
        np.testing.assert_allclose(applied[3 * k + 1], scale * np.asarray(actions[(k - 1) % 2])[:3])
        np.testing.assert_allclose(applied[3 * k + 2], scale * np.asarray(actions[k % 2])[:3])
    np.testing.assert_allclose(
        get(sim, "collective_cmd"),
        policy_io.HOVER_COLLECTIVE + policy_io.COLLECTIVE_SCALE * actions[1][3],
    )
    # VIO: a new sample every 10 ticks, held in between; the read is 4 ticks behind
    clocks, rows = np.array(clocks), np.array(rows)
    assert set(np.unique(clocks)) == set(range(10))
    for i in range(1, 30):
        if clocks[i] != 0.0:
            np.testing.assert_array_equal(rows[i], rows[i - 1])
    buf = np.asarray(get(sim, "vio_buf"))
    np.testing.assert_array_equal(get(sim, "vio_meas"), buf[4])
    assert int(get(sim, "sensor_tick")) == tick0 + 90
    # all state finite, integral bounded, observations wired to their sources
    for name, x in zip(js.names, sim):
        assert np.all(np.isfinite(np.asarray(x, dtype=np.float64))), name
    assert np.abs(np.asarray(get(sim, "p_int"))).max() <= policy_io.P_INT_LIMIT
    assert float(np.abs(np.asarray(get(sim, "p_int"))).max()) > 0.0
    obs = np.asarray(get(sim, "policy_obs"))
    np.testing.assert_allclose(np.linalg.norm(obs[0:3]), 1.0, atol=1e-9)
    np.testing.assert_array_equal(obs[3:6], get(sim, "gyro"))
    critic = np.asarray(get(sim, "critic_obs"))
    np.testing.assert_array_equal(critic[: E.OBS_DIM], obs)
    feats = critic[E.OBS_DIM + 12 :]
    assert feats[0] == pytest.approx((float(dr.mass_scale) - 1) / 0.1)
    np.testing.assert_allclose(feats[4:8], (np.asarray(dr.thrust_scale) - 1) / 0.05)
    assert feats[-4] == pytest.approx(1.0) and feats[-3] == pytest.approx(4 / 6)
    # per-motor thrust scale reaches the plant
    kt = np.asarray(get(sim, "thrust")) / np.asarray(get(sim, "motor_rpm")) ** 2
    np.testing.assert_allclose(kt / kt[2], np.asarray(dr.thrust_scale), rtol=1e-12)
    # the estimator runs and tracks
    q_hat = np.asarray(get(sim, "attitude_estimate"))
    q = np.asarray(get(sim, "world_pos"))[:4]
    err = 2 * np.arccos(min(1.0, abs(float(np.dot(q_hat, q)))))
    assert err < np.deg2rad(5.0)
    if estimator != "truth":
        assert not np.allclose(q_hat, np.asarray(state.sim[js.slot("attitude_estimate")]))
    # the env step runs over the same state
    s, o, c, r, d, info = jax.jit(environment.step)(state._replace(sim=sim), actions[0])
    assert np.all(np.isfinite(o)) and np.isfinite(float(r))


def mixer_saturated(mi: np.ndarray) -> np.ndarray:
    """Per row of motor_input: does motors.motor_input_to_pwm clamp yaw, scale roll/pitch/yaw
    or clip a motor at 0 or 1 linear throttle."""
    rf, pf, yf, _ = Config.GLOBAL.frame.motor_matrix
    roll, pitch, yaw, thr = (mi[:, k : k + 1] for k in range(4))
    out = roll * rf + pitch * pf
    room = np.where(yaw * yf > 0, 1.0 - (out + 0.5), out + 0.5)
    yaw_allowed = np.min(np.clip(room, 0.0, None) / np.abs(yf), axis=1)
    out = out + yaw * yf
    lo, hi = out.min(1), out.max(1)
    avg_max = np.clip(
        0.5 * Config.GLOBAL.control.motor_thrust_hover + 0.5 * thr[:, 0], thr[:, 0], 1
    )
    return (
        (np.abs(yaw[:, 0]) > yaw_allowed)
        | (hi - lo > 1.0)
        | (avg_max + lo < 0.0)
        | (thr[:, 0] + hi > 1.0)
        | (thr[:, 0] + lo < 0.0)
    )


@pytest.mark.parametrize("rate_pid", ["dmeas", "stock"])
def test_reset_has_no_pid_kick(rate_pid, request):
    """The rate PID starts from the first error with zero derivative: no error jump on tick 1
    and no mixer saturation over the first 10 ticks of a mild throw with a = 0 (the old zeroed
    PID state saturates it through the derivative term)."""
    environment = request.getfixturevalue("env_truth" if rate_pid == "dmeas" else "env_truth_stock")
    assert environment.rate_pid == rate_pid
    js = environment.js
    throw = D.nominal_throw()._replace(
        quat=D.quat_from_rotvec(jnp.array([np.deg2rad(10.0), 0.0, 0.0])),
        body_rate=jnp.array([0.5, -0.5, 0.25]),
    )
    state, _, _ = environment.reset_with(jax.random.key(14), D.nominal_dr(), throw)

    def first_ticks(sim):
        def body(sim, _):
            sim = js.step(sim)
            return sim, (js.get(sim, "motor_input"), js.get(sim, "rate_pid_state"))

        return jax.lax.scan(body, sim, None, length=10)[1]

    first_ticks = jax.jit(first_ticks)
    mi, pid = (np.asarray(x) for x in first_ticks(state.sim))
    pid0 = np.asarray(js.get(state.sim, "rate_pid_state"))
    np.testing.assert_allclose(pid[0, 0], pid0[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(pid[0, 2], 0.0, atol=1e-9)
    assert not mixer_saturated(mi).any()
    assert np.abs(mi[:, :3]).max() < 0.5
    old = js.set(state.sim, "rate_pid_state", jnp.zeros((3, 3)))
    mi_old, _ = (np.asarray(x) for x in first_ticks(old))
    assert mixer_saturated(mi_old).any() and np.abs(mi_old[:, :3]).max() > 2.0


def test_exploration_noise_survives(env_truth, env_truth_stock):
    """Gaussian actions (std 0.1 per channel) around a = 0 at nominal do not crash within 1.5 s
    with the default derivative-on-measurement PID; with the stock PID the D kick of every
    100 Hz action step saturates the mixer and every episode hits z < 0.3 m."""
    assert env_truth.rate_pid == "dmeas"  # the default
    n, steps = 64, 150

    def run(environment):
        def episode(key):
            k_reset, k_act = jax.random.split(key)
            s, _, _ = environment.reset(k_reset, 0.0, 0.0)

            def body(c, t):
                s, done = c
                a = 0.1 * jax.random.normal(jax.random.fold_in(k_act, t), (4,))
                s, _, _, _, d, _ = environment.step(s, a)
                return (s, done | d), None

            return jax.lax.scan(body, (s, jnp.bool_(False)), jnp.arange(steps))[0][1]

        return np.asarray(jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(19), n)))

    assert not run(env_truth).any()
    assert run(env_truth_stock).all()


def test_hover_trim_is_torque_free():
    """plant_dr.hover_trim from the motor geometry: at the trim rpm the plant's body torque
    (thrust moments plus reaction torques) vanishes, |net thrust| = m g, and the trim attitude
    turns the net thrust vertical with a nose-down pitch of ~1.89 deg (rear motors canted back)."""
    cfg = Config.GLOBAL
    trim = plant_dr.hover_trim()
    kt, kq = plant_dr.motor_constants()
    d = np.asarray(cfg.motor_thrust_directions)
    thrust = kt * trim.rpm**2
    yaw = (kq / kt) * cfg.frame.motor_matrix[2] * thrust
    torque = np.cross(cfg.motor_positions, d).T @ thrust + d.T @ yaw
    force = d.T @ thrust
    np.testing.assert_allclose(torque, 0.0, atol=1e-12)
    assert np.linalg.norm(force) == pytest.approx(cfg.mass * 9.81, rel=1e-12)
    np.testing.assert_allclose(force / np.linalg.norm(force), trim.axis, atol=1e-15)
    np.testing.assert_allclose(D.quat_rotate(D.trim_quat(), trim.axis), [0.0, 0.0, 1.0], atol=1e-15)
    pitch = np.degrees(2.0 * np.arcsin(trim.quat[1]))
    assert 1.85 < pitch < 1.95 and trim.quat[0] == pytest.approx(0) and trim.quat[2] == 0.0


def test_trim_attitude_net_thrust_vertical(env_truth):
    """The throw-0 start (q_yaw * q_trim) makes the net thrust of the settled a = 0 hover
    vertical within 0.05 deg: the per-motor thrust of the sim, summed along the motor directions
    over 0.5-1 s of 32 episodes, rotated by each episode's start attitude. The level start of
    env v1 leaves it 1.89 deg off."""
    d = jnp.asarray(Config.GLOBAL.motor_thrust_directions)
    i_thrust = env_truth.slot("thrust")

    def episode(key):
        state, _, _ = env_truth.reset(key, 0.0, 0.0)
        q0 = state.sim[env_truth.i_pos][:4]

        def body(s, _):
            s, *_ = env_truth.step(s, jnp.zeros(4))
            f = s.sim[i_thrust] @ d
            return s, f / jnp.linalg.norm(f)

        f = jax.lax.scan(body, state, None, length=100)[1][50:].mean(0)
        return D.quat_rotate(q0, f), f

    up, body = (
        np.asarray(x) for x in jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(20), 32))
    )

    def from_vertical(v):
        return np.degrees(np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1.0, 1.0)))

    assert from_vertical(up).max() < 0.05, from_vertical(up).max()
    assert from_vertical(body).min() > 1.8  # with a level start (body z up)


def test_trim_start_holds_with_frozen_motors(env_truth):
    """Noise-free plant check of the trim: with the motors frozen at the reset rpm (motor_tau
    1e12, so the rate loop cannot act), the throw-0 start at any yaw stays put for 5 s (net
    thrust vertical, |F| = m g, torque-free); a level start is pushed metres back."""
    frozen = D.nominal_dr()._replace(motor_tau=jnp.float64(1e12))

    def run(quat):
        throw = D.nominal_throw()._replace(quat=quat)
        s, _, _ = env_truth.reset_with(jax.random.key(22), frozen, throw)

        def body(s, _):
            s, *_ = env_truth.step(s, jnp.zeros(4))
            return s, s.sim[env_truth.i_pos][4:] - s.target

        return jax.lax.scan(body, s, None, length=env_truth.episode_steps)[1]

    yaw = jax.vmap(D.quat_from_rotvec)(jnp.array([[0.0, 0.0, y] for y in (0.0, 1.0, -2.5)]))
    trim = jax.vmap(D.quat_mul, in_axes=(0, None))(yaw, D.trim_quat())
    quats = jnp.concatenate([trim, jnp.array([[0.0, 0.0, 0.0, 1.0]])])
    drift = np.abs(np.asarray(jax.jit(jax.vmap(run))(quats))).max(axis=(1, 2))
    assert drift[:3].max() < 1e-8, drift
    assert drift[3] > 1.0, drift


def test_zero_action_hovers_in_place(env_truth):
    """a = 0 at nominal (dr 0, throw 0, sensor noise on) starts in the trim and holds: altitude
    within 2 cm (HOVER_COLLECTIVE calibrated at the trim) and no drift along the start heading's
    pitch axis, where the canted motors pushed a level start ~4 m back in 5 s. a = 0 only holds
    the measured rate at zero, so the stock gyro error integrates into attitude: the roll bias
    (~0.4 m sideways at 5 s) plus white noise and the bias random walk, which sensors.Noise
    draws from the same per-tick key (sd ~0.3 m per axis at 5 s). The hover is tight for 2.5 s."""
    n = 32

    def episode(key):
        state, _, _ = env_truth.reset(key, 0.0, 0.0)
        yaw0 = D.yaw_of(state.sim[env_truth.i_pos][:4])

        def body(s, _):
            s, *_ = env_truth.step(s, jnp.zeros(4))
            p = s.sim[env_truth.i_pos]
            return s, (policy_io.to_heading(yaw0, p[4:] - s.target), thrust_tilt(p[:4]))

        return jax.lax.scan(body, state, None, length=env_truth.episode_steps)[1]

    e, tilt = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(15), n))
    e, tilt = np.asarray(e), np.degrees(np.asarray(tilt))
    dz = e[:, :, 2]
    assert np.abs(dz.mean(0)).max() < 0.02
    assert np.median(np.abs(dz).max(1)) < 0.03
    assert np.linalg.norm(e[:, :250], axis=-1).max(1).max() < 0.3
    assert abs(e[:, -1, 0].mean()) < 0.2  # a level start: ~ -4 m
    assert np.median(tilt.max(1)) < 1.5 and np.percentile(tilt.max(1), 90) < 2.0


def test_p_int_gated_and_clipped(env_truth):
    """p_int integrates the VIO position error only while |error| < P_INT_GATE (1 m) and holds
    otherwise, clipped to +-P_INT_LIMIT per world axis."""
    target = jnp.array(D.TARGET)
    dt = Config.GLOBAL.dt
    assert policy_io.P_INT_GATE == 1.0 and policy_io.P_INT_LIMIT == 1.0

    def update(p_int, e):
        meas = jnp.concatenate([target + jnp.asarray(e), jnp.zeros(4)])
        return np.asarray(policy_io.update_p_int(jnp.asarray(p_int), meas, target))

    p0 = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(update(p0, [0.5, 0.5, 0.5]), p0 + 0.5 * dt, rtol=0, atol=1e-15)
    np.testing.assert_allclose(update(p0, [0.0, 0.99, 0.0]), p0 + [0, 0.99 * dt, 0], atol=1e-15)
    np.testing.assert_array_equal(update(p0, [0.0, 1.01, 0.0]), p0)
    np.testing.assert_array_equal(update(p0, [0.8, 0.0, -0.7]), p0)
    np.testing.assert_array_equal(update([1.0, -1.0, 0.0], [0.5, -0.5, 0.0]), [1.0, -1.0, 0.0])
    # in the env: a start 1.5 m off holds p_int at 0, 0.5 m off integrates it
    step = jax.jit(env_truth.step)
    for offset, moving in ((1.5, False), (0.5, True)):
        throw = D.nominal_throw()._replace(pos_offset=jnp.array([offset, 0.0, 0.0]))
        s, _, _ = env_truth.reset_with(jax.random.key(21), D.nominal_dr(), throw)
        for _ in range(10):
            s, obs, *_ = step(s, jnp.zeros(4))
        p_int = np.asarray(s.sim[env_truth.slot("p_int")])
        if moving:
            assert p_int[0] == pytest.approx(offset * 30 * dt, rel=0.02)
        else:
            np.testing.assert_array_equal(p_int, 0.0)
            np.testing.assert_array_equal(np.asarray(obs)[12:15], 0.0)


def test_defaults():
    """mekf6 is the default estimator, dmeas the default rate PID."""
    params = inspect.signature(E.DroneEnv).parameters
    assert params["estimator"].default == "mekf6" and params["rate_pid"].default == "dmeas"


def test_mekf6_reset(env_mekf6):
    """reset writes the mekf6 estimator state: q_hat = truth * exp(est_err) and the filter's
    own reset values (mekf6.reset_values)."""
    import mekf6

    throw = D.throw_sample(jax.random.key(16), 1.0)
    state, _, _ = env_mekf6.reset_with(
        jax.random.key(17), D.dr_sample(jax.random.key(18), 1.0), throw
    )
    get = env_mekf6.js.get
    q_hat = D.quat_mul(throw.quat / jnp.linalg.norm(throw.quat), D.quat_from_rotvec(throw.est_err))
    np.testing.assert_allclose(get(state.sim, "attitude_estimate"), q_hat, atol=1e-12)
    assert float(jnp.linalg.norm(throw.est_err)) > 0.0
    expected = dict(mekf6.reset_values(jnp.asarray(get(state.sim, "attitude_estimate"))))
    assert {"mekf6_bias", "mekf6_cov"} <= set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(get(state.sim, name), value, rtol=0, atol=1e-12, err_msg=name)
    # the filter state moves once the episode runs
    step = jax.jit(env_mekf6.step)
    s = state
    for _ in range(5):
        s, *_ = step(s, 0.2 * jnp.ones(4))
    assert not np.allclose(get(s.sim, "mekf6_cov"), expected["mekf6_cov"])
    assert not np.allclose(get(s.sim, "attitude_estimate"), q_hat)


def test_spawned_world_matches_reset_with(env_truth):
    """drone_init through spawn (the Cranelift/editor path) gives the same initial state as
    reset_with writes into the JAX state."""
    key = jax.random.key(8)
    dr = D.dr_sample(jax.random.key(9), 1.0)
    throw = D.throw_sample(jax.random.key(10), 1.0)
    state, _, _ = env_truth.reset_with(key, dr, throw)
    init = {k: np.asarray(v) for k, v in policy_world.drone_init(dr, throw, key, "truth").items()}
    w, _ = policy_world.world(1, "truth", [init])
    js = JaxStep(w, policy_world.system("truth"))
    assert js.names == env_truth.js.names
    for name, a, b in zip(js.names, js.init, state.sim):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_unknown_init_names_raise(env_truth, env_truth_stock, monkeypatch):
    """An init value with no state slot (a misspelled name) raises in reset_with and spawn; only
    rate_sp_prev is skipped, by the stock-PID pipeline that has no slot for it."""
    assert env_truth.js.has("rate_sp_prev") and not env_truth_stock.js.has("rate_sp_prev")
    args = (jax.random.key(0), D.nominal_dr(), D.nominal_throw())
    env_truth_stock.reset_with(*args)
    drone_init = policy_world.drone_init
    monkeypatch.setattr(
        policy_world, "drone_init", lambda *a: {**drone_init(*a), "rate_sp_prv": jnp.zeros(3)}
    )
    for environment in (env_truth, env_truth_stock):
        with pytest.raises(KeyError, match="rate_sp_prv"):
            environment.reset_with(*args)
    with pytest.raises(ValueError, match="rate_sp_prv"):
        policy_world.world(1, "truth", [{"rate_sp_prv": np.zeros(3)}])


def test_multi_drone_exec_matches_env(env_truth):
    """A 4-drone world built from drone_init and run as an Exec (jax-cpu) reproduces each
    drone's single-env JAX rollout: per-entity params, noise streams and delays stay separate."""
    n, steps = 4, 10
    keys = jax.random.split(jax.random.key(12), 3 * n).reshape(n, 3)
    actions = 0.3 * np.sin(np.arange(4 * n).reshape(n, 4) + 1.0)
    inits, expected = [], []
    step = jax.jit(env_truth.step)
    for i in range(n):
        dr = D.dr_sample(keys[i, 0], 1.0)
        throw = D.throw_sample(keys[i, 1], 1.0)
        init = policy_world.drone_init(dr, throw, keys[i, 2], "truth")
        init["policy_action"] = jnp.asarray(actions[i])
        inits.append({k: np.asarray(v) for k, v in init.items()})
        s, _, _ = env_truth.reset_with(keys[i, 2], dr, throw)
        for _ in range(steps):
            s, obs, *_ = step(s, jnp.asarray(actions[i]))
        expected.append((np.asarray(s.sim[env_truth.i_pos]), np.asarray(s.sim[env_truth.i_obs])))
    w, _ = policy_world.world(n, "truth", inits)
    ex = w.build(policy_world.system("truth"), simulation_rate=300.0, backend="jax-cpu")
    ex.run(steps * E.TICKS_PER_STEP, show_progress=False)
    cols = [f"drone_{i}.{c}" for i in range(n) for c in ("world_pos", "policy_obs")]
    df = ex.history(cols)
    for i, (pos, obs) in enumerate(expected):
        np.testing.assert_allclose(
            df[f"drone_{i}.world_pos"].to_numpy()[-1], pos, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            df[f"drone_{i}.policy_obs"].to_numpy()[-1], obs, rtol=0, atol=1e-9
        )
