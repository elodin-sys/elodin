import sys
import typing as ty
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drone"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import elodin as el
import jax
import jax.numpy as jnp
import mekf6
import numpy as np
import pytest
import sim
from vehicles import TALON_QUAD_CONFIG

# Expected tensors and scenarios from the #[cfg(test)] module of fsw/roci/adcs/src/mekf.rs.
ROCI_EXPECTED_Q = np.array(
    [
        [8.33352623e-07, 0.0, 0.0, 3.47222222e-09, 0.0, 0.0],
        [0.0, 8.33352623e-07, 0.0, 0.0, 3.47222222e-09, 0.0],
        [0.0, 0.0, 8.33352623e-07, 0.0, 0.0, 3.47222222e-09],
        [3.47222222e-09, 0.0, 0.0, 8.33333333e-07, 0.0, 0.0],
        [0.0, 3.47222222e-09, 0.0, 0.0, 8.33333333e-07, 0.0],
        [0.0, 0.0, 3.47222222e-09, 0.0, 0.0, 8.33333333e-07],
    ]
)
ROCI_YQY = np.array(
    [
        [8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0, 0.0],
        [0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09, 0.0],
        [0.0, 0.0, 8.33352623e-07, 0.0, 0.0, -3.4722e-09],
        [-3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0, 0.0],
        [0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07, 0.0],
        [0.0, 0.0, -3.4722e-09, 0.0, 0.0, 8.3333e-07],
    ]
)
ROCI_PROPAGATED_P = np.array(
    [
        [1.00007028e00, 0.0, 0.0, -8.33333681e-03, 0.0, 0.0],
        [0.0, 1.00007028e00, 5.55768107e-19, 0.0, -8.33324036e-03, -3.47220213e-05],
        [0.0, 5.55768107e-19, 1.00007028e00, 0.0, 3.47220213e-05, -8.33324036e-03],
        [-8.33333681e-03, 0.0, 0.0, 1.00000083e00, 0.0, 0.0],
        [0.0, -8.33324036e-03, 3.47220213e-05, 0.0, 1.00000083e00, 0.0],
        [0.0, -3.47220213e-05, -8.33324036e-03, 0.0, 0.0, 1.00000083e00],
    ]
)

# Final states printed by roci-adcs (State::estimate_attitude, cargo run against fsw/roci/adcs)
# for scenario 1 (= roci test_mekf, step 300) and scenario 2 (_drone_like_scenario, step 900).
RUST_S1 = {
    "q": [0.4429521194559683, 0.1834124553391549, 0.3359399302226897, 0.8107389557545034],
    "b": [3.2033808406338073e-06, 6.666677442573682e-05, -0.0011229299050264602],
    "diag_p": [
        3.371794449184112e-05,
        2.58318396676618e-05,
        2.9951778768082985e-05,
        0.0001276095046745126,
        0.0001258273714653958,
        0.00012435969817809687,
    ],
}
RUST_S2 = {
    "q": [0.06949889448789534, 0.0816156836136565, 0.15562350493037777, 0.9819827435176346],
    "b": [0.009886981498142836, -0.020246972074724867, 0.005086362773318638],
    "diag_p": [
        3.4885291011755164e-06,
        1.12856828022181e-05,
        2.877828127256395e-06,
        5.028802715800489e-06,
        7.929318794161455e-06,
        4.802634137996767e-06,
    ],
}


@pytest.fixture
def talon():
    TALON_QUAD_CONFIG.set_as_global()
    return TALON_QUAD_CONFIG


def axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def _rot_inv(q, v) -> np.ndarray:
    # nox `q.inverse() * v`
    return np.asarray(mekf6.quat_rotate(mekf6.quat_inverse(jnp.asarray(q)), jnp.asarray(v)))


def _unit(v) -> np.ndarray:
    return v / np.linalg.norm(v)


def body_vec(q, ref) -> np.ndarray:
    return _unit(_rot_inv(q, ref))


estimate_attitude = jax.jit(mekf6.estimate_attitude, static_argnums=(5,))


def test_calculate_covariance():
    q = mekf6.calculate_covariance(np.full(3, 0.01), np.full(3, 0.01), 1.0 / 120.0)
    np.testing.assert_allclose(q, ROCI_EXPECTED_Q, atol=1e-5)
    np.testing.assert_allclose(q, ROCI_EXPECTED_Q, rtol=1e-6, atol=1e-15)


def test_process_noise_matches_roci_yqy():
    yqy = mekf6.process_noise(np.full(3, 0.01), np.full(3, 0.01), 1.0 / 120.0)
    np.testing.assert_allclose(yqy, ROCI_YQY, atol=1e-4)
    np.testing.assert_allclose(yqy, ROCI_YQY, rtol=1e-4, atol=1e-15)


def test_propagate_state_covariance():
    out = mekf6.propagate_state_covariance(
        jnp.identity(6), jnp.array([1.0, 0.0, 0.0]), jnp.asarray(ROCI_YQY), 1.0 / 120.0
    )
    np.testing.assert_allclose(out, ROCI_PROPAGATED_P, atol=1e-6)


def test_propagate_quaternion():
    q = mekf6.propagate_quaternion(
        jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]), 1.0 / 60.0
    )
    np.testing.assert_allclose(q, [0.00833324, 0.0, 0.0, 0.99996528], atol=1e-5)


def test_small_rate_branches():
    dt = 1.0 / 300.0
    q = jnp.asarray(axis_angle([1.0, 2.0, 3.0], 0.3))
    omega = jnp.array([3e-6, -4e-6, 5e-6])  # |omega| < 1e-5: roci keeps q and uses Phi = [I -dt I]
    np.testing.assert_array_equal(mekf6.propagate_quaternion(q, omega, dt), q)
    p = mekf6.propagate_state_covariance(jnp.identity(6), omega, jnp.zeros((6, 6)), dt)
    phi = np.block([[np.eye(3), -dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    np.testing.assert_allclose(p, phi @ phi.T, atol=1e-15)


def test_mekf_roci_scenario():
    ref_a, ref_b = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    q = axis_angle([0.0, 0.0, 1.0], np.pi / 4.0)
    body_a, body_b = body_vec(q, ref_a), body_vec(q, ref_b)
    dt = 1.0 / 120.0
    yqy = mekf6.process_noise(np.full(3, 0.01), np.full(3, 0.01), dt)
    np.testing.assert_allclose(yqy, ROCI_YQY, atol=1e-4)
    q_hat, b_hat, p = jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.zeros(3), jnp.identity(6)
    refs, sigmas = (ref_a, ref_b), (0.03, 0.03)
    for _ in range(180):
        q_hat, b_hat, p, _ = estimate_attitude(
            q_hat, b_hat, p, jnp.zeros(3), yqy, dt, (body_a, body_b), refs, sigmas
        )
    np.testing.assert_allclose(q_hat, q, atol=1e-3)
    for _ in range(120):
        q = np.asarray(mekf6.integrate_body(jnp.asarray(q), jnp.array([1.0 / 120.0, 0.0, 0.0])))
        body_a, body_b = body_vec(q, ref_a), body_vec(q, ref_b)
        q_hat, b_hat, p, _ = estimate_attitude(
            q_hat, b_hat, p, jnp.array([1.0, 0.0, 0.0]), yqy, dt, (body_a, body_b), refs, sigmas
        )
    np.testing.assert_allclose(b_hat, np.zeros(3), atol=1e-2)
    np.testing.assert_allclose(q_hat, q, atol=1e-3)
    np.testing.assert_allclose(q_hat, RUST_S1["q"], atol=1e-9)
    np.testing.assert_allclose(b_hat, RUST_S1["b"], atol=1e-9)
    np.testing.assert_allclose(np.diag(p), RUST_S1["diag_p"], atol=1e-9)


@pytest.mark.slow
def test_matches_roci_drone_like_scenario():
    # biased gyro, drone refs, deterministic pseudo-noise; replicated step for step in Rust
    dt = 1.0 / 300.0
    up, north = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    q = axis_angle([1.0, 2.0, 3.0], 0.4)
    bias = np.array([0.01, -0.02, 0.005])
    yqy = mekf6.process_noise(np.full(3, 1e-3), np.full(3, 2e-3), dt)
    q_hat, b_hat, p = jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.zeros(3), jnp.identity(6)
    for k in range(900):
        t = float(k)
        w = np.array([0.8 * np.sin(0.05 * t), -0.5 * np.cos(0.03 * t), 0.3 * np.sin(0.07 * t + 1)])
        q = np.asarray(mekf6.integrate_body(jnp.asarray(q), jnp.asarray(w * dt)))
        na = np.array([np.sin(1.3 * t), np.cos(1.7 * t), np.sin(2.1 * t)]) * 0.02
        nm = np.array([np.cos(0.9 * t), np.sin(2.3 * t), np.cos(1.1 * t)]) * 0.01
        meas = (_unit(_rot_inv(q, up) + na), _unit(_rot_inv(q, north) + nm))
        q_hat, b_hat, p, _ = estimate_attitude(
            q_hat, b_hat, p, jnp.asarray(w + bias), yqy, dt, meas, (up, north), (0.05, 0.02)
        )
    np.testing.assert_allclose(q_hat, RUST_S2["q"], atol=1e-9)
    np.testing.assert_allclose(b_hat, RUST_S2["b"], atol=1e-9)
    np.testing.assert_allclose(np.diag(p), RUST_S2["diag_p"], atol=1e-9)
    np.testing.assert_allclose(b_hat, bias, atol=1e-3)


def test_inverse3():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(3, 3))
        m = a @ a.T + 0.1 * np.eye(3)
        np.testing.assert_allclose(mekf6.inverse3(jnp.asarray(m)), np.linalg.inv(m), rtol=1e-9)
    singular = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mekf6.inverse3(jnp.asarray(singular)), np.eye(3))
    # the fallback is relative to scale: tiny but well-conditioned matrices still invert
    for scale in (1e-5, 160e-6**2, 1e4):
        np.testing.assert_allclose(
            mekf6.inverse3(jnp.eye(3) * scale), np.eye(3) / scale, rtol=1e-12
        )
    for bad in (np.diag([1.0, 1.0, 1e-13]), -np.eye(3), np.full((3, 3), np.nan)):
        np.testing.assert_array_equal(mekf6.inverse3(jnp.asarray(bad)), np.eye(3))


def test_accel_sigma():
    s0, k = 0.3, 2.0
    at_rest = mekf6.accel_sigma(jnp.array([0.0, 0.0, 1.0]), jnp.zeros(3), s0, k)
    np.testing.assert_allclose(at_rest, s0, rtol=1e-15)
    dev = mekf6.accel_sigma(jnp.array([0.0, 0.0, 1.05]), jnp.zeros(3), s0, k)
    rate = mekf6.accel_sigma(jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 0.0]), s0, k)
    np.testing.assert_allclose(dev, s0 * (1.0 + k * 0.05 / mekf6.ACCEL_DEV_SCALE), rtol=1e-12)
    np.testing.assert_allclose(rate, s0 * (1.0 + k * 1.0 / mekf6.ACCEL_RATE_SCALE), rtol=1e-12)
    capped = mekf6.accel_sigma(jnp.array([0.0, 0.0, 0.2]), jnp.full(3, 20.0), s0, k)
    np.testing.assert_allclose(capped, s0 * (1.0 + k * mekf6.ACCEL_INFLATE_MAX), rtol=1e-15)
    # gain 0 is the fixed-sigma filter exactly; at rest the adaptive one matches it too
    dt = 1.0 / 300.0
    q = jnp.asarray(axis_angle([0.3, -0.2, 1.0], 0.2))
    b, p = jnp.zeros(3), jnp.asarray(mekf6.P0)
    accel, mag = jnp.asarray(_rot_inv(q, [0.0, 0.0, 1.0])), jnp.asarray(_rot_inv(q, [0, 1.0, 0]))
    fixed = mekf6.Sigmas(gyro=1e-3, gyro_bias=1e-3, accel=s0, mag=0.02)
    for gyro in (jnp.zeros(3), jnp.array([0.4, -0.3, 0.2])):
        a = mekf6.estimate(q, b, p, gyro, accel, mag, fixed, dt)
        sigma = float(mekf6.accel_sigma(accel, gyro, s0, k))
        c = mekf6.estimate(q, b, p, gyro, accel, mag, fixed._replace(accel=sigma), dt)
        d = mekf6.estimate(q, b, p, gyro, accel, mag, fixed._replace(accel_gain=k), dt)
        for y, z in zip(c, d):
            np.testing.assert_allclose(z, y, atol=1e-15)
        if not float(jnp.dot(gyro, gyro)):
            for x, z in zip(a, d):
                np.testing.assert_allclose(z, x, atol=1e-15)


def test_reset_values():
    q = jnp.asarray(axis_angle([1.0, 0.0, 0.0], 0.1))
    init = mekf6.reset_values(q)
    assert set(init) == {"attitude_estimate", "mekf6_bias", "mekf6_cov"}
    np.testing.assert_array_equal(init["attitude_estimate"], q)
    np.testing.assert_array_equal(init["mekf6_bias"], np.zeros(3))
    np.testing.assert_array_equal(init["mekf6_cov"], mekf6.P0)
    batched = jax.vmap(mekf6.reset_values)(jnp.stack([q, q]))
    assert batched["mekf6_cov"].shape == (2, 6, 6)


def test_skipped_measurements_and_reset():
    dt = 1.0 / 300.0
    s = mekf6.Sigmas(gyro=1e-3, gyro_bias=1e-3, accel=0.1, mag=0.02)
    q = jnp.asarray(axis_angle([0.3, -0.2, 1.0], 0.5))
    b, p = jnp.array([0.001, -0.002, 0.0005]), jnp.asarray(mekf6.P0)
    gyro, mag = jnp.array([0.1, -0.2, 0.3]), jnp.asarray(_rot_inv(q, [0.0, 1.0, 0.0]))
    yqy = mekf6.process_noise(np.full(3, s.gyro), np.full(3, s.gyro_bias), dt)
    mag_only = mekf6.estimate_attitude(q, b, p, gyro, yqy, dt, (mag,), (mekf6.MAG_REF,), (s.mag,))
    for sigmas in (s, s._replace(accel_gain=4.0)):
        for accel in (jnp.zeros(3), jnp.array([0.0, 0.0, 0.3]), jnp.array([0.0, 0.0, 1.6])):
            q1, b1, p1, gyro_est = mekf6.estimate(q, b, p, gyro, accel, mag, sigmas, dt)
            np.testing.assert_allclose(q1, mag_only[0], atol=1e-15)
            np.testing.assert_allclose(b1, mag_only[1], atol=1e-15)
            np.testing.assert_allclose(p1, mag_only[2], atol=1e-15)
            np.testing.assert_allclose(gyro_est, gyro - b1, atol=1e-15)
    q2, b2, p2, g2 = jax.jit(mekf6.estimate, static_argnums=(6, 7))(
        q, b, p * jnp.nan, gyro, jnp.array([0.0, 0.0, 1.0]), mag, s, dt
    )
    np.testing.assert_array_equal(q2, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(b2, np.zeros(3))
    np.testing.assert_array_equal(p2, mekf6.P0)
    np.testing.assert_array_equal(g2, np.zeros(3))
    # roci's own reset (estimate_attitude parity) keeps P = I
    roci = mekf6.reset_if_invalid(q, b, p * jnp.nan, gyro)
    np.testing.assert_array_equal(roci[2], np.eye(6))


def test_sim_sigmas(talon):
    s = mekf6.sim_sigmas()
    np.testing.assert_allclose(s.gyro, np.sqrt(0.001 / 900.0))
    np.testing.assert_allclose(s.gyro_bias, np.sqrt(0.001 / 900.0))
    np.testing.assert_allclose(s.mag, 0.01 * np.sqrt(3.0))
    assert (s.accel, s.accel_gain) == (mekf6.ACCEL_SIGMA, mekf6.ACCEL_SIGMA_GAIN)
    # converged innovations S ~ sigma^2 I must invert, not hit the identity fallback
    for sigma in (s.accel, s.mag, 160e-6):
        np.testing.assert_allclose(
            mekf6.inverse3(jnp.eye(3) * sigma**2), np.eye(3) / sigma**2, rtol=1e-12
        )


def _pure_step(jax_sim):
    tick = next(d for _, d, name in jax_sim.map if name == "tick")

    def step(state):
        state = list(state)
        state[tick] = state[tick] + jnp.ones((), state[tick].dtype)
        out = jax_sim.py_sim(*state)
        new = list(state)
        for c, d, _ in jax_sim.map:
            new[d] = jnp.asarray(out[c]).astype(state[d].dtype)
        return tuple(new)

    return step


def _slot(jax_sim, name: str) -> int:
    return next(d for _, d, n in jax_sim.map if n == name)


def _tilt_yaw_errors(q_true: np.ndarray, q_est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def gravity_body(q):
        u, w = -q[:, :3], q[:, 3:]
        t = 2.0 * np.cross(u, [0.0, 0.0, 1.0])
        return np.array([0.0, 0.0, 1.0]) + w * t + np.cross(u, t)

    def yaw(q):
        x, y, z, w = q.T
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    g_true, g_est = gravity_body(q_true), gravity_body(q_est)
    cos = np.sum(g_true * g_est, 1) / np.linalg.norm(g_true, axis=1) / np.linalg.norm(g_est, axis=1)
    tilt = np.arccos(np.clip(cos, -1.0, 1.0))
    yaw_err = (yaw(q_est) - yaw(q_true) + np.pi) % (2.0 * np.pi) - np.pi
    return tilt, yaw_err


def _in_sim_tracking() -> dict:
    # stock 30 s attitude flight plan; attitude_control flies on truth world_pos, so the
    # estimate is tracked inside the sim but does not feed back
    world, drone = sim.world()
    world.insert(drone, mekf6.MEKF6())
    system = mekf6.mekf6_update | sim.system()
    jax_sim = world.to_jax(system, simulation_rate=TALON_QUAD_CONFIG.simulation_rate)
    step = _pure_step(jax_sim)
    i_pos, i_est = _slot(jax_sim, "world_pos"), _slot(jax_sim, "attitude_estimate")

    def body(state, _):
        state = step(state)
        return state, (state[i_pos][:4], state[i_est])

    ticks = TALON_QUAD_CONFIG.total_sim_ticks
    run = jax.jit(lambda s: jax.lax.scan(body, s, None, length=ticks))
    _, (q_true, q_est) = run(tuple(jax_sim.state))
    q_true, q_est = np.asarray(q_true), np.asarray(q_est)
    tilt, yaw_err = _tilt_yaw_errors(q_true, q_est)
    steady = tilt[np.arange(1, ticks + 1) * TALON_QUAD_CONFIG.dt >= 22.0]  # level after 21 s
    return {
        "finite": bool(np.isfinite(q_est).all()),
        "tilt_rms": float(np.sqrt(np.mean(tilt**2))),
        "tilt_max": float(tilt.max()),
        "yaw_rms": float(np.sqrt(np.mean(yaw_err**2))),
        "steady_tilt_rms": float(np.sqrt(np.mean(steady**2))),
        "steady_tilt_max": float(steady.max()),
    }


def test_estimator_in_sim_tracking_jax(talon):
    m = _in_sim_tracking()
    print(f"\nmekf6 {m}")
    assert m["finite"]
    assert m["tilt_rms"] < 0.06 and m["tilt_max"] < 0.18 and m["yaw_rms"] < 0.01
    assert m["steady_tilt_rms"] < 0.03 and m["steady_tilt_max"] < 0.05


ProbeBias = ty.Annotated[
    jax.Array,
    el.Component("mekf6_bias_probe", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
ProbeCov = ty.Annotated[
    jax.Array,
    el.Component("mekf6_cov_probe", el.ComponentType(el.PrimitiveType.F64, (6, 6))),
]


@dataclass
class Probe(el.Archetype):
    bias: ProbeBias = field(default_factory=lambda: jnp.zeros(3))
    cov: ProbeCov = field(default_factory=lambda: jnp.zeros((6, 6)))


@el.map
def probe(b: mekf6.Mekf6Bias, p: mekf6.Mekf6Cov) -> tuple[ProbeBias, ProbeCov]:
    # mekf6_bias / mekf6_cov are transient (not recorded); mirror them for history()
    return b, p


def _history(backend: str, ticks: int, cov0=None) -> dict:
    world, drone = sim.world()
    est = mekf6.MEKF6() if cov0 is None else mekf6.MEKF6(mekf6_cov=jnp.asarray(cov0))
    world.insert(drone, [est, Probe()])
    exe = world.build(
        mekf6.mekf6_update | probe | sim.system(),
        simulation_rate=TALON_QUAD_CONFIG.simulation_rate,
        backend=backend,
    )
    exe.run(ticks, show_progress=False)
    names = ["drone.attitude_estimate", "drone.mekf6_bias_probe", "drone.mekf6_cov_probe"]
    df = exe.history(names)
    return {n: np.stack(df[n].to_numpy()) for n in names}


def test_cranelift_matches_jax_cpu(talon):
    a, b = _history("cranelift", 900), _history("jax-cpu", 900)
    for name in a:
        assert a[name].shape == b[name].shape and a[name].shape[0] == 901
        assert np.isfinite(a[name]).all()
        np.testing.assert_allclose(a[name], b[name], atol=1e-9, rtol=0, err_msg=name)


def _f64(name: str, shape: tuple) -> ty.Any:
    return ty.Annotated[
        jax.Array, el.Component(name, el.ComponentType(el.PrimitiveType.F64, shape))
    ]


TQ, TB, TP = _f64("t_q", (4,)), _f64("t_b", (3,)), _f64("t_p", (6, 6))
TG, TA, TM = _f64("t_gyro", (3,)), _f64("t_accel", (3,)), _f64("t_mag", (3,))


@dataclass
class EstimateBench(el.Archetype):
    q: TQ = field(default_factory=lambda: jnp.asarray(axis_angle([0.3, -0.5, 0.8], 0.7)))
    b: TB = field(default_factory=lambda: jnp.array([0.004, -0.007, 0.002]))
    p: TP = field(default_factory=lambda: jnp.asarray(mekf6.P0) + 1e-4)
    gyro: TG = field(default_factory=lambda: jnp.array([0.4, -0.3, 0.2]))
    accel: TA = field(default_factory=lambda: jnp.array([0.1, -0.2, 1.0]))
    mag: TM = field(default_factory=lambda: jnp.array([0.05, 0.9, -0.1]))


def _estimate_bench(sigmas: mekf6.Sigmas) -> el.System:
    @el.map
    def estimate_bench(q: TQ, b: TB, p: TP, g: TG, a: TA, m: TM) -> tuple[TQ, TB, TP]:
        return mekf6.estimate(q, b, p, g, a, m, sigmas, 1.0 / 300.0)[:3]

    return estimate_bench


@pytest.mark.parametrize("gain", [0.0, 2.0])
def test_estimate_cranelift_matches_jax_cpu(gain):
    s = mekf6.Sigmas(gyro=1e-3, gyro_bias=1e-3, accel=0.5, mag=0.02, accel_gain=gain)
    out = {}
    for backend in ("cranelift", "jax-cpu"):
        world = el.World()
        world.spawn([EstimateBench()], name="e")
        exe = world.build(_estimate_bench(s), simulation_rate=300.0, backend=backend)
        exe.run(300, show_progress=False)
        df = exe.history(["e.t_q", "e.t_b", "e.t_p"])
        out[backend] = [np.stack(df[n].to_numpy()) for n in ("e.t_q", "e.t_b", "e.t_p")]
    for a, b in zip(out["cranelift"], out["jax-cpu"]):
        assert a.shape == b.shape and a.shape[0] == 301
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)


def test_cranelift_reset_matches_jax_cpu(talon):
    nan_cov = np.full((6, 6), np.nan)
    a, b = _history("cranelift", 30, nan_cov), _history("jax-cpu", 30, nan_cov)
    for name in a:
        np.testing.assert_allclose(a[name][1:], b[name][1:], atol=1e-9, rtol=0, err_msg=name)
    np.testing.assert_allclose(a["drone.attitude_estimate"][1], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    cov = a["drone.mekf6_cov_probe"]
    np.testing.assert_allclose(cov[1].reshape(6, 6), mekf6.P0, atol=1e-15)
    assert np.isfinite(cov[1:]).all()
