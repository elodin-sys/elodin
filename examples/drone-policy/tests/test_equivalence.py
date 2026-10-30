import typing as ty
from dataclasses import dataclass, field

import elodin as el
import filter
import jax
import jax.numpy as jnp
import numpy as np
import params
import plant_dr
import policy_io
import policy_world
import pytest
import sim
import vio
from dr import dr_sample, throw_sample
from jaxstep import JaxStep

TICKS = 300
COMPARED = ("world_pos", "world_vel", "motor_rpm")


def trajectory(js: JaxStep, names: ty.Sequence[str], ticks: int = TICKS) -> dict:
    slots = [js.slot(n) for n in names]

    def body(s, _):
        s = js.step(s)
        return s, tuple(s[i] for i in slots)

    _, hist = jax.jit(lambda s: jax.lax.scan(body, s, None, length=ticks))(js.init)
    return {n: np.asarray(h) for n, h in zip(names, hist)}


E7_TICKS = 3000


@pytest.mark.parametrize("estimator", ["mekf18", "truth"])
def test_nominal_equivalence(estimator):
    """E7: nominal DR params, zero delays, unit rate gains, the stock rate PID,
    rate_flight_plan setpoints and hover + 0.05 collective through the action FIFO reproduce
    stock system(only_rate_control=True) on the JAX path, from identical initial conditions."""
    stock_world, _ = sim.world()
    stock_js = JaxStep(stock_world, sim.system(only_rate_control=True))
    w, _ = policy_world.world(1, estimator)
    system = policy_world.system(estimator, setpoints="rate_flight_plan", rate_pid="stock")
    dr_js = JaxStep(w, system)
    shared = [n for n in stock_js.names if dr_js.has(n)]
    assert set(COMPARED) | {"rate_pid_state", "inertia", "gyro", "sensor_tick"} <= set(shared)
    for name in shared:
        np.testing.assert_array_equal(
            stock_js.get(stock_js.init, name), dr_js.get(dr_js.init, name)
        )
    names = (*COMPARED, "ang_vel_setpoint")
    stock = trajectory(stock_js, names, E7_TICKS)
    dr = trajectory(dr_js, names, E7_TICKS)
    for name in COMPARED:
        err = np.max(np.abs(stock[name] - dr[name]))
        print(f"{estimator} {name}: max abs diff over {E7_TICKS} ticks = {err:.3e}")
        assert err <= 1e-9, name
    # the comparison is not vacuous: the stock rate plan sends nonzero setpoints in the window
    sp = stock["ang_vel_setpoint"]
    assert np.any(sp != 0.0) and len(np.unique(sp, axis=0)) >= 5
    np.testing.assert_allclose(dr["ang_vel_setpoint"], sp, rtol=0, atol=1e-12)
    assert np.ptp(stock["world_pos"][:, 6]) > 0.5
    assert np.ptp(stock["motor_rpm"]) > 1e4


def test_rate_pid_dmeas():
    """policy_io.rate_pid_dmeas equals the stock rate PID while the setpoint holds and has no
    derivative kick when it steps (the stock D term jumps by alpha_d * d_setpoint / dt)."""
    names = ("world_pos", "motor_rpm", "rate_pid_state", "ang_vel_setpoint")
    hist = {}
    for pid in ("stock", "dmeas"):
        w, _ = policy_world.world(1, "truth")
        system = policy_world.system("truth", setpoints="rate_flight_plan", rate_pid=pid)
        hist[pid] = trajectory(JaxStep(w, system), names, 600)
    sp = hist["stock"]["ang_vel_setpoint"]
    k = int(np.argmax(np.any(sp != sp[0], axis=1)))  # first setpoint step
    assert 0 < k < 600
    for name in names:
        np.testing.assert_array_equal(hist["stock"][name][:k], hist["dmeas"][name][:k])

    def alpha(*hz):
        return np.asarray(filter.LPF(np.array(hz), 300.0).alpha)

    a_e = alpha(params.ATC_RAT_RLL_FLTE, params.ATC_RAT_PIT_FLTE, params.ATC_RAT_YAW_FLTE)
    a_d = alpha(params.ATC_RAT_RLL_FLTD, params.ATC_RAT_PIT_FLTD, params.ATC_RAT_YAW_FLTD)
    kick = a_d * a_e * (sp[k] - sp[k - 1]) * 300.0
    assert np.abs(kick).max() > 10.0
    d = {p: hist[p]["rate_pid_state"][k, 2] for p in hist}
    np.testing.assert_allclose(d["stock"] - d["dmeas"], kick, rtol=1e-9, atol=1e-9)


@el.system
def scripted_policy(
    tick: el.Query[el.SimulationTick], a: el.Query[policy_io.PolicyAction]
) -> el.Query[policy_io.PolicyAction]:
    t = tick[0].astype(jnp.float64) / 300.0
    act = 0.4 * jnp.sin(
        2 * jnp.pi * jnp.array([1.3, 0.7, 0.9, 0.5]) * t + jnp.array([0.0, 1.0, 2.0, 3.0])
    )
    return a.map(policy_io.PolicyAction, lambda _: act)


# transient components are not recorded; mirror them into recorded test components
CriticRec = ty.Annotated[
    jax.Array,
    el.Component("critic_rec", el.ComponentType(el.PrimitiveType.F64, (policy_io.CRITIC_DIM,))),
]
FifoRec = ty.Annotated[
    jax.Array,
    el.Component("fifo_rec", el.ComponentType(el.PrimitiveType.F64, (policy_io.FIFO_LEN * 4,))),
]
VioBufRec = ty.Annotated[
    jax.Array, el.Component("vio_buf_rec", el.ComponentType(el.PrimitiveType.F64, (8 * 7,)))
]


@dataclass
class Recorded(el.Archetype):
    critic_rec: CriticRec = field(default_factory=lambda: jnp.zeros(policy_io.CRITIC_DIM))
    fifo_rec: FifoRec = field(default_factory=lambda: jnp.zeros(policy_io.FIFO_LEN * 4))
    vio_buf_rec: VioBufRec = field(default_factory=lambda: jnp.zeros(8 * 7))


@el.map
def record(
    critic: policy_io.CriticObs, fifo: policy_io.ActFifo, buf: vio.VioBuf
) -> tuple[CriticRec, FifoRec, VioBufRec]:
    return critic, fifo.reshape(-1), buf.reshape(-1)


PARITY = (
    "world_pos",
    "world_vel",
    "motor_rpm",
    "ang_vel_setpoint",
    "collective_cmd",
    "motor_input",
    "vio_meas",
    "policy_obs",
    "critic_rec",
    "fifo_rec",
    "vio_buf_rec",
)


def random_inits(n: int, seed: int = 3) -> list[dict]:
    keys = jax.random.split(jax.random.key(seed), 3 * n).reshape(n, 3)
    inits = []
    for i in range(n):
        dr = dr_sample(keys[i, 0], 1.0)
        dr = dr._replace(act_delay=jnp.int32(1 + i % 2), vio_delay=jnp.int32(3 + i % 4))
        throw = throw_sample(keys[i, 1], 0.5)
        if i % 4 == 3:  # outside the p_int gate
            throw = throw._replace(pos_offset=throw.pos_offset.at[0].set(1.2))
        init = policy_world.drone_init(dr, throw, keys[i, 2], "truth")
        inits.append({k: np.asarray(v) for k, v in init.items()})
    return inits


def backend_histories(n, inits, system, ticks, names=PARITY) -> dict:
    hist = {}
    entities = ["drone"] if n == 1 else [f"drone_{i}" for i in range(n)]
    cols = [f"{e}.{c}" for e in entities for c in names]
    for backend in ("jax-cpu", "cranelift"):
        w, _ = policy_world.world(n, "truth", inits, extra=[Recorded()])
        ex = w.build(system | record, simulation_rate=300.0, backend=backend)
        ex.run(ticks, show_progress=False)
        df = ex.history(cols)
        hist[backend] = {c: np.stack(df[c].to_numpy()) for c in cols}
    return hist


def max_diff(hist) -> dict:
    out = {}
    for c, a in hist["jax-cpu"].items():
        b = hist["cranelift"][c]
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), c
        out[c] = float(np.max(np.abs(a - b)))
    return out


@pytest.mark.parametrize("seed,rate_pid", [(3, "stock"), (11, "stock"), (3, "dmeas")])
def test_backend_parity_pipeline(seed, rate_pid):
    """The full policy pipeline (DR effectors, action FIFO with delay, collective rate control,
    VIO ring buffer, actor and critic observations) gives the same trajectory on Cranelift and
    jax-cpu for a thrown drone with full-scale DR and a time-varying action."""
    system = policy_world.system("truth", policy=scripted_policy, rate_pid=rate_pid)
    hist = backend_histories(1, random_inits(1, seed), system, TICKS)
    diff = max_diff(hist)
    print(
        f"seed {seed} {rate_pid}: worst cranelift vs jax-cpu over {TICKS} ticks", max(diff.values())
    )
    for c, err in diff.items():
        assert err <= 1e-6, f"{c}: {err:.3e}"
    sp = hist["jax-cpu"]["drone.ang_vel_setpoint"]
    assert np.ptp(sp) > 0.5  # the action path is exercised


NEW_SYSTEMS = {
    "action_apply": policy_io.action_apply,
    "rate_pid_dmeas": policy_io.rate_pid_dmeas,
    "rate_control_collective": policy_io.rate_control_collective,
    "dr_effectors": plant_dr.effectors,
    "six_dof_dr": el.six_dof(
        1.0 / 900.0, plant_dr.effectors, integrator=el.Integrator.SemiImplicit
    ),
    "vio_update": vio.vio_update,
    "actor_obs": policy_io.actor_obs,
    "critic_obs": policy_io.critic_obs,
    "truth_estimate": policy_io.truth_estimate,
}


def test_backend_parity_multi_entity_new_systems():
    """The new systems chained in one 8-drone world (per-entity params and states) match
    across backends; the per-system tests below isolate a failure."""
    chain = scripted_policy
    for s in NEW_SYSTEMS.values():
        chain = chain | s
    hist = backend_histories(8, random_inits(8), chain, 3)
    for c, err in max_diff(hist).items():
        assert err <= 1e-9, f"{c}: {err:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("name", list(NEW_SYSTEMS))
def test_backend_parity_multi_entity_systems(name):
    """Each new system alone in an 8-drone world (per-entity params and states) matches."""
    hist = backend_histories(8, random_inits(8), scripted_policy | NEW_SYSTEMS[name], 3)
    for c, err in max_diff(hist).items():
        assert err <= 1e-9, f"{name} {c}: {err:.3e}"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Cranelift multi-entity miscompile in the composed stock chain", strict=True
)
def test_backend_parity_multi_entity_pipeline():
    """Known upstream issue: in an 8-drone world, action_apply | rate_pid_state |
    rate_control_collective | motors.output | motor_response_dr gives wrong motor_rpm on
    Cranelift although every piece and shorter chains match (the stock drone pipeline NaNs at
    tick 1 with 8 entities and fails to compile with 2-4)."""
    system = policy_world.system("truth", policy=scripted_policy, rate_pid="stock")
    hist = backend_histories(8, random_inits(8), system, 30)
    for c, err in max_diff(hist).items():
        assert err <= 1e-6, f"{c}: {err:.3e}"
