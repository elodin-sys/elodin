"""Convergence order of the `six_dof` integrators against closed-form trajectories.

Every `el.Integrator` needs a METHODS entry: its order, a time-step ladder inside its
asymptotic range, and the spring state after its first step. Run this file directly to
print the error tables.
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.integrator

DURATION = 4.0
# Ratios whose finer error is below this are dominated by roundoff.
ERROR_FLOOR = 1e-11
SPRING_K = 2.25
KEPLER_E = 0.5
SPIN = np.array([1.0, 2.0, 2.0])
TORQUE_AXIS = np.array([1.0, 2.0, 2.0]) / 3.0
TOP_INERTIA = np.array([1.0, 2.0, 2.0])
TOP_OMEGA_BODY = np.array([3.0, 1.0, 0.0])


@dataclass(frozen=True)
class Method:
    order: int
    time_steps: tuple[float, ...]
    # Spring state after one step of h from (x, v) = (1, 0), as (x, v / (-k h)) in z = k h^2.
    one_step: Callable[[float], tuple[float, float]]


METHODS = {
    "SemiImplicit": Method(1, (0.02, 0.01, 0.005), lambda z: (1 - z, 1)),
    "Rk2": Method(2, (0.02, 0.01, 0.005), lambda z: (1 - z / 2, 1)),
    "Rk3": Method(3, (0.04, 0.02, 0.01), lambda z: (1 - z / 2, 1 - z / 6)),
    "Rk4": Method(4, (0.04, 0.02, 0.01), lambda z: (1 - z / 2 + z**2 / 24, 1 - z / 6)),
}

SpringConstant = el.Annotated[
    jax.Array, el.Component("spring_constant", el.ComponentType(el.PrimitiveType.F64, (1,)))
]
GravitationalParameter = el.Annotated[
    jax.Array, el.Component("gravitational_parameter", el.ComponentType(el.PrimitiveType.F64, (1,)))
]
AppliedTorque = el.Annotated[
    jax.Array, el.Component("applied_torque", el.ComponentType(el.PrimitiveType.F64, (3,)))
]


@el.map
def spring(pos: el.WorldPos, k: SpringConstant) -> el.Force:
    return el.SpatialForce(linear=-k * pos.linear())


@el.map
def gravity(pos: el.WorldPos, mu: GravitationalParameter) -> el.Force:
    r = pos.linear()
    return el.SpatialForce(linear=-mu * r / jnp.linalg.norm(r) ** 3)


@el.map
def torque(tau: AppliedTorque) -> el.Force:
    return el.SpatialForce(torque=tau)


def _qmul(a, b):
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return np.concatenate(
        [aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, -1, keepdims=True)], -1
    )


def _qconj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _qexp(rotvec):
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(rotvec, angle, out=np.zeros_like(rotvec), where=angle > 0)
    return np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], -1)


def _rotate(q, v):
    return _qmul(_qmul(q, np.concatenate([v, np.zeros_like(v[..., :1])], -1)), _qconj(q))[..., :3]


def _angle_between(qa, qb):
    d = _qmul(_qconj(qa), qb)
    return 2 * np.arctan2(np.linalg.norm(d[..., :3], axis=-1), np.abs(d[..., 3]))


def _spring_ref(t):
    w = np.sqrt(SPRING_K)
    zero = np.zeros_like(t)
    return {
        "pos": np.hstack([np.cos(w * t), zero, zero]),
        "vel": np.hstack([-w * np.sin(w * t), zero, zero]),
    }


def _kepler_ref(t):
    # mu = a = 1, starting at periapsis: solve Kepler's equation E - e sin E = t.
    e = KEPLER_E
    big_e = t.copy()
    for _ in range(30):
        big_e -= (big_e - e * np.sin(big_e) - t) / (1 - e * np.cos(big_e))
    c, s = np.cos(big_e), np.sin(big_e)
    b = np.sqrt(1 - e**2)
    rate = 1 / (1 - e * c)
    zero = np.zeros_like(t)
    return {
        "pos": np.hstack([c - e, b * s, zero]),
        "vel": np.hstack([-s * rate, b * c * rate, zero]),
    }


def _free_spin_ref(t):
    return {"att": _qexp(t * SPIN)}


def _torque_ref(t):
    return {"att": _qexp(0.5 * t**2 * TORQUE_AXIS)}


def _top_ref(t):
    # Torque-free axisymmetric body: q(t) = exp(t L/I_t) * exp(lambda t x), omega_body rotates about x.
    i_s, i_t = TOP_INERTIA[0], TOP_INERTIA[1]
    p, q, _ = TOP_OMEGA_BODY
    lam = (i_t - i_s) / i_t * p
    l_world = TOP_INERTIA * TOP_OMEGA_BODY
    att = _qmul(_qexp(t * l_world / i_t), _qexp(lam * t * np.array([1.0, 0.0, 0.0])))
    omega_body = np.hstack([np.full_like(t, p), q * np.cos(lam * t), -q * np.sin(lam * t)])
    return {"att": att, "omega": _rotate(att, omega_body)}


@dataclass(frozen=True)
class Case:
    components: Callable[[], list[Any]]
    reference: Callable[[np.ndarray], dict[str, np.ndarray]]


CASES = {
    "spring": Case(
        lambda: [
            el.Body(world_pos=el.SpatialTransform(linear=np.array([1.0, 0.0, 0.0]))),
            el.C(SpringConstant, np.array([SPRING_K])),
        ],
        _spring_ref,
    ),
    "kepler": Case(
        lambda: [
            el.Body(
                world_pos=el.SpatialTransform(linear=np.array([1 - KEPLER_E, 0.0, 0.0])),
                world_vel=el.SpatialMotion(
                    linear=np.array([0.0, np.sqrt((1 + KEPLER_E) / (1 - KEPLER_E)), 0.0])
                ),
            ),
            el.C(GravitationalParameter, np.array([1.0])),
        ],
        _kepler_ref,
    ),
    "free_spin": Case(lambda: [el.Body(world_vel=el.SpatialMotion(angular=SPIN))], _free_spin_ref),
    "constant_torque": Case(lambda: [el.Body(), el.C(AppliedTorque, TORQUE_AXIS)], _torque_ref),
    "symmetric_top": Case(
        lambda: [
            el.Body(
                world_vel=el.SpatialMotion(angular=TOP_OMEGA_BODY),
                inertia=el.SpatialInertia(1.0, inertia=TOP_INERTIA),
            )
        ],
        _top_ref,
    ),
}
METRICS = [(case, metric) for case, c in CASES.items() for metric in c.reference(np.zeros((1, 1)))]


@functools.cache
def _simulate(method: str, dt: float, steps: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    w = el.World()
    for name, case in CASES.items():
        w.spawn(case.components(), name)
    effectors = spring | gravity | torque
    sys = el.six_dof(dt, sys=effectors, integrator=getattr(el.Integrator, method))
    exec = w.build(sys, simulation_rate=1.0 / dt)
    exec.run(steps, show_progress=False)
    df = exec.history([f"{name}.{c}" for name in CASES for c in ("world_pos", "world_vel")])
    out = {}
    for name in CASES:
        pos, vel = df[f"{name}.world_pos"].to_numpy(), df[f"{name}.world_vel"].to_numpy()
        assert len(pos) == steps + 1
        out[name] = pos, vel
    return out


@functools.cache
def max_errors(method: str, dt: float) -> dict[tuple[str, str], float]:
    steps = round(DURATION / dt)
    errors = {}
    for name, (pos, vel) in _simulate(method, dt, steps).items():
        sim = {"att": pos[:, :4], "pos": pos[:, 4:], "omega": vel[:, :3], "vel": vel[:, 3:]}
        for metric, ref in CASES[name].reference(np.arange(steps + 1)[:, None] * dt).items():
            err = (
                _angle_between(sim[metric], ref)
                if metric == "att"
                else np.linalg.norm(sim[metric] - ref, axis=-1)
            )
            errors[name, metric] = float(np.max(err))
    return errors


def observed_orders(method: str, case: str, metric: str) -> list[float]:
    dts = METHODS[method].time_steps
    errs = [max_errors(method, dt)[case, metric] for dt in dts]
    return [
        float(np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1]))
        for i in range(len(dts) - 1)
        if errs[i + 1] > ERROR_FLOOR
    ]


def test_every_integrator_is_benched():
    variants = {name for name, v in vars(el.Integrator).items() if isinstance(v, el.Integrator)}
    assert variants == set(METHODS)


@pytest.mark.parametrize("method", METHODS)
def test_one_step(method):
    h = 0.02
    pos, vel = _simulate(method, h, 1)["spring"]
    x, v = METHODS[method].one_step(SPRING_K * h**2)
    assert pos[1, 4] == pytest.approx(x, abs=1e-13)
    assert vel[1, 3] == pytest.approx(-SPRING_K * h * v, abs=1e-13)


@pytest.mark.parametrize(("case", "metric"), METRICS)
@pytest.mark.parametrize("method", METHODS)
def test_order(method, case, metric):
    orders = observed_orders(method, case, metric)
    assert orders and min(orders) > METHODS[method].order - 0.5, orders


if __name__ == "__main__":
    for method, m in METHODS.items():
        print(f"\n{method} (order {m.order}): max error over {DURATION:g} s")
        print(f"{'dt':>24}" + "".join(f"{dt:>11g}" for dt in m.time_steps) + "  orders")
        for case, metric in METRICS:
            errs = "".join(f"{max_errors(method, dt)[case, metric]:11.3e}" for dt in m.time_steps)
            orders = " ".join(f"{o:5.2f}" for o in observed_orders(method, case, metric))
            print(f"  {case:16}{metric:6}{errs}  {orders}")
