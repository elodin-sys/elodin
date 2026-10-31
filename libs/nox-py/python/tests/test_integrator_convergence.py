"""Convergence order of the `six_dof` integrators against closed-form trajectories.

Run directly to print error tables for both integrators.
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable

import elodin as el
import numpy as np
import pytest

TIME_STEPS = (0.04, 0.02, 0.01, 0.005)
SPRING_K = 2.25
SPIN = np.array([1.0, 2.0, 2.0])
TORQUE_AXIS = np.array([1.0, 2.0, 2.0]) / 3.0
TOP_INERTIA = np.array([1.0, 2.0, 2.0])
TOP_OMEGA_BODY = np.array([3.0, 1.0, 0.0])


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


@el.map
def spring(pos: el.WorldPos) -> el.Force:
    return el.SpatialForce(linear=-SPRING_K * pos.linear())


@el.map
def unit_torque(_: el.Force) -> el.Force:
    return el.SpatialForce(torque=TORQUE_AXIS)


def _spring_ref(t):
    w = np.sqrt(SPRING_K)
    zero = np.zeros_like(t)
    return {
        "pos": np.hstack([np.cos(w * t), zero, zero]),
        "vel": np.hstack([-w * np.sin(w * t), zero, zero]),
    }


def _free_spin_ref(t):
    return {"att": _qexp(t * SPIN), "omega": np.broadcast_to(SPIN, (len(t), 3))}


def _torque_ref(t):
    return {"att": _qexp(0.5 * t**2 * TORQUE_AXIS), "omega": t * TORQUE_AXIS}


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
    body: Callable[[], el.Body]
    effector: Any
    duration: float
    reference: Callable[[np.ndarray], dict[str, np.ndarray]]


CASES = {
    "spring": Case(
        lambda: el.Body(world_pos=el.SpatialTransform(linear=np.array([1.0, 0.0, 0.0]))),
        spring,
        4.0,
        _spring_ref,
    ),
    "free_spin": Case(
        lambda: el.Body(world_vel=el.SpatialMotion(angular=SPIN)), None, 2.0, _free_spin_ref
    ),
    "constant_torque": Case(lambda: el.Body(), unit_torque, 2.0, _torque_ref),
    "symmetric_top": Case(
        lambda: el.Body(
            world_vel=el.SpatialMotion(angular=TOP_OMEGA_BODY),
            inertia=el.SpatialInertia(1.0, inertia=TOP_INERTIA),
        ),
        None,
        2.0,
        _top_ref,
    ),
}


def _simulate(case: Case, dt: float, steps: int, integrator: str):
    w = el.World()
    w.spawn(case.body(), "body")
    sys = el.six_dof(dt, sys=case.effector, integrator=getattr(el.Integrator, integrator))
    exec = w.build(sys, simulation_rate=1.0 / dt)
    exec.run(steps, show_progress=False)
    df = exec.history(["body.world_pos", "body.world_vel"])
    pos, vel = df["body.world_pos"].to_numpy(), df["body.world_vel"].to_numpy()
    assert len(pos) == steps + 1
    return pos, vel


@functools.cache
def max_errors(name: str, dt: float, integrator: str = "Rk4") -> dict[str, float]:
    case = CASES[name]
    pos, vel = _simulate(case, dt, round(case.duration / dt), integrator)
    sim = {"att": pos[:, :4], "pos": pos[:, 4:], "omega": vel[:, :3], "vel": vel[:, 3:]}
    ref = case.reference(np.arange(len(pos))[:, None] * dt)
    return {
        k: float(
            np.max(_angle_between(sim[k], r) if k == "att" else np.linalg.norm(sim[k] - r, axis=-1))
        )
        for k, r in ref.items()
    }


def observed_orders(name: str, metric: str, integrator: str = "Rk4") -> list[float]:
    errs = [max_errors(name, dt, integrator)[metric] for dt in TIME_STEPS]
    return [float(np.log2(a / b)) for a, b in zip(errs, errs[1:])]


def spring_one_step(h: float) -> tuple[float, float]:
    pos, vel = _simulate(CASES["spring"], h, 1, "Rk4")
    return float(pos[1, 4]), float(vel[1, 3])


def test_rk4_spring_single_step_matches_classical_rk4():
    h = 0.02
    z = SPRING_K * h**2
    x, v = spring_one_step(h)
    # Classical RK4 on x'' = -k x reproduces the Taylor series of cos/sin through h^4.
    assert x == pytest.approx(1 - z / 2 + z**2 / 24, abs=1e-13)
    assert v == pytest.approx(-SPRING_K * h * (1 - z / 6), abs=1e-13)


@pytest.mark.parametrize(
    ("name", "metric"),
    [
        ("spring", "pos"),
        ("spring", "vel"),
        ("free_spin", "att"),
        ("constant_torque", "att"),
        ("symmetric_top", "att"),
        ("symmetric_top", "omega"),
    ],
)
def test_rk4_is_fourth_order(name, metric):
    assert min(observed_orders(name, metric)) > 3.5


if __name__ == "__main__":
    x, v = spring_one_step(0.02)
    print(f"spring, one RK4 step of h=0.02: x={x:.11f} v={v:.8f}")
    for name, case in CASES.items():
        for integrator in ("Rk4", "SemiImplicit"):
            errs = [max_errors(name, dt, integrator) for dt in TIME_STEPS]
            print(f"\n{name}, {integrator}: max error over {case.duration:g} s")
            print("     dt" + "".join(f"{m:>12} {'order':>6}" for m in errs[0]))
            for i, (dt, row) in enumerate(zip(TIME_STEPS, errs)):
                cells = "".join(
                    f"{row[m]:12.3e} "
                    + (f"{np.log2(errs[i - 1][m] / row[m]):6.2f}" if i and row[m] else "     -")
                    for m in row
                )
                print(f"{dt:7.3f}{cells}")
