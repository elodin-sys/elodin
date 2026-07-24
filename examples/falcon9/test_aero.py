"""Phase 3 checks: RCS/fin control-effectiveness vs analytic, canonical aero
directions, the recorded-profile q-bar reconstruction (WHITEPAPER 8.2), plume
dominance, and the flip-time budget."""

import json
import math
from pathlib import Path

import aero
import atmosphere
import jax.numpy as jnp
import numpy as np
import propulsion
import rcs
from constants import (
    LOX_LOAD_KG,
    RP1_LOAD_KG,
    S_REF_M2,
)

DATA = Path(__file__).parent / "data"


def test_rcs_axis_purity_and_authority():
    """Each torque axis allocates to its dedicated group; off-axis torque
    cancels; realized torque matches the request until saturation."""
    cg = 22.0
    b = np.asarray(rcs.effectiveness_matrix(cg))
    assert b.shape == (6, rcs.N_RCS)
    # Per-axis authority from the effectiveness matrix itself (roll is much
    # weaker than pitch/yaw: short radial arm vs the long station arm).
    authority = [abs(b[3 + axis, :]).sum() / 2.0 for axis in range(3)]
    for axis in range(3):
        for sign in (+1.0, -1.0):
            cmd = np.zeros(3)
            cmd[axis] = sign * 0.5 * authority[axis]
            levels = rcs.allocate_torque(jnp.asarray(cmd), cg)
            force, torque = rcs.rcs_wrench(levels, cg)
            torque = np.asarray(torque)
            assert abs(torque[axis] - cmd[axis]) < 1e-6 * abs(cmd[axis]) + 1e-9, (
                f"axis {axis} sign {sign}: {torque} vs {cmd}"
            )
            off = np.delete(torque, axis)
            assert np.all(np.abs(off) < 1e-9), f"off-axis torque {torque}"
    # Saturation: an absurd command pegs levels at 1 and torque at authority.
    levels = np.asarray(rcs.allocate_torque(jnp.array([0.0, 1e9, 0.0]), cg))
    assert levels.max() <= 1.0 + 1e-12
    assert np.count_nonzero(levels > 0.99) == 2


def test_rcs_roll_force_free():
    """Roll pairs produce zero net force (opposite pods, opposite fire)."""
    cg = 22.0
    for tx in (+4.0e5, -4.0e5):
        levels = rcs.allocate_torque(jnp.array([tx, 0.0, 0.0]), cg)
        force, _ = rcs.rcs_wrench(levels, cg)
        assert np.all(np.abs(np.asarray(force)) < 1e-9)


def test_fin_mixing_axis_purity():
    """Pitch/yaw/roll fin commands produce torque about the right axis."""
    cg = 20.0
    mach, qbar = 2.0, 30_000.0
    for axis, cmd in ((1, [0.1, 0.0, 0.0]), (2, [0.0, 0.1, 0.0]), (0, [0.0, 0.0, 0.1])):
        deltas = aero.fin_mix(jnp.asarray(cmd))
        force, torque = aero.fin_wrench(deltas, mach, qbar, cg)
        torque = np.asarray(torque)
        dominant = np.argmax(np.abs(torque))
        assert dominant == axis, f"cmd {cmd}: torque {torque}"
        off = np.delete(torque, axis)
        assert np.all(np.abs(off) < 1e-9 * max(1.0, abs(torque[axis])))
    # Roll command: tangential forces cancel exactly.
    force, _ = aero.fin_wrench(aero.fin_mix(jnp.array([0.0, 0.0, 0.2])), mach, qbar, cg)
    assert np.all(np.abs(np.asarray(force)) < 1e-9)


def test_aero_canonical_directions():
    cg = 22.5
    qbar = 20_000.0
    # Nose-first axial flow: pure axial drag, no torque.
    f, t = aero.body_aero_wrench(jnp.array([500.0, 0.0, 0.0]), 1.5, qbar, cg)
    f, t = np.asarray(f), np.asarray(t)
    assert f[0] < 0.0 and abs(f[1]) < 1e-9 and abs(f[2]) < 1e-9
    assert np.all(np.abs(t) < 1e-9)
    ca_ascent = float(jnp.interp(1.5, aero.MACH_PTS, aero.CA_ASCENT))
    # The tanh config blend leaves a ~1e-9 residual of the descent table.
    assert abs(f[0] + qbar * S_REF_M2 * ca_ascent) < 1e-2
    # Engines-first flow: drag opposes motion (+X force), descent table.
    f, t = aero.body_aero_wrench(jnp.array([-500.0, 0.0, 0.0]), 1.5, qbar, cg)
    f = np.asarray(f)
    ca_descent = float(jnp.interp(1.5, aero.MACH_PTS, aero.CA_DESCENT))
    assert f[0] > 0.0
    assert abs(f[0] - qbar * S_REF_M2 * ca_descent) < 1e-2  # tanh blend residual
    assert ca_descent > 2.0 * ca_ascent  # blunt-first is much draggier
    # Pure cross-flow: force opposes flow; static moment is nonzero.
    f, t = aero.body_aero_wrench(jnp.array([0.0, 0.0, 300.0]), 0.8, qbar, cg)
    f, t = np.asarray(f), np.asarray(t)
    assert f[2] < 0.0 and abs(f[0]) < 1e-6
    assert abs(t[1]) > 0.0 and abs(t[0]) < 1e-9 and abs(t[2]) < 1e-9


def test_qbar_reconstruction_matches_whitepaper():
    """WHITEPAPER 8.2 from the vendored CRS-12 telemetry: ascent Max-Q ~22 kPa,
    descent peak ~60 kPa near t=412 s, nearly 3x the ascent load."""
    d = json.loads((DATA / "crs12" / "stage1_raw.json").read_text())
    t = np.asarray(d["time"])
    v = np.asarray(d["velocity"])
    alt_m = np.asarray(d["altitude"]) * 1000.0
    rho = np.asarray(atmosphere.density(jnp.asarray(np.maximum(alt_m, 0.0))))
    qbar = 0.5 * rho * v**2 / 1000.0  # kPa
    ascent = (t >= 40.0) & (t <= 90.0)
    descent = (t >= 390.0) & (t <= 433.0)
    q_ascent = qbar[ascent].max()
    q_descent = qbar[descent].max()
    t_descent_peak = t[descent][qbar[descent].argmax()]
    assert 18.0 < q_ascent < 26.0, f"ascent Max-Q {q_ascent:.1f} kPa"
    assert 50.0 < q_descent < 70.0, f"descent peak {q_descent:.1f} kPa"
    assert q_descent > 2.0 * q_ascent
    assert 400.0 < t_descent_peak < 425.0


def test_pitch_damping_opposes_rate():
    """Cmq < 0: a positive pitch rate produces a restoring (negative) My."""
    cg = 22.5
    qbar = 40_000.0
    v = jnp.array([-400.0, 0.0, 0.0])  # engines-first
    omega = jnp.array([0.0, 0.5, 0.0])  # +pitch rate
    _f, t0 = aero.body_aero_wrench(v, 1.5, qbar, cg, omega_body=jnp.zeros(3))
    _f, t1 = aero.body_aero_wrench(v, 1.5, qbar, cg, omega_body=omega)
    t0, t1 = np.asarray(t0), np.asarray(t1)
    # Damping contribution is negative along +ω_y.
    assert t1[1] < t0[1] - 1e3, f"expected damping My, got {t0[1]} -> {t1[1]}"


def test_plume_dominance():
    assert float(aero.plume_dominance(jnp.array(0.0), jnp.array(30_000.0))) == 0.0
    # Entry-burn class: 3 engines ~2.3 MN against 30 kPa.
    kappa = float(aero.plume_dominance(jnp.array(2.3e6), jnp.array(30_000.0)))
    assert 0.85 < kappa < 0.95
    # Landing burn near the pad: single engine, thick air, still dominant.
    kappa_land = float(aero.plume_dominance(jnp.array(5.0e5), jnp.array(40_000.0)))
    assert kappa_land > 0.5


def test_flip_time_budget():
    """Bang-bang 180 deg flip with the RCS pitch authority lands in the
    recorded MECO-to-boostback window (WHITEPAPER 10.3: ~15-20 s)."""
    # Post-MECO propellant: ~63 t remaining of the ~398 t load (budget audit).
    frac = 63_000.0 / (LOX_LOAD_KG + RP1_LOAD_KG)
    mass, cg, inertia = propulsion.stack_mass_props(LOX_LOAD_KG * frac, RP1_LOAD_KG * frac)
    b = np.asarray(rcs.effectiveness_matrix(float(cg)))
    pitch_auth = abs(b[4, 1] + b[4, 3])  # torque-y rows of the +pitch group
    i_trans = float(inertia[1])
    alpha = pitch_auth / i_trans
    t_flip = 2.0 * math.sqrt(math.pi / alpha)
    assert 8.0 < t_flip < 22.0, f"flip time {t_flip:.1f} s with authority {pitch_auth:.0f} N m"
