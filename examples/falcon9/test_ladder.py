"""Phase 1 verification ladder (WHITEPAPER 14.3).

Independent checks the plant must pass before any calibration is meaningful:
US76 anchors, apparent-gravity wiring, the classic Coriolis drop, quaternion
kinematics, a ballistic arc against an independent inertial-frame oracle, and
a one-period orbit hold — the dynamic checks under BOTH integrators so the
SemiImplicit-at-1000-Hz decision is quantified, not asserted.
"""

from __future__ import annotations

import math

import atmosphere
import jax.numpy as jnp
import numpy as np
import pytest
from constants import (
    MU_EARTH_M3S2,
    OMEGA_EARTH_RADPS,
    PAD_LAT_DEG,
    WGS84_A_M,
)
from frames import apparent_gravity, ellipsoid_up, ned_basis
from sim import build_passive, pad_ecef

import elodin as el

INTEGRATORS = [
    ("semi_implicit", el.Integrator.SemiImplicit),
    ("rk4", el.Integrator.Rk4),
]


def _run(world, system, rate_hz: float, steps: int):
    sim = world.to_jax(system, simulation_rate=rate_hz)
    sim.step(steps)
    # Single-entity world: component arrays are flat.
    pos = np.asarray(sim.get_state("world_pos"), dtype=np.float64).reshape(-1)
    vel = np.asarray(sim.get_state("world_vel"), dtype=np.float64).reshape(-1)
    # SpatialTransform: [qx qy qz qw, x y z]; SpatialMotion: [wx wy wz, vx vy vz].
    return pos[:4], pos[4:], vel[:3], vel[3:]


def test_us76_anchors():
    assert abs(float(atmosphere.density(0.0)) - 1.2250) < 1e-3
    p11, t11 = atmosphere.pressure_temperature_at_geopotential(11_000.0)
    assert abs(float(p11) - 22_632.0) < 5.0
    assert abs(float(t11) - 216.65) < 1e-9
    rho11 = float(p11) / (atmosphere.R_AIR * float(t11))
    assert abs(rho11 - 0.3639) < 1e-3
    # Sea-level speed of sound and near-vacuum above the table top.
    assert abs(float(atmosphere.speed_of_sound(0.0)) - 340.29) < 0.1
    assert float(atmosphere.density(100_000.0)) < 1e-5


def test_freefall_matches_apparent_gravity():
    """One step from rest: acceleration = gravitation + centrifugal, along -up."""
    r0 = pad_ecef()
    world, system = build_passive(r0, jnp.zeros(3))
    _, _, _, v1 = _run(world, system, 1000.0, 1)
    accel = v1 * 1000.0  # dv over one 1 ms step
    expected = np.asarray(apparent_gravity(r0))
    np.testing.assert_allclose(accel, expected, rtol=1e-9)
    lat, lon = math.radians(PAD_LAT_DEG), math.radians(-80.60433)
    up = np.asarray(ellipsoid_up(lat, lon))
    cos_angle = -accel @ up / np.linalg.norm(accel)
    # Point-mass field (no J2): plumb line within ~0.1 deg of the geodetic vertical.
    assert math.degrees(math.acos(min(1.0, cos_angle))) < 0.2


@pytest.mark.parametrize("name,integrator", INTEGRATORS)
def test_coriolis_drop(name, integrator):
    """100 m drop deflects ~1.9 cm east: d = (1/3) w g t^3 cos(lat)."""
    lat = math.radians(PAD_LAT_DEG)
    r0 = np.asarray(pad_ecef()) + np.asarray(ellipsoid_up(lat, math.radians(-80.60433))) * 100.0
    world, system = build_passive(jnp.asarray(r0), jnp.zeros(3), integrator=integrator)
    g = float(np.linalg.norm(apparent_gravity(jnp.asarray(r0))))
    t_fall = math.sqrt(2.0 * 100.0 / g)
    steps = int(round(t_fall * 1000.0))
    _, r1, _, _ = _run(world, system, 1000.0, steps)
    ned = np.asarray(ned_basis(lat, math.radians(-80.60433)))
    delta_ned = ned @ (r1 - r0)
    east = delta_ned[1]
    expected = OMEGA_EARTH_RADPS * g * t_fall**3 * math.cos(lat) / 3.0
    assert abs(east - expected) < 0.03 * expected + 2e-4
    assert abs(delta_ned[2] - 100.0) < 0.15  # fell ~100 m (down is +)


@pytest.mark.parametrize("name,integrator", INTEGRATORS)
def test_quaternion_single_axis(name, integrator):
    """1 deg/s about +Z for 90 s = 90 deg yaw, correct sign, unit norm."""
    omega = math.radians(1.0)
    world, system = build_passive(
        pad_ecef() + jnp.array([0.0, 0.0, 1e7]),  # far away: gravity irrelevant to attitude
        jnp.zeros(3),
        init_angular_vel=jnp.array([0.0, 0.0, omega]),
        integrator=integrator,
    )
    q, _, w, _ = _run(world, system, 100.0, 9000)
    np.testing.assert_allclose(w, [0.0, 0.0, omega], atol=1e-12)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6
    half = math.radians(90.0) / 2.0
    expected = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
    if q @ expected < 0.0:
        q = -q
    np.testing.assert_allclose(q, expected, atol=2e-3)


def _rotate_z(angle: float, v: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def _inertial_oracle(r_e0: np.ndarray, v_e0: np.ndarray, t_end: float, dt: float) -> np.ndarray:
    """Two-body coast propagated in the INERTIAL frame with numpy RK4, mapped
    back to ECEF — a fully independent formulation of the same physics."""
    omega = np.array([0.0, 0.0, OMEGA_EARTH_RADPS])
    r, v = r_e0.copy(), v_e0 + np.cross(omega, r_e0)

    def acc(r):
        return -MU_EARTH_M3S2 * r / np.linalg.norm(r) ** 3

    n = int(round(t_end / dt))
    for _ in range(n):
        k1v = acc(r)
        k1r = v
        k2v = acc(r + 0.5 * dt * k1r)
        k2r = v + 0.5 * dt * k1v
        k3v = acc(r + 0.5 * dt * k2r)
        k3r = v + 0.5 * dt * k2v
        k4v = acc(r + dt * k3r)
        k4r = v + dt * k3v
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return _rotate_z(-OMEGA_EARTH_RADPS * t_end, r)


# (integrator, rate Hz, coast s, tolerance m). The 1000 Hz case is the mission
# rate; the 100 Hz cases stress dt 10x coarser to expose the first-order vs
# fourth-order gap for the whitepaper record.
BALLISTIC_CASES = [
    ("semi_implicit", 1000.0, 20.0, 1.0),
    ("semi_implicit", 100.0, 200.0, 25.0),
    ("rk4", 100.0, 200.0, 0.5),
]


@pytest.mark.parametrize("name,rate_hz,coast_s,tol_m", BALLISTIC_CASES)
def test_ballistic_arc_vs_inertial_oracle(name, rate_hz, coast_s, tol_m):
    """MECO-state coast vs an independent inertial-frame oracle."""
    integrator = dict(INTEGRATORS)[name]
    lat, lon = math.radians(PAD_LAT_DEG), math.radians(-80.60433)
    up = np.asarray(ellipsoid_up(lat, lon))
    ned = np.asarray(ned_basis(lat, lon))
    r0 = np.asarray(pad_ecef()) + up * 61_000.0
    # ~1656 m/s at 45 deg flight path, heading northeast (recorded MECO class).
    v_dir = (ned[0] * 0.5 + ned[1] * 0.5) * math.cos(math.radians(45.0))
    v0 = 1656.0 * (
        v_dir / np.linalg.norm(v_dir) * math.cos(math.radians(45.0))
        + up * math.sin(math.radians(45.0))
    )
    world, system = build_passive(jnp.asarray(r0), jnp.asarray(v0), integrator=integrator)
    _, r_sim, _, _ = _run(world, system, rate_hz, int(round(coast_s * rate_hz)))
    r_ref = _inertial_oracle(r0, v0, coast_s, 0.01)
    err = np.linalg.norm(r_sim - r_ref)
    print(f"ballistic arc error [{name} @{rate_hz:.0f} Hz, {coast_s:.0f} s]: {err:.3f} m")
    assert err < tol_m


# Semi-implicit Euler is symplectic: on a circular orbit its energy error
# stays bounded, but the trajectory carries a bounded phase-space distortion
# of order dt * v_orbit (~8 km at the deliberately coarse dt = 1 s used here)
# that does not grow secularly. RK4 is not symplectic but its truncation at
# this dt is tiny. Both behaviors are expected and recorded, not bugs.
# Bounded energy oscillation for symplectic Euler is O(dt * omega_orbit)
# ~ 1e-3 relative at dt = 1 s; the endpoint sample sits inside that band.
ORBIT_RADIUS_TOL_M = {"semi_implicit": 16_000.0, "rk4": 5.0}
ORBIT_ENERGY_TOL = {"semi_implicit": 2e-3, "rk4": 1e-7}


@pytest.mark.parametrize("name,integrator", INTEGRATORS)
def test_orbit_radius_hold(name, integrator):
    """Circular 200 km orbit, one period at 1 Hz: bounded radius, conserved energy."""
    r_mag = WGS84_A_M + 200_000.0
    r0 = np.array([r_mag, 0.0, 0.0])
    v_circ = math.sqrt(MU_EARTH_M3S2 / r_mag)
    v0 = np.array([0.0, v_circ - OMEGA_EARTH_RADPS * r_mag, 0.0])  # ECEF velocity
    period = 2.0 * math.pi * math.sqrt(r_mag**3 / MU_EARTH_M3S2)
    world, system = build_passive(jnp.asarray(r0), jnp.asarray(v0), integrator=integrator)
    steps = int(round(period))
    _, r1, _, v1 = _run(world, system, 1.0, steps)
    radius_err = abs(np.linalg.norm(r1) - r_mag)
    # Inertial specific energy must be conserved.
    omega = np.array([0.0, 0.0, OMEGA_EARTH_RADPS])
    e0 = 0.5 * np.linalg.norm(v0 + np.cross(omega, r0)) ** 2 - MU_EARTH_M3S2 / r_mag
    e1 = 0.5 * np.linalg.norm(v1 + np.cross(omega, r1)) ** 2 - MU_EARTH_M3S2 / np.linalg.norm(r1)
    de_rel = abs((e1 - e0) / e0)
    print(
        f"orbit hold [{name} @1 Hz, {period:.0f} s]: radius err {radius_err:.2f} m, dE/E {de_rel:.2e}"
    )
    assert radius_err < ORBIT_RADIUS_TOL_M[name]
    assert de_rel < ORBIT_ENERGY_TOL[name]
