"""Sign battery, adapter, trim, and validity tests (guide §5, §7).

Direction conventions asserted here, all in the Elodin body frame
(X forward, Y left, Z up):

    +tau_x = roll right    +tau_y = nose down    +tau_z = nose left

and standard-aerospace inputs: +alpha nose above velocity, +beta air from
the right, +elevator TE-down (nose-down), +aileron roll-right, +rudder
TE-left (nose-left).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import bdx_model
import trim as trim_mod
from aero import (
    adapter_body_wrench,
    coefficient_set,
    reynolds_per_m,
    standard_rate_hats,
    validity_flag,
)
from class_d_fallbacks import FALLBACKS
from frames import (
    apparent_gravity,
    ecef_to_geodetic,
    enu_basis,
    geodetic_to_ecef,
    level_attitude_ecef,
    quaternion_xyzw_from_matrix,
)

MODEL = bdx_model.load()
Q_BAR = 900.0  # representative cruise dynamic pressure (Pa)
CRUISE_RE = MODEL.validity.reynolds_per_m[0]


def wrench(alpha=0.0, beta=0.0, p=0.0, q=0.0, r=0.0, de=0.0, da=0.0, dr=0.0, airspeed=37.8):
    """Body wrench for a state expressed with Elodin body rates p/q/r."""
    p_hat, q_hat, r_hat = standard_rate_hats(MODEL, jnp.array([p, q, r]), airspeed)
    coefs = coefficient_set(MODEL, FALLBACKS, alpha, beta, p_hat, q_hat, r_hat, de, da, dr)
    force, torque = adapter_body_wrench(MODEL, coefs, alpha, Q_BAR)
    return np.asarray(force), np.asarray(torque)


def test_config_stability_signs():
    lin = MODEL.aero.linearization
    assert lin.cl_alpha_per_rad > 0
    assert lin.cm_alpha_per_rad < 0
    assert FALLBACKS.aero.C_mde < 0, "C_mde must be negative (guide §3.3 sign fix)"


def test_alpha_gives_lift_and_restoring_pitch():
    _, torque_low = wrench(alpha=0.0)
    force_hi, torque_hi = wrench(alpha=math.radians(4.0))
    force_lo, _ = wrench(alpha=0.0)
    # More alpha, more lift (body +Z).
    assert force_hi[2] > force_lo[2]
    # Restoring: nose-down (+tau_y) increment for +alpha.
    assert torque_hi[1] > torque_low[1]


def test_inverted_flight_produces_negative_lift():
    """Guards against reintroducing the CL floor (guide §7.4)."""
    force, _ = wrench(alpha=math.radians(-5.0))
    assert force[2] < 0.0


def test_no_alpha_clamp():
    """Coefficients evaluate as-is outside the envelope; validity flags it."""
    lin = MODEL.aero.linearization
    alpha = math.radians(20.0)
    coefs = coefficient_set(MODEL, FALLBACKS, alpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert float(coefs[0]) == pytest.approx(lin.cl0 + lin.cl_alpha_per_rad * alpha)
    assert float(validity_flag(MODEL, alpha, 0.15, CRUISE_RE)) == 0.0
    assert float(validity_flag(MODEL, math.radians(5.0), 0.15, CRUISE_RE)) == 1.0
    assert float(validity_flag(MODEL, math.radians(5.0), 0.35, CRUISE_RE)) == 0.0


def test_validity_uses_all_package_bounds():
    """Guide §5: aero_valid is false outside any declared bound."""
    assert float(validity_flag(MODEL, math.radians(10.0), 0.15, CRUISE_RE)) == 0.0
    assert float(validity_flag(MODEL, math.radians(5.0), 0.15, CRUISE_RE)) == 1.0
    assert float(validity_flag(MODEL, math.radians(5.0), 0.15, CRUISE_RE * 2.0)) == 0.0
    lin = MODEL.aero.linearization
    re = float(reynolds_per_m(lin.reference_altitude_m, lin.reference_airspeed_mps))
    assert re == pytest.approx(CRUISE_RE, rel=0.01)


def test_beta_gives_restoring_yaw():
    # Air from the right (+beta): weathercock yaws the nose right (-tau_z).
    _, torque = wrench(beta=math.radians(5.0))
    assert torque[2] < 0.0


def test_elevator_direction():
    _, torque_base = wrench()
    force_up, torque_up = wrench(de=math.radians(5.0))
    force_base, _ = wrench()
    # +elevator (TE down): nose-down moment and a little more lift.
    assert torque_up[1] > torque_base[1]
    assert force_up[2] > force_base[2]


def test_aileron_direction_and_adverse_yaw():
    _, torque = wrench(da=math.radians(5.0))
    assert torque[0] > 0.0, "+aileron must roll right"
    assert torque[2] > 0.0, "adverse yaw: nose-left with right aileron (C_nda < 0)"


def test_rudder_direction():
    force, torque = wrench(dr=math.radians(5.0))
    assert torque[2] > 0.0, "+rudder (TE left) must yaw nose-left"
    assert force[1] < 0.0, "+rudder pushes the tail left, force to the right (-Y)"


def test_rate_damping_signs():
    # Roll right (p > 0) is opposed by tau_x < 0.
    _, torque = wrench(p=1.0)
    assert torque[0] < 0.0
    # Nose-down pitch rate (q > 0 in Y-left frame) is opposed by tau_y < 0.
    _, torque = wrench(q=1.0)
    assert torque[1] < 0.0
    # Nose-left yaw rate (r > 0 about Z-up) is opposed by tau_z < 0.
    _, torque = wrench(r=1.0)
    assert torque[2] < 0.0


def test_thrust_line_pitch_moment_sign():
    """Thrust above the CG must pitch the nose down (+tau_y in Y-left)."""
    offset = jnp.array(MODEL.propulsion.thrust_application_body_m)
    force = jnp.array(MODEL.propulsion.thrust_axis_body) * 100.0
    torque = np.asarray(jnp.cross(offset, force))
    assert offset[2] > 0.0
    assert torque[1] > 0.0


def test_cruise_trim_matches_package_anchor():
    """T7: solver reproduces the package cruise anchor (values read from the
    package, not copied literals)."""
    anchor = MODEL.performance_anchors["cruise"]
    solution = trim_mod.solve_level_trim(
        MODEL, FALLBACKS, 36.23, -116.97, anchor["altitude_m"], anchor["tas_mps"]
    )
    assert solution.valid
    assert math.degrees(solution.alpha_rad) == pytest.approx(anchor["alpha_deg"], abs=0.3)
    assert solution.effective_throttle == pytest.approx(anchor["throttle"], abs=0.03)
    assert abs(solution.elevator_rad) < math.radians(1.0)
    assert abs(solution.lift_residual_n) < 1.0
    assert abs(solution.pitch_residual_nm) < 1.0


def test_dash_regression():
    """T8: the full-throttle level dash speed tracks the package dash anchor.

    Known deviation, documented in ai-context/bdx/openair_integration_feedback.md:
    the anchor (CD 0.0316 at 85.1 m/s) was solved with the producer's
    Reynolds-dependent drag model, while the exported polar is a single-Re fit
    whose CD can never drop below CD0 = 0.0333. Consuming only the exported
    sidecars, full throttle balances ~4% slower. The 5 m/s band accepts that
    documented gap while still catching structural regressions (a wrong S,
    span, adapter sign, or map axis moves this by tens of m/s)."""
    anchor = MODEL.performance_anchors["dash"]
    altitude = anchor["altitude_m"]

    def required_throttle(tas):
        return trim_mod.solve_level_trim(MODEL, FALLBACKS, 36.23, -116.97, altitude, tas)

    low, high = 60.0, 110.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        solution = required_throttle(mid)
        if solution.effective_throttle < 1.0 and solution.valid:
            low = mid
        else:
            high = mid
    assert low == pytest.approx(anchor["tas_mps"], abs=5.0)
    assert low > 2.0 * MODEL.performance_anchors["cruise"]["tas_mps"]


def test_load_factor_dimensionalization():
    """T9: the +6 g wingbox anchor closes through our qbar and S plumbing."""
    anchor = MODEL.performance_anchors["positive_g"]
    tas = anchor["tas_mps"]
    q_bar = 0.5 * 1.225 * tas**2
    lift = q_bar * MODEL.reference.area_m2 * anchor["CL"]
    weight = MODEL.mass.mass_kg * 9.80665
    assert lift / weight == pytest.approx(anchor["load_factor"], rel=0.03)


def test_level_attitude_and_quaternion():
    lat, lon = math.radians(36.23), math.radians(-116.97)
    basis = np.asarray(enu_basis(lat, lon))
    attitude = level_attitude_ecef(lat, lon, 0.0)
    # Heading 0 = north, level: X = north, Z = up.
    assert np.allclose(attitude[:, 0], basis[1], atol=1e-12)
    assert np.allclose(attitude[:, 2], basis[2], atol=1e-12)
    assert np.allclose(attitude @ attitude.T, np.eye(3), atol=1e-12)
    q = quaternion_xyzw_from_matrix(attitude)
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_geodetic_roundtrip_and_gravity():
    lat, lon, alt = math.radians(36.23), math.radians(-116.97), 300.0
    r = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = (float(v) for v in ecef_to_geodetic(r))
    assert lat2 == pytest.approx(lat, abs=1e-9)
    assert lon2 == pytest.approx(lon, abs=1e-9)
    assert alt2 == pytest.approx(alt, abs=1e-3)
    g = float(jnp.linalg.norm(apparent_gravity(r)))
    assert 9.75 < g < 9.83
    # Apparent gravity points along the local down within deflection tolerance.
    up = np.asarray(enu_basis(lat, lon))[2]
    cos_angle = -float(np.dot(np.asarray(apparent_gravity(r)), up)) / g
    assert cos_angle > 0.99999
