"""Level-flight equilibrium solver.

Solves (alpha, elevator, effective throttle) for steady level flight at a
requested geodetic altitude and TAS, from the same package + class-D data the
plant uses (guide §9.5: scenarios must solve their own equilibrium, never
reuse a trim row off-condition).

Balance equations (standard-aerospace source frame, thrust along body X at
the package thrust line):

    lift:   L + T sin(alpha) = m g_apparent
    drag:   T cos(alpha) = D
    pitch:  Cm(alpha, de) + Cm_thrust = 0,  Cm_thrust = z_std T / (qbar S c)

where z_std is the thrust-line offset in the source frame (Z down), i.e.
minus the package's body-frame z offset. The elevator term depends on the
class-D C_mde, so any solved trim is logged as class-D-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import atmosphere
from bdx_model import BdxModel
from class_d_fallbacks import ClassDFallbacks
from frames import apparent_gravity, geodetic_to_ecef
from propulsion import trilinear

_ITERATIONS = 12


@dataclass(frozen=True)
class TrimSolution:
    altitude_m: float
    tas_mps: float
    mach: float
    alpha_rad: float
    elevator_rad: float
    effective_throttle: float
    thrust_n: float
    lift_residual_n: float
    pitch_residual_nm: float
    valid: bool


def _thrust_to_throttle(model: BdxModel, altitude_m: float, mach: float, thrust_n: float):
    """Invert the (monotone-in-throttle) propulsion map at a flight condition."""
    grid = model.propulsion_map
    axes = (
        jnp.asarray(grid.altitudes_m),
        jnp.asarray(grid.machs),
        jnp.asarray(grid.throttles),
    )
    table = jnp.asarray(grid.thrust_n)
    throttles = np.asarray(grid.throttles)
    curve = np.array([float(trilinear(axes, table, (altitude_m, mach, t))) for t in throttles])
    throttle = float(np.interp(thrust_n, curve, throttles))
    achievable = curve[0] - 1e-9 <= thrust_n <= curve[-1] + 1e-9
    return throttle, achievable


def solve_level_trim(
    model: BdxModel,
    fallbacks: ClassDFallbacks,
    site_lat_deg: float,
    site_lon_deg: float,
    altitude_m: float,
    tas_mps: float,
) -> TrimSolution:
    lin = model.aero.linearization
    polar = model.aero.drag_polar
    fb = fallbacks.aero
    s = model.reference.area_m2
    c = model.reference.mac_m

    rho = float(atmosphere.density(altitude_m))
    a_sound = float(atmosphere.speed_of_sound(altitude_m))
    mach = tas_mps / a_sound
    q_bar = 0.5 * rho * tas_mps**2

    r_ecef = geodetic_to_ecef(np.deg2rad(site_lat_deg), np.deg2rad(site_lon_deg), altitude_m)
    g = float(jnp.linalg.norm(apparent_gravity(r_ecef)))
    weight = model.mass.mass_kg * g

    # Thrust-line offset in the source frame (Z down): negate the body-frame z.
    z_std = -model.propulsion.thrust_application_body_m[2]

    alpha = (weight / (q_bar * s) - lin.cl0) / lin.cl_alpha_per_rad
    delta_e = 0.0
    thrust = 0.0
    for _ in range(_ITERATIONS):
        cl_required = (weight - thrust * np.sin(alpha)) / (q_bar * s)
        alpha = (cl_required - lin.cl0 - fb.C_Lde * delta_e) / lin.cl_alpha_per_rad
        cm_thrust = z_std * thrust / (q_bar * s * c)
        delta_e = -(lin.cm0 + lin.cm_alpha_per_rad * alpha + cm_thrust) / fb.C_mde
        cl = lin.cl0 + lin.cl_alpha_per_rad * alpha + fb.C_Lde * delta_e
        cd = polar.cd0 + polar.k * cl**2 + fb.C_Dde * abs(delta_e)
        thrust = q_bar * s * cd / np.cos(alpha)

    throttle, achievable = _thrust_to_throttle(model, altitude_m, mach, thrust)

    cl = lin.cl0 + lin.cl_alpha_per_rad * alpha + fb.C_Lde * delta_e
    lift_residual = q_bar * s * cl + thrust * np.sin(alpha) - weight
    cm_total = (
        lin.cm0
        + lin.cm_alpha_per_rad * alpha
        + fb.C_mde * delta_e
        + z_std * thrust / (q_bar * s * c)
    )
    pitch_residual = cm_total * q_bar * s * c

    alpha_lo, alpha_hi = np.deg2rad(model.validity.attached_flow_alpha_deg)
    mach_lo, mach_hi = model.validity.mach
    valid = bool(
        achievable
        and alpha_lo <= alpha <= alpha_hi
        and mach_lo <= mach <= mach_hi
        and abs(lift_residual) < 1e-3 * weight
        and abs(pitch_residual) < 1.0
    )

    return TrimSolution(
        altitude_m=altitude_m,
        tas_mps=tas_mps,
        mach=mach,
        alpha_rad=float(alpha),
        elevator_rad=float(delta_e),
        effective_throttle=throttle,
        thrust_n=float(thrust),
        lift_residual_n=float(lift_residual),
        pitch_residual_nm=float(pitch_residual),
        valid=valid,
    )
