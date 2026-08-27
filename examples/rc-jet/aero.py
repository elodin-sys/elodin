"""BDX aerodynamics: package-driven evaluation plus the one frame adapter.

Longitudinal coefficients come from the package linearization and drag polar
(analysis-correlated, class C); rate, control, and lateral-directional terms
come from the opted-in class-D fallback set. Per the package contract there
is no CL floor and no alpha clamp: coefficients evaluate as-is and validity
is published separately (`aero_valid`), policy `flag_invalid_do_not_clamp`.

Frame adapter (package `frames.coefficient_adapter`, guide §5) — the only
place source-coefficient axes meet Elodin body axes. Elodin body (X fwd,
Y left, Z up) is the standard aerospace body frame (X fwd, Y right, Z down)
rotated pi about X, so the conversion is one input map and one output map
with everything between evaluated purely in the standard frame:

    inputs   : alpha identical; beta_std = asin(v_right/V) = -asin(v_left/V);
               p_std = p; q_std = -q; r_std = -r (rate hats p b/2V, q c/2V,
               r b/2V use the wind-relative V)
    evaluate : CL, CD, CY, Cl, Cm, Cn with standard-convention derivatives
               and standard-sense controls (+de TE-down/nose-down,
               +da right-roll, +dr TE-left/nose-left)
    outputs  : F  = (-D cos(a) + L sin(a),  -CY qbar S,  D sin(a) + L cos(a))
               tau = (+Cl qbar S b,  -Cm qbar S c,  -Cn qbar S b)

For the beta/r terms this reduces to the package text "tau_z = +Cn qbar S b
after beta/r sign conversion"; spelling it as evaluate-standard-then-negate
also fixes the control/adverse-yaw senses the old mixed evaluation silently
reversed (see the sign battery in tests/test_physics.py, handoff §8).

The published `beta` component is the standard aerospace sideslip (positive
= air from the right). All velocity-derived quantities (V, alpha, beta,
qbar, Mach, rate hats) use the wind-relative body velocity (guide §9: the
old code mixed in ground-relative speed for the rate terms).
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

import atmosphere
from actuators import ControlSurfaces
from bdx_model import BdxModel
from class_d_fallbacks import ClassDFallbacks
from frames import geodetic_altitude

AngleOfAttack = ty.Annotated[
    jax.Array, el.Component("alpha", el.ComponentType.F64, metadata={"priority": "80"})
]
Sideslip = ty.Annotated[
    jax.Array, el.Component("beta", el.ComponentType.F64, metadata={"priority": "79"})
]
DynamicPressure = ty.Annotated[
    jax.Array,
    el.Component("dynamic_pressure", el.ComponentType.F64, metadata={"priority": "78"}),
]
Mach = ty.Annotated[
    jax.Array, el.Component("mach", el.ComponentType.F64, metadata={"priority": "77"})
]
AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"priority": "76", "element_names": "CL,CD,CY,Cl,Cm,Cn"},
    ),
]
AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component("aero_force", el.ComponentType.SpatialMotionF64, metadata={"priority": "75"}),
]
AeroValid = ty.Annotated[
    jax.Array,
    el.Component("aero_valid", el.ComponentType.F64, metadata={"priority": "74"}),
]
Wind = ty.Annotated[jax.Array, el.Component("wind", el.ComponentType(el.PrimitiveType.F64, (3,)))]
VelocityBody = ty.Annotated[
    jax.Array,
    el.Component(
        "velocity_body",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"priority": "81", "element_names": "u,v,w"},
    ),
]

_MIN_NORMALIZATION_SPEED = 1.0


def coefficient_set(
    model: BdxModel,
    fallbacks: ClassDFallbacks,
    alpha,
    beta,
    p_hat,
    q_hat,
    r_hat,
    delta_e,
    delta_a,
    delta_r,
):
    """Evaluate [CL, CD, CY, Cl, Cm, Cn] in the source-coefficient frame.

    Pure function (jnp or float inputs) shared by the plant system, the trim
    solver, and the tests. No clamps, no floors.
    """
    lin = model.aero.linearization
    polar = model.aero.drag_polar
    fb = fallbacks.aero

    cl = lin.cl0 + lin.cl_alpha_per_rad * alpha + fb.C_Lq * q_hat + fb.C_Lde * delta_e
    cd = polar.cd0 + polar.k * cl**2 + fb.C_Dde * jnp.abs(delta_e)
    cm = lin.cm0 + lin.cm_alpha_per_rad * alpha + fb.C_mq * q_hat + fb.C_mde * delta_e
    cy = (
        fb.C_Ybeta * beta
        + fb.C_Yp * p_hat
        + fb.C_Yr * r_hat
        + fb.C_Yda * delta_a
        + fb.C_Ydr * delta_r
    )
    c_roll = (
        fb.C_lbeta * beta
        + fb.C_lp * p_hat
        + fb.C_lr * r_hat
        + fb.C_lda * delta_a
        + fb.C_ldr * delta_r
    )
    c_yaw = (
        fb.C_nbeta * beta
        + fb.C_np * p_hat
        + fb.C_nr * r_hat
        + fb.C_nda * delta_a
        + fb.C_ndr * delta_r
    )
    return jnp.array([cl, cd, cy, c_roll, cm, c_yaw])


def adapter_body_wrench(model: BdxModel, coefs, alpha, q_bar):
    """The one coefficient-to-body-frame adapter (see module docstring).

    `coefs` are standard-frame [CL, CD, CY, Cl, Cm, Cn]; the returned force
    and torque are Elodin body-frame (X fwd, Y left, Z up), i.e. the standard
    wrench rotated pi about X: (Fy, Fz, tau_y, tau_z) negate.
    """
    cl, cd, cy, c_roll, cm, c_yaw = coefs
    s = model.reference.area_m2
    b = model.reference.span_m
    c = model.reference.mac_m

    lift = cl * q_bar * s
    drag = cd * q_bar * s

    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    force = jnp.array(
        [
            -drag * cos_a + lift * sin_a,
            -cy * q_bar * s,
            drag * sin_a + lift * cos_a,
        ]
    )
    torque = jnp.array(
        [
            c_roll * q_bar * s * b,
            -cm * q_bar * s * c,
            -c_yaw * q_bar * s * b,
        ]
    )
    return force, torque


def standard_rate_hats(model: BdxModel, ang_vel_body, airspeed):
    """Nondimensional standard-frame rates from Elodin body rates.

    Elodin pitch/yaw axes (+Y left, +Z up) are the negations of the standard
    +Y right / +Z down axes, so q and r flip; roll is shared.
    """
    v_safe = jnp.maximum(airspeed, _MIN_NORMALIZATION_SPEED)
    b = model.reference.span_m
    c = model.reference.mac_m
    p, q, r = ang_vel_body
    p_hat = p * b / (2.0 * v_safe)
    q_hat = -q * c / (2.0 * v_safe)
    r_hat = -r * b / (2.0 * v_safe)
    return p_hat, q_hat, r_hat


@el.map
def compute_velocity_body(pos: el.WorldPos, vel: el.WorldVel, wind: Wind) -> VelocityBody:
    """Wind-relative velocity in the body frame (wind is a world-frame vector)."""
    v_rel_world = vel.linear() - wind
    return pos.angular().inverse() @ v_rel_world


@el.map
def compute_aero_angles(v_body: VelocityBody) -> tuple[AngleOfAttack, Sideslip]:
    """Alpha/beta from wind-relative body velocity (X fwd, Y left, Z up).

    Nose above the velocity vector means negative w, so alpha = atan2(-w, u).
    Beta is published in the standard aerospace sense (positive = air from
    the right = negative Y-left velocity component).
    """
    u, v, w = v_body
    speed = jnp.linalg.norm(v_body)
    speed_safe = jnp.maximum(speed, _MIN_NORMALIZATION_SPEED)
    alpha = jnp.arctan2(-w, u)
    beta = jnp.arcsin(jnp.clip(-v / speed_safe, -1.0, 1.0))
    return alpha, beta


@el.map
def dynamic_pressure_and_mach(
    pos: el.WorldPos, v_body: VelocityBody
) -> tuple[DynamicPressure, Mach]:
    """Dynamic pressure and Mach from ISA density at geodetic altitude."""
    altitude = geodetic_altitude(pos.linear())
    rho = atmosphere.density(altitude)
    a = atmosphere.speed_of_sound(altitude)
    speed = jnp.linalg.norm(v_body)
    return 0.5 * rho * speed**2, speed / a


def build_aero_coefs(model: BdxModel, fallbacks: ClassDFallbacks):
    @el.map
    def compute_aero_coefs(
        alpha: AngleOfAttack,
        beta: Sideslip,
        control_surfaces: ControlSurfaces,
        pos: el.WorldPos,
        vel: el.WorldVel,
        v_body: VelocityBody,
    ) -> AeroCoefs:
        delta_e, delta_a, delta_r = control_surfaces
        ang_vel_body = pos.angular().inverse() @ vel.angular()
        airspeed = jnp.linalg.norm(v_body)
        p_hat, q_hat, r_hat = standard_rate_hats(model, ang_vel_body, airspeed)
        return coefficient_set(
            model, fallbacks, alpha, beta, p_hat, q_hat, r_hat, delta_e, delta_a, delta_r
        )

    return compute_aero_coefs


# A singleton Re/m bound is the analysis condition, not an operational band.
# ±10% is enough for hold-trim TAS wobble without inventing a flight envelope.
_SINGLETON_RE_REL = 0.10


def reynolds_per_m(altitude_m, airspeed):
    """Unit Reynolds number ρ V / μ from the shared ISA + Sutherland atmosphere."""
    mu = atmosphere.dynamic_viscosity(altitude_m)
    return atmosphere.density(altitude_m) * airspeed / mu


def _reynolds_envelope(model: BdxModel) -> tuple[float, float]:
    lo, hi = model.validity.reynolds_per_m
    if lo == hi:
        return lo * (1.0 - _SINGLETON_RE_REL), hi * (1.0 + _SINGLETON_RE_REL)
    return lo, hi


def validity_flag(model: BdxModel, alpha, mach, re_per_m):
    """1.0 inside the package validity envelope, else 0.0 (pure function).

    ANDs every numeric bound in guide §5 (attached-flow α, tabulated α, Mach,
    Re/m). Policy flag_invalid_do_not_clamp: the flag never alters the forces.
    """
    flow_lo, flow_hi = (jnp.deg2rad(v) for v in model.validity.attached_flow_alpha_deg)
    table_lo, table_hi = (jnp.deg2rad(v) for v in model.validity.polar_table_alpha_deg)
    mach_lo, mach_hi = model.validity.mach
    re_lo, re_hi = _reynolds_envelope(model)
    alpha_ok = (alpha >= flow_lo) & (alpha <= flow_hi) & (alpha >= table_lo) & (alpha <= table_hi)
    mach_ok = (mach >= mach_lo) & (mach <= mach_hi)
    re_ok = (re_per_m >= re_lo) & (re_per_m <= re_hi)
    return jnp.where(alpha_ok & mach_ok & re_ok, 1.0, 0.0)


def build_aero_validity(model: BdxModel):
    @el.map
    def aero_validity(
        alpha: AngleOfAttack, mach: Mach, pos: el.WorldPos, v_body: VelocityBody
    ) -> AeroValid:
        altitude = geodetic_altitude(pos.linear())
        airspeed = jnp.linalg.norm(v_body)
        return validity_flag(model, alpha, mach, reynolds_per_m(altitude, airspeed))

    return aero_validity


def build_aero_forces(model: BdxModel):
    @el.map
    def aero_forces(
        aero_coefs: AeroCoefs, alpha: AngleOfAttack, q_bar: DynamicPressure
    ) -> AeroForce:
        force, torque = adapter_body_wrench(model, aero_coefs, alpha, q_bar)
        return el.SpatialForce(linear=force, torque=torque)

    return aero_forces


@el.map
def apply_aero_forces(pos: el.WorldPos, aero_force: AeroForce, force: el.Force) -> el.Force:
    """Rotate the body-frame aero wrench into the world (ECEF) frame."""
    return force + pos.angular() @ aero_force
