"""
BDX Aerodynamic Model - Body-Axis Approach (following rocket example)

Uses body-axis coefficients like the rocket example:
- CA: Axial force (along body X, positive aft = drag)  
- CZ: Normal force (along body Z, positive up = lift)
- CY: Side force (along body Y)
- Cl, Cm, Cn: Roll, pitch, yaw moments

This avoids the wind-to-body transformation that was causing instability.
"""

import typing as ty
import elodin as el
import jax
import jax.numpy as jnp

from config import BDXConfig


# Component types
AngleOfAttack = ty.Annotated[
    jax.Array, 
    el.Component("alpha", el.ComponentType.F64, metadata={"priority": "80"})
]
Sideslip = ty.Annotated[
    jax.Array, 
    el.Component("beta", el.ComponentType.F64, metadata={"priority": "79"})
]
DynamicPressure = ty.Annotated[
    jax.Array, 
    el.Component("dynamic_pressure", el.ComponentType.F64, metadata={"priority": "78"})
]
Mach = ty.Annotated[
    jax.Array, 
    el.Component("mach", el.ComponentType.F64, metadata={"priority": "77"})
]

# 6-element coefficient array per whitepaper Section 8.2
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
    el.Component("aero_force", el.ComponentType.SpatialMotionF64, metadata={"priority": "75"})
]
Wind = ty.Annotated[
    jax.Array, 
    el.Component("wind", el.ComponentType(el.PrimitiveType.F64, (3,)))
]
VelocityBody = ty.Annotated[
    jax.Array, 
    el.Component("velocity_body", el.ComponentType(el.PrimitiveType.F64, (3,)), metadata={"priority": "81", "element_names": "u,v,w"})
]

ControlSurfaces = ty.Annotated[
    jax.Array,
    el.Component("control_surfaces", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@el.map
def compute_velocity_body(pos: el.WorldPos, vel: el.WorldVel, wind: Wind) -> VelocityBody:
    """Transform world velocity to body frame, accounting for wind."""
    v_rel_world = vel.linear() - wind
    v_body = pos.angular().inverse() @ v_rel_world
    return v_body


@el.map
def compute_aero_angles(v_body: VelocityBody) -> tuple[AngleOfAttack, Sideslip]:
    """
    Compute angle of attack and sideslip from body velocity.
    
    For Z-up body frame:
    - alpha: angle in X-Z plane (nose up positive)
    - beta: angle in X-Y plane (nose right positive)
    """
    u, v, w = v_body
    V = jnp.sqrt(u**2 + v**2 + w**2)
    V_safe = jnp.maximum(V, 1.0)
    
    # Alpha: angle between velocity vector and X-Y plane, projected onto X-Z plane
    # Positive when w > 0 (velocity has upward component, nose pitched up)
    alpha = jnp.arctan2(w, jnp.maximum(jnp.abs(u), 1e-6))
    
    # Beta: sideslip angle
    beta = jnp.arcsin(jnp.clip(v / V_safe, -1.0, 1.0))
    
    return alpha, beta


@el.map
def dynamic_pressure_and_mach(pos: el.WorldPos, v_body: VelocityBody) -> tuple[DynamicPressure, Mach]:
    """Compute dynamic pressure and Mach number using ISA atmosphere."""
    altitude = jnp.maximum(pos.linear()[2], 0.0)
    
    # ISA atmosphere (troposphere)
    T = 288.15 - 0.0065 * altitude
    T = jnp.clip(T, 216.65, 288.15)
    p = 101325.0 * (T / 288.15) ** 5.2561
    rho = p / (287.05 * T)
    a = jnp.sqrt(1.4 * 287.05 * T)
    
    V = jnp.linalg.norm(v_body)
    q_bar = 0.5 * rho * V**2
    mach = V / a
    
    return q_bar, mach


@el.map
def compute_aero_coefs(
    alpha: AngleOfAttack,
    beta: Sideslip,
    control_surfaces: ControlSurfaces,
    pos: el.WorldPos,
    vel: el.WorldVel,
    q_bar: DynamicPressure,
) -> AeroCoefs:
    """
    Compute aerodynamic coefficients using polynomial model from whitepaper.
    
    Returns [CL, CD, CY, Cl, Cm, Cn] in body-axis form.
    """
    config = BDXConfig.GLOBAL
    aero = config.aero
    
    delta_e, delta_a, delta_r = control_surfaces
    
    # Get angular rates in body frame
    ang_vel_body = pos.angular().inverse() @ vel.angular()
    p, q, r = ang_vel_body  # roll, pitch, yaw rates
    
    # Airspeed for non-dimensionalization
    v_body_vec = pos.angular().inverse() @ vel.linear()
    V = jnp.maximum(jnp.linalg.norm(v_body_vec), 10.0)
    
    # Non-dimensional rates
    c = config.mean_chord
    b = config.wingspan
    q_hat = q * c / (2.0 * V)  # Pitch rate
    p_hat = p * b / (2.0 * V)  # Roll rate
    r_hat = r * b / (2.0 * V)  # Yaw rate
    
    # Clamp alpha to valid range (pre-stall)
    alpha_clamped = jnp.clip(alpha, jnp.deg2rad(-12.0), jnp.deg2rad(12.0))
    
    # === LONGITUDINAL (Lift, Drag, Pitch) ===
    # Lift coefficient - ensure CL stays positive in normal flight
    CL_base = aero.C_L0 + aero.C_Lalpha * alpha_clamped + aero.C_Lq * q_hat + aero.C_Lde * delta_e
    # Ensure minimum CL to prevent negative lift (stall protection)
    CL = jnp.maximum(CL_base, 0.01)
    
    # Drag coefficient (parabolic polar)
    CD = aero.C_D0 + aero.k * CL**2 + aero.C_Dde * jnp.abs(delta_e)
    
    # Pitch moment coefficient
    Cm = (aero.C_m0 
          + aero.C_malpha * alpha_clamped 
          + aero.C_mq * q_hat 
          + aero.C_mde * delta_e)
    
    # === LATERAL-DIRECTIONAL (Side force, Roll, Yaw) ===
    # Side force coefficient
    CY = (aero.C_Ybeta * beta 
          + aero.C_Yp * p_hat 
          + aero.C_Yr * r_hat 
          + aero.C_Yda * delta_a 
          + aero.C_Ydr * delta_r)
    
    # Roll moment coefficient
    Cl = (aero.C_lbeta * beta 
          + aero.C_lp * p_hat 
          + aero.C_lr * r_hat 
          + aero.C_lda * delta_a 
          + aero.C_ldr * delta_r)
    
    # Yaw moment coefficient
    Cn = (aero.C_nbeta * beta 
          + aero.C_np * p_hat 
          + aero.C_nr * r_hat 
          + aero.C_nda * delta_a 
          + aero.C_ndr * delta_r)
    
    return jnp.array([CL, CD, CY, Cl, Cm, Cn])


@el.map
def aero_forces(
    aero_coefs: AeroCoefs,
    alpha: AngleOfAttack,
    q_bar: DynamicPressure,
) -> AeroForce:
    """
    Convert coefficients to body-frame forces and moments.
    
    Following rocket example approach: coefficients are applied
    directly in body frame to avoid transformation issues.
    
    Body frame (Z-up):
    - X forward, positive = forward
    - Y left, positive = left  
    - Z up, positive = up
    
    Forces:
    - Drag opposes motion: F_x = -D (when moving forward)
    - Lift acts upward: F_z = +L  
    - Side force: F_y = Y
    
    Moments:
    - Roll about X: tau_x = Cl_moment
    - Pitch about Y: tau_y = Cm_moment  
    - Yaw about Z: tau_z = Cn_moment
    """
    config = BDXConfig.GLOBAL
    CL, CD, CY, Cl, Cm, Cn = aero_coefs
    
    S = config.wing_area
    c = config.mean_chord
    b = config.wingspan
    
    # Force magnitudes
    L = CL * q_bar * S  # Lift
    D = CD * q_bar * S  # Drag
    Y = CY * q_bar * S  # Side force
    
    # Convert from wind/stability axis to body axis
    # This is the key transformation that was problematic before
    # 
    # Wind axis: L perpendicular to velocity, D parallel to velocity
    # Body axis: Forces along body X, Y, Z
    #
    # When alpha > 0 (nose up), velocity is below body X-axis
    # Lift acts perpendicular to velocity (up and slightly forward)
    # Drag acts opposite to velocity (back and slightly up)
    ca = jnp.cos(alpha)
    sa = jnp.sin(alpha)
    
    # Transform: [F_x, F_z] = R(-alpha) @ [-D, L]
    # where R(-alpha) rotates from wind to body frame
    F_x = -D * ca + L * sa   # Axial force (mostly drag, negative = forward)
    F_z = -D * sa + L * ca   # Normal force (mostly lift, positive = up)
    F_y = Y                   # Side force
    
    # Moments (already in body frame)
    tau_x = Cl * q_bar * S * b  # Roll moment
    tau_y = Cm * q_bar * S * c  # Pitch moment
    tau_z = Cn * q_bar * S * b  # Yaw moment
    
    return el.SpatialForce(
        linear=jnp.array([F_x, F_y, F_z]),
        torque=jnp.array([tau_x, tau_y, tau_z])
    )


@el.map
def apply_aero_forces(pos: el.WorldPos, aero_force: AeroForce, force: el.Force) -> el.Force:
    """Transform body-frame aero forces to world frame and apply."""
    force_world = pos.angular() @ aero_force
    return force + force_world
