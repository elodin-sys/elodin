"""
BDX Aerodynamic Model

Implements polynomial aerodynamic coefficient model and force/moment computation
following Section 3 of BDX_Simulation_Whitepaper.md.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la

from config import BDXConfig

# Component type definitions

AngleOfAttack = ty.Annotated[
    jax.Array,
    el.Component("alpha", el.ComponentType.F64, metadata={"priority": 80}),
]

Sideslip = ty.Annotated[
    jax.Array,
    el.Component("beta", el.ComponentType.F64, metadata={"priority": 79}),
]

DynamicPressure = ty.Annotated[
    jax.Array,
    el.Component("dynamic_pressure", el.ComponentType.F64, metadata={"priority": 78}),
]

Mach = ty.Annotated[
    jax.Array,
    el.Component("mach", el.ComponentType.F64, metadata={"priority": 77}),
]

AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "CL,CD,CY,Cl,Cm,Cn", "priority": 76},
    ),
]

AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "aero_force",
        el.ComponentType.SpatialMotionF64,
        metadata={"element_names": "τx,τy,τz,Fx,Fy,Fz", "priority": 75},
    ),
]

Wind = ty.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

VelocityBody = ty.Annotated[
    jax.Array,
    el.Component(
        "velocity_body",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "u,v,w", "priority": 81},
    ),
]

# Control surfaces component (defined locally to avoid circular imports)
# This matches the same component defined in actuators.py by name
ControlSurfaces = ty.Annotated[
    jax.Array,
    el.Component(
        "control_surfaces",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "elevator,aileron,rudder"},
    ),
]


@el.map
def compute_velocity_body(
    pos: el.WorldPos,
    vel: el.WorldVel,
    wind: Wind,
) -> VelocityBody:
    """
    Compute velocity in body frame.
    
    Transforms world velocity (relative to wind) to body frame.
    Returns [u, v, w] where:
    - u = velocity along body x-axis (forward)
    - v = velocity along body y-axis (right)
    - w = velocity along body z-axis (down)
    """
    # Wind-relative velocity in world frame
    v_rel = vel.linear() - wind
    
    # Transform to body frame
    v_body = pos.angular().inverse() @ v_rel
    
    return v_body


@el.map
def compute_aero_angles(v_body: VelocityBody) -> tuple[AngleOfAttack, Sideslip]:
    """
    Compute angle of attack (α) and sideslip angle (β) from body velocity.
    
    α: angle between velocity and body X-axis in X-Z plane
    β: angle between velocity projection and X-axis (sideslip)
    """
    u, v, w = v_body
    
    # Airspeed magnitude
    V = jnp.sqrt(u**2 + v**2 + w**2)
    
    # Angle of attack (positive when nose up relative to velocity)
    # Note: In aerospace convention, w is positive down, so α = arctan(w/u)
    alpha = jnp.arctan2(w, jnp.clip(u, 1e-6, jnp.inf))
    
    # Sideslip angle (positive when nose left of velocity)
    beta = jnp.arcsin(jnp.clip(v / jnp.clip(V, 1e-6, jnp.inf), -1.0, 1.0))
    
    return alpha, beta


@el.map
def compute_atmosphere_and_dynamic_pressure(
    pos: el.WorldPos,
    v_body: VelocityBody,
) -> tuple[DynamicPressure, Mach]:
    """
    Compute atmospheric properties and dynamic pressure using ISA model.
    
    Returns dynamic pressure (q̄) and Mach number.
    """
    # International Standard Atmosphere (ISA) model
    altitude = pos.linear()[2]
    
    # Temperature (K) - Troposphere model
    T = 288.15 - 0.0065 * altitude
    T = jnp.clip(T, 216.65, 288.15)  # Don't go below tropopause temp
    
    # Pressure (Pa)
    p = 101325.0 * (T / 288.15) ** 5.2561
    
    # Density (kg/m³)
    rho = p / (287.05 * T)
    
    # Speed of sound (m/s)
    a = jnp.sqrt(1.4 * 287.05 * T)
    
    # Airspeed magnitude
    u, v, w = v_body
    V = jnp.sqrt(u**2 + v**2 + w**2)
    
    # Dynamic pressure
    q_bar = 0.5 * rho * V**2
    q_bar = jnp.clip(q_bar, 1e-6, jnp.inf)  # Avoid division by zero
    
    # Mach number
    mach = V / a
    
    return q_bar, mach


@el.map
def compute_aero_coefs(
    alpha: AngleOfAttack,
    beta: Sideslip,
    v_body: VelocityBody,
    pos: el.WorldPos,
    vel: el.WorldVel,
    control_surfaces: ControlSurfaces,
) -> AeroCoefs:
    """
    Compute aerodynamic coefficients using polynomial stability derivative model.
    
    Implements equations from Section 3.2 of whitepaper:
    - Longitudinal: C_L, C_D, C_m
    - Lateral-directional: C_Y, C_l, C_n
    """
    config = BDXConfig.GLOBAL
    aero = config.aero
    
    # Extract control surface deflections (radians)
    delta_e, delta_a, delta_r = control_surfaces
    
    # Get body angular rates by transforming world angular velocity to body frame
    ang_vel_body = pos.angular().inverse() @ vel.angular()
    p, q, r = ang_vel_body
    
    # Airspeed for rate terms
    u, v, w = v_body
    V = jnp.sqrt(u**2 + v**2 + w**2)
    V = jnp.clip(V, 1.0, jnp.inf)  # Minimum 1 m/s
    
    # Non-dimensional rate terms
    q_bar = q * config.mean_chord / (2.0 * V)
    p_bar = p * config.wingspan / (2.0 * V)
    r_bar = r * config.wingspan / (2.0 * V)
    
    # --- LONGITUDINAL COEFFICIENTS ---
    
    # Lift coefficient
    C_L = (aero.C_L0 + 
           aero.C_Lalpha * alpha + 
           aero.C_Lq * q_bar + 
           aero.C_Lde * delta_e)
    
    # Apply stall model (simple linear rolloff beyond stall angle)
    alpha_deg = jnp.rad2deg(jnp.abs(alpha))
    stall_factor = jnp.where(
        alpha_deg < aero.alpha_stall,
        1.0,
        jnp.where(
            alpha_deg < aero.alpha_stall + 10.0,
            1.0 - 0.5 * (alpha_deg - aero.alpha_stall) / 10.0,
            0.5  # Post-stall plateau
        )
    )
    C_L = C_L * stall_factor
    
    # Drag coefficient (includes induced drag)
    C_D = aero.C_D0 + aero.k * C_L**2 + aero.C_Dde * jnp.abs(delta_e)
    
    # Pitching moment coefficient
    C_m = (aero.C_m0 + 
           aero.C_malpha * alpha + 
           aero.C_mq * q_bar + 
           aero.C_mde * delta_e)
    
    # --- LATERAL-DIRECTIONAL COEFFICIENTS ---
    
    # Side force coefficient
    C_Y = (aero.C_Ybeta * beta + 
           aero.C_Yp * p_bar + 
           aero.C_Yr * r_bar + 
           aero.C_Yda * delta_a + 
           aero.C_Ydr * delta_r)
    
    # Rolling moment coefficient
    C_l = (aero.C_lbeta * beta + 
           aero.C_lp * p_bar + 
           aero.C_lr * r_bar + 
           aero.C_lda * delta_a + 
           aero.C_ldr * delta_r)
    
    # Yawing moment coefficient
    C_n = (aero.C_nbeta * beta + 
           aero.C_np * p_bar + 
           aero.C_nr * r_bar + 
           aero.C_nda * delta_a + 
           aero.C_ndr * delta_r)
    
    return jnp.array([C_L, C_D, C_Y, C_l, C_m, C_n])


@el.map
def compute_aero_forces(
    aero_coefs: AeroCoefs,
    alpha: AngleOfAttack,
    beta: Sideslip,
    q_bar: DynamicPressure,
) -> AeroForce:
    """
    Convert aerodynamic coefficients to forces and moments in body frame.
    
    Section 3.5 of whitepaper:
    - Forces: Lift (L), Drag (D), Side force (Y)
    - Moments: Rolling (l), Pitching (m), Yawing (n)
    """
    config = BDXConfig.GLOBAL
    
    C_L, C_D, C_Y, C_l, C_m, C_n = aero_coefs
    
    S = config.wing_area
    b = config.wingspan
    c = config.mean_chord
    
    # Aerodynamic forces in wind frame
    L = q_bar * S * C_L  # Lift (perpendicular to velocity)
    D = q_bar * S * C_D  # Drag (opposite to velocity)
    Y = q_bar * S * C_Y  # Side force
    
    # Transform from wind frame to body frame
    # Wind frame: X (along velocity), Y (right), Z (down perpendicular to velocity)
    # Body frame: X (forward), Y (right), Z (down)
    ca = jnp.cos(alpha)
    sa = jnp.sin(alpha)
    cb = jnp.cos(beta)
    sb = jnp.sin(beta)
    
    # Simplified transformation (assuming small β for now)
    F_x = -D * ca + L * sa
    F_y = Y
    F_z = -D * sa - L * ca
    
    # Aerodynamic moments in body frame
    tau_x = q_bar * S * b * C_l  # Rolling moment
    tau_y = q_bar * S * c * C_m  # Pitching moment
    tau_z = q_bar * S * b * C_n  # Yawing moment
    
    return el.SpatialForce(
        linear=jnp.array([F_x, F_y, F_z]),
        torque=jnp.array([tau_x, tau_y, tau_z])
    )


@el.map
def apply_aero_forces(
    pos: el.WorldPos,
    aero_force: AeroForce,
    force: el.Force,
) -> el.Force:
    """
    Transform aerodynamic forces from body frame to world frame and apply.
    """
    # Rotate body-frame forces to world frame
    force_world = pos.angular() @ aero_force
    
    return force + force_world

