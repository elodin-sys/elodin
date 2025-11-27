"""
BDX Turbine Propulsion Model

Implements first-order spool dynamics and thrust generation for JetCat-class turbines.
Following Section 4 of BDX_Simulation_Whitepaper.md.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp

from config import BDXConfig

# Component type definitions

SpoolSpeed = ty.Annotated[
    jax.Array,
    el.Component(
        "spool_speed",
        el.ComponentType.F64,
        metadata={"priority": 60},
    ),
]

ThrottleCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "throttle_command",
        el.ComponentType.F64,
        metadata={"priority": 61},
    ),
]

Thrust = ty.Annotated[
    jax.Array,
    el.Component(
        "thrust",
        el.ComponentType.F64,
        metadata={"priority": 59},
    ),
]


@el.map
def spool_dynamics(
    throttle_cmd: ThrottleCommand,
    spool_speed: SpoolSpeed,
) -> SpoolSpeed:
    """
    First-order spool speed dynamics.
    
    From Section 4.1 of whitepaper:
    dn/dt = (n_cmd - n) / τ_spool
    
    where n is normalized spool speed (0-1) and n_cmd is throttle command.
    """
    config = BDXConfig.GLOBAL
    dt = config.dt
    tau = config.propulsion.spool_tau
    
    # Clamp throttle command to valid range
    n_cmd = jnp.clip(throttle_cmd, 0.0, 1.0)
    
    # First-order dynamics
    n_new = spool_speed + (n_cmd - spool_speed) * dt / tau
    
    # Ensure spool speed stays in valid range
    n_new = jnp.clip(n_new, 0.0, 1.0)
    
    return n_new


@el.map
def compute_thrust(
    spool_speed: SpoolSpeed,
    pos: el.WorldPos,
) -> Thrust:
    """
    Compute thrust from spool speed with atmospheric corrections.
    
    From Section 4.2-4.3 of whitepaper:
    T(n) = T_max * (a₁·n + a₂·n²) * (ρ/ρ₀) * f(M)
    
    For simplicity, we use density correction but ignore Mach effects at low speeds.
    """
    config = BDXConfig.GLOBAL
    
    # Atmospheric density at current altitude
    altitude = pos.linear()[2]
    T_atm = 288.15 - 0.0065 * altitude
    T_atm = jnp.clip(T_atm, 216.65, 288.15)
    p = 101325.0 * (T_atm / 288.15) ** 5.2561
    rho = p / (287.05 * T_atm)
    rho_0 = 1.225  # Sea level density
    
    # Quadratic thrust map
    T_max = config.propulsion.max_thrust
    a1 = config.propulsion.thrust_a1
    a2 = config.propulsion.thrust_a2
    
    # Base thrust
    T_base = T_max * (a1 * spool_speed + a2 * spool_speed**2)
    
    # Density correction (thrust lapse with altitude)
    T = T_base * (rho / rho_0)
    
    return T


@el.map
def apply_thrust(
    thrust: Thrust,
    pos: el.WorldPos,
    force: el.Force,
) -> el.Force:
    """
    Apply thrust force along body X-axis (forward).
    
    Thrust vector is aligned with aircraft centerline.
    """
    # Thrust vector in body frame (along +X axis)
    thrust_body = jnp.array([thrust, 0.0, 0.0])
    
    # Transform to world frame
    thrust_world = pos.angular() @ thrust_body
    
    # Apply as spatial force
    return force + el.SpatialForce(linear=thrust_world)

