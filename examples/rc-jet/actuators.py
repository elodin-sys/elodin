"""
BDX Control Surface Actuator Model

Implements rate-limited first-order servo dynamics for elevator, aileron, and rudder.
Following Section 5 of BDX_Simulation_Whitepaper.md.
"""

import typing as ty
from dataclasses import field

import elodin as el
import jax
import jax.numpy as jnp

from config import BDXConfig

# Component type definitions

ControlSurfaces = ty.Annotated[
    jax.Array,
    el.Component(
        "control_surfaces",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "elevator,aileron,rudder", "priority": 70},
    ),
]

ControlCommands = ty.Annotated[
    jax.Array,
    el.Component(
        "control_commands",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "elevator,aileron,rudder,throttle", "priority": 71},
    ),
]


def actuator_dynamics_single(
    delta: float,
    delta_cmd: float,
    tau: float,
    delta_dot_max: float,
    delta_max: float,
    dt: float,
) -> float:
    """
    Rate-limited first-order actuator dynamics for a single control surface.
    
    From Section 5.1 of whitepaper:
    1. First-order response: delta_dot = (delta_cmd - delta) / tau
    2. Rate limiting: clip to ±delta_dot_max
    3. Integration: delta_new = delta + delta_dot * dt
    4. Position limiting: clip to ±delta_max
    """
    # First-order response
    delta_error = delta_cmd - delta
    delta_dot = delta_error / tau
    
    # Rate limiting
    delta_dot = jnp.clip(delta_dot, -delta_dot_max, delta_dot_max)
    
    # Integration
    delta_new = delta + delta_dot * dt
    
    # Position limiting
    delta_new = jnp.clip(delta_new, -delta_max, delta_max)
    
    return delta_new


@el.map
def actuator_dynamics(
    commands: ControlCommands,
    surfaces: ControlSurfaces,
) -> ControlSurfaces:
    """
    Update all control surface positions with servo dynamics.
    
    Surfaces: [elevator, aileron, rudder] in radians
    Commands: [elevator_cmd, aileron_cmd, rudder_cmd, throttle_cmd]
    """
    config = BDXConfig.GLOBAL
    dt = config.dt
    tau = config.actuators.servo_tau
    
    # Extract current positions
    delta_e, delta_a, delta_r = surfaces
    
    # Extract commands (first 3 are surface commands, 4th is throttle)
    delta_e_cmd, delta_a_cmd, delta_r_cmd, _ = commands
    
    # Convert max deflections from degrees to radians
    max_def_rad = jnp.deg2rad(config.actuators.max_deflection_deg)
    max_rudder_rad = jnp.deg2rad(config.actuators.max_rudder_deflection_deg)
    max_rate_rad = jnp.deg2rad(config.actuators.max_rate_deg_s)
    max_rudder_rate_rad = jnp.deg2rad(config.actuators.max_rudder_rate_deg_s)
    
    # Update elevator
    delta_e_new = actuator_dynamics_single(
        delta_e, delta_e_cmd, tau, max_rate_rad, max_def_rad, dt
    )
    
    # Update aileron
    delta_a_new = actuator_dynamics_single(
        delta_a, delta_a_cmd, tau, max_rate_rad, max_def_rad, dt
    )
    
    # Update rudder (different limits)
    delta_r_new = actuator_dynamics_single(
        delta_r, delta_r_cmd, tau, max_rudder_rate_rad, max_rudder_rad, dt
    )
    
    return jnp.array([delta_e_new, delta_a_new, delta_r_new])

