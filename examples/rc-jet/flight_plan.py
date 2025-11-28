"""
BDX Flight Plan - Altitude Hold Controller

Simple cruise flight with PD altitude controller.
Maintains level flight at target altitude.
"""

import elodin as el
import jax.numpy as jnp

from actuators import ControlCommands


@el.system
def flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    pos: el.Query[el.WorldPos],
    vel: el.Query[el.WorldVel],
    commands: el.Query[ControlCommands],
) -> el.Query[ControlCommands]:
    """
    Cruise flight plan with altitude hold.
    
    Uses PD controller to maintain target altitude:
    - Proportional gain: responds to altitude error
    - Derivative gain: damps vertical velocity to prevent oscillation
    """
    current_pos = pos[0]
    current_vel = vel[0]
    altitude = current_pos.linear()[2]
    vertical_vel = current_vel.linear()[2]
    
    # Target altitude
    target_alt = 50.0
    alt_error = target_alt - altitude
    
    # PD altitude controller
    K_p = 0.05  # Proportional gain (rad/m)
    K_d = 1.5   # Derivative gain (rad/(m/s))
    elevator = -K_p * alt_error - K_d * vertical_vel
    elevator = jnp.clip(elevator, jnp.deg2rad(-8.0), jnp.deg2rad(8.0))
    
    # Throttle for level cruise at 70 m/s
    # Drag ≈ 56N, Thrust = 0.30 * 200N = 60N
    throttle = 0.30
    
    # Command: [elevator, aileron, rudder, throttle]
    cmd = jnp.array([elevator, 0.0, 0.0, throttle])
    
    return commands.map(ControlCommands, lambda _: cmd)

