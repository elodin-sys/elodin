"""
BDX Flight Control System

Implements flight plans and optional stability augmentation.
Following Section 6 of BDX_Simulation_Whitepaper.md (optional).
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

from actuators import ControlCommands


@el.system
def simple_flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    commands: el.Query[ControlCommands],
) -> el.Query[ControlCommands]:
    """
    Simple time-based flight plan for demonstration.
    
    Flight phases:
    - 0-5s: Steady level flight, throttle at 70%
    - 5-10s: Pitch up maneuver (elevator deflection)
    - 10-15s: Roll maneuver (aileron deflection)
    - 15-20s: Coordinated turn (aileron + rudder)
    - 20-25s: Return to level flight
    - 25+s: Steady cruise
    """
    time = tick[0] * dt[0]
    
    # Default commands: [elevator, aileron, rudder, throttle]
    # All surface deflections in radians, throttle 0-1
    
    def get_commands(t):
        # Phase 1: Level flight (0-5s)
        cmd1 = jnp.array([0.0, 0.0, 0.0, 0.7])
        
        # Phase 2: Pitch up maneuver (5-10s)
        elevator_cmd = jnp.where(
            (t >= 5.0) & (t < 10.0),
            jnp.deg2rad(-5.0),  # Negative for nose up
            0.0
        )
        cmd2 = jnp.array([elevator_cmd, 0.0, 0.0, 0.72])
        
        # Phase 3: Roll maneuver (10-15s)
        aileron_cmd = jnp.where(
            (t >= 10.0) & (t < 15.0),
            jnp.deg2rad(10.0),  # Positive for right roll
            0.0
        )
        cmd3 = jnp.array([0.0, aileron_cmd, 0.0, 0.7])
        
        # Phase 4: Coordinated turn (15-20s)
        aileron_turn = jnp.where(
            (t >= 15.0) & (t < 20.0),
            jnp.deg2rad(8.0),
            0.0
        )
        rudder_turn = jnp.where(
            (t >= 15.0) & (t < 20.0),
            jnp.deg2rad(3.0),
            0.0
        )
        elevator_turn = jnp.where(
            (t >= 15.0) & (t < 20.0),
            jnp.deg2rad(-2.0),  # Slight back pressure in turn
            0.0
        )
        cmd4 = jnp.array([elevator_turn, aileron_turn, rudder_turn, 0.72])
        
        # Phase 5: Return to level (20-25s)
        aileron_level = jnp.where(
            (t >= 20.0) & (t < 25.0),
            jnp.deg2rad(-5.0),  # Counter-roll
            0.0
        )
        rudder_level = jnp.where(
            (t >= 20.0) & (t < 25.0),
            jnp.deg2rad(-2.0),
            0.0
        )
        cmd5 = jnp.array([0.0, aileron_level, rudder_level, 0.7])
        
        # Select command based on time
        cmd = jax.lax.cond(
            t < 5.0,
            lambda _: cmd1,
            lambda _: jax.lax.cond(
                t < 10.0,
                lambda _: cmd2,
                lambda _: jax.lax.cond(
                    t < 15.0,
                    lambda _: cmd3,
                    lambda _: jax.lax.cond(
                        t < 20.0,
                        lambda _: cmd4,
                        lambda _: jax.lax.cond(
                            t < 25.0,
                            lambda _: cmd5,
                            lambda _: cmd1,  # Back to steady cruise
                            operand=None
                        ),
                        operand=None
                    ),
                    operand=None
                ),
                operand=None
            ),
            operand=None
        )
        
        return cmd
    
    new_cmd = get_commands(time)
    
    return commands.map(ControlCommands, lambda _: new_cmd)


@el.system
def altitude_hold_flight_plan(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    pos: el.Query[el.WorldPos],
    vel: el.Query[el.WorldVel],
    commands: el.Query[ControlCommands],
) -> el.Query[ControlCommands]:
    """
    Simple altitude hold autopilot (optional, more advanced).
    
    Maintains target altitude using proportional-derivative control on elevator.
    Throttle adjusts to maintain airspeed.
    """
    time = tick[0] * dt[0]
    
    # Target altitude (m)
    target_altitude = 100.0
    
    # Target airspeed (m/s)
    target_speed = 40.0
    
    # Get current state
    current_pos = pos[0]
    current_vel = vel[0]
    current_altitude = current_pos.linear()[2]
    current_speed = jnp.linalg.norm(current_vel.linear())
    
    # Altitude error
    alt_error = target_altitude - current_altitude
    
    # Vertical velocity (climb rate)
    vertical_velocity = current_vel.linear()[2]
    
    # PD controller for elevator
    # P gain: more elevator deflection for altitude error
    # D gain: dampen based on vertical velocity
    K_p_alt = 0.02  # rad per meter error
    K_d_alt = 0.1   # rad per m/s vertical velocity
    
    elevator_cmd = K_p_alt * alt_error - K_d_alt * vertical_velocity
    elevator_cmd = jnp.clip(elevator_cmd, jnp.deg2rad(-15.0), jnp.deg2rad(15.0))
    
    # Speed error
    speed_error = target_speed - current_speed
    
    # P controller for throttle
    K_p_speed = 0.02  # throttle per m/s error
    throttle_base = 0.7
    throttle_cmd = throttle_base + K_p_speed * speed_error
    throttle_cmd = jnp.clip(throttle_cmd, 0.2, 1.0)
    
    # Lateral control: simple bank angle hold at 0° (wings level)
    # For a more advanced autopilot, this would use roll angle feedback
    aileron_cmd = 0.0
    rudder_cmd = 0.0
    
    new_cmd = jnp.array([elevator_cmd, aileron_cmd, rudder_cmd, throttle_cmd])
    
    return commands.map(ControlCommands, lambda _: new_cmd)


@el.system
def manual_control(
    commands: el.Query[ControlCommands],
) -> el.Query[ControlCommands]:
    """
    Manual control mode - maintains current commands.
    
    This is a pass-through for external control via Impeller2.
    Commands can be set externally and will be maintained.
    """
    return commands.map(ControlCommands, lambda cmd: cmd)

