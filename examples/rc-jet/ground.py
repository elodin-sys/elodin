"""
Ground Contact Model

Implements simple ground collision to prevent aircraft from penetrating ground.
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp


GROUND_LEVEL = 0.8  # m - ground plane (landing gear keeps CG this high when on ground)


@el.map
def ground_contact(
    pos: el.WorldPos,
    vel: el.WorldVel,
    force: el.Force,
    inertia: el.Inertia,
) -> el.Force:
    """
    Apply ground contact forces to prevent aircraft from penetrating ground.
    
    Model:
    - Hard floor at Z=0: apply strong upward force only when penetrating
    - Friction during ground contact
    - Allow liftoff when aerodynamic forces overcome weight
    """
    altitude = pos.linear()[2]
    vertical_vel = vel.linear()[2]
    
    # Check if we're below ground (penetrating)
    # Add small tolerance to allow clean liftoff
    LIFTOFF_TOLERANCE = 0.05  # m - allow this much "float" before applying ground forces
    penetrating = altitude < (GROUND_LEVEL - LIFTOFF_TOLERANCE)
    penetration_depth = jnp.clip((GROUND_LEVEL - LIFTOFF_TOLERANCE) - altitude, 0.0, 10.0)
    
    # Very stiff spring to prevent penetration
    k_ground = 100000.0  # N/m - very stiff ground
    normal_force_mag = k_ground * penetration_depth
    
    # Strong damping to prevent bouncing
    c_ground = 10000.0  # N·s/m
    damping_force = -c_ground * jnp.clip(vertical_vel, -10.0, 10.0) * jnp.where(penetrating, 1.0, 0.0)
    
    # Total upward force
    vertical_force = normal_force_mag + damping_force
    
    # Friction (only when on ground and penetrating)
    mu = 0.05  # Low friction coefficient (wheeled aircraft on paved runway)
    horiz_vel = vel.linear()[:2]  # X and Y components
    horiz_speed = jnp.linalg.norm(horiz_vel)
    
    # Friction magnitude proportional to normal force
    friction_mag = mu * normal_force_mag
    
    # Friction direction opposes horizontal motion
    friction_x = jax.lax.cond(
        horiz_speed > 0.01,
        lambda _: -friction_mag * horiz_vel[0] / horiz_speed,
        lambda _: 0.0,
        operand=None
    )
    friction_y = jax.lax.cond(
        horiz_speed > 0.01,
        lambda _: -friction_mag * horiz_vel[1] / horiz_speed,
        lambda _: 0.0,
        operand=None
    )
    
    # Only apply when penetrating ground
    ground_force = jax.lax.cond(
        penetrating,
        lambda _: el.SpatialForce(linear=jnp.array([
            friction_x,
            friction_y,
            vertical_force
        ])),
        lambda _: el.SpatialForce(),
        operand=None
    )
    
    return force + ground_force

