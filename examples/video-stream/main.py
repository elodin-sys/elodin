#!/usr/bin/env python3
"""
Video Streaming Example

Demonstrates streaming video from GStreamer into Elodin DB and
displaying it in the Elodin Editor's video stream tile.

A ball rolls around on a flat surface, pushed by rotating wind and
bouncing off walls at the viewport edges.

Usage:
    elodin editor examples/video-stream/main.py

The video stream tile is automatically created via the schematic.
"""

import typing
from dataclasses import field
from pathlib import Path

import elodin as el
import jax
from jax import numpy as jnp

# =============================================================================
# Constants
# =============================================================================

SIM_TIME_STEP = 1.0 / 120.0
BALL_RADIUS = 0.3
BOUNDARY = 4.0  # Wall distance from center
BOUNCINESS = 0.95  # Energy retained on wall bounce (higher = bouncier)
FRICTION = 0.4  # Viscous damping coefficient


# =============================================================================
# Custom Components
# =============================================================================

Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@el.dataclass
class WindData(el.Archetype):
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


# =============================================================================
# Systems - Velocity/Constraint (run before integrator)
# =============================================================================


# Wind rotation rate - completes one full rotation every N ticks
WIND_ROTATION_PERIOD = 360  # 3 seconds at 120Hz for full rotation
WIND_SPEED = 8.0  # Constant wind speed


@el.system
def sample_wind(
    tick: el.Query[el.SimulationTick],
    w: el.Query[Wind],
) -> el.Query[Wind]:
    """
    Wind direction rotates in a circle over time.
    Uses built-in SimulationTick for deterministic rotation.
    """
    # Convert tick to angle (2*pi radians per rotation period)
    angle = (tick[0] / WIND_ROTATION_PERIOD) * 2.0 * jnp.pi

    # Wind rotates in X-Y plane
    wind_vec = jnp.array([jnp.cos(angle) * WIND_SPEED, jnp.sin(angle) * WIND_SPEED, 0.0])
    return w.map(Wind, lambda _: wind_vec)


@el.map
def wall_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    """Reflect velocity when ball hits boundary walls."""
    pos = p.linear()
    vel = v.linear()
    ang = v.angular()

    # Check X boundaries and reflect X velocity
    vel_x = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[0] > BOUNDARY, vel[0] > 0),
            jnp.logical_and(pos[0] < -BOUNDARY, vel[0] < 0),
        ),
        lambda _: -vel[0] * BOUNCINESS,
        lambda _: vel[0],
        operand=None,
    )

    # Check Y boundaries and reflect Y velocity
    vel_y = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[1] > BOUNDARY, vel[1] > 0),
            jnp.logical_and(pos[1] < -BOUNDARY, vel[1] < 0),
        ),
        lambda _: -vel[1] * BOUNCINESS,
        lambda _: vel[1],
        operand=None,
    )

    # Keep Z velocity zero (ball stays on surface)
    new_vel = jnp.array([vel_x, vel_y, 0.0])

    return el.SpatialMotion(angular=ang, linear=new_vel)


@el.map
def rolling_motion(v: el.WorldVel) -> el.WorldVel:
    """
    Sync angular velocity with linear velocity for rolling without slipping.
    Formula: omega = (n x v) / R, where n is the surface normal (+Z)
    """
    vel = v.linear()

    # Surface normal is +Z
    n = jnp.array([0.0, 0.0, 1.0])

    # omega = (n x v) / R
    omega = jnp.cross(n, vel) / BALL_RADIUS

    return el.SpatialMotion(angular=omega, linear=vel)


# =============================================================================
# Systems - Effectors (forces passed to integrator)
# =============================================================================

# Direct wind force coefficient (how strongly wind pushes the ball)
WIND_FORCE_COEFFICIENT = 3.0


@el.map
def apply_wind(w: Wind, f: el.Force) -> el.Force:
    """Apply direct wind force - wind pushes the ball regardless of ball velocity."""
    # Direct force proportional to wind velocity
    wind_force = w * WIND_FORCE_COEFFICIENT
    return el.SpatialForce(linear=f.force() + wind_force)


@el.map
def friction(v: el.WorldVel, f: el.Force) -> el.Force:
    """Apply viscous friction (velocity-proportional damping)."""
    vel = v.linear()

    # Viscous damping force = -friction * velocity
    friction_force = -FRICTION * vel

    return el.SpatialForce(linear=f.force() + friction_force)


# =============================================================================
# World Setup
# =============================================================================

world = el.World()

# Create ball entity with initial velocity to start moving
ball = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, BALL_RADIUS])),
            world_vel=el.SpatialMotion(
                angular=jnp.array([0.0, 0.0, 0.0]),
                linear=jnp.array([3.0, 2.0, 0.0]),  # Initial velocity to start rolling
            ),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        WindData(),
    ],
    name="ball",
)

# Register the video streaming process via S10 recipe
stream_script = Path(__file__).parent / "stream-video.sh"
video_streamer = el.s10.PyRecipe.process(
    name="video-stream",
    cmd="bash",
    args=[str(stream_script)],
    cwd=str(Path(__file__).parent),
)
world.recipe(video_streamer)

# Define schematic with top-down camera view and video stream tile
world.schematic("""
    hsplit {
        tabs share=0.6 {
            viewport name=Viewport pos="(0,0,0,0, 0,0,12)" look_at="(0,0,0,0, 0,0,0)" show_grid=#true
        }
        vsplit share=0.4 {
            video_stream "test-video" name="Test Pattern"
            graph "ball.wind" name="Wind (m/s)"
        }
    }
    object_3d ball.world_pos {
        sphere radius=0.3 {
            color orange
        }
    }
    object_3d "(0,0,0,1, 0,0,0)" {
        plane width=10 depth=10 {
            color 32 128 32 200
        }
    }
    // Wind force visualization arrow at origin, slightly above ground
    vector_arrow "ball.wind" origin="(0,0,0,1, 0,0,0.5)" scale=0.3 name="Wind" show_name=#true {
        color cyan 200
    }
""")

print("Video Streaming Example - Rolling Ball")
print("======================================")
print()
print("A ball rolls around pushed by rotating wind, bouncing off walls.")
print("The video stream tile shows a GStreamer test pattern.")
print()

# =============================================================================
# System Composition
# =============================================================================

# Constraints & velocity tweaks (run before integrator)
constraints = sample_wind | wall_bounce | rolling_motion

# Force effectors (passed to integrator)
effectors = apply_wind | friction

# Compose with semi-implicit integrator for stable, game-like motion
sys = constraints | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)

# Run simulation - real-time simulation is required for this to work, otherwise the video frames will be out of sync with the simulation.
sim = world.run(sys, SIM_TIME_STEP, run_time_step=1.0 / 60.0, start_timestamp=0)
