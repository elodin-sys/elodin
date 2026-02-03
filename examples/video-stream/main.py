#!/usr/bin/env python3
"""
Video Streaming Example

Demonstrates streaming video from GStreamer into Elodin DB and
displaying it in the Elodin Editor's video stream tile.

A ball rolls around on a flat surface, pushed by random wind and
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
from jax import random
from jax.numpy import linalg as la

# =============================================================================
# Constants
# =============================================================================

SIM_TIME_STEP = 1.0 / 120.0
BALL_RADIUS = 0.3
BOUNDARY = 4.0  # Wall distance from center
BOUNCINESS = 0.8  # Energy retained on wall bounce
WIND_STRENGTH = 1.5  # Scale factor for random wind
FRICTION = 0.1  # Viscous damping coefficient

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
    seed: el.Seed = field(default_factory=lambda: jnp.int64(0))
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))


# =============================================================================
# Systems - Velocity/Constraint (run before integrator)
# =============================================================================


@el.map
def sample_wind(s: el.Seed, _w: Wind) -> Wind:
    """Generate random wind in X-Y plane."""
    wind = random.normal(random.key(s), shape=(3,)) * WIND_STRENGTH
    # Zero out Z component - wind only blows horizontally
    return wind.at[2].set(0.0)


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


def calculate_drag(Cd: float, rho: float, V: float, A: float) -> float:
    """Calculate drag force magnitude."""
    return 0.5 * Cd * rho * V**2 * A


@el.map
def apply_wind(w: Wind, v: el.WorldVel, f: el.Force) -> el.Force:
    """Apply wind force using drag physics."""
    # Relative velocity between wind and ball
    rel_vel = w - v.linear()
    rel_speed = la.norm(rel_vel)

    # Drag parameters
    drag_coefficient = 0.5
    air_density = 1.225
    ball_surface_area = jnp.pi * BALL_RADIUS**2

    # Calculate drag force magnitude
    drag_mag = calculate_drag(drag_coefficient, air_density, rel_speed, ball_surface_area)

    # Direction of drag force (in direction of relative wind)
    # Avoid division by zero
    drag_dir = jax.lax.cond(
        rel_speed > 1e-6,
        lambda _: rel_vel / rel_speed,
        lambda _: jnp.zeros(3),
        operand=None,
    )

    drag_force = drag_mag * drag_dir

    return el.SpatialForce(linear=f.force() + drag_force)


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
                linear=jnp.array([1.0, 0.5, 0.0]),  # Initial velocity to start rolling
            ),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        WindData(seed=jnp.int64(42)),
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
        tabs share=0.5 {
            viewport name=Viewport pos="(0,0,0,0, 0,0,12)" look_at="(0,0,0,0, 0,0,0)" show_grid=#true
        }
        tabs share=0.5 {
            video_stream "test-video" name="Test Pattern"
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
""")

print("Video Streaming Example - Rolling Ball")
print("======================================")
print()
print("A ball rolls around pushed by random wind, bouncing off walls.")
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

# Run simulation
sim = world.run(sys, SIM_TIME_STEP, run_time_step=1.0 / 60.0)

