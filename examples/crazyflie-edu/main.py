#!/usr/bin/env python3
"""
Crazyflie Educational Simulation - Main Entry Point

Run with:
    elodin editor examples/crazyflie-edu/main.py

This simulation provides an educational environment for learning
quadcopter dynamics and control, inspired by UC Berkeley's ME136 course.

Labs:
    - SimLab1: System setup, motors, and sensor analysis
    - SimLab2: Powertrain identification (PWM → speed → force)
    - HWLab1: Hardware programming and communication
    - HWLab2: Hardware powertrain validation
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from config import CrazyflieConfig, create_default_config
from sim import CrazyflieDrone, MotorPwm, IsArmed, create_physics_system
from sensors import IMU, create_imu_system

# =============================================================================
# Simulation Time Component
# =============================================================================

SimTime = ty.Annotated[
    jax.Array,
    el.Component("sim_time", el.ComponentType.F64),
]


@dataclass
class SimClock(el.Archetype):
    """Simulation clock archetype."""

    sim_time: SimTime = field(default_factory=lambda: jnp.array(0.0))


# =============================================================================
# Control Components
# =============================================================================

# Motor commands from user code (PWM values 0-255)
MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "priority": 100,
            "element_names": "m1,m2,m3,m4",
        },
    ),
]


@dataclass
class Control(el.Archetype):
    """Control state archetype for user code interaction."""

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    is_armed: IsArmed = field(default_factory=lambda: jnp.array(1.0))  # Start armed for simplicity


# =============================================================================
# User Code Integration
# =============================================================================


@el.map
def update_sim_time(time: SimTime) -> SimTime:
    """Advance simulation time."""
    config = CrazyflieConfig.GLOBAL
    return time + config.dt


@el.map
def apply_motor_commands(cmd: MotorCommand, is_armed: IsArmed) -> MotorPwm:
    """
    Apply motor commands from user code to the physics simulation.

    Motor commands are only applied if the vehicle is armed.
    """
    # Only allow motor commands when armed (is_armed > 0.5 means armed)
    armed = is_armed > 0.5
    return jnp.where(armed, cmd, jnp.zeros(4))


# =============================================================================
# World Setup
# =============================================================================


def create_world() -> tuple[el.World, el.EntityId]:
    """Create the simulation world with a Crazyflie drone."""
    config = CrazyflieConfig.GLOBAL
    w = el.World()

    # Spawn the drone entity with all required components
    drone = w.spawn(
        [
            el.Body(
                world_pos=config.spatial_transform,
                inertia=config.spatial_inertia,
            ),
            CrazyflieDrone(),
            IMU(),
            Control(),
            SimClock(),
        ],
        name="crazyflie",
    )

    # Define the editor schematic (GUI layout)
    schematic = """
        theme mode="dark" scheme="default"

        tabs {
            hsplit name="Simulation" {
                viewport name="3D View" pos="crazyflie.world_pos + (0,0,0,0, 0.2, 0.2, 0.2)" look_at="crazyflie.world_pos" show_grid=#true active=#true
                vsplit share=0.35 {
                    graph "crazyflie.gyro" name="Gyroscope (rad/s)"
                    graph "crazyflie.accel" name="Accelerometer (g)"
                    graph "crazyflie.motor_pwm" name="Motor PWM"
                }
            }
            vsplit name="Motors" {
                graph "crazyflie.motor_rpm" name="Motor RPM"
                graph "crazyflie.thrust" name="Motor Thrust (N)"
                graph "crazyflie.motor_command" name="Motor Command"
            }
        }
        object_3d crazyflie.world_pos {
            glb path="crazyflie.glb" rotate="(0.0, 0.0, 0.0)" translate="(-0.01, 0.0, 0.0)" scale=0.7
        }

        // Crazyflie 2.1 dimensions: 92mm diagonal, arm_length=32.5mm, height~29mm
        // Motor position indicators (at arm_length diagonals)
        vector_arrow "(0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M1: FR" show_name=#true body_frame=#true {
            color yellow 50
        }
        vector_arrow "(0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M2: FL" show_name=#true body_frame=#true {
            color yellow 50
        }
        vector_arrow "(-0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M3: BL" show_name=#true body_frame=#true {
            color yellow 50
        }
        vector_arrow "(-0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M4: BR" show_name=#true body_frame=#true {
            color yellow 50
        }

        // Rotor disc visualization (45mm diameter = 0.0225m radius, thin cylinder)
        // Motor positions: arm_length=0.0325m at 45° diagonals, prop_height=0.012m
        // Quad-X CW/CCW pairing: M1+M3 are CW (diagonal), M2+M4 are CCW (diagonal)
        // M1: FR (+0.0325, -0.0325) - CW (cyan)
        object_3d "crazyflie.world_pos + (0,0,0,0, 0.0325, -0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color cyan 30
            }
        }
        // M2: FL (+0.0325, +0.0325) - CCW (magenta)
        object_3d "crazyflie.world_pos + (0,0,0,0, 0.0325, 0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color red 30
            }
        }
        // M3: BL (-0.0325, +0.0325) - CW (cyan) - diagonal from M1
        object_3d "crazyflie.world_pos + (0,0,0,0, -0.0325, 0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color cyan 30
            }
        }
        // M4: BR (-0.0325, -0.0325) - CCW (magenta) - diagonal from M2
        object_3d "crazyflie.world_pos + (0,0,0,0, -0.0325, -0.0325, 0.013)" body_frame=#true {
            cylinder radius=0.0225 height=0.002 {
                color red 30
            }
        }
    """

    w.schematic(schematic, "crazyflie-edu.kdl")

    return w, drone


# =============================================================================
# System Composition
# =============================================================================


def system() -> el.System:
    """Create the complete simulation system."""
    config = CrazyflieConfig.GLOBAL

    # Physics: motor dynamics, thrust, drag, gravity, ground constraint
    physics = create_physics_system()

    # Sensors: IMU with noise
    sensors = create_imu_system()

    # Control: Apply motor commands from user code
    control = apply_motor_commands

    # Time tracking
    clock = update_sim_time

    # Combine all systems
    return clock | control | physics | sensors


# =============================================================================
# Main Entry Point
# =============================================================================

# Create default configuration first (required before creating world)
config = create_default_config()

# Create world and system
world, drone_id = create_world()
sys = system()

# Run simulation
# When using `elodin editor`, this creates the visualization
# When using `python main.py run`, this runs headless
world.run(
    sys,
    sim_time_step=config.dt,
    run_time_step=1.0 / 60.0,
    max_ticks=config.total_sim_ticks,
)
