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

Keyboard Controls:
    Q           - Toggle armed state
    Left Shift  - Blue button (dead man switch, hold to enable motors)
    E/R/T       - Yellow/Green/Red buttons
    WASD        - Throttle/Yaw (for future joystick control)
    Arrows      - Pitch/Roll (for future joystick control)
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

from config import CrazyflieConfig, create_default_config
from sim import CrazyflieDrone, create_physics_system, thrust_visualization
from sensors import IMU, create_imu_system
from crazyflie_api import CrazyflieState
import user_code

# Try to import keyboard controller (optional dependency)
try:
    from keyboard_controller import KeyboardController
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: keyboard_controller not available")

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
            "external_control": "true",  # Allow writes from post_step callback
        },
    ),
]

# Button components (controlled via keyboard)
ButtonBlue = ty.Annotated[
    jax.Array,
    el.Component("button_blue", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonYellow = ty.Annotated[
    jax.Array,
    el.Component("button_yellow", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonGreen = ty.Annotated[
    jax.Array,
    el.Component("button_green", el.ComponentType.F64, metadata={"external_control": "true"}),
]
ButtonRed = ty.Annotated[
    jax.Array,
    el.Component("button_red", el.ComponentType.F64, metadata={"external_control": "true"}),
]

# Re-define IsArmed with external_control for the Control archetype
IsArmedControl = ty.Annotated[
    jax.Array,
    el.Component("is_armed_control", el.ComponentType.F64, metadata={"external_control": "true"}),
]


@dataclass
class Control(el.Archetype):
    """Control state archetype for user code interaction."""

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    is_armed_control: IsArmedControl = field(default_factory=lambda: jnp.array(0.0))  # Start disarmed
    button_blue: ButtonBlue = field(default_factory=lambda: jnp.array(0.0))
    button_yellow: ButtonYellow = field(default_factory=lambda: jnp.array(0.0))
    button_green: ButtonGreen = field(default_factory=lambda: jnp.array(0.0))
    button_red: ButtonRed = field(default_factory=lambda: jnp.array(0.0))


# =============================================================================
# User Code Integration
# =============================================================================


@el.map
def update_sim_time(time: SimTime) -> SimTime:
    """Advance simulation time."""
    config = CrazyflieConfig.GLOBAL
    return time + config.dt


# NOTE: Motor commands are now written directly from post_step callback
# This avoids conflicts between JAX system outputs and external writes


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
            vsplit name="Controls" {
                graph "crazyflie.is_armed_control" name="Armed State (Q to toggle)"
                graph "crazyflie.button_blue" name="Blue Button (Shift)"
                graph "crazyflie.button_yellow" name="Yellow Button (E)"
                graph "crazyflie.button_green" name="Green Button (R)"
                graph "crazyflie.button_red" name="Red Button (T)"
            }
        }
        object_3d crazyflie.world_pos {
            glb path="crazyflie.glb" rotate="(0.0, 0.0, 0.0)" translate="(-0.01, 0.0, 0.0)" scale=0.7
        }

        // Crazyflie 2.1 dimensions: 92mm diagonal, arm_length=32.5mm, height~29mm
        // Motor position indicators (at arm_length diagonals)
        vector_arrow "(0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M1: FR" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M2: FL" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(-0.707, 0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M3: BL" show_name=#true body_frame=#true {
            color yellow 10
        }
        vector_arrow "(-0.707, -0.707, 0)" origin="crazyflie.world_pos" scale=0.046 name="M4: BR" show_name=#true body_frame=#true {
            color yellow 10
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

        // Thrust visualization - arrows pointing DOWN from each rotor
        // M1: FR - CW (cyan)
        vector_arrow "crazyflie.thrust_viz_m1" origin="crazyflie.world_pos + (0,0,0,0, 0.0325, -0.0325, 0.013)" body_frame=#true {
            color cyan 20
        }
        // M2: FL - CCW (magenta/red)
        vector_arrow "crazyflie.thrust_viz_m2" origin="crazyflie.world_pos + (0,0,0,0, 0.0325, 0.0325, 0.013)" body_frame=#true {
            color red 20
        }
        // M3: BL - CW (cyan)
        vector_arrow "crazyflie.thrust_viz_m3" origin="crazyflie.world_pos + (0,0,0,0, -0.0325, 0.0325, 0.013)" body_frame=#true {
            color cyan 20
        }
        // M4: BR - CCW (magenta/red)
        vector_arrow "crazyflie.thrust_viz_m4" origin="crazyflie.world_pos + (0,0,0,0, -0.0325, -0.0325, 0.013)" body_frame=#true {
            color red 20
        }
    """

    w.schematic(schematic, "crazyflie-edu.kdl")

    return w, drone


# =============================================================================
# System Composition
# =============================================================================


def system() -> el.System:
    """Create the complete simulation system."""
    # Physics: motor dynamics, thrust, drag, gravity, ground constraint
    physics = create_physics_system()

    # Sensors: IMU with noise
    sensors = create_imu_system()

    # Time tracking
    clock = update_sim_time

    # Thrust visualization (compute visualization vectors after physics)
    visualization = thrust_visualization

    # NOTE: Motor commands are written directly from post_step callback
    # This avoids conflicts between JAX system outputs and external writes
    # Combine all systems
    return clock | physics | sensors | visualization


# =============================================================================
# Post-Step Callback (User Code Integration)
# =============================================================================

# Global state for keyboard controller
_keyboard_controller = None
_last_print_time = [0.0]


def user_code_post_step(tick: int, ctx: el.PostStepContext):
    """
    Post-step callback that integrates user code with the simulation.
    
    This is called after each physics step and:
    1. Reads keyboard input
    2. Updates button/arm state in the simulation
    3. Reads sensor data
    4. Calls user_code.main_loop() with the current state
    5. Writes motor commands back to the simulation
    """
    global _keyboard_controller
    
    config = CrazyflieConfig.GLOBAL
    sim_time = tick * config.dt
    
    # Initialize keyboard controller on first tick
    if _keyboard_controller is None:
        if KEYBOARD_AVAILABLE:
            _keyboard_controller = KeyboardController()
            _keyboard_controller.start()
        else:
            # Use a placeholder that always returns defaults
            class DummyController:
                def get_state(self):
                    from keyboard_controller import ControllerState
                    return ControllerState(is_armed=True, button_blue=True)  # Auto-arm for testing
            _keyboard_controller = DummyController()
            
        print("\n" + "=" * 60)
        print("  SIMULATION STARTED!")
        print("=" * 60)
        print("  Keyboard Controls:")
        print("    Q           - Toggle armed (currently DISARMED)")
        print("    Left Shift  - Blue button (hold to enable motors)")
        print("    E/R/T       - Yellow/Green/Red buttons")
        print("=" * 60 + "\n")
    
    # Read keyboard state
    if _keyboard_controller is not None:
        kb_state = _keyboard_controller.get_state()
        is_armed = kb_state.is_armed
        button_blue = kb_state.button_blue
        button_yellow = kb_state.button_yellow
        button_green = kb_state.button_green
        button_red = kb_state.button_red
    else:
        # No keyboard - default values
        is_armed = False
        button_blue = False
        button_yellow = False
        button_green = False
        button_red = False
    
    # Write control inputs to simulation for graphing/logging
    # (These are purely for visualization - arming logic is in user_code)
    ctx.write_component("crazyflie.is_armed_control", np.array([1.0 if is_armed else 0.0]))
    ctx.write_component("crazyflie.button_blue", np.array([1.0 if button_blue else 0.0]))
    ctx.write_component("crazyflie.button_yellow", np.array([1.0 if button_yellow else 0.0]))
    ctx.write_component("crazyflie.button_green", np.array([1.0 if button_green else 0.0]))
    ctx.write_component("crazyflie.button_red", np.array([1.0 if button_red else 0.0]))
    
    # Read sensor data from simulation
    try:
        gyro = np.array(ctx.read_component("crazyflie.gyro"))
        accel = np.array(ctx.read_component("crazyflie.accel"))
        motor_cmd = np.array(ctx.read_component("crazyflie.motor_command"))
    except Exception:
        # Components might not be ready on first tick
        gyro = np.zeros(3)
        accel = np.array([0.0, 0.0, 1.0])  # 1g on z-axis
        motor_cmd = np.zeros(4)
    
    # Create state object for user code
    state = CrazyflieState(
        gyro=gyro,
        accel=accel,
        is_armed=is_armed,
        button_blue=button_blue,
        button_yellow=button_yellow,
        button_green=button_green,
        button_red=button_red,
        motor_command=motor_cmd,
        time=sim_time,
        dt=config.dt,
    )
    
    # Call user code
    user_code.main_loop(state)
    
    # Write motor commands directly to motor_pwm for physics
    motor_cmds = np.array(state.motor_command, dtype=np.float64)
    
    # Write PWM values (user_code sets motor_command, we pass it to physics)
    ctx.write_component("crazyflie.motor_pwm", motor_cmds)
    
    # Also write to motor_command for logging/graphing purposes
    ctx.write_component("crazyflie.motor_command", motor_cmds)
    
    
    # Print status periodically
    if sim_time - _last_print_time[0] >= 1.0:
        armed_str = "ARMED" if is_armed else "DISARMED"
        motors = state.motor_command
        
        # Read back status
        try:
            thrust = np.array(ctx.read_component("crazyflie.thrust"))
            
            print(f"[{sim_time:6.1f}s] {armed_str} | Blue:{int(button_blue)} | "
                  f"PWM:[{motors[0]:.0f},{motors[1]:.0f},{motors[2]:.0f},{motors[3]:.0f}] | "
                  f"Thrust:[{thrust[0]:.4f},{thrust[1]:.4f},{thrust[2]:.4f},{thrust[3]:.4f}]")
        except Exception as e:
            print(f"[{sim_time:6.1f}s] {armed_str} | Blue:{int(button_blue)} | Read error: {e}")
        
        _last_print_time[0] = sim_time


# =============================================================================
# Main Entry Point
# =============================================================================

# Create default configuration first (required before creating world)
config = create_default_config()

# Create world and system
world, drone_id = create_world()
sys = system()

# Run simulation with user code callback
# When using `elodin editor`, this creates the visualization
# When using `python main.py run`, this runs headless
world.run(
    sys,
    sim_time_step=config.dt,
    run_time_step=1.0 / 60.0,
    max_ticks=config.total_sim_ticks,
    post_step=user_code_post_step,
)
