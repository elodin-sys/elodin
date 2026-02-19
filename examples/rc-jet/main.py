#!/usr/bin/env python3
"""
BDX RC Jet Turbine Simulation

Main entry point for the Elite Aerosports BDX simulation.
Implements a 6-DOF fixed-wing jet aircraft with aerodynamics,
turbine propulsion, and control surface dynamics.

Usage:
    elodin editor main.py       # Run with 3D visualization

The RC controller starts automatically with the simulation.
WASD / Arrow keys for keyboard control.
"""

from dataclasses import field
from pathlib import Path

import elodin as el
import jax.numpy as jnp

from config import BDXConfig
from sim import BDXJet, system


@el.dataclass
class StaticMarker(el.Archetype):
    """A static visual marker with no physics (no inertia, force, velocity)."""

    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


# Create configuration
config = BDXConfig()
config.set_as_global()


def setup_world(config: BDXConfig) -> tuple[el.World, el.EntityId, el.EntityId]:
    """
    Create and configure the simulation world with a BDX jet and target drone.

    Returns:
        tuple: (world, jet_entity_id, target_entity_id)
    """
    world = el.World()

    # Calculate initial position and velocity
    initial_pos = jnp.array([0.0, 0.0, config.initial_altitude])

    # Initial velocity: forward flight at cruise speed
    # Transform body velocity to world frame using initial attitude
    v_body = config.initial_velocity_body
    v_world = config.initial_attitude @ v_body

    # Spawn the BDX jet
    jet = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=config.initial_attitude,
                    linear=initial_pos,
                ),
                world_vel=el.SpatialMotion(
                    linear=v_world,
                    angular=jnp.zeros(3),
                ),
                inertia=config.spatial_inertia,
            ),
            BDXJet(),
        ],
        name="bdx",
    )

    # Spawn target drone (static visual marker) - positioned along initial flight path
    # Jet starts at [0,0,50] with heading 35° and speed 70 m/s
    # World velocity ≈ [57.3, 40.2, 0] m/s, so target at ~6 seconds ahead
    # Using StaticMarker (no Inertia/Force) makes it immune to physics systems
    target_position = jnp.array([350.0, 245.0, 55.0])
    target = world.spawn(
        StaticMarker(
            world_pos=el.SpatialTransform(
                angular=el.Quaternion.identity(),
                linear=target_position,
            ),
        ),
        name="target",
    )

    # Create schematic for visualization
    world.schematic(
        """
        tabs {
            hsplit name="Main View" {
                viewport name=Viewport pos="bdx.world_pos.translate_world(-8.0,-8.0,4.0)" look_at="bdx.world_pos" show_grid=#true active=#true
                vsplit share=0.4 {
                    vsplit {
                        graph "bdx.alpha" name="Angle of Attack (rad)"
                        graph "bdx.thrust" name="Thrust (N)"
                        viewport name=TGTViewport pos="target.world_pos.translate_world(1,1,0.2)" look_at="bdx.world_pos" show_grid=#true
                        viewport name=FPVViewport pos="bdx.world_pos.rotate_z(-90).translate_y(-2.0)" show_grid=#true
                    }
                }
            }
            vsplit name="Flight Data" {
                hsplit {
                    graph "bdx.velocity_body" name="Body Velocity (m/s)"
                    graph "bdx.world_pos.q0, bdx.world_pos.q1, bdx.world_pos.q2, bdx.world_pos.q3" name="Attitude (quat)"
                }
                hsplit {
                    graph "bdx.control_surfaces" name="Control Surfaces (rad)"
                    graph "bdx.dynamic_pressure, bdx.mach" name="Dynamic Pressure / Mach"
                }
            }
            vsplit name="Propulsion" {
                graph "bdx.thrust" name="Thrust (N)"
                graph "bdx.spool_speed" name="Spool Speed (normalized)"
                graph "bdx.throttle_command" name="Throttle Command"
            }
            vsplit name="Control Input" {
                graph "bdx.control_commands" name="Control Commands (rad/normalized)"
                graph "bdx.control_surfaces" name="Control Surfaces (rad)"
            }
            vsplit name="Aerodynamics" {
                hsplit {
                    graph "bdx.alpha" name="Angle of Attack (rad)"
                    graph "bdx.beta" name="Sideslip (rad)"
                }
                hsplit {
                    graph "bdx.aero_coefs.CL, bdx.aero_coefs.CD" name="CL, CD"
                    graph "bdx.aero_coefs.Cm" name="Pitch Moment Cm"
                }
            }
            hsplit name="Navigation" {
                viewport name="Top-Down View" pos="bdx.world_pos.translate_world(0.0, 0.0, 150.0)" look_at="bdx.world_pos" fov=60.0 show_grid=#true
                query_plot name="Ground Track (XY)" query="SELECT bdx_world_pos.bdx_world_pos[5], bdx_world_pos.bdx_world_pos[6] FROM bdx_world_pos" type="sql" mode="xy" x_label="X Position (m)" y_label="Y Position (m)" auto_refresh=#true refresh_interval=500 {
                    color cyan
                }
            }
        }

        object_3d bdx.world_pos {
            glb path="f22.glb" scale=0.01 translate="(0.0, 0.0, 0.0)" rotate="(0.0, 0.0, 0.0)"
            icon builtin="flight_takeoff" {
                visibility_range min=500.0
                color 76 175 80
            }
        }
        
        object_3d target.world_pos {
            glb path="edu-450-v2-drone.glb"
            icon builtin="adjust" {
                visibility_range min=500.0
                color 244 67 54
            }
        }
        
        vector_arrow "(1, 0, 0)" origin="bdx.world_pos" scale=1.0 name="Forward (X)" show_name=#true body_frame=#true {
           color red 150
        }
        vector_arrow "(0, 1, 0)" origin="bdx.world_pos" scale=1.0 name="Left (Y)" show_name=#true body_frame=#true {
           color green 150
        }
        vector_arrow "(0, 0, 1)" origin="bdx.world_pos" scale=1.0 name="Up (Z)" show_name=#true body_frame=#true {
           color blue 150
        }
        
        line_3d bdx.world_pos line_width=3.0 perspective=#false {
            color yolk
        }
        """,
        "bdx.kdl",
    )

    return world, jet, target


# Setup world
world, jet, target = setup_world(config)

# Build system
sim_system = system()

print("BDX RC Jet Simulation")
print("=====================")
print(f"Initial altitude: {config.initial_altitude:.1f} m")
print(f"Initial speed: {config.initial_speed:.1f} m/s")
print(f"Initial heading: {config.initial_yaw_deg:.1f}° (0°=East, 90°=North)")
print(f"Mass: {config.mass:.1f} kg")
print(f"Max thrust: {config.propulsion.max_thrust:.1f} N")
print("Target position: (350.0, 245.0, 55.0) m - along flight path at ~6s")
print(f"Simulation time: {config.simulation_time:.1f} s")
print(f"Time step: {config.dt:.6f} s ({1 / config.dt:.0f} Hz)")
print(f"Total ticks: {config.total_ticks}")
print()

# Register the RC controller to run alongside the simulation
controller_path = Path(__file__).parent / "controller"
controller = el.s10.PyRecipe.cargo(name="controller", path=str(controller_path))
world.recipe(controller)

# Run simulation in real-time mode for responsive RC control
world.run(
    sim_system,
    sim_time_step=config.dt,
    run_time_step=config.dt,
    max_ticks=config.total_ticks,
)
