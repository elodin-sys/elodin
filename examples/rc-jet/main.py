#!/usr/bin/env python3
"""
BDX RC Jet Turbine Simulation

Main entry point for the Elite Aerosports BDX simulation.
Implements a 6-DOF fixed-wing jet aircraft with aerodynamics,
turbine propulsion, and control surface dynamics.

Usage:
    elodin editor main.py       # Run with 3D visualization

For RC controller input, run in a separate terminal:
    cargo run -p rc-jet-controller

WASD / Arrow keys for keyboard control.
"""

import elodin as el
import jax.numpy as jnp

from config import BDXConfig
from sim import BDXJet, system

# Create configuration
config = BDXConfig()
config.set_as_global()


def setup_world(config: BDXConfig) -> tuple[el.World, el.EntityId]:
    """
    Create and configure the simulation world with a BDX jet.

    Returns:
        tuple: (world, jet_entity_id)
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
        }

        object_3d bdx.world_pos {
            glb path="f22.glb" scale=0.01 translate="(0.0, 0.0, 0.0)" rotate="(0.0, 0.0, 0.0)"
        }
        
        vector_arrow "(1, 0, 0)" origin="bdx.world_pos" scale=1.0 name="Forward (X)" display_name=#true body_frame=#true {
           color red 150
        }
        vector_arrow "(0, 1, 0)" origin="bdx.world_pos" scale=1.0 name="Left (Y)" display_name=#true body_frame=#true {
           color green 150
        }
        vector_arrow "(0, 0, 1)" origin="bdx.world_pos" scale=1.0 name="Up (Z)" display_name=#true body_frame=#true {
           color blue 150
        }
        
        line_3d bdx.world_pos line_width=3.0 perspective=#false {
            color yolk
        }
        """,
        "bdx.kdl",
    )

    return world, jet


# Setup world
world, jet = setup_world(config)

# Build system
sim_system = system()

print("BDX RC Jet Simulation")
print("=====================")
print(f"Initial altitude: {config.initial_altitude:.1f} m")
print(f"Initial speed: {config.initial_speed:.1f} m/s")
print(f"Initial heading: {config.initial_yaw_deg:.1f}° (0°=East, 90°=North)")
print(f"Mass: {config.mass:.1f} kg")
print(f"Max thrust: {config.propulsion.max_thrust:.1f} N")
print(f"Simulation time: {config.simulation_time:.1f} s")
print(f"Time step: {config.dt:.6f} s ({1 / config.dt:.0f} Hz)")
print(f"Total ticks: {config.total_ticks}")
print()

# Run simulation in real-time mode for responsive RC control
# When using RC controller, run: cargo run -p rc-jet-controller
world.run(
    sim_system,
    sim_time_step=config.dt,
    run_time_step=2 * config.dt,  # Setting this higher than sim allows for real-time control
    max_ticks=config.total_ticks,
)
