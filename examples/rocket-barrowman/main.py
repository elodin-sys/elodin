#!/usr/bin/env python3
"""
Calisto Rocket Simulation - Elodin Integration
Runs the RocketPy-compatible 6-DOF solver and visualizes in Elodin editor

Usage:
    # From elodin root directory, in nix shell:
    nix develop
    python3 examples/rocket-barrowman/main.py
"""

import sys
import os
import numpy as np

# Add the rocket-barrowman directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    import elodin as el
except ImportError:
    print("ERROR: elodin not found. Make sure you're in the nix shell:")
    print("  cd /home/kush-mahajan/elodin")
    print("  nix develop")
    print("  python3 examples/rocket-barrowman/main.py")
    sys.exit(1)

# Import our rocket simulation
from environment import Environment
from motor_model import Motor
from rocket_model import Rocket as RocketModel
from flight_solver import FlightSolver, FlightResult
from calisto_builder import build_calisto


def main():
    """Run Calisto rocket simulation and visualize in Elodin."""
    
    print("\n" + "="*70)
    print("CALISTO ROCKET SIMULATION - ELODIN")
    print("="*70)
    
    # Build Calisto rocket
    rocket_raw, motor_raw = build_calisto()
    rocket = RocketModel(rocket_raw)
    motor = Motor.from_openrocket(motor_raw)
    
    print(f"\n✓ Calisto loaded:")
    print(f"  Dry mass: {rocket.dry_mass:.3f} kg")
    print(f"  Dry CG: {rocket.dry_cg:.3f} m")
    print(f"  Reference diameter: {rocket.reference_diameter*1000:.1f} mm")
    print(f"  Parachutes: {len(rocket.parachutes)}")
    
    # Environment (Spaceport America: elevation=1400m, no wind)
    env = Environment(elevation=1400.0)
    
    # Run simulation
    print(f"\nRunning simulation...")
    solver = FlightSolver(
        rocket=rocket,
        motor=motor,
        environment=env,
        rail_length=5.2,  # meters
        inclination_deg=90.0,  # degrees from horizontal (90 = vertical)
        heading_deg=0.0,  # degrees
    )
    
    result = solver.run(max_time=200.0)
    
    # Print summary
    if len(result.history) == 0:
        print(f"\n❌ Simulation failed - no history recorded")
        return
    
    max_alt = max(s.z for s in result.history)
    apogee_time = next(s.time for s in result.history if s.z == max_alt)
    max_v = max(np.linalg.norm(s.velocity) for s in result.history)
    
    print(f"\n✓ Simulation complete:")
    print(f"  Max altitude: {max_alt:.1f} m (AGL)")
    print(f"  Apogee time: {apogee_time:.2f} s")
    print(f"  Max velocity: {max_v:.1f} m/s")
    print(f"  History length: {len(result.history)} snapshots")
    
    # Debug: print first few states
    print(f"\nFirst 3 states:")
    for i, s in enumerate(result.history[:3]):
        print(f"  t={s.time:.2f}s: pos={s.position}, z={s.z:.2f}m")
    
    print(f"\n{'='*70}")
    print(f"Simulation complete! Elodin visualization coming soon...")
    print(f"{'='*70}\n")
    return
    
    # TODO: Convert to Elodin format
    print(f"\nConverting to Elodin format...")
    
    # Create Elodin world
    world_builder = el.WorldBuilder()
    
    # Create rocket entity
    rocket_entity = world_builder.spawn(
        el.Body(
            world_pos=el.SpatialTransform(
                pos=el.Component(shape=[3]),
                rot=el.Component(shape=[4]),
            ),
            world_vel=el.SpatialMotion(
                linear=el.Component(shape=[3]),
                angular=el.Component(shape=[3]),
            ),
        ),
        name="rocket",
    )
    
    # Add metadata components
    rocket_entity.insert(
        el.Metadata(
            altitude=el.Component(shape=[]),
            velocity_magnitude=el.Component(shape=[]),
            mach=el.Component(shape=[]),
            angle_of_attack=el.Component(shape=[]),
            dynamic_pressure=el.Component(shape=[]),
        )
    )
    
    # Build world
    world = world_builder.build()
    
    # Create time series data
    times = np.array([s.time for s in result.history])
    positions = np.array([s.position for s in result.history])
    velocities = np.array([s.velocity for s in result.history])
    quaternions = np.array([s.quaternion for s in result.history])
    angular_velocities = np.array([s.angular_velocity for s in result.history])
    
    # Metadata
    altitudes = np.array([s.z for s in result.history])
    velocity_mags = np.array([np.linalg.norm(s.velocity) for s in result.history])
    machs = np.array([s.mach for s in result.history])
    aoas = np.array([s.angle_of_attack for s in result.history])
    dynamic_pressures = np.array([s.dynamic_pressure for s in result.history])
    
    # Create exec
    exec = el.Executor(world)
    
    # Set initial state
    exec.set_component(rocket_entity.id(), "world_pos.pos", positions[0])
    exec.set_component(rocket_entity.id(), "world_pos.rot", quaternions[0])
    exec.set_component(rocket_entity.id(), "world_vel.linear", velocities[0])
    exec.set_component(rocket_entity.id(), "world_vel.angular", angular_velocities[0])
    exec.set_component(rocket_entity.id(), "altitude", altitudes[0])
    exec.set_component(rocket_entity.id(), "velocity_magnitude", velocity_mags[0])
    exec.set_component(rocket_entity.id(), "mach", machs[0])
    exec.set_component(rocket_entity.id(), "angle_of_attack", aoas[0])
    exec.set_component(rocket_entity.id(), "dynamic_pressure", dynamic_pressures[0])
    
    # Create schematic
    schematic = """
    hsplit {
        tabs share=0.8 {
            viewport name=Viewport pos="rocket.world_pos + (0.0,0.0,0.0,0.0, 5.0, 0.0, 1.0)" look_at="rocket.world_pos" hdr=#true
        }
        vsplit share=0.4 {
            graph "rocket.altitude" name="Altitude (m)"
            graph "rocket.velocity_magnitude" name="Velocity (m/s)"
            graph "rocket.mach" name="Mach Number"
            graph "rocket.angle_of_attack" name="Angle of Attack (rad)"
            graph "rocket.dynamic_pressure" name="Dynamic Pressure (Pa)"
        }
    }

    object_3d "(0,0,0,1, rocket.world_pos[4],rocket.world_pos[5],rocket.world_pos[6])" {
        glb path="compass.glb"
    }
    object_3d rocket.world_pos {
        glb path="https://storage.googleapis.com/elodin-assets/rocket.glb"
    }
    line_3d rocket.world_pos line_width=11.0 perspective=#false {
        color yolk
    }
    """
    
    # Run simulation with visualization
    print(f"\nLaunching Elodin editor...")
    
    for i, t in enumerate(times):
        exec.set_component(rocket_entity.id(), "world_pos.pos", positions[i])
        exec.set_component(rocket_entity.id(), "world_pos.rot", quaternions[i])
        exec.set_component(rocket_entity.id(), "world_vel.linear", velocities[i])
        exec.set_component(rocket_entity.id(), "world_vel.angular", angular_velocities[i])
        exec.set_component(rocket_entity.id(), "altitude", altitudes[i])
        exec.set_component(rocket_entity.id(), "velocity_magnitude", velocity_mags[i])
        exec.set_component(rocket_entity.id(), "mach", machs[i])
        exec.set_component(rocket_entity.id(), "angle_of_attack", aoas[i])
        exec.set_component(rocket_entity.id(), "dynamic_pressure", dynamic_pressures[i])
        
        # Step with delta time
        if i < len(times) - 1:
            dt = times[i + 1] - times[i]
            exec.step(dt)
    
    # Save schematic and open editor
    with open("rocket.kdl", "w") as f:
        f.write(schematic)
    
    print(f"\n✓ Schematic saved to rocket.kdl")
    print(f"✓ Opening Elodin editor...")
    
    # Open in editor
    exec.run_with_schematic(schematic)
    
    print(f"\n{'='*70}")
    print(f"Simulation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
