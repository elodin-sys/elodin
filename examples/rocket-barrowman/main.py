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
import typing as ty
from dataclasses import dataclass, field

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
        inclination_deg=0.0,  # degrees from vertical (0 = straight up, 90 = horizontal)
        heading_deg=0.0,  # degrees (north)
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
    
    visualize_in_elodin(result, solver)


def visualize_in_elodin(result: FlightResult, solver: FlightSolver) -> None:
    print(f"\nConverting to Elodin format and launching editor...")

    history = result.history
    times = np.array([s.time for s in history])
    positions = np.array([s.position for s in history])
    velocities = np.array([s.velocity for s in history])
    quaternions = np.array([s.quaternion for s in history])
    angular_velocities = np.array([s.angular_velocity for s in history])

    altitudes = np.array([s.z for s in history])
    velocity_mags = np.array([np.linalg.norm(s.velocity) for s in history])
    machs = np.nan_to_num(np.array([s.mach for s in history]))
    aoas = np.array([s.angle_of_attack for s in history])
    dynamic_pressures = np.array([s.dynamic_pressure for s in history])
    aero_forces = np.array([s.total_aero_force for s in history])

    fin_control_trim_data = angular_velocities
    fin_deflect_data = aero_forces
    aero_coefs_data = np.stack([machs, aoas, dynamic_pressures], axis=1)

    AltitudeComp = ty.Annotated[
        float,
        el.Component("altitude", el.ComponentType(el.PrimitiveType.F64, ())),
    ]
    VelocityMagComp = ty.Annotated[
        float,
        el.Component("velocity_magnitude", el.ComponentType(el.PrimitiveType.F64, ())),
    ]
    MachComp = ty.Annotated[
        float,
        el.Component("mach", el.ComponentType(el.PrimitiveType.F64, ())),
    ]
    AoAComp = ty.Annotated[
        float,
        el.Component("angle_of_attack", el.ComponentType(el.PrimitiveType.F64, ())),
    ]
    DynPressureComp = ty.Annotated[
        float,
        el.Component("dynamic_pressure", el.ComponentType(el.PrimitiveType.F64, ())),
    ]
    Vector3Type = el.ComponentType(el.PrimitiveType.F64, (3,))
    FinControlTrimComp = ty.Annotated[
        np.ndarray,
        el.Component("fin_control_trim", Vector3Type, metadata={"element_names": "x,y,z"}),
    ]
    FinDeflectComp = ty.Annotated[
        np.ndarray,
        el.Component("fin_deflect", Vector3Type, metadata={"element_names": "x,y,z"}),
    ]
    AeroCoefsComp = ty.Annotated[
        np.ndarray,
        el.Component("aero_coefs", Vector3Type, metadata={"element_names": "mach,aoa,q"}),
    ]

    @dataclass
    class RocketTelemetry(el.Archetype):
        altitude: AltitudeComp = 0.0
        velocity_magnitude: VelocityMagComp = 0.0
        mach: MachComp = 0.0
        angle_of_attack: AoAComp = 0.0
        dynamic_pressure: DynPressureComp = 0.0
        fin_control_trim: FinControlTrimComp = field(default_factory=lambda: np.zeros(3))
        fin_deflect: FinDeflectComp = field(default_factory=lambda: np.zeros(3))
        aero_coefs: AeroCoefsComp = field(default_factory=lambda: np.zeros(3))

    world = el.World()
    mass0 = solver.mass_model.total_mass(0.0)
    inertia_diag0 = solver.mass_model.inertia_diag(0.0)

    rocket_entity = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    linear=positions[0],
                    angular=el.Quaternion.from_array(quaternions[0]),
                ),
                inertia=el.SpatialInertia(
                    mass=mass0,
                    inertia=inertia_diag0,
                ),
            ),
            RocketTelemetry(),
        ],
        name="rocket",
    )

    schematic = """
    hsplit {
        tabs share=0.8 {
            viewport name=Viewport pos="rocket.world_pos + (0.0,0.0,0.0,0.0, 5.0, 0.0, 1.0)" look_at="rocket.world_pos" hdr=#true
        }
        vsplit share=0.4 {
            graph "rocket.fin_control_trim" name="Trim Control"
            graph "rocket.fin_deflect" name="Fin Deflection"
            graph "rocket.aero_coefs" name="Aero Coefficients"
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

    if hasattr(world, "schematic"):
        world.schematic(schematic, "rocket.kdl")

    with open("rocket.kdl", "w") as f:
        f.write(schematic)

    exec = el.Executor(world)

    quats_el = [el.Quaternion.from_array(q) for q in quaternions]

    for i in range(len(times)):
        exec.set_component(rocket_entity, "world_pos.linear", positions[i])
        exec.set_component(rocket_entity, "world_pos.angular", quats_el[i])
        exec.set_component(rocket_entity, "world_vel.linear", velocities[i])
        exec.set_component(rocket_entity, "world_vel.angular", angular_velocities[i])
        exec.set_component(rocket_entity, "altitude", float(altitudes[i]))
        exec.set_component(rocket_entity, "velocity_magnitude", float(velocity_mags[i]))
        exec.set_component(rocket_entity, "mach", float(machs[i]))
        exec.set_component(rocket_entity, "angle_of_attack", float(aoas[i]))
        exec.set_component(rocket_entity, "dynamic_pressure", float(dynamic_pressures[i]))
        exec.set_component(rocket_entity, "fin_control_trim", np.asarray(fin_control_trim_data[i]))
        exec.set_component(rocket_entity, "fin_deflect", np.asarray(fin_deflect_data[i]))
        exec.set_component(rocket_entity, "aero_coefs", np.asarray(aero_coefs_data[i]))
        if i < len(times) - 1:
            exec.step(times[i + 1] - times[i])

    print(f"\n✓ Schematic: rocket.kdl")
    print(f"✓ Launching Elodin editor...")

    exec.run_with_schematic(schematic)

    print(f"\n{'='*70}")
    print(f"Simulation + visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
