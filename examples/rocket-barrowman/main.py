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
import jax.numpy as jnp

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
    times_np = np.array([s.time for s in history])
    dt = float(times_np[1] - times_np[0]) if len(times_np) > 1 else 0.01

    positions = jnp.asarray([s.position for s in history])
    velocities = jnp.asarray([s.velocity for s in history])
    quaternions = jnp.asarray([s.quaternion for s in history])
    angular_velocities = jnp.asarray([s.angular_velocity for s in history])

    altitudes = jnp.asarray([s.z for s in history])
    velocity_mags = jnp.asarray([jnp.linalg.norm(jnp.asarray(s.velocity)) for s in history])
    machs = jnp.nan_to_num(jnp.asarray([s.mach for s in history]))
    aoas = jnp.asarray([s.angle_of_attack for s in history])
    dynamic_pressures = jnp.asarray([s.dynamic_pressure for s in history])
    aero_forces = jnp.asarray([s.total_aero_force for s in history])

    fin_control_trim_data = angular_velocities
    fin_deflect_data = aero_forces
    aero_coefs_data = jnp.stack([machs, aoas, dynamic_pressures], axis=1)

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
        fin_control_trim: FinControlTrimComp = field(default_factory=lambda: jnp.zeros(3))
        fin_deflect: FinDeflectComp = field(default_factory=lambda: jnp.zeros(3))
        aero_coefs: AeroCoefsComp = field(default_factory=lambda: jnp.zeros(3))

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
        id="rocket",
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

    frame_count = positions.shape[0]
    frame_max = float(frame_count - 1)

    def frame_index(tick_value: jnp.ndarray) -> jnp.ndarray:
        idx = jnp.clip(jnp.floor(tick_value), 0.0, frame_max)
        return idx.astype(jnp.int32)

    @el.system
    def playback_world_pos(
        tick: el.Query[el.SimulationTick],
        pos_q: el.Query[el.WorldPos],
    ) -> el.Query[el.WorldPos]:
        idx = frame_index(tick[0])
        transform = el.SpatialTransform(
            linear=positions[idx],
            angular=el.Quaternion.from_array(quaternions[idx]),
        )
        return pos_q.map(el.WorldPos, lambda _: transform)

    @el.system
    def playback_world_vel(
        tick: el.Query[el.SimulationTick],
        vel_q: el.Query[el.WorldVel],
    ) -> el.Query[el.WorldVel]:
        idx = frame_index(tick[0])
        motion = el.SpatialMotion(
            linear=velocities[idx],
            angular=angular_velocities[idx],
        )
        return vel_q.map(el.WorldVel, lambda _: motion)

    @el.system
    def playback_altitude(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AltitudeComp],
    ) -> el.Query[AltitudeComp]:
        idx = frame_index(tick[0])
        return comp.map(AltitudeComp, lambda _: float(altitudes[idx]))

    @el.system
    def playback_velocity_mag(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[VelocityMagComp],
    ) -> el.Query[VelocityMagComp]:
        idx = frame_index(tick[0])
        return comp.map(VelocityMagComp, lambda _: float(velocity_mags[idx]))

    @el.system
    def playback_mach(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[MachComp],
    ) -> el.Query[MachComp]:
        idx = frame_index(tick[0])
        return comp.map(MachComp, lambda _: float(machs[idx]))

    @el.system
    def playback_aoa(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AoAComp],
    ) -> el.Query[AoAComp]:
        idx = frame_index(tick[0])
        return comp.map(AoAComp, lambda _: float(aoas[idx]))

    @el.system
    def playback_dynamic_pressure(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[DynPressureComp],
    ) -> el.Query[DynPressureComp]:
        idx = frame_index(tick[0])
        return comp.map(DynPressureComp, lambda _: float(dynamic_pressures[idx]))

    @el.system
    def playback_fin_trim(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[FinControlTrimComp],
    ) -> el.Query[FinControlTrimComp]:
        idx = frame_index(tick[0])
        return comp.map(FinControlTrimComp, lambda _: fin_control_trim_data[idx])

    @el.system
    def playback_fin_deflect(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[FinDeflectComp],
    ) -> el.Query[FinDeflectComp]:
        idx = frame_index(tick[0])
        return comp.map(FinDeflectComp, lambda _: fin_deflect_data[idx])

    @el.system
    def playback_aero_coefs(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AeroCoefsComp],
    ) -> el.Query[AeroCoefsComp]:
        idx = frame_index(tick[0])
        return comp.map(AeroCoefsComp, lambda _: aero_coefs_data[idx])

    playback_system = (
        playback_world_pos
        | playback_world_vel
        | playback_altitude
        | playback_velocity_mag
        | playback_mach
        | playback_aoa
        | playback_dynamic_pressure
        | playback_fin_trim
        | playback_fin_deflect
        | playback_aero_coefs
    )

    print(f"\n✓ Schematic: rocket.kdl")
    print(f"✓ Launching Elodin editor...")

    world.run(
        playback_system,
        sim_time_step=dt,
        run_time_step=dt,
        default_playback_speed=1.0,
        max_ticks=frame_count,
    )

    print(f"\n{'='*70}")
    print(f"Simulation + visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
