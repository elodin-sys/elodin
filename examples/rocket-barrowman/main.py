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
import jax
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
    aero_forces = jnp.asarray([s.total_aero_force for s in history])

    altitudes = positions[:, 2]
    downrange = jnp.linalg.norm(positions[:, :2], axis=1)
    speed = jnp.linalg.norm(velocities, axis=1)

    accel_vec = (velocities[1:] - velocities[:-1]) / dt
    accel_vec = jnp.concatenate((accel_vec[:1], accel_vec), axis=0)
    accel_mag = jnp.linalg.norm(accel_vec, axis=1)

    machs = jnp.nan_to_num(jnp.asarray([s.mach for s in history]))
    aoas = jnp.asarray([s.angle_of_attack for s in history])
    dynamic_pressures = jnp.asarray([s.dynamic_pressure for s in history])
    aero_force_mag = jnp.linalg.norm(aero_forces, axis=1)

    def quat_to_euler_deg(q: jnp.ndarray) -> jnp.ndarray:
        x, y, z, w = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = jnp.where(jnp.abs(sinp) >= 1.0, jnp.sign(sinp) * (jnp.pi / 2.0), jnp.arcsin(sinp))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        return jnp.rad2deg(jnp.array([roll, pitch, yaw]))

    euler_deg = jax.vmap(quat_to_euler_deg)(quaternions)

    angular_rates_deg = jnp.rad2deg(angular_velocities)

    Altitude = ty.Annotated[jax.Array, el.Component("altitude_m", el.ComponentType.F64)]
    Downrange = ty.Annotated[jax.Array, el.Component("downrange_m", el.ComponentType.F64)]
    Speed = ty.Annotated[jax.Array, el.Component("speed_ms", el.ComponentType.F64)]
    Accel = ty.Annotated[jax.Array, el.Component("accel_ms2", el.ComponentType.F64)]
    MachComp = ty.Annotated[jax.Array, el.Component("mach", el.ComponentType.F64)]
    AngleOfAttackComp = ty.Annotated[jax.Array, el.Component("angle_of_attack_deg", el.ComponentType.F64)]
    DynamicPressureComp = ty.Annotated[jax.Array, el.Component("dynamic_pressure_pa", el.ComponentType.F64)]
    AeroForceMag = ty.Annotated[jax.Array, el.Component("aero_force_n", el.ComponentType.F64)]
    EulerAngles = ty.Annotated[
        jax.Array,
        el.Component(
            "euler_deg",
            el.ComponentType(el.PrimitiveType.F64, (3,)),
            metadata={"element_names": "roll,pitch,yaw"},
        ),
    ]
    BodyRates = ty.Annotated[
        jax.Array,
        el.Component(
            "angular_rates_deg", el.ComponentType(el.PrimitiveType.F64, (3,)), metadata={"element_names": "p,q,r"}
        ),
    ]

    @el.dataclass
    class PlaybackTelemetry(el.Archetype):
        altitude: Altitude = field(default_factory=lambda: jnp.float64(0.0))
        downrange: Downrange = field(default_factory=lambda: jnp.float64(0.0))
        speed: Speed = field(default_factory=lambda: jnp.float64(0.0))
        acceleration: Accel = field(default_factory=lambda: jnp.float64(0.0))
        mach: MachComp = field(default_factory=lambda: jnp.float64(0.0))
        angle_of_attack: AngleOfAttackComp = field(default_factory=lambda: jnp.float64(0.0))
        dynamic_pressure: DynamicPressureComp = field(default_factory=lambda: jnp.float64(0.0))
        aero_force: AeroForceMag = field(default_factory=lambda: jnp.float64(0.0))
        euler: EulerAngles = field(default_factory=lambda: jnp.zeros(3))
        angular_rates: BodyRates = field(default_factory=lambda: jnp.zeros(3))

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
            PlaybackTelemetry(),
        ],
        name="rocket",
        id="rocket",
    )

    schematic = """
    tabs {
        hsplit share=0.65 name="Flight View" {
            viewport name=Viewport pos="rocket.world_pos + (0,0,0,0, 6,3,2)" look_at="rocket.world_pos" hdr=#true show_grid=#true active=#true
            vsplit share=0.45 {
                graph "rocket.altitude_m,rocket.downrange_m" name="Altitude / Downrange"
                graph "rocket.speed_ms,rocket.accel_ms2" name="Speed / Accel"
                graph "rocket.dynamic_pressure_pa,rocket.mach,rocket.aero_force_n" name="Aero Loads"
            }
        }
        hsplit share=0.35 name="Orientation" {
            graph "rocket.euler_deg[0],rocket.euler_deg[1],rocket.euler_deg[2]" name="Euler Angles"
            graph "rocket.angular_rates_deg[0],rocket.angular_rates_deg[1],rocket.angular_rates_deg[2]" name="Body Rates"
        }
    }

    object_3d "(0,0,0,1, rocket.world_pos[4],rocket.world_pos[5],rocket.world_pos[6])" {
        glb path="compass.glb"
    }
    object_3d rocket.world_pos {
        glb path="https://storage.googleapis.com/elodin-assets/rocket.glb"
    }
    line_3d rocket.world_pos line_width=6.0 perspective=#false {
        color 255 223 0
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
        comp: el.Query[Altitude],
    ) -> el.Query[Altitude]:
        idx = frame_index(tick[0])
        return comp.map(Altitude, lambda _: jnp.float64(altitudes[idx]))

    @el.system
    def playback_downrange(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Downrange],
    ) -> el.Query[Downrange]:
        idx = frame_index(tick[0])
        return comp.map(Downrange, lambda _: jnp.float64(downrange[idx]))

    @el.system
    def playback_speed(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Speed],
    ) -> el.Query[Speed]:
        idx = frame_index(tick[0])
        return comp.map(Speed, lambda _: jnp.float64(speed[idx]))

    @el.system
    def playback_accel(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Accel],
    ) -> el.Query[Accel]:
        idx = frame_index(tick[0])
        return comp.map(Accel, lambda _: jnp.float64(accel_mag[idx]))

    @el.system
    def playback_mach(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[MachComp],
    ) -> el.Query[MachComp]:
        idx = frame_index(tick[0])
        return comp.map(MachComp, lambda _: jnp.float64(machs[idx]))

    @el.system
    def playback_aoa(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AngleOfAttackComp],
    ) -> el.Query[AngleOfAttackComp]:
        idx = frame_index(tick[0])
        return comp.map(AngleOfAttackComp, lambda _: jnp.float64(aoas[idx]))

    @el.system
    def playback_dynamic_pressure(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[DynamicPressureComp],
    ) -> el.Query[DynamicPressureComp]:
        idx = frame_index(tick[0])
        return comp.map(DynamicPressureComp, lambda _: jnp.float64(dynamic_pressures[idx]))

    @el.system
    def playback_aero_force(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AeroForceMag],
    ) -> el.Query[AeroForceMag]:
        idx = frame_index(tick[0])
        return comp.map(AeroForceMag, lambda _: jnp.float64(aero_force_mag[idx]))

    @el.system
    def playback_euler(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[EulerAngles],
    ) -> el.Query[EulerAngles]:
        idx = frame_index(tick[0])
        return comp.map(EulerAngles, lambda _: euler_deg[idx])

    @el.system
    def playback_body_rates(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[BodyRates],
    ) -> el.Query[BodyRates]:
        idx = frame_index(tick[0])
        return comp.map(BodyRates, lambda _: angular_rates_deg[idx])

    playback_system = (
        playback_world_pos
        | playback_world_vel
        | playback_altitude
        | playback_downrange
        | playback_speed
        | playback_accel
        | playback_mach
        | playback_aoa
        | playback_dynamic_pressure
        | playback_aero_force
        | playback_euler
        | playback_body_rates
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
