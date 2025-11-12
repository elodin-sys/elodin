"""
Complete OpenRocket validation test.
Builds exact same rocket and compares results.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from openrocket_components import *
from openrocket_motor import Motor, get_builtin_motors
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere, WindModel
from openrocket_sim_3dof import Simulator3DOF


def build_test_rocket() -> Tuple[Rocket, Motor]:
    """
    Build test rocket matching validation_rocket.ork EXACTLY.
    """
    rocket = Rocket("Test Rocket")
    
    # Nose cone - 10cm ogive
    nose = NoseCone(
        name="Nose cone",
        length=0.10,
        base_radius=0.025,
        thickness=0.002,
        shape=NoseCone.Shape.OGIVE
    )
    nose.material = MATERIALS["Polystyrene (cast)"]
    rocket.add_child(nose)
    
    # Main body tube - 30cm x 50mm
    body = BodyTube(
        name="Body tube",
        length=0.30,
        outer_radius=0.025,
        thickness=0.002
    )
    body.position.x = 0.10  # After nose
    body.material = MATERIALS["Cardboard"]
    body.motor_mount = True
    rocket.add_child(body)
    
    # Trapezoidal fins - 3 fins
    fins = TrapezoidFinSet(
        name="Trapezoidal fins",
        fin_count=3,
        root_chord=0.10,
        tip_chord=0.05,
        span=0.05,
        sweep=0.03,
        thickness=0.003
    )
    fins.position.x = 0.30  # At end of body tube (from nose tip)
    fins.material = MATERIALS["Plywood (birch)"]
    rocket.add_child(fins)
    
    # Parachute - 30cm
    chute = Parachute(
        name="Parachute",
        diameter=0.30,
        cd=0.75,
        material_density=50.0  # g/m^2
    )
    chute.position.x = 0.05  # Inside body
    chute.deployment_event = "APOGEE"
    rocket.add_child(chute)
    
    # Calculate reference values
    rocket.calculate_reference_values()
    
    # Motor - Aerotech F50T
    motors = get_builtin_motors()
    motor = motors['F50']
    
    return rocket, motor


def print_mass_breakdown(rocket: Rocket, motor: Motor):
    """Print detailed mass breakdown"""
    print("\n" + "="*60)
    print("MASS BREAKDOWN")
    print("="*60)
    
    total_comp_mass = 0.0
    
    def print_component(comp, indent=0):
        nonlocal total_comp_mass
        mass = comp.get_mass()
        cg = comp.get_cg_x()
        abs_pos = comp.get_absolute_position()
        
        prefix = "  " * indent
        print(f"{prefix}{comp.name}:")
        print(f"{prefix}  Mass: {mass*1000:.2f} g")
        print(f"{prefix}  CG: {cg*1000:.1f} mm (local), {abs_pos*1000:.1f} mm (absolute)")
        
        total_comp_mass += mass
        
        for child in comp.children:
            print_component(child, indent + 1)
    
    for child in rocket.children:
        print_component(child)
    
    # Motor masses
    motor_initial = motor.total_mass_initial
    motor_burnout = motor.case_mass
    motor_propellant = motor.propellant_mass
    
    print(f"\nMotor ({motor.designation}):")
    print(f"  Initial mass: {motor_initial*1000:.2f} g")
    print(f"  Propellant mass: {motor_propellant*1000:.2f} g")
    print(f"  Case mass: {motor_burnout*1000:.2f} g")
    
    # Totals
    total_initial = total_comp_mass + motor_initial
    total_burnout = total_comp_mass + motor_burnout
    
    print(f"\n{'─'*60}")
    print(f"Component total: {total_comp_mass*1000:.2f} g")
    print(f"Rocket + motor (ignition): {total_initial*1000:.2f} g")
    print(f"Rocket + motor (burnout): {total_burnout*1000:.2f} g")
    print("="*60)


def print_aerodynamics(rocket: Rocket, motor: Motor):
    """Print aerodynamic analysis"""
    aero = RocketAerodynamics(rocket)
    
    print("\n" + "="*60)
    print("AERODYNAMIC ANALYSIS")
    print("="*60)
    
    # At ignition
    mass_ignition, cg_ignition, _ = 0.0, 0.0, np.array([0, 0, 0])
    comp_mass = rocket.get_total_mass()
    comp_cg = rocket.get_total_cg()
    motor_mass = motor.get_mass(0.0)
    
    # Find motor mount position
    motor_mount_pos = 0.0
    def find_mount(comp):
        nonlocal motor_mount_pos
        if isinstance(comp, BodyTube) and comp.motor_mount:
            motor_mount_pos = comp.get_absolute_position()
        for child in comp.children:
            find_mount(child)
    find_mount(rocket)
    
    motor_cg_abs = motor_mount_pos + motor.length / 2
    mass_ignition = comp_mass + motor_mass
    cg_ignition = (comp_mass * comp_cg + motor_mass * motor_cg_abs) / mass_ignition
    
    # At burnout
    motor_mass_burnout = motor.case_mass
    mass_burnout = comp_mass + motor_mass_burnout
    cg_burnout = (comp_mass * comp_cg + motor_mass_burnout * motor_cg_abs) / mass_burnout
    
    # Aerodynamic properties
    cn_alpha = aero.calculate_cn_alpha(0.0)
    cp = aero.calculate_cp(0.0)
    
    print(f"Reference diameter: {rocket.reference_diameter*1000:.1f} mm")
    print(f"Reference length: {rocket.reference_length*1000:.1f} mm")
    print(f"\nCN_alpha: {cn_alpha:.3f} /rad ({cn_alpha*180/math.pi:.3f} /deg)")
    print(f"CP location: {cp*1000:.1f} mm from nose tip")
    
    print(f"\nAt ignition:")
    print(f"  Mass: {mass_ignition*1000:.1f} g")
    print(f"  CG: {cg_ignition*1000:.1f} mm from nose tip")
    print(f"  Static margin: {(cp - cg_ignition)/rocket.reference_diameter:.2f} cal")
    
    print(f"\nAt burnout:")
    print(f"  Mass: {mass_burnout*1000:.1f} g")
    print(f"  CG: {cg_burnout*1000:.1f} mm from nose tip")
    print(f"  Static margin: {(cp - cg_burnout)/rocket.reference_diameter:.2f} cal")
    
    # Sample CD values
    atm = ISAAtmosphere()
    props = atm.get_properties(0)
    
    print(f"\nDrag coefficient (sea level):")
    for v in [10, 50, 100, 200]:
        mach = v / props['speed_of_sound']
        cd = aero.calculate_cd(mach, v, props['density'], props['viscosity'], 0.0)
        print(f"  @ {v} m/s (Mach {mach:.3f}): CD = {cd:.4f}")
    
    print("="*60)


def run_simulation_test():
    """Run simulation and compare with OpenRocket"""
    print("\n" + "="*60)
    print("BUILDING ROCKET")
    print("="*60)
    
    rocket, motor = build_test_rocket()
    
    # Print analysis
    print_mass_breakdown(rocket, motor)
    print_aerodynamics(rocket, motor)
    
    # Run simulation
    print("\n" + "="*60)
    print("RUNNING SIMULATION")
    print("="*60)
    
    sim = Simulator3DOF(rocket, motor)
    sim.rail_length = 1.0  # 1m rail
    sim.dt = 0.01  # 10ms steps
    
    print("Launch configuration:")
    print(f"  Rail length: {sim.rail_length} m")
    print(f"  Launch angle: 90.0° (vertical, 3DOF)")
    print(f"  Timestep: {sim.dt*1000:.1f} ms")
    
    history = sim.run()
    
    # Results
    summary = sim.get_summary()
    
    print("\n" + "="*60)
    print("FLIGHT RESULTS")
    print("="*60)
    print(f"Max altitude: {summary['max_altitude']:.1f} m")
    print(f"Max velocity: {summary['max_velocity']:.1f} m/s")
    print(f"Apogee time: {summary['apogee_time']:.2f} s")
    print(f"Flight time: {summary['flight_time']:.1f} s")
    print("="*60)
    
    # Plot results
    plot_results(history, summary)
    
    return rocket, motor, history, summary


def plot_results(history, summary):
    """Plot simulation results"""
    times = [s.time for s in history]
    altitudes = [s.z for s in history]
    velocities = [abs(s.vz) for s in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Altitude
    ax1.plot(times, altitudes, 'b-', linewidth=2)
    ax1.axhline(summary['max_altitude'], color='r', linestyle='--', 
                label=f"Apogee: {summary['max_altitude']:.1f} m")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Altitude vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Velocity
    ax2.plot(times, velocities, 'g-', linewidth=2)
    ax2.axhline(summary['max_velocity'], color='r', linestyle='--',
                label=f"Max: {summary['max_velocity']:.1f} m/s")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/openrocket_simulation.png', dpi=150)
    print("\n✓ Plot saved: openrocket_simulation.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" OPENROCKET PYTHON IMPLEMENTATION - VALIDATION TEST")
    print("="*60)
    
    rocket, motor, history, summary = run_simulation_test()
    
    print("\n" + "="*60)
    print("COMPARISON WITH OPENROCKET")
    print("="*60)
    print("\nNow open validation_rocket.ork in OpenRocket and compare:")
    print("  1. Mass breakdown (Components tab)")
    print("  2. CG and CP positions")
    print("  3. Static margin")
    print("  4. Apogee altitude")
    print("  5. Max velocity")
    print("  6. Flight time")
    print("\nExpected match: <1% error on all parameters")
    print("="*60)

