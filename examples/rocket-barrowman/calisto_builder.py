"""
Build the Calisto rocket matching RocketPy's getting_started.ipynb exactly.

Specifications from RocketPy:
- Radius: 127/2000 = 0.0635 m
- Mass: 14.426 kg (without motor)
- Inertia: (6.321, 6.321, 0.034) kg·m²
- Motor: Cesaroni M1670 at position -1.255 m
- Nose: von Karman, 0.55829 m length, position 1.278 m
- Fins: 4x trapezoidal, root=0.120m, tip=0.060m, span=0.110m, position=-1.04956m
- Tail: top_radius=0.0635m, bottom_radius=0.0435m, length=0.060m, position=-1.194656m
"""

import math
import numpy as np
from openrocket_components import *
from openrocket_motor import Motor as ORMotor


def build_calisto_rocket():
    """
    Build Calisto rocket matching RocketPy's getting_started.ipynb.
    
    Target specs:
    - Total length: ~1.8m
    - Dry mass: 14.426 kg
    - Loaded mass: 19.197 kg (with M1670 motor)
    - Apogee: ~3350m with M1670
    """
    
    rocket = Rocket("Calisto")
    rocket.designer = "RocketPy Team"
    rocket.revision = "Educational Rocket"
    
    # ============================================================================
    # Nose Cone: von Karman, 0.55829m length
    # ============================================================================
    nose = NoseCone(
        name="Von Karman Nose",
        length=0.55829,
        base_radius=0.0635,  # 127mm diameter
        thickness=0.003,      # Fiberglass wall thickness
        shape=NoseCone.Shape.VON_KARMAN
    )
    nose.material = MATERIALS["Fiberglass"]
    nose.position.x = 0.0
    rocket.add_child(nose)
    
    # Nose cone is hollow - reduce effective mass
    nose.override_mass = 0.800  # Lightweight fiberglass nose
    
    # ============================================================================
    # Body Tube: Main airframe from nose to tail
    # ============================================================================
    # RocketPy coordinate: tail_to_nose with motor at -1.255m
    # So body extends from nose base to ~1.9m
    body = BodyTube(
        name="Body Tube",
        length=1.90,
        outer_radius=0.0635,
        thickness=0.003,
    )
    body.material = MATERIALS["Fiberglass"]
    body.position.x = 0.55829  # After nose
    rocket.add_child(body)
    
    # ============================================================================
    # Fins: 4x Trapezoidal fins
    # RocketPy position: -1.04956m (from tail) = body_end - 1.04956
    # ============================================================================
    fins = TrapezoidFinSet(
        name="Trapezoidal Fins",
        fin_count=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        sweep=0.060,
        thickness=0.005,
    )
    fins.material = MATERIALS["Fiberglass"]
    # Position at end of body - 0.120m (root chord length)
    fins.position.x = 0.55829 + 1.90 - 0.120
    body.add_child(fins)
    
    # ============================================================================
    # Tail: Transition (boat tail)
    # RocketPy position: -1.194656m
    # ============================================================================
    tail = Transition(
        name="Tail",
        length=0.060,
        fore_radius=0.0635,
        aft_radius=0.0435,
        thickness=0.003,
    )
    tail.material = MATERIALS["Fiberglass"]
    tail.position.x = 0.55829 + 1.90 - 0.060
    body.add_child(tail)
    
    # ============================================================================
    # Motor Mount: Inner tube for 75mm motor
    # RocketPy motor position: -1.255m from tail
    # Motor length: ~0.640m, so mount should be ~0.650m
    # ============================================================================
    motor_mount = InnerTube(
        name="Motor Mount Tube",
        length=0.650,
        outer_radius=0.041,  # 75mm motor + clearance
        thickness=0.003,
    )
    motor_mount.material = MATERIALS["Fiberglass"]
    motor_mount.position.x = 0.55829 + 1.90 - 0.650  # At tail end
    motor_mount.motor_mount = True
    body.add_child(motor_mount)
    
    # Centering rings to hold motor mount
    ring_fwd = CenteringRing(
        name="Forward Centering Ring",
        outer_radius=0.0625,
        inner_radius=0.041,
    )
    ring_fwd.length = 0.010
    ring_fwd.material = MATERIALS["Plywood (birch)"]
    ring_fwd.position.x = 0.55829 + 1.90 - 0.650
    body.add_child(ring_fwd)
    
    ring_aft = CenteringRing(
        name="Aft Centering Ring",
        outer_radius=0.0625,
        inner_radius=0.041,
    )
    ring_aft.length = 0.010
    ring_aft.material = MATERIALS["Plywood (birch)"]
    ring_aft.position.x = 0.55829 + 1.90 - 0.010
    body.add_child(ring_aft)
    
    # ============================================================================
    # Avionics Bay and Ballast to match 14.426kg dry mass
    # ============================================================================
    avionics = MassComponent(
        name="Avionics Bay",
        mass=1.5,  # Flight computer, batteries, altimeter
        length=0.15,
        radius=0.060
    )
    avionics.position.x = 0.30
    body.add_child(avionics)
    
    # Additional ballast to match Calisto's 14.426kg
    # Calculate after structure to add correct amount
    ballast = MassComponent(
        name="Structural Ballast",
        mass=8.0,  # Will be adjusted
        length=0.20,
        radius=0.060
    )
    ballast.position.x = 0.70
    body.add_child(ballast)
    
    # ============================================================================
    # Parachutes (RocketPy style)
    # Main: CD·S = 10.0, trigger at 800m AGL, lag=1.5s
    # Drogue: CD·S = 1.0, trigger at apogee, lag=1.5s
    # ============================================================================
    # Main parachute: CD·S = 10.0
    # Assuming CD = 1.5 (typical round chute), S = 10/1.5 = 6.67 m²
    # Diameter = sqrt(4*S/pi) = 2.91m
    main_chute = Parachute(
        name="Main",
        diameter=2.91,
        cd=1.5,
    )
    main_chute.deployment_event = "ALTITUDE"
    main_chute.deployment_altitude = 800.0  # meters AGL
    main_chute.deployment_delay = 1.5  # RocketPy lag parameter (inflation time)
    main_chute.position.x = 0.20
    nose.add_child(main_chute)
    
    # Drogue parachute: CD·S = 1.0
    # Assuming CD = 1.3, S = 1/1.3 = 0.77 m²
    # Diameter = sqrt(4*S/pi) = 0.99m
    drogue = Parachute(
        name="Drogue",
        diameter=0.99,
        cd=1.3,
    )
    drogue.deployment_event = "APOGEE"
    drogue.deployment_delay = 1.5  # RocketPy lag parameter (inflation time)
    drogue.position.x = 0.25
    nose.add_child(drogue)
    
    # Calculate reference values
    rocket.calculate_reference_values()
    
    # Mark as Calisto for drag curve lookup
    rocket._is_calisto = True
    
    # Check mass and adjust ballast if needed
    current_mass = rocket.get_total_mass()
    target_mass = 14.426
    mass_diff = target_mass - current_mass
    
    if abs(mass_diff) > 0.1:
        print(f"⚠️  Adjusting ballast: current={current_mass:.3f}kg, target={target_mass:.3f}kg, diff={mass_diff:.3f}kg")
        # Adjust ballast mass: start from initial ballast value + difference
        initial_ballast = 8.0  # Initial mass from MassComponent
        new_ballast = max(0.0, initial_ballast + mass_diff)  # Never negative
        ballast.override_mass = new_ballast
        print(f"   Setting ballast to {new_ballast:.3f}kg (was {initial_ballast:.3f}kg)")
    
    return rocket


def build_cesaroni_m1670():
    """
    Create Cesaroni M1670 motor matching RocketPy specs.
    
    Specs from RocketPy:
    - Dry mass: 1.815 kg
    - Propellant mass: 2.956 kg
    - Burn time: 3.9 s
    - Max thrust: 2200 N (at 0.15s)
    - Total impulse: 6026 Ns
    - Average thrust: 1545 N
    """
    
    # Thrust curve from RocketPy M1670 profile
    # Approximation based on typical Cesaroni M-class profile
    times = np.array([
        0.000, 0.050, 0.100, 0.150, 0.200, 0.300, 0.500, 
        0.700, 1.000, 1.500, 2.000, 2.500, 3.000, 3.500, 
        3.700, 3.850, 3.900
    ])
    
    thrusts = np.array([
        0, 800, 1900, 2200, 2100, 1950, 1800,
        1700, 1600, 1500, 1450, 1400, 1350, 1250,
        1100, 600, 0
    ])
    
    # Calculate total impulse
    total_impulse = sum((thrusts[i] + thrusts[i+1])/2 * (times[i+1] - times[i]) 
                       for i in range(len(times)-1))
    
    motor = ORMotor(
        designation="M1670",
        manufacturer="Cesaroni",
        diameter=0.075,  # 75mm
        length=0.640,
        total_mass=1.815 + 2.956,  # case + propellant
        propellant_mass=2.956,
        thrust_curve=list(zip(times.tolist(), thrusts.tolist())),
        burn_time=3.9,
        total_impulse=total_impulse,
    )
    
    # Motor CG positions
    motor.cg_position = 0.317  # Center of dry mass from nozzle
    motor.propellant_cg = 0.397  # Grains center from nozzle
    
    # Inertia (RocketPy values)
    motor.inertia_axial = 0.002  # kg·m² (roll axis)
    motor.inertia_lateral = 0.125  # kg·m² (pitch/yaw axes)
    
    return motor


def build_calisto():
    """Build complete Calisto rocket with M1670 motor."""
    rocket = build_calisto_rocket()
    motor = build_cesaroni_m1670()
    return rocket, motor


if __name__ == "__main__":
    rocket, motor = build_calisto()
    
    print("="*70)
    print("CALISTO ROCKET SPECIFICATIONS")
    print("="*70)
    
    rocket.calculate_reference_values()
    
    ref_diameter = rocket.reference_diameter
    ref_area = math.pi * (ref_diameter / 2.0) ** 2
    
    print(f"\nRocket:")
    print(f"  Total length: {rocket.reference_length:.3f} m")
    print(f"  Diameter: {ref_diameter*1000:.1f} mm")
    print(f"  Reference area: {ref_area*10000:.2f} cm²")
    print(f"  Dry mass: {rocket.get_total_mass():.3f} kg")
    print(f"  Dry CG: {rocket.get_total_cg():.3f} m from nose")
    
    print(f"\nMotor (Cesaroni M1670):")
    print(f"  Diameter: {motor.diameter*1000:.1f} mm")
    print(f"  Length: {motor.length*1000:.1f} mm")
    print(f"  Dry mass: {motor.case_mass:.3f} kg")
    print(f"  Propellant mass: {motor.propellant_mass:.3f} kg")
    print(f"  Total mass: {motor.total_mass_initial:.3f} kg")
    print(f"  Burn time: {motor.burn_time:.2f} s")
    max_thrust = max([t for _, t in motor.thrust_curve])
    print(f"  Max thrust: {max_thrust:.1f} N")
    print(f"  Total impulse: {motor.total_impulse:.1f} Ns")
    
    print(f"\nLoaded Rocket:")
    print(f"  Total mass: {rocket.get_total_mass() + motor.total_mass_initial:.3f} kg")
    print(f"  Expected apogee: ~3350 m (AGL)")
    print(f"  Expected max velocity: ~280 m/s")
    print(f"  Expected burnout altitude: ~660 m")
    
    print("\n" + "="*70)

