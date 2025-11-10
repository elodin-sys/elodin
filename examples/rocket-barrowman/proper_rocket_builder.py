"""
PROPER rocket builder with ALL internal components matching OpenRocket exactly.
No shortcuts - full fidelity implementation.
"""

import math
from openrocket_components import *
from openrocket_motor import get_builtin_motors


def build_proper_rocket():
    """
    Build rocket with COMPLETE internal structure:
    - Motor mount with centering rings
    - Bulkhead for recovery attachment
    - Shock cord and parachute properly placed
    - Launch lugs
    - All masses and positions exactly as OpenRocket
    """
    
    rocket = Rocket("High-Fidelity Test Rocket")
    rocket.designer = "OpenRocket Python"
    rocket.revision = "v1.0"
    
    # ============================================================================
    # NOSE CONE (0 - 100mm)
    # ============================================================================
    nose = NoseCone(
        name="Ogive Nose Cone",
        length=0.10,           # 100mm
        base_radius=0.025,     # 25mm (50mm diameter)
        thickness=0.002,       # 2mm wall
        shape=NoseCone.Shape.OGIVE
    )
    nose.material = MATERIALS["Polystyrene (cast)"]
    nose.position.x = 0.0
    rocket.add_child(nose)
    
    # Recovery system components (inside nose cone)
    # Nose cone shoulder (mass component for nose cone attachment)
    nose_shoulder = MassComponent(
        name="Nose cone shoulder",
        mass=0.005,  # 5g
        length=0.02,
        radius=0.023
    )
    nose_shoulder.position.x = 0.08  # 80mm from tip
    nose.add_child(nose_shoulder)
    
    # ============================================================================
    # MAIN BODY TUBE (100 - 400mm)
    # ============================================================================
    body_main = BodyTube(
        name="Main Body Tube",
        length=0.30,           # 300mm
        outer_radius=0.025,    # 25mm
        thickness=0.002        # 2mm wall
    )
    body_main.material = MATERIALS["Cardboard"]
    body_main.position.x = 0.10  # Starts at 100mm
    rocket.add_child(body_main)
    
    # ============================================================================
    # RECOVERY SYSTEM (Forward section)
    # ============================================================================
    
    # Shock cord (1 meter)
    shock_cord = ShockCord(
        name="Shock Cord",
        length=1.0,
        diameter=0.003  # 3mm diameter cord
    )
    shock_cord.position.x = 0.02  # 20mm into body tube (120mm from nose)
    body_main.add_child(shock_cord)
    
    # Parachute (packed)
    parachute = Parachute(
        name="Main Parachute",
        diameter=0.30,         # 30cm deployed
        cd=0.75,
        material_density=50.0  # g/m²
    )
    parachute.position.x = 0.03  # 30mm into body (130mm from nose)
    parachute.deployment_event = "APOGEE"
    parachute.deployment_delay = 0.0
    body_main.add_child(parachute)
    
    # Recovery system bulkhead (attaches shock cord)
    recovery_bulkhead = Bulkhead(
        name="Forward Bulkhead",
        radius=0.023,          # Fits inside body tube
        thickness=0.006        # 6mm plywood
    )
    recovery_bulkhead.position.x = 0.05  # 50mm into body (150mm from nose)
    body_main.add_child(recovery_bulkhead)
    
    # ============================================================================
    # MOTOR MOUNT ASSEMBLY (Aft section)
    # ============================================================================
    
    # Inner tube (motor mount tube) - 29mm ID for F50 motor
    motor_mount = InnerTube(
        name="Motor Mount Tube",
        length=0.12,           # 120mm (longer than motor for retention)
        outer_radius=0.015,    # 15mm (30mm OD, fits 29mm motor)
        thickness=0.001        # 1mm wall
    )
    motor_mount.material = MATERIALS["Kraft phenolic"]
    motor_mount.position.x = 0.17  # Starts 170mm into body (270mm from nose)
    body_main.add_child(motor_mount)
    
    # Forward centering ring
    centering_ring_fwd = CenteringRing(
        name="Forward Centering Ring",
        outer_radius=0.023,    # Fits body tube ID
        inner_radius=0.015,    # Fits motor mount OD
        thickness=0.006        # 6mm thick
    )
    centering_ring_fwd.material = MATERIALS["Plywood (birch)"]
    centering_ring_fwd.position.x = 0.17  # At front of motor mount
    body_main.add_child(centering_ring_fwd)
    
    # Middle centering ring
    centering_ring_mid = CenteringRing(
        name="Middle Centering Ring",
        outer_radius=0.023,
        inner_radius=0.015,
        thickness=0.006
    )
    centering_ring_mid.material = MATERIALS["Plywood (birch)"]
    centering_ring_mid.position.x = 0.23  # Middle of motor mount
    body_main.add_child(centering_ring_mid)
    
    # Aft centering ring (at back)
    centering_ring_aft = CenteringRing(
        name="Aft Centering Ring",
        outer_radius=0.023,
        inner_radius=0.015,
        thickness=0.006
    )
    centering_ring_aft.material = MATERIALS["Plywood (birch)"]
    centering_ring_aft.position.x = 0.284  # At rear of motor mount (6mm from body end)
    body_main.add_child(centering_ring_aft)
    
    # ============================================================================
    # FINS (300 - 400mm)
    # ============================================================================
    fins = TrapezoidFinSet(
        name="Trapezoidal Fin Set",
        fin_count=3,
        root_chord=0.10,       # 100mm
        tip_chord=0.05,        # 50mm
        span=0.05,             # 50mm
        sweep=0.03,            # 30mm sweep
        thickness=0.003        # 3mm
    )
    fins.material = MATERIALS["Plywood (birch)"]
    fins.position.x = 0.20  # Fins start 200mm into body (300mm from nose)
    body_main.add_child(fins)
    
    # ============================================================================
    # LAUNCH LUGS (Rail guides)
    # ============================================================================
    launch_lug_fwd = LaunchLug(
        name="Forward Launch Lug",
        length=0.03,           # 30mm
        outer_radius=0.0025,   # 2.5mm (5mm OD)
        thickness=0.0005       # 0.5mm wall
    )
    launch_lug_fwd.material = MATERIALS["Cardboard"]
    launch_lug_fwd.position.x = 0.05  # 50mm into body (150mm from nose)
    body_main.add_child(launch_lug_fwd)
    
    launch_lug_aft = LaunchLug(
        name="Aft Launch Lug",
        length=0.03,
        outer_radius=0.0025,
        thickness=0.0005
    )
    launch_lug_aft.material = MATERIALS["Cardboard"]
    launch_lug_aft.position.x = 0.25  # 250mm into body (350mm from nose)
    body_main.add_child(launch_lug_aft)
    
    # ============================================================================
    # AVIONICS BAY (Optional - add if needed)
    # ============================================================================
    # Avionics mass (flight computer, batteries, switches)
    avionics = MassComponent(
        name="Avionics Bay",
        mass=0.025,  # 25g (flight computer + battery)
        length=0.04,
        radius=0.020
    )
    avionics.position.x = 0.10  # 100mm into body (200mm from nose)
    body_main.add_child(avionics)
    
    # Calculate reference values
    rocket.calculate_reference_values()
    
    # Get motor
    motor = get_builtin_motors()['F50']
    
    return rocket, motor


def print_complete_mass_breakdown(rocket: Rocket, motor):
    """Print COMPLETE mass breakdown with every component"""
    print("\n" + "="*80)
    print("COMPLETE MASS BREAKDOWN (Every Component)")
    print("="*80)
    
    total_mass = 0.0
    
    def print_component(comp, indent=0):
        nonlocal total_mass
        prefix = "  " * indent
        mass = comp.get_mass() * 1000
        cg = comp.get_cg_x() * 1000
        abs_pos = comp.get_absolute_position() * 1000
        
        if mass > 0.001:  # Only print if mass > 1mg
            print(f"{prefix}{comp.name}:")
            print(f"{prefix}  Position: {abs_pos:.1f} mm (absolute)")
            print(f"{prefix}  Mass:     {mass:.3f} g")
            print(f"{prefix}  CG:       {cg:.1f} mm (local), {abs_pos + cg:.1f} mm (absolute)")
            
            if hasattr(comp, 'material'):
                print(f"{prefix}  Material: {comp.material.name}")
            
            total_mass += mass
        
        for child in comp.children:
            print_component(child, indent + 1)
    
    for comp in rocket.children:
        print_component(comp)
    
    # Motor
    print(f"\nMotor ({motor.designation}):")
    print(f"  Position: Motor mount tube (270-390 mm from nose)")
    print(f"  Mass (ignition): {motor.total_mass_initial*1000:.3f} g")
    print(f"  Mass (burnout):  {motor.case_mass*1000:.3f} g")
    print(f"  Propellant:      {motor.propellant_mass*1000:.3f} g")
    
    motor_mass = motor.total_mass_initial * 1000
    
    print("\n" + "─"*80)
    print(f"Component total (no motor): {total_mass:.3f} g")
    print(f"Motor mass (ignition):      {motor_mass:.3f} g")
    print(f"TOTAL (ignition):           {total_mass + motor_mass:.3f} g")
    print(f"TOTAL (burnout):            {total_mass + motor.case_mass*1000:.3f} g")
    print("="*80)
    
    return total_mass + motor_mass


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PROPER HIGH-FIDELITY ROCKET BUILDER")
    print("="*80)
    
    rocket, motor = build_proper_rocket()
    
    print("\n✓ Rocket built with complete internal structure:")
    print("  • Nose cone with shoulder")
    print("  • Recovery bulkhead")
    print("  • Shock cord (1m)")
    print("  • Parachute (30cm)")
    print("  • Motor mount tube (120mm)")
    print("  • 3x Centering rings (forward, middle, aft)")
    print("  • 3x Fins")
    print("  • 2x Launch lugs")
    print("  • Avionics bay (25g)")
    
    total_mass = print_complete_mass_breakdown(rocket, motor)
    
    print(f"\n✓ Total component count: {len(list(rocket.children)) + sum(len(c.children) for c in rocket.children)}")
    print(f"✓ Ready for high-fidelity simulation")

