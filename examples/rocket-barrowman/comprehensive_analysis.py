"""
Comprehensive rocket analysis and visualization.
First-order, second-order analysis matching OpenRocket.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
from openrocket_components import *
from openrocket_motor import Motor, get_builtin_motors
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere
from openrocket_sim_3dof import Simulator3DOF


def build_rocket():
    """Build the test rocket"""
    rocket = Rocket("Test Rocket - Aerotech F50")
    
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
    body.position.x = 0.10
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
    fins.position.x = 0.30
    fins.material = MATERIALS["Plywood (birch)"]
    rocket.add_child(fins)
    
    # Parachute - 30cm
    chute = Parachute(
        name="Parachute",
        diameter=0.30,
        cd=0.75,
        material_density=50.0
    )
    chute.position.x = 0.05
    chute.deployment_event = "APOGEE"
    rocket.add_child(chute)
    
    rocket.calculate_reference_values()
    
    motor = get_builtin_motors()['F50']
    
    return rocket, motor


def print_first_order_analysis(rocket: Rocket, motor: Motor):
    """First-order analysis - basic geometry and mass"""
    print("\n" + "="*80)
    print("FIRST-ORDER ANALYSIS: GEOMETRY AND MASS")
    print("="*80)
    
    print("\n📐 ROCKET GEOMETRY")
    print("─"*80)
    print(f"Overall length:        {rocket.reference_length*1000:.1f} mm")
    print(f"Maximum diameter:      {rocket.reference_diameter*1000:.1f} mm")
    print(f"Reference area:        {math.pi*(rocket.reference_diameter/2)**2*10000:.2f} cm²")
    
    # Component breakdown
    print("\n🔧 COMPONENT BREAKDOWN")
    print("─"*80)
    print(f"{'Component':<25} {'Length (mm)':<15} {'Diameter (mm)':<15} {'Mass (g)':<12}")
    print("─"*80)
    
    total_comp_mass = 0.0
    for comp in rocket.children:
        mass = comp.get_mass() * 1000
        total_comp_mass += mass
        
        if isinstance(comp, NoseCone):
            length = comp.length * 1000
            diameter = comp.base_radius * 2 * 1000
            print(f"{comp.name:<25} {length:<15.1f} {diameter:<15.1f} {mass:<12.2f}")
        elif isinstance(comp, BodyTube):
            length = comp.length * 1000
            diameter = comp.outer_radius * 2 * 1000
            print(f"{comp.name:<25} {length:<15.1f} {diameter:<15.1f} {mass:<12.2f}")
        elif isinstance(comp, TrapezoidFinSet):
            span = comp.span * 1000
            root = comp.root_chord * 1000
            print(f"{comp.name:<25} {f'{comp.fin_count}x fins':<15} {span:<15.1f} {mass:<12.2f}")
        elif isinstance(comp, Parachute):
            diameter = comp.diameter * 1000
            print(f"{comp.name:<25} {'N/A':<15} {diameter:<15.1f} {mass:<12.2f}")
    
    print("─"*80)
    print(f"{'Component total':<55} {total_comp_mass:<12.2f}")
    
    print(f"\n⚙️  MOTOR: {motor.designation}")
    print("─"*80)
    print(f"Manufacturer:          {motor.manufacturer}")
    print(f"Diameter:              {motor.diameter*1000:.1f} mm")
    print(f"Length:                {motor.length*1000:.1f} mm")
    print(f"Total mass (ignition): {motor.total_mass_initial*1000:.2f} g")
    print(f"Propellant mass:       {motor.propellant_mass*1000:.2f} g")
    print(f"Case mass (burnout):   {motor.case_mass*1000:.2f} g")
    print(f"Burn time:             {motor.burn_time:.2f} s")
    print(f"Total impulse:         {motor.total_impulse:.1f} N·s")
    print(f"Average thrust:        {motor.average_thrust:.1f} N")
    print(f"Impulse class:         {motor.impulse_class}")
    
    # Total masses
    mass_ignition = total_comp_mass + motor.total_mass_initial * 1000
    mass_burnout = total_comp_mass + motor.case_mass * 1000
    
    print("\n📊 TOTAL MASS")
    print("─"*80)
    print(f"At ignition (T+0):     {mass_ignition:.2f} g  ({mass_ignition/1000:.4f} kg)")
    print(f"At burnout (T+{motor.burn_time:.1f}s):  {mass_burnout:.2f} g  ({mass_burnout/1000:.4f} kg)")
    print(f"Mass loss during burn: {mass_ignition - mass_burnout:.2f} g")


def print_second_order_analysis(rocket: Rocket, motor: Motor):
    """Second-order analysis - CG, CP, stability"""
    print("\n" + "="*80)
    print("SECOND-ORDER ANALYSIS: CENTER OF GRAVITY AND STABILITY")
    print("="*80)
    
    aero = RocketAerodynamics(rocket)
    
    # Find motor mount position
    motor_mount_pos = 0.0
    def find_mount(comp):
        nonlocal motor_mount_pos
        if isinstance(comp, BodyTube) and comp.motor_mount:
            motor_mount_pos = comp.get_absolute_position()
        for child in comp.children:
            find_mount(child)
    find_mount(rocket)
    
    # CG calculations at different times
    comp_mass = rocket.get_total_mass()
    comp_cg = rocket.get_total_cg()
    
    times = [0.0, motor.burn_time/2, motor.burn_time, motor.burn_time + 5.0]
    time_labels = ["Ignition (T+0)", f"Mid-burn (T+{motor.burn_time/2:.1f}s)", 
                   f"Burnout (T+{motor.burn_time:.1f}s)", "Coast (T+7s)"]
    
    print("\n⚖️  CENTER OF GRAVITY TRAVEL")
    print("─"*80)
    print(f"{'Time':<20} {'Motor Mass (g)':<18} {'Total Mass (g)':<18} {'CG (mm)':<15}")
    print("─"*80)
    
    cgs = []
    for t, label in zip(times, time_labels):
        motor_mass = motor.get_mass(t) * 1000
        motor_cg_local = motor.get_cg(t)
        motor_cg_abs = (motor_mount_pos + motor_cg_local) * 1000
        
        total_mass = (comp_mass + motor.get_mass(t)) * 1000
        total_cg = (comp_mass * comp_cg + motor.get_mass(t) * (motor_mount_pos + motor_cg_local)) / (comp_mass + motor.get_mass(t)) * 1000
        
        cgs.append(total_cg / 1000)
        print(f"{label:<20} {motor_mass:<18.2f} {total_mass:<18.2f} {total_cg:<15.1f}")
    
    print(f"\nCG travel: {(max(cgs) - min(cgs))*1000:.1f} mm")
    
    # CP and stability
    cp = aero.calculate_cp(0.0) * 1000
    cn_alpha = aero.calculate_cn_alpha(0.0)
    
    print("\n🎯 CENTER OF PRESSURE AND AERODYNAMICS")
    print("─"*80)
    print(f"CP location:           {cp:.1f} mm from nose tip")
    print(f"CN_alpha:              {cn_alpha:.3f} per radian")
    print(f"                       {cn_alpha * 180 / math.pi:.3f} per degree")
    
    print("\n📏 STATIC STABILITY MARGIN")
    print("─"*80)
    print(f"{'Condition':<20} {'CG (mm)':<15} {'CP (mm)':<15} {'Margin (cal)':<15} {'Status':<15}")
    print("─"*80)
    
    for i, (t, label) in enumerate(zip(times[:3], time_labels[:3])):
        cg_mm = cgs[i] * 1000
        margin = (cp/1000 - cgs[i]) / rocket.reference_diameter
        
        if margin >= 2.0:
            status = "✓ Very Stable"
        elif margin >= 1.0:
            status = "✓ Stable"
        elif margin >= 0.5:
            status = "⚠ Marginal"
        else:
            status = "✗ Unstable"
        
        print(f"{label:<20} {cg_mm:<15.1f} {cp:<15.1f} {margin:<15.2f} {status:<15}")
    
    print("\n💡 Recommended static margin: 1.0 - 2.5 calibers")


def print_third_order_analysis(rocket: Rocket, motor: Motor):
    """Third-order analysis - drag, thrust, flight performance"""
    print("\n" + "="*80)
    print("THIRD-ORDER ANALYSIS: AERODYNAMIC DRAG AND FLIGHT PERFORMANCE")
    print("="*80)
    
    aero = RocketAerodynamics(rocket)
    atm = ISAAtmosphere()
    props = atm.get_properties(0)
    
    print("\n🌬️  DRAG COEFFICIENT vs VELOCITY")
    print("─"*80)
    print(f"{'Velocity (m/s)':<18} {'Mach':<12} {'CD':<12} {'Drag @ 0m (N)':<15}")
    print("─"*80)
    
    ref_area = math.pi * (rocket.reference_diameter / 2)**2
    
    for v in [10, 30, 50, 75, 100, 150, 200, 250, 300]:
        mach = v / props['speed_of_sound']
        cd = aero.calculate_cd(mach, v, props['density'], props['viscosity'], 0.0)
        drag = 0.5 * props['density'] * v**2 * ref_area * cd
        print(f"{v:<18} {mach:<12.3f} {cd:<12.4f} {drag:<15.2f}")
    
    print("\n🚀 THRUST CURVE")
    print("─"*80)
    print(f"{'Time (s)':<15} {'Thrust (N)':<15} {'Cumulative Impulse (N·s)':<25}")
    print("─"*80)
    
    cumulative_impulse = 0.0
    prev_t = 0.0
    for i in range(0, 25):
        t = i * motor.burn_time / 24
        thrust = motor.get_thrust(t)
        
        if i > 0:
            dt = t - prev_t
            avg_thrust = (thrust + motor.get_thrust(prev_t)) / 2
            cumulative_impulse += avg_thrust * dt
        
        print(f"{t:<15.2f} {thrust:<15.2f} {cumulative_impulse:<25.2f}")
        prev_t = t
    
    # Performance estimates
    print("\n📈 PREDICTED FLIGHT PERFORMANCE (3DOF Simulation)")
    print("─"*80)
    
    sim = Simulator3DOF(rocket, motor)
    sim.rail_length = 1.0
    sim.dt = 0.01
    
    history = sim.run()
    summary = sim.get_summary()
    
    print(f"Maximum altitude:      {summary['max_altitude']:.1f} m ({summary['max_altitude']*3.281:.0f} ft)")
    print(f"Maximum velocity:      {summary['max_velocity']:.1f} m/s ({summary['max_velocity']*2.237:.1f} mph)")
    print(f"Time to apogee:        {summary['apogee_time']:.2f} s")
    print(f"Total flight time:     {summary['flight_time']:.1f} s")
    
    # Rail exit velocity
    for s in history:
        if s.z >= sim.rail_length:
            print(f"Rail exit velocity:    {abs(s.vz):.1f} m/s (at {s.time:.3f} s)")
            break
    
    # Max acceleration
    max_accel = 0.0
    for i in range(1, len(history)):
        if history[i].time < motor.burn_time:
            dv = history[i].vz - history[i-1].vz
            dt = history[i].time - history[i-1].time
            accel = abs(dv / dt) / 9.81
            max_accel = max(max_accel, accel)
    
    print(f"Maximum acceleration:  {max_accel:.1f} g")
    
    # Descent rate with parachute
    for i in range(len(history)-1, 0, -1):
        if history[i].parachute_deployed and history[i].vz < -1.0:
            print(f"Descent rate (chute):  {abs(history[i].vz):.1f} m/s")
            break
    
    return history, summary


def visualize_rocket(rocket: Rocket, motor: Motor):
    """Create detailed rocket visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== SUBPLOT 1: Rocket Assembly =====
    ax1.set_title("Rocket Assembly (Side View)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Position (mm)")
    ax1.set_ylabel("Radius (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_aspect('equal')
    
    aero = RocketAerodynamics(rocket)
    
    # Find motor mount
    motor_mount_pos = 0.0
    def find_mount(comp):
        nonlocal motor_mount_pos
        if isinstance(comp, BodyTube) and comp.motor_mount:
            motor_mount_pos = comp.get_absolute_position()
        for child in comp.children:
            find_mount(child)
    find_mount(rocket)
    
    # Draw components
    for comp in rocket.children:
        pos = comp.get_absolute_position() * 1000
        
        if isinstance(comp, NoseCone):
            # Draw nose cone
            L = comp.length * 1000
            R = comp.base_radius * 1000
            if comp.shape == NoseCone.Shape.CONICAL:
                nose_x = [pos, pos + L, pos]
                nose_y = [0, R, -R]
                ax1.fill(nose_x, nose_y, color='lightcoral', edgecolor='darkred', linewidth=2, label='Nose Cone')
            else:  # Ogive
                theta = np.linspace(0, np.pi, 50)
                x = pos + L * (1 - np.cos(theta))
                y = R * np.sin(theta)
                ax1.fill(np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]]), 
                        color='lightcoral', edgecolor='darkred', linewidth=2, label='Nose Cone')
        
        elif isinstance(comp, BodyTube):
            # Draw body tube
            L = comp.length * 1000
            R = comp.outer_radius * 1000
            rect_top = Rectangle((pos, 0), L, R, facecolor='lightblue', edgecolor='darkblue', linewidth=2)
            rect_bot = Rectangle((pos, -R), L, R, facecolor='lightblue', edgecolor='darkblue', linewidth=2)
            ax1.add_patch(rect_top)
            ax1.add_patch(rect_bot)
            if comp == rocket.children[1]:  # First body tube
                ax1.text(pos + L/2, R + 10, comp.name, ha='center', fontsize=9)
        
        elif isinstance(comp, TrapezoidFinSet):
            # Draw fins
            root = comp.root_chord * 1000
            tip = comp.tip_chord * 1000
            span = comp.span * 1000
            sweep = comp.sweep * 1000
            
            # Fin outline
            fin_x = [pos, pos + sweep, pos + sweep + tip, pos + root, pos]
            fin_y_top = [0, span, span, 0, 0]
            fin_y_bot = [0, -span, -span, 0, 0]
            
            ax1.fill(fin_x, fin_y_top, color='wheat', edgecolor='saddlebrown', linewidth=2, label='Fins (3x)')
            ax1.fill(fin_x, fin_y_bot, color='wheat', edgecolor='saddlebrown', linewidth=2)
            ax1.text(pos + root/2, span + 10, f"{comp.fin_count}x Fins", ha='center', fontsize=9)
    
    # Draw motor
    motor_x = motor_mount_pos * 1000
    motor_L = motor.length * 1000
    motor_R = motor.diameter / 2 * 1000
    motor_rect_top = Rectangle((motor_x, 0), motor_L, motor_R, facecolor='gray', edgecolor='black', linewidth=2, alpha=0.7)
    motor_rect_bot = Rectangle((motor_x, -motor_R), motor_L, motor_R, facecolor='gray', edgecolor='black', linewidth=2, alpha=0.7)
    ax1.add_patch(motor_rect_top)
    ax1.add_patch(motor_rect_bot)
    ax1.text(motor_x + motor_L/2, 0, motor.designation, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # CG and CP markers
    comp_mass = rocket.get_total_mass()
    comp_cg = rocket.get_total_cg()
    motor_mass_ign = motor.get_mass(0.0)
    motor_cg_ign = motor_mount_pos + motor.get_cg(0.0)
    cg_ign = (comp_mass * comp_cg + motor_mass_ign * motor_cg_ign) / (comp_mass + motor_mass_ign) * 1000
    
    cp = aero.calculate_cp(0.0) * 1000
    
    ax1.axvline(cg_ign, color='blue', linestyle='--', linewidth=2, label=f'CG (ignition) = {cg_ign:.1f} mm')
    ax1.axvline(cp, color='red', linestyle='--', linewidth=2, label=f'CP = {cp:.1f} mm')
    ax1.scatter([cg_ign], [0], color='blue', s=200, marker='o', zorder=10, edgecolor='darkblue', linewidth=2)
    ax1.scatter([cp], [0], color='red', s=200, marker='x', zorder=10, linewidth=3)
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(-20, rocket.reference_length * 1000 + 60)
    ax1.set_ylim(-80, 80)
    
    # ===== SUBPLOT 2: Mass Distribution =====
    ax2.set_title("Mass Distribution Along Rocket", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Position from nose tip (mm)")
    ax2.set_ylabel("Component Mass (g)")
    ax2.grid(True, alpha=0.3)
    
    positions = []
    masses = []
    labels = []
    
    for comp in rocket.children:
        pos = comp.get_absolute_position() * 1000
        mass = comp.get_mass() * 1000
        positions.append(pos)
        masses.append(mass)
        labels.append(comp.name)
    
    # Add motor
    positions.append(motor_mount_pos * 1000)
    masses.append(motor.total_mass_initial * 1000)
    labels.append(f"Motor ({motor.designation})")
    
    colors = ['lightcoral', 'lightblue', 'wheat', 'lightgreen', 'gray']
    ax2.bar(positions, masses, width=30, color=colors[:len(positions)], edgecolor='black', linewidth=1.5)
    
    for i, (pos, mass, label) in enumerate(zip(positions, masses, labels)):
        ax2.text(pos, mass + 2, f"{mass:.1f}g", ha='center', fontsize=8, fontweight='bold')
        ax2.text(pos, -5, label, ha='center', fontsize=7, rotation=0)
    
    ax2.axvline(cg_ign, color='blue', linestyle='--', linewidth=2, label=f'CG = {cg_ign:.1f} mm')
    ax2.legend()
    
    # ===== SUBPLOT 3: Stability Diagram =====
    ax3.set_title("Static Stability vs Time", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Static Margin (calibers)")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Minimum recommended (1.0)')
    ax3.axhline(2.0, color='green', linestyle='--', alpha=0.5, label='Optimal (2.0)')
    
    times = np.linspace(0, motor.burn_time + 2, 100)
    margins = []
    
    for t in times:
        motor_mass = motor.get_mass(t)
        motor_cg_abs = motor_mount_pos + motor.get_cg(t)
        total_mass = comp_mass + motor_mass
        total_cg = (comp_mass * comp_cg + motor_mass * motor_cg_abs) / total_mass
        margin = (cp/1000 - total_cg) / rocket.reference_diameter
        margins.append(margin)
    
    ax3.plot(times, margins, 'b-', linewidth=2, label='Static Margin')
    ax3.fill_between(times, 0, margins, alpha=0.3)
    ax3.axvspan(0, motor.burn_time, alpha=0.1, color='red', label='Motor burn')
    ax3.legend()
    ax3.set_ylim(0, max(margins) * 1.2)
    
    # ===== SUBPLOT 4: Thrust and Drag =====
    ax4.set_title("Thrust Curve", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Thrust (N)", color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3)
    
    thrust_times = []
    thrusts = []
    for i in range(100):
        t = i * motor.burn_time / 99
        thrust_times.append(t)
        thrusts.append(motor.get_thrust(t))
    
    ax4.plot(thrust_times, thrusts, 'r-', linewidth=2, label='Thrust')
    ax4.fill_between(thrust_times, 0, thrusts, alpha=0.3, color='red')
    
    # Add average thrust line
    ax4.axhline(motor.average_thrust, color='darkred', linestyle='--', linewidth=1.5, label=f'Average: {motor.average_thrust:.1f} N')
    
    ax4.legend(loc='upper right')
    ax4.set_xlim(0, motor.burn_time * 1.1)
    
    plt.tight_layout()
    plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/rocket_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: rocket_analysis.png")
    plt.close()


def main():
    """Run comprehensive analysis"""
    print("\n" + "="*80)
    print("  COMPREHENSIVE ROCKET ANALYSIS - OPENROCKET VALIDATION")
    print("="*80)
    
    rocket, motor = build_rocket()
    
    # First-order: Geometry and mass
    print_first_order_analysis(rocket, motor)
    
    # Second-order: CG, CP, stability
    print_second_order_analysis(rocket, motor)
    
    # Third-order: Drag, thrust, performance
    history, summary = print_third_order_analysis(rocket, motor)
    
    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION")
    print("="*80)
    visualize_rocket(rocket, motor)
    
    # Flight trajectory plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    times = [s.time for s in history]
    altitudes = [s.z for s in history]
    velocities = [abs(s.vz) for s in history]
    
    ax1.plot(times, altitudes, 'b-', linewidth=2)
    ax1.axhline(summary['max_altitude'], color='r', linestyle='--', alpha=0.5, label=f"Apogee: {summary['max_altitude']:.1f} m")
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Altitude vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(times, velocities, 'g-', linewidth=2)
    ax2.axhline(summary['max_velocity'], color='r', linestyle='--', alpha=0.5, label=f"Max: {summary['max_velocity']:.1f} m/s")
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title('Velocity vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/flight_trajectory.png', dpi=150)
    print("✓ Flight trajectory saved: flight_trajectory.png")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("  - rocket_analysis.png (4-panel rocket visualization)")
    print("  - flight_trajectory.png (altitude and velocity plots)")
    print("\nNext steps:")
    print("  1. Open validation_rocket.ork in OpenRocket")
    print("  2. Compare mass breakdown, CG, CP, static margin")
    print("  3. Run OpenRocket simulation and compare apogee/velocity")
    print("  4. Verify <5% error on all parameters")
    print("="*80)


if __name__ == "__main__":
    main()

