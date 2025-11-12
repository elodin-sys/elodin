"""
Complete flight analysis with full telemetry and visualization.
Shows rocket build, flight trajectory, and ALL flight data.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from proper_rocket_builder import build_proper_rocket
from openrocket_sim_3dof import Simulator3DOF
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere
from openrocket_components import (NoseCone, BodyTube, InnerTube, CenteringRing, 
                                   Bulkhead, TrapezoidFinSet, Parachute, ShockCord,
                                   LaunchLug, MassComponent)


def visualize_complete_rocket(rocket, motor):
    """Detailed rocket visualization with ALL components"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[:, 0])  # Main rocket view (full height)
    ax_mass = fig.add_subplot(gs[0, 1])  # Mass distribution
    ax_stab = fig.add_subplot(gs[1, 1])  # Stability
    ax_comp = fig.add_subplot(gs[0, 2])  # Component table
    ax_thrust = fig.add_subplot(gs[1, 2])  # Thrust curve
    
    # ===== MAIN ROCKET VIEW =====
    ax_main.set_title("Complete Rocket Assembly\n(All Internal Components)", fontsize=14, fontweight='bold')
    ax_main.set_xlabel("Position from nose tip (mm)")
    ax_main.set_ylabel("Radius (mm)")
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(0, color='k', linewidth=0.5)
    ax_main.set_aspect('equal')
    
    aero = RocketAerodynamics(rocket)
    
    # Colors for different component types
    colors = {
        'nose': '#FF6B6B',
        'body': '#4ECDC4',
        'fin': '#FFE66D',
        'motor_mount': '#95E1D3',
        'centering': '#A8E6CF',
        'bulkhead': '#8B7355',
        'parachute': '#FFB6C1',
        'shock_cord': '#DDA15E',
        'lug': '#C9ADA7',
        'avionics': '#7209B7',
        'motor': '#6C757D'
    }
    
    # Draw all components
    def draw_component(comp, parent_x=0):
        abs_x = (parent_x + comp.position.x) * 1000  # mm
        
        if isinstance(comp, NoseCone):
            L = comp.length * 1000
            R = comp.base_radius * 1000
            theta = np.linspace(0, np.pi, 50)
            x = abs_x + L * (1 - np.cos(theta))
            y = R * np.sin(theta)
            ax_main.fill(np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]]), 
                        color=colors['nose'], edgecolor='darkred', linewidth=2, alpha=0.8, label='Nose Cone')
            ax_main.text(abs_x + L/2, R + 15, comp.name, ha='center', fontsize=8, fontweight='bold')
        
        elif isinstance(comp, BodyTube):
            L = comp.length * 1000
            R = comp.outer_radius * 1000
            rect_top = Rectangle((abs_x, 0), L, R, facecolor=colors['body'], edgecolor='darkblue', linewidth=2, alpha=0.7)
            rect_bot = Rectangle((abs_x, -R), L, R, facecolor=colors['body'], edgecolor='darkblue', linewidth=2, alpha=0.7)
            ax_main.add_patch(rect_top)
            ax_main.add_patch(rect_bot)
            ax_main.text(abs_x + L/2, R + 8, comp.name, ha='center', fontsize=8)
        
        elif isinstance(comp, InnerTube):
            L = comp.length * 1000
            R = comp.outer_radius * 1000
            rect_top = Rectangle((abs_x, 0), L, R, facecolor=colors['motor_mount'], edgecolor='darkgreen', linewidth=2, alpha=0.9)
            rect_bot = Rectangle((abs_x, -R), L, R, facecolor=colors['motor_mount'], edgecolor='darkgreen', linewidth=2, alpha=0.9)
            ax_main.add_patch(rect_top)
            ax_main.add_patch(rect_bot)
            ax_main.text(abs_x + L/2, R + 5, "Motor Mount", ha='center', fontsize=7, fontweight='bold')
        
        elif isinstance(comp, CenteringRing):
            R_out = comp.outer_radius * 1000
            R_in = comp.inner_radius * 1000
            T = comp.thickness * 1000
            # Draw as thick annulus
            for y_sign in [1, -1]:
                ring = Rectangle((abs_x, y_sign * R_in), T, y_sign * (R_out - R_in), 
                               facecolor=colors['centering'], edgecolor='brown', linewidth=1.5, alpha=0.9)
                ax_main.add_patch(ring)
        
        elif isinstance(comp, Bulkhead):
            R = comp.radius * 1000
            T = comp.thickness * 1000
            rect_top = Rectangle((abs_x, 0), T, R, facecolor=colors['bulkhead'], edgecolor='black', linewidth=2, alpha=0.9)
            rect_bot = Rectangle((abs_x, -R), T, R, facecolor=colors['bulkhead'], edgecolor='black', linewidth=2, alpha=0.9)
            ax_main.add_patch(rect_top)
            ax_main.add_patch(rect_bot)
        
        elif isinstance(comp, TrapezoidFinSet):
            root = comp.root_chord * 1000
            tip = comp.tip_chord * 1000
            span = comp.span * 1000
            sweep = comp.sweep * 1000
            
            fin_x = [abs_x, abs_x + sweep, abs_x + sweep + tip, abs_x + root, abs_x]
            fin_y_top = [0, span, span, 0, 0]
            fin_y_bot = [0, -span, -span, 0, 0]
            
            ax_main.fill(fin_x, fin_y_top, color=colors['fin'], edgecolor='saddlebrown', linewidth=2, alpha=0.8)
            ax_main.fill(fin_x, fin_y_bot, color=colors['fin'], edgecolor='saddlebrown', linewidth=2, alpha=0.8)
            ax_main.text(abs_x + root/2, span + 10, f"{comp.fin_count}x Fins", ha='center', fontsize=8, fontweight='bold')
        
        elif isinstance(comp, Parachute):
            # Draw as circle icon
            circle = Circle((abs_x, 0), 8, facecolor=colors['parachute'], edgecolor='red', linewidth=2, alpha=0.9)
            ax_main.add_patch(circle)
            ax_main.plot([abs_x, abs_x], [-5, 5], 'r--', linewidth=1.5)
        
        elif isinstance(comp, ShockCord):
            # Draw as wavy line
            x_cord = np.linspace(abs_x, abs_x + 20, 20)
            y_cord = 3 * np.sin(x_cord)
            ax_main.plot(x_cord, y_cord, color=colors['shock_cord'], linewidth=3, alpha=0.8)
        
        elif isinstance(comp, LaunchLug):
            L = comp.length * 1000
            R = comp.outer_radius * 1000
            rect = Rectangle((abs_x, 0), L, R, facecolor=colors['lug'], edgecolor='black', linewidth=1, alpha=0.8)
            ax_main.add_patch(rect)
        
        elif isinstance(comp, MassComponent):
            L = comp.length * 1000
            R = comp.radius * 1000
            rect_top = Rectangle((abs_x, 0), L, R, facecolor=colors['avionics'], edgecolor='purple', linewidth=2, alpha=0.9)
            rect_bot = Rectangle((abs_x, -R), L, R, facecolor=colors['avionics'], edgecolor='purple', linewidth=2, alpha=0.9)
            ax_main.add_patch(rect_top)
            ax_main.add_patch(rect_bot)
            if "Avionics" in comp.name:
                ax_main.text(abs_x + L/2, 0, "AVIONICS", ha='center', va='center', fontsize=6, fontweight='bold', color='white')
        
        # Recurse for children
        for child in comp.children:
            draw_component(child, abs_x / 1000)
    
    # Draw rocket components
    for comp in rocket.children:
        draw_component(comp)
    
    # Draw motor (find motor mount position)
    motor_mount_pos = 0.0
    for comp in rocket.children:
        for child in comp.children:
            if isinstance(child, InnerTube) and "Motor Mount" in child.name:
                motor_mount_pos = (comp.position.x + child.position.x) * 1000
                break
    
    motor_x = motor_mount_pos
    motor_L = motor.length * 1000
    motor_R = motor.diameter / 2 * 1000
    motor_rect_top = Rectangle((motor_x, 0), motor_L, motor_R, facecolor=colors['motor'], edgecolor='black', linewidth=2, alpha=0.8)
    motor_rect_bot = Rectangle((motor_x, -motor_R), motor_L, motor_R, facecolor=colors['motor'], edgecolor='black', linewidth=2, alpha=0.8)
    ax_main.add_patch(motor_rect_top)
    ax_main.add_patch(motor_rect_bot)
    ax_main.text(motor_x + motor_L/2, 0, motor.designation, ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # CG and CP markers
    comp_mass = rocket.get_total_mass()
    comp_cg = rocket.get_total_cg()
    motor_mass = motor.get_mass(0.0)
    motor_cg_abs = motor_mount_pos/1000 + motor.get_cg(0.0)
    cg = (comp_mass * comp_cg + motor_mass * motor_cg_abs) / (comp_mass + motor_mass) * 1000
    cp = aero.calculate_cp(0.0) * 1000
    
    ax_main.axvline(cg, color='blue', linestyle='--', linewidth=2.5, label=f'CG = {cg:.1f} mm', alpha=0.8)
    ax_main.axvline(cp, color='red', linestyle='--', linewidth=2.5, label=f'CP = {cp:.1f} mm', alpha=0.8)
    ax_main.scatter([cg], [0], color='blue', s=300, marker='o', zorder=10, edgecolor='darkblue', linewidth=3)
    ax_main.scatter([cp], [0], color='red', s=300, marker='x', zorder=10, linewidth=4)
    
    ax_main.legend(loc='upper left', fontsize=9)
    ax_main.set_xlim(-20, rocket.reference_length * 1000 + 50)
    ax_main.set_ylim(-70, 70)
    
    # ===== MASS DISTRIBUTION =====
    ax_mass.set_title("Mass Distribution", fontsize=12, fontweight='bold')
    ax_mass.set_xlabel("Position (mm)")
    ax_mass.set_ylabel("Mass (g)")
    ax_mass.grid(True, alpha=0.3)
    
    # Collect component masses
    comp_data = []
    def collect_masses(comp, parent_x=0):
        abs_x = (parent_x + comp.position.x) * 1000
        mass = comp.get_mass() * 1000
        if mass > 0.1:
            comp_data.append((abs_x, mass, comp.name[:20]))
        for child in comp.children:
            collect_masses(child, abs_x/1000)
    
    for comp in rocket.children:
        collect_masses(comp)
    
    positions = [d[0] for d in comp_data]
    masses = [d[1] for d in comp_data]
    
    ax_mass.bar(positions, masses, width=20, color='steelblue', edgecolor='navy', linewidth=1.5, alpha=0.8)
    ax_mass.axvline(cg, color='blue', linestyle='--', linewidth=2, label=f'CG = {cg:.1f} mm')
    ax_mass.legend()
    
    # ===== STABILITY =====
    ax_stab.set_title("Static Stability Margin", fontsize=12, fontweight='bold')
    ax_stab.set_xlabel("Time (s)")
    ax_stab.set_ylabel("Static Margin (cal)")
    ax_stab.grid(True, alpha=0.3)
    
    times = np.linspace(0, motor.burn_time + 2, 100)
    margins = []
    for t in times:
        motor_mass_t = motor.get_mass(t)
        motor_cg_t = motor_mount_pos/1000 + motor.get_cg(t)
        total_mass = comp_mass + motor_mass_t
        cg_t = (comp_mass * comp_cg + motor_mass_t * motor_cg_t) / total_mass
        margin = (cp/1000 - cg_t) / rocket.reference_diameter
        margins.append(margin)
    
    ax_stab.plot(times, margins, 'b-', linewidth=2.5)
    ax_stab.fill_between(times, 0, margins, alpha=0.3)
    ax_stab.axhline(1.0, color='orange', linestyle='--', alpha=0.7, label='Min (1.0)')
    ax_stab.axhline(2.0, color='green', linestyle='--', alpha=0.7, label='Optimal (2.0)')
    ax_stab.axvspan(0, motor.burn_time, alpha=0.1, color='red', label='Motor burn')
    ax_stab.legend(fontsize=8)
    
    # ===== COMPONENT TABLE =====
    ax_comp.axis('off')
    ax_comp.set_title("Component List", fontsize=12, fontweight='bold')
    
    table_data = [["Component", "Mass (g)", "Position (mm)"]]
    for pos, mass, name in sorted(comp_data, key=lambda x: x[0]):
        table_data.append([name, f"{mass:.2f}", f"{pos:.0f}"])
    table_data.append(["Motor", f"{motor.total_mass_initial*1000:.2f}", f"{motor_mount_pos:.0f}"])
    
    table = ax_comp.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    
    # Header styling
    for i in range(3):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')
    
    # ===== THRUST CURVE =====
    ax_thrust.set_title("Motor Thrust Curve", fontsize=12, fontweight='bold')
    ax_thrust.set_xlabel("Time (s)")
    ax_thrust.set_ylabel("Thrust (N)")
    ax_thrust.grid(True, alpha=0.3)
    
    thrust_times = np.linspace(0, motor.burn_time, 100)
    thrusts = [motor.get_thrust(t) for t in thrust_times]
    
    ax_thrust.plot(thrust_times, thrusts, 'r-', linewidth=2.5)
    ax_thrust.fill_between(thrust_times, 0, thrusts, alpha=0.3, color='red')
    ax_thrust.axhline(motor.average_thrust, color='darkred', linestyle='--', linewidth=2, 
                     label=f'Avg: {motor.average_thrust:.1f} N')
    ax_thrust.legend()
    
    plt.tight_layout()
    plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/complete_rocket_build.png', dpi=150, bbox_inches='tight')
    print("\n✓ Complete rocket build saved: complete_rocket_build.png")
    plt.close()


def run_complete_simulation(rocket, motor):
    """Run simulation and return complete telemetry"""
    sim = Simulator3DOF(rocket, motor)
    sim.rail_length = 1.0
    sim.dt = 0.01
    
    print("\nRunning complete flight simulation...")
    history = sim.run()
    summary = sim.get_summary()
    
    return history, summary


def plot_complete_telemetry(history, summary, rocket, motor):
    """Plot ALL flight telemetry data"""
    
    atm = ISAAtmosphere()
    aero = RocketAerodynamics(rocket)
    
    # Extract data
    times = np.array([s.time for s in history])
    altitudes = np.array([s.z for s in history])
    velocities = np.array([s.vz for s in history])
    
    # Calculate additional parameters
    accelerations = np.zeros(len(history))
    for i in range(1, len(history)):
        dt = history[i].time - history[i-1].time
        if dt > 0:
            accelerations[i] = (history[i].vz - history[i-1].vz) / dt / 9.81  # in g's
    
    # Mach number
    machs = []
    for s in history:
        props = atm.get_properties(max(0, s.z))
        mach = abs(s.vz) / props['speed_of_sound']
        machs.append(mach)
    machs = np.array(machs)
    
    # Dynamic pressure
    dynamic_pressures = []
    for s in history:
        props = atm.get_properties(max(0, s.z))
        q = 0.5 * props['density'] * s.vz**2
        dynamic_pressures.append(q / 1000)  # kPa
    dynamic_pressures = np.array(dynamic_pressures)
    
    # Drag force
    drag_forces = []
    for s in history:
        props = atm.get_properties(max(0, s.z))
        v = abs(s.vz)
        if v > 0.1:
            mach = v / props['speed_of_sound']
            cd = aero.calculate_cd(mach, v, props['density'], props['viscosity'], 0.0)
            ref_area = math.pi * (rocket.reference_diameter / 2)**2
            q = 0.5 * props['density'] * v**2
            drag = cd * q * ref_area
            drag_forces.append(drag)
        else:
            drag_forces.append(0)
    drag_forces = np.array(drag_forces)
    
    # Thrust
    thrusts = []
    for s in history:
        if s.motor_ignited and s.motor_time <= motor.burn_time:
            thrusts.append(motor.get_thrust(s.motor_time))
        else:
            thrusts.append(0)
    thrusts = np.array(thrusts)
    
    # Mass
    masses = []
    comp_mass = rocket.get_total_mass()
    for s in history:
        motor_mass = motor.get_mass(s.motor_time)
        masses.append((comp_mass + motor_mass) * 1000)  # grams
    masses = np.array(masses)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # ===== ALTITUDE =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, altitudes, 'b-', linewidth=2.5)
    ax1.axhline(summary['max_altitude'], color='r', linestyle='--', alpha=0.6, 
                label=f"Apogee: {summary['max_altitude']:.0f} m")
    ax1.axvline(summary['apogee_time'], color='r', linestyle=':', alpha=0.6)
    ax1.fill_between(times, 0, altitudes, alpha=0.2)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Altitude (m)', fontsize=11)
    ax1.set_title('Altitude vs Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # ===== VELOCITY =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(times, velocities, 'g-', linewidth=2.5)
    ax2.axhline(summary['max_velocity'], color='r', linestyle='--', alpha=0.6,
                label=f"Max: {summary['max_velocity']:.0f} m/s")
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(times, 0, velocities, alpha=0.2, color='green')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Velocity (m/s)', fontsize=11)
    ax2.set_title('Velocity vs Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # ===== ACCELERATION =====
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(times, accelerations, 'r-', linewidth=2.5)
    ax3.axhline(max(accelerations), color='darkred', linestyle='--', alpha=0.6,
                label=f"Max: {max(accelerations):.1f} g")
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.axvspan(0, motor.burn_time, alpha=0.1, color='orange', label='Motor burn')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Acceleration (g)', fontsize=11)
    ax3.set_title('Acceleration vs Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # ===== MACH NUMBER =====
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(times, machs, 'purple', linewidth=2.5)
    ax4.axhline(max(machs), color='darkviolet', linestyle='--', alpha=0.6,
                label=f"Max: Mach {max(machs):.3f}")
    ax4.axhline(1.0, color='orange', linestyle=':', alpha=0.5, label='Mach 1.0')
    ax4.fill_between(times, 0, machs, alpha=0.2, color='purple')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Mach Number', fontsize=11)
    ax4.set_title('Mach Number vs Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # ===== DYNAMIC PRESSURE =====
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(times, dynamic_pressures, 'orange', linewidth=2.5)
    ax5.axhline(max(dynamic_pressures), color='darkorange', linestyle='--', alpha=0.6,
                label=f"Max Q: {max(dynamic_pressures):.1f} kPa")
    ax5.fill_between(times, 0, dynamic_pressures, alpha=0.2, color='orange')
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Dynamic Pressure (kPa)', fontsize=11)
    ax5.set_title('Dynamic Pressure vs Time', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # ===== FORCES =====
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(times, thrusts, 'r-', linewidth=2.5, label='Thrust')
    ax6.plot(times, drag_forces, 'b-', linewidth=2.5, label='Drag')
    ax6.axvline(motor.burn_time, color='red', linestyle=':', alpha=0.5, label='Burnout')
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Force (N)', fontsize=11)
    ax6.set_title('Forces vs Time', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    
    # ===== MASS =====
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(times, masses, 'brown', linewidth=2.5)
    ax7.axhline(masses[0], color='darkred', linestyle='--', alpha=0.6, label=f"Ignition: {masses[0]:.1f} g")
    burnout_idx = int(motor.burn_time / 0.01)
    if burnout_idx < len(masses):
        ax7.axhline(masses[burnout_idx], color='darkgreen', linestyle='--', alpha=0.6, 
                   label=f"Burnout: {masses[burnout_idx]:.1f} g")
    ax7.fill_between(times, 0, masses, alpha=0.2, color='brown')
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('Mass (g)', fontsize=11)
    ax7.set_title('Rocket Mass vs Time', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=9)
    
    # ===== TRAJECTORY (Altitude vs Range) =====
    ax8 = fig.add_subplot(gs[2, 1])
    # For 3DOF vertical, range is always 0 - show as vertical line
    range_values = np.zeros_like(altitudes)
    ax8.plot(range_values, altitudes, 'b-', linewidth=2.5)
    ax8.scatter([0], [summary['max_altitude']], color='red', s=200, marker='*', 
               label=f"Apogee: {summary['max_altitude']:.0f} m", zorder=5, edgecolor='darkred', linewidth=2)
    ax8.scatter([0], [0], color='green', s=200, marker='o', 
               label="Launch/Landing", zorder=5, edgecolor='darkgreen', linewidth=2)
    ax8.set_xlabel('Downrange Distance (m)', fontsize=11)
    ax8.set_ylabel('Altitude (m)', fontsize=11)
    ax8.set_title('Flight Trajectory (Vertical)', fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)
    ax8.set_xlim(-50, 50)
    
    # ===== SUMMARY TABLE =====
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    ax9.set_title('Flight Summary', fontsize=13, fontweight='bold')
    
    summary_data = [
        ["Parameter", "Value"],
        ["Max Altitude", f"{summary['max_altitude']:.1f} m ({summary['max_altitude']*3.281:.0f} ft)"],
        ["Apogee Time", f"{summary['apogee_time']:.2f} s"],
        ["Max Velocity", f"{summary['max_velocity']:.1f} m/s (Mach {max(machs):.3f})"],
        ["Max Acceleration", f"{max(accelerations):.1f} g"],
        ["Max Dynamic Press.", f"{max(dynamic_pressures):.1f} kPa"],
        ["Burnout Altitude", f"{altitudes[burnout_idx]:.1f} m" if burnout_idx < len(altitudes) else "N/A"],
        ["Coast Time", f"{summary['apogee_time'] - motor.burn_time:.2f} s"],
        ["Flight Time", f"{summary['flight_time']:.1f} s"],
        ["Rail Exit Vel.", f"{velocities[int(0.11/0.01)]:.1f} m/s" if len(velocities) > 11 else "N/A"],
    ]
    
    table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')
    
    plt.suptitle('COMPLETE FLIGHT TELEMETRY ANALYSIS', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/complete_flight_telemetry.png', dpi=150, bbox_inches='tight')
    print("✓ Complete flight telemetry saved: complete_flight_telemetry.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*100)
    print(" " * 30 + "COMPLETE FLIGHT ANALYSIS")
    print("="*100)
    
    # Build proper rocket
    rocket, motor = build_proper_rocket()
    
    # Visualize rocket build
    print("\n1. Creating complete rocket visualization...")
    visualize_complete_rocket(rocket, motor)
    
    # Run simulation
    print("\n2. Running flight simulation...")
    history, summary = run_complete_simulation(rocket, motor)
    
    # Plot telemetry
    print("\n3. Generating complete telemetry plots...")
    plot_complete_telemetry(history, summary, rocket, motor)
    
    print("\n" + "="*100)
    print("✅ COMPLETE ANALYSIS FINISHED")
    print("="*100)
    print("\nGenerated files:")
    print("  • complete_rocket_build.png - Full rocket assembly with all components")
    print("  • complete_flight_telemetry.png - 9-panel flight data visualization")
    print("\nFlight Summary:")
    print(f"  Apogee:        {summary['max_altitude']:.1f} m ({summary['max_altitude']*3.281:.0f} ft)")
    print(f"  Max Velocity:  {summary['max_velocity']:.1f} m/s")
    print(f"  Flight Time:   {summary['flight_time']:.1f} s")
    print("="*100)

