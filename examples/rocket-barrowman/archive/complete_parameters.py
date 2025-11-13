"""
COMPLETE OpenRocket parameter suite.
Every coefficient, derivative, ratio, and metric that OpenRocket calculates.
"""

import math
import numpy as np
from proper_rocket_builder import build_proper_rocket
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere
from openrocket_sim_3dof import Simulator3DOF


def calculate_all_parameters(rocket, motor):
    """Calculate EVERY parameter OpenRocket provides"""
    
    aero = RocketAerodynamics(rocket)
    atm = ISAAtmosphere()
    
    # Find motor mount position
    motor_mount_pos = 0.0
    def find_mount(comp):
        nonlocal motor_mount_pos
        from openrocket_components import BodyTube, InnerTube
        if isinstance(comp, InnerTube) and "Motor Mount" in comp.name:
            motor_mount_pos = comp.get_absolute_position()
        for child in comp.children:
            find_mount(child)
    find_mount(rocket)
    
    params = {}
    
    # =========================================================================
    # 1. GEOMETRIC PARAMETERS
    # =========================================================================
    params['geometry'] = {
        'total_length': rocket.reference_length,  # m
        'max_diameter': rocket.reference_diameter,  # m
        'reference_area': math.pi * (rocket.reference_diameter / 2)**2,  # m²
        'fineness_ratio': rocket.reference_length / rocket.reference_diameter,
    }
    
    # =========================================================================
    # 2. MASS PROPERTIES (Time-varying)
    # =========================================================================
    times = [0.0, motor.burn_time/2, motor.burn_time, motor.burn_time + 5]
    labels = ['ignition', 'mid_burn', 'burnout', 'coast']
    
    params['mass'] = {}
    for t, label in zip(times, labels):
        comp_mass = rocket.get_total_mass()
        comp_cg = rocket.get_total_cg()
        motor_mass = motor.get_mass(t)
        motor_cg_abs = motor_mount_pos + motor.get_cg(t)
        
        total_mass = comp_mass + motor_mass
        total_cg = (comp_mass * comp_cg + motor_mass * motor_cg_abs) / total_mass
        
        # Inertia (simplified - cylinder approximation)
        L = rocket.reference_length
        R = rocket.reference_diameter / 2
        Ixx = Iyy = (1/12) * total_mass * L**2 + (1/4) * total_mass * R**2
        Izz = (1/2) * total_mass * R**2
        
        params['mass'][label] = {
            'total_mass_kg': total_mass,
            'total_mass_g': total_mass * 1000,
            'cg_position_m': total_cg,
            'cg_position_mm': total_cg * 1000,
            'moment_of_inertia_pitch_yaw': Ixx,  # kg·m²
            'moment_of_inertia_roll': Izz,  # kg·m²
            'radius_of_gyration_pitch': math.sqrt(Ixx / total_mass),  # m
            'radius_of_gyration_roll': math.sqrt(Izz / total_mass),  # m
        }
    
    # =========================================================================
    # 3. AERODYNAMIC COEFFICIENTS (Mach-dependent)
    # =========================================================================
    machs = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    params['aero_coefficients'] = {}
    
    for mach in machs:
        cp = aero.calculate_cp(mach)
        cn_alpha = aero.calculate_cn_alpha(mach)
        
        # Get atmospheric properties at sea level
        props = atm.get_properties(0)
        v = mach * props['speed_of_sound']
        cd = aero.calculate_cd(mach, v, props['density'], props['viscosity'], 0.0)
        
        # Stability derivatives
        # CN_alpha is already normal force coefficient derivative
        # CM_alpha (pitching moment coefficient derivative)
        cg_ign = params['mass']['ignition']['cg_position_m']
        cm_alpha = -cn_alpha * (cp - cg_ign) / rocket.reference_diameter
        
        # Damping derivatives (approximate)
        # These depend on velocity and require complex calculations
        # Using simplified estimates based on OpenRocket methodology
        S_ref = params['geometry']['reference_area']
        d_ref = rocket.reference_diameter
        
        # Pitch damping derivative CN_q (per radian of pitch rate)
        # Approximation: CN_q ≈ -2 * CN_alpha * (CP - CG) / L
        cn_q = -2 * cn_alpha * (cp - cg_ign) / rocket.reference_length
        
        # Roll damping (from fins)
        cn_p = -0.01  # Small roll damping
        
        params['aero_coefficients'][f'mach_{mach:.1f}'] = {
            'mach_number': mach,
            'cn_alpha': cn_alpha,  # Normal force coefficient derivative (per radian)
            'cn_alpha_deg': cn_alpha * 180 / math.pi,  # per degree
            'cm_alpha': cm_alpha,  # Pitching moment coefficient derivative
            'cd_total': cd,  # Total drag coefficient
            'cd_zero_lift': cd,  # Drag at zero AoA
            'cp_position_m': cp,
            'cp_position_mm': cp * 1000,
            'cn_q': cn_q,  # Pitch damping derivative
            'cm_q': -cn_q,  # Pitching moment damping
            'cn_p': cn_p,  # Roll damping
        }
    
    # =========================================================================
    # 4. STABILITY PARAMETERS
    # =========================================================================
    cp = aero.calculate_cp(0.0)
    cg_ign = params['mass']['ignition']['cg_position_m']
    cg_burn = params['mass']['burnout']['cg_position_m']
    
    params['stability'] = {
        'static_margin_ignition_cal': (cp - cg_ign) / rocket.reference_diameter,
        'static_margin_burnout_cal': (cp - cg_burn) / rocket.reference_diameter,
        'static_margin_ignition_percent': ((cp - cg_ign) / rocket.reference_length) * 100,
        'static_margin_burnout_percent': ((cp - cg_burn) / rocket.reference_length) * 100,
        'cg_travel_mm': (cg_burn - cg_ign) * 1000,
        'cp_position_mm': cp * 1000,
        'cg_to_cp_distance_mm': (cp - cg_ign) * 1000,
    }
    
    # =========================================================================
    # 5. DRAG BREAKDOWN (at Mach 0.3)
    # =========================================================================
    props = atm.get_properties(0)
    v = 0.3 * props['speed_of_sound']  # ~100 m/s
    
    # Component drag contributions (approximate)
    ref_area = params['geometry']['reference_area']
    
    # Nose drag
    cd_nose = aero.calculator.nose_pressure_drag(rocket.children[0], 0.3)
    
    # Body skin friction
    body_length = rocket.reference_length
    body_diameter = rocket.reference_diameter
    cd_body = aero.calculator.body_skin_friction_drag(
        body_length, body_diameter, v, props['density'], props['viscosity']
    )
    
    # Fin drag
    from openrocket_components import TrapezoidFinSet
    fin_component = None
    for comp in rocket.children:
        for child in comp.children:
            if isinstance(child, TrapezoidFinSet):
                fin_component = child
                break
    
    cd_fin = 0.0
    if fin_component:
        cd_fin = aero.calculator.fin_drag(fin_component, 0.3, ref_area, v, props['density'], props['viscosity'])
    
    # Base drag
    cd_base = aero.calculator.base_drag(0.3, ref_area, ref_area)
    
    cd_total = aero.calculate_cd(0.3, v, props['density'], props['viscosity'], 0.0)
    
    params['drag_breakdown'] = {
        'cd_nose': cd_nose,
        'cd_body_friction': cd_body,
        'cd_fins': cd_fin,
        'cd_base': cd_base,
        'cd_total': cd_total,
        'cd_pressure': cd_nose + cd_base,
        'cd_skin_friction': cd_body + cd_fin,
    }
    
    # =========================================================================
    # 6. FLIGHT PERFORMANCE (from simulation)
    # =========================================================================
    sim = Simulator3DOF(rocket, motor)
    sim.rail_length = 1.0
    sim.dt = 0.01
    history = sim.run()
    summary = sim.get_summary()
    
    # Extract detailed performance
    max_altitude = summary['max_altitude']
    max_velocity = summary['max_velocity']
    apogee_time = summary['apogee_time']
    
    # Find burnout conditions
    burnout_idx = 0
    for i, s in enumerate(history):
        if s.motor_time >= motor.burn_time:
            burnout_idx = i
            break
    
    burnout_state = history[burnout_idx] if burnout_idx < len(history) else history[-1]
    
    # Find max acceleration
    max_accel_g = 0.0
    max_accel_time = 0.0
    for i in range(1, len(history)):
        if history[i].time < motor.burn_time:
            dv = history[i].vz - history[i-1].vz
            dt = history[i].time - history[i-1].time
            if dt > 0:
                accel_g = (dv / dt) / 9.81
                if accel_g > max_accel_g:
                    max_accel_g = accel_g
                    max_accel_time = history[i].time
    
    # Rail exit
    rail_exit_velocity = 0.0
    rail_exit_time = 0.0
    for s in history:
        if s.z >= sim.rail_length:
            rail_exit_velocity = abs(s.vz)
            rail_exit_time = s.time
            break
    
    # Descent rate
    descent_rate = 0.0
    for i in range(len(history)-1, 0, -1):
        if history[i].parachute_deployed and history[i].vz < -1.0:
            descent_rate = abs(history[i].vz)
            break
    
    # Dynamic pressure at max Q
    max_q = 0.0
    max_q_time = 0.0
    max_q_altitude = 0.0
    for s in history:
        props_alt = atm.get_properties(max(0, s.z))
        q = 0.5 * props_alt['density'] * s.vz**2
        if q > max_q:
            max_q = q
            max_q_time = s.time
            max_q_altitude = s.z
    
    params['flight_performance'] = {
        'apogee_m': max_altitude,
        'apogee_ft': max_altitude * 3.281,
        'apogee_time_s': apogee_time,
        'max_velocity_ms': max_velocity,
        'max_velocity_fps': max_velocity * 3.281,
        'max_velocity_mph': max_velocity * 2.237,
        'max_mach': max_velocity / atm.get_speed_of_sound(0),
        'max_acceleration_g': max_accel_g,
        'max_acceleration_time_s': max_accel_time,
        'rail_exit_velocity_ms': rail_exit_velocity,
        'rail_exit_time_s': rail_exit_time,
        'burnout_altitude_m': burnout_state.z,
        'burnout_velocity_ms': abs(burnout_state.vz),
        'coast_time_s': apogee_time - motor.burn_time,
        'descent_rate_ms': descent_rate,
        'descent_rate_fps': descent_rate * 3.281,
        'flight_time_s': summary['flight_time'],
        'max_dynamic_pressure_pa': max_q,
        'max_q_time_s': max_q_time,
        'max_q_altitude_m': max_q_altitude,
    }
    
    # =========================================================================
    # 7. MOTOR PERFORMANCE
    # =========================================================================
    params['motor'] = {
        'designation': motor.designation,
        'manufacturer': motor.manufacturer,
        'diameter_mm': motor.diameter * 1000,
        'length_mm': motor.length * 1000,
        'total_mass_g': motor.total_mass_initial * 1000,
        'propellant_mass_g': motor.propellant_mass * 1000,
        'case_mass_g': motor.case_mass * 1000,
        'burn_time_s': motor.burn_time,
        'total_impulse_ns': motor.total_impulse,
        'average_thrust_n': motor.average_thrust,
        'peak_thrust_n': max(motor.thrusts),
        'impulse_class': motor.impulse_class,
        'thrust_to_weight_ignition': motor.average_thrust / (params['mass']['ignition']['total_mass_kg'] * 9.81),
        'thrust_to_weight_burnout': motor.average_thrust / (params['mass']['burnout']['total_mass_kg'] * 9.81),
    }
    
    # =========================================================================
    # 8. ATMOSPHERIC CONDITIONS (at key altitudes)
    # =========================================================================
    altitudes = [0, 100, 500, 1000, max_altitude]
    params['atmosphere'] = {}
    
    for alt in altitudes:
        props = atm.get_properties(alt)
        params['atmosphere'][f'altitude_{int(alt)}m'] = {
            'altitude_m': alt,
            'temperature_k': props['temperature'],
            'temperature_c': props['temperature'] - 273.15,
            'pressure_pa': props['pressure'],
            'density_kgm3': props['density'],
            'speed_of_sound_ms': props['speed_of_sound'],
            'dynamic_viscosity_pas': props['viscosity'],
            'kinematic_viscosity_m2s': props['viscosity'] / props['density'],
        }
    
    return params, history


def print_all_parameters(params):
    """Print ALL parameters in organized format"""
    
    print("\n" + "="*100)
    print(" " * 35 + "COMPLETE PARAMETER SUITE")
    print("="*100)
    
    # 1. GEOMETRY
    print("\n📐 GEOMETRIC PARAMETERS")
    print("─"*100)
    g = params['geometry']
    print(f"Total Length:          {g['total_length']*1000:.2f} mm  ({g['total_length']*39.37:.3f} in)")
    print(f"Maximum Diameter:      {g['max_diameter']*1000:.2f} mm  ({g['max_diameter']*39.37:.3f} in)")
    print(f"Reference Area:        {g['reference_area']*10000:.2f} cm²")
    print(f"Fineness Ratio:        {g['fineness_ratio']:.2f}")
    
    # 2. MASS PROPERTIES
    print("\n⚖️  MASS PROPERTIES (Time-Varying)")
    print("─"*100)
    print(f"{'Condition':<20} {'Mass (g)':<15} {'CG (mm)':<15} {'Ixx=Iyy (kg·m²)':<20} {'Izz (kg·m²)':<15}")
    print("─"*100)
    for label in ['ignition', 'mid_burn', 'burnout', 'coast']:
        m = params['mass'][label]
        print(f"{label.capitalize():<20} {m['total_mass_g']:<15.3f} {m['cg_position_mm']:<15.1f} "
              f"{m['moment_of_inertia_pitch_yaw']:<20.6f} {m['moment_of_inertia_roll']:<15.6f}")
    
    # 3. AERODYNAMIC COEFFICIENTS
    print("\n🌬️  AERODYNAMIC COEFFICIENTS vs MACH")
    print("─"*100)
    print(f"{'Mach':<8} {'CN_α (/rad)':<12} {'CM_α':<12} {'CD':<10} {'CP (mm)':<12} {'CN_q':<12} {'CM_q':<12}")
    print("─"*100)
    for key in sorted(params['aero_coefficients'].keys()):
        a = params['aero_coefficients'][key]
        print(f"{a['mach_number']:<8.1f} {a['cn_alpha']:<12.4f} {a['cm_alpha']:<12.4f} "
              f"{a['cd_total']:<10.4f} {a['cp_position_mm']:<12.1f} {a['cn_q']:<12.4f} {a['cm_q']:<12.4f}")
    
    # 4. STABILITY
    print("\n📏 STABILITY PARAMETERS")
    print("─"*100)
    s = params['stability']
    print(f"Static Margin (ignition):  {s['static_margin_ignition_cal']:.3f} cal  ({s['static_margin_ignition_percent']:.1f}%)")
    print(f"Static Margin (burnout):   {s['static_margin_burnout_cal']:.3f} cal  ({s['static_margin_burnout_percent']:.1f}%)")
    print(f"CG Travel:                 {s['cg_travel_mm']:.2f} mm")
    print(f"CP Position:               {s['cp_position_mm']:.1f} mm from nose tip")
    print(f"CG to CP Distance:         {s['cg_to_cp_distance_mm']:.1f} mm")
    
    # 5. DRAG BREAKDOWN
    print("\n🎯 DRAG BREAKDOWN (at Mach 0.3)")
    print("─"*100)
    d = params['drag_breakdown']
    print(f"Total CD:              {d['cd_total']:.4f}")
    print(f"  Nose Pressure:       {d['cd_nose']:.4f}  ({d['cd_nose']/d['cd_total']*100:.1f}%)")
    print(f"  Body Friction:       {d['cd_body_friction']:.4f}  ({d['cd_body_friction']/d['cd_total']*100:.1f}%)")
    print(f"  Fin Drag:            {d['cd_fins']:.4f}  ({d['cd_fins']/d['cd_total']*100:.1f}%)")
    print(f"  Base Drag:           {d['cd_base']:.4f}  ({d['cd_base']/d['cd_total']*100:.1f}%)")
    print(f"Pressure Drag Total:   {d['cd_pressure']:.4f}")
    print(f"Friction Drag Total:   {d['cd_skin_friction']:.4f}")
    
    # 6. FLIGHT PERFORMANCE
    print("\n🚀 FLIGHT PERFORMANCE")
    print("─"*100)
    f = params['flight_performance']
    print(f"Apogee:                {f['apogee_m']:.1f} m  ({f['apogee_ft']:.0f} ft)  @ {f['apogee_time_s']:.2f} s")
    print(f"Max Velocity:          {f['max_velocity_ms']:.1f} m/s  ({f['max_velocity_fps']:.0f} ft/s, {f['max_velocity_mph']:.1f} mph)")
    print(f"Max Mach:              {f['max_mach']:.3f}")
    print(f"Max Acceleration:      {f['max_acceleration_g']:.1f} g  @ {f['max_acceleration_time_s']:.3f} s")
    print(f"Rail Exit Velocity:    {f['rail_exit_velocity_ms']:.1f} m/s  @ {f['rail_exit_time_s']:.3f} s")
    print(f"Burnout Altitude:      {f['burnout_altitude_m']:.1f} m")
    print(f"Burnout Velocity:      {f['burnout_velocity_ms']:.1f} m/s")
    print(f"Coast Time:            {f['coast_time_s']:.2f} s")
    print(f"Descent Rate:          {f['descent_rate_ms']:.1f} m/s  ({f['descent_rate_fps']:.1f} ft/s)")
    print(f"Flight Time:           {f['flight_time_s']:.1f} s")
    print(f"Max Dynamic Pressure:  {f['max_dynamic_pressure_pa']:.0f} Pa  @ {f['max_q_altitude_m']:.1f} m, {f['max_q_time_s']:.2f} s")
    
    # 7. MOTOR
    print("\n⚙️  MOTOR PERFORMANCE")
    print("─"*100)
    m = params['motor']
    print(f"Designation:           {m['designation']} ({m['manufacturer']})")
    print(f"Dimensions:            {m['diameter_mm']:.1f} mm x {m['length_mm']:.1f} mm")
    print(f"Total Mass:            {m['total_mass_g']:.2f} g")
    print(f"Propellant Mass:       {m['propellant_mass_g']:.2f} g")
    print(f"Case Mass:             {m['case_mass_g']:.2f} g")
    print(f"Burn Time:             {m['burn_time_s']:.2f} s")
    print(f"Total Impulse:         {m['total_impulse_ns']:.1f} N·s  (Class {m['impulse_class']})")
    print(f"Average Thrust:        {m['average_thrust_n']:.1f} N")
    print(f"Peak Thrust:           {m['peak_thrust_n']:.1f} N")
    print(f"T/W (ignition):        {m['thrust_to_weight_ignition']:.2f}")
    print(f"T/W (burnout):         {m['thrust_to_weight_burnout']:.2f}")
    
    # 8. ATMOSPHERE
    print("\n🌍 ATMOSPHERIC CONDITIONS")
    print("─"*100)
    print(f"{'Altitude':<15} {'Temp (°C)':<12} {'Pressure (Pa)':<18} {'Density (kg/m³)':<18} {'Sound (m/s)':<15}")
    print("─"*100)
    for key in sorted(params['atmosphere'].keys(), key=lambda x: params['atmosphere'][x]['altitude_m']):
        a = params['atmosphere'][key]
        print(f"{a['altitude_m']:<15.0f} {a['temperature_c']:<12.1f} {a['pressure_pa']:<18.0f} "
              f"{a['density_kgm3']:<18.4f} {a['speed_of_sound_ms']:<15.1f}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    rocket, motor = build_proper_rocket()
    
    print("\nCalculating ALL parameters (this may take a minute for simulation)...")
    params, history = calculate_all_parameters(rocket, motor)
    
    print_all_parameters(params)
    
    print("\n✅ COMPLETE PARAMETER SUITE CALCULATED")
    print(f"   Total parameters: {sum(len(v) if isinstance(v, dict) else 1 for v in params.values())}")
    print("   Ready for OpenRocket comparison!")

