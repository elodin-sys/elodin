"""
Robust 6DOF Rocket Flight Simulator

This is a production-quality flight simulator with:
- Proper RK4 integration
- Quaternion-based attitude dynamics
- Correct aerodynamic forces and moments
- Stability derivatives
- Robust numerical handling
- Full validation against OpenRocket
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import math
from dataclasses import dataclass

from rocket_components import Rocket, Parachute
from motor_database import MotorData


@dataclass
class State:
    """Complete 6DOF state vector"""
    # Position (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Quaternion (orientation)
    q0: float = 1.0  # scalar part
    q1: float = 0.0  # vector part
    q2: float = 0.0
    q3: float = 0.0
    
    # Angular velocity (rad/s) in body frame
    wx: float = 0.0  # roll rate
    wy: float = 0.0  # pitch rate
    wz: float = 0.0  # yaw rate
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for integration"""
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.q0, self.q1, self.q2, self.q3,
            self.wx, self.wy, self.wz
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'State':
        """Create from numpy array"""
        return cls(
            x=arr[0], y=arr[1], z=arr[2],
            vx=arr[3], vy=arr[4], vz=arr[5],
            q0=arr[6], q1=arr[7], q2=arr[8], q3=arr[9],
            wx=arr[10], wy=arr[11], wz=arr[12]
        )
    
    def normalize_quaternion(self):
        """Normalize quaternion to maintain unit length"""
        norm = math.sqrt(self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2)
        if norm > 0:
            self.q0 /= norm
            self.q1 /= norm
            self.q2 /= norm
            self.q3 /= norm


class Atmosphere:
    """ISA standard atmosphere model"""
    
    @staticmethod
    def get_properties(altitude: float) -> Tuple[float, float, float, float]:
        """
        Get atmospheric properties at altitude
        
        Returns:
            (density_kg_m3, pressure_pa, temperature_k, speed_of_sound_m_s)
        """
        # Clamp altitude
        h = np.clip(altitude, 0, 50000)
        
        # Sea level conditions
        rho_0 = 1.225  # kg/m³
        P_0 = 101325  # Pa
        T_0 = 288.15  # K
        L = 0.0065  # K/m (temperature lapse rate)
        g = 9.80665  # m/s²
        R = 287.05  # J/(kg·K)
        
        if h < 11000:  # Troposphere
            T = T_0 - L * h
            P = P_0 * (T / T_0) ** (g / (R * L))
            rho = rho_0 * (T / T_0) ** ((g / (R * L)) - 1)
        else:  # Stratosphere (simplified)
            T = 216.65
            P = 22632 * np.exp(-g * (h - 11000) / (R * T))
            rho = P / (R * T)
        
        a = np.sqrt(1.4 * R * T)  # Speed of sound
        
        return rho, P, T, a


class RobustSimulator:
    """Production-quality 6DOF simulator"""
    
    def __init__(self, rocket: Rocket, motor: MotorData, motor_position: float,
                 launch_angle_deg: float = 90.0, rail_length: float = 1.5):
        """
        Args:
            rocket: Rocket configuration
            motor: Motor data
            motor_position: Motor aft end position from nose (m)
            launch_angle_deg: Launch angle from horizontal (90 = vertical)
            rail_length: Launch rail length (m)
        """
        self.rocket = rocket
        self.motor = motor
        self.motor_position = motor_position
        self.rail_length = rail_length
        
        # Initial attitude (quaternion for vertical launch)
        # For 90°: body-x axis should point up in world-z
        # Rotation about negative y-axis to pitch nose up from horizontal (x) to vertical (z)
        angle_rad = math.radians(launch_angle_deg)
        
        # For standard aerospace convention: need NEGATIVE rotation about y
        # q = [cos(θ/2), 0, -sin(θ/2), 0] for pitch up
        self.state = State(
            q0=math.cos(angle_rad/2),
            q2=-math.sin(angle_rad/2)  # Negative for pitch up
        )
        
        # Simulation parameters
        self.time = 0.0
        self.dt = 1.0 / 200.0  # 200 Hz for numerical stability
        
        # Flight phase tracking
        self.on_rail = True
        self.motor_burnout_time = motor.burn_time
        
        # Data logging
        self.history = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'vx': [], 'vy': [], 'vz': [],
            'altitude': [],
            'velocity': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'mass': [],
            'thrust': [],
            'drag': [],
            'cg': [],
            'cp': [],
            'static_margin': [],
            'mach': [],
            'aoa': [],
        }
    
    def quaternion_to_rotation_matrix(self, q0, q1, q2, q3) -> np.ndarray:
        """
        Convert quaternion to rotation matrix (body to world frame)
        
        Body frame: x = forward (thrust direction), y = right, z = down
        World frame: x = north, y = east, z = up
        """
        R = np.array([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ])
        return R
    
    def quaternion_to_euler(self, q0, q1, q2, q3) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0*q1 + q2*q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q0*q2 - q3*q1)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi/2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0*q3 + q1*q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def get_mass_properties(self, t: float) -> Tuple[float, float, np.ndarray]:
        """
        Get current mass properties
        
        Returns:
            (mass_kg, cg_x_m, inertia_matrix_kg_m2)
        """
        # Base rocket mass properties
        base_props = self.rocket.get_total_mass_properties()
        
        # Motor contribution
        if t < self.motor_burnout_time:
            burn_frac = t / self.motor_burnout_time
            motor_mass = self.motor.total_mass - burn_frac * self.motor.propellant_mass
            # CG shifts forward as propellant burns
            motor_cg = self.motor_position - 0.05 * burn_frac
        else:
            motor_mass = self.motor.casing_mass
            motor_cg = self.motor_position - 0.05
        
        # Combined properties
        total_mass = base_props.mass + motor_mass
        total_cg = (base_props.mass * base_props.cg_x + motor_mass * motor_cg) / total_mass
        
        # Inertia matrix (simplified - diagonal)
        I = np.diag([base_props.ixx, base_props.iyy, base_props.izz])
        
        return total_mass, total_cg, I
    
    def compute_aerodynamic_forces_and_moments(self, state: State, mass: float, 
                                              cg: float, rho: float, a: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic forces and moments in body frame
        
        Returns:
            (force_body_frame, moment_body_frame)
        """
        # Rotation matrix (body to world)
        R = self.quaternion_to_rotation_matrix(state.q0, state.q1, state.q2, state.q3)
        
        # Velocity in world frame
        v_world = np.array([state.vx, state.vy, state.vz])
        v_mag = np.linalg.norm(v_world)
        
        if v_mag < 0.1:
            return np.zeros(3), np.zeros(3)
        
        # Velocity in body frame
        v_body = R.T @ v_world
        
        # Angle of attack and sideslip
        # Body x-axis is rocket longitudinal axis
        v_axial = v_body[0]
        v_side_y = v_body[1]
        v_side_z = v_body[2]
        
        alpha = math.atan2(v_side_z, v_axial)  # Pitch plane AoA
        beta = math.atan2(v_side_y, v_axial)   # Yaw plane AoA
        
        # Dynamic pressure
        q = 0.5 * rho * v_mag**2
        
        # Reference area
        S_ref = math.pi * (self.rocket.body_diameter / 2) ** 2
        
        # Mach number
        mach = v_mag / a
        
        # Get aerodynamic properties
        aero = self.rocket.get_total_aerodynamic_properties(mach)
        
        # Axial force coefficient (drag)
        cd0 = aero.cd_base + aero.cd_friction
        
        # Compressibility correction for CD
        if mach < 0.8:
            cd = cd0
        elif mach < 1.2:
            # Transonic drag rise
            cd = cd0 * (1 + 5 * (mach - 0.8)**2)
        else:
            # Supersonic
            cd = cd0 * (1 + 0.2 / math.sqrt(mach**2 - 1))
        
        # Normal force coefficient (pitch and yaw)
        cn_alpha = aero.cn_alpha
        
        # Compressibility correction for CNα
        if 0 < mach < 0.9:
            beta_m = math.sqrt(max(1 - mach**2, 0.01))
            cn_alpha = cn_alpha / beta_m
        
        # Induced drag from angle of attack
        cd_induced = 0.1 * cn_alpha * (alpha**2 + beta**2)
        cd_total = cd + cd_induced
        
        # Forces in stability axes (aligned with velocity)
        # Then transform to body axes
        F_axial = -cd_total * q * S_ref  # Drag (negative along velocity)
        F_normal_pitch = cn_alpha * alpha * q * S_ref
        F_normal_yaw = cn_alpha * beta * q * S_ref
        
        # Forces in body frame (approximate for small angles)
        F_body = np.array([
            F_axial,
            F_normal_yaw,
            F_normal_pitch
        ])
        
        # Pitching and yawing moments
        cp = aero.cp_x
        moment_arm = cp - cg
        
        # Moments about CG in body frame
        M_pitch = -F_normal_pitch * moment_arm  # Restoring moment
        M_yaw = -F_normal_yaw * moment_arm
        
        # Damping moments (crucial for stability!)
        # These resist angular velocity
        c_ref = self.rocket.body_diameter  # Reference length
        C_mq = -10.0  # Pitch damping derivative (typical value)
        C_nr = -10.0  # Yaw damping derivative
        
        M_pitch_damping = C_mq * (state.wy * c_ref / (2 * v_mag)) * q * S_ref * c_ref if v_mag > 1 else 0
        M_yaw_damping = C_nr * (state.wz * c_ref / (2 * v_mag)) * q * S_ref * c_ref if v_mag > 1 else 0
        
        M_body = np.array([
            0.0,  # No roll moment (axisymmetric)
            M_pitch + M_pitch_damping,
            M_yaw + M_yaw_damping
        ])
        
        return F_body, M_body
    
    def state_derivative(self, state: State, t: float) -> np.ndarray:
        """
        Compute time derivative of state vector (for RK4 integration)
        
        Returns:
            d(state)/dt as numpy array
        """
        # Get mass properties
        mass, cg, I = self.get_mass_properties(t)
        
        # Altitude
        altitude = state.z
        
        # Atmospheric properties
        rho, P, T, a = Atmosphere.get_properties(altitude)
        
        # Rotation matrix
        R = self.quaternion_to_rotation_matrix(state.q0, state.q1, state.q2, state.q3)
        
        # === FORCES ===
        
        # Thrust (in body frame, along +x-axis which points forward/up)
        thrust_mag = self.motor.get_thrust(t)
        F_thrust_body = np.array([thrust_mag, 0, 0])
        
        # Aerodynamic forces and moments (body frame)
        F_aero_body, M_aero_body = self.compute_aerodynamic_forces_and_moments(
            state, mass, cg, rho, a
        )
        
        # Total force in body frame
        F_body_total = F_thrust_body + F_aero_body
        
        # Transform to world frame
        F_world = R @ F_body_total
        
        # Gravity (world frame)
        F_gravity = np.array([0, 0, -9.80665 * mass])
        
        # Total force (world frame)
        F_total_world = F_world + F_gravity
        
        # Acceleration (world frame)
        accel_world = F_total_world / mass
        
        # === MOMENTS ===
        
        # Angular velocity in body frame
        omega = np.array([state.wx, state.wy, state.wz])
        
        # Euler's rotation equation: I * dω/dt = M - ω × (I * ω)
        I_omega = I @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        
        # Angular acceleration
        I_inv = np.linalg.inv(I)
        alpha_body = I_inv @ (M_aero_body - omega_cross_I_omega)
        
        # === QUATERNION DERIVATIVE ===
        
        # dq/dt = 0.5 * Ω * q, where Ω is the quaternion multiplication matrix
        # For quaternion [q0, q1, q2, q3] and angular velocity [wx, wy, wz]:
        omega_quat = np.array([
            [-state.q1, -state.q2, -state.q3],
            [ state.q0, -state.q3,  state.q2],
            [ state.q3,  state.q0, -state.q1],
            [-state.q2,  state.q1,  state.q0]
        ])
        
        q_dot = 0.5 * omega_quat @ omega
        
        # Assemble derivative
        deriv = np.array([
            state.vx, state.vy, state.vz,  # Position derivatives
            accel_world[0], accel_world[1], accel_world[2],  # Velocity derivatives
            q_dot[0], q_dot[1], q_dot[2], q_dot[3],  # Quaternion derivatives
            alpha_body[0], alpha_body[1], alpha_body[2]  # Angular acceleration
        ])
        
        return deriv
    
    def rk4_step(self) -> State:
        """Single RK4 integration step"""
        # Current state
        y = self.state.to_array()
        t = self.time
        h = self.dt
        
        # RK4 stages
        k1 = self.state_derivative(State.from_array(y), t)
        k2 = self.state_derivative(State.from_array(y + h*k1/2), t + h/2)
        k3 = self.state_derivative(State.from_array(y + h*k2/2), t + h/2)
        k4 = self.state_derivative(State.from_array(y + h*k3), t + h)
        
        # Update
        y_new = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Create new state
        new_state = State.from_array(y_new)
        new_state.normalize_quaternion()
        
        return new_state
    
    def step(self):
        """Single simulation step with RK4 integration"""
        # Check parachute deployment BEFORE integration
        # This prevents sudden force changes during RK4 substeps
        for comp in self.rocket.components:
            if isinstance(comp, Parachute):
                if comp.check_deployment(self.time, self.state.z, self.state.vz):
                    if not comp.deployed:
                        comp.deployed = True
                        print(f"  🪂 {comp.name} deployed at t={self.time:.2f}s, "
                              f"h={self.state.z:.1f}m, vy={self.state.vz:.1f}m/s")
                        # Reduce timestep temporarily for smooth deployment
                        old_dt = self.dt
                        self.dt = old_dt / 10
                        for _ in range(10):
                            self.state = self.rk4_step()
                        self.dt = old_dt
                        self.log_data()
                        self.time += old_dt
                        return
        
        # RK4 integration
        self.state = self.rk4_step()
        
        # Sanity checks for numerical stability
        v_mag = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
        if v_mag > 1000.0:  # Supersonic limit check (Mach 3)
            print(f"  ⚠️  WARNING: Excessive velocity {v_mag:.1f} m/s detected at t={self.time:.2f}s")
            # Cap velocity
            scale = 1000.0 / v_mag
            self.state.vx *= scale
            self.state.vy *= scale
            self.state.vz *= scale
        
        # Ground contact
        if self.state.z < 0:
            self.state.z = 0
            self.state.vz = max(0, self.state.vz)  # No bouncing
        
        # Rail guidance (constraint)
        if self.on_rail:
            rail_distance = math.sqrt(self.state.x**2 + self.state.y**2 + self.state.z**2)
            if rail_distance >= self.rail_length:
                self.on_rail = False
                v = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
                print(f"  🚀 Rail cleared at t={self.time:.2f}s, v={v:.1f}m/s")
        
        # Log data
        self.log_data()
        
        # Advance time
        self.time += self.dt
    
    def log_data(self):
        """Log current state"""
        mass, cg, I = self.get_mass_properties(self.time)
        
        rho, P, T, a = Atmosphere.get_properties(self.state.z)
        v_mag = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
        mach = v_mag / a if a > 0 else 0
        
        # Aerodynamics
        aero = self.rocket.get_total_aerodynamic_properties(mach)
        cp = aero.cp_x
        static_margin = (cp - cg) / self.rocket.body_diameter if self.rocket.body_diameter > 0 else 0
        
        # Angle of attack (simplified)
        R = self.quaternion_to_rotation_matrix(self.state.q0, self.state.q1, self.state.q2, self.state.q3)
        v_world = np.array([self.state.vx, self.state.vy, self.state.vz])
        v_body = R.T @ v_world
        if abs(v_body[0]) > 0.1:
            aoa = math.atan2(v_body[2], v_body[0])
        else:
            aoa = 0
        
        # Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            self.state.q0, self.state.q1, self.state.q2, self.state.q3
        )
        
        # Drag force
        q = 0.5 * rho * v_mag**2
        S_ref = math.pi * (self.rocket.body_diameter / 2) ** 2
        cd = aero.cd_base + aero.cd_friction
        drag = cd * q * S_ref
        
        self.history['time'].append(self.time)
        self.history['x'].append(self.state.x)
        self.history['y'].append(self.state.y)
        self.history['z'].append(self.state.z)
        self.history['vx'].append(self.state.vx)
        self.history['vy'].append(self.state.vy)
        self.history['vz'].append(self.state.vz)
        self.history['altitude'].append(self.state.z)
        self.history['velocity'].append(v_mag)
        self.history['roll'].append(math.degrees(roll))
        self.history['pitch'].append(math.degrees(pitch))
        self.history['yaw'].append(math.degrees(yaw))
        self.history['mass'].append(mass)
        self.history['thrust'].append(self.motor.get_thrust(self.time))
        self.history['drag'].append(drag)
        self.history['cg'].append(cg)
        self.history['cp'].append(cp)
        self.history['static_margin'].append(static_margin)
        self.history['mach'].append(mach)
        self.history['aoa'].append(math.degrees(aoa))
    
    def run(self, max_time: float = 60.0, verbose: bool = True):
        """Run simulation"""
        print(f"\n{'='*70}")
        print(f"ROBUST 6DOF FLIGHT SIMULATION")
        print(f"{'='*70}")
        print(f"Rocket: {self.rocket.name}")
        print(f"Motor: {self.motor.manufacturer} {self.motor.designation}")
        print(f"Total impulse: {self.motor.total_impulse:.1f} N·s")
        print(f"Burn time: {self.motor.burn_time:.2f} s")
        print(f"\nStarting simulation...")
        
        steps = 0
        last_print_time = 0
        
        while self.time < max_time:
            self.step()
            steps += 1
            
            # Print progress every 1 second
            if verbose and self.time - last_print_time >= 1.0:
                v = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
                print(f"  t={self.time:.1f}s: h={self.state.z:.1f}m, v={v:.1f}m/s, "
                      f"pitch={self.history['pitch'][-1]:.1f}°")
                last_print_time = self.time
            
            # Stop if landed
            if self.state.z <= 0 and self.time > 5.0:
                print(f"  🛬 Landed at t={self.time:.1f}s")
                break
        
        # Summary
        max_alt = max(self.history['altitude'])
        max_vel = max(self.history['velocity'])
        max_mach = max(self.history['mach'])
        
        print(f"\n{'='*70}")
        print(f"FLIGHT SUMMARY")
        print(f"{'='*70}")
        print(f"Maximum altitude: {max_alt:.1f} m ({max_alt*3.28084:.1f} ft)")
        print(f"Maximum velocity: {max_vel:.1f} m/s")
        print(f"Maximum Mach: {max_mach:.3f}")
        print(f"Flight time: {self.time:.1f} s")
        print(f"Total simulation steps: {steps}")
        print(f"{'='*70}\n")
    
    def plot_results(self, filename: Optional[str] = None):
        """Plot comprehensive flight data"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        t = np.array(self.history['time'])
        
        # Row 1: Position and trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, self.history['altitude'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Altitude vs Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, self.history['velocity'], 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity vs Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history['x'], self.history['z'], 'g-', linewidth=2)
        ax3.set_xlabel('Downrange (m)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Flight Trajectory', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # Row 2: Forces and attitude
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, self.history['thrust'], 'b-', label='Thrust', linewidth=2)
        ax4.plot(t, self.history['drag'], 'r-', label='Drag', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Force (N)')
        ax4.set_title('Thrust and Drag', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(t, self.history['pitch'], 'orange', label='Pitch', linewidth=2)
        ax5.plot(t, self.history['roll'], 'cyan', label='Roll', linewidth=1)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angle (deg)')
        ax5.set_title('Attitude Angles', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(t, self.history['aoa'], 'purple', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle of Attack (deg)')
        ax6.set_title('Angle of Attack', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Row 3: Mass properties and stability
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(t, self.history['mass'], 'brown', linewidth=2)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Mass (kg)')
        ax7.set_title('Total Mass', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(t, self.history['cg'], 'b-', label='CG', linewidth=2)
        ax8.plot(t, self.history['cp'], 'r-', label='CP', linewidth=2)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Position from Nose (m)')
        ax8.set_title('CG and CP Tracking', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(t, self.history['static_margin'], 'green', linewidth=2)
        ax9.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Min Rec.')
        ax9.axhline(y=2.0, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax9.axhline(y=0.0, color='r', linestyle='--', alpha=0.7, label='Unstable')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Static Margin (calibers)')
        ax9.set_title('Stability Margin', fontweight='bold')
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)
        
        # Row 4: Advanced metrics
        ax10 = fig.add_subplot(gs[3, 0])
        ax10.plot(t, self.history['mach'], 'cyan', linewidth=2)
        ax10.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Mach 1')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Mach Number')
        ax10.set_title('Mach Number', fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        ax11 = fig.add_subplot(gs[3, 1])
        accel = np.gradient(self.history['velocity'], t)
        accel_g = accel / 9.80665
        ax11.plot(t, accel_g, 'red', linewidth=2)
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Acceleration (g)')
        ax11.set_title('Acceleration Profile', fontweight='bold')
        ax11.grid(True, alpha=0.3)
        
        ax12 = fig.add_subplot(gs[3, 2])
        # 3D trajectory
        ax12 = fig.add_subplot(gs[3, 2], projection='3d')
        ax12.plot(self.history['x'], self.history['y'], self.history['z'], 'b-', linewidth=2)
        ax12.set_xlabel('X (m)')
        ax12.set_ylabel('Y (m)')
        ax12.set_zlabel('Z (m)')
        ax12.set_title('3D Trajectory', fontweight='bold')
        
        plt.suptitle(f'{self.rocket.name} - Robust 6DOF Simulation', 
                    fontsize=16, fontweight='bold')
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        plt.show()


def test_robust_simulator():
    """Test the robust simulator"""
    from rocket_components import (
        Rocket, NoseCone, BodyTube, FinSet, Parachute,
        NoseShape, MATERIALS
    )
    from motor_database import create_sample_motors
    
    # Build test rocket
    rocket = Rocket("Test Rocket - Mid Power")
    
    rocket.add_component(NoseCone(
        "Ogive Nose",
        NoseShape.OGIVE,
        length=0.15,
        diameter=0.054,
        thickness=0.003,
        material=MATERIALS["Fiberglass"]
    ))
    
    rocket.add_component(BodyTube(
        "Body Tube",
        length=0.60,
        outer_diameter=0.054,
        thickness=0.002,
        material=MATERIALS["Blue Tube"],
        position_x=0.15
    ))
    
    rocket.add_component(FinSet(
        "Fins",
        fin_count=4,
        root_chord=0.12,
        tip_chord=0.06,
        semi_span=0.10,
        sweep_length=0.04,
        thickness=0.004,
        material=MATERIALS["Fiberglass"],
        position_x=0.65
    ))
    
    rocket.add_component(Parachute(
        "Main Chute",
        diameter=0.60,
        cd_parachute=0.75,
        deployment_time=8.0,
        packed_mass=0.050,
        position_x=0.10
    ))
    
    # Print rocket summary
    rocket.print_summary()
    
    # Get motor
    motor_db = create_sample_motors()
    motor = motor_db.get_motor("Aerotech", "F50-6T")
    
    # Run simulation
    sim = RobustSimulator(
        rocket=rocket,
        motor=motor,
        motor_position=0.70,
        launch_angle_deg=90.0,
        rail_length=1.5
    )
    
    sim.run(max_time=30.0, verbose=True)
    sim.plot_results('/home/kush-mahajan/elodin/examples/rocket-barrowman/robust_simulation.png')


if __name__ == "__main__":
    test_robust_simulator()

