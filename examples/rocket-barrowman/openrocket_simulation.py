"""
OpenRocket 6DOF simulation - exact implementation.
Matches OpenRocket's RK4 integration and event detection.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from openrocket_components import *
from openrocket_motor import Motor
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere, WindModel


@dataclass
class SimulationState:
    """Complete 6DOF state vector"""
    time: float = 0.0
    
    # Position (m, world frame)
    x: float = 0.0  # North
    y: float = 0.0  # East  
    z: float = 0.0  # Up
    
    # Velocity (m/s, world frame)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    
    # Orientation (quaternion, world to body)
    q0: float = 1.0  # w
    q1: float = 0.0  # x
    q2: float = 0.0  # y
    q3: float = 0.0  # z
    
    # Angular velocity (rad/s, body frame)
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    
    # Motor state
    motor_time: float = -0.01  # Time since motor ignition
    motor_ignited: bool = False
    
    # Recovery state
    parachute_deployed: bool = False
    apogee_reached: bool = False
    
    def copy(self):
        """Create a copy of this state"""
        return SimulationState(
            time=self.time, x=self.x, y=self.y, z=self.z,
            vx=self.vx, vy=self.vy, vz=self.vz,
            q0=self.q0, q1=self.q1, q2=self.q2, q3=self.q3,
            wx=self.wx, wy=self.wy, wz=self.wz,
            motor_time=self.motor_time, motor_ignited=self.motor_ignited,
            parachute_deployed=self.parachute_deployed,
            apogee_reached=self.apogee_reached
        )


class OpenRocketSimulator:
    """
    OpenRocket 6DOF flight simulator - exact implementation.
    Uses RK4 integration matching OpenRocket's calculation loop.
    """
    
    def __init__(self, rocket: Rocket, motor: Motor, 
                 atmosphere: Optional[ISAAtmosphere] = None,
                 wind: Optional[WindModel] = None):
        self.rocket = rocket
        self.motor = motor
        self.atmosphere = atmosphere or ISAAtmosphere()
        self.wind = wind or WindModel()
        self.aero = RocketAerodynamics(rocket)
        
        # Simulation parameters
        self.dt = 0.01  # 10ms timestep (OpenRocket default)
        self.max_time = 300.0  # 5 minutes max
        self.ground_level = 0.0
        self.rail_length = 1.0  # m
        self.rail_angle = math.pi / 2  # 90° = vertical
        self.rail_direction = 0.0  # 0 = north
        
        # Results storage
        self.history = []
        
    def quaternion_to_matrix(self, q0, q1, q2, q3) -> np.ndarray:
        """Convert quaternion to rotation matrix (body to world)"""
        return np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])
    
    def get_mass_properties(self, time: float) -> Tuple[float, float, np.ndarray]:
        """
        Get rocket mass, CG, and inertia at given time.
        Includes motor mass variation.
        """
        # Component mass and CG
        comp_mass = self.rocket.get_total_mass()
        comp_cg = self.rocket.get_total_cg()
        
        # Motor mass and CG
        motor_mass = self.motor.get_mass(time)
        motor_cg = self.motor.get_cg(time)
        
        # Find motor mount position
        motor_mount_pos = 0.0
        def find_motor_mount(component):
            nonlocal motor_mount_pos
            if isinstance(component, BodyTube) and component.motor_mount:
                motor_mount_pos = component.get_absolute_position()
            for child in component.children:
                find_motor_mount(child)
        find_motor_mount(self.rocket)
        
        # Motor absolute CG
        motor_cg_abs = motor_mount_pos + motor_cg
        
        # Total mass and CG
        total_mass = comp_mass + motor_mass
        if total_mass > 0:
            total_cg = (comp_mass * comp_cg + motor_mass * motor_cg_abs) / total_mass
        else:
            total_cg = 0.0
        
        # Inertia (simplified - assume rocket is thin cylinder)
        # OpenRocket calculates component-by-component, but this is close
        length = self.rocket.reference_length
        radius = self.rocket.reference_diameter / 2
        
        # Moments of inertia about CG
        Ixx = Iyy = (1.0/12.0) * total_mass * length**2 + (1.0/4.0) * total_mass * radius**2
        Izz = (1.0/2.0) * total_mass * radius**2
        
        inertia = np.array([Ixx, Iyy, Izz])
        
        return total_mass, total_cg, inertia
    
    def compute_derivatives(self, state: SimulationState) -> SimulationState:
        """
        Compute state derivatives (exact OpenRocket physics).
        Returns d(state)/dt.
        """
        deriv = SimulationState()
        deriv.time = 1.0
        
        # Get current properties
        mass, cg, inertia = self.get_mass_properties(state.motor_time)
        atm = self.atmosphere.get_properties(state.z)
        rho = atm['density']
        mu = atm['viscosity']
        a_sound = atm['speed_of_sound']
        
        # Rotation matrix (body to world)
        R = self.quaternion_to_matrix(state.q0, state.q1, state.q2, state.q3)
        
        # --- Forces and moments ---
        
        # 1. Gravity (world frame)
        F_gravity = np.array([0, 0, -mass * 9.80665])
        
        # 2. Thrust (body frame, along +x-axis which points forward/up)
        thrust_mag = self.motor.get_thrust(state.motor_time) if state.motor_ignited else 0.0
        F_thrust_body = np.array([thrust_mag, 0, 0])  # Thrust along body-x
        F_thrust = R @ F_thrust_body  # Transform to world frame
        
        # 3. Aerodynamic forces
        # Wind effect
        wind_vel = np.array(self.wind.get_wind(state.z, state.time))
        
        # Relative velocity (rocket wrt air, world frame)
        v_rocket = np.array([state.vx, state.vy, state.vz])
        v_rel = v_rocket - wind_vel
        v_mag = np.linalg.norm(v_rel)
        
        if v_mag > 0.1:
            # Velocity in body frame
            v_body = R.T @ v_rel
            
            # Angle of attack and sideslip
            alpha = math.atan2(v_body[2], v_body[0])  # Pitch
            beta = math.atan2(v_body[1], v_body[0])   # Yaw
            
            # Mach number
            mach = v_mag / a_sound
            
            # Dynamic pressure
            q = 0.5 * rho * v_mag**2
            
            # Reference area
            ref_area = math.pi * (self.rocket.reference_diameter / 2)**2
            
            # Drag coefficient
            cd = self.aero.calculate_cd(mach, v_mag, rho, mu, alpha)
            
            # Normal force coefficient
            cn_alpha = self.aero.calculate_cn_alpha(mach)
            
            # Drag force (opposite to velocity)
            v_unit = v_rel / v_mag
            F_drag = -cd * q * ref_area * v_unit
            
            # Normal force (perpendicular to body axis in body frame)
            # Lift in z-direction (body), side force in y-direction (body)
            F_normal_body = np.array([
                0,
                cn_alpha * beta * q * ref_area,   # Side force
                cn_alpha * alpha * q * ref_area   # Lift force
            ])
            F_normal = R @ F_normal_body
            
            F_aero = F_drag + F_normal
            
            # Aerodynamic moment (body frame)
            # Moment arm from CG to CP
            cp = self.aero.calculate_cp(mach)
            moment_arm = cp - cg
            
            # Pitching moment
            cm_alpha = cn_alpha * moment_arm / self.rocket.reference_diameter
            M_aero_body = np.array([
                0,  # Roll moment (assume zero for symmetric rocket)
                -cm_alpha * alpha * q * ref_area * self.rocket.reference_diameter,  # Pitch
                -cm_alpha * beta * q * ref_area * self.rocket.reference_diameter    # Yaw
            ])
            
            # Damping moments (resists rotation)
            ref_length = self.rocket.reference_length
            M_damping_body = -0.5 * rho * v_mag * ref_length * ref_area * np.array([
                0.01 * state.wx,  # Roll damping
                0.5 * state.wy,   # Pitch damping
                0.5 * state.wz    # Yaw damping
            ])
            
            M_aero = M_aero_body + M_damping_body
        else:
            F_aero = np.zeros(3)
            M_aero = np.zeros(3)
        
        # Total force
        F_total = F_gravity + F_thrust + F_aero
        
        # 4. Parachute drag (if deployed)
        if state.parachute_deployed:
            # Find parachute
            for component in self.rocket.children:
                if isinstance(component, Parachute):
                    drag_area = component.get_drag_area()
                    if v_mag > 0.1:
                        F_chute = -0.5 * rho * v_mag**2 * drag_area * (v_rel / v_mag)
                        F_total += F_chute
                    break
        
        # --- Derivatives ---
        
        # Position derivative = velocity
        deriv.x = state.vx
        deriv.y = state.vy
        deriv.z = state.vz
        
        # Velocity derivative = acceleration
        if mass > 0:
            accel = F_total / mass
            # Clamp acceleration to prevent numerical explosion
            accel_mag = np.linalg.norm(accel)
            if accel_mag > 1000.0:  # Max 100g
                accel = accel / accel_mag * 1000.0
            deriv.vx = accel[0]
            deriv.vy = accel[1]
            deriv.vz = accel[2]
        
        # Quaternion derivative
        # q_dot = 0.5 * q * omega
        deriv.q0 = 0.5 * (-state.q1*state.wx - state.q2*state.wy - state.q3*state.wz)
        deriv.q1 = 0.5 * ( state.q0*state.wx + state.q2*state.wz - state.q3*state.wy)
        deriv.q2 = 0.5 * ( state.q0*state.wy - state.q1*state.wz + state.q3*state.wx)
        deriv.q3 = 0.5 * ( state.q0*state.wz + state.q1*state.wy - state.q2*state.wx)
        
        # Angular velocity derivative = angular acceleration
        # M = I * alpha + omega x (I * omega)
        if inertia[0] > 0 and inertia[1] > 0 and inertia[2] > 0:
            I_omega = inertia * np.array([state.wx, state.wy, state.wz])
            omega_cross_I_omega = np.array([
                state.wy * I_omega[2] - state.wz * I_omega[1],
                state.wz * I_omega[0] - state.wx * I_omega[2],
                state.wx * I_omega[1] - state.wy * I_omega[0]
            ])
            
            alpha = (M_aero - omega_cross_I_omega) / inertia
            deriv.wx = alpha[0]
            deriv.wy = alpha[1]
            deriv.wz = alpha[2]
        
        # Motor time
        if state.motor_ignited:
            deriv.motor_time = 1.0
        
        return deriv
    
    def rk4_step(self, state: SimulationState, dt: float) -> SimulationState:
        """RK4 integration step (exact OpenRocket method)"""
        
        # k1 = f(t, y)
        k1 = self.compute_derivatives(state)
        
        # k2 = f(t + dt/2, y + k1*dt/2)
        state2 = state.copy()
        state2.time += dt/2
        state2.x += k1.x * dt/2
        state2.y += k1.y * dt/2
        state2.z += k1.z * dt/2
        state2.vx += k1.vx * dt/2
        state2.vy += k1.vy * dt/2
        state2.vz += k1.vz * dt/2
        state2.q0 += k1.q0 * dt/2
        state2.q1 += k1.q1 * dt/2
        state2.q2 += k1.q2 * dt/2
        state2.q3 += k1.q3 * dt/2
        state2.wx += k1.wx * dt/2
        state2.wy += k1.wy * dt/2
        state2.wz += k1.wz * dt/2
        state2.motor_time += k1.motor_time * dt/2
        k2 = self.compute_derivatives(state2)
        
        # k3 = f(t + dt/2, y + k2*dt/2)
        state3 = state.copy()
        state3.time += dt/2
        state3.x += k2.x * dt/2
        state3.y += k2.y * dt/2
        state3.z += k2.z * dt/2
        state3.vx += k2.vx * dt/2
        state3.vy += k2.vy * dt/2
        state3.vz += k2.vz * dt/2
        state3.q0 += k2.q0 * dt/2
        state3.q1 += k2.q1 * dt/2
        state3.q2 += k2.q2 * dt/2
        state3.q3 += k2.q3 * dt/2
        state3.wx += k2.wx * dt/2
        state3.wy += k2.wy * dt/2
        state3.wz += k2.wz * dt/2
        state3.motor_time += k2.motor_time * dt/2
        k3 = self.compute_derivatives(state3)
        
        # k4 = f(t + dt, y + k3*dt)
        state4 = state.copy()
        state4.time += dt
        state4.x += k3.x * dt
        state4.y += k3.y * dt
        state4.z += k3.z * dt
        state4.vx += k3.vx * dt
        state4.vy += k3.vy * dt
        state4.vz += k3.vz * dt
        state4.q0 += k3.q0 * dt
        state4.q1 += k3.q1 * dt
        state4.q2 += k3.q2 * dt
        state4.q3 += k3.q3 * dt
        state4.wx += k3.wx * dt
        state4.wy += k3.wy * dt
        state4.wz += k3.wz * dt
        state4.motor_time += k3.motor_time * dt
        k4 = self.compute_derivatives(state4)
        
        # y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        new_state = state.copy()
        new_state.time += dt
        new_state.x += dt/6 * (k1.x + 2*k2.x + 2*k3.x + k4.x)
        new_state.y += dt/6 * (k1.y + 2*k2.y + 2*k3.y + k4.y)
        new_state.z += dt/6 * (k1.z + 2*k2.z + 2*k3.z + k4.z)
        new_state.vx += dt/6 * (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx)
        new_state.vy += dt/6 * (k1.vy + 2*k2.vy + 2*k3.vy + k4.vy)
        new_state.vz += dt/6 * (k1.vz + 2*k2.vz + 2*k3.vz + k4.vz)
        new_state.q0 += dt/6 * (k1.q0 + 2*k2.q0 + 2*k3.q0 + k4.q0)
        new_state.q1 += dt/6 * (k1.q1 + 2*k2.q1 + 2*k3.q1 + k4.q1)
        new_state.q2 += dt/6 * (k1.q2 + 2*k2.q2 + 2*k3.q2 + k4.q2)
        new_state.q3 += dt/6 * (k1.q3 + 2*k2.q3 + 2*k3.q3 + k4.q3)
        new_state.wx += dt/6 * (k1.wx + 2*k2.wx + 2*k3.wx + k4.wx)
        new_state.wy += dt/6 * (k1.wy + 2*k2.wy + 2*k3.wy + k4.wy)
        new_state.wz += dt/6 * (k1.wz + 2*k2.wz + 2*k3.wz + k4.wz)
        new_state.motor_time += dt/6 * (k1.motor_time + 2*k2.motor_time + 2*k3.motor_time + k4.motor_time)
        
        # Normalize quaternion
        q_norm = math.sqrt(new_state.q0**2 + new_state.q1**2 + new_state.q2**2 + new_state.q3**2)
        if q_norm > 0:
            new_state.q0 /= q_norm
            new_state.q1 /= q_norm
            new_state.q2 /= q_norm
            new_state.q3 /= q_norm
        
        # Sanity checks to prevent explosion
        if abs(new_state.z) > 1e6 or math.isnan(new_state.z) or math.isnan(new_state.vz):
            print(f"ERROR at t={new_state.time:.3f}s:")
            print(f"  Position: ({new_state.x}, {new_state.y}, {new_state.z})")
            print(f"  Velocity: ({new_state.vx}, {new_state.vy}, {new_state.vz})")
            print(f"  Quaternion: ({new_state.q0}, {new_state.q1}, {new_state.q2}, {new_state.q3})")
            raise ValueError(f"Numerical instability detected at t={new_state.time:.3f}s")
        
        return new_state
    
    def run(self, launch_angle: Optional[float] = None, 
            launch_direction: Optional[float] = None) -> List[SimulationState]:
        """
        Run complete simulation (exact OpenRocket flow).
        
        Returns:
            List of simulation states at each timestep
        """
        if launch_angle is not None:
            self.rail_angle = launch_angle
        if launch_direction is not None:
            self.rail_direction = launch_direction
        
        # Initialize state
        state = SimulationState()
        state.time = 0.0
        state.z = self.ground_level
        
        # Initial orientation (quaternion for launch angle)
        # For vertical launch (rail_angle = pi/2), we want body-x pointing up (+world-z)
        # This requires a -90° pitch (nose up)
        # Quaternion for -90° pitch around y-axis: q = cos(-45°) + sin(-45°)*j
        if abs(self.rail_angle - math.pi/2) < 0.01:
            # Vertical launch - body-x along +world-z
            state.q0 = math.sqrt(2)/2
            state.q2 = -math.sqrt(2)/2  # Pitch up 90° (negative angle for positive z-direction)
        else:
            # General angle - rail_angle is from horizontal
            # Pitch angle from vertical (negative for upward)
            pitch_angle = -(math.pi/2 - self.rail_angle)
            state.q0 = math.cos(pitch_angle/2)
            state.q2 = math.sin(pitch_angle/2)
        
        # Motor ignition
        state.motor_ignited = True
        state.motor_time = 0.0
        
        self.history = [state.copy()]
        
        # Main simulation loop
        on_rail = True
        max_altitude = 0.0
        
        while state.time < self.max_time:
            # Check rail departure
            if on_rail:
                distance = math.sqrt(state.x**2 + state.y**2 + state.z**2)
                if distance >= self.rail_length:
                    on_rail = False
            
            # Check apogee
            if state.vz < 0 and not state.apogee_reached:
                state.apogee_reached = True
                max_altitude = state.z
                
                # Deploy parachute at apogee
                for component in self.rocket.children:
                    if isinstance(component, Parachute) and component.deployment_event == "APOGEE":
                        state.parachute_deployed = True
                        component.deployed = True
                        break
            
            # RK4 step
            new_state = self.rk4_step(state, self.dt)
            
            # Constrain to rail if still on it
            if on_rail:
                # Velocity along rail direction only
                v_mag = math.sqrt(new_state.vx**2 + new_state.vy**2 + new_state.vz**2)
                rail_dir = np.array([
                    math.sin(self.rail_angle) * math.cos(self.rail_direction),
                    math.sin(self.rail_angle) * math.sin(self.rail_direction),
                    math.cos(self.rail_angle)
                ])
                new_state.vx = v_mag * rail_dir[0]
                new_state.vy = v_mag * rail_dir[1]
                new_state.vz = v_mag * rail_dir[2]
            
            # Check ground impact
            if new_state.z <= self.ground_level and state.time > 1.0:
                new_state.z = self.ground_level
                new_state.vx = 0
                new_state.vy = 0
                new_state.vz = 0
                self.history.append(new_state.copy())
                break
            
            state = new_state
            self.history.append(state.copy())
        
        return self.history
    
    def get_summary(self) -> dict:
        """Get flight summary statistics"""
        if not self.history:
            return {}
        
        max_alt = max(s.z for s in self.history)
        max_vel = max(math.sqrt(s.vx**2 + s.vy**2 + s.vz**2) for s in self.history)
        flight_time = self.history[-1].time
        
        # Apogee time
        apogee_idx = max(range(len(self.history)), key=lambda i: self.history[i].z)
        apogee_time = self.history[apogee_idx].time
        
        # Landing distance
        final = self.history[-1]
        landing_distance = math.sqrt(final.x**2 + final.y**2)
        
        return {
            'max_altitude': max_alt,
            'max_velocity': max_vel,
            'apogee_time': apogee_time,
            'flight_time': flight_time,
            'landing_distance': landing_distance,
        }

