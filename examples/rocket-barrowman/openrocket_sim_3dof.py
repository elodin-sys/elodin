"""
Simplified 3DOF OpenRocket simulation (no rotation).
Gets the basic physics working first before adding 6DOF complexity.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from openrocket_components import *
from openrocket_motor import Motor
from openrocket_aero import RocketAerodynamics
from openrocket_atmosphere import ISAAtmosphere


@dataclass
class State3DOF:
    """3DOF state (position + velocity only)"""
    time: float = 0.0
    z: float = 0.0  # Altitude
    vz: float = 0.0  # Vertical velocity
    motor_time: float = -0.01
    motor_ignited: bool = False
    parachute_deployed: bool = False
    
    def copy(self):
        return State3DOF(
            time=self.time, z=self.z, vz=self.vz,
            motor_time=self.motor_time,
            motor_ignited=self.motor_ignited,
            parachute_deployed=self.parachute_deployed
        )


class Simulator3DOF:
    """Simple 3DOF vertical flight simulator"""
    
    def __init__(self, rocket: Rocket, motor: Motor):
        self.rocket = rocket
        self.motor = motor
        self.atmosphere = ISAAtmosphere()
        self.aero = RocketAerodynamics(rocket)
        self.dt = 0.01
        self.max_time = 300.0
        self.rail_length = 1.0
        self.history = []
        
    def get_mass(self, time: float) -> float:
        """Get total mass at time"""
        comp_mass = self.rocket.get_total_mass()
        motor_mass = self.motor.get_mass(time)
        return comp_mass + motor_mass
    
    def compute_acceleration(self, state: State3DOF) -> float:
        """Compute vertical acceleration"""
        mass = self.get_mass(state.motor_time)
        if mass <= 0:
            return -9.81
        
        # Atmospheric properties
        atm = self.atmosphere.get_properties(max(0, state.z))
        rho = atm['density']
        mu = atm['viscosity']
        a_sound = atm['speed_of_sound']
        
        # Thrust (vertical)
        thrust = self.motor.get_thrust(state.motor_time) if state.motor_ignited else 0.0
        
        # Gravity
        gravity = -9.80665
        
        # Drag
        v_mag = abs(state.vz)
        if v_mag > 0.1:
            mach = v_mag / a_sound
            ref_area = math.pi * (self.rocket.reference_diameter / 2)**2
            cd = self.aero.calculate_cd(mach, v_mag, rho, mu, 0.0)
            q = 0.5 * rho * v_mag**2
            drag_force = cd * q * ref_area
            
            # Drag opposes motion
            if state.vz > 0:
                drag = -drag_force / mass
            else:
                drag = drag_force / mass
        else:
            drag = 0.0
        
        # Parachute drag
        chute_drag = 0.0
        if state.parachute_deployed and v_mag > 0.1:
            for comp in self.rocket.children:
                if isinstance(comp, Parachute):
                    drag_area = comp.get_drag_area()
                    chute_force = 0.5 * rho * v_mag**2 * drag_area
                    if state.vz < 0:  # Falling
                        chute_drag = chute_force / mass
                    break
        
        # Total acceleration
        accel = gravity + thrust/mass + drag + chute_drag
        
        return accel
    
    def rk4_step(self, state: State3DOF, dt: float) -> State3DOF:
        """RK4 integration step"""
        
        def deriv(s):
            ds = State3DOF()
            ds.time = 1.0
            ds.z = s.vz
            ds.vz = self.compute_acceleration(s)
            ds.motor_time = 1.0 if s.motor_ignited else 0.0
            return ds
        
        k1 = deriv(state)
        
        s2 = state.copy()
        s2.time += dt/2
        s2.z += k1.z * dt/2
        s2.vz += k1.vz * dt/2
        s2.motor_time += k1.motor_time * dt/2
        k2 = deriv(s2)
        
        s3 = state.copy()
        s3.time += dt/2
        s3.z += k2.z * dt/2
        s3.vz += k2.vz * dt/2
        s3.motor_time += k2.motor_time * dt/2
        k3 = deriv(s3)
        
        s4 = state.copy()
        s4.time += dt
        s4.z += k3.z * dt
        s4.vz += k3.vz * dt
        s4.motor_time += k3.motor_time * dt
        k4 = deriv(s4)
        
        new_state = state.copy()
        new_state.time += dt
        new_state.z += dt/6 * (k1.z + 2*k2.z + 2*k3.z + k4.z)
        new_state.vz += dt/6 * (k1.vz + 2*k2.vz + 2*k3.vz + k4.vz)
        new_state.motor_time += dt/6 * (k1.motor_time + 2*k2.motor_time + 2*k3.motor_time + k4.motor_time)
        
        return new_state
    
    def run(self) -> List[State3DOF]:
        """Run simulation"""
        state = State3DOF()
        state.time = 0.0
        state.z = 0.0
        state.vz = 0.0
        state.motor_ignited = True
        state.motor_time = 0.0
        
        self.history = [state.copy()]
        
        on_rail = True
        apogee_reached = False
        
        while state.time < self.max_time:
            # Check rail departure
            if on_rail and state.z >= self.rail_length:
                on_rail = False
            
            # Check apogee
            if state.vz < 0 and not apogee_reached:
                apogee_reached = True
                # Deploy parachute
                for comp in self.rocket.children:
                    if isinstance(comp, Parachute) and comp.deployment_event == "APOGEE":
                        state.parachute_deployed = True
                        comp.deployed = True
                        break
            
            # Integrate
            new_state = self.rk4_step(state, self.dt)
            
            # Check ground
            if new_state.z <= 0 and state.time > 1.0:
                new_state.z = 0
                new_state.vz = 0
                self.history.append(new_state.copy())
                break
            
            state = new_state
            self.history.append(state.copy())
        
        return self.history
    
    def get_summary(self) -> dict:
        """Get flight summary"""
        if not self.history:
            return {}
        
        max_alt = max(s.z for s in self.history)
        max_vel = max(abs(s.vz) for s in self.history)
        flight_time = self.history[-1].time
        apogee_idx = max(range(len(self.history)), key=lambda i: self.history[i].z)
        apogee_time = self.history[apogee_idx].time
        
        return {
            'max_altitude': max_alt,
            'max_velocity': max_vel,
            'apogee_time': apogee_time,
            'flight_time': flight_time,
        }

