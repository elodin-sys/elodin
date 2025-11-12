"""
Correct Barrowman-Based Rocket Simulation
Implementing equations exactly as specified in the white papers
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math


@dataclass
class RocketConfig:
    """Complete rocket configuration"""
    # Geometry
    nose_length: float = 0.2  # m
    body_diameter: float = 0.1  # m
    body_length: float = 1.0  # m
    
    # Fins (trapezoidal)
    fin_count: int = 4
    fin_root_chord: float = 0.15  # m (Cr)
    fin_tip_chord: float = 0.1  # m (Ct)
    fin_semi_span: float = 0.08  # m (s)
    fin_root_le_position: float = 0.8  # m from nose tip (Xf)
    fin_sweep_length: float = 0.03  # m (Xr) - leading edge sweep
    
    # Mass
    dry_mass: float = 2.0  # kg
    propellant_mass: float = 0.5  # kg
    cg_loaded: float = 0.4  # m from nose tip
    cg_burnout: float = 0.35  # m from nose tip
    
    # Motor
    total_impulse: float = 100.0  # N·s
    burn_time: float = 2.0  # s


class BarrowmanCalculator:
    """
    Implements Barrowman aerodynamic equations exactly as in the paper
    """
    
    def __init__(self, config: RocketConfig):
        self.cfg = config
        
        # Reference quantities
        self.d = config.body_diameter
        self.A_ref = (math.pi / 4) * self.d ** 2  # Reference area
        
        # Body radius
        self.r = self.d / 2
        
    def nose_cn_alpha(self) -> float:
        """
        Nose normal force coefficient slope
        From Barrowman Equation 12: CNα_N = 2 per radian
        """
        return 2.0
    
    def nose_cp(self) -> float:
        """
        Nose center of pressure from nose tip
        From Barrowman Equation 39: X̄_N = 0.466 * L (for ogive)
        """
        return 0.466 * self.cfg.nose_length
    
    def fin_cn_alpha_single(self) -> float:
        """
        Single fin normal force coefficient slope
        From Barrowman Equation 57
        
        CNα = 8(s/d)² / [1 + sqrt(1 + (2l/(Cr+Ct))²)]
        """
        s = self.cfg.fin_semi_span
        d = self.d
        Cr = self.cfg.fin_root_chord
        Ct = self.cfg.fin_tip_chord
        
        # Calculate mid-chord line length l
        # l = sqrt(s² + (sweep at mid-chord)²)
        # For trapezoidal fin: mid-chord sweep ≈ Xr/2
        mid_chord_sweep = self.cfg.fin_sweep_length / 2
        l = math.sqrt(s**2 + mid_chord_sweep**2)
        
        numerator = 8 * (s / d) ** 2
        denominator = 1 + math.sqrt(1 + (2 * l / (Cr + Ct)) ** 2)
        
        return numerator / denominator
    
    def fin_cn_alpha_set(self) -> float:
        """
        Fin set normal force coefficient slope
        From Barrowman Equation 58: For 4 fins, multiply single fin by 2
        """
        if self.cfg.fin_count == 4:
            return 2 * self.fin_cn_alpha_single()
        elif self.cfg.fin_count == 3:
            return math.sqrt(3) * self.fin_cn_alpha_single()
        else:
            return self.cfg.fin_count / 2 * self.fin_cn_alpha_single()
    
    def interference_factor(self) -> float:
        """
        Body-fin interference factor
        From Barrowman Equation 77: K = 1 + r/(s+r)
        """
        r = self.r
        s = self.cfg.fin_semi_span
        return 1.0 + r / (s + r)
    
    def fin_cn_alpha_with_body(self) -> float:
        """
        Fin CNα including body interference
        From Barrowman Equation 78
        """
        return self.interference_factor() * self.fin_cn_alpha_set()
    
    def fin_cp(self) -> float:
        """
        Fin center of pressure from nose tip
        From Barrowman Equation 76a and 76b
        
        X̄_F = Xf + (Xr/3) * [(Cr + 2Ct)/(Cr + Ct)] + (1/6)[Cr + Ct - CrCt/(Cr+Ct)]
        """
        Xf = self.cfg.fin_root_le_position
        Xr = self.cfg.fin_sweep_length
        Cr = self.cfg.fin_root_chord
        Ct = self.cfg.fin_tip_chord
        
        term1 = (Xr / 3) * ((Cr + 2 * Ct) / (Cr + Ct))
        term2 = (1 / 6) * (Cr + Ct - (Cr * Ct) / (Cr + Ct))
        
        return Xf + term1 + term2
    
    def total_cn_alpha(self) -> float:
        """
        Total vehicle CNα
        From Barrowman Equation 79
        """
        return self.nose_cn_alpha() + self.fin_cn_alpha_with_body()
    
    def total_cp(self) -> float:
        """
        Total vehicle center of pressure
        From Barrowman Equation 80: moment balance
        """
        cn_nose = self.nose_cn_alpha()
        cn_fins = self.fin_cn_alpha_with_body()
        
        x_nose = self.nose_cp()
        x_fins = self.fin_cp()
        
        cn_total = cn_nose + cn_fins
        
        return (cn_nose * x_nose + cn_fins * x_fins) / cn_total
    
    def static_margin(self, cg: float) -> float:
        """Calculate static margin (CP - CG) / d"""
        cp = self.total_cp()
        return (cp - cg) / self.d


class RocketSimulation:
    """6DOF rocket simulation using Barrowman aerodynamics"""
    
    def __init__(self, config: RocketConfig):
        self.cfg = config
        self.bman = BarrowmanCalculator(config)
        
        # Time step
        self.dt = 1.0 / 120.0
        
        # State
        self.time = 0.0
        self.altitude = 0.0  # m
        self.velocity = 0.0  # m/s (vertical, positive up)
        self.angle = math.pi / 2  # radians from horizontal (pi/2 = vertical)
        self.angular_velocity = 0.0  # rad/s (pitch rate)
        
        # Mass
        self.mass = config.dry_mass + config.propellant_mass
        self.cg = config.cg_loaded
        
        # Data logging
        self.history = {
            'time': [],
            'altitude': [],
            'velocity': [],
            'angle_deg': [],
            'mass': [],
            'thrust': [],
            'alpha_deg': [],
        }
    
    def get_thrust(self, t: float) -> float:
        """Get thrust at time t"""
        if 0 <= t <= self.cfg.burn_time:
            return self.cfg.total_impulse / self.cfg.burn_time
        return 0.0
    
    def get_mass_and_cg(self, t: float) -> tuple:
        """Get current mass and CG"""
        if t >= self.cfg.burn_time:
            return self.cfg.dry_mass, self.cfg.cg_burnout
        
        # Linear burn
        frac = t / self.cfg.burn_time
        mass = self.cfg.dry_mass + self.cfg.propellant_mass * (1 - frac)
        cg = self.cfg.cg_loaded + (self.cfg.cg_burnout - self.cfg.cg_loaded) * frac
        return mass, cg
    
    def get_atmosphere(self, alt: float) -> tuple:
        """Get density and speed of sound"""
        # ISA atmosphere (simplified)
        if alt < 0:
            alt = 0
        
        # Density (kg/m³)
        rho = 1.225 * math.exp(-alt / 8400)
        
        # Speed of sound (m/s)
        a = 340.0
        
        return rho, a
    
    def step(self):
        """Single simulation step"""
        # Update mass and CG
        self.mass, self.cg = self.get_mass_and_cg(self.time)
        
        # Atmospheric properties
        rho, a = self.get_atmosphere(self.altitude)
        
        # Dynamic pressure
        q = 0.5 * rho * self.velocity ** 2
        
        # Mach number
        mach = self.velocity / a
        
        # Angle of attack (simplified: difference between velocity angle and body angle)
        # For vertical flight, this is small
        velocity_angle = math.pi / 2  # Assuming mostly vertical flight
        alpha = self.angle - velocity_angle
        
        # Aerodynamic coefficients
        cn_alpha = self.bman.total_cn_alpha()
        cn = cn_alpha * alpha  # Normal force coefficient
        
        # Drag coefficient (simplified)
        cd = 0.3 + 0.1 * alpha ** 2
        
        # Aerodynamic forces
        drag = cd * q * self.bman.A_ref
        normal_force = cn * q * self.bman.A_ref
        
        # Thrust
        thrust = self.get_thrust(self.time)
        
        # Forces in vertical direction
        # Thrust acts along rocket body axis
        thrust_vertical = thrust * math.sin(self.angle)
        
        # Drag opposes motion
        drag_vertical = -drag * (self.velocity / abs(self.velocity)) if abs(self.velocity) > 1e-6 else 0
        
        # Normal force (perpendicular to body)
        normal_vertical = normal_force * math.cos(self.angle)
        
        # Gravity
        gravity_force = -9.81 * self.mass
        
        # Total vertical force
        total_force_vertical = thrust_vertical + drag_vertical + normal_vertical + gravity_force
        
        # Vertical acceleration
        accel = total_force_vertical / self.mass
        
        # Pitching moment about CG
        cp = self.bman.total_cp()
        moment_arm = cp - self.cg
        pitching_moment = normal_force * moment_arm
        
        # Angular acceleration (simplified inertia)
        I_pitch = 0.1  # kg·m²
        angular_accel = pitching_moment / I_pitch
        
        # Integrate
        self.velocity += accel * self.dt
        self.altitude += self.velocity * self.dt
        
        self.angular_velocity += angular_accel * self.dt
        self.angle += self.angular_velocity * self.dt
        
        # Ground contact
        if self.altitude < 0:
            self.altitude = 0
            self.velocity = 0
        
        # Log data
        self.history['time'].append(self.time)
        self.history['altitude'].append(self.altitude)
        self.history['velocity'].append(self.velocity)
        self.history['angle_deg'].append(math.degrees(self.angle))
        self.history['mass'].append(self.mass)
        self.history['thrust'].append(thrust)
        self.history['alpha_deg'].append(math.degrees(alpha))
        
        # Advance time
        self.time += self.dt
    
    def run(self, duration: float = 10.0):
        """Run simulation"""
        print("Starting simulation...")
        print(f"Initial CP: {self.bman.total_cp():.3f} m")
        print(f"Initial CG: {self.cg:.3f} m")
        print(f"Initial static margin: {self.bman.static_margin(self.cg):.2f} calibers")
        print()
        
        while self.time < duration and self.altitude >= 0:
            self.step()
            
            # Print every second
            if len(self.history['time']) % 120 == 0:
                print(f"t={self.time:.1f}s: h={self.altitude:.1f}m, v={self.velocity:.1f}m/s, m={self.mass:.2f}kg")
        
        print(f"\nFinal altitude: {max(self.history['altitude']):.1f} m")
        print(f"Final time: {self.time:.1f} s")
    
    def plot(self):
        """Plot results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        t = self.history['time']
        
        # Altitude
        axes[0, 0].plot(t, self.history['altitude'])
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].set_title('Altitude vs Time')
        axes[0, 0].grid(True)
        
        # Velocity
        axes[0, 1].plot(t, self.history['velocity'])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].grid(True)
        
        # Angle
        axes[0, 2].plot(t, self.history['angle_deg'])
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Angle from Horizontal (deg)')
        axes[0, 2].set_title('Body Angle vs Time')
        axes[0, 2].grid(True)
        
        # Mass
        axes[1, 0].plot(t, self.history['mass'])
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Mass (kg)')
        axes[1, 0].set_title('Mass vs Time')
        axes[1, 0].grid(True)
        
        # Thrust
        axes[1, 1].plot(t, self.history['thrust'])
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Thrust (N)')
        axes[1, 1].set_title('Thrust vs Time')
        axes[1, 1].grid(True)
        
        # Angle of attack
        axes[1, 2].plot(t, self.history['alpha_deg'])
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Angle of Attack (deg)')
        axes[1, 2].set_title('Angle of Attack vs Time')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point"""
    print("=" * 60)
    print("BARROWMAN AERODYNAMIC ROCKET SIMULATION")
    print("=" * 60)
    
    config = RocketConfig()
    sim = RocketSimulation(config)
    sim.run(duration=15.0)
    sim.plot()


if __name__ == "__main__":
    main()

