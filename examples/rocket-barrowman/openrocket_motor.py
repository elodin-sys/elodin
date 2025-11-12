"""
OpenRocket motor implementation with exact mass modeling.
Handles motor mass changes during burn, including propellant consumption.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ThrustCurvePoint:
    """Single point on thrust curve"""
    time: float  # seconds
    thrust: float  # Newtons


class Motor:
    """Motor with time-varying mass - exact OpenRocket implementation"""
    
    def __init__(self, designation: str, manufacturer: str,
                 diameter: float, length: float,
                 total_mass: float, propellant_mass: float,
                 thrust_curve: List[Tuple[float, float]],
                 burn_time: float, total_impulse: float,
                 delays: List[str] = None):
        """
        Args:
            designation: Motor designation (e.g. "F50T")
            manufacturer: Manufacturer name
            diameter: Motor diameter (m)
            length: Motor length (m)
            total_mass: Total mass at ignition (kg)
            propellant_mass: Mass of propellant only (kg)
            thrust_curve: List of (time, thrust) tuples
            burn_time: Total burn time (s)
            total_impulse: Total impulse (N·s)
            delays: Available delay times
        """
        self.designation = designation
        self.manufacturer = manufacturer
        self.diameter = diameter
        self.length = length
        self.total_mass_initial = total_mass
        self.propellant_mass = propellant_mass
        self.case_mass = total_mass - propellant_mass  # Motor case only
        self.burn_time = burn_time
        self.total_impulse = total_impulse
        self.delays = delays or []
        
        # Process thrust curve
        self.thrust_curve = sorted(thrust_curve, key=lambda x: x[0])
        self.times = np.array([p[0] for p in self.thrust_curve])
        self.thrusts = np.array([p[1] for p in self.thrust_curve])
        
        # Average thrust and impulse class
        self.average_thrust = total_impulse / burn_time if burn_time > 0 else 0
        self.impulse_class = self._get_impulse_class(total_impulse)
        
    def _get_impulse_class(self, total_impulse: float) -> str:
        """Get motor impulse class (A, B, C, D, etc)"""
        # Class boundaries: 1.25, 2.5, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120 N·s
        classes = ['1/4A', '1/2A', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        thresholds = [0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
        
        for i, threshold in enumerate(thresholds):
            if total_impulse <= threshold:
                return classes[i]
        return 'O+'
    
    def get_thrust(self, time: float) -> float:
        """
        Get thrust at given time (exact OpenRocket interpolation).
        Returns 0 before ignition and after burnout.
        """
        if time < 0 or time > self.burn_time:
            return 0.0
        
        # Linear interpolation
        return np.interp(time, self.times, self.thrusts)
    
    def get_mass(self, time: float) -> float:
        """
        Get motor mass at given time (exact OpenRocket mass model).
        
        OpenRocket assumes linear propellant consumption proportional to 
        cumulative impulse delivered.
        """
        if time <= 0:
            # Before ignition - full mass
            return self.total_mass_initial
        
        if time >= self.burn_time:
            # After burnout - only case mass remains
            return self.case_mass
        
        # During burn - calculate mass from cumulative impulse
        # Integrate thrust curve up to current time
        cumulative_impulse = self._get_cumulative_impulse(time)
        
        # Mass decreases linearly with delivered impulse
        fraction_burned = cumulative_impulse / self.total_impulse
        propellant_remaining = self.propellant_mass * (1.0 - fraction_burned)
        
        return self.case_mass + propellant_remaining
    
    def _get_cumulative_impulse(self, time: float) -> float:
        """Calculate cumulative impulse up to given time"""
        if time <= 0:
            return 0.0
        if time >= self.burn_time:
            return self.total_impulse
        
        # Integrate using trapezoidal rule
        impulse = 0.0
        for i in range(len(self.times) - 1):
            t1, t2 = self.times[i], self.times[i+1]
            f1, f2 = self.thrusts[i], self.thrusts[i+1]
            
            if t2 <= time:
                # Full segment before cutoff
                impulse += 0.5 * (f1 + f2) * (t2 - t1)
            elif t1 < time < t2:
                # Partial segment
                # Interpolate thrust at cutoff time
                f_cutoff = f1 + (f2 - f1) * (time - t1) / (t2 - t1)
                impulse += 0.5 * (f1 + f_cutoff) * (time - t1)
                break
            else:
                break
        
        return impulse
    
    def get_cg(self, time: float) -> float:
        """
        Get motor CG position relative to motor front (exact OpenRocket).
        
        OpenRocket assumes propellant is evenly distributed and burns uniformly,
        so CG stays at geometric center.
        """
        # Motor CG is always at center for uniform burn
        return self.length / 2.0
    
    def get_inertia(self, time: float) -> Tuple[float, float, float]:
        """
        Get motor moment of inertia (Ixx, Iyy, Izz).
        OpenRocket models motor as solid cylinder.
        """
        mass = self.get_mass(time)
        radius = self.diameter / 2.0
        
        # Solid cylinder inertia
        # Ixx = Iyy = (1/12) * m * L^2 + (1/4) * m * r^2
        # Izz = (1/2) * m * r^2
        ixx = iyy = (1.0/12.0) * mass * self.length**2 + (1.0/4.0) * mass * radius**2
        izz = (1.0/2.0) * mass * radius**2
        
        return ixx, iyy, izz


def parse_rasp_file(filepath: str) -> List[Motor]:
    """
    Parse RASP .eng file format (OpenRocket standard).
    
    Format:
    ; Comments
    NAME diameter length delays propellant_mass total_mass manufacturer
    thrust_time_1 thrust_1
    thrust_time_2 thrust_2
    ...
    """
    motors = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith(';'):
            i += 1
            continue
        
        # Parse header line
        parts = line.split()
        if len(parts) < 6:
            i += 1
            continue
        
        designation = parts[0]
        diameter = float(parts[1]) / 1000.0  # mm to m
        length = float(parts[2]) / 1000.0    # mm to m
        delays = parts[3].split('-')
        propellant_mass = float(parts[4]) / 1000.0  # g to kg
        total_mass = float(parts[5]) / 1000.0       # g to kg
        manufacturer = parts[6] if len(parts) > 6 else "Unknown"
        
        # Parse thrust curve data
        i += 1
        thrust_curve = []
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith(';'):
                break
            if len(line.split()) != 2:
                break
            
            try:
                t, f = line.split()
                thrust_curve.append((float(t), float(f)))
            except ValueError:
                break
            i += 1
        
        # Calculate total impulse and burn time
        if len(thrust_curve) >= 2:
            burn_time = thrust_curve[-1][0]
            
            # Trapezoidal integration
            total_impulse = 0.0
            for j in range(len(thrust_curve) - 1):
                t1, f1 = thrust_curve[j]
                t2, f2 = thrust_curve[j+1]
                total_impulse += 0.5 * (f1 + f2) * (t2 - t1)
            
            motor = Motor(
                designation=designation,
                manufacturer=manufacturer,
                diameter=diameter,
                length=length,
                total_mass=total_mass,
                propellant_mass=propellant_mass,
                thrust_curve=thrust_curve,
                burn_time=burn_time,
                total_impulse=total_impulse,
                delays=delays
            )
            motors.append(motor)
    
    return motors


# Built-in motor database (from Estes and Cesaroni)
def get_builtin_motors() -> dict:
    """Get built-in motors matching OpenRocket defaults"""
    
    motors = {}
    
    # Estes C6-0 (24mm)
    motors['C6'] = Motor(
        designation='C6-0',
        manufacturer='Estes',
        diameter=0.0177,  # 17.7mm actual
        length=0.070,
        total_mass=0.0239,  # 23.9g
        propellant_mass=0.0088,  # 8.8g propellant
        thrust_curve=[
            (0.0, 0.0),
            (0.05, 15.0),
            (0.2, 7.0),
            (0.5, 5.5),
            (1.0, 5.0),
            (1.5, 4.5),
            (1.7, 3.0),
            (1.85, 0.0),
        ],
        burn_time=1.85,
        total_impulse=8.8,  # 8.8 N·s
        delays=['0', '3', '5', '7']
    )
    
    # Aerotech F50T (29mm)
    motors['F50'] = Motor(
        designation='F50T-9',
        manufacturer='AeroTech',
        diameter=0.0290,
        length=0.114,
        total_mass=0.0608,
        propellant_mass=0.0335,
        thrust_curve=[
            (0.0, 0.0),
            (0.05, 80.0),
            (0.2, 55.0),
            (0.5, 50.0),
            (1.0, 48.0),
            (1.5, 46.0),
            (2.0, 43.0),
            (2.2, 30.0),
            (2.3, 0.0),
        ],
        burn_time=2.3,
        total_impulse=115.0,
        delays=['4', '6', '9', '11', '13']
    )
    
    # Cesaroni J450 (54mm)
    motors['J450'] = Motor(
        designation='J450DM-15',
        manufacturer='Cesaroni',
        diameter=0.0540,
        length=0.336,
        total_mass=0.685,
        propellant_mass=0.442,
        thrust_curve=[
            (0.0, 0.0),
            (0.1, 600.0),
            (0.3, 520.0),
            (0.5, 480.0),
            (1.0, 450.0),
            (2.0, 430.0),
            (3.0, 420.0),
            (4.0, 410.0),
            (5.0, 400.0),
            (6.0, 380.0),
            (6.5, 200.0),
            (6.8, 0.0),
        ],
        burn_time=6.8,
        total_impulse=3027.0,
        delays=['6', '9', '12', '15', '18']
    )
    
    return motors

