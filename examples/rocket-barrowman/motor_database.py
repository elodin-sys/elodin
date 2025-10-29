"""
Motor Database and Thrust Curve Integration

Supports RASP .eng file format (standard used by thrustcurve.org and OpenRocket)

File Format:
; Comment lines start with semicolon
<name> <diameter> <length> <delays> <prop_weight> <total_weight> <manufacturer>
<time1> <thrust1>
<time2> <thrust2>
...

Example:
Estes C6 18 70 5 6.2 11.0 Estes
0.0 0.0
0.02 5.5
1.7 5.5
1.8 0.0
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import glob


@dataclass
class MotorData:
    """Complete motor specification"""
    # Identification
    manufacturer: str
    designation: str  # e.g., "C6-5"
    common_name: str  # e.g., "C6"
    
    # Physical properties
    diameter: float  # mm
    length: float  # mm
    propellant_mass: float  # g
    total_mass: float  # kg (includes casing)
    casing_mass: float  # kg
    
    # Performance
    delays: str  # Available delays, e.g., "5-7-9"
    total_impulse: float  # N·s
    average_thrust: float  # N
    max_thrust: float  # N
    burn_time: float  # s
    
    # Thrust curve
    time_points: np.ndarray  # s
    thrust_points: np.ndarray  # N
    
    # Metadata
    motor_class: str  # A, B, C, D, E, F, G, etc.
    data_source: str  # File path or source
    
    def __repr__(self):
        return (f"Motor({self.manufacturer} {self.designation}, "
                f"{self.diameter}mm×{self.length}mm, "
                f"{self.total_impulse:.1f}N·s, "
                f"Iavg={self.average_thrust:.1f}N)")
    
    def get_thrust(self, t: float) -> float:
        """Get thrust at time t using linear interpolation"""
        return float(np.interp(t, self.time_points, self.thrust_points))
    
    def get_mass(self, t: float) -> float:
        """Get motor mass at time t (assumes linear propellant consumption)"""
        if t < 0:
            return self.total_mass
        elif t >= self.burn_time:
            return self.casing_mass
        else:
            # Linear propellant depletion
            burn_fraction = t / self.burn_time
            remaining_propellant = self.propellant_mass * (1.0 - burn_fraction)
            return self.casing_mass + remaining_propellant
    
    def get_cg_offset(self, t: float, motor_mount_position: float) -> float:
        """
        Get CG offset from motor mount position due to propellant burn
        
        Args:
            t: Time in seconds
            motor_mount_position: Position of motor mount from nose tip (m)
        
        Returns:
            CG offset in meters (assumes propellant burns from aft to forward)
        """
        if t < 0:
            # Full propellant, CG at motor center
            return motor_mount_position + self.length / 2000.0  # Convert mm to m
        elif t >= self.burn_time:
            # Empty, CG slightly forward (casing only)
            return motor_mount_position + self.length / 2500.0
        else:
            # Linear CG shift forward as propellant burns
            burn_fraction = t / self.burn_time
            cg_shift = burn_fraction * (self.length / 10000.0)  # Small shift
            return motor_mount_position + self.length / 2000.0 - cg_shift
    
    @property
    def impulse_class(self) -> str:
        """Get motor impulse class (A, B, C, etc.)"""
        impulse = self.total_impulse
        if impulse <= 2.5:
            return "A"
        elif impulse <= 5.0:
            return "B"
        elif impulse <= 10.0:
            return "C"
        elif impulse <= 20.0:
            return "D"
        elif impulse <= 40.0:
            return "E"
        elif impulse <= 80.0:
            return "F"
        elif impulse <= 160.0:
            return "G"
        elif impulse <= 320.0:
            return "H"
        elif impulse <= 640.0:
            return "I"
        elif impulse <= 1280.0:
            return "J"
        elif impulse <= 2560.0:
            return "K"
        elif impulse <= 5120.0:
            return "L"
        elif impulse <= 10240.0:
            return "M"
        elif impulse <= 20480.0:
            return "N"
        else:
            return "O+"


class MotorDatabase:
    """Motor database management"""
    
    def __init__(self):
        self.motors: Dict[str, MotorData] = {}
        self._by_manufacturer: Dict[str, List[str]] = {}
        self._by_class: Dict[str, List[str]] = {}
    
    def load_eng_file(self, filepath: str) -> MotorData:
        """
        Load a RASP .eng file
        
        Args:
            filepath: Path to .eng file
        
        Returns:
            MotorData object
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith(';')]
        
        if len(lines) < 2:
            raise ValueError(f"Invalid .eng file: {filepath}")
        
        # Parse header line
        header = lines[0].split()
        if len(header) < 7:
            raise ValueError(f"Invalid header in {filepath}: {lines[0]}")
        
        designation = header[0]
        diameter = float(header[1])  # mm
        length = float(header[2])  # mm
        delays = header[3]
        propellant_mass = float(header[4])  # g
        total_mass = float(header[5])  # g
        manufacturer = ' '.join(header[6:])  # Handle multi-word manufacturers
        
        # Parse thrust curve data
        time_points = []
        thrust_points = []
        
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                time_points.append(float(parts[0]))
                thrust_points.append(float(parts[1]))
        
        time_array = np.array(time_points)
        thrust_array = np.array(thrust_points)
        
        # Calculate derived properties
        total_impulse = np.trapz(thrust_array, time_array)
        burn_time = time_array[-1] - time_array[0]
        average_thrust = total_impulse / burn_time if burn_time > 0 else 0.0
        max_thrust = np.max(thrust_array)
        
        # Convert masses to kg
        propellant_mass_kg = propellant_mass / 1000.0
        total_mass_kg = total_mass / 1000.0
        casing_mass_kg = total_mass_kg - propellant_mass_kg
        
        # Extract common name (remove delay designation)
        common_name = designation.split('-')[0]
        
        # Determine motor class
        motor_class = common_name[0] if common_name else "?"
        
        motor = MotorData(
            manufacturer=manufacturer,
            designation=designation,
            common_name=common_name,
            diameter=diameter,
            length=length,
            propellant_mass=propellant_mass_kg,
            total_mass=total_mass_kg,
            casing_mass=casing_mass_kg,
            delays=delays,
            total_impulse=total_impulse,
            average_thrust=average_thrust,
            max_thrust=max_thrust,
            burn_time=burn_time,
            time_points=time_array,
            thrust_points=thrust_array,
            motor_class=motor_class,
            data_source=filepath
        )
        
        return motor
    
    def add_motor(self, motor: MotorData):
        """Add a motor to the database"""
        key = f"{motor.manufacturer}_{motor.designation}"
        self.motors[key] = motor
        
        # Index by manufacturer
        if motor.manufacturer not in self._by_manufacturer:
            self._by_manufacturer[motor.manufacturer] = []
        self._by_manufacturer[motor.manufacturer].append(key)
        
        # Index by class
        if motor.motor_class not in self._by_class:
            self._by_class[motor.motor_class] = []
        self._by_class[motor.motor_class].append(key)
    
    def load_directory(self, directory: str, pattern: str = "*.eng"):
        """Load all .eng files from a directory"""
        eng_files = glob.glob(os.path.join(directory, pattern))
        
        loaded = 0
        failed = []
        
        for filepath in eng_files:
            try:
                motor = self.load_eng_file(filepath)
                self.add_motor(motor)
                loaded += 1
            except Exception as e:
                failed.append((filepath, str(e)))
        
        print(f"Loaded {loaded} motors from {directory}")
        if failed:
            print(f"Failed to load {len(failed)} files:")
            for filepath, error in failed[:5]:  # Show first 5 errors
                print(f"  {os.path.basename(filepath)}: {error}")
    
    def get_motor(self, manufacturer: str, designation: str) -> Optional[MotorData]:
        """Get a specific motor"""
        key = f"{manufacturer}_{designation}"
        return self.motors.get(key)
    
    def list_motors(self, manufacturer: Optional[str] = None, 
                   motor_class: Optional[str] = None) -> List[MotorData]:
        """List motors, optionally filtered by manufacturer or class"""
        if manufacturer:
            keys = self._by_manufacturer.get(manufacturer, [])
            return [self.motors[k] for k in keys]
        elif motor_class:
            keys = self._by_class.get(motor_class, [])
            return [self.motors[k] for k in keys]
        else:
            return list(self.motors.values())
    
    def find_motors(self, diameter_mm: float, max_length_mm: float = None,
                   impulse_class: str = None) -> List[MotorData]:
        """
        Find motors matching physical constraints
        
        Args:
            diameter_mm: Motor diameter in mm (e.g., 18, 24, 29)
            max_length_mm: Maximum motor length in mm
            impulse_class: Desired impulse class (A, B, C, etc.)
        
        Returns:
            List of matching motors
        """
        results = []
        
        for motor in self.motors.values():
            # Check diameter (allow 1mm tolerance)
            if abs(motor.diameter - diameter_mm) > 1.0:
                continue
            
            # Check length
            if max_length_mm and motor.length > max_length_mm:
                continue
            
            # Check impulse class
            if impulse_class and motor.impulse_class != impulse_class:
                continue
            
            results.append(motor)
        
        # Sort by total impulse
        results.sort(key=lambda m: m.total_impulse)
        
        return results
    
    def print_summary(self):
        """Print database summary"""
        print(f"\nMotor Database Summary:")
        print(f"  Total motors: {len(self.motors)}")
        print(f"  Manufacturers: {len(self._by_manufacturer)}")
        print(f"  Impulse classes: {sorted(self._by_class.keys())}")
        
        print(f"\nMotors by class:")
        for cls in sorted(self._by_class.keys()):
            count = len(self._by_class[cls])
            print(f"  {cls}: {count} motors")


def create_sample_motors():
    """Create sample motors for testing (when no .eng files available)"""
    db = MotorDatabase()
    
    # Estes C6-5 (Model rocket standard)
    c6 = MotorData(
        manufacturer="Estes",
        designation="C6-5",
        common_name="C6",
        diameter=18.0,
        length=70.0,
        propellant_mass=0.0088,  # 8.8g
        total_mass=0.0175,  # 17.5g
        casing_mass=0.0087,  # 8.7g
        delays="3-5-7",
        total_impulse=8.82,
        average_thrust=5.0,
        max_thrust=6.5,
        burn_time=1.76,
        time_points=np.array([0.0, 0.02, 0.2, 1.5, 1.76]),
        thrust_points=np.array([0.0, 6.5, 5.5, 5.0, 0.0]),
        motor_class="C",
        data_source="built-in"
    )
    
    # Aerotech F50-6T (Mid-power standard)
    f50 = MotorData(
        manufacturer="Aerotech",
        designation="F50-6T",
        common_name="F50",
        diameter=29.0,
        length=114.0,
        propellant_mass=0.0385,  # 38.5g
        total_mass=0.0726,  # 72.6g
        casing_mass=0.0341,  # 34.1g
        delays="4-6-8-10",
        total_impulse=57.6,
        average_thrust=50.0,
        max_thrust=64.0,
        burn_time=1.15,
        time_points=np.array([0.0, 0.05, 0.1, 0.9, 1.05, 1.15]),
        thrust_points=np.array([0.0, 60.0, 55.0, 48.0, 42.0, 0.0]),
        motor_class="F",
        data_source="built-in"
    )
    
    # Cesaroni J450 (High power)
    j450 = MotorData(
        manufacturer="Cesaroni",
        designation="J450-10A",
        common_name="J450",
        diameter=54.0,
        length=355.0,
        propellant_mass=0.388,  # 388g
        total_mass=0.669,  # 669g
        casing_mass=0.281,  # 281g
        delays="P",  # Plugged (no ejection charge)
        total_impulse=512.0,
        average_thrust=450.0,
        max_thrust=524.0,
        burn_time=1.14,
        time_points=np.array([0.0, 0.05, 0.2, 0.8, 1.0, 1.14]),
        thrust_points=np.array([0.0, 500.0, 480.0, 440.0, 380.0, 0.0]),
        motor_class="J",
        data_source="built-in"
    )
    
    db.add_motor(c6)
    db.add_motor(f50)
    db.add_motor(j450)
    
    return db


def main():
    """Demo and test motor database"""
    print("=" * 60)
    print("ROCKET MOTOR DATABASE")
    print("=" * 60)
    
    # Create sample database
    db = create_sample_motors()
    db.print_summary()
    
    # Test motor lookup
    print(f"\n\nLooking up Estes C6-5:")
    c6 = db.get_motor("Estes", "C6-5")
    if c6:
        print(c6)
        print(f"  Thrust at t=0.5s: {c6.get_thrust(0.5):.2f}N")
        print(f"  Mass at t=0.5s: {c6.get_mass(0.5)*1000:.2f}g")
    
    # Test motor search
    print(f"\n\n18mm diameter motors:")
    motors_18mm = db.find_motors(diameter_mm=18.0)
    for motor in motors_18mm:
        print(f"  {motor}")
    
    print(f"\n\nF-class motors:")
    motors_f = db.list_motors(motor_class="F")
    for motor in motors_f:
        print(f"  {motor}")
    
    # Plot thrust curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, motor in enumerate([c6, db.get_motor("Aerotech", "F50-6T"), 
                                     db.get_motor("Cesaroni", "J450-10A")]):
            if motor:
                axes[idx].plot(motor.time_points, motor.thrust_points, 'b-', linewidth=2)
                axes[idx].fill_between(motor.time_points, motor.thrust_points, alpha=0.3)
                axes[idx].set_xlabel('Time (s)')
                axes[idx].set_ylabel('Thrust (N)')
                axes[idx].set_title(f'{motor.manufacturer} {motor.designation}\n'
                                  f'{motor.total_impulse:.1f}N·s total impulse')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim(left=0)
                axes[idx].set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig('/home/kush-mahajan/elodin/examples/rocket-barrowman/motor_thrust_curves.png', dpi=150)
        print(f"\n\nThrust curves saved to motor_thrust_curves.png")
        plt.show()
    except ImportError:
        print("\nMatplotlib not available, skipping plots")


if __name__ == "__main__":
    main()

