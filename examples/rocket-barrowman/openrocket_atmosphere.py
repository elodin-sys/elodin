"""
OpenRocket atmospheric model - exact ISA implementation.
"""

import math


class ISAAtmosphere:
    """
    International Standard Atmosphere (ISA) - exact OpenRocket implementation.
    Valid from sea level to 86 km altitude.
    """
    
    # Sea level constants
    P0 = 101325.0  # Pa
    T0 = 288.15    # K
    RHO0 = 1.225   # kg/m^3
    G = 9.80665    # m/s^2 (standard gravity)
    R = 287.05     # J/(kg·K) (specific gas constant for dry air)
    GAMMA = 1.4    # Heat capacity ratio for air
    MU0 = 1.7894e-5  # Pa·s (dynamic viscosity at sea level)
    
    # Atmospheric layers (altitude, lapse rate, base temperature, base pressure)
    LAYERS = [
        (0.0,     -0.0065, 288.15,   101325.0),     # Troposphere
        (11000.0,  0.0,    216.65,   22632.1),      # Tropopause
        (20000.0,  0.001,  216.65,   5474.89),      # Stratosphere 1
        (32000.0,  0.0028, 228.65,   868.019),      # Stratosphere 2
        (47000.0,  0.0,    270.65,   110.906),      # Stratopause
        (51000.0, -0.0028, 270.65,   66.9389),      # Mesosphere 1
        (71000.0, -0.002,  214.65,   3.95642),      # Mesosphere 2
        (86000.0,  0.0,    186.946,  0.373384),     # Mesopause
    ]
    
    def __init__(self, temperature_offset: float = 0.0, pressure_ratio: float = 1.0):
        """
        Args:
            temperature_offset: Temperature offset from ISA (K)
            pressure_ratio: Pressure ratio multiplier (for weather variations)
        """
        self.temperature_offset = temperature_offset
        self.pressure_ratio = pressure_ratio
    
    def get_properties(self, altitude: float) -> dict:
        """
        Get atmospheric properties at given altitude (exact OpenRocket).
        
        Args:
            altitude: Geometric altitude (m)
            
        Returns:
            dict with keys: temperature, pressure, density, speed_of_sound, viscosity, mach
        """
        # Clamp altitude to valid range
        altitude = max(0.0, min(altitude, 86000.0))
        
        # Find appropriate atmospheric layer
        layer_idx = 0
        for i in range(len(self.LAYERS) - 1):
            if altitude >= self.LAYERS[i][0] and altitude < self.LAYERS[i+1][0]:
                layer_idx = i
                break
        
        h_base, lapse_rate, T_base, P_base = self.LAYERS[layer_idx]
        
        # Height above layer base
        dh = altitude - h_base
        
        # Temperature
        temperature = T_base + lapse_rate * dh + self.temperature_offset
        temperature = max(temperature, 1.0)  # Prevent zero/negative
        
        # Pressure (barometric formula)
        if abs(lapse_rate) < 1e-6:
            # Isothermal layer
            pressure = P_base * math.exp(-self.G * dh / (self.R * T_base))
        else:
            # Gradient layer
            pressure = P_base * (T_base / temperature) ** (self.G / (self.R * lapse_rate))
        
        pressure *= self.pressure_ratio
        
        # Density (ideal gas law)
        density = pressure / (self.R * temperature)
        
        # Speed of sound
        speed_of_sound = math.sqrt(self.GAMMA * self.R * temperature)
        
        # Dynamic viscosity (Sutherland's formula)
        T_ref = 288.15  # K
        mu_ref = 1.7894e-5  # Pa·s
        S = 110.4  # K (Sutherland's constant for air)
        viscosity = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': speed_of_sound,
            'viscosity': viscosity,
        }
    
    def get_temperature(self, altitude: float) -> float:
        """Get temperature at altitude (K)"""
        return self.get_properties(altitude)['temperature']
    
    def get_pressure(self, altitude: float) -> float:
        """Get pressure at altitude (Pa)"""
        return self.get_properties(altitude)['pressure']
    
    def get_density(self, altitude: float) -> float:
        """Get density at altitude (kg/m^3)"""
        return self.get_properties(altitude)['density']
    
    def get_speed_of_sound(self, altitude: float) -> float:
        """Get speed of sound at altitude (m/s)"""
        return self.get_properties(altitude)['speed_of_sound']
    
    def get_viscosity(self, altitude: float) -> float:
        """Get dynamic viscosity at altitude (Pa·s)"""
        return self.get_properties(altitude)['viscosity']


class WindModel:
    """Wind model for rocket simulation"""
    
    def __init__(self, wind_speed: float = 0.0, wind_direction: float = 0.0,
                 turbulence_intensity: float = 0.0):
        """
        Args:
            wind_speed: Average wind speed (m/s)
            wind_direction: Wind direction (radians, 0 = north, pi/2 = east)
            turbulence_intensity: Turbulence intensity (0-1)
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.turbulence_intensity = turbulence_intensity
        
        # Wind components
        self.wind_north = wind_speed * math.cos(wind_direction)
        self.wind_east = wind_speed * math.sin(wind_direction)
        self.wind_up = 0.0
    
    def get_wind(self, altitude: float, time: float = 0.0) -> tuple:
        """
        Get wind velocity vector at altitude and time.
        
        Returns:
            (north, east, up) velocity components (m/s)
        """
        # Base wind
        north = self.wind_north
        east = self.wind_east
        up = self.wind_up
        
        # Altitude dependence (power law)
        if altitude > 0:
            # Wind increases with altitude
            altitude_factor = (altitude / 10.0) ** 0.143  # 1/7 power law
            north *= altitude_factor
            east *= altitude_factor
        
        # Turbulence (simplified - OpenRocket uses more complex model)
        if self.turbulence_intensity > 0:
            import random
            turb_scale = self.turbulence_intensity * self.wind_speed * 0.3
            north += random.gauss(0, turb_scale)
            east += random.gauss(0, turb_scale)
            up += random.gauss(0, turb_scale * 0.5)
        
        return north, east, up

