"""
Robust atmospheric models using national database libraries.

This module provides multiple atmospheric models:
- ISA (International Standard Atmosphere) - fallback
- NRLMSISE-00 - for high-altitude modeling (up to 1000 km)
- Weather data integration - from ECMWF, NOAA GFS, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Try to import optional dependencies
try:
    import nrlmsise00
    NRLMSISE_AVAILABLE = True
except ImportError:
    NRLMSISE_AVAILABLE = False
    nrlmsise00 = None

try:
    import netCDF4
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    netCDF4 = None

from .openrocket_atmosphere import ISAAtmosphere


@dataclass
class AtmosphericProperties:
    """Container for atmospheric properties at a given altitude."""

    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m^3
    speed_of_sound: float  # m/s
    viscosity: float  # Pa·s
    molecular_weight: Optional[float] = None  # kg/mol (for NRLMSISE-00)


class NRLMSISE00Atmosphere:
    """
    NRLMSISE-00 atmospheric model for high-altitude modeling (0-1000 km).
    
    This is the Naval Research Laboratory Mass Spectrometer and Incoherent Scatter
    model, used by RocketPy for accurate high-altitude atmospheric properties.
    """

    def __init__(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        year: int = 2024,
        day_of_year: int = 1,
        f107: float = 150.0,  # Solar flux at 10.7 cm
        f107a: float = 150.0,  # 81-day average of f107
        ap: float = 4.0,  # Geomagnetic activity index
    ):
        """
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            year: Year (for time-dependent calculations)
            day_of_year: Day of year (1-365/366)
            f107: Solar flux at 10.7 cm (affects upper atmosphere)
            f107a: 81-day average of f107
            ap: Geomagnetic activity index (affects upper atmosphere)
        """
        if not NRLMSISE_AVAILABLE:
            raise ImportError(
                "nrlmsise00 package not available. Install with: pip install nrlmsise00"
            )

        self.latitude = latitude
        self.longitude = longitude
        self.year = year
        self.day_of_year = day_of_year
        self.f107 = f107
        self.f107a = f107a
        self.ap = ap

    def get_properties(self, altitude: float) -> AtmosphericProperties:
        """
        Get atmospheric properties using NRLMSISE-00 model.

        Args:
            altitude: Geometric altitude above sea level (m)

        Returns:
            AtmosphericProperties object
        """
        # NRLMSISE-00 expects altitude in km
        alt_km = altitude / 1000.0

        # Call NRLMSISE-00 model
        output = nrlmsise00.run(
            self.year,
            self.day_of_year,
            alt_km,
            self.latitude,
            self.longitude,
            local_time=12.0,  # Local solar time (hours)
            f107=self.f107,
            f107a=self.f107a,
            ap=self.ap,
        )

        # Extract properties from output
        # NRLMSISE-00 returns densities in particles/cm^3, temperature in K
        # We need to convert to standard units
        temperature = output[0][1]  # K
        total_density = output[0][5]  # Total mass density (g/cm^3)

        # Convert density from g/cm^3 to kg/m^3
        density = total_density * 1000.0  # kg/m^3

        # Calculate pressure using ideal gas law
        # Using average molecular weight from NRLMSISE-00
        R = 287.05  # J/(kg·K) - specific gas constant for air
        pressure = density * R * temperature  # Pa

        # Speed of sound
        gamma = 1.4  # Heat capacity ratio
        speed_of_sound = math.sqrt(gamma * R * temperature)  # m/s

        # Dynamic viscosity (Sutherland's formula)
        T_ref = 288.15  # K
        mu_ref = 1.7894e-5  # Pa·s
        S = 110.4  # K
        viscosity = mu_ref * (temperature / T_ref) ** 1.5 * (T_ref + S) / (temperature + S)

        return AtmosphericProperties(
            temperature=temperature,
            pressure=pressure,
            density=density,
            speed_of_sound=speed_of_sound,
            viscosity=viscosity,
        )


class WeatherDataAtmosphere:
    """
    Atmospheric model using weather data from external sources (ECMWF, NOAA GFS, etc.).
    
    This class can load atmospheric profiles from NetCDF files containing
    weather reanalysis or forecast data.
    """

    def __init__(self, data_file: Optional[str] = None, fallback: Optional[ISAAtmosphere] = None):
        """
        Args:
            data_file: Path to NetCDF file with weather data
            fallback: Fallback atmosphere model if data unavailable
        """
        self.data_file = data_file
        self.fallback = fallback or ISAAtmosphere()
        self._data = None
        self._altitudes = None
        self._temperatures = None
        self._pressures = None
        self._densities = None

        if data_file and NETCDF_AVAILABLE:
            self._load_data(data_file)

    def _load_data(self, filename: str):
        """Load atmospheric data from NetCDF file."""
        try:
            with netCDF4.Dataset(filename, "r") as nc:
                # Try to find altitude/pressure levels
                # Common variable names in weather data
                if "level" in nc.variables:
                    levels = nc.variables["level"][:]
                elif "pressure" in nc.variables:
                    levels = nc.variables["pressure"][:]
                else:
                    # Fallback: use standard pressure levels
                    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]

                # Extract temperature and pressure data
                # This is a simplified version - real implementation would need
                # to handle different data formats and coordinate systems
                if "t" in nc.variables:
                    self._temperatures = nc.variables["t"][:]
                if "sp" in nc.variables:  # Surface pressure
                    surface_pressure = nc.variables["sp"][:]
                elif "ps" in nc.variables:
                    surface_pressure = nc.variables["ps"][:]

                # Convert pressure levels to altitudes (simplified)
                self._altitudes = []
                self._pressures = []
                for level in levels:
                    # Approximate altitude from pressure level (hPa)
                    # Using barometric formula
                    p_hpa = float(level)
                    if p_hpa > 0:
                        # Simplified conversion
                        h = 44330 * (1 - (p_hpa / 1013.25) ** 0.1903)  # meters
                        self._altitudes.append(h)
                        self._pressures.append(p_hpa * 100.0)  # Convert to Pa

        except Exception as e:
            print(f"Warning: Could not load weather data from {filename}: {e}")
            print("Falling back to ISA model")
            self._data = None

    def get_properties(self, altitude: float) -> AtmosphericProperties:
        """
        Get atmospheric properties from weather data or fallback.

        Args:
            altitude: Geometric altitude above sea level (m)

        Returns:
            AtmosphericProperties object
        """
        # If no weather data loaded, use fallback
        if self._altitudes is None or len(self._altitudes) == 0:
            props = self.fallback.get_properties(altitude)
            return AtmosphericProperties(
                temperature=props["temperature"],
                pressure=props["pressure"],
                density=props["density"],
                speed_of_sound=props["speed_of_sound"],
                viscosity=props["viscosity"],
            )

        # Interpolate from weather data
        # Find altitude bracket
        if altitude <= self._altitudes[0]:
            idx = 0
        elif altitude >= self._altitudes[-1]:
            idx = len(self._altitudes) - 1
        else:
            # Linear interpolation
            for i in range(len(self._altitudes) - 1):
                if self._altitudes[i] <= altitude < self._altitudes[i + 1]:
                    idx = i
                    frac = (altitude - self._altitudes[i]) / (
                        self._altitudes[i + 1] - self._altitudes[i]
                    )
                    break
            else:
                idx = len(self._altitudes) - 1
                frac = 0.0

        # Interpolate properties
        if idx < len(self._altitudes) - 1 and "frac" in locals():
            T = self._temperatures[idx] + frac * (
                self._temperatures[idx + 1] - self._temperatures[idx]
            )
            P = self._pressures[idx] + frac * (self._pressures[idx + 1] - self._pressures[idx])
        else:
            T = self._temperatures[idx]
            P = self._pressures[idx]

        # Calculate density and other properties
        R = 287.05  # J/(kg·K)
        density = P / (R * T)  # kg/m^3

        gamma = 1.4
        speed_of_sound = math.sqrt(gamma * R * T)

        T_ref = 288.15
        mu_ref = 1.7894e-5
        S = 110.4
        viscosity = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

        return AtmosphericProperties(
            temperature=float(T),
            pressure=float(P),
            density=float(density),
            speed_of_sound=float(speed_of_sound),
            viscosity=float(viscosity),
        )


class HybridAtmosphere:
    """
    Hybrid atmospheric model that uses different models at different altitudes.
    
    - ISA or Weather Data: 0-86 km (troposphere, stratosphere, mesosphere)
    - NRLMSISE-00: 86-1000 km (thermosphere, exosphere)
    """

    def __init__(
        self,
        low_altitude_model: ISAAtmosphere | WeatherDataAtmosphere | None = None,
        high_altitude_model: NRLMSISE00Atmosphere | None = None,
        transition_altitude: float = 86000.0,  # 86 km
    ):
        """
        Args:
            low_altitude_model: Model for altitudes below transition (ISA or WeatherData)
            high_altitude_model: Model for altitudes above transition (NRLMSISE-00)
            transition_altitude: Altitude where models switch (m)
        """
        self.low_altitude_model = low_altitude_model or ISAAtmosphere()
        self.high_altitude_model = high_altitude_model
        self.transition_altitude = transition_altitude

    def get_properties(self, altitude: float) -> AtmosphericProperties:
        """
        Get atmospheric properties using appropriate model for altitude.

        Args:
            altitude: Geometric altitude above sea level (m)

        Returns:
            AtmosphericProperties object
        """
        if altitude < self.transition_altitude:
            # Use low-altitude model
            if isinstance(self.low_altitude_model, ISAAtmosphere):
                props = self.low_altitude_model.get_properties(altitude)
                return AtmosphericProperties(
                    temperature=props["temperature"],
                    pressure=props["pressure"],
                    density=props["density"],
                    speed_of_sound=props["speed_of_sound"],
                    viscosity=props["viscosity"],
                )
            else:
                return self.low_altitude_model.get_properties(altitude)
        else:
            # Use high-altitude model (NRLMSISE-00)
            if self.high_altitude_model is None:
                if NRLMSISE_AVAILABLE:
                    self.high_altitude_model = NRLMSISE00Atmosphere()
                else:
                    # Fallback: extend ISA model (not accurate but better than nothing)
                    props = self.low_altitude_model.get_properties(self.transition_altitude)
                    # Extrapolate exponentially
                    scale = math.exp(-(altitude - self.transition_altitude) / 8000.0)
                    return AtmosphericProperties(
                        temperature=props["temperature"] * scale,
                        pressure=props["pressure"] * scale,
                        density=props["density"] * scale,
                        speed_of_sound=props["speed_of_sound"] * math.sqrt(scale),
                        viscosity=props["viscosity"] * scale,
                    )

            return self.high_altitude_model.get_properties(altitude)

