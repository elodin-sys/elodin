"""RocketPy-inspired environment module for the Elodin solver.

This module packages atmospheric models and dynamic wind models so they can be
consumed by the RocketPy-style solver. The API mirrors the naming of RocketPy's
:mod:`rocketpy.environment` and supports:
- ISA (International Standard Atmosphere) - basic fallback
- NRLMSISE-00 - high-altitude atmospheric modeling (0-1000 km)
- Weather data integration - from ECMWF, NOAA GFS, etc.
- Hybrid models - combining multiple models for different altitude ranges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Tuple

from ..environment.atmospheric_models import (
    ISAAtmosphere,
    NRLMSISE00Atmosphere,
    WeatherDataAtmosphere,
    HybridAtmosphere,
    NRLMSISE_AVAILABLE,
    NETCDF_AVAILABLE,
)
from ..environment.dynamic_wind import DynamicWindModel


@dataclass
class Environment:
    """
    Container for atmosphere and wind models with support for multiple atmospheric models.

    Supports:
    - ISA (International Standard Atmosphere) - default
    - NRLMSISE-00 - for high-altitude modeling (requires nrlmsise00 package)
    - Weather Data - from NetCDF files (requires netCDF4 package)
    - Hybrid - combines models for different altitude ranges
    """

    atmosphere: ISAAtmosphere | NRLMSISE00Atmosphere | WeatherDataAtmosphere | HybridAtmosphere
    wind_model: DynamicWindModel
    elevation: float  # Ground elevation above sea level (m)

    def __init__(
        self,
        atmosphere: (
            ISAAtmosphere | NRLMSISE00Atmosphere | WeatherDataAtmosphere | HybridAtmosphere | None
        ) = None,
        wind_model: DynamicWindModel | None = None,
        elevation: float = 0.0,
    ) -> None:
        self.atmosphere = atmosphere or ISAAtmosphere()
        self.wind_model = wind_model or DynamicWindModel()
        self.elevation = elevation

    # ------------------------------------------------------------------
    # Atmosphere helpers (RocketPy naming)
    # ------------------------------------------------------------------

    def set_atmospheric_model(
        self,
        *,
        type: str = "ISA",  # noqa: A003
        latitude: float = 0.0,
        longitude: float = 0.0,
        year: int = 2024,
        day_of_year: int = 1,
        f107: float = 150.0,
        f107a: float = 150.0,
        ap: float = 4.0,
        weather_file: Optional[str] = None,
        transition_altitude: float = 86000.0,
        **kwargs,
    ) -> None:
        """
        Set atmospheric model type.

        Args:
            type: Model type - "ISA", "NRLMSISE00", "WEATHER", or "HYBRID"
            latitude: Latitude in degrees (for NRLMSISE-00)
            longitude: Longitude in degrees (for NRLMSISE-00)
            year: Year (for NRLMSISE-00)
            day_of_year: Day of year 1-365/366 (for NRLMSISE-00)
            f107: Solar flux at 10.7 cm (for NRLMSISE-00)
            f107a: 81-day average of f107 (for NRLMSISE-00)
            ap: Geomagnetic activity index (for NRLMSISE-00)
            weather_file: Path to NetCDF weather data file (for WEATHER)
            transition_altitude: Altitude where models switch in HYBRID mode (m)
            **kwargs: Additional arguments passed to ISA model
        """
        model_type = type.upper()

        if model_type == "ISA":
            self.atmosphere = ISAAtmosphere(**kwargs)
        elif model_type == "NRLMSISE00":
            if not NRLMSISE_AVAILABLE:
                raise ImportError(
                    "NRLMSISE-00 model requires nrlmsise00 package. "
                    "Install with: pip install nrlmsise00"
                )
            self.atmosphere = NRLMSISE00Atmosphere(
                latitude=latitude,
                longitude=longitude,
                year=year,
                day_of_year=day_of_year,
                f107=f107,
                f107a=f107a,
                ap=ap,
            )
        elif model_type == "WEATHER":
            if not NETCDF_AVAILABLE:
                raise ImportError(
                    "Weather data model requires netCDF4 package. Install with: pip install netCDF4"
                )
            if weather_file is None:
                raise ValueError("weather_file is required for WEATHER model")
            self.atmosphere = WeatherDataAtmosphere(data_file=weather_file)
        elif model_type == "HYBRID":
            # Create hybrid model with ISA/NRLMSISE-00
            low_model = ISAAtmosphere(**kwargs)
            high_model = None
            if NRLMSISE_AVAILABLE:
                high_model = NRLMSISE00Atmosphere(
                    latitude=latitude,
                    longitude=longitude,
                    year=year,
                    day_of_year=day_of_year,
                    f107=f107,
                    f107a=f107a,
                    ap=ap,
                )
            self.atmosphere = HybridAtmosphere(
                low_altitude_model=low_model,
                high_altitude_model=high_model,
                transition_altitude=transition_altitude,
            )
        else:
            raise ValueError(
                f"Unknown atmospheric model type: {type}. "
                "Supported types: ISA, NRLMSISE00, WEATHER, HYBRID"
            )

    def air_properties(self, altitude: float) -> dict[str, float]:
        """Get air properties at altitude above ground level.

        Args:
            altitude: Altitude above ground level (m)

        Returns:
            Dictionary with density, pressure, temperature, viscosity, speed_of_sound
        """
        # Convert AGL to ASL (above sea level) for calculation
        altitude_asl = altitude + self.elevation

        # Get properties from atmosphere model
        if isinstance(self.atmosphere, ISAAtmosphere):
            # ISA returns dict directly
            return self.atmosphere.get_properties(altitude_asl)
        else:
            # Other models return AtmosphericProperties object
            props = self.atmosphere.get_properties(altitude_asl)
            return {
                "temperature": props.temperature,
                "pressure": props.pressure,
                "density": props.density,
                "speed_of_sound": props.speed_of_sound,
                "viscosity": props.viscosity,
            }

    # ------------------------------------------------------------------
    # Wind helpers (RocketPy naming)
    # ------------------------------------------------------------------

    def set_wind_model(self, *, profile_points: Iterable, **kwargs) -> None:
        self.wind_model = DynamicWindModel(profile_points=profile_points, **kwargs)

    def wind_velocity(self, altitude: float, time: float) -> Tuple[float, float, float]:
        return self.wind_model.get_wind(altitude, time)

    # ------------------------------------------------------------------
    # Convenience factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_coordinates(
        cls,
        latitude: float,
        longitude: float,
        datetime_obj: datetime,
        elevation: float = 0.0,
        use_weather_data: bool = True,
        use_nrlmsise: bool = True,
    ) -> "Environment":
        """
        Create Environment automatically from coordinates and datetime.

        Fetches weather data from national databases (ECMWF ERA5) and sets up
        atmospheric and wind models automatically.

        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)
            datetime_obj: Datetime object for the requested time
            elevation: Ground elevation above sea level (m). Default: 0.0
            use_weather_data: If True, fetch and use weather data. Default: True
            use_nrlmsise: If True, use NRLMSISE-00 for high altitudes. Default: True

        Returns:
            Environment object configured with weather data

        Example:
            >>> from datetime import datetime
            >>> from core import Environment
            >>>
            >>> # Spaceport America launch conditions
            >>> launch_time = datetime(2024, 6, 15, 12, 0)
            >>> env = Environment.from_coordinates(
            ...     latitude=33.0,
            ...     longitude=-106.5,
            ...     datetime_obj=launch_time,
            ...     elevation=1400.0,
            ... )
            >>> # Now use env for simulation
            >>> props = env.air_properties(altitude=1000.0)
        """
        from ..environment.weather_fetcher import create_environment_from_coordinates

        env, _ = create_environment_from_coordinates(
            latitude=latitude,
            longitude=longitude,
            datetime_obj=datetime_obj,
            elevation=elevation,
            use_weather_data=use_weather_data,
            use_nrlmsise=use_nrlmsise,
        )
        return env
