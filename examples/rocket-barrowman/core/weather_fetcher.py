"""
Weather data fetcher for automatic environment initialization.

This module provides functionality to automatically fetch weather data from
national databases (ECMWF ERA5, NOAA GFS) based on coordinates and datetime,
then initialize the Environment with real atmospheric and wind conditions.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .environment import Environment

# Import availability flags
from .atmospheric_models import NRLMSISE_AVAILABLE

# Try to import optional dependencies
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False
    cdsapi = None

try:
    import netCDF4
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    netCDF4 = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


def fetch_era5_data(
    latitude: float,
    longitude: float,
    datetime_obj: datetime,
    output_file: Optional[str] = None,
    variables: Optional[list[str]] = None,
) -> Optional[str]:
    """
    Fetch ERA5 reanalysis data from ECMWF for given coordinates and datetime.

    Args:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        datetime_obj: Datetime object for the requested time
        output_file: Optional path to save NetCDF file. If None, creates temp file.
        variables: List of variables to fetch. Default: ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']

    Returns:
        Path to downloaded NetCDF file, or None if failed

    Requires:
        - CDS API key configured (see https://cds.climate.copernicus.eu/)
        - cdsapi package: pip install cdsapi
    """
    if not CDSAPI_AVAILABLE:
        raise ImportError(
            "cdsapi package required for ERA5 data. Install with: pip install cdsapi"
        )

    if not NETCDF_AVAILABLE:
        raise ImportError(
            "netCDF4 package required. Install with: pip install netCDF4"
        )

    # Check for CDS API credentials
    if not os.getenv("CDS_API_URL") or not os.getenv("CDS_API_KEY"):
        raise ValueError(
            "CDS API credentials not found. Set CDS_API_URL and CDS_API_KEY environment variables.\n"
            "Get credentials at: https://cds.climate.copernicus.eu/"
        )

    if variables is None:
        variables = [
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ]

    if output_file is None:
        # Create temp file
        cache_dir = Path(__file__).parent / "weather_cache"
        cache_dir.mkdir(exist_ok=True)
        timestamp = datetime_obj.strftime("%Y%m%d_%H%M")
        output_file = str(cache_dir / f"era5_{latitude:.2f}_{longitude:.2f}_{timestamp}.nc")

    client = cdsapi.Client()

    # ERA5 request parameters
    request_params = {
        "product_type": "reanalysis",
        "variable": variables,
        "pressure_level": [
            "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125", "150",
            "175", "200", "225", "250", "300", "350", "400", "450", "500", "550", "600",
            "650", "700", "750", "775", "800", "825", "850", "875", "900", "925", "950",
            "975", "1000",
        ],
        "year": datetime_obj.strftime("%Y"),
        "month": datetime_obj.strftime("%m"),
        "day": datetime_obj.strftime("%d"),
        "time": datetime_obj.strftime("%H:00"),
        "area": [
            latitude + 1,  # North
            longitude - 1,  # West
            latitude - 1,  # South
            longitude + 1,  # East
        ],
        "format": "netcdf",
    }

    try:
        print(f"Fetching ERA5 data for {latitude:.2f}°N, {longitude:.2f}°E at {datetime_obj}")
        client.retrieve("reanalysis-era5-pressure-levels", request_params, output_file)
        print(f"✓ Downloaded ERA5 data to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error fetching ERA5 data: {e}")
        return None


def fetch_noaa_gfs_data(
    latitude: float,
    longitude: float,
    datetime_obj: datetime,
    output_file: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch NOAA GFS forecast data for given coordinates and datetime.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        datetime_obj: Datetime object for the requested time
        output_file: Optional path to save NetCDF file

    Returns:
        Path to downloaded NetCDF file, or None if failed

    Note:
        NOAA GFS data is available for recent dates (typically last 10 days).
        For historical data, use ERA5 instead.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests package required. Install with: pip install requests")

    if not NETCDF_AVAILABLE:
        raise ImportError("netCDF4 package required. Install with: pip install netCDF4")

    # NOAA GFS data URL (example - actual implementation would need proper API)
    # This is a simplified version - real implementation would use proper NOAA API
    print("Warning: NOAA GFS direct API access requires proper implementation")
    print("For now, use ERA5 data or provide NetCDF files manually")
    return None


def create_environment_from_coordinates(
    latitude: float,
    longitude: float,
    datetime_obj: datetime,
    elevation: float = 0.0,
    use_weather_data: bool = True,
    use_nrlmsise: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Environment, Optional[str]]:
    """
    Automatically create Environment with weather data from coordinates and datetime.

    This function:
    1. Fetches weather data (ERA5) for the given location and time
    2. Sets up atmospheric model with the weather data
    3. Extracts wind profile from the weather data
    4. Creates a complete Environment object

    Args:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        datetime_obj: Datetime object for the requested time
        elevation: Ground elevation above sea level (m). Default: 0.0
        use_weather_data: If True, fetch and use weather data. Default: True
        use_nrlmsise: If True, use NRLMSISE-00 for high altitudes. Default: True
        cache_dir: Directory to cache weather data files. Default: core/weather_cache

    Returns:
        Tuple of (Environment object, path to weather data file or None)

    Example:
        >>> from datetime import datetime
        >>> from core.weather_fetcher import create_environment_from_coordinates
        >>>
        >>> # Spaceport America: 33.0°N, 106.5°W, elevation 1400m
        >>> launch_time = datetime(2024, 6, 15, 12, 0)  # June 15, 2024, noon
        >>> env, weather_file = create_environment_from_coordinates(
        ...     latitude=33.0,
        ...     longitude=-106.5,
        ...     datetime_obj=launch_time,
        ...     elevation=1400.0,
        ... )
        >>> print(f"Environment created with weather data from {weather_file}")
    """
    from .environment import Environment
    from .atmospheric_models import (
        WeatherDataAtmosphere,
        NRLMSISE00Atmosphere,
        HybridAtmosphere,
        ISAAtmosphere,
    )
    from .dynamic_wind import DynamicWindModel, ProfilePoint

    weather_file = None

    # Try to fetch weather data
    if use_weather_data and CDSAPI_AVAILABLE:
        try:
            weather_file = fetch_era5_data(latitude, longitude, datetime_obj)
        except Exception as e:
            print(f"Warning: Could not fetch weather data: {e}")
            print("Falling back to ISA model")
            use_weather_data = False

    # Set up atmospheric model
    if use_weather_data and weather_file:
        # Use weather data for low altitudes
        if use_nrlmsise:
            # Hybrid: Weather data + NRLMSISE-00
            low_model = WeatherDataAtmosphere(data_file=weather_file)
            high_model = None
            try:
                high_model = NRLMSISE00Atmosphere(
                    latitude=latitude,
                    longitude=longitude,
                    year=datetime_obj.year,
                    day_of_year=datetime_obj.timetuple().tm_yday,
                )
            except ImportError:
                pass  # NRLMSISE not available, use weather data only

            if high_model:
                atmosphere = HybridAtmosphere(
                    low_altitude_model=low_model,
                    high_altitude_model=high_model,
                    transition_altitude=86000.0,
                )
            else:
                atmosphere = low_model
        else:
            # Just weather data
            atmosphere = WeatherDataAtmosphere(data_file=weather_file)
    elif use_nrlmsise and NRLMSISE_AVAILABLE:
        # NRLMSISE-00 only
        try:
            atmosphere = NRLMSISE00Atmosphere(
                latitude=latitude,
                longitude=longitude,
                year=datetime_obj.year,
                day_of_year=datetime_obj.timetuple().tm_yday,
            )
        except ImportError:
            # Fallback to ISA if NRLMSISE fails
            atmosphere = ISAAtmosphere()
    else:
        # Fallback to ISA
        atmosphere = ISAAtmosphere()

    # Extract wind profile from weather data if available
    wind_model = None
    if weather_file and NETCDF_AVAILABLE:
        try:
            wind_profile = extract_wind_profile_from_netcdf(weather_file, latitude, longitude)
            if wind_profile:
                wind_model = DynamicWindModel(profile_points=wind_profile)
        except Exception as e:
            print(f"Warning: Could not extract wind profile: {e}")
            print("Using default wind model")

    # Create default wind model if none created
    if wind_model is None:
        wind_model = DynamicWindModel()

    # Create environment
    env = Environment(
        atmosphere=atmosphere,
        wind_model=wind_model,
        elevation=elevation,
    )

    return env, weather_file


def extract_wind_profile_from_netcdf(
    netcdf_file: str,
    latitude: float,
    longitude: float,
) -> Optional[list[ProfilePoint]]:
    """
    Extract wind profile from NetCDF weather data file.

    Args:
        netcdf_file: Path to NetCDF file
        latitude: Target latitude
        longitude: Target longitude

    Returns:
        List of ProfilePoint objects, or None if extraction failed
    """
    if not NETCDF_AVAILABLE:
        return None

    try:
        from .dynamic_wind import ProfilePoint
        import math

        with netCDF4.Dataset(netcdf_file, "r") as nc:
            # Extract wind components (u, v) and altitude/pressure levels
            # This is a simplified extraction - real implementation would
            # interpolate to exact lat/lon and handle different data formats

            if "level" in nc.variables:
                levels = nc.variables["level"][:]
            elif "pressure" in nc.variables:
                levels = nc.variables["pressure"][:]
            else:
                return None

            # Get wind components
            u_wind = None
            v_wind = None
            if "u" in nc.variables:
                u_wind = nc.variables["u"][:]
            elif "u_component_of_wind" in nc.variables:
                u_wind = nc.variables["u_component_of_wind"][:]

            if "v" in nc.variables:
                v_wind = nc.variables["v"][:]
            elif "v_component_of_wind" in nc.variables:
                v_wind = nc.variables["v_component_of_wind"][:]

            if u_wind is None or v_wind is None:
                return None

            # Convert pressure levels to altitudes (simplified)
            profile_points = []
            for i, level in enumerate(levels):
                # Approximate altitude from pressure (hPa)
                p_hpa = float(level)
                if p_hpa > 0:
                    h = 44330 * (1 - (p_hpa / 1013.25) ** 0.1903)  # meters

                    # Get wind components (simplified - would need proper interpolation)
                    u = float(u_wind[0, i, 0, 0]) if len(u_wind.shape) >= 2 else 0.0
                    v = float(v_wind[0, i, 0, 0]) if len(v_wind.shape) >= 2 else 0.0

                    # Calculate speed and direction
                    speed = math.sqrt(u**2 + v**2)
                    direction_rad = math.atan2(v, u)  # radians

                    profile_points.append(
                        ProfilePoint(
                            altitude=h,
                            speed=speed,
                            direction_rad=direction_rad,
                            vertical=0.0,
                        )
                    )

            return sorted(profile_points, key=lambda p: p.altitude)

    except Exception as e:
        print(f"Error extracting wind profile: {e}")
        return None

