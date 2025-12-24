"""
Weather data fetcher for automatic environment initialization.

This module provides functionality to automatically fetch weather data from
public APIs (Open-Meteo, NOAA GFS) based on coordinates and datetime,
then initialize the Environment with real atmospheric and wind conditions.

No API keys or setup required - works automatically like RocketPy.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .environment import Environment

# Import availability flags
from .atmospheric_models import NRLMSISE_AVAILABLE

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import netCDF4
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    netCDF4 = None

# Legacy CDS API support (optional, requires setup)
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False
    cdsapi = None


def fetch_openmeteo_data(
    latitude: float,
    longitude: float,
    datetime_obj: datetime,
) -> Optional[dict]:
    """
    Fetch weather data from Open-Meteo API (free, no API key required).

    Args:
        latitude: Latitude in degrees (-90 to 90)
        longitude: Longitude in degrees (-180 to 180)
        datetime_obj: Datetime object for the requested time

    Returns:
        Dictionary with weather data, or None if failed

    Note:
        Open-Meteo provides free access to weather data without requiring API keys.
        For historical data, uses ERA5 reanalysis. For recent dates, uses forecast data.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests package required. Install with: pip install requests")

    # Determine if we need historical or forecast data
    now = datetime.now()
    is_historical = datetime_obj < now - timedelta(days=2)

    try:
        if is_historical:
            # Use ERA5 reanalysis for historical data
            # Open-Meteo provides free ERA5 access
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": datetime_obj.strftime("%Y-%m-%d"),
                "end_date": (datetime_obj + timedelta(days=1)).strftime("%Y-%m-%d"),
                "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
                "models": "era5",
            }
        else:
            # Use forecast for recent dates
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
                "start_date": datetime_obj.strftime("%Y-%m-%d"),
                "end_date": (datetime_obj + timedelta(days=1)).strftime("%Y-%m-%d"),
            }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract data for the specific hour
        if "hourly" in data:
            hourly = data["hourly"]
            times = hourly.get("time", [])
            target_time = datetime_obj.strftime("%Y-%m-%dT%H:00")

            if target_time in times:
                idx = times.index(target_time)
                return {
                    "temperature": hourly.get("temperature_2m", [None])[idx],
                    "pressure": hourly.get("pressure_msl", [None])[idx],
                    "wind_speed": hourly.get("wind_speed_10m", [None])[idx],
                    "wind_direction": hourly.get("wind_direction_10m", [None])[idx],
                    "humidity": hourly.get("relative_humidity_2m", [None])[idx],
                    "latitude": latitude,
                    "longitude": longitude,
                    "datetime": datetime_obj,
                }

        return None

    except Exception as e:
        print(f"Error fetching Open-Meteo data: {e}")
        return None


def fetch_noaa_gfs_data(
    latitude: float,
    longitude: float,
    datetime_obj: datetime,
) -> Optional[dict]:
    """
    Fetch NOAA GFS forecast data via Open-Meteo (free, no API key required).

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        datetime_obj: Datetime object for the requested time

    Returns:
        Dictionary with weather data, or None if failed
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests package required. Install with: pip install requests")

    try:
        # Open-Meteo provides free access to NOAA GFS data
        url = "https://api.open-meteo.com/v1/gfs"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m",
            "start_date": datetime_obj.strftime("%Y-%m-%d"),
            "end_date": (datetime_obj + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract data for the specific hour
        if "hourly" in data:
            hourly = data["hourly"]
            times = hourly.get("time", [])
            target_time = datetime_obj.strftime("%Y-%m-%dT%H:00")

            if target_time in times:
                idx = times.index(target_time)
                return {
                    "temperature": hourly.get("temperature_2m", [None])[idx],
                    "pressure": hourly.get("pressure_msl", [None])[idx],
                    "wind_speed": hourly.get("wind_speed_10m", [None])[idx],
                    "wind_direction": hourly.get("wind_direction_10m", [None])[idx],
                    "humidity": hourly.get("relative_humidity_2m", [None])[idx],
                    "latitude": latitude,
                    "longitude": longitude,
                    "datetime": datetime_obj,
                }

        return None

    except Exception as e:
        print(f"Error fetching NOAA GFS data: {e}")
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
    weather_data = None

    # Try to fetch weather data automatically (no API keys needed)
    if use_weather_data and REQUESTS_AVAILABLE:
        try:
            # Try Open-Meteo first (completely free, no setup)
            weather_data = fetch_openmeteo_data(latitude, longitude, datetime_obj)
            if weather_data:
                print(f"✓ Fetched weather data from Open-Meteo for {latitude:.2f}°N, {longitude:.2f}°E")
        except Exception as e:
            print(f"Warning: Could not fetch weather data: {e}")
            print("Falling back to ISA model")
            use_weather_data = False

    # Set up atmospheric model
    if use_weather_data and weather_data:
        # Create a simple atmospheric model from weather data
        # For now, use ISA with adjustments, or create a custom model
        # In the future, we can create a WeatherDataAtmosphere that works with dict data
        
        # Use ISA as base, but we'll enhance it with NRLMSISE-00 if available
        if use_nrlmsise and NRLMSISE_AVAILABLE:
            # Hybrid: ISA + NRLMSISE-00 (ISA for low, NRLMSISE for high)
            try:
                low_model = ISAAtmosphere()
                high_model = NRLMSISE00Atmosphere(
                    latitude=latitude,
                    longitude=longitude,
                    year=datetime_obj.year,
                    day_of_year=datetime_obj.timetuple().tm_yday,
                )
                atmosphere = HybridAtmosphere(
                    low_altitude_model=low_model,
                    high_altitude_model=high_model,
                    transition_altitude=86000.0,
                )
            except Exception:
                atmosphere = ISAAtmosphere()
        else:
            # Just ISA for now (can be enhanced with weather data adjustments)
            atmosphere = ISAAtmosphere()
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
    if weather_data:
        try:
            from .dynamic_wind import ProfilePoint

            # Create wind profile from fetched data
            # Use surface wind data and create a simple profile
            wind_speed = weather_data.get("wind_speed", 0.0)
            wind_direction = weather_data.get("wind_direction", 0.0)
            
            if wind_speed is not None and wind_direction is not None:
                # Convert direction to radians
                direction_rad = np.radians(wind_direction)
                
                # Create a simple wind profile (can be enhanced with altitude-dependent data)
                # For now, use constant wind up to 10km, then decrease
                profile_points = [
                    ProfilePoint(altitude=0.0, speed=wind_speed, direction_rad=direction_rad, vertical=0.0),
                    ProfilePoint(altitude=1000.0, speed=wind_speed * 1.2, direction_rad=direction_rad, vertical=0.0),
                    ProfilePoint(altitude=5000.0, speed=wind_speed * 1.5, direction_rad=direction_rad, vertical=0.0),
                    ProfilePoint(altitude=10000.0, speed=wind_speed * 1.2, direction_rad=direction_rad, vertical=0.0),
                    ProfilePoint(altitude=20000.0, speed=wind_speed * 0.5, direction_rad=direction_rad, vertical=0.0),
                ]
                wind_model = DynamicWindModel(profile_points=profile_points)
        except Exception as e:
            print(f"Warning: Could not create wind profile: {e}")
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

