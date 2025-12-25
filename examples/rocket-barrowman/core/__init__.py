"""Core physics and rocket modeling components.

Reorganized structure:
- physics/ - Flight solver and numerical methods
- models/ - Rocket and motor models
- components/ - OpenRocket-compatible components
- environment/ - Atmospheric and wind models
- builders/ - Rocket configuration builders
- data/ - Motor database and data sources
"""

# Re-export everything for backward compatibility
from .physics import FlightSolver, FlightResult, StateSnapshot, Matrix, Vector
from .models import Rocket, Motor
from .environment import (
    Environment,
    ISAAtmosphere,
    NRLMSISE00Atmosphere,
    WeatherDataAtmosphere,
    HybridAtmosphere,
    AtmosphericProperties,
    NRLMSISE_AVAILABLE,
    NETCDF_AVAILABLE,
    DynamicWindModel,
    ProfilePoint,
    create_environment_from_coordinates,
)
from .builders import build_calisto, build_calisto_rocket, build_cesaroni_m1670
# Data module - import directly from motor_scraper if needed
# from .data import motor_scraper

__all__ = [
    # Physics
    "FlightSolver",
    "FlightResult",
    "StateSnapshot",
    "Matrix",
    "Vector",
    # Models
    "Rocket",
    "Motor",
    # Environment
    "Environment",
    "ISAAtmosphere",
    "NRLMSISE00Atmosphere",
    "WeatherDataAtmosphere",
    "HybridAtmosphere",
    "AtmosphericProperties",
    "NRLMSISE_AVAILABLE",
    "NETCDF_AVAILABLE",
    "DynamicWindModel",
    "ProfilePoint",
    "create_environment_from_coordinates",
    # Builders
    "build_calisto",
    "build_calisto_rocket",
    "build_cesaroni_m1670",
    # Data - import from core.data.motor_scraper directly
]
