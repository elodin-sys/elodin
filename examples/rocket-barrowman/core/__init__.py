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
from .data import search_motors, get_motor_data

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
    # Data
    "search_motors",
    "get_motor_data",
]

