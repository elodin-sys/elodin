"""RocketPy-inspired environment module for the Elodin solver.

This module packages the ISA atmosphere and dynamic wind model so they can be
consumed by the forthcoming RocketPy-style solver. The API mirrors the naming
of RocketPy's :mod:`rocketpy.environment` without depending on any external
packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from openrocket_atmosphere import ISAAtmosphere
from dynamic_wind import DynamicWindModel


@dataclass
class Environment:
    """Container for atmosphere and wind models."""

    atmosphere: ISAAtmosphere
    wind_model: DynamicWindModel
    elevation: float  # Ground elevation above sea level (m)

    def __init__(
        self,
        atmosphere: ISAAtmosphere | None = None,
        wind_model: DynamicWindModel | None = None,
        elevation: float = 0.0,
    ) -> None:
        self.atmosphere = atmosphere or ISAAtmosphere()
        self.wind_model = wind_model or DynamicWindModel()
        self.elevation = elevation

    # ------------------------------------------------------------------
    # Atmosphere helpers (RocketPy naming)
    # ------------------------------------------------------------------

    def set_atmospheric_model(self, *, type: str = "ISA", **kwargs) -> None:  # noqa: A003
        if type.upper() != "ISA":
            raise NotImplementedError(f"Atmospheric model '{type}' not supported")
        self.atmosphere = ISAAtmosphere(**kwargs)

    def air_properties(self, altitude: float) -> dict[str, float]:
        """Get air properties at altitude above ground level.
        
        Args:
            altitude: Altitude above ground level (m)
            
        Returns:
            Dictionary with density, pressure, temperature, viscosity, speed_of_sound
        """
        # Convert AGL to ASL (above sea level) for ISA calculation
        altitude_asl = altitude + self.elevation
        return self.atmosphere.get_properties(altitude_asl)

    # ------------------------------------------------------------------
    # Wind helpers (RocketPy naming)
    # ------------------------------------------------------------------

    def set_wind_model(self, *, profile_points: Iterable, **kwargs) -> None:
        self.wind_model = DynamicWindModel(profile_points=profile_points, **kwargs)

    def wind_velocity(self, altitude: float, time: float) -> Tuple[float, float, float]:
        return self.wind_model.get_wind(altitude, time)
