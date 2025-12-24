"""OpenRocket-compatible component definitions with extensions."""

from ..components.openrocket_components import (
    # Base classes
    RocketComponent,
    Material,
    MATERIALS,
    Position,
    FinishType,
    # Core components
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    EllipticalFinSet,
    MassComponent,
    Parachute,
    Streamer,
    Transition,
    InnerTube,
    CenteringRing,
    Rocket,
    # Airfoil system
    AirfoilType,
    AirfoilProfile,
    # Advanced fins
    FinShape,
    AdvancedFinSet,
    # External protuberances
    RailButton,
    LaunchLug,
    CameraShroud,
    # Recovery components
    Bulkhead,
    UBolt,
    EyeBolt,
    ShockCord,
    # Structural components
    Coupler,
    MotorRetainer,
    ThrustPlate,
    # Avionics
    AvionicsBay,
    SwitchBand,
)
from ..components.openrocket_aero import RocketAerodynamics
from ..components.openrocket_motor import Motor as OpenRocketMotor

__all__ = [
    # Base classes
    "RocketComponent",
    "Material",
    "MATERIALS",
    "Position",
    "FinishType",
    # Core components
    "NoseCone",
    "BodyTube",
    "TrapezoidFinSet",
    "EllipticalFinSet",
    "MassComponent",
    "Parachute",
    "Streamer",
    "Transition",
    "InnerTube",
    "CenteringRing",
    "Rocket",
    # Airfoil system
    "AirfoilType",
    "AirfoilProfile",
    # Advanced fins
    "FinShape",
    "AdvancedFinSet",
    # External protuberances
    "RailButton",
    "LaunchLug",
    "CameraShroud",
    # Recovery components
    "Bulkhead",
    "UBolt",
    "EyeBolt",
    "ShockCord",
    # Structural components
    "Coupler",
    "MotorRetainer",
    "ThrustPlate",
    # Avionics
    "AvionicsBay",
    "SwitchBand",
    # Aero
    "RocketAerodynamics",
    "OpenRocketMotor",
]

