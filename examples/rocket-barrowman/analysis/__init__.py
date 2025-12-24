"""Flight analysis, structural analysis, and aerospace-grade metrics."""

from .flight_analysis import FlightAnalyzer, FlightMetrics, StabilityDerivatives, compute_flight_phases
from .structural_analysis import (
    StructuralAnalyzer,
    StructuralAnalysisResult,
    FinFlutterAnalyzer,
    FlutterResult,
    LoadsResult,
    StressResult,
    MaterialProperties,
    STRUCTURAL_MATERIALS,
)

__all__ = [
    # Flight analysis
    "FlightAnalyzer",
    "FlightMetrics",
    "StabilityDerivatives",
    "compute_flight_phases",
    # Structural analysis
    "StructuralAnalyzer",
    "StructuralAnalysisResult",
    "FinFlutterAnalyzer",
    "FlutterResult",
    "LoadsResult",
    "StressResult",
    "MaterialProperties",
    "STRUCTURAL_MATERIALS",
]

