"""Physics and numerical methods.

Note: Flight solver has been moved to solvers/barrowman/ for better organization.
"""

# Re-export from solvers for backward compatibility
import sys
from pathlib import Path

# Add parent directory to path for absolute imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from solvers.barrowman import FlightSolver, FlightResult, StateSnapshot, Matrix, Vector

__all__ = ["FlightSolver", "FlightResult", "StateSnapshot", "Matrix", "Vector"]

