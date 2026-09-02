"""Shared pytest setup for the Betaflight SITL example."""

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))
