"""Puts examples/drone first on sys.path: the drone-policy modules build on its plant, sensors,
controllers and config. Import it before any examples/drone module (the leading underscore
keeps it first in a sorted import block)."""

import os
import sys

DRONE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drone"))
if DRONE not in sys.path:
    sys.path.insert(0, DRONE)
