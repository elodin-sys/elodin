import sys
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1]
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))
