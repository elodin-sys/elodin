import sys
from pathlib import Path

_RC_JET_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _RC_JET_DIR.parents[1]
sys.path.insert(0, str(_RC_JET_DIR))
sys.path.insert(0, str(_REPO_ROOT))
