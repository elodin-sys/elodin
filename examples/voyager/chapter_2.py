"""Run Voyager Chapter 2: heliocentric relative dynamics."""

import os
from pathlib import Path
import runpy


os.environ["VOYAGER_DYNAMICS_CHAPTER"] = "2"
runpy.run_path(Path(__file__).with_name("main.py"), run_name="__main__")
