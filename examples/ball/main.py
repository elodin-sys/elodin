import os
from sim import SIM_TIME_STEP, system, world

db_path = os.environ.get("BALL_DB_PATH")
kwargs = {}
if db_path:
    kwargs["db_path"] = db_path
world().run(system(), simulation_rate=1.0 / SIM_TIME_STEP, max_ticks=1200, **kwargs)
