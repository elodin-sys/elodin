# Uncomment the next line to use the ENU ball simulation.
from sim import SIM_TIME_STEP, system, world
# Uncomment the next line to use the NED ball simulation.
# from sim_ned import SIM_TIME_STEP, system, world

world().run(system(), simulation_rate=1.0 / SIM_TIME_STEP, max_ticks=1200)
