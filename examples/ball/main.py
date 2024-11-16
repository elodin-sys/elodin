from sim import SIM_TIME_STEP, system, world

world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=1200)
