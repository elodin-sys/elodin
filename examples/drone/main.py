#!/usr/bin/env uv run

import sys

import polars as pl
import vehicles
from config import Config
from sim import system, world

vehicles.TALON_QUAD_CONFIG.set_as_global()
# vehicles.EDU_450_CONFIG.set_as_global()

args = sys.argv[1:]
world, drone = world()
if "--telemetry" in args:
    exec = world.build(
        system(only_rate_control=True), simulation_rate=Config.GLOBAL.simulation_rate
    )
    exec.run(Config.GLOBAL.total_sim_ticks)
    df = exec.history(["drone.body_ang_vel", "drone.motor_ang_vel"])
    print(df)
    df = df.select(
        pl.col("drone.body_ang_vel").arr.get(0).alias("body_ang_vel_x"),
        pl.col("drone.body_ang_vel").arr.get(1).alias("body_ang_vel_y"),
        pl.col("drone.body_ang_vel").arr.get(2).alias("body_ang_vel_z"),
        pl.col("drone.motor_ang_vel").arr.get(0).alias("motor_ang_vel_1"),
        pl.col("drone.motor_ang_vel").arr.get(1).alias("motor_ang_vel_2"),
        pl.col("drone.motor_ang_vel").arr.get(2).alias("motor_ang_vel_3"),
        pl.col("drone.motor_ang_vel").arr.get(3).alias("motor_ang_vel_4"),
    )
    df.write_csv("telemetry.csv")
else:
    world.run(
        system(),
        simulation_rate=Config.GLOBAL.simulation_rate,
        telemetry_rate=Config.GLOBAL.simulation_rate / 3.0,
        generate_real_time=True,
        max_ticks=int(Config.GLOBAL.total_sim_ticks or 1200),
        interactive=False,
    )
