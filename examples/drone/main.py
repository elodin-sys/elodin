import jax.numpy as jnp
import math

from sim import world, system
from config import Config
import util

EDU_450_CONFIG = Config(
    drone_glb="https://storage.googleapis.com/elodin-assets/edu-450-v2-drone.glb",
    mass=1.0,
    inertia_diagonal=jnp.array([0.1, 0.1, 0.2]),
    start_pos=jnp.array([0.0, 0.0, 2.0]),
    start_euler_angles=jnp.array([0.0, 0.0, 0.0]),
    motor_positions=util.motor_positions(
        math.pi * jnp.array([0.75, 0.25, -0.25, -0.75]), 0.24
    ),
    motor_spin_dir=jnp.array([1, -1, 1, -1]),
    time_step=(1.0 / 300.0),
    fast_loop_time_step=(1.0 / 900.0),
    simulation_time=30.0,
    sensor_noise=True,
)

EDU_450_CONFIG.set_as_global()

world().run(
    system(),
    run_time_step=0.0,
    sim_time_step=Config.GLOBAL.dt,
    output_time_step=Config.GLOBAL.dt,
    max_ticks=int(Config.GLOBAL.simulation_time / Config.GLOBAL.dt),
)
