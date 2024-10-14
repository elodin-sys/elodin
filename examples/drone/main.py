import elodin as el
import jax.numpy as jnp
import numpy as np
import util
from config import Config, Control, Frame
from sim import system, world

EDU_450_CONFIG = Config(
    control=Control(
        rate_pid_gains=np.array(
            [
                [0.40, 0.40, 2.50],  # P
                [0.02, 0.02, 0.02],  # I
                [0.05, 0.05, 0.01],  # D
            ]
        ),
        angle_p_gains=np.array([4.0, 4.0, 1.0]),
        motor_thrust_exponent=0.65,
        motor_thrust_hover=0.35,
        attitude_control_input_tc=0.1,
        pilot_yaw_rate_tc=0.1,
    ),
    drone_glb="https://storage.googleapis.com/elodin-assets/edu-450-v2-drone.glb",
    mass=1.0,
    inertia_diagonal=np.array([0.1, 0.1, 0.2]),
    start_pos=np.array([0.0, 0.0, 2.0]),
    start_euler_angles=np.array([0.0, 0.0, 0.0]),
    motor_positions=util.motor_positions(np.pi * np.array([0.25, -0.75, 0.75, -0.25]), 0.24),
    motor_thrust_directions=np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    motor_thrust_curve_path="./motor_thrust_curve.csv",
    sim_time_step=(1.0 / 300.0),
    frame=Frame.QUAD_X,
    fast_loop_time_step=(1.0 / 900.0),
    simulation_time=30.0,
    sensor_noise=True,
)

# Motor configuration: Quad-X
# Motor 1: Front right motor, counter-clockwise
# Motor 2: Front left motor, clockwise
# Motor 3: Rear right motor, clockwise
# Motor 4: Rear left motor, counter-clockwise
#  (CW) 3 1 (CCW)
#        X
# (CCW) 2 4 (CW)

q_tilt_right = el.Quaternion.from_axis_angle(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=np.deg2rad(5.0),
)
q_tilt_left = el.Quaternion.from_axis_angle(
    axis=np.array([1.0, 0.0, 0.0]),
    angle=np.deg2rad(-5.0),
)
q_tilt_back = el.Quaternion.from_axis_angle(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=np.deg2rad(-3.75),
)
up = np.array([0.0, 0.0, 1.0])
motor_thrust_directions = jnp.array(
    [
        q_tilt_right @ up,
        (q_tilt_left * q_tilt_back) @ up,
        q_tilt_left @ up,
        (q_tilt_right * q_tilt_back) @ up,
    ]
)

# Assume CG is at the center of the quad motor positions
TALON_QUAD_CONFIG = Config(
    control=Control(
        rate_pid_gains=np.array(
            [
                [0.32, 0.32, 1.10],  # P
                [0.05, 0.05, 0.08],  # I
                [0.12, 0.08, 0.01],  # D
            ]
        ),
        angle_p_gains=np.array([4.0, 4.0, 1.0]),
        motor_thrust_exponent=0.833,
        motor_thrust_hover=0.689,
        attitude_control_input_tc=0.2,  # soft
        pilot_yaw_rate_tc=0.25,  # soft
    ),
    drone_glb="https://storage.googleapis.com/elodin-assets/talon-quad-v2.glb",
    mass=2.586,
    inertia_diagonal=np.array([0.0854, 0.1149, 0.1604]),
    start_pos=np.array([0.0, 0.0, 2.0]),
    start_euler_angles=np.array([0.0, 0.0, 0.0]),
    motor_positions=np.array(
        [
            [0.26, -0.26, 0.26, -0.26],  # X position for each motor
            [-0.2075, 0.2015, 0.2075, -0.2015],  # Y position for each motor
            [-0.0215, 0.0215, -0.0215, 0.0215],  # Z position for each motor
        ]
    ).T,
    motor_thrust_directions=np.array(motor_thrust_directions),
    motor_thrust_curve_path="./motor_thrust_curve.csv",
    sim_time_step=(1.0 / 300.0),
    frame=Frame.QUAD_X,
    fast_loop_time_step=(1.0 / 900.0),
    simulation_time=30.0,
    sensor_noise=False,
)
TALON_QUAD_CONFIG.set_as_global()
# EDU_450_CONFIG.set_as_global()

world().run(
    system(),
    sim_time_step=Config.GLOBAL.dt,
    run_time_step=0.0,
    output_time_step=Config.GLOBAL.dt,
    max_ticks=int(Config.GLOBAL.total_sim_ticks),
)
