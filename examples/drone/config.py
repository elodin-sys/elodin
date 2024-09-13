import os
import typing as ty
from dataclasses import dataclass
from enum import Enum

import elodin as el
import numpy as np
import util
from numpy._typing import NDArray


class Frame(Enum):
    # QUAD X:
    #  (CW) 3 1 (CCW)
    #        X
    # (CCW) 2 4 (CW)
    QUAD_X = 0

    @property
    def motor_matrix(self) -> NDArray[np.float64]:
        motor_angles = None
        yaw_factor = None
        throttle_factor = None
        if self == Frame.QUAD_X:
            motor_angles = np.pi * np.array([0.25, -0.75, 0.75, -0.25])
            yaw_factor = np.array([-1.0, -1.0, 1.0, 1.0])
            throttle_factor = np.ones(4)
        else:
            raise ValueError(f"Unsupported frame: {self}")
        pitch_factor = np.sin(motor_angles)
        roll_factor = np.sin(motor_angles - np.pi / 2)

        # scale factors to [-0.5, 0.5] for each axis
        roll_factor /= 2 * np.max(np.abs(roll_factor))
        pitch_factor /= 2 * np.max(np.abs(pitch_factor))
        yaw_factor /= 2 * np.max(np.abs(yaw_factor))

        return np.array([roll_factor, pitch_factor, yaw_factor, throttle_factor])


@dataclass
class Control:
    rate_pid_gains: NDArray[np.float64]
    angle_p_gains: NDArray[np.float64]
    motor_thrust_exponent: float
    motor_thrust_hover: float
    attitude_control_input_tc: float
    pilot_yaw_rate_tc: float


@dataclass
class Config:
    _GLOBAL: ty.ClassVar[ty.Self | None] = None

    # Cascade PID controller gains
    control: Control

    # Drone GLB asset
    drone_glb: str

    # Mass in kg
    mass: float
    # Moment of inertia diagonal in kg*m^2
    inertia_diagonal: NDArray[np.float64]
    # Initial linear position in meters
    start_pos: NDArray[np.float64]
    # Initial attitude in euler angles in degrees
    start_euler_angles: NDArray[np.float64]
    # Motor positions
    motor_positions: NDArray[np.float64]
    # Motor thrust directions
    motor_thrust_directions: NDArray[np.float64]
    # Motor thrust curve
    motor_thrust_curve_path: str

    # Frame type
    frame: Frame

    # Simulation time step in seconds
    time_step: float
    # Fast loop time step in seconds
    fast_loop_time_step: float
    # Total simulation time in seconds
    simulation_time: float

    # Enable sensor noise
    sensor_noise: bool

    @property
    def dt(self) -> float:
        return self.time_step

    @property
    def total_sim_ticks(self) -> int:
        return int(self.simulation_time / self.dt)

    @property
    def attitude(self) -> el.Quaternion:
        return util.euler_to_quat(self.start_euler_angles)

    @property
    def spatial_transform(self) -> el.SpatialTransform:
        return el.SpatialTransform(
            linear=self.start_pos,
            angular=self.attitude,
        )

    @property
    def spatial_inertia(self) -> el.SpatialInertia:
        return el.SpatialInertia(
            mass=self.mass,
            inertia=self.inertia_diagonal,
        )

    @property
    def motor_torque_axes(self) -> NDArray[np.float64]:
        return np.cross(self.motor_positions, self.motor_thrust_directions)

    def thrust_curve(self) -> np.ndarray:
        path = os.path.join(os.path.dirname(__file__), self.motor_thrust_curve_path)
        return np.genfromtxt(path, delimiter=",", skip_header=1).transpose()

    @classmethod
    @property
    def GLOBAL(cls) -> ty.Self:
        if cls._GLOBAL is None:
            raise ValueError("No global config set")
        return cls._GLOBAL

    def set_as_global(self):
        Config._GLOBAL = self
