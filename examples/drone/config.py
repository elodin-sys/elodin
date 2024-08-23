from dataclasses import dataclass
import jax
import jax.numpy as jnp
import typing as ty
import elodin as el

import util

# Motor indexes:
#
#  0   1
#   \ /
#    X
#   / \
#  3   2


@dataclass
class Config:
    _GLOBAL: ty.ClassVar[ty.Self | None] = None

    # Drone GLB asset
    drone_glb: str

    # Mass in kg
    mass: float
    # Moment of inertia diagonal in kg*m^2
    inertia_diagonal: jax.Array
    # Initial linear position in meters
    start_pos: jax.Array
    # Initial attitude in euler angles in degrees
    start_euler_angles: jax.Array
    # Motor positions
    motor_positions: jax.Array
    # Motor spin directions (1 for CW, -1 for CCW)
    motor_spin_dir: jax.Array

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
        return util.euler_to_quat(jnp.array(self.start_euler_angles))

    @property
    def spatial_transform(self) -> el.SpatialTransform:
        return el.SpatialTransform(
            linear=jnp.array(self.start_pos),
            angular=self.attitude,
        )

    @property
    def spatial_inertia(self) -> el.SpatialInertia:
        return el.SpatialInertia(
            mass=self.mass,
            inertia=jnp.array(self.inertia_diagonal),
        )

    @property
    def motor_torque_axes(self) -> jax.Array:
        return util.motor_torque_axes(self.motor_positions)

    @classmethod
    @property
    def GLOBAL(cls) -> ty.Self:
        if cls._GLOBAL is None:
            raise ValueError("No global config set")
        return cls._GLOBAL

    def set_as_global(self):
        Config._GLOBAL = self
