#!/usr/bin/env python3

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Platform:
    mass: float
    side_length: float
    cable_length: float
    oscillation_time_period: float

    def moi(self, additional_mass: float, oscillation_time_period: float) -> float:
        g = 9.80665
        m = self.mass + additional_mass
        r = self.side_length * math.sqrt(3) / 3
        tau = oscillation_time_period
        s = self.cable_length
        return g * m * r**2 * tau**2 / (4 * math.pi**2 * s)

    def moi_platform(self) -> float:
        return self.moi(0.0, self.oscillation_time_period)

    def moi_object(self, mass: float, oscillation_time_period: float) -> float:
        moi_total = self.moi(mass, oscillation_time_period)
        return moi_total - self.moi_platform()


platform = Platform(
    mass=2.662,
    side_length=1.1,
    cable_length=0.93,
    oscillation_time_period=np.mean([15.26, 15.40]).item() / 9.5,
)
moi_platform = platform.moi_platform()

square_mass = 2.176
square_length = 2 * 0.3048  # 2 feet in meters
square_oscillation_time_period = np.mean([12.68, 12.75]).item() / 9.5
moi_square = platform.moi_object(square_mass, square_oscillation_time_period)

moi_square_theoretical = square_mass * square_length**2 / 6.0

talon_mass = 2.586 - 0.546
talon_roll_oscillation_time_period = np.mean([12.21, 12.19, 12.13]).item() / 9.5
talon_yaw_oscillation_time_period = np.mean([12.80, 12.67, 12.67]).item() / 9.5
talon_pitch_oscillation_time_period = np.mean([12.46, 12.36, 12.35]).item() / 9.5
moi_talon_roll = platform.moi_object(talon_mass, talon_roll_oscillation_time_period)
moi_talon_yaw = platform.moi_object(talon_mass, talon_yaw_oscillation_time_period)
moi_talon_pitch = platform.moi_object(talon_mass, talon_pitch_oscillation_time_period)

print(f"Platform MOI: {moi_platform:.4f}")
print(f"Square MOI: {moi_square:.4f}")
print(f"Square MOI (theoretical): {moi_square_theoretical:.4f}")
print(
    f"Talon MOI (r,p,y): {moi_talon_roll:.4f}, {moi_talon_pitch:.4f}, {moi_talon_yaw:.4f}"
)
