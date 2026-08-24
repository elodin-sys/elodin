"""ISA troposphere model, keyed to geodetic altitude.

Same formulation the pre-campaign example used (and the whitepaper §6.1);
kept as the one atmosphere for aero, propulsion, and trim so dynamic
pressure, Mach, and the propulsion-map interpolation stay consistent.
"""

from __future__ import annotations

import jax.numpy as jnp

SEA_LEVEL_TEMPERATURE_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
SEA_LEVEL_PRESSURE_PA = 101325.0
GAS_CONSTANT_AIR = 287.05
GAMMA_AIR = 1.4
TROPOPAUSE_TEMPERATURE_K = 216.65


def temperature(altitude_m):
    t = SEA_LEVEL_TEMPERATURE_K - LAPSE_RATE_K_PER_M * altitude_m
    return jnp.clip(t, TROPOPAUSE_TEMPERATURE_K, None)


def pressure(altitude_m):
    t = temperature(altitude_m)
    return SEA_LEVEL_PRESSURE_PA * (t / SEA_LEVEL_TEMPERATURE_K) ** 5.2561


def density(altitude_m):
    return pressure(altitude_m) / (GAS_CONSTANT_AIR * temperature(altitude_m))


def speed_of_sound(altitude_m):
    return jnp.sqrt(GAMMA_AIR * GAS_CONSTANT_AIR * temperature(altitude_m))
