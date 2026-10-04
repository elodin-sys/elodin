"""U.S. Standard Atmosphere 1976 (WHITEPAPER 7.1) as pure JAX functions.

Piecewise-linear temperature in geopotential altitude, hydrostatic pressure
per layer, ideal-gas density. Valid to the 86 km table top; above it the last
isothermal layer decays exponentially (rho < 1e-5 kg/m^3, effectively vacuum
for this mission's brief 86-118 km apogee arc).

Anchors the tests assert: rho(0) = 1.2250 kg/m^3, p(H=11 km) = 22,632 Pa,
rho(H=11 km) = 0.3639 kg/m^3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

G0 = 9.80665
R_STAR = 8.31432  # J/(mol K)
M_AIR = 28.9644e-3  # kg/mol
R_AIR = R_STAR / M_AIR  # 287.053 J/(kg K)
GMR = G0 * M_AIR / R_STAR  # 0.0341632 K/m
GAMMA = 1.4
R0_GEOPOT_M = 6_356_766.0
P_SL_PA = 101_325.0

# Layer bases: geopotential altitude (m), base temperature (K), lapse rate (K/m).
_H_B = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0])
_T_B = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
_L_B = np.array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3, 0.0])


def _base_pressures() -> np.ndarray:
    p = np.empty_like(_H_B)
    p[0] = P_SL_PA
    for i in range(1, len(_H_B)):
        dh = _H_B[i] - _H_B[i - 1]
        t_b, lapse = _T_B[i - 1], _L_B[i - 1]
        if lapse == 0.0:
            p[i] = p[i - 1] * np.exp(-GMR * dh / t_b)
        else:
            p[i] = p[i - 1] * (t_b / (t_b + lapse * dh)) ** (GMR / lapse)
    return p


_P_B = _base_pressures()

H_B = jnp.asarray(_H_B)
T_B = jnp.asarray(_T_B)
L_B = jnp.asarray(_L_B)
P_B = jnp.asarray(_P_B)


def geopotential_altitude(h_geometric_m):
    return R0_GEOPOT_M * h_geometric_m / (R0_GEOPOT_M + h_geometric_m)


def pressure_temperature_at_geopotential(h_geopot_m):
    """(pressure Pa, temperature K) at geopotential altitude."""
    h = jnp.clip(h_geopot_m, 0.0, 250_000.0)
    i = jnp.clip(jnp.searchsorted(H_B, h, side="right") - 1, 0, len(_H_B) - 1)
    t_b, lapse, p_b, h_b = T_B[i], L_B[i], P_B[i], H_B[i]
    dh = h - h_b
    temp = t_b + lapse * dh
    # Guard the lapse==0 division; jnp.where picks the isothermal branch there.
    lapse_safe = jnp.where(lapse == 0.0, 1.0, lapse)
    p_gradient = p_b * (t_b / temp) ** (GMR / lapse_safe)
    p_isothermal = p_b * jnp.exp(-GMR * dh / t_b)
    press = jnp.where(lapse == 0.0, p_isothermal, p_gradient)
    return press, temp


def pressure_temperature(h_geometric_m):
    return pressure_temperature_at_geopotential(geopotential_altitude(h_geometric_m))


def pressure(h_geometric_m):
    return pressure_temperature(h_geometric_m)[0]


def density(h_geometric_m):
    p, t = pressure_temperature(h_geometric_m)
    return p / (R_AIR * t)


def speed_of_sound(h_geometric_m):
    _, t = pressure_temperature(h_geometric_m)
    return jnp.sqrt(GAMMA * R_AIR * t)
