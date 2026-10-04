"""Propulsion, actuator, mass-property, and tank physics (WHITEPAPER 9-10).

Pure JAX-compatible functions, unit-tested standalone in test_propulsion.py
and wired into Elodin systems by sim.py. Body frame is Elodin's: +X nose,
engines at station x = 0 thrusting along +X.
"""

import jax
import jax.numpy as jnp
from constants import (
    ENGINE_A_E_M2,
    ENGINE_ISP_VAC_S,
    ENGINE_T_VAC_N,
    G0,
    LOX_LOAD_KG,
    OF_RATIO,
    RP1_LOAD_KG,
    S_REF_M2,
    STAGE1_DRY_MASS_KG,
    STAGE1_LENGTH_M,
    TANK_P_NOM_PA,
)

jax.config.update("jax_enable_x64", True)

# --- Stage geometry stations (m from engine plane, EST) ------------------------
DRY_CG_STATION_M = 18.8
RHO_LOX = 1220.0  # EST densified
RHO_RP1 = 830.0  # EST chilled
TANK_AREA_M2 = S_REF_M2
RP1_TANK_BOTTOM_M = 3.0
LOX_TANK_BOTTOM_M = 17.5
TANK_ULLAGE_FRAC = 0.05
V_TANK_LOX_M3 = LOX_LOAD_KG / RHO_LOX * (1.0 + TANK_ULLAGE_FRAC)
V_TANK_RP1_M3 = RP1_LOAD_KG / RHO_RP1 * (1.0 + TANK_ULLAGE_FRAC)
STAGE_RADIUS_M = 1.83

# --- Tank pressurization (Level-1 ullage model, EST) ----------------------------
P_REGULATOR_PA = TANK_P_NOM_PA + 0.2e5  # helium regulator setpoint
K_INFILL_PER_S = 0.5  # infill valve authority (fraction of deficit per second)
K_VENT_PER_S = 0.3  # vent valve authority
P_AMBIENT_MIN_PA = 1.0e4  # vent back-pressure floor


def engine_thrust_per_engine(throttle, p_ambient_pa):
    """T(u, h) = u * T_vac - p_a * A_e, floored at zero (WHITEPAPER 9.1)."""
    return jnp.maximum(throttle * ENGINE_T_VAC_N - p_ambient_pa * ENGINE_A_E_M2, 0.0)


def cluster_mdot(engines_lit, throttle):
    """Total propellant mass flow: mdot = T_vac / (Isp_vac * g0) per engine."""
    return engines_lit * throttle * ENGINE_T_VAC_N / (ENGINE_ISP_VAC_S * G0)


def split_mdot(mdot_total):
    """LOX/RP-1 split by mixture ratio."""
    mdot_lox = mdot_total * OF_RATIO / (1.0 + OF_RATIO)
    return mdot_lox, mdot_total - mdot_lox


def actuator_step(x, cmd, dt, tau, rate_limit=None, lo=None, hi=None):
    """Rate-limited first-order actuator, exact exponential discretization
    (WHITEPAPER 10.1). Valid at any dt relative to tau."""
    alpha = 1.0 - jnp.exp(-dt / tau)
    dx = alpha * (cmd - x)
    if rate_limit is not None:
        dx = jnp.clip(dx, -rate_limit * dt, rate_limit * dt)
    x_new = x + dx
    if lo is not None or hi is not None:
        x_new = jnp.clip(x_new, lo, hi)
    return x_new


def _column(mass, rho, bottom_m):
    """Propellant column (fills from the tank bottom, drains top-down):
    (cg station, own transverse inertia, own axial inertia)."""
    length = mass / (rho * TANK_AREA_M2)
    cg = bottom_m + 0.5 * length
    r2 = STAGE_RADIUS_M**2
    i_trans = mass * (length**2 / 12.0 + r2 / 4.0)
    i_axial = 0.5 * mass * r2
    return cg, i_trans, i_axial


# Stage 2 + payload ride as a point-ish cylinder above the interstage.
STAGE2_CG_STATION_M = 58.0  # EST
STAGE2_LENGTH_M = 16.0  # EST


def stack_mass_props(m_lox, m_rp1, m_upper=0.0):
    """Cylinder-stack mass model (WHITEPAPER 9.4).

    `m_upper` is the attached stage-2 + payload mass (zero after separation).
    Returns (total mass, CG station from engine plane, body inertia diagonal
    (Ix axial, Iy, Iz transverse) about the CG).
    """
    r2 = STAGE_RADIUS_M**2
    dry_i_trans = STAGE1_DRY_MASS_KG * STAGE1_LENGTH_M**2 / 12.0
    dry_i_axial = 0.5 * STAGE1_DRY_MASS_KG * r2

    cg_lox, it_lox, ia_lox = _column(m_lox, RHO_LOX, LOX_TANK_BOTTOM_M)
    cg_rp1, it_rp1, ia_rp1 = _column(m_rp1, RHO_RP1, RP1_TANK_BOTTOM_M)
    up_i_trans = m_upper * STAGE2_LENGTH_M**2 / 12.0
    up_i_axial = 0.5 * m_upper * r2

    mass = STAGE1_DRY_MASS_KG + m_lox + m_rp1 + m_upper
    cg = (
        STAGE1_DRY_MASS_KG * DRY_CG_STATION_M
        + m_lox * cg_lox
        + m_rp1 * cg_rp1
        + m_upper * STAGE2_CG_STATION_M
    ) / mass

    def par_axis(i_own, m, station):
        return i_own + m * (station - cg) ** 2

    i_trans = (
        par_axis(dry_i_trans, STAGE1_DRY_MASS_KG, DRY_CG_STATION_M)
        + par_axis(it_lox, m_lox, cg_lox)
        + par_axis(it_rp1, m_rp1, cg_rp1)
        + par_axis(up_i_trans, m_upper, STAGE2_CG_STATION_M)
    )
    i_axial = dry_i_axial + ia_lox + ia_rp1 + up_i_axial
    return mass, cg, jnp.array([i_axial, i_trans, i_trans])


def tank_pressure_step(p, m_prop, mdot_out, v_tank, rho, infill, vent, dt):
    """Isothermal ullage pressure update (WHITEPAPER 9.5, Level 1).

    Draining propellant grows the ullage (pressure falls as p * dV/V);
    the helium infill valve feeds toward the regulator setpoint; the vent
    valve bleeds toward ambient. `infill`/`vent` are valve states in [0, 1].
    """
    v_ullage = jnp.maximum(v_tank - m_prop / rho, 1e-2 * v_tank)
    dv_ullage = mdot_out / rho * dt
    p_drain = p * v_ullage / (v_ullage + dv_ullage)
    dp_infill = K_INFILL_PER_S * (P_REGULATOR_PA - p_drain) * infill * dt
    dp_vent = K_VENT_PER_S * (p_drain - P_AMBIENT_MIN_PA) * vent * dt
    return jnp.maximum(p_drain + jnp.maximum(dp_infill, 0.0) - jnp.maximum(dp_vent, 0.0), 0.0)


def inlet_pressure(p_tank, m_prop, rho, bottom_m, cg_m, a_axial_mps2, mdot):
    """Engine-inlet pressure: tank + acceleration head - line loss
    (WHITEPAPER 9.5). The column head uses the axial specific force."""
    column_len = m_prop / (rho * TANK_AREA_M2)
    head_height = bottom_m + column_len  # column top to the engine plane
    p_head = rho * jnp.maximum(a_axial_mps2, 0.0) * head_height
    k_line = 2.0e-2  # EST Pa per (kg/s)^2
    return p_tank + p_head - k_line * mdot**2
