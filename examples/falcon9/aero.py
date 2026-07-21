"""Aerodynamics: config-blended tables, plume dominance, grid fins
(WHITEPAPER 8). Pure JAX functions; sim.py wires them into systems.

Body frame: +X nose. Ascent flies +X first; descent flies engines (-X) first.
All coefficient tables are EST calibration parameters with public-estimate
priors; the Monte Carlo campaign owns their scale factors.
"""

import jax
import jax.numpy as jnp
from constants import S_REF_M2, STAGE1_DIAMETER_M, STAGE1_LENGTH_M

jax.config.update("jax_enable_x64", True)

# --- Mach breakpoints shared by all tables --------------------------------------
MACH_PTS = jnp.array([0.0, 0.6, 0.9, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0])

# Axial-force coefficient, nose-first (slender body, transonic rise). EST.
CA_ASCENT = jnp.array([0.30, 0.32, 0.45, 0.55, 0.50, 0.42, 0.35, 0.30, 0.28])
# Axial-force coefficient, engines-first (blunt base + deployed grid fins +
# body lift at attitude — an effective drag prior; the recorded CRS-12
# descent q-bar peak of ~60 kPa is the calibration anchor). EST.
CA_DESCENT = jnp.array([1.90, 1.95, 2.10, 2.40, 2.30, 2.20, 2.10, 2.00, 1.90])
# Cross-flow (normal) coefficient: continuous through 90 deg AoA (the flip). EST.
CN_CROSS = jnp.array([1.20, 1.20, 1.25, 1.35, 1.30, 1.25, 1.20, 1.15, 1.10])

# Center-of-pressure stations (m from engine plane). Nose-first flow puts the
# bare-body CP ahead of the CG (unstable, TVC holds it); engines-first with
# fins deployed puts it downstream of the CG (stable). EST.
X_CP_ASCENT_M = 28.0
X_CP_DESCENT_M = 26.0

# Pitch/yaw damping derivatives Cmq (∂Cm/∂(q L/2V)), negative = stable.
# Engines-first + fins is much more heavily damped than the slender ascent
# stack. EST priors; WHITEPAPER §8.1.
#
# Dynamic derivatives are nondimensionalized on body length (not diameter):
# with L=D the same Cmq numbers under-damp the short-period mode by ~L_body²/D²
# and re-enable the fin/aero limit cycle at max-q̄.
CMQ_ASCENT = -2.5
CMQ_DESCENT = -12.0
L_REF_DAMP_M = STAGE1_LENGTH_M

# Plume-dominance blend kappa(C_T) = C_T / (C_T + C_T0) (WHITEPAPER 8.3). EST.
PLUME_CT0 = 1.0

# --- Grid fins (WHITEPAPER 8.4) --------------------------------------------------
FIN_STATION_M = 44.0
S_FIN_M2 = 1.5  # EST per fin
# Fin normal-force effectiveness per radian of deflection with transonic dip. EST.
CN_DELTA_FIN = jnp.array([1.2, 1.2, 0.9, 0.8, 1.1, 1.3, 1.25, 1.2, 1.1])
# X-configuration fin azimuths about body +X (rad), measured from +Y toward +Z.
FIN_AZIMUTH = jnp.deg2rad(jnp.array([45.0, 135.0, 225.0, 315.0]))
# Tangential force direction of a positive deflection for fin i.
FIN_FORCE_DIR = jnp.stack([jnp.zeros(4), -jnp.sin(FIN_AZIMUTH), jnp.cos(FIN_AZIMUTH)], axis=1)
FIN_POS = jnp.stack(
    [
        jnp.full((4,), FIN_STATION_M),
        1.83 * jnp.cos(FIN_AZIMUTH),
        1.83 * jnp.sin(FIN_AZIMUTH),
    ],
    axis=1,
)
# Mixing: fin deflections from (pitch, yaw, roll) commands, X-config.
# pitch: +Z force -> fins with dir_z sign; yaw: +Y force; roll: all same sign.
FIN_MIX = jnp.stack(
    [
        FIN_FORCE_DIR[:, 2],  # pitch column: project force onto +Z
        FIN_FORCE_DIR[:, 1],  # yaw column: project force onto +Y
        jnp.ones(4),  # roll column: common tangential deflection
    ],
    axis=1,
)


def config_blend(v_axial_body: jnp.ndarray) -> jnp.ndarray:
    """0 = engines-first (descent tables), 1 = nose-first (ascent tables).

    Smooth in the body-axial air-relative velocity so the flip crosses the
    tables continuously.
    """
    return 0.5 * (1.0 + jnp.tanh(v_axial_body / 50.0))


def plume_dominance(thrust_n: jnp.ndarray, qbar_pa: jnp.ndarray) -> jnp.ndarray:
    """kappa in [0, 1): fraction of aerodynamic force erased by the plume."""
    ct = thrust_n / jnp.maximum(qbar_pa * S_REF_M2, 1.0)
    return ct / (ct + PLUME_CT0)


def body_aero_wrench(
    v_air_body,
    mach,
    qbar_pa,
    cg_station_m,
    omega_body=None,
    ca_scale=1.0,
    cn_scale=1.0,
):
    """Continuous all-attitude body aero force + moment (body frame).

    F = -qbar S [C_ax (v_hat . x) x_hat + C_n (v_hat - (v_hat . x) x_hat)]
    applied at the config-blended CP station, plus pitch/yaw damping
    M = q̄ S L²/(2V) Cmq ω_⊥ (WHITEPAPER §8.1).
    """
    speed = jnp.linalg.norm(v_air_body)
    v_hat = v_air_body / jnp.maximum(speed, 1e-6)
    blend = config_blend(v_air_body[0])
    ca = (
        blend * jnp.interp(mach, MACH_PTS, CA_ASCENT)
        + (1.0 - blend) * jnp.interp(mach, MACH_PTS, CA_DESCENT)
    ) * ca_scale
    cn = jnp.interp(mach, MACH_PTS, CN_CROSS) * cn_scale
    x_hat = jnp.array([1.0, 0.0, 0.0])
    axial = v_hat[0]
    cross = v_hat - axial * x_hat
    force = -qbar_pa * S_REF_M2 * (ca * axial * x_hat + cn * cross)
    x_cp = blend * X_CP_ASCENT_M + (1.0 - blend) * X_CP_DESCENT_M
    lever = (x_cp - cg_station_m) * x_hat
    torque = jnp.cross(lever, force)
    # Rotational damping on pitch/yaw (roll neglected — slender axisymmetric).
    if omega_body is None:
        omega_body = jnp.zeros(3)
    cmq = blend * CMQ_ASCENT + (1.0 - blend) * CMQ_DESCENT
    damp = (
        qbar_pa
        * S_REF_M2
        * (L_REF_DAMP_M**2)
        / (2.0 * jnp.maximum(speed, 1.0))
        * cmq
    )
    omega_perp = jnp.array([0.0, omega_body[1], omega_body[2]])
    torque = torque + damp * omega_perp
    return force, torque


def fin_wrench(deltas_rad, mach, qbar_pa, cg_station_m, eff_scale=1.0):
    """Grid-fin control wrench from four deflections (body frame)."""
    cnd = jnp.interp(mach, MACH_PTS, CN_DELTA_FIN) * eff_scale
    per_fin = qbar_pa * S_FIN_M2 * cnd * deltas_rad  # (4,)
    forces = per_fin[:, None] * FIN_FORCE_DIR  # (4, 3)
    cg = jnp.array([cg_station_m, 0.0, 0.0])
    torques = jnp.cross(FIN_POS - cg, forces)
    return jnp.sum(forces, axis=0), jnp.sum(torques, axis=0)


def fin_mix(pitch_yaw_roll: jnp.ndarray) -> jnp.ndarray:
    """Map (pitch, yaw, roll) commands (rad) to four fin deflections (rad)."""
    return FIN_MIX @ pitch_yaw_roll


def reference_length() -> float:
    return STAGE1_LENGTH_M
