"""Falcon 9 booster plant in rotating WGS84 ECEF (WHITEPAPER sections 4-10).

Phase 1 scope: the passive plant — point-mass gravity plus the rotating-frame
Coriolis/centrifugal terms, integrated by `el.six_dof`, with geodetic
telemetry. Effectors (engines, aero, TVC, RCS, fins) land in later phases.

NOTE: no `from __future__ import annotations` here — `@el.map` introspects
real annotation objects, not the stringized ForwardRefs that import creates.
"""

import math
import typing as ty
from pathlib import Path

import aero
import atmosphere
import jax
import jax.numpy as jnp
import propulsion
import rcs
import sensors as sn
from constants import (
    ENGINE_SHUTDOWN_TAU_S,
    ENGINE_SPINUP_TAU_S,
    ENGINE_THROTTLE_TAU_S,
    FIN_MAX_DEG,
    FIN_RATE_DPS,
    FIN_TAU_S,
    G0,
    LOX_LOAD_KG,
    N_ENGINES,
    PAD_ALT_M,
    PAD_LAT_DEG,
    PAD_LON_DEG,
    RCS_VALVE_TAU_S,
    RELIGHT_CAPABLE_ENGINES,
    RP1_LOAD_KG,
    SIM_RATE_HZ,
    TANK_P_NOM_PA,
    THROTTLE_MIN,
    TVC_MAX_DEG,
    TVC_RATE_DPS,
    TVC_TAU_S,
    VALVE_TAU_S,
)
from frames import (
    ecef_to_geodetic,
    ellipsoid_up,
    frame_accel,
    geodetic_to_ecef,
    gravity_accel,
    ned_basis,
)

import elodin as el

SIM_TIME_STEP = 1.0 / SIM_RATE_HZ

# Valve indices (WHITEPAPER 9.5).
VALVE_HE_INFILL_LOX = 0
VALVE_HE_VENT_LOX = 1
VALVE_HE_INFILL_RP1 = 2
VALVE_HE_VENT_RP1 = 3
VALVE_MAIN_LOX = 4
VALVE_MAIN_RP1 = 5
VALVE_TEATEB = 6
VALVE_N2_PURGE = 7
N_VALVES = 8

# TEA-TEB charges: all engines light once at liftoff; the center engine and
# its two relight-capable neighbors (indices 0-2) carry three more relights.
INITIAL_TEATEB_CHARGES = jnp.array(
    [4.0] * RELIGHT_CAPABLE_ENGINES + [1.0] * (N_ENGINES - RELIGHT_CAPABLE_ENGINES)
)

# --- Telemetry components -----------------------------------------------------
AltitudeGeodetic = ty.Annotated[
    jax.Array,
    el.Component("altitude_geodetic", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
GroundSpeed = ty.Annotated[
    jax.Array,
    el.Component("ground_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]

# --- Propulsion components (Phase 2) -------------------------------------------
# Commanded per-engine throttle in [0 | THROTTLE_MIN..1]; written by the FSW.
EngineCmd = ty.Annotated[
    jax.Array,
    el.Component(
        "engine_cmd",
        el.ComponentType(el.PrimitiveType.F64, (N_ENGINES,)),
        metadata={"external_control": "true"},
    ),
]
# Commanded valve states in [0, 1]; written by the FSW.
ValveCmd = ty.Annotated[
    jax.Array,
    el.Component(
        "valve_cmd",
        el.ComponentType(el.PrimitiveType.F64, (N_VALVES,)),
        metadata={"external_control": "true"},
    ),
]
EngineSpool = ty.Annotated[
    jax.Array,
    el.Component("engine_spool", el.ComponentType(el.PrimitiveType.F64, (N_ENGINES,))),
]
EngineArmed = ty.Annotated[
    jax.Array,
    el.Component("engine_armed", el.ComponentType(el.PrimitiveType.F64, (N_ENGINES,))),
]
TeaTebCharges = ty.Annotated[
    jax.Array,
    el.Component("teateb_charges", el.ComponentType(el.PrimitiveType.F64, (N_ENGINES,))),
]
ValveState = ty.Annotated[
    jax.Array,
    el.Component("valve_state", el.ComponentType(el.PrimitiveType.F64, (N_VALVES,))),
]
ThrustTotal = ty.Annotated[
    jax.Array,
    el.Component("thrust_total", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
MdotTotal = ty.Annotated[
    jax.Array,
    el.Component("mdot_total", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
PropellantLox = ty.Annotated[
    jax.Array,
    el.Component("propellant_lox", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
PropellantRp1 = ty.Annotated[
    jax.Array,
    el.Component("propellant_rp1", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TankPressureLox = ty.Annotated[
    jax.Array,
    el.Component("tank_pressure_lox", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TankPressureRp1 = ty.Annotated[
    jax.Array,
    el.Component("tank_pressure_rp1", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
InletPressureLox = ty.Annotated[
    jax.Array,
    el.Component("inlet_pressure_lox", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
InletPressureRp1 = ty.Annotated[
    jax.Array,
    el.Component("inlet_pressure_rp1", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
CgStation = ty.Annotated[
    jax.Array,
    el.Component("cg_station", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
AxialSpecificForce = ty.Annotated[
    jax.Array,
    el.Component("axial_specific_force", el.ComponentType(el.PrimitiveType.F64, (1,))),
]

# --- Aero / effector components (Phase 3) ---------------------------------------
# Wind profile hook (NED components rotated to ECEF at build; zero default).
WindEcef = ty.Annotated[
    jax.Array,
    el.Component("wind_ecef", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
Qbar = ty.Annotated[
    jax.Array,
    el.Component("qbar", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Mach = ty.Annotated[
    jax.Array,
    el.Component("mach", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# TVC gimbal (pitch, yaw) commanded by the in-sim attitude loop (rad).
TvcCmd = ty.Annotated[
    jax.Array,
    el.Component("tvc_cmd", el.ComponentType(el.PrimitiveType.F64, (2,))),
]
TvcState = ty.Annotated[
    jax.Array,
    el.Component("tvc_state", el.ComponentType(el.PrimitiveType.F64, (2,))),
]
# Grid-fin (pitch, yaw, roll) command from the FSW (rad).
FinCmd = ty.Annotated[
    jax.Array,
    el.Component(
        "fin_cmd",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"external_control": "true"},
    ),
]
FinState = ty.Annotated[
    jax.Array,
    el.Component("fin_state", el.ComponentType(el.PrimitiveType.F64, (4,))),
]
# Desired body torque for the RCS (from the in-sim attitude loop).
RcsTorqueCmd = ty.Annotated[
    jax.Array,
    el.Component("rcs_torque_cmd", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
RcsLevels = ty.Annotated[
    jax.Array,
    el.Component("rcs_levels", el.ComponentType(el.PrimitiveType.F64, (rcs.N_RCS,))),
]
NitrogenKg = ty.Annotated[
    jax.Array,
    el.Component("nitrogen_kg", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Attitude setpoint quaternion commanded by the FSW.
AttitudeSetpoint = ty.Annotated[
    el.Quaternion,
    el.Component("attitude_setpoint", metadata={"external_control": "true"}),
]
# FSW attitude-authority enable: [tvc, rcs].
CtrlEnable = ty.Annotated[
    jax.Array,
    el.Component(
        "ctrl_enable",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"external_control": "true"},
    ),
]
# FSW mission phase (telemetry/display only; the FSW owns the state machine).
FswPhase = ty.Annotated[
    jax.Array,
    el.Component(
        "fsw_phase",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
Landed = ty.Annotated[
    jax.Array,
    el.Component("landed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Attached stage-2 + payload mass; the bridge zeroes it at the FSW separation
# command (the plant models the mass event, WHITEPAPER 15).
UpperMass = ty.Annotated[
    jax.Array,
    el.Component(
        "upper_mass",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
# Launch clamp latch: 0 until cluster thrust first exceeds weight, then 1.
Lifted = ty.Annotated[
    jax.Array,
    el.Component("lifted", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Sim time of clamp release (s); the recorded profile's t=0 is liftoff, so
# scoring and the truth ghost index the reference by (t_sim - t_liftoff).
LiftoffTime = ty.Annotated[
    jax.Array,
    el.Component("liftoff_time", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Touchdown metrics latched at first ground contact (WHITEPAPER 13.2):
# [vertical speed, lateral speed, tilt-from-vertical deg].
TouchdownMetrics = ty.Annotated[
    jax.Array,
    el.Component("touchdown_metrics", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
# Wrench audits: [force_body(3), torque_body(3)] per effector (WHITEPAPER 5.3).
AeroWrench = ty.Annotated[
    jax.Array,
    el.Component("aero_wrench", el.ComponentType(el.PrimitiveType.F64, (6,))),
]
FinWrench = ty.Annotated[
    jax.Array,
    el.Component("fin_wrench", el.ComponentType(el.PrimitiveType.F64, (6,))),
]
RcsWrench = ty.Annotated[
    jax.Array,
    el.Component("rcs_wrench", el.ComponentType(el.PrimitiveType.F64, (6,))),
]
EngineWrench = ty.Annotated[
    jax.Array,
    el.Component("engine_wrench", el.ComponentType(el.PrimitiveType.F64, (6,))),
]

FIN_MAX_RAD = math.radians(FIN_MAX_DEG)
FIN_RATE_RADPS = math.radians(FIN_RATE_DPS)
TVC_MAX_RAD = math.radians(TVC_MAX_DEG)
TVC_RATE_RADPS = math.radians(TVC_RATE_DPS)
N2_ISP_S = 70.0  # EST cold-gas nitrogen


@el.map
def gravity_and_frame_forces(
    force: el.Force, inertia: el.Inertia, pos: el.WorldPos, vel: el.WorldVel
) -> el.Force:
    """Gravitation + Coriolis + centrifugal, as a world-frame force (WHITEPAPER 5.1)."""
    r = pos.linear()
    v = vel.linear()
    accel = gravity_accel(r) + frame_accel(r, v)
    return force + el.SpatialForce(linear=accel * inertia.mass())


# --- Propulsion systems (Phase 2) ----------------------------------------------


@el.map
def valve_dynamics(valve_state: ValveState, valve_cmd: ValveCmd) -> ValveState:
    """Solenoid open/close response (~15 ms first-order, exact discretization)."""
    return propulsion.actuator_step(
        valve_state, jnp.clip(valve_cmd, 0.0, 1.0), SIM_TIME_STEP, VALVE_TAU_S, lo=0.0, hi=1.0
    )


def make_engine_dynamics(thrust_scale: float = 1.0, isp_scale: float = 1.0) -> el.System:
    """Per-engine ignition/spool/shutdown state (WHITEPAPER 9.3).

    An engine lights only on a commanded rising edge with a TEA-TEB charge
    available, the igniter isolation valve open, and both main propellant
    valves open. Spool tracks commanded throttle with asymmetric first-order
    time constants; thrust dies with the feed (valves or empty tanks).
    `thrust_scale`/`isp_scale` are the campaign's vehicle calibration knobs.
    """

    @el.map
    def engine_dynamics(
        pos: el.WorldPos,
        cmd: EngineCmd,
        spool: EngineSpool,
        armed: EngineArmed,
        charges: TeaTebCharges,
        valves: ValveState,
        lox: PropellantLox,
        rp1: PropellantRp1,
    ) -> tuple[EngineSpool, EngineArmed, TeaTebCharges, ThrustTotal, MdotTotal]:
        cmd_c = jnp.clip(cmd, 0.0, 1.0)
        cmd_on = cmd_c >= THROTTLE_MIN * 0.5
        feed_open = (valves[VALVE_MAIN_LOX] > 0.5) & (valves[VALVE_MAIN_RP1] > 0.5)
        teateb_open = valves[VALVE_TEATEB] > 0.5
        prop_ok = (lox[0] > 0.0) & (rp1[0] > 0.0)

        lighting = cmd_on & (armed < 0.5) & (charges >= 1.0) & feed_open & teateb_open & prop_ok
        charges_next = charges - jnp.where(lighting, 1.0, 0.0)
        armed_next = jnp.where(cmd_on & ((armed > 0.5) | lighting), 1.0, 0.0)

        burn_ok = (armed_next > 0.5) & feed_open & prop_ok
        target = jnp.where(burn_ok, jnp.maximum(cmd_c, THROTTLE_MIN), 0.0)
        # Three regimes: slow turbopump spin-up from cold, fast throttle
        # response once running, fast shutdown (WHITEPAPER 9.3).
        running = spool > 0.5 * THROTTLE_MIN
        tau = jnp.where(
            target > spool,
            jnp.where(running, ENGINE_THROTTLE_TAU_S, ENGINE_SPINUP_TAU_S),
            ENGINE_SHUTDOWN_TAU_S,
        )
        spool_next = propulsion.actuator_step(spool, target, SIM_TIME_STEP, tau, lo=0.0, hi=1.0)

        _, _, alt = ecef_to_geodetic(pos.linear())
        p_amb = atmosphere.pressure(jnp.maximum(alt, 0.0))
        lit = spool_next > 1e-3
        thrust_per = jnp.where(
            lit, propulsion.engine_thrust_per_engine(spool_next, p_amb) * thrust_scale, 0.0
        )
        mdot = propulsion.cluster_mdot(jnp.where(lit, 1.0, 0.0), spool_next) * (
            thrust_scale / isp_scale
        )
        return (
            spool_next,
            armed_next,
            charges_next,
            jnp.array([jnp.sum(thrust_per)]),
            jnp.array([jnp.sum(mdot)]),
        )

    return engine_dynamics


@el.map
def mass_props(
    mdot: MdotTotal,
    lox: PropellantLox,
    rp1: PropellantRp1,
    thrust: ThrustTotal,
    upper: UpperMass,
) -> tuple[PropellantLox, PropellantRp1, el.Inertia, CgStation, AxialSpecificForce]:
    """Deplete propellant and rebuild mass/CG/inertia from the cylinder stack
    (including the attached stage 2 + payload until separation)."""
    mdot_lox, mdot_rp1 = propulsion.split_mdot(mdot[0])
    lox_next = jnp.maximum(lox[0] - mdot_lox * SIM_TIME_STEP, 0.0)
    rp1_next = jnp.maximum(rp1[0] - mdot_rp1 * SIM_TIME_STEP, 0.0)
    mass, cg, inertia_diag = propulsion.stack_mass_props(
        lox_next, rp1_next, jnp.maximum(upper[0], 0.0)
    )
    return (
        jnp.array([lox_next]),
        jnp.array([rp1_next]),
        el.SpatialInertia(mass, inertia_diag),
        jnp.array([cg]),
        jnp.array([thrust[0] / mass]),
    )


@el.map
def tank_dynamics(
    p_lox: TankPressureLox,
    p_rp1: TankPressureRp1,
    lox: PropellantLox,
    rp1: PropellantRp1,
    mdot: MdotTotal,
    valves: ValveState,
    axial: AxialSpecificForce,
    cg: CgStation,
) -> tuple[TankPressureLox, TankPressureRp1, InletPressureLox, InletPressureRp1]:
    """Ullage pressures + engine-inlet pressures with acceleration head."""
    mdot_lox, mdot_rp1 = propulsion.split_mdot(mdot[0])
    p_lox_next = propulsion.tank_pressure_step(
        p_lox[0],
        lox[0],
        mdot_lox,
        propulsion.V_TANK_LOX_M3,
        propulsion.RHO_LOX,
        valves[VALVE_HE_INFILL_LOX],
        valves[VALVE_HE_VENT_LOX],
        SIM_TIME_STEP,
    )
    p_rp1_next = propulsion.tank_pressure_step(
        p_rp1[0],
        rp1[0],
        mdot_rp1,
        propulsion.V_TANK_RP1_M3,
        propulsion.RHO_RP1,
        valves[VALVE_HE_INFILL_RP1],
        valves[VALVE_HE_VENT_RP1],
        SIM_TIME_STEP,
    )
    inlet_lox = propulsion.inlet_pressure(
        p_lox_next,
        lox[0],
        propulsion.RHO_LOX,
        propulsion.LOX_TANK_BOTTOM_M,
        cg[0],
        axial[0],
        mdot_lox,
    )
    inlet_rp1 = propulsion.inlet_pressure(
        p_rp1_next,
        rp1[0],
        propulsion.RHO_RP1,
        propulsion.RP1_TANK_BOTTOM_M,
        cg[0],
        axial[0],
        mdot_rp1,
    )
    return (
        jnp.array([p_lox_next]),
        jnp.array([p_rp1_next]),
        jnp.array([inlet_lox]),
        jnp.array([inlet_rp1]),
    )


# --- Effector wrenches (Phase 3) -------------------------------------------------


@el.map
def tvc_actuators(tvc_state: TvcState, tvc_cmd: TvcCmd) -> TvcState:
    return propulsion.actuator_step(
        tvc_state,
        jnp.clip(tvc_cmd, -TVC_MAX_RAD, TVC_MAX_RAD),
        SIM_TIME_STEP,
        TVC_TAU_S,
        rate_limit=TVC_RATE_RADPS,
        lo=-TVC_MAX_RAD,
        hi=TVC_MAX_RAD,
    )


@el.map
def fin_actuators(fin_state: FinState, fin_cmd: FinCmd) -> FinState:
    deltas_cmd = aero.fin_mix(jnp.clip(fin_cmd, -FIN_MAX_RAD, FIN_MAX_RAD))
    return propulsion.actuator_step(
        fin_state,
        jnp.clip(deltas_cmd, -FIN_MAX_RAD, FIN_MAX_RAD),
        SIM_TIME_STEP,
        FIN_TAU_S,
        rate_limit=FIN_RATE_RADPS,
        lo=-FIN_MAX_RAD,
        hi=FIN_MAX_RAD,
    )


@el.map
def engine_wrench(thrust: ThrustTotal, tvc: TvcState, cg: CgStation) -> EngineWrench:
    """Cluster thrust through the TVC gimbal, applied at the engine plane
    (WHITEPAPER 10.2): d_B ~ (1, delta_yaw, -delta_pitch) normalized."""
    d = jnp.array([1.0, tvc[1], -tvc[0]])
    d = d / jnp.linalg.norm(d)
    force = thrust[0] * d
    lever = jnp.array([-cg[0], 0.0, 0.0])  # engine plane is station 0
    torque = jnp.cross(lever, force)
    return jnp.concatenate([force, torque])


@el.map
def rcs_dynamics(
    levels: RcsLevels,
    torque_cmd: RcsTorqueCmd,
    cg: CgStation,
    n2: NitrogenKg,
) -> tuple[RcsLevels, RcsWrench, NitrogenKg]:
    """Allocate desired torque to thrusters, apply valve dynamics, meter N2."""
    have_gas = n2[0] > 0.0
    cmd_levels = jnp.where(have_gas, rcs.allocate_torque(torque_cmd, cg[0]), jnp.zeros(rcs.N_RCS))
    levels_next = propulsion.actuator_step(
        levels, cmd_levels, SIM_TIME_STEP, RCS_VALVE_TAU_S, lo=0.0, hi=1.0
    )
    force, torque = rcs.rcs_wrench(levels_next, cg[0])
    thrust_sum = jnp.sum(levels_next) * rcs.RCS_THRUST_PER_THRUSTER_N
    n2_next = jnp.maximum(n2[0] - thrust_sum / (N2_ISP_S * G0) * SIM_TIME_STEP, 0.0)
    return levels_next, jnp.concatenate([force, torque]), jnp.array([n2_next])


def make_aero_dynamics(ca_scale: float = 1.0, cn_scale: float = 1.0) -> el.System:
    """Air data + body aero + grid fins, with plume dominance (WHITEPAPER 7-8).
    `ca_scale`/`cn_scale` are the campaign's aero calibration knobs."""

    @el.map
    def aero_dynamics(
        pos: el.WorldPos,
        vel: el.WorldVel,
        wind: WindEcef,
        thrust: ThrustTotal,
        fins: FinState,
        cg: CgStation,
    ) -> tuple[Qbar, Mach, AeroWrench, FinWrench]:
        _, _, alt = ecef_to_geodetic(pos.linear())
        alt = jnp.maximum(alt, 0.0)
        rho = atmosphere.density(alt)
        a_sound = atmosphere.speed_of_sound(alt)
        v_air_ecef = vel.linear() - wind
        v_air_body = pos.angular().inverse() @ v_air_ecef
        speed = jnp.linalg.norm(v_air_body)
        qbar = 0.5 * rho * speed**2
        mach = speed / a_sound
        f_aero, t_aero = aero.body_aero_wrench(
            v_air_body, mach, qbar, cg[0], ca_scale=ca_scale, cn_scale=cn_scale
        )
        kappa = aero.plume_dominance(thrust[0], qbar)
        f_aero = f_aero * (1.0 - kappa)
        t_aero = t_aero * (1.0 - kappa)
        f_fin, t_fin = aero.fin_wrench(fins, mach, qbar, cg[0])
        return (
            jnp.array([qbar]),
            jnp.array([mach]),
            jnp.concatenate([f_aero, t_aero]),
            jnp.concatenate([f_fin, t_fin]),
        )

    return aero_dynamics


@el.map
def apply_body_wrenches(
    engine: EngineWrench,
    aero_w: AeroWrench,
    fin_w: FinWrench,
    rcs_w: RcsWrench,
    force: el.Force,
    pos: el.WorldPos,
) -> el.Force:
    """Sum all body-frame effector wrenches and rotate into the world frame."""
    total = engine + aero_w + fin_w + rcs_w
    q = pos.angular()
    return force + el.SpatialForce(linear=q @ total[:3], torque=q @ total[3:])


# --- In-sim attitude inner loop (Phase 5, WHITEPAPER 11.6) --------------------------
# Inertia-scaled quaternion-error PD: tau = I (wn^2 err - 2 zeta wn omega).
ATT_WN_TVC = 0.9  # rad/s, powered flight
ATT_ZETA_TVC = 0.9
ATT_WN_RCS = 0.35  # rad/s, coast/flip
ATT_ZETA_RCS = 0.8


@el.map
def attitude_control(
    pos: el.WorldPos,
    vel: el.WorldVel,
    setpoint: AttitudeSetpoint,
    enable: CtrlEnable,
    inertia: el.Inertia,
    thrust: ThrustTotal,
    cg: CgStation,
) -> tuple[TvcCmd, RcsTorqueCmd]:
    """Execute the FSW attitude setpoint at 1000 Hz: TVC takes pitch/yaw when
    engines burn, the RCS takes roll always and all axes unpowered."""
    q = pos.angular()
    q_err = q.inverse() * setpoint
    err = q_err.vector()
    sign = jnp.where(err[3] >= 0.0, 1.0, -1.0)
    err_vec = sign * err[:3]
    omega_body = q.inverse() @ vel.angular()
    i_diag = inertia.inertia_diag()

    tvc_on = (enable[0] > 0.5) & (thrust[0] > 2.0e5)
    rcs_on = enable[1] > 0.5

    wn = jnp.where(tvc_on, ATT_WN_TVC, ATT_WN_RCS)
    zeta = jnp.where(tvc_on, ATT_ZETA_TVC, ATT_ZETA_RCS)
    torque_des = i_diag * (wn**2 * err_vec - 2.0 * zeta * wn * omega_body)

    # TVC: tau_y = -cg T dp, tau_z = -cg T dy  (engine_wrench geometry).
    lever = jnp.maximum(cg[0] * thrust[0], 1.0)
    tvc_cmd = jnp.where(
        tvc_on,
        jnp.array([-torque_des[1] / lever, -torque_des[2] / lever]),
        jnp.zeros(2),
    )
    # RCS: roll always (single-axis engines have no roll arm); everything
    # when the engines are off. A deadband (~1 deg attitude, ~0.6 deg/s rate)
    # keeps the cold-gas valves shut near the setpoint — continuous chatter
    # would drain the nitrogen budget long before the landing burn.
    in_deadband = (jnp.linalg.norm(err_vec) < 0.009) & (jnp.linalg.norm(omega_body) < 0.01)
    rcs_torque = jnp.where(
        tvc_on,
        jnp.array([torque_des[0], 0.0, 0.0]),
        torque_des,
    )
    rcs_torque = jnp.where(rcs_on & ~in_deadband, rcs_torque, jnp.zeros(3))
    return tvc_cmd, rcs_torque


@el.map
def ground_contact(
    pos: el.WorldPos,
    vel: el.WorldVel,
    landed: Landed,
    metrics: TouchdownMetrics,
    lifted: Lifted,
) -> tuple[el.WorldPos, el.WorldVel, Landed, TouchdownMetrics]:
    """Touchdown: latch impact metrics on the contact tick (before state is
    clobbered — the 'too perfect' trap), then pin the booster to the surface
    (WHITEPAPER 13.2, 15). Only active after liftoff."""
    r = pos.linear()
    lat, lon, alt = ecef_to_geodetic(r)
    sin_lat, cos_lat = jnp.sin(lat), jnp.cos(lat)
    sin_lon, cos_lon = jnp.sin(lon), jnp.cos(lon)
    up = jnp.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    contact = jnp.logical_and(lifted[0] > 0.5, alt <= 0.0)
    was_landed = landed[0] > 0.5
    landed_now = jnp.logical_or(was_landed, contact)
    first_contact = jnp.logical_and(~was_landed, contact)

    v = vel.linear()
    v_up = jnp.dot(v, up)
    v_lat = jnp.linalg.norm(v - v_up * up)
    body_x = pos.angular() @ jnp.array([1.0, 0.0, 0.0])
    tilt_deg = jnp.rad2deg(jnp.arccos(jnp.clip(jnp.dot(body_x, up), -1.0, 1.0)))
    latched = jnp.where(first_contact, jnp.array([jnp.abs(v_up), v_lat, tilt_deg]), metrics)

    pinned_pos = jnp.where(landed_now, r - alt * up, r)
    pinned_lin = jnp.where(landed_now, jnp.zeros(3), v)
    pinned_ang = jnp.where(landed_now, jnp.zeros(3), vel.angular())
    return (
        el.SpatialTransform(angular=pos.angular(), linear=pinned_pos),
        el.SpatialMotion(angular=pinned_ang, linear=pinned_lin),
        jnp.array([jnp.where(landed_now, 1.0, 0.0)]),
        latched,
    )


@el.system
def pad_clamp(
    tick: el.Query[el.SimulationTick],
    q: el.Query[el.WorldPos, el.WorldVel, Lifted, LiftoffTime, ThrustTotal, el.Inertia],
) -> el.Query[el.WorldPos, el.WorldVel, Lifted, LiftoffTime]:
    """Hold-down clamps: pin the stack to the pad until thrust exceeds
    weight, then release for good (the real T-0 release condition), latching
    the release time for liftoff-referenced truth comparison."""
    t_s = tick[0] * SIM_TIME_STEP
    pad = pad_ecef()

    def update(pos, vel, lifted, t_liftoff, thrust, inertia):
        r = pos.linear()
        weight = inertia.mass() * 9.79
        was_lifted = lifted[0] > 0.5
        release = jnp.logical_or(was_lifted, thrust[0] > weight)
        first = jnp.logical_and(~was_lifted, release)
        held_pos = el.SpatialTransform(angular=pos.angular(), linear=jnp.where(release, r, pad))
        held_vel = el.SpatialMotion(
            angular=jnp.where(release, vel.angular(), jnp.zeros(3)),
            linear=jnp.where(release, vel.linear(), jnp.zeros(3)),
        )
        return (
            held_pos,
            held_vel,
            jnp.array([jnp.where(release, 1.0, 0.0)]),
            jnp.where(first, jnp.array([t_s]), t_liftoff),
        )

    return q.map((el.WorldPos, el.WorldVel, Lifted, LiftoffTime), update)


# --- Sensors (Phase 4, WHITEPAPER 12) ---------------------------------------------


@el.map
def imu_model(
    tick: sn.SensorTick,
    pos: el.WorldPos,
    vel: el.WorldVel,
    inertia: el.Inertia,
    engine: EngineWrench,
    aero_w: AeroWrench,
    fin_w: FinWrench,
    rcs_w: RcsWrench,
) -> tuple[sn.SensorTick, sn.ImuAccel, sn.ImuGyro]:
    """Specific force = summed non-gravitational body force / mass; gyro
    measures the inertial rate (frame rate + Earth rate) — WHITEPAPER 12.1."""
    count = tick[0] + 1.0
    f_body = (engine[:3] + aero_w[:3] + fin_w[:3] + rcs_w[:3]) / inertia.mass()
    q_inv = pos.angular().inverse()
    gyro = q_inv @ (vel.angular() + sn.OMEGA_E_VEC)
    accel_meas = f_body + sn._noise(count, 1, (3,), sn.IMU_ACCEL_SIGMA)
    gyro_meas = gyro + sn._noise(count, 2, (3,), sn.IMU_GYRO_SIGMA)
    return jnp.array([count]), accel_meas, gyro_meas


@el.map
def gps_model(
    tick: sn.SensorTick,
    timer: sn.GpsTimer,
    pos: el.WorldPos,
    vel: el.WorldVel,
    mach: Mach,
    thrust: ThrustTotal,
    gps_pos: sn.GpsPos,
    gps_vel: sn.GpsVel,
    count: sn.GpsCount,
) -> tuple[sn.GpsTimer, sn.GpsPos, sn.GpsVel, sn.GpsCount]:
    """25 Hz ECEF position/velocity, timer-accumulator + hold, with the
    retropropulsion plasma blackout (WHITEPAPER 12.2)."""
    t = timer[0] + SIM_TIME_STEP
    fired = t >= sn.GPS_DT_S
    t = jnp.where(fired, t - sn.GPS_DT_S, t)
    blackout = (mach[0] > sn.BLACKOUT_MACH_MIN) & (thrust[0] > sn.BLACKOUT_THRUST_MIN_N)
    fresh = fired & ~blackout
    n = count[0] + jnp.where(fresh, 1.0, 0.0)
    pos_meas = pos.linear() + sn._noise(n, 3, (3,), sn.GPS_POS_SIGMA)
    vel_meas = vel.linear() + sn._noise(n, 4, (3,), sn.GPS_VEL_SIGMA)
    return (
        jnp.array([t]),
        jnp.where(fresh, pos_meas, gps_pos),
        jnp.where(fresh, vel_meas, gps_vel),
        jnp.array([n]),
    )


@el.map
def radar_altimeter_model(
    timer: sn.RadarTimer,
    pos: el.WorldPos,
    rng_prev: sn.RadarRange,
    count: sn.RadarCount,
) -> tuple[sn.RadarTimer, sn.RadarRange, sn.RadarCount]:
    """Boresight (-X body) range to terrain at 40 Hz within FOV/max-range
    gates; -1 when invalid (WHITEPAPER 12.2)."""
    t = timer[0] + SIM_TIME_STEP
    fired = t >= sn.RADAR_DT_S
    t = jnp.where(fired, t - sn.RADAR_DT_S, t)
    lat, lon, alt = ecef_to_geodetic(pos.linear())
    sin_lat, cos_lat = jnp.sin(lat), jnp.cos(lat)
    sin_lon, cos_lon = jnp.sin(lon), jnp.cos(lon)
    up = jnp.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    bore_world = pos.angular() @ jnp.array([-1.0, 0.0, 0.0])
    cos_tilt = jnp.dot(bore_world, -up)
    slant = alt / jnp.maximum(cos_tilt, 1e-3)
    n = count[0] + jnp.where(fired, 1.0, 0.0)
    valid = (cos_tilt > sn.RADAR_FOV_COS) & (slant <= sn.RADAR_MAX_RANGE_M) & (alt > 0.0)
    meas = jnp.where(valid, slant + sn._noise(n, 5, (), sn.RADAR_SIGMA_M), -1.0)
    return (
        jnp.array([t]),
        jnp.where(fired, jnp.array([meas]), rng_prev),
        jnp.array([n]),
    )


@el.map
def pressure_transducers(
    tick: sn.SensorTick,
    p_lox: TankPressureLox,
    p_rp1: TankPressureRp1,
    i_lox: InletPressureLox,
    i_rp1: InletPressureRp1,
) -> sn.PressureMeas:
    truth = jnp.array([p_lox[0], p_rp1[0], i_lox[0], i_rp1[0]])
    return truth + sn._noise(tick[0], 6, (4,), sn.PRESSURE_SIGMA_PA)


@el.map
def display_model(pos: el.WorldPos, vel: el.WorldVel) -> tuple[sn.DisplaySpeed, sn.DisplayAlt]:
    """The webcast observables: ground-relative speed quantized to 1 km/h,
    geodetic altitude quantized to 0.1 km (WHITEPAPER 12.3)."""
    speed = jnp.linalg.norm(vel.linear())
    _, _, alt = ecef_to_geodetic(pos.linear())
    return (
        jnp.array([jnp.round(speed / sn.DISPLAY_SPEED_STEP) * sn.DISPLAY_SPEED_STEP]),
        jnp.array([jnp.round(alt / sn.DISPLAY_ALT_STEP) * sn.DISPLAY_ALT_STEP]),
    )


def sensor_systems() -> el.System:
    return imu_model | gps_model | radar_altimeter_model | pressure_transducers | display_model


@el.map
def derive_geodetic_telemetry(
    pos: el.WorldPos, vel: el.WorldVel
) -> tuple[AltitudeGeodetic, GroundSpeed]:
    """The two webcast observables: geodetic altitude and ground-relative speed."""
    _, _, alt = ecef_to_geodetic(pos.linear())
    return (
        jnp.array([alt]),
        jnp.array([jnp.linalg.norm(vel.linear())]),
    )


def pad_ecef() -> jnp.ndarray:
    return geodetic_to_ecef(math.radians(PAD_LAT_DEG), math.radians(PAD_LON_DEG), PAD_ALT_M)


# Pad-local basis for the truth-ghost track and initial attitude.
_PAD_LAT_RAD = math.radians(PAD_LAT_DEG)
_PAD_LON_RAD = math.radians(PAD_LON_DEG)
PAD_UP = ellipsoid_up(_PAD_LAT_RAD, _PAD_LON_RAD)
_PAD_NED = ned_basis(_PAD_LAT_RAD, _PAD_LON_RAD)
# Ascent ground-track azimuth ~45 deg (northeast, ISS inclination class). EST.
PAD_TRACK_DIR = (_PAD_NED[0] + _PAD_NED[1]) / jnp.linalg.norm(_PAD_NED[0] + _PAD_NED[1])


def upright_attitude() -> el.Quaternion:
    """Body +X along the pad's geodetic up (launch attitude)."""
    up = jnp.asarray(PAD_UP)
    x = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.cross(x, up)
    axis = axis / jnp.linalg.norm(axis)
    angle = jnp.arccos(jnp.clip(jnp.dot(x, up), -1.0, 1.0))
    return el.Quaternion.from_axis_angle(axis, angle)


# --- Truth ghost + scoring (Phase 4) ----------------------------------------------
TruthMarker = ty.Annotated[
    jax.Array,
    el.Component("truth_marker", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# [sum speed_err^2, sum alt_err^2, samples] accumulated in-sim (WHITEPAPER 13.2).
ScoreState = ty.Annotated[
    jax.Array,
    el.Component("score_state", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@el.dataclass
class StaticSceneObject(el.Archetype):
    world_pos: el.WorldPos


def truth_ghost_components(ref) -> list:
    return [
        StaticSceneObject(el.WorldPos(angular=upright_attitude(), linear=pad_ecef())),
        el.C(TruthMarker, jnp.array([1.0])),
        el.C(AltitudeGeodetic, jnp.array([0.0])),
        el.C(GroundSpeed, jnp.array([0.0])),
    ]


def make_truth_playback(ref, display_lag_s: float = 0.0) -> el.System:
    """Kinematic replay of the recorded flight on the ghost entity.

    `display_lag_s` models the webcast overlay latency (WHITEPAPER 12.3):
    the recorded series lags reality, so reality at liftoff-relative time t
    corresponds to data sample (t + lag).
    """
    lag = float(display_lag_s)
    ref_t = jnp.asarray(ref.time_s)
    ref_speed = jnp.asarray(ref.speed_mps)
    ref_alt = jnp.asarray(ref.altitude_m)
    ref_dr = jnp.asarray(ref.downrange_m)
    pad = pad_ecef()
    up = jnp.asarray(PAD_UP)
    track = jnp.asarray(PAD_TRACK_DIR)
    ghost_att = upright_attitude()

    @el.system
    def truth_playback(
        tick: el.Query[el.SimulationTick],
        lifted_q: el.Query[Lifted],
        liftoff_q: el.Query[LiftoffTime],
        truth: el.Query[TruthMarker],
    ) -> el.Query[el.WorldPos, AltitudeGeodetic, GroundSpeed]:
        # Reference t=0 is liftoff (clamp release), not sim start.
        t_s = tick[0] * SIM_TIME_STEP
        lifted = jnp.asarray(lifted_q[0]).reshape(())
        t_liftoff = jnp.asarray(liftoff_q[0]).reshape(())
        t_ref = jnp.where(lifted > 0.5, t_s - t_liftoff + lag, lag)
        alt = jnp.interp(t_ref, ref_t, ref_alt)
        speed = jnp.interp(t_ref, ref_t, ref_speed)
        downrange = jnp.interp(t_ref, ref_t, ref_dr)
        pose = el.SpatialTransform(angular=ghost_att, linear=pad + up * alt + track * downrange)
        return truth.map(
            (el.WorldPos, AltitudeGeodetic, GroundSpeed),
            lambda _m: (pose, jnp.array([alt]), jnp.array([speed])),
        )

    return truth_playback


def make_display_scoring(ref, display_lag_s: float = 0.0) -> el.System:
    """Accumulate display-space squared errors vs the recorded profile."""
    lag = float(display_lag_s)
    ref_t = jnp.asarray(ref.time_s)
    ref_speed = jnp.asarray(ref.speed_mps)
    ref_alt = jnp.asarray(ref.altitude_m)
    t_end = float(ref.t_end)

    @el.system
    def display_scoring(
        tick: el.Query[el.SimulationTick],
        q: el.Query[sn.DisplaySpeed, sn.DisplayAlt, Lifted, LiftoffTime, ScoreState],
    ) -> el.Query[ScoreState]:
        t_s = tick[0] * SIM_TIME_STEP

        def update(speed, alt, lifted, t_liftoff, score):
            t_ref = t_s - t_liftoff[0] + lag
            in_window = (lifted[0] > 0.5) & (t_ref >= lag) & (t_ref <= t_end)
            err_v = speed[0] - jnp.interp(t_ref, ref_t, ref_speed)
            err_h = alt[0] - jnp.interp(t_ref, ref_t, ref_alt)
            return jnp.where(
                in_window,
                score + jnp.array([err_v**2, err_h**2, 1.0]),
                score,
            )

        return q.map(ScoreState, update)

    return display_scoring


def build_passive(
    init_pos_ecef,
    init_vel_ecef,
    init_attitude: el.Quaternion | None = None,
    init_angular_vel=None,
    mass_kg: float = 30_000.0,
    inertia_diag=None,
    integrator: el.Integrator = el.Integrator.SemiImplicit,
) -> tuple[el.World, el.System]:
    """A single passive booster body — the verification-ladder test article."""
    world = el.World()
    if init_attitude is None:
        init_attitude = el.Quaternion.identity()
    if init_angular_vel is None:
        init_angular_vel = jnp.zeros(3)
    if inertia_diag is None:
        inertia_diag = jnp.array([1.0, 1.0, 1.0]) * mass_kg
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=init_attitude, linear=jnp.asarray(init_pos_ecef, dtype=jnp.float64)
                ),
                world_vel=el.SpatialMotion(
                    angular=jnp.asarray(init_angular_vel, dtype=jnp.float64),
                    linear=jnp.asarray(init_vel_ecef, dtype=jnp.float64),
                ),
                inertia=el.SpatialInertia(mass_kg, jnp.asarray(inertia_diag, dtype=jnp.float64)),
            ),
            el.C(AltitudeGeodetic, jnp.array([0.0])),
            el.C(GroundSpeed, jnp.array([0.0])),
        ],
        name="booster",
    )
    system = (
        el.six_dof(sys=gravity_and_frame_forces, integrator=integrator) | derive_geodetic_telemetry
    )
    return world, system


N2_INITIAL_KG = 800.0  # EST cold-gas budget (flip + coast + descent attitude)


def booster_propulsion_components(lox_kg: float = LOX_LOAD_KG, rp1_kg: float = RP1_LOAD_KG) -> list:
    """The propulsion/effector component set for the booster entity."""
    return [
        el.C(EngineCmd, jnp.zeros(N_ENGINES)),
        el.C(ValveCmd, jnp.zeros(N_VALVES)),
        el.C(EngineSpool, jnp.zeros(N_ENGINES)),
        el.C(EngineArmed, jnp.zeros(N_ENGINES)),
        el.C(TeaTebCharges, INITIAL_TEATEB_CHARGES),
        el.C(ValveState, jnp.zeros(N_VALVES)),
        el.C(ThrustTotal, jnp.array([0.0])),
        el.C(MdotTotal, jnp.array([0.0])),
        el.C(PropellantLox, jnp.array([lox_kg])),
        el.C(PropellantRp1, jnp.array([rp1_kg])),
        el.C(TankPressureLox, jnp.array([TANK_P_NOM_PA])),
        el.C(TankPressureRp1, jnp.array([TANK_P_NOM_PA])),
        el.C(InletPressureLox, jnp.array([TANK_P_NOM_PA])),
        el.C(InletPressureRp1, jnp.array([TANK_P_NOM_PA])),
        el.C(CgStation, jnp.array([propulsion.DRY_CG_STATION_M])),
        el.C(AxialSpecificForce, jnp.array([0.0])),
        el.C(WindEcef, jnp.zeros(3)),
        el.C(Qbar, jnp.array([0.0])),
        el.C(Mach, jnp.array([0.0])),
        el.C(TvcCmd, jnp.zeros(2)),
        el.C(TvcState, jnp.zeros(2)),
        el.C(FinCmd, jnp.zeros(3)),
        el.C(FinState, jnp.zeros(4)),
        el.C(RcsTorqueCmd, jnp.zeros(3)),
        el.C(RcsLevels, jnp.zeros(rcs.N_RCS)),
        el.C(NitrogenKg, jnp.array([N2_INITIAL_KG])),
        el.C(AeroWrench, jnp.zeros(6)),
        el.C(FinWrench, jnp.zeros(6)),
        el.C(RcsWrench, jnp.zeros(6)),
        el.C(EngineWrench, jnp.zeros(6)),
        el.C(AttitudeSetpoint, upright_attitude()),
        el.C(CtrlEnable, jnp.zeros(2)),
        el.C(FswPhase, jnp.array([0.0])),
        el.C(Landed, jnp.array([0.0])),
        *sn.sensor_components(),
    ]


def propulsion_systems(
    thrust_scale: float = 1.0,
    isp_scale: float = 1.0,
    ca_scale: float = 1.0,
    cn_scale: float = 1.0,
) -> el.System:
    """Non-effector plant pipeline: attitude inner loop -> valves/actuators ->
    engines -> mass -> tanks -> effector wrenches."""
    return (
        attitude_control
        | valve_dynamics
        | tvc_actuators
        | fin_actuators
        | make_engine_dynamics(thrust_scale, isp_scale)
        | mass_props
        | tank_dynamics
        | rcs_dynamics
        | engine_wrench
        | make_aero_dynamics(ca_scale, cn_scale)
    )


def build_powered(
    init_pos_ecef,
    init_vel_ecef,
    init_attitude: el.Quaternion | None = None,
    lox_kg: float = LOX_LOAD_KG,
    rp1_kg: float = RP1_LOAD_KG,
    upper_kg: float = 0.0,
    thrust_scale: float = 1.0,
    isp_scale: float = 1.0,
    ca_scale: float = 1.0,
    cn_scale: float = 1.0,
    extra_systems: el.System | None = None,
    integrator: el.Integrator = el.Integrator.SemiImplicit,
) -> tuple[el.World, el.System]:
    """Booster with the full powered plant (engines + TVC + aero + fins + RCS).

    `extra_systems` runs before the plant each tick — open-loop test scripts
    drive EngineCmd/ValveCmd/RcsTorqueCmd/FinCmd through it.
    """
    world = el.World()
    if init_attitude is None:
        init_attitude = el.Quaternion.identity()
    mass0, _cg0, inertia0 = propulsion.stack_mass_props(lox_kg, rp1_kg, upper_kg)
    # The hold-down clamp only applies to pad starts; test articles spawned
    # aloft are born released.
    on_pad = float(jnp.linalg.norm(jnp.asarray(init_pos_ecef) - pad_ecef())) < 100.0
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=init_attitude, linear=jnp.asarray(init_pos_ecef, dtype=jnp.float64)
                ),
                world_vel=el.SpatialMotion(linear=jnp.asarray(init_vel_ecef, dtype=jnp.float64)),
                inertia=el.SpatialInertia(float(mass0), inertia0),
            ),
            el.C(AltitudeGeodetic, jnp.array([0.0])),
            el.C(GroundSpeed, jnp.array([0.0])),
            el.C(ScoreState, jnp.zeros(3)),
            el.C(UpperMass, jnp.array([upper_kg])),
            el.C(Lifted, jnp.array([0.0 if on_pad else 1.0])),
            el.C(LiftoffTime, jnp.array([0.0])),
            el.C(TouchdownMetrics, jnp.zeros(3)),
            *booster_propulsion_components(lox_kg, rp1_kg),
        ],
        name="booster",
    )
    effectors = gravity_and_frame_forces | apply_body_wrenches
    plant = (
        propulsion_systems(thrust_scale, isp_scale, ca_scale, cn_scale)
        | el.six_dof(sys=effectors, integrator=integrator)
        | pad_clamp
        | ground_contact
    )
    system = (extra_systems | plant) if extra_systems is not None else plant
    return world, system | derive_geodetic_telemetry | sensor_systems()


def build_mission(
    ref,
    lox_kg: float = LOX_LOAD_KG,
    rp1_kg: float = RP1_LOAD_KG,
    upper_kg: float = 0.0,
    thrust_scale: float = 1.0,
    isp_scale: float = 1.0,
    ca_scale: float = 1.0,
    cn_scale: float = 1.0,
    display_lag_s: float = 0.0,
    extra_systems: el.System | None = None,
    integrator: el.Integrator = el.Integrator.SemiImplicit,
) -> tuple[el.World, el.System]:
    """The full mission world: booster upright on the pad, truth ghost
    replaying `ref`, display scoring accumulating in-sim."""
    world, system = build_powered(
        pad_ecef(),
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        lox_kg=lox_kg,
        rp1_kg=rp1_kg,
        upper_kg=upper_kg,
        thrust_scale=thrust_scale,
        isp_scale=isp_scale,
        ca_scale=ca_scale,
        cn_scale=cn_scale,
        extra_systems=extra_systems,
        integrator=integrator,
    )
    world.spawn(truth_ghost_components(ref), name="booster_truth")
    world.spawn(
        StaticSceneObject(el.WorldPos(linear=jnp.zeros(3))),
        name="earth",
    )
    schematic_path = Path(__file__).with_name("falcon9.kdl")
    world.schematic(schematic_path.read_text(), schematic_path.name)
    return world, system | make_truth_playback(ref, display_lag_s) | make_display_scoring(
        ref, display_lag_s
    )
