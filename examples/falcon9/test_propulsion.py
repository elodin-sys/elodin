"""Phase 2 propulsion checks: thrust anchors, budget audit, actuator exactness,
mass-property stack, ignition gating, and an open-loop vertical burn against
an independent 1-D oracle."""

import math

import atmosphere
import jax.numpy as jnp
import numpy as np
import propulsion
from constants import (
    ENGINE_ISP_SL_S,
    ENGINE_T_SL_N,
    ENGINE_T_VAC_N,
    G0,
    LIFTOFF_MASS_KG,
    LOX_LOAD_KG,
    N_ENGINES,
    P_SL_PA,
    RP1_LOAD_KG,
    STAGE1_DRY_MASS_KG,
    STAGE1_LENGTH_M,
    STAGE1_PROP_KG,
)
from frames import apparent_gravity, ellipsoid_up
from sim import (
    N_VALVES,
    VALVE_HE_INFILL_LOX,
    VALVE_HE_INFILL_RP1,
    VALVE_MAIN_LOX,
    VALVE_MAIN_RP1,
    VALVE_TEATEB,
    EngineCmd,
    ValveCmd,
    build_powered,
    pad_ecef,
)

import elodin as el

PAD_LAT_RAD = math.radians(28.60839)
PAD_LON_RAD = math.radians(-80.60433)


def test_thrust_altitude_anchors():
    """WHITEPAPER 9.1: sea-level/vacuum pair and exit-area self-consistency."""
    t_sl = float(propulsion.engine_thrust_per_engine(1.0, P_SL_PA))
    t_vac = float(propulsion.engine_thrust_per_engine(1.0, 0.0))
    assert abs(t_sl - ENGINE_T_SL_N) < 1.0
    assert abs(t_vac - ENGINE_T_VAC_N) < 1.0
    a_e = (t_vac - t_sl) / P_SL_PA
    assert abs(a_e - 0.681) < 1e-3


def test_propellant_budget_audit():
    """WHITEPAPER 9.2: ~275 kg/s per engine; the four-burn budget closes."""
    mdot = float(propulsion.cluster_mdot(1.0, 1.0))
    assert abs(mdot - ENGINE_T_SL_N / (ENGINE_ISP_SL_S * G0)) < 0.5
    assert abs(mdot - 275.0) < 3.0
    ascent = 9 * mdot * 147.0 - 9 * mdot * 39.0 * 0.3  # bucket credit ~30%
    boostback = 3 * mdot * 46.0
    entry = 3 * mdot * 14.0
    landing = 1 * mdot * 0.7 * 33.0
    total = ascent + boostback + entry + landing
    assert 0.93 * STAGE1_PROP_KG < total < 1.01 * STAGE1_PROP_KG


def test_liftoff_thrust_to_weight():
    t_total = N_ENGINES * float(propulsion.engine_thrust_per_engine(1.0, P_SL_PA))
    twr = t_total / (LIFTOFF_MASS_KG * G0)
    assert 1.2 < twr < 1.4


def test_actuator_exact_discretization():
    """The e^(-dt/tau) update is exact at any dt and never overshoots."""
    tau = 0.007
    # dt = 1 ms steps: after exactly tau seconds, 63.2% of the step.
    x = jnp.array(0.0)
    for _ in range(7):
        x = propulsion.actuator_step(x, 1.0, 0.001, tau)
    assert abs(float(x) - (1.0 - math.exp(-1.0))) < 1e-9
    # One giant step (dt = 100 tau) lands on the command without overshoot.
    x_big = propulsion.actuator_step(jnp.array(0.0), 1.0, 0.7, tau)
    assert 0.0 < float(x_big) <= 1.0
    assert abs(float(x_big) - 1.0) < 1e-9
    # Rate limit engages: 10/s limit over 1 ms caps the step at 0.01.
    x_rl = propulsion.actuator_step(jnp.array(0.0), 1.0, 0.001, 1e-6, rate_limit=10.0)
    assert abs(float(x_rl) - 0.01) < 1e-12


def test_stack_mass_props():
    mass, cg, inertia = propulsion.stack_mass_props(LOX_LOAD_KG, RP1_LOAD_KG)
    assert abs(float(mass) - (STAGE1_DRY_MASS_KG + STAGE1_PROP_KG)) < 1.0
    assert 0.0 < float(cg) < STAGE1_LENGTH_M
    assert np.all(np.asarray(inertia) > 0.0)
    # Transverse inertia dominates axial for a slender stage.
    assert float(inertia[1]) > 10.0 * float(inertia[0])
    # CG walks down while propellant drains (columns drain top-down, LOX tank
    # on top); near-empty the low-sitting residual columns fall below the dry
    # CG, so the fully-dry stage pops back up to the dry station.
    cgs = []
    for frac in (1.0, 0.6, 0.3):
        _, cg_f, _ = propulsion.stack_mass_props(LOX_LOAD_KG * frac, RP1_LOAD_KG * frac)
        cgs.append(float(cg_f))
    assert cgs[0] > cgs[1] > cgs[2]
    _, cg_dry, _ = propulsion.stack_mass_props(0.0, 0.0)
    assert abs(float(cg_dry) - propulsion.DRY_CG_STATION_M) < 1e-6
    assert cgs[2] < float(cg_dry) < cgs[0]


def _script(valve_profile, engine_profile) -> el.System:
    """Tick-driven open-loop command script for EngineCmd/ValveCmd."""

    @el.system
    def script(
        tick: el.Query[el.SimulationTick], q: el.Query[EngineCmd]
    ) -> el.Query[EngineCmd, ValveCmd]:
        t = tick[0] * 0.001
        return q.map(
            (EngineCmd, ValveCmd),
            lambda _cmd: (engine_profile(t), valve_profile(t)),
        )

    return script


def _open_feed(t):
    v = jnp.zeros(N_VALVES)
    v = v.at[VALVE_MAIN_LOX].set(1.0)
    v = v.at[VALVE_MAIN_RP1].set(1.0)
    v = v.at[VALVE_TEATEB].set(1.0)
    v = v.at[VALVE_HE_INFILL_LOX].set(1.0)
    v = v.at[VALVE_HE_INFILL_RP1].set(1.0)
    return v


def _up_attitude() -> el.Quaternion:
    """Quaternion rotating body +X onto the pad's geodetic up."""
    up = np.asarray(ellipsoid_up(PAD_LAT_RAD, PAD_LON_RAD))
    x = np.array([1.0, 0.0, 0.0])
    axis = np.cross(x, up)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(float(np.clip(x @ up, -1.0, 1.0)))
    return el.Quaternion.from_axis_angle(jnp.asarray(axis), jnp.array(angle))


def _run_powered(valve_profile, engine_profile, steps, lox=LOX_LOAD_KG, rp1=RP1_LOAD_KG):
    world, system = build_powered(
        pad_ecef(),
        jnp.zeros(3),
        init_attitude=_up_attitude(),
        lox_kg=lox,
        rp1_kg=rp1,
        extra_systems=_script(valve_profile, engine_profile),
    )
    sim = world.to_jax(system, simulation_rate=1000.0)
    sim.step(steps)

    def get(name):
        return np.asarray(sim.get_state(name), dtype=np.float64).reshape(-1)

    return get


def test_ignition_gating_and_relight_budget():
    """No TEA-TEB valve, no light; a lit-then-cut outer engine cannot relight."""

    # Feed valves open but igniter isolation closed: engines must stay dark.
    def valves_no_teateb(t):
        v = _open_feed(t)
        return v.at[VALVE_TEATEB].set(0.0)

    get = _run_powered(valves_no_teateb, lambda t: jnp.ones(N_ENGINES), 200)
    assert float(get("thrust_total")[0]) == 0.0
    np.testing.assert_allclose(get("teateb_charges"), [4, 4, 4, 1, 1, 1, 1, 1, 1])

    # Light all 9, cut at t=2 s, re-command all at t=4 s: only the three
    # relight-capable engines (one charge spent at liftoff) come back.
    def engines(t):
        on = jnp.ones(N_ENGINES)
        return jnp.where((t >= 0.1) & (t < 2.0), on, jnp.where(t >= 4.0, on, jnp.zeros(N_ENGINES)))

    get = _run_powered(_open_feed, engines, 6000)
    spool = get("engine_spool")
    charges = get("teateb_charges")
    assert np.all(spool[:3] > 0.5), f"relight-capable engines should burn: {spool}"
    assert np.all(spool[3:] < 1e-3), f"outer engines must not relight: {spool}"
    np.testing.assert_allclose(charges, [2, 2, 2, 0, 0, 0, 0, 0, 0])


def test_open_loop_vertical_burn_vs_oracle():
    """20 s pad-vertical full-throttle burn vs an independent 1-D oracle."""
    t_light = 0.5

    def engines(t):
        return jnp.where(t >= t_light, jnp.ones(N_ENGINES), jnp.zeros(N_ENGINES))

    get = _run_powered(_open_feed, engines, 20_000)
    alt_sim = float(get("altitude_geodetic")[0])
    speed_sim = float(get("ground_speed")[0])
    lox = float(get("propellant_lox")[0])
    rp1 = float(get("propellant_rp1")[0])
    inlet_lox = float(get("inlet_pressure_lox")[0])
    tank_lox = float(get("tank_pressure_lox")[0])

    # Independent 1-D oracle: same thrust/mass model, apparent gravity at the
    # pad, same exponential spool, same hold-down-clamp release (thrust >
    # weight) as the plant's pad_clamp system.
    g = float(np.linalg.norm(apparent_gravity(pad_ecef())))
    dt = 0.001
    h, v = 3.0, 0.0
    m = float(propulsion.stack_mass_props(LOX_LOAD_KG, RP1_LOAD_KG)[0])
    spool = 0.0
    released = False
    for i in range(20_000):
        t = i * dt
        target = 1.0 if t >= t_light else 0.0
        # Three-regime spool, mirroring sim.engine_dynamics: slow turbopump
        # spin-up from cold, fast throttle response once running.
        running = spool > 0.5 * 0.57
        if target > spool:
            tau = 0.15 if running else 1.5
        else:
            tau = 0.35
        spool += (1.0 - math.exp(-dt / tau)) * (target - spool)
        level = spool if spool > 1e-3 else 0.0
        p_amb = float(atmosphere.pressure(max(h, 0.0)))
        thrust = 9.0 * max(level * ENGINE_T_VAC_N - p_amb * 0.681, 0.0)
        mdot = 9.0 * level * ENGINE_T_VAC_N / (propulsion.ENGINE_ISP_VAC_S * G0) if level else 0.0
        released = released or thrust > m * 9.79
        a = thrust / m - g
        # Semi-implicit ordering to match the plant integrator.
        if released:
            v += a * dt
            h += v * dt
        m -= mdot * dt
    m_burned_oracle = LOX_LOAD_KG + RP1_LOAD_KG - (lox + rp1)
    m_burned_expected = float(propulsion.stack_mass_props(LOX_LOAD_KG, RP1_LOAD_KG)[0]) - m

    assert abs(alt_sim - h) < 0.03 * max(h, 1.0) + 5.0, f"sim {alt_sim} vs oracle {h}"
    assert abs(speed_sim - v) < 0.03 * v + 2.0, f"sim {speed_sim} vs oracle {v}"
    assert abs(m_burned_oracle - m_burned_expected) < 0.02 * m_burned_expected
    # Inlet pressure carries the acceleration head above tank pressure.
    assert inlet_lox > tank_lox
    # Note: the ~3.6 g near-MECO figure is asserted in the Phase 5 closed-loop
    # ascent test where the full 147 s burn runs in the compiled hot loop.


def test_valve_response_time():
    """Valve state reaches >86% of a step command in ~2 tau (30 ms)."""

    def valves(t):
        return jnp.where(t >= 0.1, jnp.ones(N_VALVES), jnp.zeros(N_VALVES))

    get = _run_powered(valves, lambda t: jnp.zeros(N_ENGINES), 130)
    v = get("valve_state")
    expected = 1.0 - math.exp(-0.030 / 0.015)
    np.testing.assert_allclose(v, expected, atol=0.02)
