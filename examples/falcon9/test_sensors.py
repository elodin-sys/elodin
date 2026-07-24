"""Phase 4 checks: specific-force identity, Earth-rate gyro, GPS cadence and
hold, radar geometry gates, display quantization, and in-sim scoring."""

import math

import jax.numpy as jnp
import numpy as np
import sensors as sn
from constants import N_ENGINES, OMEGA_EARTH_RADPS
from reference import build_reference, sanity_check
from sim import (
    N_VALVES,
    VALVE_HE_INFILL_LOX,
    VALVE_HE_INFILL_RP1,
    VALVE_MAIN_LOX,
    VALVE_MAIN_RP1,
    VALVE_TEATEB,
    EngineCmd,
    ValveCmd,
    build_mission,
    build_powered,
    pad_ecef,
    upright_attitude,
)

import elodin as el


def _script(valve_profile, engine_profile) -> el.System:
    @el.system
    def script(
        tick: el.Query[el.SimulationTick], q: el.Query[EngineCmd]
    ) -> el.Query[EngineCmd, ValveCmd]:
        t = tick[0] * 0.001
        return q.map((EngineCmd, ValveCmd), lambda _c: (engine_profile(t), valve_profile(t)))

    return script


def _no_valves(t):
    return jnp.zeros(N_VALVES)


def _open_feed(t):
    v = jnp.zeros(N_VALVES)
    for i in (
        VALVE_MAIN_LOX,
        VALVE_MAIN_RP1,
        VALVE_TEATEB,
        VALVE_HE_INFILL_LOX,
        VALVE_HE_INFILL_RP1,
    ):
        v = v.at[i].set(1.0)
    return v


def _engines_off(t):
    return jnp.zeros(N_ENGINES)


def _run(world, system, steps, rate=1000.0):
    sim = world.to_jax(system, simulation_rate=rate)
    sim.step(steps)

    def get(name, entity=None):
        val = sim.get_state(name, entity) if entity else sim.get_state(name)
        return np.asarray(val, dtype=np.float64).reshape(-1)

    return get


def test_reference_profiles():
    for mission in ("crs12", "crs11"):
        sanity_check(build_reference(mission))


def test_imu_specific_force_identity_and_earth_rate_gyro():
    """Freefall, engines off: accelerometer ~0 (specific force!), gyro reads
    exactly the Earth rate in the body frame."""
    world, system = build_powered(
        pad_ecef(),
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        extra_systems=_script(_no_valves, _engines_off),
    )
    get = _run(world, system, 50)
    accel = get("imu_accel")
    gyro = get("imu_gyro")
    assert np.linalg.norm(accel) < 5 * sn.IMU_ACCEL_SIGMA + 1e-6
    assert abs(np.linalg.norm(gyro) - OMEGA_EARTH_RADPS) < 5 * sn.IMU_GYRO_SIGMA


def test_imu_reads_thrust_specific_force():
    """Engines burning: f_B = T/m along body +X."""

    def engines(t):
        return jnp.where(t >= 0.05, jnp.ones(N_ENGINES), jnp.zeros(N_ENGINES))

    world, system = build_powered(
        pad_ecef(),
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        extra_systems=_script(_open_feed, engines),
    )
    get = _run(world, system, 8000)
    accel = get("imu_accel")
    thrust = float(get("thrust_total")[0])
    lox, rp1 = float(get("propellant_lox")[0]), float(get("propellant_rp1")[0])
    import propulsion

    mass = float(propulsion.stack_mass_props(lox, rp1)[0])
    assert thrust > 5e6
    assert abs(accel[0] - thrust / mass) < 0.05
    assert abs(accel[1]) < 0.1 and abs(accel[2]) < 0.1


def test_gps_cadence_hold_and_display_quantization():
    world, system = build_powered(
        pad_ecef(),
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        extra_systems=_script(_no_valves, _engines_off),
    )
    get = _run(world, system, 1000)  # exactly 1 s
    count = float(get("gps_count")[0])
    assert abs(count - 25.0) <= 1.0, f"GPS samples in 1 s: {count}"
    # Held measurement is close to the (slowly falling) truth position.
    gps_pos = get("gps_pos")
    truth = np.asarray(get("world_pos"))[4:]
    assert np.linalg.norm(gps_pos - truth) < 25.0
    # Display quantization steps.
    dspeed = float(get("display_speed")[0])
    dalt = float(get("display_alt")[0])
    assert abs(dspeed / sn.DISPLAY_SPEED_STEP - round(dspeed / sn.DISPLAY_SPEED_STEP)) < 1e-9
    assert abs(dalt / sn.DISPLAY_ALT_STEP - round(dalt / sn.DISPLAY_ALT_STEP)) < 1e-9


def test_radar_geometry_gates():
    # 100 m up, nose up: boresight (-X body) looks straight down -> range ~ alt.
    import frames

    up = frames.ellipsoid_up(math.radians(28.60839), math.radians(-80.60433))
    world, system = build_powered(
        pad_ecef() + up * 100.0,
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        extra_systems=_script(_no_valves, _engines_off),
    )
    get = _run(world, system, 100)
    rng_meas = float(get("radar_range")[0])
    alt = float(get("altitude_geodetic")[0])
    assert rng_meas > 0.0
    assert abs(rng_meas - alt) < 2.0
    # 45-degree tilt about the pad's local NORTH axis exceeds the 35-degree
    # FOV: invalid (-1). (Tilting about a world axis would not tilt from
    # vertical by the same angle.)
    north = frames.ned_basis(math.radians(28.60839), math.radians(-80.60433))[0]
    tilt = el.Quaternion.from_axis_angle(jnp.asarray(north), jnp.array(math.radians(45.0)))
    world, system = build_powered(
        pad_ecef() + up * 100.0,
        jnp.zeros(3),
        init_attitude=tilt * upright_attitude(),
        extra_systems=_script(_no_valves, _engines_off),
    )
    get = _run(world, system, 100)
    assert float(get("radar_range")[0]) < 0.0
    # 10 km up: beyond max range, invalid.
    world, system = build_powered(
        pad_ecef() + up * 10_000.0,
        jnp.zeros(3),
        init_attitude=upright_attitude(),
        extra_systems=_script(_no_valves, _engines_off),
    )
    get = _run(world, system, 100)
    assert float(get("radar_range")[0]) < 0.0


def test_mission_scoring_gates_on_liftoff():
    """Scoring is liftoff-referenced: zero samples while clamped, accumulating
    once the clamp releases (recorded t=0 is liftoff, not sim start)."""
    ref = build_reference("crs12")
    # Engines dark: never lifts, never scores.
    world, system = build_mission(ref, extra_systems=_script(_no_valves, _engines_off))
    get = _run(world, system, 500)
    score = get("score_state")
    assert score[2] == 0.0
    # Ghost holds the liftoff state while clamped.
    ghost_alt = float(get("altitude_geodetic", "booster_truth")[0])
    assert abs(ghost_alt - ref.altitude(0.0)) < 150.0

    # Engines lit: the clamp releases once thrust > weight, scoring begins.
    def engines(t):
        return jnp.where(t >= 0.05, jnp.ones(N_ENGINES), jnp.zeros(N_ENGINES))

    world, system = build_mission(ref, extra_systems=_script(_open_feed, engines))
    get = _run(world, system, 4000)
    assert float(get("lifted")[0]) == 1.0
    liftoff_t = float(get("liftoff_time")[0])
    assert 0.0 < liftoff_t < 3.0
    score = get("score_state")
    expected_samples = 4000 - round(liftoff_t * 1000.0)
    assert abs(score[2] - expected_samples) <= 1.0
    assert score[0] >= 0.0
    # Ghost is being driven: altitude tracks the reference, not the pad.
    # (altitude_geodetic exists on both entities; index the ghost's row.)
    ghost_alt = float(get("altitude_geodetic", "booster_truth")[0])
    assert abs(ghost_alt - ref.altitude(0.5)) < 200.0
