"""Closed-loop simulation acceptance tests (guide §7, §9.7).

These build the same world main.py flies (minus visualization and the RC
controller) and step it headless with the compiled backend.
"""

import math

import elodin as el
import jax.numpy as jnp
import numpy as np
import pytest

import bdx_model
from class_d_fallbacks import FALLBACKS
from scenario import MOJAVE_RC_FIELD, Numerics, Scenario, _initial_state, load_scenario
from sim import build_system, make_jet

MODEL = bdx_model.load()
NUMERICS = Numerics()


def build_exec(scenario: Scenario) -> el.Exec:
    world = el.World()
    init = scenario.initial
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=el.Quaternion(jnp.asarray(init.quat_xyzw)),
                    linear=jnp.asarray(init.pos_ecef),
                ),
                world_vel=el.SpatialMotion(linear=jnp.asarray(init.vel_ecef), angular=jnp.zeros(3)),
                inertia=el.SpatialInertia(
                    mass=MODEL.mass.operating_empty_mass_kg + init.fuel_kg,
                    inertia=np.array(
                        [
                            FALLBACKS.inertia.ixx_kg_m2,
                            FALLBACKS.inertia.iyy_kg_m2,
                            FALLBACKS.inertia.izz_kg_m2,
                        ]
                    ),
                ),
            ),
            make_jet(scenario),
        ],
        name="bdx",
    )
    system = build_system(MODEL, FALLBACKS, scenario, NUMERICS)
    return world.build(system, simulation_rate=1.0 / NUMERICS.dt)


def column(df, name) -> np.ndarray:
    series = df[name]
    return np.stack([np.asarray(row.to_numpy()) for row in series])


def scalar_column(df, name) -> np.ndarray:
    return np.asarray(df[name].to_numpy(), dtype=np.float64)


def test_validation_scenario_holds_trim():
    """T7/T13: the trim-row initialization flies level for 30 s."""
    scenario = load_scenario(MODEL, FALLBACKS, name="validation")
    exec = build_exec(scenario)
    exec.run(30 * 300, show_progress=False)
    df = exec.history(["bdx.geodetic", "bdx.alpha", "bdx.aero_valid", "bdx.velocity_body"])

    # Row 0 is the archetype default (pre-step); derived telemetry starts at tick 1.
    altitude = column(df, "bdx.geodetic")[1:, 2]
    anchor = MODEL.performance_anchors["cruise"]
    assert np.all(np.abs(altitude - anchor["altitude_m"]) < 5.0), (
        f"altitude drifted to [{altitude.min():.1f}, {altitude.max():.1f}]"
    )

    alpha_deg = np.degrees(scalar_column(df, "bdx.alpha"))
    assert abs(alpha_deg[-1] - anchor["alpha_deg"]) < 0.5

    assert np.all(scalar_column(df, "bdx.aero_valid") == 1.0)

    speed = np.linalg.norm(column(df, "bdx.velocity_body"), axis=1)
    assert abs(speed[-1] - anchor["tas_mps"]) < 2.0


def test_wind_invariance():
    """T6: matched airspeed state in steady wind leaves alpha/beta/qbar
    unchanged (rate terms now use wind-relative velocity, guide §9)."""
    calm = load_scenario(MODEL, FALLBACKS, name="validation")
    windy = load_scenario(MODEL, FALLBACKS, name="validation", wind_enu=(5.0, 3.0, 0.0))

    exec_calm = build_exec(calm)
    exec_windy = build_exec(windy)
    ticks = 300
    exec_calm.run(ticks, show_progress=False)
    exec_windy.run(ticks, show_progress=False)

    names = ["bdx.alpha", "bdx.beta", "bdx.dynamic_pressure"]
    df_calm = exec_calm.history(names)
    df_windy = exec_windy.history(names)
    # Tolerances absorb the Coriolis difference of the wind-carried ground
    # velocity (~1e-3 m/s^2 over one second).
    assert np.allclose(
        scalar_column(df_calm, "bdx.alpha"), scalar_column(df_windy, "bdx.alpha"), atol=2e-4
    )
    assert np.allclose(
        scalar_column(df_calm, "bdx.beta"), scalar_column(df_windy, "bdx.beta"), atol=2e-4
    )
    assert np.allclose(
        scalar_column(df_calm, "bdx.dynamic_pressure"),
        scalar_column(df_windy, "bdx.dynamic_pressure"),
        rtol=2e-4,
    )


def test_fuel_integrates_map_flow():
    """T10: fuel decreases exactly by the integrated map fuel flow."""
    scenario = load_scenario(MODEL, FALLBACKS, name="validation")
    exec = build_exec(scenario)
    ticks = 3000
    exec.run(ticks, show_progress=False)
    df = exec.history(["bdx.fuel_mass", "bdx.fuel_flow"])
    fuel = scalar_column(df, "bdx.fuel_mass")
    flow = scalar_column(df, "bdx.fuel_flow")
    burned = fuel[0] - fuel[-1]
    integrated = float(np.sum(flow[1:]) * NUMERICS.dt)
    assert burned == pytest.approx(integrated, abs=1e-4)
    assert burned > 0.01, "cruise for 10 s must burn measurable fuel"


def test_ground_rest():
    """T13: a dead-stick aircraft settles on the field surface without
    bouncing or sinking (contact normal along the geodetic up)."""
    site = MOJAVE_RC_FIELD
    from ground import GEAR_HEIGHT_M

    init = _initial_state(
        MODEL,
        site,
        altitude_m=site.field_elevation_m + GEAR_HEIGHT_M + 0.3,
        tas_mps=0.0,
        heading_deg=350.0,
        alpha_rad=0.0,
        elevator_rad=0.0,
        throttle=0.0,
        wind_enu=(0.0, 0.0, 0.0),
    )
    # Dead stick: no fuel means flameout, so idle thrust cannot drag the
    # airframe across the pad during the settling check.
    init = type(init)(**{**init.__dict__, "fuel_kg": 0.0})
    scenario = Scenario(
        name="ground-rest",
        site=site,
        altitude_m=init.pos_ecef[2],
        tas_mps=0.0,
        heading_deg=350.0,
        wind_enu=(0.0, 0.0, 0.0),
        initial=init,
        trim=None,
    )
    exec = build_exec(scenario)
    exec.run(1500, show_progress=False)
    df = exec.history(["bdx.geodetic", "bdx.velocity_body", "bdx.thrust"])
    altitude = column(df, "bdx.geodetic")[1:, 2]
    assert np.all(np.isfinite(altitude))
    rest_altitude = site.field_elevation_m + GEAR_HEIGHT_M
    assert altitude[-1] == pytest.approx(rest_altitude, abs=0.1)
    speed = np.linalg.norm(column(df, "bdx.velocity_body"), axis=1)
    assert speed[-1] < 0.1
    assert np.all(scalar_column(df, "bdx.thrust") == 0.0), "flameout must zero thrust"


def test_demo_scenario_solves_equilibrium():
    """The demo scenario refuses to spawn off-equilibrium and its solved trim
    stays inside the validity envelope."""
    scenario = load_scenario(MODEL, FALLBACKS, name="demo")
    assert scenario.trim.valid
    assert math.degrees(scenario.trim.alpha_rad) < MODEL.validity.attached_flow_alpha_deg[1]
