import math
import os
from pathlib import Path

import elodin as el
import jax.numpy as jnp
import numpy as np
import pytest

from frames import ecef_to_geodetic, enu_basis, geodetic_to_ecef
from ground import (
    GEAR_HEIGHT_M,
    LIFTOFF_TOLERANCE_M,
    TerrainContact,
    build_ground_contact,
    build_terrain_constraint,
    remember_terrain_pose,
)


def grid(heights=(100, 100, 100, 100)):
    return {
        "version": 1,
        "asset_id": 1,
        "ellipsoid": "WGS84",
        "bounds_degrees": [-0.001, -0.001, 0.001, 0.001],
        "rows": 2,
        "columns": 2,
        "heights_m": list(heights),
    }


def test_surface_height_slope_and_coverage():
    terrain = el.TerrainHeightmap(grid((100, 120, 100, 120)))
    height, normal, covered = terrain.sample_geodetic(0.0, 0.0)
    assert float(height) == pytest.approx(110.0)
    assert bool(covered)
    assert normal[0] > 0.99 and normal[1] < 0.0
    assert float(np.linalg.norm(normal)) == pytest.approx(1.0)
    assert not bool(terrain.sample_geodetic(0.0, math.radians(0.002))[2])


@pytest.mark.parametrize(
    "change",
    [
        {"ellipsoid": "MSL"},
        {"heights_m": [0, 0, 0, float("nan")]},
        {"heights_m": [0]},
        {"rows": 1},
        {"bounds_degrees": [1, 0, -1, 1]},
    ],
)
def test_invalid_ground_cannot_fall_back_to_a_flat_pad(change):
    with pytest.raises(ValueError):
        el.TerrainHeightmap({**grid(), **change})


def run_contact(terrain, altitude, velocity_enu, ticks=600):
    world = el.World()
    basis = enu_basis(0.0, 0.0)
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=geodetic_to_ecef(0.0, 0.0, altitude)),
                world_vel=el.SpatialMotion(linear=basis.T @ jnp.asarray(velocity_enu)),
                inertia=el.SpatialInertia(mass=20.0, inertia=jnp.ones(3)),
            ),
            TerrainContact(),
        ],
        name="body",
    )

    @el.map
    def gravity(pos: el.WorldPos, force: el.Force, inertia: el.Inertia) -> el.Force:
        lat, lon, _ = ecef_to_geodetic(pos.linear())
        return force + el.SpatialForce(linear=-9.81 * inertia.mass() * enu_basis(lat, lon)[2])

    system = (
        remember_terrain_pose
        | el.six_dof(
            sys=gravity | build_ground_contact(0.0, terrain),
            integrator=el.Integrator.SemiImplicit,
        )
        | build_terrain_constraint(terrain)
    )
    simulation = world.build(system, simulation_rate=300.0)
    simulation.run(ticks, show_progress=False)
    return simulation.history(
        ["body.terrain_clearance_m", "body.terrain_status", "body.world_pos", "body.world_vel"]
    )


def test_fast_impact_cannot_tunnel_through_terrain_above_the_old_pad():
    history = run_contact(el.TerrainHeightmap(grid()), 105.0, (0.0, 0.0, -100.0))
    clearance = np.asarray(history["body.terrain_clearance_m"].to_numpy())[1:]
    assert np.min(clearance) >= GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M - 1e-6
    assert np.any(np.asarray(history["body.terrain_status"].to_numpy()) == 1.0)
    assert np.all(np.isfinite(clearance))


def test_soft_touchdown_keeps_contact_without_stopping_the_flight():
    history = run_contact(el.TerrainHeightmap(grid()), 100.9, (0.0, 0.0, -0.2), ticks=120)
    status = np.asarray(history["body.terrain_status"].to_numpy())
    assert np.any(status == 1.0)
    assert not np.any(status >= 2.0)


def test_coverage_limit_stops_the_body_instead_of_using_fake_ground():
    history = run_contact(el.TerrainHeightmap(grid()), 500.0, (200.0, 0.0, 0.0))
    status = np.asarray(history["body.terrain_status"].to_numpy())
    assert status[-1] == 2.0
    first = int(np.flatnonzero(status == 2.0)[0])
    positions = np.stack([np.asarray(row.to_numpy()) for row in history["body.world_pos"]])
    np.testing.assert_allclose(
        positions[first:],
        np.broadcast_to(positions[first], positions[first:].shape),
        atol=1e-9,
        rtol=0,
    )


@pytest.mark.skipif(
    not os.environ.get("ELODIN_RC_JET_TEST_GROUND_FILE"), reason="requires prepared Mojave terrain"
)
def test_original_mountain_route_contacts_the_real_height_grid(monkeypatch):
    from test_sim import MODEL, FALLBACKS, NUMERICS, build_exec, column
    from scenario import load_scenario

    monkeypatch.setenv("ELODIN_RC_JET_HEADING_DEG", "350")
    terrain = el.TerrainHeightmap.load(Path(os.environ["ELODIN_RC_JET_TEST_GROUND_FILE"]))
    scenario = load_scenario(MODEL, FALLBACKS, name="demo")
    simulation = build_exec(scenario, terrain)
    simulation.run(90000, show_progress=False)
    history = simulation.history(
        ["bdx.terrain_clearance_m", "bdx.terrain_status", "bdx.world_pos", "bdx.world_vel"]
    )
    clearance = np.asarray(history["bdx.terrain_clearance_m"].to_numpy())[1:]
    status = np.asarray(history["bdx.terrain_status"].to_numpy())[1:]
    assert np.min(clearance) >= GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M - 1e-6
    assert np.any(status == 1.0), "The old mountain route must reach terrain contact"
    assert not np.any(status == 2.0), "The five-minute route must stay within coverage"
    assert np.all(np.isfinite(column(history, "bdx.world_pos")))
    assert np.all(np.isfinite(column(history, "bdx.world_vel")))
    first = np.flatnonzero(status == 1.0)[0] * NUMERICS.dt
    assert 100 < first < 240
