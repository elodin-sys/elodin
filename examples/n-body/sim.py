#!/usr/bin/env python3

import csv
from pathlib import Path

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
from body_metadata import (
    BODY_META,
    CSV_PATHS,
    SUN_MASS_SOLAR,
    Body,
    normalize_body_name,
)
from schematic import build as build_schematic

# Truth data is in AU/day; Elodin tick rates are in Hz (seconds), so convert
# the Gaussian gravitational constant to AU^3 / (solar_mass * s^2).
K_SQUARED_DAY = 2.9591220828e-4
SOFTENING_AU2 = 1.0e-10
SECONDS_PER_DAY = 86_400.0
K_SQUARED = K_SQUARED_DAY / (SECONDS_PER_DAY * SECONDS_PER_DAY)
SIMULATION_RATE_HZ = 1.0 / 3600.0  # 1 tick per hour
TELEMETRY_RATE_HZ = 1.0 / SECONDS_PER_DAY  # 1 sample per simulated day
TICKS_PER_DAY = int(round(SECONDS_PER_DAY * SIMULATION_RATE_HZ))
TICKS_PER_TELEMETRY = int(round(SIMULATION_RATE_HZ / TELEMETRY_RATE_HZ))
if abs(TICKS_PER_TELEMETRY * TELEMETRY_RATE_HZ - SIMULATION_RATE_HZ) > 1e-12:
    raise ValueError("telemetry_rate_hz must evenly divide simulation_rate_hz")
DEFAULT_DB_PATH = "dbs/n-body-solar-system"
DB_NAME_ENV = "DBNAME"
TruthIdx = el.Annotated[
    jax.Array,
    el.Component("truth_idx", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TruthWorldPos = el.Annotated[
    jax.Array,
    el.Component("truth_world_pos", el.ComponentType(el.PrimitiveType.F64, (7,))),
]
GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]

TRUTH_POSITIONS: jax.Array | None = None
TRUTH_POSITIONS_NP: np.ndarray | None = None
MAX_DAY_INDEX: int = 0
TRUTH_DAY_COUNT: int = 0
BODIES: list[Body] = []


@el.dataclass
class GravityConstraint(el.Archetype):
    edge: GravityEdge

    def __init__(self, src: el.EntityId, dst: el.EntityId):
        self.edge = GravityEdge(src, dst)


@el.dataclass
class TruthBody(el.Archetype):
    truth_world_pos: TruthWorldPos
    truth_idx: TruthIdx


def load_truth(csv_paths: tuple[Path, ...] = CSV_PATHS) -> tuple[jax.Array, np.ndarray, list[str]]:
    global BODIES

    positions_by_id: dict[int, list[list[float]]] = {}
    velocities_by_id: dict[int, list[list[float]]] = {}
    dates_by_id: dict[int, list[str]] = {}
    names_by_id: dict[int, str] = {}
    body_order: list[int] = []

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                body_id = int(row["naif_id"])
                body_name = normalize_body_name(row["name"])
                existing_name = names_by_id.get(body_id)
                if existing_name is None:
                    names_by_id[body_id] = body_name
                    body_order.append(body_id)
                    positions_by_id[body_id] = []
                    velocities_by_id[body_id] = []
                    dates_by_id[body_id] = []
                elif existing_name != body_name:
                    raise ValueError(
                        f"conflicting body names for naif_id={body_id}: "
                        f"{existing_name!r} vs {body_name!r}"
                    )

                positions_by_id[body_id].append(
                    [float(row["x_au"]), float(row["y_au"]), float(row["z_au"])]
                )
                velocities_by_id[body_id].append(
                    [
                        float(row["vx_au_per_day"]) / SECONDS_PER_DAY,
                        float(row["vy_au_per_day"]) / SECONDS_PER_DAY,
                        float(row["vz_au_per_day"]) / SECONDS_PER_DAY,
                    ]
                )
                dates_by_id[body_id].append(row["date"])

    discovered_bodies: list[Body] = []
    for body_id in body_order:
        body_name = names_by_id[body_id]
        meta = BODY_META.get(body_name)
        if meta is None:
            print(f"warning: skipping unsupported body {body_name!r} (naif_id={body_id})")
            continue
        discovered_bodies.append(Body(name=body_name, naif_id=body_id, meta=meta))

    if not discovered_bodies:
        raise ValueError("no supported bodies found in configured CSV files")

    reference_id = discovered_bodies[0].naif_id
    dates = dates_by_id[reference_id]
    n_bodies = len(discovered_bodies)
    n_days = len(dates)
    pos = np.zeros((n_bodies, n_days, 3), dtype=np.float64)
    vel = np.zeros((n_bodies, n_days, 3), dtype=np.float64)

    for idx, body in enumerate(discovered_bodies):
        body_dates = dates_by_id[body.naif_id]
        if len(body_dates) != n_days:
            raise ValueError(
                f"missing truth rows for body {body.name} ({body.naif_id}): "
                f"{len(body_dates)} != {n_days}"
            )
        if body_dates != dates:
            raise ValueError(
                f"date mismatch for body {body.name} ({body.naif_id}) against reference timeline"
            )
        pos[idx] = np.asarray(positions_by_id[body.naif_id], dtype=np.float64)
        vel[idx] = np.asarray(velocities_by_id[body.naif_id], dtype=np.float64)

    BODIES = discovered_bodies
    return jnp.array(pos), vel, dates


def build_world() -> el.World:
    global TRUTH_POSITIONS, TRUTH_POSITIONS_NP, MAX_DAY_INDEX, TRUTH_DAY_COUNT
    TRUTH_POSITIONS, truth_velocities, dates = load_truth()
    TRUTH_POSITIONS_NP = np.asarray(TRUTH_POSITIONS)
    TRUTH_DAY_COUNT = len(dates)
    MAX_DAY_INDEX = len(dates) - 1

    world = el.World()
    sun = world.spawn(
        el.Body(
            world_pos=el.SpatialTransform(),
            world_vel=el.SpatialMotion(),
            inertia=el.SpatialInertia(mass=SUN_MASS_SOLAR),
        ),
        name="sun",
        id="sun",
    )
    sim_entities: list[el.EntityId] = [sun]
    for body_idx, body in enumerate(BODIES):
        sim_entity = world.spawn(
            el.Body(
                world_pos=el.SpatialTransform(linear=TRUTH_POSITIONS[body_idx, 0]),
                world_vel=el.SpatialMotion(linear=jnp.array(truth_velocities[body_idx, 0])),
                inertia=el.SpatialInertia(mass=body.meta.mass_solar),
            ),
            name=body.name,
            id=body.name,
        )
        sim_entities.append(sim_entity)
        world.spawn(
            TruthBody(
                truth_world_pos=jnp.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        float(TRUTH_POSITIONS[body_idx, 0, 0]),
                        float(TRUTH_POSITIONS[body_idx, 0, 1]),
                        float(TRUTH_POSITIONS[body_idx, 0, 2]),
                    ],
                    dtype=jnp.float64,
                ),
                truth_idx=jnp.array([float(body_idx)], dtype=jnp.float64),
            ),
            name=f"truth_{body.name}",
            id=f"truth_{body.name}",
        )

    for i, src in enumerate(sim_entities):
        for j, dst in enumerate(sim_entities):
            if i == j:
                continue
            world.spawn(GravityConstraint(src, dst))

    world.schematic(build_schematic(BODIES), "solar-system.kdl")
    return world


@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    q: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_fn(
        acc: el.Force,
        a_pos: el.WorldPos,
        a_inertia: el.Inertia,
        b_pos: el.WorldPos,
        b_inertia: el.Inertia,
    ) -> el.Force:
        r = b_pos.linear() - a_pos.linear()
        dist_sq = jnp.dot(r, r) + SOFTENING_AU2
        inv_dist = jnp.reciprocal(jnp.sqrt(dist_sq))
        inv_dist3 = inv_dist * inv_dist * inv_dist
        scalar = K_SQUARED * a_inertia.mass() * b_inertia.mass() * inv_dist3
        return acc + el.SpatialForce(linear=scalar * r)

    return graph.edge_fold(
        left_query=q,
        right_query=q,
        return_type=el.Force,
        init_value=el.SpatialForce(),
        fold_fn=gravity_fn,
    )


def make_truth_post_step():
    if TRUTH_POSITIONS_NP is None:
        raise RuntimeError("truth positions not initialized; call build_world() first")

    def post_step(tick: int, ctx: el.StepContext):
        day_idx = min(int(tick // TICKS_PER_DAY), MAX_DAY_INDEX)
        for body_idx, body in enumerate(BODIES):
            pos = TRUTH_POSITIONS_NP[body_idx, day_idx]
            world_pos = np.array(
                [0.0, 0.0, 0.0, 1.0, float(pos[0]), float(pos[1]), float(pos[2])],
                dtype=np.float64,
            )
            ctx.write_component(
                f"truth_{body.name}.truth_world_pos",
                world_pos,
            )

    return post_step


def build_system() -> el.System:
    return el.six_dof(sys=gravity, integrator=el.Integrator.Rk4)


def get_default_max_ticks() -> int:
    if TRUTH_DAY_COUNT <= 1:
        return TICKS_PER_DAY
    # Simulate the full truth timeline from day 0 to final day.
    return (TRUTH_DAY_COUNT - 1) * TICKS_PER_DAY
