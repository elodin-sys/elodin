#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from pathlib import Path

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

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
AU_IN_KM = 149_597_870.7
CSV_PATHS: tuple[Path, ...] = (
    Path(__file__).with_name("planets_truth.csv"),
    # Path(__file__).with_name("moons_truth.csv"),
)
SUN_MASS_SOLAR = 1.0
SUN_RADIUS_KM = 696_340.0
SUN_COLOR = (255, 220, 120)
TRUTH_COLOR = "180 180 180"

PLANET_ICON = "public"
MOON_ICON = "circle"


@dataclass(frozen=True)
class BodyMeta:
    mass_solar: float
    radius_km: float
    color_rgb: tuple[int, int, int]
    icon: str


@dataclass(frozen=True)
class Body:
    name: str
    naif_id: int
    meta: BodyMeta


BODY_META: dict[str, BodyMeta] = {
    "mercury": BodyMeta(1.6605e-7, 2439.7, (185, 185, 185), PLANET_ICON),
    "venus": BodyMeta(2.4478e-6, 6051.8, (240, 200, 120), PLANET_ICON),
    "earth": BodyMeta(3.0035e-6, 6371.0, (90, 150, 255), PLANET_ICON),
    "mars": BodyMeta(3.2272e-7, 3389.5, (255, 120, 80), PLANET_ICON),
    "jupiter": BodyMeta(9.5459e-4, 69911.0, (210, 170, 130), PLANET_ICON),
    "saturn": BodyMeta(2.8588e-4, 58232.0, (230, 210, 150), PLANET_ICON),
    "uranus": BodyMeta(4.3662e-5, 25362.0, (150, 240, 255), PLANET_ICON),
    "neptune": BodyMeta(5.1514e-5, 24622.0, (90, 120, 255), PLANET_ICON),
    "pluto": BodyMeta(6.55e-9, 1188.3, (190, 180, 170), PLANET_ICON),
    "moon": BodyMeta(3.694e-8, 1737.4, (210, 210, 210), MOON_ICON),
    "mimas": BodyMeta(1.98e-11, 198.2, (170, 170, 170), MOON_ICON),
    "enceladus": BodyMeta(5.4e-10, 252.1, (190, 210, 230), MOON_ICON),
    "tethys": BodyMeta(3.09e-9, 531.1, (170, 170, 180), MOON_ICON),
    "dione": BodyMeta(5.5e-9, 561.7, (180, 180, 190), MOON_ICON),
    "rhea": BodyMeta(1.16e-8, 763.8, (190, 190, 200), MOON_ICON),
    "titan": BodyMeta(6.763e-8, 2574.7, (230, 180, 130), MOON_ICON),
    "hyperion": BodyMeta(2.8e-12, 135.0, (180, 170, 150), MOON_ICON),
    "iapetus": BodyMeta(9.05e-9, 734.5, (200, 180, 150), MOON_ICON),
    "phoebe": BodyMeta(4.2e-12, 106.5, (110, 110, 120), MOON_ICON),
    "helene": BodyMeta(1.3e-16, 17.6, (150, 150, 160), MOON_ICON),
    "telesto": BodyMeta(1.0e-15, 12.4, (150, 150, 160), MOON_ICON),
    "calypso": BodyMeta(6.0e-16, 10.7, (150, 150, 160), MOON_ICON),
    "methone": BodyMeta(1.0e-16, 1.6, (130, 130, 140), MOON_ICON),
    "polydeuces": BodyMeta(1.0e-16, 1.3, (130, 130, 140), MOON_ICON),
    "charon": BodyMeta(7.6e-10, 606.0, (170, 170, 170), MOON_ICON),
    "nix": BodyMeta(2.0e-14, 24.5, (150, 150, 160), MOON_ICON),
    "hydra": BodyMeta(2.0e-14, 31.0, (150, 150, 160), MOON_ICON),
    "kerberos": BodyMeta(1.0e-14, 12.0, (150, 150, 160), MOON_ICON),
    "styx": BodyMeta(1.0e-14, 8.0, (150, 150, 160), MOON_ICON),
}

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


def _radius_au(radius_km: float) -> float:
    return radius_km / AU_IN_KM


def _normalize_body_name(raw_name: str) -> str:
    body_name = raw_name.split(maxsplit=1)[-1].strip().lower()
    return body_name.replace("-", "_").replace(" ", "_")


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
                body_name = _normalize_body_name(row["name"])
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


def _kdl_color(rgb: tuple[int, int, int]) -> str:
    return f"{rgb[0]} {rgb[1]} {rgb[2]}"


def _build_schematic() -> str:
    lines: list[str] = [
        """
coordinate frame=ECEF
timeline follow_latest=#true

tabs {
    hsplit {
        tabs share=0.75 {
            viewport name=SolarSystem pos="(0,0,0,1, -6,6,6)" look_at="(0,0,0,1, 0,0,0)" hdr=#true show_grid=#true active=#true
            viewport name=TopDown pos="(0,0,0,1, 0,0,25)" look_at="(0,0,0,1, 0,0,0)" hdr=#true show_grid=#true 
        }
        tabs share=0.25 {
            graph "earth.world_pos[4],earth.world_pos[5],earth.world_pos[6],truth_earth.truth_world_pos[4],truth_earth.truth_world_pos[5],truth_earth.truth_world_pos[6]" name="Earth vs Truth (AU)"
            hierarchy
            inspector
        }
    }
}
"""
    ]
    sun_color = _kdl_color(SUN_COLOR)
    sun_radius = _radius_au(SUN_RADIUS_KM)
    lines.append(
        f"""
object_3d sun.world_pos {{
    sphere radius={sun_radius:.8f} {{
        color {sun_color}
    }}
    icon builtin="wb_sunny" {{
        visibility_range min=1.0 fade_distance=5.0
        color {sun_color}
    }}
}}
line_3d sun.world_pos line_width=1.5 perspective=#false {{
    color {sun_color}
}}
vector_arrow "(0,0,0.03)" origin="sun.world_pos" scale=1.0 name="sun" show_name=#true arrow_thickness=0.02 label_position=1.0 {{
    color {sun_color}
}}
"""
    )
    for body in BODIES:
        label = body.name.replace("_", " ")
        color = _kdl_color(body.meta.color_rgb)
        radius = _radius_au(body.meta.radius_km)
        truth_radius = radius * 0.6
        lines.append(
            f"""
object_3d {body.name}.world_pos {{
    sphere radius={radius:.8f} {{
        color {color}
    }}
    icon builtin="{body.meta.icon}" {{
        visibility_range min=1.0 fade_distance=5.0
        color {color}
    }}
}}
line_3d {body.name}.world_pos line_width=1.5 perspective=#false {{
    color {color}
}}
vector_arrow "(0,0,0.03)" origin="{body.name}.world_pos" scale=1.0 name="{label}" show_name=#true arrow_thickness=0.02 label_position=1.0 {{
    color {color}
}}
object_3d truth_{body.name}.truth_world_pos {{
    sphere radius={truth_radius:.8f} {{
        color {TRUTH_COLOR} 120
    }}
}}
line_3d truth_{body.name}.truth_world_pos line_width=1.0 perspective=#false {{
    color {TRUTH_COLOR} 80
}}
"""
        )
    return "".join(lines)


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

    world.schematic(_build_schematic(), "solar-system.kdl")
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
