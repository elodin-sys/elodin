"""Shared Voyager world construction and gravity systems."""

import typing as ty

import elodin as el
import jax
from jax import numpy as jnp
from jax.numpy import linalg as la
import numpy as np
import spiceypy as spice

from dynamics import heliocentric_relative_acceleration
from gravity_parameters import DE440_GM_M3_S2


GravitationalParameter = ty.Annotated[
    jax.Array,
    el.Component(
        "gravitational_parameter_m3_s2",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
    ),
]
GravityEdge = el.Annotated[
    el.Edge,
    el.Component("gravity_edge", el.ComponentType.Edge),
]


@el.dataclass
class GravityConstraint(el.Archetype):
    a: GravityEdge

    def __init__(self, a: el.EntityId, b: el.EntityId):
        self.a = GravityEdge(a, b)


@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    probe_query: el.Query[el.WorldPos, el.Inertia],
    source_query: el.Query[el.WorldPos, GravitationalParameter],
) -> el.Query[el.Force]:
    """Chapter 1 direct source-body gravity."""

    def gravity_fn(force, probe_pos, probe_inertia, source_pos, source_gm):
        r = probe_pos.linear() - source_pos.linear()
        mass = probe_inertia.mass()
        mu = source_gm[0]
        norm = la.norm(r)
        f = mu * mass * r / (norm * norm * norm)
        return el.Force(linear=force.force() - f)

    return graph.edge_fold(
        left_query=probe_query,
        right_query=source_query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_fn,
    )


@el.system
def heliocentric_gravity(
    graph: el.GraphQuery[GravityEdge],
    probe_query: el.Query[el.WorldPos, el.Inertia],
    source_query: el.Query[el.WorldPos, GravitationalParameter],
) -> el.Query[el.Force]:
    """Chapter 2 heliocentric-relative source-body gravity."""

    def gravity_fn(force, probe_pos, probe_inertia, source_pos, source_gm):
        acc = heliocentric_relative_acceleration(
            probe_pos.linear(), source_pos.linear(), source_gm[0]
        )
        return el.Force(linear=force.force() + probe_inertia.mass() * acc)

    return graph.edge_fold(
        left_query=probe_query,
        right_query=source_query,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=gravity_fn,
    )


PLANETS = [
    {
        "spice_name": "MERCURY BARYCENTER",
        "entity_name": "mercury",
        "radius": 2000000000.0,
        "color": "white",
        "trail_color": "white 220",
        "mass": 3.3011e23,
    },
    {
        "spice_name": "VENUS BARYCENTER",
        "entity_name": "venus",
        "radius": 3000000000.0,
        "color": "peach",
        "trail_color": "peach 220",
        "mass": 4.8675e24,
    },
    {
        "spice_name": "EARTH",
        "entity_name": "earth",
        "radius": 6000000000.0,
        "color": "hyperblue",
        "trail_color": "hyperblue 220",
        "mass": 5.97219e24,
    },
    {
        "spice_name": "MARS BARYCENTER",
        "entity_name": "mars",
        "radius": 4000000000.0,
        "color": "red",
        "trail_color": "red 220",
        "mass": 6.4171e23,
    },
    {
        "spice_name": "JUPITER BARYCENTER",
        "entity_name": "jupiter",
        "radius": 12000000000.0,
        "color": "orange",
        "trail_color": "orange 220",
        "mass": 1.898125e27,
    },
    {
        "spice_name": "SATURN BARYCENTER",
        "entity_name": "saturn",
        "radius": 10000000000.0,
        "color": "yolk",
        "trail_color": "yolk 220",
        "mass": 5.6834e26,
    },
    {
        "spice_name": "URANUS BARYCENTER",
        "entity_name": "uranus",
        "radius": 8000000000.0,
        "color": "cyan",
        "trail_color": "cyan 220",
        "mass": 8.6813e25,
    },
    {
        "spice_name": "NEPTUNE BARYCENTER",
        "entity_name": "neptune",
        "radius": 8000000000.0,
        "color": "blue",
        "trail_color": "blue 220",
        "mass": 1.02413e26,
    },
]
for planet in PLANETS:
    planet["gm"] = DE440_GM_M3_S2[planet["spice_name"]]

SUN_MASS_KG = 1.9885e30
SUN_GM_M3_S2 = DE440_GM_M3_S2["SUN"]


def spice_state(
    spice_name: str,
    epoch_et: float,
    *,
    frame: str = "ECLIPJ2000",
    observer: str = "SUN",
) -> tuple[np.ndarray, np.ndarray]:
    """Return one SPICE state in SI units."""

    state_km, _ = spice.spkezr(spice_name, epoch_et, frame, "NONE", observer)
    state = np.asarray(state_km, dtype=np.float64)
    return state[:3] * 1000.0, state[3:] * 1000.0


def build_world(
    start_time_et: float,
    *,
    probes: ty.Sequence[dict],
    truth_probes: ty.Sequence[dict] = (),
    extra_components: ty.Callable[[dict], ty.Sequence] | None = None,
    frame: str = "ECLIPJ2000",
    observer: str = "SUN",
    log_initial_states: bool = False,
) -> tuple[el.World, dict[str, el.EntityId]]:
    """Build the shared Voyager gravity world used by the example and validator."""

    world = el.World()
    sun = world.spawn(
        [
            el.Body(
                world_pos=el.WorldPos(linear=jnp.array([0.0, 0.0, 0.0])),
                world_vel=el.WorldVel(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.Inertia(SUN_MASS_KG),
            ),
            el.C(
                GravitationalParameter,
                jnp.array([SUN_GM_M3_S2], dtype=jnp.float64),
            ),
        ],
        name="Sun",
    )
    body_entity_ids = {"Sun": sun}

    for body in [*PLANETS, *probes, *truth_probes]:
        position_m, velocity_mps = spice_state(
            body["spice_name"],
            start_time_et,
            frame=frame,
            observer=observer,
        )

        if log_initial_states:
            print(body["spice_name"])
            print(position_m)
            print(velocity_mps)

        components = [
            el.Body(
                world_pos=el.WorldPos(linear=jnp.asarray(position_m)),
                world_vel=el.WorldVel(linear=jnp.asarray(velocity_mps)),
                inertia=el.Inertia(body["mass"]),
            )
        ]
        if "gm" in body:
            components.append(
                el.C(
                    GravitationalParameter,
                    jnp.array([body["gm"]], dtype=jnp.float64),
                )
            )
        if extra_components is not None:
            components.extend(extra_components(body))

        body_entity_ids[body["entity_name"]] = world.spawn(
            components,
            name=body["entity_name"],
        )

    source_names = ["Sun", *[planet["entity_name"] for planet in PLANETS]]
    for probe in probes:
        probe_id = body_entity_ids[probe["entity_name"]]
        for source_name in source_names:
            world.spawn(
                GravityConstraint(probe_id, body_entity_ids[source_name]),
                name=f"{probe['entity_name']} -> {source_name}",
            )

    return world, body_entity_ids


def make_ephemeris_pre_step(
    start_time_et: float,
    step_seconds: float,
    bodies: ty.Sequence[dict],
    *,
    frame: str = "ECLIPJ2000",
    observer: str = "SUN",
):
    """Create the shared once-per-tick SPICE source refresh callback."""

    def pre_step(tick: int, ctx: el.StepContext) -> None:
        current_time_et = start_time_et + tick * step_seconds
        for body in bodies:
            position_m, velocity_mps = spice_state(
                body["spice_name"],
                current_time_et,
                frame=frame,
                observer=observer,
            )
            ctx.write_component(
                f"{body['entity_name']}.world_pos",
                np.array(
                    [0.0, 0.0, 0.0, 1.0, *position_m],
                    dtype=np.float64,
                ),
            )
            ctx.write_component(
                f"{body['entity_name']}.world_vel",
                np.array(
                    [0.0, 0.0, 0.0, *velocity_mps],
                    dtype=np.float64,
                ),
            )

    return pre_step


def chapter_gravity_system(chapter: int | str):
    chapter = str(chapter)
    if chapter == "1":
        return gravity
    if chapter == "2":
        return heliocentric_gravity
    raise ValueError("chapter must be 1 or 2")
