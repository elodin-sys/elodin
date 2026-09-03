"""Run the focused Voyager 1 Jupiter validation through Elodin's RK4 integrator.

The validation uses the same hourly source-body timing as the Voyager example:
planet states are refreshed from SPICE at the start of each tick, then Elodin's
six-DOF RK4 propagator advances the bodies through that tick. The selected
Feb 22-28 arc excludes the documented impulsive maneuver times in the 1995 JPL
reanalysis. Small attitude-control accelerations from that analysis remain out
of scope.
"""

import hashlib
import json
import tempfile
import typing as ty
from pathlib import Path

import elodin as el
import jax
from jax import numpy as jnp
from jax.numpy import linalg as la
import numpy as np
import spiceypy as spice

from dynamics import heliocentric_relative_acceleration
from gravity_parameters import DE440_GM_M3_S2
from validation_case import (
    ENCOUNTER_KERNEL,
    ENCOUNTER_KERNEL_SHA256,
    FRAME,
    INITIALIZATION_UTC,
    KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC,
    OBSERVER,
    PROBE,
    checkpoints,
)

STEP_SECONDS = 3600.0
SIMULATION_RATE_HZ = 1.0 / STEP_SECONDS
SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
PROBE_ENTITY_NAME = "voyager1"
PROBE_MASS_KG = 825.0
SUN_MASS_KG = 1.9885e30

PLANETS = (
    ("MERCURY BARYCENTER", "mercury", 3.3011e23),
    ("VENUS BARYCENTER", "venus", 4.8675e24),
    ("EARTH", "earth", 5.97219e24),
    ("MARS BARYCENTER", "mars", 6.4171e23),
    ("JUPITER BARYCENTER", "jupiter", 1.898125e27),
    ("SATURN BARYCENTER", "saturn", 5.6834e26),
    ("URANUS BARYCENTER", "uranus", 8.6813e25),
    ("NEPTUNE BARYCENTER", "neptune", 1.02413e26),
)

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
    edge: GravityEdge

    def __init__(self, probe: el.EntityId, source: el.EntityId):
        self.edge = GravityEdge(probe, source)


@el.system
def direct_gravity(
    graph: el.GraphQuery[GravityEdge],
    probe_query: el.Query[el.WorldPos, el.Inertia],
    source_query: el.Query[el.WorldPos, GravitationalParameter],
) -> el.Query[el.Force]:
    """Chapter 1 direct source-body gravity, matching the Voyager example."""

    def gravity_fn(force, probe_pos, probe_inertia, source_pos, source_gm):
        r = probe_pos.linear() - source_pos.linear()
        mass = probe_inertia.mass()
        mu = source_gm[0]
        norm = la.norm(r)
        source_force = mu * mass * r / (norm * norm * norm)
        return el.Force(linear=force.force() - source_force)

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
    """Chapter 2 heliocentric-relative gravity, matching the Voyager example."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth_state(epoch_et: float) -> np.ndarray:
    state_km, _ = spice.spkezr(PROBE, epoch_et, FRAME, "NONE", OBSERVER)
    state = np.asarray(state_km, dtype=np.float64)
    return np.concatenate((state[:3] * 1000.0, state[3:] * 1000.0))


def _body_state(spice_name: str, epoch_et: float) -> tuple[np.ndarray, np.ndarray]:
    state_km, _ = spice.spkezr(spice_name, epoch_et, FRAME, "NONE", OBSERVER)
    state = np.asarray(state_km, dtype=np.float64)
    return state[:3] * 1000.0, state[3:] * 1000.0


def _record(
    checkpoint: dict[str, int | str], state: np.ndarray, truth: np.ndarray
) -> dict:
    position_residual_m = state[:3] - truth[:3]
    velocity_residual_mps = state[3:] - truth[3:]
    return {
        "checkpoint": checkpoint["name"],
        "role": checkpoint["role"],
        "utc": checkpoint["utc"],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
        "position_error_km": float(np.linalg.norm(position_residual_m) / 1000.0),
        "velocity_error_mps": float(np.linalg.norm(velocity_residual_mps)),
        "position_residual_km": [
            float(component / 1000.0) for component in position_residual_m
        ],
        "velocity_residual_mps": [
            float(component) for component in velocity_residual_mps
        ],
    }


def _spawn_source(
    world: el.World,
    *,
    entity_name: str,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    mass_kg: float,
    gm_m3_s2: float,
) -> el.EntityId:
    return world.spawn(
        [
            el.Body(
                world_pos=el.WorldPos(linear=jnp.asarray(position_m)),
                world_vel=el.WorldVel(linear=jnp.asarray(velocity_mps)),
                inertia=el.Inertia(mass_kg),
            ),
            el.C(
                GravitationalParameter,
                jnp.array([gm_m3_s2], dtype=jnp.float64),
            ),
        ],
        name=entity_name,
    )


def run_chapter(chapter: int) -> list[dict]:
    if chapter not in (1, 2):
        raise ValueError("chapter must be 1 or 2")

    checkpoint_list = checkpoints()
    checkpoint_by_elapsed = {
        int(checkpoint["elapsed_seconds"]): checkpoint for checkpoint in checkpoint_list
    }
    final_elapsed = int(checkpoint_list[-1]["elapsed_seconds"])
    if final_elapsed % int(STEP_SECONDS) != 0:
        raise ValueError("final checkpoint must land on the fixed hourly timestep")

    start_et = spice.utc2et(INITIALIZATION_UTC)
    initial_probe_state = _truth_state(start_et)
    records = [_record(checkpoint_list[0], initial_probe_state, initial_probe_state)]

    world = el.World()
    source_ids: dict[str, el.EntityId] = {}

    source_ids["Sun"] = _spawn_source(
        world,
        entity_name="Sun",
        position_m=np.zeros(3, dtype=np.float64),
        velocity_mps=np.zeros(3, dtype=np.float64),
        mass_kg=SUN_MASS_KG,
        gm_m3_s2=DE440_GM_M3_S2["SUN"],
    )

    for spice_name, entity_name, mass_kg in PLANETS:
        position_m, velocity_mps = _body_state(spice_name, start_et)
        source_ids[entity_name] = _spawn_source(
            world,
            entity_name=entity_name,
            position_m=position_m,
            velocity_mps=velocity_mps,
            mass_kg=mass_kg,
            gm_m3_s2=DE440_GM_M3_S2[spice_name],
        )

    probe_id = world.spawn(
        el.Body(
            world_pos=el.WorldPos(linear=jnp.asarray(initial_probe_state[:3])),
            world_vel=el.WorldVel(linear=jnp.asarray(initial_probe_state[3:])),
            inertia=el.Inertia(PROBE_MASS_KG),
        ),
        name=PROBE_ENTITY_NAME,
    )

    for source_name, source_id in source_ids.items():
        world.spawn(
            GravityConstraint(probe_id, source_id),
            name=f"{PROBE_ENTITY_NAME} -> {source_name}",
        )

    def pre_step(tick: int, ctx: el.StepContext) -> None:
        current_time_et = start_et + tick * STEP_SECONDS
        for spice_name, entity_name, _ in PLANETS:
            position_m, velocity_mps = _body_state(spice_name, current_time_et)
            ctx.write_component(
                f"{entity_name}.world_pos",
                np.array(
                    [0.0, 0.0, 0.0, 1.0, *position_m],
                    dtype=np.float64,
                ),
            )
            ctx.write_component(
                f"{entity_name}.world_vel",
                np.array(
                    [0.0, 0.0, 0.0, *velocity_mps],
                    dtype=np.float64,
                ),
            )

    def post_step(tick: int, ctx: el.StepContext) -> None:
        elapsed = int((tick + 1) * STEP_SECONDS)
        checkpoint = checkpoint_by_elapsed.get(elapsed)
        if checkpoint is None:
            return

        simulated_pos = np.asarray(
            ctx.read_component(f"{PROBE_ENTITY_NAME}.world_pos"),
            dtype=np.float64,
        )[4:7]
        simulated_vel = np.asarray(
            ctx.read_component(f"{PROBE_ENTITY_NAME}.world_vel"),
            dtype=np.float64,
        )[3:6]
        simulated_state = np.concatenate((simulated_pos, simulated_vel))
        truth = _truth_state(start_et + elapsed)
        records.append(_record(checkpoint, simulated_state, truth))

    gravity_system = direct_gravity if chapter == 1 else heliocentric_gravity
    system = el.six_dof(
        sys=gravity_system,
        integrator=el.Integrator.Rk4,
    )

    with tempfile.TemporaryDirectory(prefix=f"voyager-ch{chapter}-") as db_path:
        world.run(
            system,
            simulation_rate=SIMULATION_RATE_HZ,
            pre_step=pre_step,
            post_step=post_step,
            max_ticks=final_elapsed // int(STEP_SECONDS),
            db_path=db_path,
            interactive=False,
        )

    return records


def main() -> None:
    kernels = (
        SPICE_DIR / "naif0012.tls",
        SPICE_DIR / ENCOUNTER_KERNEL,
        # Load DE440 last so its planetary segments take precedence while the
        # encounter SPK remains the source for Voyager 1.
        SPICE_DIR / "de440.bsp",
    )
    missing = [str(kernel) for kernel in kernels if not kernel.exists()]
    if missing:
        raise FileNotFoundError(
            "missing SPICE kernels; run examples/voyager/download_spice_data.sh first: "
            + ", ".join(missing)
        )

    actual_hash = _sha256(SPICE_DIR / ENCOUNTER_KERNEL)
    if actual_hash != ENCOUNTER_KERNEL_SHA256:
        raise ValueError(
            f"unexpected {ENCOUNTER_KERNEL} SHA-256: {actual_hash}; "
            f"expected {ENCOUNTER_KERNEL_SHA256}"
        )

    spice.kclear()
    for kernel in kernels:
        spice.furnsh(str(kernel))

    try:
        result = {
            "case": "voyager1_jupiter_1979",
            "encounter_kernel": ENCOUNTER_KERNEL,
            "encounter_kernel_sha256": actual_hash,
            "frame": FRAME,
            "observer": OBSERVER,
            "initialization_utc": INITIALIZATION_UTC,
            "step_seconds": STEP_SECONDS,
            "integrator": "Elodin six_dof / Integrator.Rk4",
            "source_sampling": "SPICE once per tick + Elodin RK4 body propagation",
            "gravity_parameters": "DE440 GM values in m^3/s^2",
            "documented_impulsive_maneuver_events_utc": (
                KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC
            ),
            "chapters": {
                "1": run_chapter(1),
                "2": run_chapter(2),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        spice.kclear()


if __name__ == "__main__":
    main()
