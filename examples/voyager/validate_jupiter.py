"""Run the focused Voyager 1 Jupiter validation through the shared Voyager model.

The selected Feb 22-28 arc excludes the documented impulsive maneuver times in
the 1995 JPL reanalysis. Small attitude-control accelerations from that analysis
remain out of scope.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import elodin as el
import numpy as np
import spiceypy as spice

from simulation import (
    PLANETS,
    build_world,
    chapter_gravity_system,
    make_ephemeris_pre_step,
)
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
VALIDATION_PROBE = {
    "spice_name": PROBE,
    "entity_name": PROBE_ENTITY_NAME,
    "mass": PROBE_MASS_KG,
}


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


def run_chapter(chapter: int) -> list[dict]:
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

    world, _ = build_world(
        start_et,
        probes=(VALIDATION_PROBE,),
        frame=FRAME,
        observer=OBSERVER,
    )
    pre_step = make_ephemeris_pre_step(
        start_et,
        STEP_SECONDS,
        PLANETS,
        frame=FRAME,
        observer=OBSERVER,
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

    system = el.six_dof(
        sys=chapter_gravity_system(chapter),
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
            "source_sampling": "shared Voyager SPICE refresh + Elodin RK4",
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
