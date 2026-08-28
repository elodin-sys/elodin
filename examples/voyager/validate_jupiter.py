"""Run the single reconstructed Voyager 1 Jupiter diagnostic case.

The propagation matches the Voyager example's source-body timing: each planet is
refreshed from SPICE once at the start of an hourly tick, then its sampled
velocity carries it through the RK4 substeps. The selected Feb 22-28 arc excludes
the documented impulsive maneuver times in the 1995 JPL reanalysis. Small
attitude-control accelerations from that analysis are still not modeled here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import spiceypy as spice

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
SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"

# Use the same DE440 gravitational parameters as the Voyager example. Barycenter
# sources use the full system GM, which matters during the Jupiter encounter.
PLANET_NAMES = (
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
)
PLANETS = tuple((name, DE440_GM_M3_S2[name]) for name in PLANET_NAMES)
SUN_GM = DE440_GM_M3_S2["SUN"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_acceleration(
    probe_position_m: np.ndarray,
    source_position_m: np.ndarray,
    mu: float,
) -> np.ndarray:
    delta = source_position_m - probe_position_m
    distance = np.linalg.norm(delta)
    return mu * delta / distance**3


def _heliocentric_relative_acceleration(
    probe_position_m: np.ndarray,
    source_position_m: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Chapter 2 direct pull minus the source's acceleration of the Sun."""
    direct = _direct_acceleration(probe_position_m, source_position_m, mu)
    source_distance = np.linalg.norm(source_position_m)
    if source_distance == 0.0:
        return direct
    return direct - mu * source_position_m / source_distance**3


def _planet_states(epoch_et: float) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    states = []
    for spice_name, _ in PLANETS:
        state_km, _ = spice.spkezr(spice_name, epoch_et, FRAME, "NONE", OBSERVER)
        states.append(
            (
                np.asarray(state_km[:3], dtype=np.float64) * 1000.0,
                np.asarray(state_km[3:], dtype=np.float64) * 1000.0,
            )
        )
    return tuple(states)


def _acceleration(
    chapter: int,
    position_m: np.ndarray,
    source_positions_m: tuple[np.ndarray, ...],
) -> np.ndarray:
    acceleration = _direct_acceleration(
        position_m, np.zeros(3, dtype=np.float64), SUN_GM
    )

    for (_, mu), source_position_m in zip(PLANETS, source_positions_m, strict=True):
        if chapter == 1:
            acceleration += _direct_acceleration(position_m, source_position_m, mu)
        elif chapter == 2:
            acceleration += _heliocentric_relative_acceleration(
                position_m, source_position_m, mu
            )
        else:
            raise ValueError("chapter must be 1 or 2")

    return acceleration


def _rk4_step(
    chapter: int,
    state: np.ndarray,
    source_states: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Advance one tick using the same source-body timing as the Elodin example."""

    def source_positions(offset_seconds: float) -> tuple[np.ndarray, ...]:
        return tuple(
            position_m + offset_seconds * velocity_mps
            for position_m, velocity_mps in source_states
        )

    def derivative(candidate: np.ndarray, offset_seconds: float) -> np.ndarray:
        return np.concatenate(
            (
                candidate[3:],
                _acceleration(
                    chapter, candidate[:3], source_positions(offset_seconds)
                ),
            )
        )

    half = STEP_SECONDS / 2.0
    k1 = derivative(state, 0.0)
    k2 = derivative(state + half * k1, half)
    k3 = derivative(state + half * k2, half)
    k4 = derivative(state + STEP_SECONDS * k3, STEP_SECONDS)
    return state + STEP_SECONDS * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


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
    state = _truth_state(start_et)
    records = [_record(checkpoint_list[0], state, state)]

    for elapsed in range(
        int(STEP_SECONDS), final_elapsed + int(STEP_SECONDS), int(STEP_SECONDS)
    ):
        step_start_et = start_et + elapsed - STEP_SECONDS
        state = _rk4_step(chapter, state, _planet_states(step_start_et))
        checkpoint = checkpoint_by_elapsed.get(elapsed)
        if checkpoint is not None:
            truth = _truth_state(start_et + elapsed)
            records.append(_record(checkpoint, state, truth))

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
            "source_sampling": "SPICE once per tick + linear source drift through RK4 stages",
            "gravity_parameters": "DE440 GM values in m^3/s^2",
            "documented_impulsive_maneuver_events_utc": KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC,
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
