"""Run the single reconstructed Voyager 1 Jupiter validation case.

This intentionally stays small: one encounter SPK, one initialization epoch,
one hourly RK4 integration, and the fixed checkpoints from ``validation_case``.
It prints Chapter 1 and Chapter 2 residuals against the same SPICE reference;
it does not write campaign/result artifacts into the repository.
"""

import json
from pathlib import Path

import numpy as np
import spiceypy as spice

from validation_case import (
    ENCOUNTER_KERNEL,
    FRAME,
    INITIALIZATION_UTC,
    OBSERVER,
    PROBE,
    checkpoints,
)

G = 6.6743e-11
SUN_MASS = 1.9885e30
STEP_SECONDS = 3600.0
SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"

# Keep the same source names and masses as the Voyager example.
PLANETS = (
    ("MERCURY BARYCENTER", 3.3011e23),
    ("VENUS BARYCENTER", 4.8675e24),
    ("EARTH", 5.97219e24),
    ("MARS BARYCENTER", 6.4171e23),
    ("JUPITER BARYCENTER", 1.898125e27),
    ("SATURN BARYCENTER", 5.6834e26),
    ("URANUS BARYCENTER", 8.6813e25),
    ("NEPTUNE BARYCENTER", 1.02413e26),
)


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


def _acceleration(chapter: int, epoch_et: float, position_m: np.ndarray) -> np.ndarray:
    sun_position = np.zeros(3, dtype=np.float64)
    acceleration = _direct_acceleration(position_m, sun_position, G * SUN_MASS)

    for spice_name, mass in PLANETS:
        state_km, _ = spice.spkezr(spice_name, epoch_et, FRAME, "NONE", OBSERVER)
        source_position_m = np.asarray(state_km[:3], dtype=np.float64) * 1000.0
        mu = G * mass
        if chapter == 1:
            acceleration += _direct_acceleration(position_m, source_position_m, mu)
        elif chapter == 2:
            acceleration += _heliocentric_relative_acceleration(
                position_m, source_position_m, mu
            )
        else:
            raise ValueError("chapter must be 1 or 2")

    return acceleration


def _derivative(chapter: int, epoch_et: float, state: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (state[3:], _acceleration(chapter, epoch_et, state[:3]))
    )


def _rk4_step(chapter: int, epoch_et: float, state: np.ndarray) -> np.ndarray:
    half = STEP_SECONDS / 2.0
    k1 = _derivative(chapter, epoch_et, state)
    k2 = _derivative(chapter, epoch_et + half, state + half * k1)
    k3 = _derivative(chapter, epoch_et + half, state + half * k2)
    k4 = _derivative(chapter, epoch_et + STEP_SECONDS, state + STEP_SECONDS * k3)
    return state + STEP_SECONDS * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _truth_state(epoch_et: float) -> np.ndarray:
    state_km, _ = spice.spkezr(PROBE, epoch_et, FRAME, "NONE", OBSERVER)
    state = np.asarray(state_km, dtype=np.float64)
    return np.concatenate((state[:3] * 1000.0, state[3:] * 1000.0))


def _record(checkpoint: dict[str, int | str], state: np.ndarray, truth: np.ndarray) -> dict:
    position_residual_m = state[:3] - truth[:3]
    velocity_residual_mps = state[3:] - truth[3:]
    return {
        "checkpoint": checkpoint["name"],
        "utc": checkpoint["utc"],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
        "position_error_km": float(np.linalg.norm(position_residual_m) / 1000.0),
        "velocity_error_mps": float(np.linalg.norm(velocity_residual_mps)),
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

    for elapsed in range(int(STEP_SECONDS), final_elapsed + int(STEP_SECONDS), int(STEP_SECONDS)):
        state = _rk4_step(chapter, start_et + elapsed - STEP_SECONDS, state)
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

    spice.kclear()
    for kernel in kernels:
        spice.furnsh(str(kernel))

    try:
        result = {
            "case": "voyager1_jupiter_1979",
            "encounter_kernel": ENCOUNTER_KERNEL,
            "frame": FRAME,
            "observer": OBSERVER,
            "initialization_utc": INITIALIZATION_UTC,
            "step_seconds": STEP_SECONDS,
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
