"""Compare Voyager Chapter 1 and Chapter 2 against one reconstructed Jupiter arc.

This is intentionally a small validation case for issue #794: one Voyager 1
encounter kernel, one initialization epoch, and three fixed checkpoints. It
uses the same rounded masses, one-hour timestep, and Sun-centered force models
as the Voyager example. Planet states are refreshed from DE440 once per step,
then move through the RK4 stages using their SPICE velocities, matching how
six_dof integrates the ephemeris bodies between pre-step updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spiceypy as spice

G = 6.6743e-11
DT_S = 3600.0
START_UTC = "1979-02-05T00:00:00"
CHECKPOINTS_S = (0, 5 * 86400, 10 * 86400)

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
KERNELS = (
    SPICE_DIR / "naif0012.tls",
    SPICE_DIR / "vgr1_jup230.bsp",
    SPICE_DIR / "de440.bsp",
)

SUN_MASS = 1.9885e30
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


@dataclass(frozen=True)
class ErrorSample:
    elapsed_days: float
    chapter1_position_error_km: float
    chapter1_velocity_error_mps: float
    chapter2_position_error_km: float
    chapter2_velocity_error_mps: float


def point_mass_acceleration(position_m: np.ndarray, source_m: np.ndarray, mu: float) -> np.ndarray:
    delta = source_m - position_m
    distance = np.linalg.norm(delta)
    return mu * delta / distance**3


def acceleration(
    chapter: int,
    position_m: np.ndarray,
    source_positions_m: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Return the Chapter 1 or Chapter 2 Sun-centered acceleration."""
    total = point_mass_acceleration(position_m, np.zeros(3), G * SUN_MASS)

    for (_, mass), source_m in zip(PLANETS, source_positions_m, strict=True):
        mu = G * mass
        total += point_mass_acceleration(position_m, source_m, mu)
        if chapter == 2:
            source_distance = np.linalg.norm(source_m)
            total -= mu * source_m / source_distance**3

    return total


def rk4_step(
    chapter: int,
    state: np.ndarray,
    source_states: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Advance one step with source bodies drifting at their pre-step velocities."""

    def source_positions(offset_s: float) -> tuple[np.ndarray, ...]:
        return tuple(position_m + offset_s * velocity_mps for position_m, velocity_mps in source_states)

    def derivative(candidate: np.ndarray, offset_s: float) -> np.ndarray:
        position_m = candidate[:3]
        velocity_mps = candidate[3:]
        return np.concatenate(
            (velocity_mps, acceleration(chapter, position_m, source_positions(offset_s)))
        )

    k1 = derivative(state, 0.0)
    k2 = derivative(state + 0.5 * DT_S * k1, 0.5 * DT_S)
    k3 = derivative(state + 0.5 * DT_S * k2, 0.5 * DT_S)
    k4 = derivative(state + DT_S * k3, DT_S)
    return state + (DT_S / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def planet_states(epoch_et: float) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    states = []
    for name, _ in PLANETS:
        state_km, _ = spice.spkezr(name, epoch_et, "ECLIPJ2000", "NONE", "SUN")
        states.append(
            (
                np.asarray(state_km[:3]) * 1000.0,
                np.asarray(state_km[3:]) * 1000.0,
            )
        )
    return tuple(states)


def voyager_truth(epoch_et: float) -> np.ndarray:
    state_km, _ = spice.spkezr("VOYAGER 1", epoch_et, "ECLIPJ2000", "NONE", "SUN")
    return np.concatenate(
        (np.asarray(state_km[:3]) * 1000.0, np.asarray(state_km[3:]) * 1000.0)
    )


def error_against_truth(state: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    position_error_km = np.linalg.norm(state[:3] - truth[:3]) / 1000.0
    velocity_error_mps = np.linalg.norm(state[3:] - truth[3:])
    return position_error_km, velocity_error_mps


def run() -> list[ErrorSample]:
    missing = [str(kernel) for kernel in KERNELS if not kernel.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Voyager validation kernels. Run download_spice_data.sh first:\n"
            + "\n".join(missing)
        )

    spice.kclear()
    try:
        # The encounter kernel supplies Voyager 1. DE440 is loaded last so its
        # planetary segments take precedence if kernels overlap.
        for kernel in KERNELS:
            spice.furnsh(str(kernel))

        start_et = spice.utc2et(START_UTC)
        initial_state = voyager_truth(start_et)
        states = {1: initial_state.copy(), 2: initial_state.copy()}
        samples: list[ErrorSample] = []

        max_elapsed = max(CHECKPOINTS_S)
        tick_count = int(max_elapsed / DT_S)
        checkpoints = set(CHECKPOINTS_S)

        for tick in range(tick_count + 1):
            elapsed_s = tick * DT_S
            epoch_et = start_et + elapsed_s

            if elapsed_s in checkpoints:
                truth = voyager_truth(epoch_et)
                c1_pos, c1_vel = error_against_truth(states[1], truth)
                c2_pos, c2_vel = error_against_truth(states[2], truth)
                samples.append(
                    ErrorSample(
                        elapsed_days=elapsed_s / 86400.0,
                        chapter1_position_error_km=c1_pos,
                        chapter1_velocity_error_mps=c1_vel,
                        chapter2_position_error_km=c2_pos,
                        chapter2_velocity_error_mps=c2_vel,
                    )
                )

            if tick == tick_count:
                break

            sources = planet_states(epoch_et)
            states[1] = rk4_step(1, states[1], sources)
            states[2] = rk4_step(2, states[2], sources)

        return samples
    finally:
        spice.kclear()


def main() -> None:
    samples = run()
    print(
        "days  ch1_pos_km  ch1_vel_mps  ch2_pos_km  ch2_vel_mps  "
        "ch2_pos_reduction_pct"
    )
    for sample in samples:
        if sample.chapter1_position_error_km == 0.0:
            reduction = 0.0
        else:
            reduction = 100.0 * (
                sample.chapter1_position_error_km - sample.chapter2_position_error_km
            ) / sample.chapter1_position_error_km
        print(
            f"{sample.elapsed_days:4.0f}  "
            f"{sample.chapter1_position_error_km:10.3f}  "
            f"{sample.chapter1_velocity_error_mps:11.6f}  "
            f"{sample.chapter2_position_error_km:10.3f}  "
            f"{sample.chapter2_velocity_error_mps:11.6f}  "
            f"{reduction:21.3f}"
        )


if __name__ == "__main__":
    main()
