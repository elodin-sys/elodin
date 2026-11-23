"""Compare Voyager force models against the corresponding SPICE trajectory."""

import argparse
import json

import spiceypy as spice
from fast_sim import (
    GALILEAN_MOONS,
    MANIFEST_FILE,
    SATURN_MOONS,
    V1_JUP,
    V1_SAT,
    V2_JUP,
    V2_SAT,
    ModelConfig,
    init_spice,
    run_simulation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=["v1_jupiter", "v2_jupiter", "v1_saturn", "v2_saturn", "v1_cruise"]
    )
    args = parser.parse_args()
    moons = [moon["name"] for moon in GALILEAN_MOONS]
    if args.case == "v1_cruise":
        init_spice()
        for label, maneuvers in (
            ("gravity only", []),
            (
                "with estimated April correction",
                [
                    {
                        "epoch_et": spice.utc2et("1979-04-10T00:00:00"),
                        "delta_v": [-63.525, -13.349, 0.363],
                    }
                ],
            ),
        ):
            config = ModelConfig(
                giant_planet_state="center",
                jupiter_moons=moons,
                saturn_moons=["titan", "rhea", "iapetus", "dione", "tethys"],
                maneuvers=maneuvers,
            )
            result = run_simulation("VOYAGER 1", "1979-04-01T00:00:00", 590, 3600, config)
            print(f"{label}: {result['trajectory_pos_err_km'][-1]:,.1f} km at day 590", flush=True)
        return

    first_probe = args.case.startswith("v1")
    probe = "VOYAGER 1" if first_probe else "VOYAGER 2"
    case = next(
        case for case in json.loads(MANIFEST_FILE.read_text())["cases"] if case["id"] == args.case
    )
    if args.case.endswith("saturn"):
        init_spice(V1_SAT if first_probe else V2_SAT)
        closest = next(
            point for point in case["checkpoints"] if point["name"] == "closest_approach"
        )
        start = spice.et2utc(spice.utc2et(closest["epoch_utc"]) - 3 * 86400, "ISOC", 6)
        for label, config in (
            ("point masses", ModelConfig()),
            (
                "Saturn center, moons and shape",
                ModelConfig(
                    giant_planet_state="center",
                    saturn_moons=[moon["name"] for moon in SATURN_MOONS],
                    saturn_j2=True,
                    saturn_j4=True,
                ),
            ),
        ):
            result = run_simulation(probe, start, 3, 60, config)
            print(
                f"{label}: {result['trajectory_pos_err_km'][-1]:,.2f} km / {result['trajectory_vel_err_mps'][-1]:.3f} m/s near closest approach",
                flush=True,
            )
        return

    init_spice(V1_JUP if first_probe else V2_JUP)
    for label, config in (
        ("point masses", ModelConfig()),
        (
            "Jupiter center and moons",
            ModelConfig(gm_source="naif_de440", giant_planet_state="center", jupiter_moons=moons),
        ),
    ):
        result = run_simulation(
            probe,
            case["initialization_utc"],
            case["duration_days"],
            300,
            config,
            case["checkpoints"],
        )
        closest = next(
            point for point in result["checkpoints"] if point["name"] == "closest_approach"
        )
        print(
            f"{label}: {closest['pos_err_km']:,.2f} km / {closest['vel_err_mps']:.3f} m/s near closest approach",
            flush=True,
        )


if __name__ == "__main__":
    main()
