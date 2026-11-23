"""Compare Jupiter gravity models against both Voyager encounter kernels."""

import argparse
import os
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

DAY = 86400
PLANETS = (1, 2, 3, 4, 5, 6, 7, 8)
JUPITER_AND_MOONS = (599, 501, 502, 503, 504)
ENCOUNTERS = (
    (-31, "vgr1_jup230.bsp", "1979-03-05T12:04:46.794"),
    (-32, "vgr2_jup230.bsp", "1979-07-09T22:28:52.806"),
)


def state(body, et):
    return np.asarray(spice.spkez(body, et, "ECLIPJ2000", "NONE", 10)[0])


def propagate(probe, start, epochs, bodies, use_j2, max_step):
    gm_sun = spice.bodvcd(10, "GM", 1)[1][0]
    gm = np.array([spice.bodvcd(body, "GM", 1)[1][0] for body in bodies])
    if use_j2:
        gm_jupiter = gm[bodies.index(599)]

    def j2_acceleration(relative, pole):
        distance = np.linalg.norm(relative)
        latitude_sine = relative @ pole / distance
        scale = 1.5 * gm_jupiter * 0.014696514 * 71492.0**2 / distance**5
        return scale * ((5 * latitude_sine**2 - 1) * relative - 2 * latitude_sine * distance * pole)

    def rhs(et, y):
        position = y[:3]
        sources = np.array([state(body, et)[:3] for body in bodies])
        relative = sources - position
        direct = relative / np.linalg.norm(relative, axis=1)[:, None] ** 3
        indirect = sources / np.linalg.norm(sources, axis=1)[:, None] ** 3
        acceleration = -gm_sun * position / np.linalg.norm(position) ** 3 + gm @ (direct - indirect)
        if use_j2:
            jupiter = sources[bodies.index(599)]
            pole = spice.pxform("IAU_JUPITER", "ECLIPJ2000", et)[:, 2]
            acceleration += j2_acceleration(position - jupiter, pole) - j2_acceleration(
                -jupiter, pole
            )
        return np.r_[y[3:], acceleration]

    solution = solve_ivp(
        rhs,
        (start, epochs[-1]),
        state(probe, start),
        t_eval=epochs,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    residuals = solution.y.T - np.array([state(probe, et) for et in epochs])
    return np.linalg.norm(residuals[:, :3], axis=1), 1000 * np.linalg.norm(residuals[:, 3:], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-step", type=float, default=300, help="maximum integration step in seconds"
    )
    args = parser.parse_args()
    if not np.isfinite(args.max_step) or args.max_step <= 0:
        parser.error("--max-step must be a positive number")

    default_dir = Path(__file__).resolve().parent / "nasa_spice_data"
    kernel_dir = Path(os.environ.get("VOYAGER_SPICE_DIR", default_dir))
    kernels = ("naif0012.tls", "de440.bsp", "gm_de440.tpc", "pck00011.tpc", "jup365.bsp")
    for name in (*kernels, *(case[1] for case in ENCOUNTERS)):
        kernel = kernel_dir / name
        if not kernel.is_file():
            parser.error(f"missing {kernel}; run download_spice_data.sh --validation first")
        spice.furnsh(str(kernel))

    resolved_jupiter = tuple(body for body in PLANETS if body != 5) + JUPITER_AND_MOONS
    models = (
        ("point masses", PLANETS, False),
        ("Jupiter + moons", resolved_jupiter, False),
        ("Jupiter + moons + J2", resolved_jupiter, True),
    )
    for probe, _, closest_utc in ENCOUNTERS:
        closest = spice.utc2et(closest_utc)
        epochs = (closest, closest + 3 * DAY)
        for days_before in (8, 3):
            print(
                f"Voyager {-probe - 30}, start {days_before} days before closest approach",
                flush=True,
            )
            for label, bodies, use_j2 in models:
                position, velocity = propagate(
                    probe, closest - days_before * DAY, epochs, bodies, use_j2, args.max_step
                )
                print(
                    f"  {label:22s} closest: {position[0]:9.2f} km / {velocity[0]:7.3f} m/s"
                    f"   +3 days: {position[1]:9.2f} km / {velocity[1]:7.3f} m/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
