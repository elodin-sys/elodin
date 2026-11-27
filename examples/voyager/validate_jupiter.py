"""Compare Jupiter gravity models with Elodin RK4 on both Voyager flybys."""

import argparse
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import spiceypy as spice

DAY = 86400
BODIES = (1, 2, 3, 4, 6, 7, 8, 5, 599, 501, 502, 503, 504)
JUPITER_BARYCENTER = BODIES.index(5)
JUPITER_CENTER = BODIES.index(599)
ENCOUNTERS = (
    (-31, "vgr1_jup230.bsp", "1979-03-05T12:04:46.794"),
    (-32, "vgr2_jup230.bsp", "1979-07-09T22:28:52.806"),
)
Sources = el.Annotated[
    jax.Array, el.Component("sources", el.ComponentType(el.PrimitiveType.F64, (len(BODIES), 3)))
]
Pole = el.Annotated[jax.Array, el.Component("pole", el.ComponentType(el.PrimitiveType.F64, (3,)))]
GravityGM = el.Annotated[
    jax.Array, el.Component("gravity_gm", el.ComponentType(el.PrimitiveType.F64, (len(BODIES),)))
]
JupiterJ2 = el.Annotated[
    jax.Array, el.Component("jupiter_j2", el.ComponentType(el.PrimitiveType.F64, ()))
]


def state(body, et):
    return 1000 * np.asarray(spice.spkez(body, et, "ECLIPJ2000", "NONE", 10)[0])


def gm(body):
    return 1e9 * spice.bodvcd(body, "GM", 1)[1][0]


def compare(probe, closest, days_before, max_step, port):
    steps_per_day = math.ceil(DAY / max_step)
    dt = DAY / steps_per_day
    start = closest - days_before * DAY
    scoring_ticks = {days_before * steps_per_day - 1: closest}
    scoring_ticks[(days_before + 3) * steps_per_day - 1] = closest + 3 * DAY
    total_gm = np.asarray([gm(body) for body in BODIES])
    point_gm = total_gm.copy()
    point_gm[JUPITER_CENTER:] = 0
    moon_gm = total_gm.copy()
    moon_gm[JUPITER_BARYCENTER] = 0
    models = (
        ("point", "point masses", point_gm, 0.0),
        ("moons", "Jupiter + moons", moon_gm, 0.0),
        ("shape", "Jupiter + moons + J2", moon_gm, 0.014696514),
    )
    sun_gm = gm(10)
    jupiter_gm = total_gm[JUPITER_CENTER]

    @el.map
    def gravity(
        pos: el.WorldPos, sources: Sources, pole: Pole, strengths: GravityGM, j2: JupiterJ2
    ) -> el.Force:
        position = pos.linear()
        delta = sources - position
        direct = delta / jnp.linalg.norm(delta, axis=1)[:, None] ** 3
        indirect = sources / jnp.linalg.norm(sources, axis=1)[:, None] ** 3
        acceleration = -sun_gm * position / jnp.linalg.norm(position) ** 3
        acceleration += strengths @ (direct - indirect)

        def oblateness(relative):
            distance = jnp.linalg.norm(relative)
            latitude_sine = relative @ pole / distance
            scale = 1.5 * jupiter_gm * j2 * (71_492_000.0 / distance) ** 2 / distance**3
            return scale * (
                (5 * latitude_sine**2 - 1) * relative - 2 * latitude_sine * distance * pole
            )

        jupiter = sources[JUPITER_CENTER]
        acceleration += oblateness(position - jupiter) - oblateness(-jupiter)
        return el.SpatialForce(linear=acceleration)

    world = el.World()
    initial = state(probe, start)
    for entity, _, strengths, j2 in models:
        world.spawn(
            [
                el.Body(
                    world_pos=el.WorldPos(linear=jnp.asarray(initial[:3])),
                    world_vel=el.WorldVel(linear=jnp.asarray(initial[3:])),
                    inertia=el.Inertia(1.0),
                ),
                el.C(Sources, np.zeros((len(BODIES), 3))),
                el.C(Pole, np.zeros(3)),
                el.C(GravityGM, strengths),
                el.C(JupiterJ2, np.asarray(j2)),
            ],
            name=entity,
        )

    def pre_step(tick, ctx):
        # Use the midpoint ephemeris for each RK4 step.
        et = start + (tick + 0.5) * dt
        sources = np.asarray([state(body, et)[:3] for body in BODIES])
        pole = spice.pxform("IAU_JUPITER", "ECLIPJ2000", et)[:, 2].copy()
        for entity, _, _, _ in models:
            ctx.write_component(f"{entity}.sources", sources)
            ctx.write_component(f"{entity}.pole", pole)

    def post_step(tick, ctx):
        if tick not in scoring_ticks:
            return
        reference = state(probe, scoring_ticks[tick])
        checkpoint = "closest approach" if scoring_ticks[tick] == closest else "+3 days"
        print(f"  {checkpoint}", flush=True)
        for entity, label, _, _ in models:
            position = np.asarray(ctx.read_component(f"{entity}.world_pos"))[4:7]
            velocity = np.asarray(ctx.read_component(f"{entity}.world_vel"))[3:6]
            error_km = np.linalg.norm(position - reference[:3]) / 1000
            error_mps = np.linalg.norm(velocity - reference[3:])
            print(f"    {label:22s} {error_km:9.2f} km / {error_mps:7.3f} m/s", flush=True)

    with TemporaryDirectory(prefix="voyager-jupiter-") as db:
        # Each World.run starts its own server with two adjacent companion ports.
        sys.argv[1:] = ["run", f"127.0.0.1:{port}"]
        world.run(
            el.six_dof(sys=gravity, integrator=el.Integrator.Rk4),
            simulation_rate=1 / dt,
            max_ticks=max(scoring_ticks) + 1,
            interactive=False,
            pre_step=pre_step,
            post_step=post_step,
            db_path=str(Path(db) / "db"),
            log_level="warn",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-step", type=float, default=30, help="maximum RK4 step in seconds")
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

    port = 2240
    for probe, _, closest_utc in ENCOUNTERS:
        closest = spice.utc2et(closest_utc)
        for days_before in (8, 3):
            print(
                f"Voyager {-probe - 30}, start {days_before} days before closest approach",
                flush=True,
            )
            compare(probe, closest, days_before, args.max_step, port)
            port += 3


if __name__ == "__main__":
    main()
