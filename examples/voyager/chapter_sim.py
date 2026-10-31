from datetime import datetime, timezone
from pathlib import Path

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import spiceypy as spice

SPICE_DIR = Path(__file__).resolve().parent / "nasa_spice_data"
START = "1979-02-22T00:00:00"
END = "1979-02-28T00:00:00"
TIME_STEP = 3600.0
MASS_KG = 825.0

for name in (
    "naif0012.tls",
    "gm_de440.tpc",
    "vgr1_jup230.bsp",
    "de440.bsp",
):
    kernel = SPICE_DIR / name
    if not kernel.is_file():
        raise FileNotFoundError(f"Missing {kernel}; run download_spice_data.sh first")
    spice.furnsh(str(kernel))

start_et = spice.utc2et(START)
end_et = spice.utc2et(END)
steps = round((end_et - start_et) / TIME_STEP)
sun_gm = float(spice.bodvrd("SUN", "GM", 1)[1][0]) * 1.0e9
jupiter_gm = float(spice.bodvrd("JUPITER BARYCENTER", "GM", 1)[1][0]) * 1.0e9
JupiterPosition = el.Annotated[
    jax.Array, el.Component("jupiter_position", el.ComponentType(el.PrimitiveType.F64, (3,)))
]
# Jupiter's position and velocity at the start and end of the current step.
JupiterTrack = el.Annotated[
    jax.Array, el.Component("jupiter_track", el.ComponentType(el.PrimitiveType.F64, (2, 6)))
]


@el.system
def interpolate_jupiter(
    stage_time: el.Query[el.StageTime], q: el.Query[JupiterTrack]
) -> el.Query[JupiterPosition]:
    # Cubic Hermite basis for [p0, v0, p1, v1]; the velocity terms scale by the step length.
    s = stage_time[0] / TIME_STEP
    basis = jnp.stack(
        [
            2 * s**3 - 3 * s**2 + 1,
            (s**3 - 2 * s**2 + s) * TIME_STEP,
            3 * s**2 - 2 * s**3,
            (s**3 - s**2) * TIME_STEP,
        ]
    )
    return q.map(JupiterPosition, lambda track: basis @ track.reshape(4, 3))


def reference_state(et: float, target: str = "VOYAGER 1") -> np.ndarray:
    state, _ = spice.spkezr(target, et, "ECLIPJ2000", "NONE", "SUN")
    # SPICE uses km and km/s, but Elodin uses m and m/s
    return np.asarray(state, dtype=np.float64) * 1000.0


def run(gravity, chapter: int) -> None:
    initial_state = reference_state(start_et)
    world = el.World()
    world.spawn(
        [
            el.Body(
                world_pos=el.WorldPos(linear=jnp.asarray(initial_state[:3])),
                world_vel=el.WorldVel(linear=jnp.asarray(initial_state[3:])),
                inertia=el.Inertia(MASS_KG),
            ),
            el.C(JupiterPosition, reference_state(start_et, "JUPITER BARYCENTER")[:3]),
            el.C(JupiterTrack, np.zeros((2, 6))),
        ],
        name="voyager1",
    )
    position_errors_km = []

    def pre_step(tick: int, ctx: el.StepContext) -> None:
        et = start_et + tick * TIME_STEP
        track = [reference_state(t, "JUPITER BARYCENTER") for t in (et, et + TIME_STEP)]
        ctx.write_component("voyager1.jupiter_track", np.stack(track))

    def post_step(tick: int, ctx: el.StepContext) -> None:
        et = start_et + (tick + 1) * TIME_STEP
        spice_state = reference_state(et)
        sim_position = np.asarray(ctx.read_component("voyager1.world_pos"))[4:7]
        sim_velocity = np.asarray(ctx.read_component("voyager1.world_vel"))[3:6]
        position_error_km = np.linalg.norm(sim_position - spice_state[:3]) / 1000.0
        position_errors_km.append(position_error_km)
        if tick != steps - 1:
            return
        velocity_error_mps = np.linalg.norm(sim_velocity - spice_state[3:])
        print(f"Final position error: {position_error_km:.3f} km")
        print(f"Final velocity error: {velocity_error_mps:.6f} m/s")
        print(f"Mean position error: {np.mean(position_errors_km):.3f} km")
        print(f"Max position error: {max(position_errors_km):.3f} km")

    print(f"Voyager 1 chapter {chapter}: {START} to {END} ({steps} hourly steps)")
    print("Sun and Jupiter gravity only; other planets, SRP and thrust are omitted.")
    start_timestamp = int(
        datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp() * 1_000_000
    )
    world.run(
        el.six_dof(
            sys=interpolate_jupiter | gravity,
            integrator=el.Integrator.Rk78,
        ),
        simulation_rate=1.0 / TIME_STEP,
        max_ticks=steps,
        start_timestamp=start_timestamp,
        pre_step=pre_step,
        post_step=post_step,
        db_path=f"dbs/voyager-chapter{chapter}",
        interactive=False,
    )
