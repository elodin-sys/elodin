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


def reference_state(et: float, target: str = "VOYAGER 1") -> np.ndarray:
    state, _ = spice.spkezr(target, et, "ECLIPJ2000", "NONE", "SUN")
    return np.asarray(state, dtype=np.float64) * 1000.0


initial_state = reference_state(start_et)
reference_state(end_et)
world = el.World()
world.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.asarray(initial_state[:3])),
            world_vel=el.WorldVel(linear=jnp.asarray(initial_state[3:])),
            inertia=el.Inertia(MASS_KG),
        ),
        el.C(JupiterPosition, reference_state(start_et, "JUPITER BARYCENTER")[:3]),
    ],
    name="voyager1",
)
position_errors_km = []


def pre_step(tick: int, ctx: el.StepContext) -> None:
    et = start_et + (tick + 0.5) * TIME_STEP
    ctx.write_component("voyager1.jupiter_position", reference_state(et, "JUPITER BARYCENTER")[:3])


@el.map
def gravity(pos: el.WorldPos, inertia: el.Inertia, jupiter: JupiterPosition) -> el.Force:
    r = pos.linear()
    acceleration = -sun_gm * r / jnp.linalg.norm(r) ** 3
    to_jupiter = jupiter - r
    acceleration += jupiter_gm * (
        to_jupiter / jnp.linalg.norm(to_jupiter) ** 3 - jupiter / jnp.linalg.norm(jupiter) ** 3
    )
    return el.SpatialForce(linear=inertia.mass() * acceleration)


def post_step(tick: int, ctx: el.StepContext) -> None:
    et = start_et + (tick + 1) * TIME_STEP
    truth = reference_state(et)
    position = np.asarray(ctx.read_component("voyager1.world_pos"))[4:7]
    velocity = np.asarray(ctx.read_component("voyager1.world_vel"))[3:6]
    position_error_km = np.linalg.norm(position - truth[:3]) / 1000.0
    position_errors_km.append(position_error_km)
    if tick != steps - 1:
        return
    velocity_error_mps = np.linalg.norm(velocity - truth[3:])
    print(f"Final position error: {position_error_km:.3f} km")
    print(f"Final velocity error: {velocity_error_mps:.6f} m/s")
    print(f"Mean position error: {np.mean(position_errors_km):.3f} km")
    print(f"Max position error: {max(position_errors_km):.3f} km")


print(f"Voyager 1 Sun/Jupiter baseline: {START} to {END} ({steps} hourly steps)")
print("Sun and Jupiter gravity only; other planets, SRP and thrust are omitted.")
start_timestamp = int(
    datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp() * 1_000_000
)
world.run(
    el.six_dof(sys=gravity, integrator=el.Integrator.Rk4),
    simulation_rate=1.0 / TIME_STEP,
    max_ticks=steps,
    start_timestamp=start_timestamp,
    pre_step=pre_step,
    post_step=post_step,
    db_path="dbs/voyager-chapter1",
    interactive=False,
)
