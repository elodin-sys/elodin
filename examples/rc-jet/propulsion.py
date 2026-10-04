"""BDX propulsion: package map interpolation, spool state, fuel, thrust line.

Steady thrust and fuel flow come from trilinear interpolation of the
package's `propulsion_map.csv` over (geodetic altitude, Mach, effective
throttle). The map is class-D (analytic lapse/TSFC, no identified engine
deck) — see the package provenance. Elodin owns the dynamics around it:

- stick 0..1 maps to effective throttle `min_throttle`..1 (turbines idle at
  min_throttle; this replaces the old dead `idle_spool` field),
- a first-order spool state lags the commanded effective throttle before the
  map lookup (spool tau is a class-D estimate),
- fuel integrates map fuel flow and drives total mass; an empty tank is a
  flameout (thrust and flow drop to zero, fuel never goes negative),
- thrust applies at the package thrust line (+0.044 m above the CG), which
  adds the nose-down pitch moment the pre-campaign example omitted.

Queries outside the map's grid hull are clamped to the hull (the map's own
axes bound the class-D model's stated domain).
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

from aero import Mach
from actuators import ControlCommands
from bdx_model import BdxModel
from class_d_fallbacks import ClassDFallbacks
from frames import geodetic_altitude

SpoolSpeed = ty.Annotated[
    jax.Array,
    el.Component("spool_speed", el.ComponentType.F64, metadata={"priority": 60}),
]
ThrottleCommand = ty.Annotated[
    jax.Array,
    el.Component("throttle_command", el.ComponentType.F64, metadata={"priority": 61}),
]
Thrust = ty.Annotated[
    jax.Array,
    el.Component("thrust", el.ComponentType.F64, metadata={"priority": 59}),
]
FuelMass = ty.Annotated[
    jax.Array,
    el.Component("fuel_mass", el.ComponentType.F64, metadata={"priority": 58}),
]
FuelFlow = ty.Annotated[
    jax.Array,
    el.Component("fuel_flow", el.ComponentType.F64, metadata={"priority": 57}),
]


def _axis_fraction(axis: jnp.ndarray, value):
    index = jnp.clip(jnp.searchsorted(axis, value, side="right") - 1, 0, axis.size - 2)
    x0 = axis[index]
    x1 = axis[index + 1]
    fraction = jnp.clip((value - x0) / (x1 - x0), 0.0, 1.0)
    return index, fraction


def trilinear(axes, table, point):
    """Trilinear interpolation on a regular grid, clamped to the hull.

    axes: three sorted 1-D arrays; table: array shaped by the axes;
    point: three scalars. jnp-traceable.
    """
    i, fi = _axis_fraction(axes[0], point[0])
    j, fj = _axis_fraction(axes[1], point[1])
    k, fk = _axis_fraction(axes[2], point[2])
    result = 0.0
    for di, wi in ((0, 1.0 - fi), (1, fi)):
        for dj, wj in ((0, 1.0 - fj), (1, fj)):
            for dk, wk in ((0, 1.0 - fk), (1, fk)):
                result = result + wi * wj * wk * table[i + di, j + dj, k + dk]
    return result


def effective_throttle(model: BdxModel, stick):
    """Commanded effective throttle: the stick value floored at engine idle.

    Keeping stick == effective throttle above idle makes the telemetry read
    directly against the package trim rows and anchors (cruise 0.2125).
    """
    return jnp.maximum(jnp.clip(stick, 0.0, 1.0), model.propulsion.min_throttle)


@el.map
def extract_throttle_command(commands: ControlCommands) -> ThrottleCommand:
    return commands[3]


def build_spool_dynamics(model: BdxModel, fallbacks: ClassDFallbacks, dt: float):
    tau = fallbacks.propulsion.spool_tau_s

    @el.map
    def spool_dynamics(throttle_cmd: ThrottleCommand, spool: SpoolSpeed) -> SpoolSpeed:
        target = effective_throttle(model, throttle_cmd)
        return jnp.clip(spool + (target - spool) * dt / tau, 0.0, 1.0)

    return spool_dynamics


def build_thrust_and_fuel_flow(model: BdxModel):
    grid = model.propulsion_map
    axes = (
        jnp.asarray(grid.altitudes_m),
        jnp.asarray(grid.machs),
        jnp.asarray(grid.throttles),
    )
    thrust_table = jnp.asarray(grid.thrust_n)
    fuel_table = jnp.asarray(grid.fuel_flow_kg_s)

    @el.map
    def compute_thrust(
        pos: el.WorldPos, mach: Mach, spool: SpoolSpeed, fuel: FuelMass
    ) -> tuple[Thrust, FuelFlow]:
        altitude = geodetic_altitude(pos.linear())
        point = (altitude, mach, spool)
        running = jnp.where(fuel > 0.0, 1.0, 0.0)  # empty tank = flameout
        thrust = running * trilinear(axes, thrust_table, point)
        flow = running * trilinear(axes, fuel_table, point)
        return thrust, flow

    return compute_thrust


def build_fuel_integration(model: BdxModel, dt: float):
    capacity = model.mass.fuel_capacity_kg

    @el.map
    def integrate_fuel(fuel: FuelMass, flow: FuelFlow) -> FuelMass:
        return jnp.clip(fuel - flow * dt, 0.0, capacity)

    return integrate_fuel


def build_mass_update(model: BdxModel, fallbacks: ClassDFallbacks):
    empty_mass = model.mass.operating_empty_mass_kg
    inertia_diag = jnp.array(
        [
            fallbacks.inertia.ixx_kg_m2,
            fallbacks.inertia.iyy_kg_m2,
            fallbacks.inertia.izz_kg_m2,
        ]
    )

    @el.map
    def update_mass(fuel: FuelMass, _inertia: el.Inertia) -> el.Inertia:
        """Total mass tracks fuel burn; CG stays at the package point and the
        class-D inertia diagonal is held constant (no mass-distribution model)."""
        return el.SpatialInertia(mass=empty_mass + fuel, inertia=inertia_diag)

    return update_mass


def build_apply_thrust(model: BdxModel):
    axis = jnp.array(model.propulsion.thrust_axis_body)
    offset = jnp.array(model.propulsion.thrust_application_body_m)

    @el.map
    def apply_thrust(thrust: Thrust, pos: el.WorldPos, force: el.Force) -> el.Force:
        force_body = axis * thrust
        torque_body = jnp.cross(offset, force_body)
        wrench = el.SpatialForce(linear=force_body, torque=torque_body)
        return force + pos.angular() @ wrench

    return apply_thrust
