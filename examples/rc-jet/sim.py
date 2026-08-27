"""BDX system composition: package-driven plant in a rotating ECEF world."""

from dataclasses import field

import elodin as el
import jax.numpy as jnp

from actuators import ControlCommands, ControlSurfaces, build_actuator_dynamics
from aero import (
    AeroCoefs,
    AeroForce,
    AeroValid,
    AngleOfAttack,
    DynamicPressure,
    Mach,
    Sideslip,
    VelocityBody,
    Wind,
    apply_aero_forces,
    build_aero_coefs,
    build_aero_forces,
    build_aero_validity,
    compute_aero_angles,
    compute_velocity_body,
    dynamic_pressure_and_mach,
)
from bdx_model import BdxModel
from class_d_fallbacks import ClassDFallbacks
from frames import frame_accel, gravity_accel
from ground import build_ground_contact
from propulsion import (
    FuelFlow,
    FuelMass,
    SpoolSpeed,
    Thrust,
    ThrottleCommand,
    build_apply_thrust,
    build_fuel_integration,
    build_mass_update,
    build_spool_dynamics,
    build_thrust_and_fuel_flow,
    extract_throttle_command,
)
from scenario import Numerics, Scenario
from telemetry import Geodetic, PosENU, build_geodetic_telemetry


@el.dataclass
class BDXJet(el.Archetype):
    """BDX jet subsystem states (rigid-body states live in el.Body)."""

    velocity_body: VelocityBody = field(default_factory=lambda: jnp.zeros(3))
    alpha: AngleOfAttack = field(default_factory=lambda: jnp.float64(0.0))
    beta: Sideslip = field(default_factory=lambda: jnp.float64(0.0))
    dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(0.0))
    mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
    aero_coefs: AeroCoefs = field(default_factory=lambda: jnp.zeros(6))
    aero_force: AeroForce = field(default_factory=el.SpatialForce)
    aero_valid: AeroValid = field(default_factory=lambda: jnp.float64(1.0))

    spool_speed: SpoolSpeed = field(default_factory=lambda: jnp.float64(0.0))
    throttle_command: ThrottleCommand = field(default_factory=lambda: jnp.float64(0.0))
    thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))
    fuel_mass: FuelMass = field(default_factory=lambda: jnp.float64(0.0))
    fuel_flow: FuelFlow = field(default_factory=lambda: jnp.float64(0.0))

    control_surfaces: ControlSurfaces = field(default_factory=lambda: jnp.zeros(3))
    control_commands: ControlCommands = field(default_factory=lambda: jnp.zeros(4))

    wind: Wind = field(default_factory=lambda: jnp.zeros(3))
    geodetic: Geodetic = field(default_factory=lambda: jnp.zeros(3))
    pos_enu: PosENU = field(default_factory=lambda: jnp.zeros(3))


def make_jet(scenario: Scenario) -> BDXJet:
    init = scenario.initial
    tas = scenario.tas_mps
    alpha = init.alpha_rad
    return BDXJet(
        velocity_body=jnp.array([tas * jnp.cos(alpha), 0.0, -tas * jnp.sin(alpha)]),
        alpha=jnp.float64(alpha),
        spool_speed=jnp.float64(init.throttle),
        throttle_command=jnp.float64(init.throttle),
        fuel_mass=jnp.float64(init.fuel_kg),
        control_surfaces=jnp.array([init.elevator_rad, 0.0, 0.0]),
        control_commands=jnp.array([init.elevator_rad, 0.0, 0.0, init.throttle]),
        wind=jnp.asarray(init.wind_ecef),
    )


@el.map
def gravity_and_frame_forces(
    force: el.Force, inertia: el.Inertia, pos: el.WorldPos, vel: el.WorldVel
) -> el.Force:
    """Point-mass gravitation plus Coriolis/centrifugal of the rotating ECEF frame."""
    accel = gravity_accel(pos.linear()) + frame_accel(pos.linear(), vel.linear())
    return force + el.SpatialForce(linear=accel * inertia.mass())


def build_system(
    model: BdxModel, fallbacks: ClassDFallbacks, scenario: Scenario, numerics: Numerics
) -> el.System:
    dt = numerics.dt
    site = scenario.site

    non_effectors = (
        extract_throttle_command
        | compute_velocity_body
        | compute_aero_angles
        | dynamic_pressure_and_mach
        | build_actuator_dynamics(fallbacks, dt)
        | build_spool_dynamics(model, fallbacks, dt)
        | build_aero_coefs(model, fallbacks)
        | build_aero_validity(model)
        | build_aero_forces(model)
        | build_thrust_and_fuel_flow(model)
        | build_fuel_integration(model, dt)
        | build_mass_update(model, fallbacks)
        | build_geodetic_telemetry(site.lat_deg, site.lon_deg, site.field_elevation_m)
    )
    effectors = (
        gravity_and_frame_forces
        | build_apply_thrust(model)
        | apply_aero_forces
        | build_ground_contact(site.field_elevation_m)
    )
    return non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)
