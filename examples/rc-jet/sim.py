"""
BDX Simulation System Composition

Integrates all subsystems into a complete 6-DOF simulation.
Following Section 8 of BDX_Simulation_Whitepaper.md.
"""

from dataclasses import field

import elodin as el
import jax.numpy as jnp

# Import all component types from the proper aero module
from aero import (
    AngleOfAttack,
    Sideslip,
    DynamicPressure,
    Mach,
    AeroCoefs,
    AeroForce,
    Wind,
    VelocityBody,
    compute_velocity_body,
    compute_aero_angles,
    dynamic_pressure_and_mach,
    compute_aero_coefs,
    aero_forces,
    apply_aero_forces,
)
from propulsion import (
    SpoolSpeed,
    ThrottleCommand,
    Thrust,
    extract_throttle_command,
    spool_dynamics,
    compute_thrust,
    apply_thrust,
)
from actuators import (
    ControlSurfaces,
    ControlCommands,
    actuator_dynamics,
)
from ground import ground_contact


@el.dataclass
class BDXJet(el.Archetype):
    """
    Complete BDX jet aircraft archetype combining all subsystems.

    Components:
    - Aerodynamics: alpha, beta, aero_coefs, aero_force, dynamic_pressure, mach, velocity_body
    - Propulsion: spool_speed, throttle_command, thrust
    - Actuators: control_surfaces, control_commands
    - Environment: wind
    """

    # Aerodynamic states
    velocity_body: VelocityBody = field(default_factory=lambda: jnp.array([70.0, 0.0, 0.0]))
    alpha: AngleOfAttack = field(default_factory=lambda: jnp.float64(0.0))
    beta: Sideslip = field(default_factory=lambda: jnp.float64(0.0))  # Added missing beta
    dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(1.0))
    mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
    aero_coefs: AeroCoefs = field(default_factory=lambda: jnp.zeros(6))  # 6 coefficients now
    aero_force: AeroForce = field(default_factory=el.SpatialForce)

    # Propulsion states
    spool_speed: SpoolSpeed = field(default_factory=lambda: jnp.float64(0.3))
    throttle_command: ThrottleCommand = field(default_factory=lambda: jnp.float64(0.3))
    thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))

    # Actuator states
    control_surfaces: ControlSurfaces = field(default_factory=lambda: jnp.zeros(3))
    control_commands: ControlCommands = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.3])
    )

    # Environment
    wind: Wind = field(default_factory=lambda: jnp.zeros(3))


@el.map
def gravity(inertia: el.Inertia, force: el.Force) -> el.Force:
    """Apply gravitational force (Z-up world frame)."""
    return force + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


def system() -> el.System:
    """
    Compose the complete BDX simulation system.

    System graph (Section 8.4 of whitepaper):
    1. Non-effectors: Compute derived quantities (angles, pressures, coefficients)
    2. Effectors: Apply forces (gravity, thrust, aerodynamics)
    3. Integration: 6-DOF rigid body dynamics
    """

    # Non-effector systems (compute derived quantities in dependency order)
    non_effectors = (
        extract_throttle_command  # Extract throttle from control commands
        | compute_velocity_body  # Transform world vel to body frame
        | compute_aero_angles  # Calculate α, β
        | dynamic_pressure_and_mach  # Calculate q̄, M
        | actuator_dynamics  # Update control surface positions
        | spool_dynamics  # Update engine spool speed
        | compute_aero_coefs  # Calculate aerodynamic coefficients
        | aero_forces  # Convert coefficients to forces/moments
        | compute_thrust  # Calculate thrust from spool speed
    )

    # Effector systems (apply forces to rigid body)
    effectors = (
        gravity  # Apply gravitational force
        | apply_thrust  # Apply propulsion force
        | apply_aero_forces  # Apply aerodynamic forces
        | ground_contact  # Apply ground contact forces
    )

    # Compose with 6-DOF integrator
    return non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)
