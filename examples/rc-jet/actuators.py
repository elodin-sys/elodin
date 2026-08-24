"""Control-surface servo dynamics (rate-limited first order).

Deflection/rate limits are class-D fallbacks: Elite Aerosports publishes
linear throws in millimeters, not hinge angles (handoff §13), and the
package carries no control geometry yet.
"""

import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

from class_d_fallbacks import ClassDFallbacks

ControlSurfaces = ty.Annotated[
    jax.Array,
    el.Component(
        "control_surfaces",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "elevator,aileron,rudder", "priority": 70},
    ),
]

ControlCommands = ty.Annotated[
    jax.Array,
    el.Component(
        "control_commands",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "element_names": "elevator,aileron,rudder,throttle",
            "priority": 71,
            "external_control": "true",
        },
    ),
]


def servo_step(delta, delta_cmd, tau, rate_limit, deflection_limit, dt):
    """One rate-limited first-order servo update (whitepaper §5.1)."""
    rate = jnp.clip((delta_cmd - delta) / tau, -rate_limit, rate_limit)
    return jnp.clip(delta + rate * dt, -deflection_limit, deflection_limit)


def build_actuator_dynamics(fallbacks: ClassDFallbacks, dt: float):
    servos = fallbacks.actuators
    tau = servos.servo_tau_s
    max_surface = jnp.deg2rad(servos.max_deflection_deg)
    max_rudder = jnp.deg2rad(servos.max_rudder_deflection_deg)
    max_rate = jnp.deg2rad(servos.max_rate_deg_s)
    max_rudder_rate = jnp.deg2rad(servos.max_rudder_rate_deg_s)

    @el.map
    def actuator_dynamics(commands: ControlCommands, surfaces: ControlSurfaces) -> ControlSurfaces:
        delta_e, delta_a, delta_r = surfaces
        delta_e_cmd, delta_a_cmd, delta_r_cmd, _throttle = commands
        return jnp.array(
            [
                servo_step(delta_e, delta_e_cmd, tau, max_rate, max_surface, dt),
                servo_step(delta_a, delta_a_cmd, tau, max_rate, max_surface, dt),
                servo_step(delta_r, delta_r_cmd, tau, max_rudder_rate, max_rudder, dt),
            ]
        )

    return actuator_dynamics
