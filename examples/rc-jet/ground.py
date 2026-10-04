"""Ground contact against the local ellipsoid-height field surface.

Spring-damper normal force along the geodetic up at the vehicle's own
position, with Coulomb-style friction in the local tangent plane. The field
surface is the scenario site's elevation — a flat analytic pad, not the
terrain heightfield (visual terrain and contact physics are decoupled, as in
the pre-campaign example).
"""

import elodin as el
import jax.numpy as jnp

from frames import ecef_to_geodetic, ellipsoid_up

GEAR_HEIGHT_M = 0.8  # CG height above the surface on the landing gear (class-D estimate)
LIFTOFF_TOLERANCE_M = 0.05
SPRING_N_PER_M = 100_000.0
DAMPING_N_S_PER_M = 10_000.0
FRICTION_COEFFICIENT = 0.05


def build_ground_contact(field_elevation_m: float):
    contact_altitude = field_elevation_m + GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M

    @el.map
    def ground_contact(
        pos: el.WorldPos, vel: el.WorldVel, force: el.Force, inertia: el.Inertia
    ) -> el.Force:
        lat, lon, alt = ecef_to_geodetic(pos.linear())
        up = ellipsoid_up(lat, lon)

        penetration = jnp.clip(contact_altitude - alt, 0.0, 10.0)
        contact = penetration > 0.0

        v = vel.linear()
        v_up = jnp.dot(v, up)
        normal_mag = SPRING_N_PER_M * penetration - DAMPING_N_S_PER_M * jnp.clip(v_up, -10.0, 10.0)
        normal_mag = jnp.maximum(normal_mag, 0.0)

        v_tangent = v - v_up * up
        tangent_speed = jnp.linalg.norm(v_tangent)
        friction = jnp.where(
            tangent_speed > 0.01,
            -FRICTION_COEFFICIENT * normal_mag * v_tangent / jnp.maximum(tangent_speed, 0.01),
            jnp.zeros(3),
        )

        ground_force = jnp.where(contact, normal_mag * up + friction, jnp.zeros(3))
        return force + el.SpatialForce(linear=ground_force)

    return ground_contact
