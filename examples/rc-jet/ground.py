"""Spring-damper contact with a fixed terrain surface or the reference field pad."""

from dataclasses import field
import typing as ty

import elodin as el
import jax
import jax.numpy as jnp

from frames import ecef_to_geodetic, ellipsoid_up, geodetic_to_ecef

GEAR_HEIGHT_M = 0.8  # CG height above the surface on the landing gear (class-D estimate)
LIFTOFF_TOLERANCE_M = 0.05
SPRING_N_PER_M = 100_000.0
DAMPING_N_S_PER_M = 10_000.0
FRICTION_COEFFICIENT = 0.05


TerrainClearance = ty.Annotated[
    jax.Array, el.Component("terrain_clearance_m", el.ComponentType.F64)
]
TerrainStatus = ty.Annotated[jax.Array, el.Component("terrain_status", el.ComponentType.F64)]
TerrainPreviousPose = ty.Annotated[
    el.SpatialTransform, el.Component("terrain_previous_pose", el.ComponentType.SpatialPosF64)
]
TerrainChasePose = ty.Annotated[
    el.SpatialTransform, el.Component("terrain_chase_pose", el.ComponentType.SpatialPosF64)
]


@el.dataclass
class TerrainContact(el.Archetype):
    """Status: 0 airborne, 1 contact, 2 coverage edge."""

    terrain_clearance_m: TerrainClearance = field(default_factory=lambda: jnp.asarray(0.0))
    terrain_status: TerrainStatus = field(default_factory=lambda: jnp.asarray(0.0))
    terrain_previous_pose: TerrainPreviousPose = field(default_factory=el.SpatialTransform)
    terrain_chase_pose: TerrainChasePose = field(default_factory=el.SpatialTransform)


@el.map
def remember_terrain_pose(pos: el.WorldPos, _status: TerrainStatus) -> TerrainPreviousPose:
    return pos


def build_terrain_constraint(terrain: el.TerrainHeightmap):
    """Resolve penetration after integration; stop at coverage edges, never invent ground."""

    @el.map
    def constrain(
        pos: el.WorldPos, vel: el.WorldVel, previous: TerrainPreviousPose, status: TerrainStatus
    ) -> tuple[el.WorldPos, el.WorldVel, TerrainClearance, TerrainStatus]:
        lat, lon, altitude = ecef_to_geodetic(pos.linear())
        height, normal, covered = terrain.sample_geodetic(lat, lon)
        stopped = (~covered) | (status == 2.0)
        floor = height + GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M
        contact = altitude <= floor
        resolved = el.SpatialTransform(
            angular=pos.angular(), linear=geodetic_to_ecef(lat, lon, jnp.maximum(altitude, floor))
        )
        # Keep airborne coordinates bit-for-bit; project only a penetrating pose.
        resolved = jax.tree.map(lambda new, old: jnp.where(contact, new, old), resolved, pos)
        resolved = jax.tree.map(lambda new, old: jnp.where(stopped, old, new), resolved, previous)
        inward_speed = jnp.minimum(jnp.dot(vel.linear(), normal), 0.0)
        resolved_linear = vel.linear() - inward_speed * normal
        tangent = resolved_linear - jnp.dot(resolved_linear, normal) * normal
        friction_fraction = jnp.minimum(
            FRICTION_COEFFICIENT * -inward_speed / jnp.maximum(jnp.linalg.norm(tangent), 1e-9), 1.0
        )
        resolved_linear = resolved_linear - friction_fraction * tangent
        linear = jnp.where(contact, resolved_linear, vel.linear())
        velocity = el.SpatialMotion(
            linear=jnp.where(stopped, jnp.zeros(3), linear),
            angular=jnp.where(stopped, jnp.zeros(3), vel.angular()),
        )
        final_lat, final_lon, final_alt = ecef_to_geodetic(resolved.linear())
        final_height, _, _ = terrain.sample_geodetic(final_lat, final_lon)
        return (
            resolved,
            velocity,
            final_alt - final_height,
            jnp.where(
                status == 2.0,
                status,
                jnp.where(~covered, 2.0, jnp.where(contact, 1.0, 0.0)),
            ),
        )

    @el.map
    def chase_camera(pos: el.WorldPos, _status: TerrainStatus) -> TerrainChasePose:
        # The chase camera is outside the airframe: keep it above the same terrain.
        location = pos.linear() + jnp.array([-3.128, -11.885, -10.827])
        lat, lon, altitude = ecef_to_geodetic(location)
        height, _, _ = terrain.sample_geodetic(lat, lon)
        return el.SpatialTransform(
            linear=geodetic_to_ecef(lat, lon, jnp.maximum(altitude, height + 5.0))
        )

    return constrain | chase_camera


def build_ground_contact(field_elevation_m: float, terrain: el.TerrainHeightmap | None = None):
    contact_altitude = field_elevation_m + GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M

    @el.map
    def ground_contact(
        pos: el.WorldPos, vel: el.WorldVel, force: el.Force, inertia: el.Inertia
    ) -> el.Force:
        lat, lon, alt = ecef_to_geodetic(pos.linear())
        up = ellipsoid_up(lat, lon)
        surface = contact_altitude
        normal = up
        if terrain is not None:
            height, normal, _ = terrain.sample_geodetic(lat, lon)
            surface = height + GEAR_HEIGHT_M - LIFTOFF_TOLERANCE_M

        distance = surface - alt
        if terrain is not None:
            distance = distance * jnp.dot(up, normal)
        penetration = jnp.clip(distance, 0.0, 10.0)
        contact = penetration > 0.0

        v = vel.linear()
        v_up = jnp.dot(v, normal)
        normal_mag = SPRING_N_PER_M * penetration - DAMPING_N_S_PER_M * jnp.clip(v_up, -10.0, 10.0)
        normal_mag = jnp.maximum(normal_mag, 0.0)

        v_tangent = v - v_up * normal
        tangent_speed = jnp.linalg.norm(v_tangent)
        friction = jnp.where(
            tangent_speed > 0.01,
            -FRICTION_COEFFICIENT * normal_mag * v_tangent / jnp.maximum(tangent_speed, 0.01),
            jnp.zeros(3),
        )

        ground_force = jnp.where(contact, normal_mag * normal + friction, jnp.zeros(3))
        return force + el.SpatialForce(linear=ground_force)

    return ground_contact
