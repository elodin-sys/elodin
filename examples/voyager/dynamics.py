"""Small, testable pieces of the Voyager gravity models."""

from collections.abc import Iterable, Mapping

import jax.numpy as jnp


def direct_planetary_acceleration(
    probe_position,
    source_position,
    gravitational_parameter,
):
    """Acceleration of a probe toward one gravity source, in m/s^2."""
    source_to_probe = source_position - probe_position
    distance = jnp.linalg.norm(source_to_probe)
    return gravitational_parameter * source_to_probe / distance**3


def planetary_acceleration_of_sun(source_position, gravitational_parameter):
    """Acceleration of the heliocentric origin toward one planet, in m/s^2."""
    source_distance = jnp.linalg.norm(source_position)
    safe_distance = jnp.where(source_distance > 0.0, source_distance, 1.0)
    return gravitational_parameter * source_position / safe_distance**3


def heliocentric_relative_acceleration(
    probe_position,
    source_position,
    gravitational_parameter,
):
    """Probe acceleration relative to the accelerating Sun, in m/s^2."""
    direct_acceleration = direct_planetary_acceleration(
        probe_position, source_position, gravitational_parameter
    )
    sun_acceleration = planetary_acceleration_of_sun(source_position, gravitational_parameter)
    return direct_acceleration - sun_acceleration


def gravity_source_entity_names(planets: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    """Return the Sun and planet entities that may source probe gravity."""
    return ("Sun", *(str(planet["entity_name"]) for planet in planets))
