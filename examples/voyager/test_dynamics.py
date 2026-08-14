"""Focused checks for the Voyager Chapter 2 force equation."""

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dynamics import (
    direct_planetary_acceleration,
    gravity_source_entity_names,
    heliocentric_relative_acceleration,
    planetary_acceleration_of_sun,
)


def test_direct_planetary_acceleration_points_toward_source():
    probe_position = jnp.array([2.0, 0.0, 0.0])
    source_position = jnp.array([5.0, 0.0, 0.0])

    acceleration = direct_planetary_acceleration(
        probe_position,
        source_position,
        gravitational_parameter=18.0,
    )

    np.testing.assert_allclose(acceleration, [2.0, 0.0, 0.0])


def test_sun_acceleration_is_subtracted_from_direct_acceleration():
    probe_position = jnp.array([5.0, 0.0, 0.0])
    source_position = jnp.array([3.0, 0.0, 0.0])
    gravitational_parameter = 9.0

    direct_acceleration = direct_planetary_acceleration(
        probe_position, source_position, gravitational_parameter
    )
    sun_acceleration = planetary_acceleration_of_sun(source_position, gravitational_parameter)
    relative_acceleration = heliocentric_relative_acceleration(
        probe_position, source_position, gravitational_parameter
    )

    np.testing.assert_allclose(direct_acceleration, [-2.25, 0.0, 0.0])
    np.testing.assert_allclose(sun_acceleration, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(relative_acceleration, [-3.25, 0.0, 0.0])


def test_relative_acceleration_accumulates_over_multiple_sources():
    probe_position = jnp.array([4.0, 0.0, 0.0])
    sources = (
        (jnp.array([2.0, 0.0, 0.0]), 8.0),
        (jnp.array([-2.0, 0.0, 0.0]), 18.0),
    )

    relative_acceleration = sum(
        (
            heliocentric_relative_acceleration(probe_position, source_position, mu)
            for source_position, mu in sources
        ),
        start=jnp.zeros(3),
    )

    # The two independently hand-computed contributions are -4 and +4 m/s^2.
    np.testing.assert_allclose(relative_acceleration, [0.0, 0.0, 0.0])


def test_source_at_origin_has_no_indirect_acceleration():
    probe_position = jnp.array([2.0, 0.0, 0.0])
    source_position = jnp.zeros(3)

    direct_acceleration = direct_planetary_acceleration(
        probe_position, source_position, gravitational_parameter=16.0
    )
    sun_acceleration = planetary_acceleration_of_sun(source_position, gravitational_parameter=16.0)

    relative_acceleration = heliocentric_relative_acceleration(
        probe_position, source_position, gravitational_parameter=16.0
    )

    np.testing.assert_allclose(sun_acceleration, jnp.zeros(3))
    np.testing.assert_allclose(relative_acceleration, direct_acceleration)
    np.testing.assert_allclose(relative_acceleration, [-4.0, 0.0, 0.0])


def test_truth_probes_are_not_wired_into_chapter_2_gravity():
    planets = (
        {"entity_name": "earth"},
        {"entity_name": "jupiter"},
    )
    assert gravity_source_entity_names(planets) == ("Sun", "earth", "jupiter")

    main_tree = ast.parse(Path(__file__).with_name("main.py").read_text())
    gravity_source_calls = [
        node
        for node in ast.walk(main_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "gravity_source_entity_names"
    ]
    assert len(gravity_source_calls) == 1
    assert ast.unparse(gravity_source_calls[0].args[0]) == "PLANETS"

    relative_acceleration_calls = [
        node
        for node in ast.walk(main_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "heliocentric_relative_acceleration"
    ]
    assert len(relative_acceleration_calls) == 1
