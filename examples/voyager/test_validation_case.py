"""Focused contract tests for the single Voyager 1 Jupiter validation case."""

import numpy as np

from dynamics import heliocentric_relative_acceleration
from gravity_parameters import DE440_GM_M3_S2
from validate_jupiter import PLANETS, SUN_GM, _heliocentric_relative_acceleration
from validation_case import (
    CHECKPOINT_ROLES,
    CHECKPOINT_UTCS,
    ENCOUNTER_KERNEL,
    ENCOUNTER_KERNEL_SHA256,
    INITIALIZATION_UTC,
    KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC,
    PDS_COVERAGE_UTC,
    checkpoints,
    parse_utc,
)


def test_v1_jupiter_case_is_fixed_and_inside_encounter_coverage():
    records = checkpoints()
    coverage_start, coverage_end = map(parse_utc, PDS_COVERAGE_UTC)

    assert ENCOUNTER_KERNEL == "vgr1_jup230.bsp"
    assert len(ENCOUNTER_KERNEL_SHA256) == 64
    int(ENCOUNTER_KERNEL_SHA256, 16)
    assert CHECKPOINT_UTCS[0] == INITIALIZATION_UTC
    assert len(CHECKPOINT_ROLES) == len(CHECKPOINT_UTCS)
    assert [record["elapsed_seconds"] for record in records] == [
        0,
        2 * 86400,
        4 * 86400,
        6 * 86400,
    ]
    assert all(
        coverage_start <= parse_utc(str(record["utc"])) <= coverage_end
        for record in records
    )


def test_selected_arc_excludes_documented_impulsive_maneuvers():
    start = parse_utc(INITIALIZATION_UTC)
    end = parse_utc(CHECKPOINT_UTCS[-1])
    events = tuple(map(parse_utc, KNOWN_IMPULSIVE_MANEUVER_EVENTS_UTC))

    assert not any(start < event <= end for event in events)
    assert max(event for event in events if event <= start) == parse_utc(
        "1979-02-21T03:58:00Z"
    )
    assert min(event for event in events if event > end) == parse_utc(
        "1979-03-01T23:00:00Z"
    )


def test_validation_uses_de440_system_gravity_parameters():
    planet_gms = dict(PLANETS)

    assert SUN_GM == DE440_GM_M3_S2["SUN"]
    assert planet_gms["JUPITER BARYCENTER"] == 1.2671276409999998e17
    assert planet_gms["JUPITER BARYCENTER"] == DE440_GM_M3_S2["JUPITER BARYCENTER"]


def test_validation_chapter_two_term_matches_shared_dynamics_helper():
    probe = np.array([7.5e11, -2.0e11, 1.0e10], dtype=np.float64)
    source = np.array([7.0e11, -1.0e11, 2.0e10], dtype=np.float64)
    mu = DE440_GM_M3_S2["JUPITER BARYCENTER"]

    expected = np.asarray(
        heliocentric_relative_acceleration(probe, source, mu), dtype=np.float64
    )
    actual = _heliocentric_relative_acceleration(probe, source, mu)

    # JAX may evaluate the shared helper in float32 depending on repo settings.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
