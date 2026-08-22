"""Focused contract tests for the single Voyager 1 Jupiter validation case."""

import numpy as np

from dynamics import heliocentric_relative_acceleration
from validate_jupiter import _heliocentric_relative_acceleration
from validation_case import (
    CHECKPOINT_UTCS,
    ENCOUNTER_KERNEL,
    ENCOUNTER_KERNEL_SHA256,
    INITIALIZATION_UTC,
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
    assert [record["elapsed_seconds"] for record in records] == [
        0,
        3 * 86400,
        7 * 86400,
        11 * 86400,
        12 * 86400,
        13 * 86400,
    ]
    assert all(
        coverage_start <= parse_utc(str(record["utc"])) <= coverage_end
        for record in records
    )


def test_validation_chapter_two_term_matches_shared_dynamics_helper():
    probe = np.array([7.5e11, -2.0e11, 1.0e10], dtype=np.float64)
    source = np.array([7.0e11, -1.0e11, 2.0e10], dtype=np.float64)
    mu = 1.26686534e17

    expected = np.asarray(
        heliocentric_relative_acceleration(probe, source, mu), dtype=np.float64
    )
    actual = _heliocentric_relative_acceleration(probe, source, mu)

    # JAX may evaluate the shared helper in float32 depending on repo settings.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
