"""Focused regression test for the Voyager 1 Jupiter validation path."""

import numpy as np

from dynamics import heliocentric_relative_acceleration
from gravity_parameters import DE440_GM_M3_S2
from validate_jupiter import _heliocentric_relative_acceleration


def test_validation_chapter_two_term_matches_shared_dynamics_helper():
    """Catch drift between the standalone validator and Chapter 2 dynamics."""
    probe = np.array([7.5e11, -2.0e11, 1.0e10], dtype=np.float64)
    source = np.array([7.0e11, -1.0e11, 2.0e10], dtype=np.float64)
    mu = DE440_GM_M3_S2["JUPITER BARYCENTER"]

    expected = np.asarray(
        heliocentric_relative_acceleration(probe, source, mu), dtype=np.float64
    )
    actual = _heliocentric_relative_acceleration(probe, source, mu)

    # JAX may evaluate the shared helper in float32 depending on repo settings.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
