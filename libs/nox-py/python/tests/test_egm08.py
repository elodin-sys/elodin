"""EGM08 gravity against an independent spherical-harmonic reference.

The reference sums the fully normalized potential directly and differentiates it
numerically, so it shares no code or recursions with the Pines formulation in
`elodin.egm08`.
"""

from math import factorial

import jax
import numpy as np
import pytest
from elodin.egm08 import EGM08

jax.config.update("jax_enable_x64", True)

R_REF = 6.378e6
MU = 3.986004418e14
POSITION = np.array([4.1e6, -3.3e6, 4.4e6])


def normalized_legendre(degree, order, x):
    """Fully normalized associated Legendre function, geodesy convention (no Condon-Shortley)."""
    p_mm = float(np.prod(np.arange(1, 2 * order, 2))) * (1.0 - x * x) ** (order / 2)
    if degree == order:
        p = p_mm
    else:
        p_prev, p = p_mm, x * (2 * order + 1) * p_mm
        for l in range(order + 2, degree + 1):
            p_prev, p = p, ((2 * l - 1) * x * p - (l + order - 1) * p_prev) / (l - order)
    norm = (2 - (order == 0)) * (2 * degree + 1) * factorial(degree - order)
    return p * np.sqrt(norm / factorial(degree + order))


def potential(position, c_bar, s_bar):
    x, y, z = position
    r = np.linalg.norm(position)
    sin_lat, lon = z / r, np.arctan2(y, x)
    total = 0.0
    for l in range(c_bar.shape[0]):
        for m in range(l + 1):
            total += (
                (R_REF / r) ** l
                * normalized_legendre(l, m, sin_lat)
                * (c_bar[l, m] * np.cos(m * lon) + s_bar[l, m] * np.sin(m * lon))
            )
    return MU / r * total


def reference_acceleration(position, c_bar, s_bar, h=2.0):
    acceleration = np.zeros(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        acceleration[i] = (
            potential(position + step, c_bar, s_bar) - potential(position - step, c_bar, s_bar)
        ) / (2 * h)
    return acceleration


def model(tmp_path, max_degree, c_bar, s_bar):
    # Writing the coefficient files into the cache skips the download.
    np.save(tmp_path / "C_normal.npy", c_bar)
    np.save(tmp_path / "S_normal.npy", s_bar)
    return EGM08(max_degree, cache_directory=str(tmp_path))


def single_coefficients():
    for degree in range(5):
        for order in range(degree + 1):
            yield f"C{degree}{order}", degree, order
            if order:
                yield f"S{degree}{order}", degree, order


@pytest.mark.parametrize("name,degree,order", list(single_coefficients()))
def test_each_coefficient_matches_reference(tmp_path, name, degree, order):
    max_degree = 4
    c_bar = np.zeros((max_degree + 1, max_degree + 1))
    s_bar = np.zeros_like(c_bar)
    (c_bar if name[0] == "C" else s_bar)[degree, order] = 1.0

    expected = reference_acceleration(POSITION, c_bar, s_bar)
    actual = np.asarray(model(tmp_path, max_degree, c_bar, s_bar).compute_field(*POSITION, 1.0))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6 * np.linalg.norm(expected))


def test_max_degree_two_includes_j2(tmp_path):
    c_bar = np.zeros((3, 3))
    s_bar = np.zeros_like(c_bar)
    c_bar[0, 0] = 1.0
    c_bar[2, 0] = -4.84165143790815e-4

    actual = np.asarray(model(tmp_path, 2, c_bar, s_bar).compute_field(*POSITION, 1.0))

    x, y, z = POSITION
    r = np.linalg.norm(POSITION)
    j2 = -np.sqrt(5.0) * c_bar[2, 0]
    k = 1.5 * j2 * MU * R_REF**2 / r**5
    expected = -MU * POSITION / r**3 - k * np.array(
        [x * (1 - 5 * z**2 / r**2), y * (1 - 5 * z**2 / r**2), z * (3 - 5 * z**2 / r**2)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
