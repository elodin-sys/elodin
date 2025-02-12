import jax.numpy as np
from jax.numpy import linalg as la


class J2:
    def __init__(self):
        self.r_ref = 6.378e6  # Earth's equatorial radius in meters
        self.mu_earth = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2
        self.J2 = 1.08262668e-3

    def compute_field(self, x, y, z, mass):
        r = np.array([z, y, z])
        m = mass
        norm = la.norm(r)
        e_r = r / norm
        f = -self.mu_earth * m * r / (norm * norm * norm)
        e_z = np.array([0.0, 0.0, 1.0])
        j2 = (
            -self.mu_earth
            * m
            * self.J2
            * self.r_ref**2
            * (
                3 * z / (norm**5) * e_z
                + (3.0 / (2.0 * norm**4) - 15.0 * z**2 / (2.0 * norm**6.0)) * e_r
            )
        )
        field = f + j2
        return np.array(field)
