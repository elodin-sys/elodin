# ignore ambiguous variable names since they are based on math literature
# ruff: noqa: E741
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)


def k_delta(degree):
    return jnp.where(degree == 0, 1.0, 2.0)


class EGM08:
    def __init__(
        self,
        max_degree,
        c_bar_path="",
        s_bar_path="",
    ):
        self.r_ref = 6.378e6  # Earth's equatorial radius in meters
        self.mu_earth = 3.986004418e14  # Earth's gravitational parameter in m^3/s^2

        self.max_degree = max_degree

        self.a_0_0 = 1.0
        self.r_0 = 1.0
        self.i_0 = 0.0

        self.c_bar = jnp.load(c_bar_path)[: max_degree + 1, : max_degree + 1].astype(jnp.float64)
        self.s_bar = jnp.load(s_bar_path)[: max_degree + 1, : max_degree + 1].astype(jnp.float64)

        self.l = jnp.arange(max_degree + 1)
        self.m = jnp.arange(max_degree + 1)

        self.nq1 = jnp.zeros((max_degree + 1, max_degree + 1))
        self.nq2 = jnp.zeros((max_degree + 1, max_degree + 1))
        self.i_r_m = jnp.zeros((max_degree + 1, 2))
        self.rho_l = jnp.zeros(max_degree + 1)

        self.a_1 = 0.0
        self.a_2 = 0.0
        self.a_3 = 0.0
        self.a_4 = 0.0

        self.initialize_parameters()

    def compute_a_bar_diagonal(self, a_0_0, l):
        numerator = (2 * l + 1) * k_delta(l)
        denominator = (2 * l) * k_delta(l - 1)
        sqrt_term = jnp.sqrt(numerator / denominator)
        a_current = a_0_0 * sqrt_term
        return jnp.where(l == 0, a_0_0, a_current), jnp.where(l == 0, a_0_0, a_current)

    def compute_a_bar_off_diagonal(self, unused, l):
        numerator = (2 * l) * k_delta(l - 1)
        denominator = k_delta(l)
        sqrt_term = jnp.sqrt(numerator / denominator)
        a_current = self.a_bar[l, l] * sqrt_term * self.u
        return jnp.where(l == 0, 0.0, a_current), jnp.where(l == 0, 0.0, a_current)

    def compute_n1(self, l, m):
        numerator = (2 * l + 1) * (2 * l - 1)
        denominator = (l + m) * (l - m)
        sqrt_term = jnp.sqrt(numerator / denominator)
        return jnp.where(l >= m + 2, sqrt_term, 0.0)

    def compute_n2(self, l, m):
        numerator = (l + m - 1) * (l - m - 1) * (2 * l + 1)
        denominator = (2 * l - 3) * (l + m) * (l - m)
        sqrt_term = jnp.sqrt(numerator / denominator)
        return jnp.where(l >= m + 2, sqrt_term, 0.0)

    def compute_i_r_m(self, i_r_inits, m):  # i_r_inits = [i_m1, r_m1]
        i_m = jnp.where(m == 0, i_r_inits[0], self.s * i_r_inits[0] + self.t * i_r_inits[1])
        r_m = jnp.where(m == 0, i_r_inits[1], self.s * i_r_inits[1] - self.t * i_r_inits[0])
        i_r_inits = (i_m, r_m)
        return i_r_inits, i_r_inits

    def compute_rho_l(self, l):
        rho_l = (self.mu_earth / self.r) * ((self.r_ref / self.r) ** l)
        return rho_l

    def compute_a_bar_full_m(self, m):
        def compute_a_full_l(a_inits, l):  # a_inits = [a_l-1_m, a_l-2_m]
            val = self.u * self.compute_n1(l, m) * a_inits[0] - self.compute_n2(l, m) * a_inits[1]
            a_l_m = jnp.where(l >= m + 2, val, self.a_bar[l, m])
            a_inits = (a_l_m, a_inits[0])
            return a_inits, a_l_m

        _, a_l = jax.lax.scan(compute_a_full_l, (self.a_bar[0, 0], self.a_bar[1, 0]), self.l)
        return a_l

    def compute_nq1(self, m):
        def compute_nq1_l(unused, l):
            numerator = (l - m) * (k_delta(m)) * (l + m + 1)
            denominator = k_delta(m + 1)
            nq1 = jnp.where(numerator < 0, 0.0, jnp.sqrt(numerator / denominator))
            return 0, nq1

        _, nq1 = jax.lax.scan(compute_nq1_l, 0, self.l)
        return nq1

    def compute_nq2(self, m):
        def compute_nq2_l(unused, l):
            numerator = (l + m + 2) * (l + m + 1) * (2 * l + 1) * k_delta(m)
            denominator = (2 * l + 3) * k_delta(m + 1)
            nq2 = jnp.where(numerator < 0, 0.0, jnp.sqrt(numerator / denominator))
            return 0, nq2

        _, nq2 = jax.lax.scan(compute_nq2_l, 0, self.l)
        return nq2

    def compute_components(self):
        rho_l_1 = jnp.roll(self.rho_l, -1).at[self.rho_l.shape[0] - 1].set(0.0)
        r_m_1 = jnp.roll(self.i_r_m[1], 1).at[0].set(0.0)
        i_m_1 = jnp.roll(self.i_r_m[0], 1).at[0].set(0.0)
        e = self.c_bar * r_m_1 + self.s_bar * i_m_1
        m = jnp.roll(self.m, -1).at[self.m.shape[0] - 1].set(0.0)
        a_1 = ((rho_l_1 / self.r_ref) * self.a_bar.T).T * m * e
        self.a_1 = jnp.sum(jnp.sum(a_1, axis=1), axis=0)

        f = self.s_bar * r_m_1 - self.c_bar * i_m_1
        a_2 = ((rho_l_1 / self.r_ref) * self.a_bar.T).T * m * f
        self.a_2 = jnp.sum(jnp.sum(a_2, axis=1), axis=0)

        d = self.c_bar * self.i_r_m[1] + self.s_bar * self.i_r_m[0]
        a_bar1 = (
            jnp.roll(self.a_bar, -1, axis=1)
            .at[:, self.a_bar.shape[1] - 1]
            .set(jnp.zeros(self.a_bar.shape[0]))
        )
        a_3 = ((rho_l_1 / self.r_ref) * a_bar1.T).T * m * self.nq1 * d
        self.a_3 = jnp.sum(jnp.sum(a_3, axis=1), axis=0)

        a_bar2 = (
            jnp.roll(
                jnp.roll(self.a_bar, -1, axis=1)
                .at[:, self.a_bar.shape[1] - 1]
                .set(jnp.zeros(self.a_bar.shape[0])),
                -1,
                axis=0,
            )
            .at[self.a_bar.shape[0] - 1, :]
            .set(jnp.zeros(self.a_bar.shape[1]))
        )
        a_4 = ((rho_l_1 / self.r_ref) * a_bar2.T).T * m * self.nq2 * d * (-1)
        self.a_4 = jnp.sum(jnp.sum(a_4, axis=1), axis=0)

    def initialize_parameters(self):
        _, a_bar_diag = jax.lax.scan(self.compute_a_bar_diagonal, self.a_0_0, self.l)
        self.a_bar = jnp.diag(a_bar_diag)

    def pre_compute_parameters(self):
        _, a_bar_off_diag = jax.lax.scan(self.compute_a_bar_off_diagonal, 0, self.l)
        a_bar_off_diag = jnp.diag(a_bar_off_diag)
        self.a_bar = jax.numpy.roll(a_bar_off_diag, -1, axis=1) + self.a_bar

    def compute_field(self, x, y, z, mass):
        self.r = jnp.sqrt(x**2 + y**2 + z**2)
        self.s = x / self.r
        self.t = y / self.r
        self.u = z / self.r

        self.pre_compute_parameters()
        self.a_bar = jax.vmap(self.compute_a_bar_full_m, in_axes=(0))(self.m)
        self.a_bar = self.a_bar.T
        _, self.i_r_m = jax.lax.scan(self.compute_i_r_m, (self.i_0, self.r_0), self.m)
        self.rho_l = jax.vmap(self.compute_rho_l, in_axes=(0))(self.l)
        self.nq1 = (jax.vmap(self.compute_nq1, in_axes=(0))(self.m)).T
        self.nq2 = (jax.vmap(self.compute_nq2, in_axes=(0))(self.m)).T
        self.compute_components()
        return mass * jnp.array(
            [
                self.a_1 + self.s * self.a_4,
                self.a_2 + self.t * self.a_4,
                self.a_3 + self.u * self.a_4,
            ]
        )
