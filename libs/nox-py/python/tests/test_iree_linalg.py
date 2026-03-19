"""Tests for pure-JAX linalg shims (elodin._iree_linalg).

Each test verifies:
  - Correctness vs ``jnp.linalg.*`` at multiple matrix sizes
  - f64 precision
  - JAX-traceability (wrap in ``jax.jit``)
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from elodin._iree_linalg import (  # noqa: E402
    _cholesky,
    _cho_solve,
    _det,
    _eigh,
    _inv,
    _lu,
    _lu_no_pivot,
    _qr,
    _scipy_solve,
    _slogdet,
    _solve,
    _svd,
    iree_safe_linalg,
)

SIZES = [2, 3, 6, 10, 18]


def _spd_matrix(key, n):
    """Generate a random symmetric positive-definite matrix."""
    A = jax.random.normal(key, (n, n))
    return A @ A.T + 0.5 * jnp.eye(n)


def _well_conditioned(key, n):
    """Generate a well-conditioned random square matrix."""
    A = jax.random.normal(key, (n, n))
    return A + n * jnp.eye(n)


# -----------------------------------------------------------------------
# Cholesky
# -----------------------------------------------------------------------


class TestCholesky:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_jnp(self, n):
        A = _spd_matrix(jax.random.PRNGKey(0), n)
        L_ref = jnp.linalg.cholesky(A)
        L_ours = _cholesky(A)
        assert jnp.allclose(L_ref, L_ours, atol=1e-10), (
            f"cholesky mismatch at n={n}: max diff={jnp.max(jnp.abs(L_ref - L_ours))}"
        )

    @pytest.mark.parametrize("n", [3, 6])
    def test_jit(self, n):
        A = _spd_matrix(jax.random.PRNGKey(1), n)
        L = jax.jit(_cholesky)(A)
        assert jnp.allclose(A, L @ L.T, atol=1e-10)


# -----------------------------------------------------------------------
# LU (no pivot)
# -----------------------------------------------------------------------


class TestLU:
    @pytest.mark.parametrize("n", SIZES)
    def test_reconstructs(self, n):
        A = _well_conditioned(jax.random.PRNGKey(2), n)
        L, U = _lu_no_pivot(A)
        recon = L @ U
        assert jnp.allclose(A, recon, atol=1e-8), (
            f"LU reconstruction mismatch at n={n}: max diff={jnp.max(jnp.abs(A - recon))}"
        )


# -----------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------


class TestSolve:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_jnp(self, n):
        key = jax.random.PRNGKey(3)
        A = _well_conditioned(key, n)
        b = jax.random.normal(jax.random.PRNGKey(4), (n,))
        x_ref = jnp.linalg.solve(A, b)
        x_ours = _solve(A, b)
        assert jnp.allclose(x_ref, x_ours, atol=1e-8), (
            f"solve mismatch at n={n}: max diff={jnp.max(jnp.abs(x_ref - x_ours))}"
        )

    def test_matrix_rhs(self):
        A = _well_conditioned(jax.random.PRNGKey(5), 4)
        B = jax.random.normal(jax.random.PRNGKey(6), (4, 3))
        X_ref = jnp.linalg.solve(A, B)
        X_ours = _solve(A, B)
        assert jnp.allclose(X_ref, X_ours, atol=1e-8)

    @pytest.mark.parametrize("n", [3, 6])
    def test_jit(self, n):
        A = _well_conditioned(jax.random.PRNGKey(7), n)
        b = jax.random.normal(jax.random.PRNGKey(8), (n,))
        x = jax.jit(_solve)(A, b)
        assert jnp.allclose(A @ x, b, atol=1e-8)


# -----------------------------------------------------------------------
# Inv
# -----------------------------------------------------------------------


class TestInv:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_jnp(self, n):
        A = _well_conditioned(jax.random.PRNGKey(9), n)
        Ainv_ref = jnp.linalg.inv(A)
        Ainv_ours = _inv(A)
        assert jnp.allclose(Ainv_ref, Ainv_ours, atol=1e-4), (
            f"inv mismatch at n={n}: max diff={jnp.max(jnp.abs(Ainv_ref - Ainv_ours))}"
        )

    @pytest.mark.parametrize("n", [3, 6])
    def test_identity_product(self, n):
        A = _well_conditioned(jax.random.PRNGKey(10), n)
        Ainv = _inv(A)
        eye_approx = A @ Ainv
        assert jnp.allclose(eye_approx, jnp.eye(n), atol=1e-4)


# -----------------------------------------------------------------------
# QR
# -----------------------------------------------------------------------


class TestQR:
    @pytest.mark.parametrize("n", SIZES)
    def test_reconstruction(self, n):
        A = jax.random.normal(jax.random.PRNGKey(11), (n, n))
        Q, R = _qr(A)
        recon = Q @ R
        assert jnp.allclose(A, recon, atol=1e-8), (
            f"QR reconstruction mismatch at n={n}: max diff={jnp.max(jnp.abs(A - recon))}"
        )

    @pytest.mark.parametrize("n", SIZES)
    def test_orthogonality(self, n):
        A = jax.random.normal(jax.random.PRNGKey(12), (n, n))
        Q, _ = _qr(A)
        eye_approx = Q.T @ Q
        assert jnp.allclose(eye_approx, jnp.eye(n), atol=1e-8)

    def test_r_mode(self):
        A = jax.random.normal(jax.random.PRNGKey(13), (4, 4))
        R = _qr(A, mode="r")
        assert R.shape == (4, 4)


# -----------------------------------------------------------------------
# SVD
# -----------------------------------------------------------------------


class TestSVD:
    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_reconstruction(self, n):
        A = jax.random.normal(jax.random.PRNGKey(14), (n, n))
        U, s, Vt = _svd(A)
        recon = U[:, :n] @ jnp.diag(s) @ Vt[:n, :]
        assert jnp.allclose(A, recon, atol=1e-6), (
            f"SVD reconstruction mismatch at n={n}: max diff={jnp.max(jnp.abs(A - recon))}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_singular_values_match(self, n):
        A = jax.random.normal(jax.random.PRNGKey(15), (n, n))
        s_ref = jnp.sort(jnp.linalg.svd(A, compute_uv=False))
        s_ours = jnp.sort(_svd(A, compute_uv=False))
        assert jnp.allclose(s_ref, s_ours, atol=1e-6), f"SVD singular values mismatch at n={n}"

    def test_no_uv(self):
        A = jax.random.normal(jax.random.PRNGKey(16), (3, 3))
        s = _svd(A, compute_uv=False)
        assert s.shape == (3,)


# -----------------------------------------------------------------------
# Det / Slogdet
# -----------------------------------------------------------------------


class TestDet:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_jnp(self, n):
        A = _well_conditioned(jax.random.PRNGKey(17), n)
        d_ref = jnp.linalg.det(A)
        d_ours = _det(A)
        assert jnp.allclose(d_ref, d_ours, rtol=1e-6), (
            f"det mismatch at n={n}: ref={d_ref}, ours={d_ours}"
        )


class TestSlogdet:
    @pytest.mark.parametrize("n", SIZES)
    def test_matches_jnp(self, n):
        A = _well_conditioned(jax.random.PRNGKey(18), n)
        sign_ref, logdet_ref = jnp.linalg.slogdet(A)
        sign_ours, logdet_ours = _slogdet(A)
        assert jnp.allclose(sign_ref, sign_ours, atol=1e-10)
        assert jnp.allclose(logdet_ref, logdet_ours, atol=1e-6), (
            f"slogdet mismatch at n={n}: ref={logdet_ref}, ours={logdet_ours}"
        )


# -----------------------------------------------------------------------
# Eigh
# -----------------------------------------------------------------------


class TestEigh:
    @pytest.mark.parametrize("n", [2, 3, 4, 6, 10])
    def test_eigenvalues_match(self, n):
        A = _spd_matrix(jax.random.PRNGKey(19), n)
        w_ref, _ = jnp.linalg.eigh(A)
        w_ours, _ = _eigh(A)
        assert jnp.allclose(w_ref, w_ours, atol=1e-8), (
            f"eigh eigenvalue mismatch at n={n}: max diff={jnp.max(jnp.abs(w_ref - w_ours))}"
        )

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_reconstruction(self, n):
        A = _spd_matrix(jax.random.PRNGKey(20), n)
        w, V = _eigh(A)
        recon = V @ jnp.diag(w) @ V.T
        assert jnp.allclose(A, recon, atol=1e-8), f"eigh reconstruction mismatch at n={n}"


# -----------------------------------------------------------------------
# scipy.linalg wrappers
# -----------------------------------------------------------------------


class TestScipySolve:
    def test_matches_solve(self):
        A = _well_conditioned(jax.random.PRNGKey(21), 4)
        b = jax.random.normal(jax.random.PRNGKey(22), (4,))
        x = _scipy_solve(A, b)
        assert jnp.allclose(A @ x, b, atol=1e-8)


class TestLUWrapper:
    def test_reconstruction(self):
        A = _well_conditioned(jax.random.PRNGKey(23), 4)
        P, L, U = _lu(A)
        recon = P @ L @ U
        assert jnp.allclose(A, recon, atol=1e-8)


class TestChoSolve:
    def test_matches_solve(self):
        A = _spd_matrix(jax.random.PRNGKey(24), 4)
        b = jax.random.normal(jax.random.PRNGKey(25), (4,))
        L = _cholesky(A)
        x = _cho_solve((L, True), b)
        assert jnp.allclose(A @ x, b, atol=1e-8)


# -----------------------------------------------------------------------
# Context manager
# -----------------------------------------------------------------------


class TestContextManager:
    def test_patches_and_restores(self):
        original_solve = jnp.linalg.solve
        with iree_safe_linalg():
            assert jnp.linalg.solve is not original_solve
        assert jnp.linalg.solve is original_solve

    def test_patched_solve_works(self):
        A = _well_conditioned(jax.random.PRNGKey(26), 4)
        b = jax.random.normal(jax.random.PRNGKey(27), (4,))
        with iree_safe_linalg():
            x = jnp.linalg.solve(A, b)
        assert jnp.allclose(A @ x, b, atol=1e-8)

    def test_patched_jit_traces(self):
        A = _well_conditioned(jax.random.PRNGKey(28), 3)
        b = jax.random.normal(jax.random.PRNGKey(29), (3,))
        with iree_safe_linalg():
            fn = jax.jit(lambda a, b: jnp.linalg.solve(a, b))
            lowered = fn.lower(A, b)
            hlo_text = lowered.as_text()
        assert "custom_call" not in hlo_text.lower() or "lapack" not in hlo_text.lower()
