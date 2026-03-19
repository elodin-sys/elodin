"""Pure-JAX replacements for LAPACK-backed linalg functions.

These implementations avoid ``stablehlo.custom_call`` operations AND
``stablehlo.scatter`` (which IREE rejects) so that the resulting
StableHLO compiles cleanly with IREE.  All array mutations use
``jnp.where`` with masks instead of ``.at[].set()``.

Every function is JAX-traceable (uses ``lax.fori_loop`` / ``lax.scan``
instead of Python loops with data-dependent control flow) and targets
small-to-medium matrices (up to ~20x20) typical of aerospace simulations.

The public entry point is :func:`iree_safe_linalg`, a context manager
that temporarily monkey-patches ``jnp.linalg.*`` and
``jax.scipy.linalg.*`` with these implementations.
"""

from __future__ import annotations

import contextlib
import logging

import jax
import jax.numpy as jnp
from jax import lax

logger = logging.getLogger(__name__)


def _set_col(M, k, col):
    """Set column ``k`` of ``M`` to ``col`` using ``where`` (no scatter)."""
    n = M.shape[1]
    mask = (jnp.arange(n) == k)[None, :]
    return jnp.where(mask, col[:, None], M)


def _set_row(M, k, row):
    """Set row ``k`` of ``M`` to ``row`` using ``where`` (no scatter)."""
    m = M.shape[0]
    mask = (jnp.arange(m) == k)[:, None]
    return jnp.where(mask, row[None, :], M)


def _set_elem(v, i, val):
    """Set element ``i`` of vector ``v`` to ``val`` using ``where``."""
    return jnp.where(jnp.arange(v.shape[0]) == i, val, v)


def _get_elem(v, i):
    """Extract scalar element ``i`` from vector ``v`` (no dynamic_slice)."""
    return jnp.dot(v, jnp.where(jnp.arange(v.shape[0]) == i, 1.0, 0.0))


# ---------------------------------------------------------------------------
# Cholesky  (column-by-column, scatter-free)
# ---------------------------------------------------------------------------


def _cholesky(a):
    """Pure-JAX Cholesky decomposition.  Returns lower-triangular ``L``
    such that ``a = L @ L.T``.  Input must be symmetric positive-definite.
    """
    a = jnp.asarray(a)
    n = a.shape[-1]
    L = jnp.zeros_like(a)
    idx = jnp.arange(n)

    def body(k, L):
        e_k = jnp.where(idx == k, 1.0, 0.0)
        Lk_row = L.T @ e_k
        prev = L @ Lk_row
        col = a @ e_k - prev
        diag_val = jnp.sqrt(jnp.maximum(jnp.dot(col, e_k), 0.0))
        safe_diag = jnp.where(diag_val == 0, 1.0, diag_val)
        col = jnp.where(idx == k, diag_val, col / safe_diag)
        col = jnp.where(idx < k, 0.0, col)
        L = _set_col(L, k, col)
        return L

    return lax.fori_loop(0, n, body, L)


# ---------------------------------------------------------------------------
# Triangular solve helpers (forward / back substitution, scatter-free)
# ---------------------------------------------------------------------------


def _solve_lower(L, b):
    """Forward substitution: solve ``L @ x = b``, ``L`` lower-triangular."""
    n = L.shape[-1]
    x = jnp.zeros_like(b)
    idx = jnp.arange(n)

    def body(i, x):
        e_i = jnp.where(idx == i, 1.0, 0.0)
        Lx_i = jnp.dot(e_i, L @ x)
        bi = jnp.dot(b, e_i)
        lii = jnp.dot(e_i, L @ e_i)
        xi = (bi - Lx_i) / lii
        return jnp.where(idx == i, xi, x)

    return lax.fori_loop(0, n, body, x)


def _solve_upper(U, b):
    """Back substitution: solve ``U @ x = b``, ``U`` upper-triangular."""
    n = U.shape[-1]
    x = jnp.zeros_like(b)
    idx = jnp.arange(n)

    def body(k, x):
        i = n - 1 - k
        e_i = jnp.where(idx == i, 1.0, 0.0)
        Ux_i = jnp.dot(e_i, U @ x)
        bi = jnp.dot(b, e_i)
        uii = jnp.dot(e_i, U @ e_i)
        xi = (bi - Ux_i) / uii
        return jnp.where(idx == i, xi, x)

    return lax.fori_loop(0, n, body, x)


# ---------------------------------------------------------------------------
# LU decomposition (Doolittle, no pivoting, scatter-free)
# ---------------------------------------------------------------------------


def _lu_no_pivot(a):
    """Pivotless LU factorisation.  Returns ``(L, U)`` with ``L`` unit
    lower-triangular and ``U`` upper-triangular such that ``a ~ L @ U``.
    """
    n = a.shape[-1]
    L = jnp.eye(n, dtype=a.dtype)
    U = jnp.zeros_like(a)
    idx = jnp.arange(n)

    def body(k, state):
        L, U = state
        e_k = jnp.where(idx == k, 1.0, 0.0)

        Lk_row = L.T @ e_k
        row_k = a.T @ e_k - U.T @ Lk_row
        row_k = jnp.where(idx >= k, row_k, 0.0)
        U = _set_row(U, k, row_k)

        ukk = jnp.dot(row_k, e_k)
        ukk_safe = jnp.where(ukk == 0, 1.0, ukk)
        Uk_col = U @ e_k
        col = (a @ e_k - L @ Uk_col) / ukk_safe
        col = jnp.where(idx > k, col, jnp.where(idx == k, 1.0, 0.0))
        L = _set_col(L, k, col)

        return (L, U)

    L, U = lax.fori_loop(0, n, body, (L, U))
    return L, U


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def _solve(a, b):
    """Solve ``a @ x = b`` via pivotless LU + forward/back substitution."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    L, U = _lu_no_pivot(a)
    if b.ndim == 1:
        y = _solve_lower(L, b)
        return _solve_upper(U, y)
    else:

        def solve_col(bi):
            y = _solve_lower(L, bi)
            return _solve_upper(U, y)

        return jax.vmap(solve_col, in_axes=-1, out_axes=-1)(b)


# ---------------------------------------------------------------------------
# inv  (Newton-Schulz iteration)
# ---------------------------------------------------------------------------


def _inv(a):
    """Matrix inverse via Newton-Schulz iteration (10 steps)."""
    a = jnp.asarray(a)
    norm_sq = jnp.sum(a * a)
    X = a.T / jnp.maximum(norm_sq, 1e-30)

    def step(_, X):
        return 2.0 * X - X @ a @ X

    return lax.fori_loop(0, 10, step, X)


# ---------------------------------------------------------------------------
# QR  (Householder reflections, scatter-free)
# ---------------------------------------------------------------------------


def _qr(a, mode="reduced"):
    """Pure-JAX QR decomposition via Householder reflections."""
    a = jnp.asarray(a)
    m, n = a.shape
    k = min(m, n)
    Q = jnp.eye(m, dtype=a.dtype)
    R = a + 0.0
    row_idx = jnp.arange(m)
    col_idx = jnp.arange(n)

    def body(j, state):
        Q, R = state
        e_j_col = jnp.where(col_idx == j, 1.0, 0.0)
        e_j_row = jnp.where(row_idx == j, 1.0, 0.0)
        r_col = R @ e_j_col
        x = jnp.where(row_idx >= j, r_col, 0.0)
        alpha = jnp.sqrt(jnp.dot(x, x))
        sign = jnp.where(jnp.dot(r_col, e_j_row) >= 0, 1.0, -1.0)
        alpha = sign * alpha

        xj = jnp.dot(x, e_j_row)
        v = _set_elem(x, j, xj + alpha)
        v_norm_sq = jnp.dot(v, v)
        v_norm_sq = jnp.maximum(v_norm_sq, 1e-30)

        R = R - (2.0 / v_norm_sq) * jnp.outer(v, v @ R)
        Q = Q - (2.0 / v_norm_sq) * jnp.outer(Q @ v, v)
        return Q, R

    Q, R = lax.fori_loop(0, k, body, (Q, R))

    if mode == "reduced":
        return Q[:, :k], R[:k, :]
    elif mode == "complete":
        return Q, R
    elif mode == "r":
        return R[:k, :]
    return Q[:, :k], R[:k, :]


# ---------------------------------------------------------------------------
# SVD  (one-sided Jacobi, scatter-free)
# ---------------------------------------------------------------------------


def _svd(a, full_matrices=True, compute_uv=True):
    """Pure-JAX SVD via one-sided Jacobi rotations with fixed sweeps."""
    a = jnp.asarray(a)
    m, n = a.shape
    k = min(m, n)
    num_sweeps = 15

    B = a + 0.0
    V = jnp.eye(n, dtype=a.dtype)

    n_pairs = (n * (n - 1)) // 2
    pairs_p = []
    pairs_q = []
    for p_val in range(n):
        for q_val in range(p_val + 1, n):
            pairs_p.append(p_val)
            pairs_q.append(q_val)
    pairs_p = jnp.array(pairs_p, dtype=jnp.int32)
    pairs_q = jnp.array(pairs_q, dtype=jnp.int32)

    col_idx = jnp.arange(n)

    def sweep(_, state):
        B, V = state

        def rotate(carry, pair_idx):
            B, V = carry
            p = pairs_p[pair_idx]
            q = pairs_q[pair_idx]

            bp = B[:, p]
            bq = B[:, q]
            alpha = jnp.dot(bp, bp)
            beta = jnp.dot(bq, bq)
            gamma = jnp.dot(bp, bq)

            converged = jnp.abs(gamma) < 1e-15 * jnp.sqrt(alpha * beta + 1e-300)
            tau = (beta - alpha) / (2.0 * gamma + 1e-300)
            t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1.0 + tau**2))
            c = 1.0 / jnp.sqrt(1.0 + t**2)
            s = t * c

            c = jnp.where(converged, 1.0, c)
            s = jnp.where(converged, 0.0, s)

            new_bp = c * bp - s * bq
            new_bq = s * bp + c * bq

            mask_p = (col_idx == p)[None, :]
            mask_q = (col_idx == q)[None, :]
            B = jnp.where(mask_p, new_bp[:, None], B)
            B = jnp.where(mask_q, new_bq[:, None], B)

            vp = V[:, p]
            vq = V[:, q]
            new_vp = c * vp - s * vq
            new_vq = s * vp + c * vq
            V = jnp.where(mask_p, new_vp[:, None], V)
            V = jnp.where(mask_q, new_vq[:, None], V)

            return (B, V), None

        (B, V), _ = lax.scan(rotate, (B, V), jnp.arange(n_pairs))
        return (B, V)

    B, V = lax.fori_loop(0, num_sweeps, sweep, (B, V))

    sigma = jnp.sqrt(jnp.sum(B**2, axis=0))
    safe_sigma = jnp.maximum(sigma, 1e-300)
    U_out = B / safe_sigma[None, :]

    order = jnp.argsort(-sigma)
    sigma = sigma[order]
    U_out = U_out[:, order]
    V = V[:, order]

    if not compute_uv:
        return sigma[:k]
    if full_matrices:
        return U_out, sigma[:k], V.T
    return U_out[:, :k], sigma[:k], V[:, :k].T


# ---------------------------------------------------------------------------
# det / slogdet  (via pivotless LU)
# ---------------------------------------------------------------------------


def _det(a):
    """Determinant via pivotless LU decomposition."""
    a = jnp.asarray(a)
    _, U = _lu_no_pivot(a)
    return jnp.prod(jnp.diag(U))


def _slogdet(a):
    """Sign and log-absolute-determinant via pivotless LU."""
    a = jnp.asarray(a)
    _, U = _lu_no_pivot(a)
    diag = jnp.diag(U)
    sign = jnp.prod(jnp.sign(diag))
    logabsdet = jnp.sum(jnp.log(jnp.abs(diag)))
    return sign, logabsdet


# ---------------------------------------------------------------------------
# eigh  (cyclic Jacobi for symmetric matrices, scatter-free)
# ---------------------------------------------------------------------------


def _eigh(a, UPLO="L"):
    """Eigendecomposition for symmetric matrices via cyclic Jacobi."""
    a = jnp.asarray(a)
    n = a.shape[-1]
    num_sweeps = 20

    A = 0.5 * (a + a.T)
    V = jnp.eye(n, dtype=a.dtype)

    n_pairs = (n * (n - 1)) // 2
    pp = []
    pq = []
    for p_val in range(n):
        for q_val in range(p_val + 1, n):
            pp.append(p_val)
            pq.append(q_val)
    pairs_p = jnp.array(pp, dtype=jnp.int32)
    pairs_q = jnp.array(pq, dtype=jnp.int32)

    row_idx = jnp.arange(n)

    def sweep(_, state):
        A, V = state

        def rotate(carry, pair_idx):
            A, V = carry
            p = pairs_p[pair_idx]
            q = pairs_q[pair_idx]

            app = A[p, p]
            aqq = A[q, q]
            apq = A[p, q]

            converged = jnp.abs(apq) < 1e-15 * jnp.sqrt(jnp.abs(app * aqq) + 1e-300)

            tau = (aqq - app) / (2.0 * apq + 1e-300)
            t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1.0 + tau**2))
            c = 1.0 / jnp.sqrt(1.0 + t**2)
            s = t * c

            c = jnp.where(converged, 1.0, c)
            s = jnp.where(converged, 0.0, s)

            row_p = A[p, :]
            row_q = A[q, :]
            new_row_p = c * row_p - s * row_q
            new_row_q = s * row_p + c * row_q

            mask_rp = (row_idx == p)[:, None]
            mask_rq = (row_idx == q)[:, None]
            A = jnp.where(mask_rp, new_row_p[None, :], A)
            A = jnp.where(mask_rq, new_row_q[None, :], A)

            col_p = A[:, p]
            col_q = A[:, q]
            new_col_p = c * col_p - s * col_q
            new_col_q = s * col_p + c * col_q

            mask_cp = (row_idx == p)[None, :]
            mask_cq = (row_idx == q)[None, :]
            A = jnp.where(mask_cp, new_col_p[:, None], A)
            A = jnp.where(mask_cq, new_col_q[:, None], A)

            vp = V[:, p]
            vq = V[:, q]
            new_vp = c * vp - s * vq
            new_vq = s * vp + c * vq
            V = jnp.where(mask_cp, new_vp[:, None], V)
            V = jnp.where(mask_cq, new_vq[:, None], V)

            return (A, V), None

        (A, V), _ = lax.scan(rotate, (A, V), jnp.arange(n_pairs))
        return (A, V)

    A, V = lax.fori_loop(0, num_sweeps, sweep, (A, V))

    eigenvalues = jnp.diag(A)
    order = jnp.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    V = V[:, order]

    return eigenvalues, V


# ---------------------------------------------------------------------------
# scipy.linalg wrappers
# ---------------------------------------------------------------------------


def _scipy_solve(
    a,
    b,
    lower=False,
    overwrite_a=False,
    overwrite_b=False,
    check_finite=True,
    assume_a="gen",
    transposed=False,
):
    """Drop-in for ``jax.scipy.linalg.solve``."""
    return _solve(a, b)


def _lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    """Drop-in for ``jax.scipy.linalg.lu``."""
    a = jnp.asarray(a)
    n = a.shape[-1]
    L, U = _lu_no_pivot(a)
    P = jnp.eye(n, dtype=a.dtype)
    if permute_l:
        return P @ L, U
    return P, L, U


def _cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Drop-in for ``jax.scipy.linalg.cho_solve``."""
    c, lower = c_and_lower
    if lower:
        y = _solve_lower(c, b)
        return _solve_upper(c.T, y)
    else:
        y = _solve_lower(c.T, b)
        return _solve_upper(c, y)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def _build_replacements():
    """Build the replacement registry lazily so jax modules are ready."""
    import jax.numpy as _jnp
    import jax.scipy.linalg as _jscl

    return [
        (_jnp.linalg, "solve", _solve),
        (_jnp.linalg, "inv", _inv),
        (_jnp.linalg, "cholesky", _cholesky),
        (_jnp.linalg, "qr", _qr),
        (_jnp.linalg, "svd", _svd),
        (_jnp.linalg, "det", _det),
        (_jnp.linalg, "slogdet", _slogdet),
        (_jnp.linalg, "eigh", _eigh),
        (_jscl, "solve", _scipy_solve),
        (_jscl, "lu", _lu),
        (_jscl, "cho_solve", _cho_solve),
    ]


@contextlib.contextmanager
def iree_safe_linalg():
    """Temporarily replace LAPACK-backed linalg functions with pure-JAX impls.

    Use as a context manager around ``jax.jit(fn).lower(...)`` during IREE
    compilation so that the resulting StableHLO contains no
    ``custom_call`` ops targeting LAPACK kernels.
    """
    replacements = _build_replacements()
    originals: list[tuple] = []
    for mod, name, replacement in replacements:
        originals.append((mod, name, getattr(mod, name)))
        setattr(mod, name, replacement)
    logger.debug("iree_safe_linalg: patched %d linalg functions", len(originals))
    try:
        yield
    finally:
        for mod, name, original in originals:
            setattr(mod, name, original)
        logger.debug("iree_safe_linalg: restored %d linalg functions", len(originals))
