# Linear Algebra Validation

This example validates that all LAPACK-backed linear algebra operations compile
and run correctly. It runs Kalman filter simulations of varying sizes that
exercise `jnp.linalg.cholesky`, `jnp.linalg.solve`, `jnp.linalg.inv`,
`jnp.linalg.qr`, `jnp.linalg.svd`, `jnp.linalg.det`, `jnp.linalg.slogdet`,
and `jnp.linalg.eigh`.

## Run

From repo root:

```bash
nix develop
```

Default (Cranelift backend, 600 ticks):

```bash
elodin run examples/linalg/main.py
```

Bench mode for CI:

```bash
python examples/linalg/main.py bench --ticks 100
```

Override backend:

```bash
ELODIN_BACKEND=jax-cpu python examples/linalg/main.py bench --ticks 100
```

## What this example exercises

The simulation systems in `sim.py` run Kalman filter ticks that call:

1. `jnp.linalg.cholesky(S)` -- factorize innovation covariance (lower factor)
2. `jax.scipy.linalg.cholesky(A, lower=False)` -- upper factor U s.t. A = U.T @ U
3. `jax.scipy.linalg.cholesky(A, lower=True)` -- lower factor L s.t. A = L @ L.T
4. `jnp.linalg.cholesky(A_batch)` -- batched Cholesky over leading dim
5. `jnp.linalg.solve(S, y)` -- solve-based Kalman gain (LU factorization)
6. `jnp.linalg.qr(P_upd)` -- square-root factorization of updated covariance
7. `jnp.linalg.det(S)` -- determinant for log-likelihood
8. `jnp.linalg.slogdet(S)` -- sign + log-determinant for numerically stable log-likelihood
9. `jnp.linalg.svd(S)` -- SVD-based pseudoinverse (6-state EKF)
10. `jnp.linalg.eigh(P)` -- symmetric eigendecomposition of covariance
11. `jnp.linalg.inv(P)` -- matrix inverse (2-state system)

The Cholesky variants (2--4) each reconstruct their input from the returned
factor and report the Frobenius norm of the residual in the `chol_res_norms`
component. A correct implementation produces residuals near machine epsilon;
the norms should be visible in telemetry as `[upper, lower, batched]`.
