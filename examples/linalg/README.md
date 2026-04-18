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

1. `jnp.linalg.cholesky(S)` -- factorize innovation covariance
2. `jnp.linalg.solve(S, y)` -- solve-based Kalman gain (LU factorization)
3. `jnp.linalg.qr(P_upd)` -- square-root factorization of updated covariance
4. `jnp.linalg.det(S)` -- determinant for log-likelihood
5. `jnp.linalg.slogdet(S)` -- sign + log-determinant for numerically stable log-likelihood
6. `jnp.linalg.svd(S)` -- SVD-based pseudoinverse (6-state EKF)
7. `jnp.linalg.eigh(P)` -- symmetric eigendecomposition of covariance
8. `jnp.linalg.inv(P)` -- matrix inverse (2-state system)
