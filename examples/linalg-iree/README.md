# IREE Linear Algebra Shims Validation

This example validates that Elodin's IREE-safe linear algebra shims compile and
run correctly on the IREE backend. It runs a simplified 3-state Kalman filter
that calls `jnp.linalg.cholesky`, `jnp.linalg.inv`, `jnp.linalg.qr`,
`jnp.linalg.det`, and `jnp.linalg.slogdet` -- all functions that would
normally produce LAPACK `custom_call` ops that IREE cannot compile.

## Background

JAX's `jnp.linalg.*` functions lower to LAPACK kernel calls
(`stablehlo.custom_call` with targets like `lapack_dgetrf`, `lapack_dgesdd`,
etc.). IREE has no mechanism to resolve these calls, so any simulation using
standard JAX linear algebra would fail at compile time.

Elodin solves this transparently. During IREE compilation, the
`iree_safe_linalg()` context manager in `elodin._iree_linalg` temporarily
replaces every LAPACK-backed `jnp.linalg` and `jax.scipy.linalg` function with
a pure-JAX implementation. The replacement functions:

- Use `lax.fori_loop` / `lax.scan` for all iteration (JAX-traceable).
- Use `jnp.where` with masks for all array mutations (no `.at[].set()` scatter).
- Use `jnp.dot` with one-hot vectors for element access (no `dynamic_slice`).

This produces clean StableHLO with no `custom_call` or `scatter` ops, which
IREE compiles without issues.

Users do not need to change their code. The shims activate automatically when
`backend="iree-cpu"` or `backend="iree-gpu"` is selected.

## Shims provided

| Function | Algorithm | Iterations / Sweeps |
|---|---|---|
| `jnp.linalg.cholesky` | Cholesky-Banachiewicz (column-by-column) | n steps |
| `jnp.linalg.solve` | Pivotless Doolittle LU + triangular solves | n steps |
| `jnp.linalg.inv` | Newton-Schulz | 10 steps |
| `jnp.linalg.qr` | Householder reflections | min(m,n) steps |
| `jnp.linalg.svd` | One-sided Jacobi rotations | 15 sweeps |
| `jnp.linalg.det` | Pivotless LU, product of diagonal | n steps |
| `jnp.linalg.slogdet` | Pivotless LU, sign + log-abs of diagonal | n steps |
| `jnp.linalg.eigh` | Cyclic Jacobi eigenvalue | 20 sweeps |
| `jax.scipy.linalg.solve` | Same as `jnp.linalg.solve` | -- |
| `jax.scipy.linalg.lu` | Pivotless Doolittle LU | n steps |
| `jax.scipy.linalg.cho_solve` | Cholesky forward/back substitution | n steps |

All implementations target matrices up to ~20x20, which covers the vast
majority of aerospace EKF / control / navigation use cases.

## Run

From repo root:

```bash
nix develop
```

Default (IREE backend, 600 ticks):

```bash
elodin run examples/linalg-iree/main.py
```

Bench mode for CI:

```bash
python examples/linalg-iree/main.py bench --ticks 100
```

Override backend:

```bash
ELODIN_BACKEND=jax-cpu python examples/linalg-iree/main.py bench --ticks 100
```

## Unit tests

The shim implementations have their own test suite:

```bash
uv run python -m pytest libs/nox-py/python/tests/test_iree_linalg.py -v
```

Tests verify correctness against `jnp.linalg.*` reference implementations at
matrix sizes 2x2 through 18x18 in f64 precision.

## Known limitations

- **Pivotless LU**: The `solve`, `det`, and `slogdet` shims use LU
  decomposition without pivoting. This is numerically unstable for
  ill-conditioned matrices but sufficient for the well-conditioned systems
  typical of aerospace EKFs (condition numbers < 10^6).

- **Newton-Schulz convergence**: The `inv` shim uses 10 Newton-Schulz
  iterations with initial guess `X0 = A^T / ||A||_F^2`. For well-conditioned
  matrices this achieves ~1e-4 accuracy relative to LAPACK. Poorly conditioned
  matrices may not converge.

- **IREE O2 optimizer bug**: When multiple `lax.fori_loop`-heavy operations
  (e.g., LU-based `solve` + Householder `qr`) appear in the same `@el.map`
  function, IREE's `-O2` optimization pass can produce invalid SSA dominance in
  the lowered IR. Elodin defaults to `-O1` to work around this. Individual
  operations compile fine at `-O2`.

- **SVD / eigh sweep count**: The Jacobi-based SVD and eigh use fixed sweep
  counts (15 and 20 respectively). For matrices larger than ~10x10, convergence
  should be verified for the specific application.

## What this example exercises

The `kf_step` system in `sim.py` runs a single Kalman filter tick that calls:

1. `jnp.linalg.cholesky(S)` -- factorize innovation covariance
2. `jnp.linalg.inv(S)` -- compute Kalman gain via `K = P H^T S^{-1}`
3. `jnp.linalg.qr(P_upd)` -- square-root factorization of updated covariance
4. `jnp.linalg.det(S)` -- determinant for log-likelihood
5. `jnp.linalg.slogdet(S)` -- sign + log-determinant for numerically stable log-likelihood

All five calls go through the IREE-safe shims during compilation, producing a
VMFB that executes in the pure-Rust IREE runtime with no Python per tick.
