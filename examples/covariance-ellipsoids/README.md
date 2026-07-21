# Covariance Ellipsoids

Displays the same positive-definite covariance as two ellipsoids:

- `error_covariance_cholesky` receives the lower-triangular factor `L`.
- `error_covariance` receives `P = LLᵀ` directly.

The Cholesky factor varies continuously over an eight-second cycle, while the
direct covariance is recomputed as `P = LLᵀ` every simulation tick. Their
principal extents should remain synchronized. The direct covariance side uses
the example's explicit ECEF frame, while the Cholesky side inherits ENU.

Run from the repository root:

```sh
nix develop --command elodin editor examples/covariance-ellipsoids/main.py
```
