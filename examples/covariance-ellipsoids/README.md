# Covariance Ellipsoids

Displays the same positive-definite covariance as two ellipsoids:

- `error_covariance_cholesky` receives the lower-triangular factor `L`.
- `error_covariance` receives `P = LLᵀ` directly.

Both viewports use the same camera offset and confidence interval, so the
ellipsoids should have identical shape and orientation.

Run from the repository root:

```sh
nix develop --command elodin editor examples/covariance-ellipsoids/main.py
```
