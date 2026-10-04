# Elodin model package — bdx-baseline

- Phase: `baseline`
- Credibility: **analysis-correlated**
- Pipeline run: `d861cb7409614557a8da39b11ce20652`
- Design SHA-256: `8296a967c139c2836a630e4d96cb7c73ffe177ea2c4ec34f4cdc3d2de4362bdb`
- Source git commit: `65f48ec93494dc737e36e3501ba31f03bb5e47b6`

## Evidence classes

- A: manufacturer-supported source
- B: independent corroboration
- C: engineering derivation / solver analysis
- D: provisional placeholder

## Allowances and limitations

- Attached-flow aerodynamics only; emit a validity flag outside the declared domain.
- Aero/structures solver agreement is verification, not physical-aircraft validation.
- Propulsion map is evaluated from the analytic lapse/TSFC model, not an identified engine deck.
- No measured inertia tensor is available; consumers must not invent one from this package.
- NACA 0012 may be a documented surrogate rather than the physical aircraft section.

This package binds one results phase only. Agreement between solvers is
verification, not validation against a physical aircraft.
