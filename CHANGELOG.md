# Changelog

## [unreleased]

## [v0.3.19]

- **(breaking)** Allow different graph query parameters via the new `edge_fold` API.
- Make graph colors deterministic.
- Drop milliseconds from the x-axis in graphs.
- Prevent grid from changing origin.
- Add ranges to graphs to allow zooming + panning.
- Add support for spawning graphs and splits from code.
- Switch to GPU based plotting.
- Add configurable line width for graphs.

## [v0.3.18]

- **(breaking)** Make RK4 the default integrator (can still override to use semi-implicit euler).
- Add element names to graphs.
- Add component priorities.
- Components in inspector are displayed in descending order of priority.
  - Priority can be set via metadata. E.g.: `metadata={"element_names": "q0,q1,q2,q3,x,y,z", "priority": 5}`.
  - Default priority, if unset, is 10.
  - Components with priority of < 0 are hidden from the inspector.
- Increase y-axis margin in graphs to prevent axis labels from getting cut off.
- Use scientific notation for large values.
- Prevent graph cursor modal from being cut off.
- Add `interia_diag()` helper to el.SpatialInertia.
- Fix graph data interpolation issues.

## [v0.3.17]

- Reduce memory usage of plots.
- Add rocket example with thrust curve interpolation.
- Clear graphs on sim update.
- Have a stable ordering of components in inspector.
- Make decimal point stable in component inspector.
- Add a built-in Time component + system.
- Remember window size on restart.
- Add configurable labels for component elements.

[unreleased]: https://github.com/elodin-sys/paracosm/compare/v0.3.19...HEAD
[v0.3.19]: https://github.com/elodin-sys/paracosm/compare/v0.3.18...v0.3.19
[v0.3.18]: https://github.com/elodin-sys/paracosm/compare/v0.3.17...v0.3.18
[v0.3.17]: https://github.com/elodin-sys/paracosm/compare/v0.3.16...v0.3.17
