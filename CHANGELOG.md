# Changelog

## [unreleased]

## [v0.3.23]
- Add Status Bar to the editor (currently shows FPS/TPS and basic version of the connection status)
- **(fix)** When a simulation file was changed, the associated pycache files would also be updated, causing the simulation to be re-built multiple times in some cases. This is now fixed.
- `elodin editor <path/to/sim>` now watches the parent directory of the simulation file for changes in addition to the file itself. This is useful for multi-file projects. This is also the case when using the `--watch` flag directly. E.g. `python <path/to/sim> run --watch`.
- Export simulation data to a directory using `exec.write_to_dir(path)`.
- Render the Z-axis as up in the editor (instead of the Y-axis). This is a purely visual change and does not affect simulation results, but it's recommended to update simulations to match the new visual orientation. Typically, this requires swapping the Y and Z axes when operating on spatial positions, velocities, and forces.

## [v0.3.22]
- **(fix)** Fix errors when using `vmap` with `scan`, and non-scalar values
    - When the arguments to a scan operation were non-scalar values (i.e their rank was above 0), scan would error in various ways when combined with vmap.
    The core issue is that some of our logic accidentally assumed an empty shape array, and when that array was non-empty, dimensions would be inserted into the wrong place.

## [v0.3.21]
- **(fix)** Fix missing 1/2 factor in angular velocity integration and `Quaternion::from_axis_angle` .
    - In `nox`, the constant `Field::two` returned a `1` constant. This constant was only used in the implementation of `Add` between `SpatialMotion` and `SpatialTransform` and in `Quaternion::from_axis_angle`. Unfortunately, this caused angular velocity integration to return incorrect results. This bug caused the applied angular velocity to be multiplied by a factor of 2.
    - The most significant impact of this bug is on the stability of any attitude control system. This bug has led to an increase in small oscillations, potentially affecting the performance of PID controllers tuned to work with previous versions of Elodin. PID controllers tuned to work with earlier versions of Elodin will likely need to be re-tuned
    - We regret not catching this bug earlier. To prevent a bug like this happening ever again, we have taken the following steps:
        1. We created a set of unit tests for the 6 DOF implementation that compare Elodin's results with Simulink. They have confirmed that our implementation now matched Simulink's [6DOF Quaternion block](https://www.mathworks.com/help/aeroblks/6dofquaternion.html).
        2. We will expand our set of unit tests to have 100% coverage of our math modules. We will test each module against Simulink and other trusted implementations.
        3. We will publish documentation on our testing methodology and reports for each module.
- **(fix)** Fix incorrent angular acceleration due to mismatched coordinate frames.
    - Spatial force and motion are defined in the world frame. The inertia tensor, however, is defined in the body frame. To correctly calculate angular acceleration, torque and inertia tensor must be in the same frame. To fix this, we now transform torque from the world frame to the body frame before calculating angular acceleration. Then, we transform the angular acceleration back to the world frame.
- **(breaking)** Split `el.Pbr` into `el.Mesh`, `el.Material`, `el.Shape`, and `el.Glb`.
  - Use the `el.shape` and `el.glb` helpers instead
    ```python
    # before:
    w.spawn(el.Body(pbr = w.insert_asset( el.Pbr(el.Mesh.sphere(0.2), el.Material.color(0.0, 10.0, 10.0)))))
    w.spawn(el.Body(w.insert_asset(el.Pbr.from_url("https://storage.googleapis.com/elodin-marketing/models/earth.glb"))))
    # after
    w.spawn([
      el.Body(),
      w.shape(el.Mesh.sphere(0.2), el.Material.color(0.0, 10.0, 10.0))
    ])
    w.spawn([
      el.Body(),
      w.glb("https://storage.googleapis.com/elodin-marketing/models/earth.glb")
    ])
    ```
- **(breaking)** Remove `SpatialInertia.from_mass()`.
  - Use the `SpatialInertia()` constructor instead, which now accepts inertia tensor as an optional keyword argument.
    ```python
    # before:
    inertia=el.SpatialInertia.from_mass(1.0 / G),
    # after:
    inertia=el.SpatialInertia(1.0 / G),
    ```
- If a `Component` base class provides `ComponentType` information via `__metadata__`, then it can be omitted from the `Component(...)` annotation as it can be inferred from the base class instead.
  - `Quaternion`, `Handle`, `Edge`, and all spatial classes (`SpatialTransform`, `SpatialForce`, `SpatialMotion`, `SpatialInertia`) have been updated to provide `ComponentType` information directly. So, components that annotate these classes can now be defined more simply:
    ```python
    # before:
    ControlForce = Annotated[
      el.SpatialForce, el.Component("control_force", el.ComponentType.SpatialMotionF64)
    ]
    # after:
    ControlForce = Annotated[
      el.SpatialForce, el.Component("control_force")
    ]
    ```
- Add visualization of the current tick in the graph views of the editor.
- Add `TotalEdge`, an edge type that connects every entity to every other entity.
- Fix issue where stepping backwards or forwards in the editor would sometimes not work.
- Add Command Palette to the editor

## [v0.3.20]

- **(breaking)** Replace entity builder pattern with a more flexible `spawn` API.
  - To migrate to the new API, replace `name()` with the keyword argument:
    ```python
    # before:
    w.spawn(el.Body()).name("entity_name")
    # after:
    w.spawn(el.Body(), name="entity_name")
    ```
  - If the entity has multiple archetypes, just provide a list of archetypes as the first positional argument instead of using `insert()`:
    ```python
    # before:
    w.spawn(el.Body()).name("entity_name").insert(OtherArchetype())
    # after:
    w.spawn([el.Body(), OtherArchetype()], name="entity_name")
    ```
  - Also, `spawn` now returns `EntityId` directly instead of `Entity`. So, there's no need to call `id()` to reference the entity in an edge component or viewport.
    ```python
    a = w.spawn(...)
    b = w.spawn(...)
    # before:
    w.spawn(Rel(el.Edge(a.id(), b.id())))
    # after:
    w.spawn(Rel(el.Edge(a, b)))
    ```
- Add multi-file support for Monte Carlo runs.
- Add ability to use ranges in viewports for replay.

## [v0.3.19]

- **(breaking)** Allow querying different components from the left and right entities via the new `edge_fold` API.
  - To migrate to the new API, move the graph query's input components to a separate query parameter:
    ```python
    # before:
    @el.system
    def gravity(
        graph: el.GraphQuery[GravityEdge, el.WorldPos, el.Inertia],
    )

    # after:
    @el.system
    def gravity(
        graph: el.GraphQuery[GravityEdge],
        query: el.Query[el.WorldPos, el.Inertia],
    )
    ```
    And then reference it in `edge_fold`:
    ```python
    # before:
    return graph.edge_fold(el.Force, ...

    # after:
    return graph.edge_fold(query, query, el.Force, ...
    ```
  - To query different components from the left and right entities, use multiple queries:
    ```python
    @el.system
    def rw_effector(
        rw_force: el.GraphQuery[RWEdge],
        force_query: el.Query[el.Force],
        rw_query: el.Query[RWForce]
    ) -> el.Query[el.Force]:
        return rw_force.edge_fold(force_query, rw_query, el.Force, ...
    ```
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

[unreleased]: https://github.com/elodin-sys/paracosm/compare/v0.3.23...HEAD
[v0.3.23]: https://github.com/elodin-sys/paracosm/compare/v0.3.22...v0.3.23
[v0.3.22]: https://github.com/elodin-sys/paracosm/compare/v0.3.21...v0.3.22
[v0.3.21]: https://github.com/elodin-sys/paracosm/compare/v0.3.20...v0.3.21
[v0.3.20]: https://github.com/elodin-sys/paracosm/compare/v0.3.19...v0.3.20
[v0.3.19]: https://github.com/elodin-sys/paracosm/compare/v0.3.18...v0.3.19
[v0.3.18]: https://github.com/elodin-sys/paracosm/compare/v0.3.17...v0.3.18
[v0.3.17]: https://github.com/elodin-sys/paracosm/compare/v0.3.16...v0.3.17
