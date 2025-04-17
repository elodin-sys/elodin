# Changelog

## unreleased

- **(breaking)** Switch vtables to use a new bytecode based format. This changes makes vtables simpler and more flexible. This change also effects the Lua format. You can now formulate a vtable with the `vtable_msg` function:

  ```lua
  vtable =  vtable_msg(1, {
      field(0, 12, schema("f32", {3}, pair(1, "accel"))), # field 
      field(12, 12, schema("f32", {3}, pair(1, "gyro"))),
  })

  ```
- **(feat)** Add VTableStream to elodin-db. VTableStream lets you specify a vtable for the database to populate, and have data streamed directly into it. You can also specify aggreator operations like `mean`. For example to setup a stream that calculates the mean of every 10 values:

  ```lua
  client:vtable_stream({field(0, 4, mean(10, schema("f32", {}, pair(1, "temp"))))})
  ```
- Add CSV as an archive data format. From `elodin-db lua`: `client:save_archive("path/to/dir", "csv")`.
- Don't add "Globals" entity + "Tick" component on `elodin-db` initialization. This entity and component are only required for simulation purposes, not HITL.

## v0.13

### v0.13.3
- **(fix)** Fix a bug where the db/editor's time window would be smaller than the actual data's time window.

### v0.13.2
- **(fix)** Fix a bug where the dump schema and dump metadata responses didn't include the request id.

### v0.13.1
- **(fix)** Fix a bug where the db wouldn't stream data to the editor correctly when there's a vtable that's registered but not used.

### v0.13.0
- **(breaking)** Remove `exec.write_to_dir(path)`. This is made obsolete by the new `SaveArchive` method in elodin-db.
- **(fix)** Fix an issue where the component inspector would show incorrect data for some multi-dimensional components.

## v0.12

### v0.12.3
- **(feat)** Add "presets" to the editor. Presets allow you to save your current editor layout (graphs, viewports, etc), and load them later
- **(feat)** add `SaveArchive` method to db that allows saving arrow ipc or parquet archives
  `SaveArchive` will dump the current contents of the database to the file system at the specified path. By default this uses arrow-ipc as the file format, but it can be switched to use parquet
- **(feat)** add C++ code generation to elodin-db

### v0.12.2
- **(breaking)** Make the elodin-db HTTP server optional. If an explicit `--http-addr` argument is not provided, the HTTP server will not be started.
- Add UDP unicast support to elodin-db. Configure using the `UdpUnicast` message.
- Add ability to mirror elodin-db over UDP unicast. Simply configure one db instance to stream data to another db instance's listen address.

### v0.12.1
- **(feat)** Add HTTP API to elodin-db.

### v0.12.0
- **(breaking)** Add a request_id field to `PacketHeader` and shorten `PacketId` to `[u8; 3]`. In order to support better request-reply semantics, a u8 request_id was added to `PacketHeader`.
- **(breaking)** The primitive types in the lua api have been lowercased so `F64` is now `f64` and so on.
- **(breaking)** When using `send_msg` and `send_msgs` you no longer have to call `:msg()` on every msg. See the updated `examples/db-config.lua` for the new API
- **(feat:db)** Add support for recording data at timestamps to elodin-db
elodin-db now no longer operates on the concept of "ticks". Instead each new value is recorded as a pair of timestamp and value. This essentially makes elodin-db into a timer series database. It also fixes the problem where elodin-db would record a new entry, even when the value is unchanged.
- **(feat)** Add support for SQL queries in elodin-db. These can be accessed from the CLI using the `:sql` command or through the editor using the new SQL pane. We used datafusion-sql to implement SQL support. You can find their docs here (https://datafusion.apache.org/user-guide/sql/index.html) for the syntax.
- **(feat)** Add startup screen to editor that allows connecting to elodin-db or simulations, and running sims from files
- **(feat:editor)** The component monitor UI has been redesigned to be easier to read
- **(feat)** Add touch screen support to plots
- **(fix)** Fix plotting x axis tick marks placed in the wrong position

## v0.11

### v0.11.3
- **(fix)** Fix bug where it wasn't possible to set entity or component metadata from Lua config. The following should now work:
  ```lua
  SetComponentMetadata({ component_id = time_id, name = "time", metadata = { priority = "100" } }):msg()
  ```
- **(fix)** Fix an issue where a panic could be caused when a stellarator executor was dropped from a thread local. Now the executor is manually dropped when `stellarator::run` finished.

### v0.11.2
- **(breaking)** Replace message-specific `elodin-db` Lua methods with generic `send_msg()` and `send_msgs()`.
- **(breaking)** Reduce `impeller` packet header size from 16 bytes to 8 bytes. The `len` field is now a `u32` instead of a `u64`. The `req_id` field has also been removed.
- Add a `--config` argument to `elodin-db run` to provide a custom Lua config file that can pre-configure the database with necessary vtables and metadata.

### v0.11.1
- **(fix)** Fix bug where live reload would fail with an "Address already in use" error.
- **(fix)** Fix bug where duplicate panels would be created on live reload.

### v0.11.0
- **(breaking)** Add required `component_name` and `entity_id` arguments to `Exec.history` function.
- Change playback speed in the editor using the "Set Playback Speed" command in the command palette.
- Display plot data as soon as it is available rather than waiting for the simulation playback.

## v0.10

### v0.10.1
- Add EGM2008 gravity model to the Python API.

### v0.10.0
- **(breaking)** Add "optimize" as an optional argument to `World.run` (defaults to `False`). This will enable additional optimization passes on the simulation code, which results in faster simulation run times at the cost of slower simulation startup times. This is useful for rapidly iterating on simulation code.
- **(breaking)** Replace "output_time_step" with "default_playback_speed" in `World.run`. This argument sets the default playback speed of the simulation in the editor. A value of 1.0 means that the simulation will run in real-time. A value of 0.5 means that the simulation will run at half speed, and so on.
- Expose more default arguments via type hints.

## v0.9

### v0.9.0
- **(breaking)** Remove deprecated `time_step` argument from `World.run`. Use `sim_time_step` instead.

## v0.8

### v0.8.0
- **(breaking)** Remove deprecated `from_linear`, `from_angular`, and `zero` static methods from spatial types. Use the constructor instead with optional arguments.

## v0.7

### v0.7.4
- **(fix)** Fix bug where `editor run` would fail if the simulation file was in the current directory because it would attempt to get the parent directory without canonicalizing the path.
- **(fix)** Fix `elodin login` in WSL. Previously, it would fail when attempting to open the browser. Now, it will print the URL to the console and you can copy and paste it into your browser.

### v0.7.2
- Use logical scan codes for the editor's key bindings. This means that keyboard shortcuts will respect your keyboard layout. For example, if you remap your Caps Lock key to Ctrl, the editor will treat Caps Lock as Ctrl as well.

### v0.7.1
- On Windows, use Ctrl instead of Super for launching the command palette (Ctrl + P).

### v0.7.0
- **(breaking)** Replace `--watch` option with `elodin run`.
As part of adding process supervision to Elodin, we refactored how watching a simulation works. Now instead of running `python3 sim.py run --watch`, you run `elodin run sim.py`.
- Add locking axis when panning and zooming plots.
  Now you can just move the x or y axis when panning or zooming a plot. Hold down Ctrl to only move along the x axis, and hold down Shift to only move along the y axis.
- Add ability to run processes alongside simulations.
  You can now add sidecar processes that are started alongside the main simulation. This is super useful for testing out flight software alongside your simulation.
  ```python
  world.recipe(el.s10.Recipe.Process(name = "test", cmd = "yes", args = [], cwd = None, env = {}, restart_policy = el.s10.RestartPolicy.Never, no_watch = False)) # to run an arbitrary process

  world.recipe(el.s10.Recipe.Cargo(name = "test", path = "./test-crate", args = [], cwd = None, env = {}, restart_policy = el.s10.RestartPolicy.Never, package = None, bin = None)) # to build and run a crate
  ```

## v0.6

### v0.6.2
- **(fix)** Various WASM editor bug fixes.

## v0.5

### v0.5.0
- **(fix)** Fix inconsistent axes colors in the editor. Now, +X is red, +Y is green, and +Z is blue across both the navigation widget and the grid lines.
- Respect `.gitignore` when uploading simulation artifacts for monte carlo runs. Also, ignore hidden files and directories.

## v0.4

### v0.4.1
- Add local cache for remote GLB assets to improve model load times and reduce network usage.
- Add more built-in templates for creating new simulations (`ball` and `drone`).

### v0.4.0
- **(fix)** Fix cutoff titlebar in editor on browser and windows.
- **(fix)** Fix bug in the drone example's rate controller where the derivative term wasn't filtered correctly (https://github.com/elodin-sys/elodin/issues/18). It effectively caused the integral gain to amplified and the derivative gain to be suppressed.
- **(breaking)** Replace `el.Time` and `el.advance_time` with `el.SimulationTick` and `el.SimulationTimeStep`.
  - The simulation tick is automatically advanced by a built-in system.
  - The simulation time step is set to the same value as the "sim_time_step" provided in `World.run`.
- Add a built-in `SystemGlobals` archetype that is automatically spawned for every simulation.
  - This archetype is only associated with a single managed entity and contains global variables that can be accessed by any system.
  - Currently, it contains `el.SimulationTick` and `el.SimulationTimeStep` which are automatically set to the current tick and simulation time step respectively.
- Add icon for the editor on windows.
- Add support for panning and zooming plots in the editor.
- Enable vsync on Windows and Linux.
- Don't throttle FPS when cursor is not moving on Windows and Linux.

## v0.3

### v0.3.30
- **(fix)** Fix issue where `edge_fold` would not filter out edges where the "right" entity doesn't match the "right" query.

### v0.3.29
- Support adding `el.SpatialMotion` to `el.SpatialTransform`.
- Add body-frame quaternion integration to `el.Quaternion` via `integrate_body`.

### v0.3.28
- Various bug fixes.

### v0.3.27
- **(fix)** Set the correct type information for the return value of `el.Quaternion` `__matmul__`. Previously, it would always return a `jax.BatchTracer` even if a spatial type was being multiplied.
- **(fix)** Fix issue where CLI is unable to verify the license key.

### v0.3.26
- Improve simulation startup times.

### v0.3.25
- **(breaking)** Use variable positional arguments for `el.Panel.vsplit`, `el.Panel.hsplit`, `el.Panel.graph`, and `el.GraphEntity` instead of lists. This results in much less verbose panel configuration code. To update your code, either replace the list of arguments with individual arguments separated by commas or unpack the list with `*`.
  ```python
  # before:
  world.spawn(
      el.Panel.hsplit(
          [
              el.Panel.vsplit(
                  [
                      el.Panel.graph(
                          [
                              el.GraphEntity(
                                  sat,
                                  [
                                      el.Component.index(SunRef),
                                  ],
                              )
                          ]
                      ),
                  ]
              ),
          ]
      ),
  )
  world.spawn(
      el.Panel.graph(
          [el.GraphEntity(b, el.Component.index(el.WorldPos)[4:]) for b in balls]
      ),
  )

  # after:
  world.spawn(
      el.Panel.hsplit(
          el.Panel.vsplit(
              el.Panel.graph(el.GraphEntity(sat, el.Component.index(SunRef))),
          ),
      ),
  )
  world.spawn(
      el.Panel.graph(
          *[el.GraphEntity(b, *el.Component.index(el.WorldPos)[4:]) for b in balls]
      ),
  )
  ```
- **(fix)**: Fix editor crash on Windows due to:
  > Requested alpha mode PostMultiplied is not in the list of supported alpha modes: [Opaque]
- `GraphEntity` constructor now accepts component classes directly instead of `el.Component.index(ComponentClass)`. However, `el.Component.index` is still needed for slicing and indexing into individual elements of a component.
  ```python
  # before:
  el.GraphEntity(sat, el.Component.index(SunRef))

  # after:
  el.GraphEntity(sat, SunRef)
  ```
- Add better default names for viewports and graphs. E.g. Track: "\<entity
name\>" for viewports and "\<entity name\>: \<component name\>" for graphs.
- Add basic support for 3D plots/traces in the editor.
  - Trace an entity's position by spawning in `elodin.Line3d` assets:
    ```python
    # The entity id is required as the first positional argument.
    world.spawn(el.Line3d(rocket))

    # You can specify the color and width of the line.
    w.spawn(el.Line3d(rocket, line_width=5.0, color=el.Color(1.0, 0.0, 0.0)))

    # You can enable or disable perspective rendering, which makes the line scale with distance to the camera. This is enabled by default.
    w.spawn(el.Line3d(rocket, perspective=False))
    ```

### v0.3.24
- Decouple simulation and playback running. You can now pause and rewind the editor without pausing the simulation. You can also change the playback speed by using `output_time_step` on `WorldBuilder.run`. We are deprecating the `time_step` parameter and replacing it with `sim_time_step`. This is to disambiguate it with `run_time_step`, which allows you to control the amount of time elodin waits between ticks of the simulation. Setting `run_time_step` to `0.0` effectively lets you simulate maximum speed.
- Add `max_ticks` parameter to `WorldBuilder.run`. The simulation will run until the specified number of ticks is reached.
- Add body frame visualization option.
  To try it out, either open the command palette and type `Toggle Body Axes` or add the following code to your simulation file:
  ```python
    w.spawn(el.BodyAxes(entity, scale=1.0))
  ```
- Add the ability to create graphs and viewports entirely from the command palette. Try spawning a new graph by typing `Create Graph`, to see the new workflow.
- **(fix)** If there were pytest failures in a  Monte Carlo sample, the sample would still be reported as a pass. This is now fixed.
- **(fix)** Fix line pixelation in long-running plots.
- Add `SpatialTransform()` constructor that replaces `SpatialTransform.zero()`, `SpatialTransform.from_linear(linear)`, and `SpatialTransform.from_angular(quaternion)` by making use of optional arguments. The old methods are still available but deprecated.
  ```python
  # before
  world_pos = el.SpatialTransform.from_linear(jnp.array([0.0, 0.0, 1.0])) + el.SpatialTransform.from_angular(euler_to_quat(jnp.array([0.0, 70.0, 0.0])))

  # after
  world_pos = el.SpatialTransform(linear=jnp.array([0.0, 0.0, 1.0]), angular=euler_to_quat(jnp.array([0.0, 70.0, 0.0])))

  # before
  world_pos = el.SpatialTransform.zero()

  # after
  world_pos = el.SpatialTransform()
  ```
- Add `SpatialForce()` constructor that replaces `SpatialForce.zero()`, `SpatialForce.from_linear(linear)`, and `SpatialForce.from_torque(torque)` by making use of optional arguments. The old methods are still available but deprecated.
  ```python
  # before
  force = el.SpatialForce.from_linear(jnp.array([0.0, 0.0, 1.0])) + el.SpatialForce.from_torque(jnp.array([0.0, 70.0, 0.0]))

  # after
  force = el.SpatialForce(linear=jnp.array([0.0, 0.0, 1.0]), torque=jnp.array([0.0, 70.0, 0.0]))

  # before
  force = el.SpatialForce.zero()

  # after
  force = el.SpatialForce()
  ```
- Deprecate `SpatialTransform.from_axis_angle(axis, angle)` in favor of `SpatialTransform(angular=Quaternion.from_axis_angle(axis, angle))`.
- Deprecate `SpatialMotion.from_linear(linear)` and `SpatialMotion.from_angular(angular)` in favor of `SpatialMotion(linear=linear, angular=angular)`.
- Add filtered search for entities and components.
- Always include zero as a tick in the y-axis of plots.
- Replace "Welcome" panel with a new UI for creating viewports and graphs.
- Add "Save Replay" command to command palette (Cmd + P).
- Show progress bar when executing `exec.run(ticks)` unless explicitly disabled with `exec.run(ticks, show_progress=False)`.
- Add a create command for templates to the Elodin CLI
- Allowing naming of viewports and graphs from code.
  ```python
  el.Panel.graph(
    [ el.GraphEntity(drone, [ el.Component.index(mekf.AttEstError) ]) ],
    name="Attitude Estimation Error",
  )
  ```

### v0.3.23
- **(breaking)** Render the Z-axis as up in the editor (instead of the Y-axis). This is a purely visual change and does not affect simulation results, but it's recommended to update simulations to match the new visual orientation. Typically, this requires swapping the Y and Z axes when operating on spatial positions, velocities, and forces.
- **(fix)** When a simulation file was changed, the associated pycache files would also be updated, causing the simulation to be re-built multiple times in some cases. This is now fixed.
- Add Status Bar to the editor (currently shows FPS/TPS and basic version of the connection status).
- `elodin editor <path/to/sim>` now watches the parent directory of the simulation file for changes in addition to the file itself. This is useful for multi-file projects. This is also the case when using the `--watch` flag directly. E.g. `python <path/to/sim> run --watch`.
- Export simulation data to a directory using `exec.write_to_dir(path)`.

### v0.3.22
- **(fix)** Fix errors when using `vmap` with `scan`, and non-scalar values
    - When the arguments to a scan operation were non-scalar values (i.e their rank was above 0), scan would error in various ways when combined with vmap. The core issue is that some of our logic accidentally assumed an empty shape array, and when that array was non-empty, dimensions would be inserted into the wrong place.

### v0.3.21
- **(fix)** Fix missing 1/2 factor in angular velocity integration and `Quaternion::from_axis_angle` .
    - In `nox`, the constant `Field::two` returned a `1` constant. This constant was only used in the implementation of `Add` between `SpatialMotion` and `SpatialTransform` and in `Quaternion::from_axis_angle`. Unfortunately, this caused angular velocity integration to return incorrect results. This bug caused the applied angular velocity to be multiplied by a factor of 2.
    - The most significant impact of this bug is on the stability of any attitude control system. This bug has led to an increase in small oscillations, potentially affecting the performance of PID controllers tuned to work with previous versions of Elodin. PID controllers tuned to work with earlier versions of Elodin will likely need to be re-tuned
    - We regret not catching this bug earlier. To prevent a bug like this happening ever again, we have taken the following steps:
        1. We created a set of unit tests for the 6 DOF implementation that compare Elodin's results with Simulink. They have confirmed that our implementation now matched Simulink's [6DOF Quaternion block](https://www.mathworks.com/help/aeroblks/6dofquaternion.html).
        2. We will expand our set of unit tests to have 100% coverage of our math modules. We will test each module against Simulink and other trusted implementations.
        3. We will publish documentation on our testing methodology and reports for each module.
- **(fix)** Fix incorrect angular acceleration due to mismatched coordinate frames.
    - Spatial force and motion are defined in the world frame. The inertia tensor, however, is defined in the body frame. To correctly calculate angular acceleration, torque and inertia tensor must be in the same frame. To fix this, we now transform torque from the world frame to the body frame before calculating angular acceleration. Then, we transform the angular acceleration back to the world frame.
- **(breaking)** Split `el.Pbr` into `el.Mesh`, `el.Material`, `el.Shape`, and `el.Glb`.
  - Use the `el.shape` and `el.glb` helpers instead
    ```python
    # before:
    w.spawn(el.Body(pbr = w.insert_asset( el.Pbr(el.Mesh.sphere(0.2), el.Material.color(0.0, 10.0, 10.0)))))
    w.spawn(el.Body(w.insert_asset(el.Pbr.from_url("https://storage.googleapis.com/elodin-assets/earth.glb"))))
    # after
    w.spawn([
      el.Body(),
      w.shape(el.Mesh.sphere(0.2), el.Material.color(0.0, 10.0, 10.0))
    ])
    w.spawn([
      el.Body(),
      w.glb("https://storage.googleapis.com/elodin-assets/earth.glb")
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

### v0.3.20

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

### v0.3.19

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

### v0.3.18

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

### v0.3.17

- Reduce memory usage of plots.
- Add rocket example with thrust curve interpolation.
- Clear graphs on sim update.
- Have a stable ordering of components in inspector.
- Make decimal point stable in component inspector.
- Add a built-in Time component + system.
- Remember window size on restart.
- Add configurable labels for component elements.
