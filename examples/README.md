# Examples

These examples show how to use Elodin — physics, vehicles, editor objects, and
integrations. An example can appear in more than one section.

Run from the repository root, inside `nix develop` (after `just install`):

```sh
elodin editor examples/<name>/main.py   # 3D editor
elodin run examples/<name>/main.py      # headless
```

Some folders have their own `README.md` with extra setup (SITL binaries, terrain
atlases, Monte Carlo campaigns). A few are headless checks, not editor scenes:
`frames`, `linalg`, `stablehlo`, `cube-sat-pysim`.

---

## Table of Contents

- [Start here](#start-here)
- [By domain](#by-domain)
  - [Drones & aircraft](#drones--aircraft)
  - [Rockets & launch vehicles](#rockets--launch-vehicles)
  - [Spacecraft & landers](#spacecraft--landers)
  - [Orbital & n-body dynamics](#orbital--n-body-dynamics)
  - [Physics fundamentals](#physics-fundamentals)
  - [Coordinate frames & geodesy](#coordinate-frames--geodesy)
  - [Software-in-the-loop](#software-in-the-loop)
  - [Editor scenes & visualization](#editor-scenes--visualization)
  - [Telemetry & external clients](#telemetry--external-clients)
  - [Compiler & numerics](#compiler--numerics)
- [By Elodin object](#by-elodin-object)
  - [World, Body, six_dof](#world-body-six_dof)
  - [Component / Archetype](#component--archetype)
  - [GraphQuery / Edge](#graphquery--edge)
  - [StepContext](#stepcontext)
  - [s10 recipes](#s10-recipes)
  - [Monte Carlo](#monte-carlo)
  - [Coordinate frames](#coordinate-frames)
  - [Gravity models](#gravity-models)
  - [Execution modes & entry points](#execution-modes--entry-points)
  - [KDL: viewport](#kdl-viewport)
  - [KDL: object_3d / glb](#kdl-object_3d--glb)
  - [KDL: graph](#kdl-graph)
  - [KDL: line_3d](#kdl-line_3d)
  - [KDL: vector_arrow](#kdl-vector_arrow)
  - [KDL: thruster](#kdl-thruster)
  - [KDL: world_mesh](#kdl-world_mesh)
  - [KDL: ellipsoid](#kdl-ellipsoid)
  - [sensor_camera / sensor_view](#sensor_camera--sensor_view)
  - [Gauges / monitors](#gauges--monitors)
  - [Video / logs](#video--logs)
  - [Truth-replay ghosts](#truth-replay-ghosts)
  - [Cranelift / compiler internals](#cranelift--compiler-internals)

---

## Start here

| Example | Description |
| --- | --- |
| [ball](./ball) | Gravity, wind, bounce — the smallest full `six_dof` sim |
| [rotating-cube](./rotating-cube) | One spinning `Body` plus editor gauges |

---

## By domain

### Drones & aircraft

| Example | Description |
| --- | --- |
| [drone](./drone) | Quadcopter / quadplane, motor mixing, cascaded PID, MEKF |
| [rc-jet](./rc-jet) | Turbine RC jet: polynomial aero, ADI gauge, FPV camera, Death Valley terrain |
| [betaflight-sitl](./betaflight-sitl) | Real Betaflight PID loop in lockstep over UDP |
| [crazyflie-edu](./crazyflie-edu) | Crazyflie labs — same C controller in SITL and HITL |
| [db-client](./db-client) | Synthetic Crazyflie written with the standalone `elodin.db` client |
| [ellipsoid](./ellipsoid) | Drone GLB flying inside an ellipsoid (frustum coverage) |

### Rockets & launch vehicles

| Example | Description |
| --- | --- |
| [rocket](./rocket) | 6DOF model rocket: thrust curve, aero tables, PID fins |
| [rocket-barrowman](./rocket-barrowman) | Barrowman stability + Streamlit design UI |
| [falcon9](./falcon9) | Falcon 9 RTLS in ECEF, Rust FSW, truth ghost, Monte Carlo |

### Spacecraft & landers

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | Apollo 11 powered descent (P63/P64/P66), LGC-style SITL |
| [cube-sat](./cube-sat) | CubeSat ADCS: CSS, mag, gyro, MEKF, LQR, reaction wheels, EGM08 |
| [cube-sat-pysim](./cube-sat-pysim) | Lighter CubeSat via `World.to_jax` + Matplotlib |
| [voyager](./voyager) | Voyager 1/2 under planetary gravity vs SPICE truth |

### Orbital & n-body dynamics

| Example | Description |
| --- | --- |
| [three-body](./three-body) | Periodic three-body orbit via `GraphQuery` gravity edges |
| [n-body](./n-body) | Solar-system all-pairs gravity + CSV truth overlay |
| [voyager](./voyager) | Probes + planets; SPICE truth vs integrated trajectories |

### Physics fundamentals

| Example | Description |
| --- | --- |
| [ball](./ball) | Gravity, drag, wind, bounce — canonical single-entity physics |
| [rotating-cube](./rotating-cube) | One `Body` spinning at constant angular velocity |

### Coordinate frames & geodesy

| Example | Description |
| --- | --- |
| [frames](./frames) | Headless invariance checks: ENU vs NED, ECI vs GCRF |
| [geo-frames](./geo-frames) | ECEF / NED / ENU markers, trails, and axis arrows at Earth scale |
| [f32-quant-repro](./f32-quant-repro) | ECEF vs local ENU trails at Earth radius (display-side f32 cast) |

### Software-in-the-loop

An external flight-software process driven over UDP, alongside the sim.

| Example | Description |
| --- | --- |
| [betaflight-sitl](./betaflight-sitl) | Betaflight binary, lockstep at flight-controller rate |
| [crazyflie-edu](./crazyflie-edu) | C control code in SITL, then the same code on hardware (HITL) |
| [apollo-lander](./apollo-lander) | Rust LGC flying the descent, scored by a Monte Carlo campaign |
| [falcon9](./falcon9) | Rust FSW flying the full RTLS mission |
| [rc-jet](./rc-jet) | External RC / autopilot controller |
| [monte-carlo](./monte-carlo) | Minimal plant ↔ controller campaign with per-run scoring |

### Editor scenes & visualization

| Example | Description |
| --- | --- |
| [terrain](./terrain) | Planar `world_mesh` (Brienz atlas) |
| [sensor-camera](./sensor-camera) | Bouncing balls carrying onboard RGB / thermal cameras |
| [ellipsoid](./ellipsoid) | Frustum ∩ ellipsoid coverage and far-plane projection |
| [covariance-ellipsoids](./covariance-ellipsoids) | Cholesky factor vs direct `P` covariance |

### Telemetry & external clients

| Example | Description |
| --- | --- |
| [video-stream](./video-stream) | GStreamer H.264 into Elodin DB + editor overlay |
| [logstream](./logstream) | C++ log client → editor log panel |
| [db-client](./db-client) | Standalone `elodin.db`: embedded server, writers, live replay |

### Compiler & numerics

| Example | Description |
| --- | --- |
| [linalg](./linalg) | LAPACK-backed `jnp.linalg` through Cranelift |
| [stablehlo](./stablehlo) | StableHLO / CHLO op coverage through the JIT |

---

## By Elodin object

Same idea as Bevy's "2D Rendering / ECS / Shaders" pages: pick the primitive
you want to learn, then open the example that uses it.

### World, Body, six_dof

`el.World`, `el.Body` (`WorldPos` / `WorldVel` / `Inertia` / `Force`), and
`el.six_dof(...)`.

| Example | Description |
| --- | --- |
| [ball](./ball) | Smallest complete `six_dof` pipeline |
| [drone](./drone) | Thrust / drag effectors inside `six_dof`, motor mixing |
| [rocket](./rocket) | RK4 6DOF with aero + thrust effectors |
| [falcon9](./falcon9) | ECEF 6DOF booster, SemiImplicit integrator |
| [apollo-lander](./apollo-lander) | Lunar-gravity 6DOF + throttle / RCS effectors |
| [rotating-cube](./rotating-cube) | One `Body` with constant angular velocity |

### Component / Archetype

Custom `el.Component` + `@el.dataclass` archetypes beyond `el.Body`.

| Example | Description |
| --- | --- |
| [rocket](./rocket) | Thrust curve, fin deflection, aero coefficients |
| [cube-sat](./cube-sat) | Sensors, MEKF state, reaction-wheel archetypes |
| [drone](./drone) | Motor PWM/RPM, gyro / accel / mag sensors, MEKF estimate |
| [falcon9](./falcon9) | Propellant, TVC, grid fins, FSW phase |
| [apollo-lander](./apollo-lander) | Throttle, RCS, landed flags, derived nav |

### GraphQuery / Edge

`el.Edge`, `el.GraphQuery`, `edge_fold`, optional `el.RevEdge`.

| Example | Description |
| --- | --- |
| [three-body](./three-body) | Bidirectional gravity edges, `edge_fold` |
| [n-body](./n-body) | All-pairs solar-system gravity |
| [voyager](./voyager) | Gravity graph + SPICE-driven planet forces |
| [cube-sat](./cube-sat) | CSS sensors and reaction wheels attached by edges |

### StepContext

`pre_step` / `post_step` with `el.StepContext` (read/write components, lockstep).

| Example | Description |
| --- | --- |
| [betaflight-sitl](./betaflight-sitl) | Sensor out / motor in at flight-controller rate |
| [apollo-lander](./apollo-lander) | UDP bridge to the LGC-style controller |
| [falcon9](./falcon9) | Closed-loop SITL boundary with Rust FSW |
| [monte-carlo](./monte-carlo) | Plant ↔ controller over UDP, campaign ports |
| [voyager](./voyager) | SPICE kernel updates each tick |
| [n-body](./n-body) | Truth-index advance in `post_step` |

### s10 recipes

`el.s10.PyRecipe` — spawn an external process next to the sim.

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | Rust LGC controller (`process` / `cargo`) |
| [falcon9](./falcon9) | Rust FSW controller |
| [rc-jet](./rc-jet) | External RC / autopilot controller |
| [betaflight-sitl](./betaflight-sitl) | Betaflight SITL binary |
| [video-stream](./video-stream) | GStreamer + OBS / RTSP receivers |
| [logstream](./logstream) | C++ `log-client` into the DB |
| [monte-carlo](./monte-carlo) | `world.recipe(...)` + campaign worker ports |

### Monte Carlo

`el.monte_carlo.params_spec`, `params`, `result`, `port`, campaign TOML.

| Example | Description |
| --- | --- |
| [monte-carlo](./monte-carlo) | Canonical native campaign (hooks, scoring, shared constants) |
| [apollo-lander](./apollo-lander) | Descent-parameter campaign vs Apollo 11 truth |
| [falcon9](./falcon9) | LHS calibration vs recorded CRS flights |

### Coordinate frames

`coordinate frame=…` in KDL, ECEF / ENU / NED poses, geodetic gauges.

| Example | Description |
| --- | --- |
| [frames](./frames) | Physics invariance: ENU vs NED gravity, ECI vs GCRF n-body |
| [geo-frames](./geo-frames) | Same markers drawn in ECEF, NED, and ENU |
| [falcon9](./falcon9) | Full mission in ECEF + geodetic derived components |
| [rotating-cube](./rotating-cube) | ENU origin + NED / ECEF attitude gauges |
| [f32-quant-repro](./f32-quant-repro) | ECEF vs local ENU trails at Earth radius |

### Gravity models

| Example | Description |
| --- | --- |
| [cube-sat](./cube-sat) | `elodin.egm08.EGM08` spherical harmonics |
| [three-body](./three-body) / [n-body](./n-body) / [voyager](./voyager) | Inverse-square via `GraphQuery` |
| [apollo-lander](./apollo-lander) | Constant lunar g with orbital centrifugal relief |
| [ball](./ball) / [drone](./drone) / [rocket](./rocket) | Flat-world `−9.81` effector |

### Execution modes & entry points

Alternatives to the default `World.run(...)` on the Cranelift backend.

| Example | Description |
| --- | --- |
| [cube-sat-pysim](./cube-sat-pysim) | `World.to_jax` — pure JAX step loop + Matplotlib (RL-friendly) |
| [db-client](./db-client) | Standalone `elodin.db`: embedded server, `table_writer`, no tick loop |
| [n-body](./n-body) | `ELODIN_BACKEND` switch: `cranelift` / `jax-cpu` / `jax-gpu` |

### KDL: viewport

| Example | Description |
| --- | --- |
| [rotating-cube](./rotating-cube) | Single `look_at` viewport on the cube |
| [falcon9](./falcon9) | Cinematic chase (mission), landing / night-sky in `visual_check.kdl` |
| [rc-jet](./rc-jet) | Chase, FPV, top-down, target intercept |
| [geo-frames](./geo-frames) | ECEF equator + NED local views |
| [ellipsoid](./ellipsoid) | Dual viewports + frustum overlays |

### KDL: object_3d / glb

| Example | Description |
| --- | --- |
| [drone](./drone) / [betaflight-sitl](./betaflight-sitl) | Quad GLB |
| [rc-jet](./rc-jet) | Aircraft GLB |
| [falcon9](./falcon9) | Booster + pad + LZ-1 |
| [apollo-lander](./apollo-lander) | LM + Moon + Earth |
| [cube-sat](./cube-sat) | OreSat mesh |
| [n-body](./n-body) / [voyager](./voyager) | Procedural spheres for bodies |

### KDL: graph

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | Altitude, rates, pitch, throttle vs truth |
| [falcon9](./falcon9) | Speed, Mach, tanks, fins, TVC vs recorded |
| [cube-sat](./cube-sat) | CSS values + attitude estimate |
| [rc-jet](./rc-jet) | α, β, CL/CD, surfaces, spool |
| [rocket](./rocket) | EQL `query_plot` for angle of attack and speed |
| [voyager](./voyager) | Position / velocity error vs SPICE |

### KDL: line_3d

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | Blue sim trail + green truth trail |
| [falcon9](./falcon9) | Booster vs recorded trajectory |
| [n-body](./n-body) / [voyager](./voyager) | Orbital trails |
| [f32-quant-repro](./f32-quant-repro) | ECEF staircase vs smooth ENU |
| [geo-frames](./geo-frames) | Per-frame trails at Earth scale |
| [rc-jet](./rc-jet) | Flight-path ribbon |

### KDL: vector_arrow

| Example | Description |
| --- | --- |
| [rc-jet](./rc-jet) | Body-frame X/Y/Z axes |
| [n-body](./n-body) | Tiny arrows used as body labels |
| [geo-frames](./geo-frames) | NED / ENU / ECEF Y-axis arrows |

### KDL: thruster

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | DPS plume + 16 RCS puffs + ground dust |
| [falcon9](./falcon9) | Merlin core, exhaust smoke, 8 RCS darts, pad / LZ smoke |

### KDL: world_mesh

| Example | Description |
| --- | --- |
| [terrain](./terrain) | Brienz planar atlas (`world_mesh "brienz"`) |
| [rc-jet](./rc-jet) | Death Valley under the jet |

### KDL: ellipsoid

| Example | Description |
| --- | --- |
| [covariance-ellipsoids](./covariance-ellipsoids) | `error_covariance` vs `error_covariance_cholesky` |
| [ellipsoid](./ellipsoid) | Frustum ∩ ellipsoid coverage + far-plane projection |

### sensor_camera / sensor_view

`world.sensor_camera(...)` in Python, `sensor_view` in KDL.

| Example | Description |
| --- | --- |
| [sensor-camera](./sensor-camera) | Onboard camera, FPS / latency knobs |
| [ellipsoid](./ellipsoid) | Frustum creation + `sensor_view` next to 3D |
| [rc-jet](./rc-jet) | FPV camera (`create_frustum=False`) |

### Gauges / monitors

| Example | Description |
| --- | --- |
| [rotating-cube](./rotating-cube) | `geo_position_gauge`, `orientation_gauge`, `component_monitor` |
| [rc-jet](./rc-jet) | `horizon_gauge` ADI on `bdx.world_pos` (ENU body frame) |

### Video / logs

| Example | Description |
| --- | --- |
| [video-stream](./video-stream) | H.264 Annex-B into the DB, viewport overlay |
| [logstream](./logstream) | MSG log stream in the editor log panel |

### Truth-replay ghosts

Kinematic entity (often no `el.Body`) driven by recorded / ephemeris data.

| Example | Description |
| --- | --- |
| [apollo-lander](./apollo-lander) | `lander_truth` from reconstructed Apollo 11 descent |
| [falcon9](./falcon9) | `booster_truth` from CRS telemetry |
| [n-body](./n-body) | `truth_*` bodies from CSV ephemerides |
| [voyager](./voyager) | SPICE kernels for planets and real Voyagers |

### Cranelift / compiler internals

| Example | Description |
| --- | --- |
| [stablehlo](./stablehlo) | Every implemented StableHLO/CHLO op through the JIT |
| [linalg](./linalg) | Cholesky, solve, SVD, eigh, … on Cranelift |
| [f32-quant-repro](./f32-quant-repro) | Display-path f32 cast at ECEF magnitude |
