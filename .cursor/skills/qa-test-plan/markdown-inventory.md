# Repository Markdown Inventory

> Every markdown document in the Elodin repository (excluding `ai-context/`), with location and a one-line summary.
> Generated 2026-07-10 at `822eb89a9` (main) as groundwork for QA test planning.
> Companion document: [`feature-catalog.md`](feature-catalog.md) — the itemized feature/workflow catalog derived from these sources.

**Total: 163 documents** (162 files + `CLAUDE.md`, a symlink to `AGENTS.md`).

## Root

- `README.md` — Monorepo overview: component map (flight software, nox-py, editor), Nix/`just install` setup, running examples, macOS/Linux manual build alternatives.
- `AGENTS.md` — Contributor/agent rules: `nix develop`, `uv`, CI checks, and an index of `.cursor/skills/` guidance by product area.
- `CLAUDE.md` — Symlink to `AGENTS.md` (identical content).
- `CHANGELOG.md` — Versioned release notes (v0.3–v0.17+) covering features, fixes, and breaking changes across editor, DB, SDK, and Aleph.
- `FAQ.md` — Fix for the Nix "ignoring untrusted substituter" warning via `trusted-users` and Elodin binary cache configuration.
- `VIDEOS.md` — Index of YouTube explainer videos (dev environment intro, Tracy profiling) with the format for adding new entries.

## CI & Build Scripts

- `.buildkite/README.md` — Buildkite EC2 agent bootstrap: overlayroot, S3 Nix cache signing, x86/ARM queues, deploy keys, troubleshooting.
- `scripts/ci/README.md` — Simulation regression gate `regress.sh`: benchmark, DB export, and CSV/profile-metric comparison against baselines for five examples.

## Agent Skills & Commands (`.cursor/`)

- `.cursor/commands/optimize-glb.md` — Cursor command workflow: inspect a GLB, run `scripts/optimize-glb.sh`, verify Bevy-safe output, visually QA.
- `.cursor/skills/bevy/SKILL.md` — Bevy ECS performance patterns for editor code: `Local<T>`, query filters, `run_if`, Events vs Messages, `bevy_defer`.
- `.cursor/skills/elodin-aleph/SKILL.md` — Aleph Jetson Orin workflows: `deploy.sh`, SD image, FSW service modules, STM32 flash/reset, on-vehicle elodin-db, SITL/HITL.
- `.cursor/skills/elodin-cranelift/SKILL.md` — Cranelift-MLIR JIT backend development: `ELODIN_BACKEND`, adding ops, checkpoint tests, regression baselines, debug env vars.
- `.cursor/skills/elodin-db/SKILL.md` — Elodin-DB usage: run/lua/merge/trim/follow, Impeller2 clients (C/C++/Rust/Python), replication, editor connection.
- `.cursor/skills/elodin-dev/SKILL.md` — Monorepo architecture and contributor setup: `nix develop`, `just install`, CI checks, workspace crate map.
- `.cursor/skills/elodin-dev/ci-checks.md` — Local CI mirror commands: cargo fmt/test/clippy, ruff, alejandra, plus the Apollo Monte Carlo Buildkite step.
- `.cursor/skills/elodin-editor-dev/SKILL.md` — Editor development: cargo run/watch, env vars, Bevy/Egui layout, KDL schematics, plots, video, command palette.
- `.cursor/skills/elodin-monte-carlo/SKILL.md` — Truth-data Monte Carlo workflows: vendoring reference profiles, truth ghosts, spec/campaign TOML, scoring, calibration loop.
- `.cursor/skills/elodin-nix/SKILL.md` — Nix dev shell, binary cache, trusted users, OrbStack remote Linux builds, Alejandra formatting, troubleshooting.
- `.cursor/skills/elodin-simulation/SKILL.md` — Python SDK simulation authoring: components/systems/6DOF, backends, SITL/HITL, gravity models, regression, Monte Carlo basics.
- `.cursor/skills/elodin-simulation/api-reference.md` — Quick API reference: `World.run/build/to_jax`, `StepContext`, components, spatial types, queries, panels, sim CLI.
- `.cursor/skills/elodin-simulation/examples.md` — Annotated example patterns: ball, three-body, rocket, cube-sat, Betaflight SITL, drone, RC jet, video stream.
- `.cursor/skills/elodin-tracy/SKILL.md` — Tracy profiling on Linux: `just install tracy`, per-process ports, capture/export, DB bench, custom instrumentation.
- `.cursor/skills/gltf-asset-optimization/SKILL.md` — Shrinking LFS-tracked GLBs for Bevy 0.18 (no Draco/meshopt): inspect, decimate, verify.
- `.cursor/skills/nox-py-dev/SKILL.md` — nox-py internals: PyO3 binding architecture, `WorldExec`, system compilation, adding components/decorators, pytest.
- `.cursor/skills/qa-test-plan/SKILL.md` — Agentic QA plan authoring/instantiation/execution: case anatomy, ID areas, execution modes, plan locations under `.cursor/skills/qa-test-plan/`.
- `.cursor/skills/qa-test-plan/template.md` — Reusable release QA plan template with SDK/SIM/DB/EDITOR/RUST/LINT cases, execution rules, run summary.

## Documentation Site & Internal Docs (`docs/`)

- `docs/internal/editor-cache-refactor-plan.md` — Draft plan to decouple editor playback from Elodin DB via a local TelemetryCache and PlaybackController.
- `docs/internal/nix.md` — Nix flakes dev shell setup and OrbStack VM remote-build configuration for building Linux binaries from macOS.
- `docs/internal/release.md` — Maintainer release workflow: changelog curation, versioning, `just public-changelog`, tagging, post-release alpha bumps.
- `docs/memserve/README.md` — Historical in-memory static HTTP docs server; superseded by the Zola-based docs site.
- `docs/public/README.md` — Public docs site stack (Zola, Tera, AdiDoks theme) and local `zola serve` development workflow.
- `docs/public/DEPLOYMENT.md` — Docs deployment: GitHub Pages auto-deploy, PR preview URLs, manual workflow dispatch, content/asset editing guide.
- `docs/public/content/_index.md` — Docs site root redirect to `/home` with top-level menu entries.
- `docs/public/content/ES01/_index.md` — Redirect to the ES01 Aleph flight-computer hardware datasheet PDF.
- `docs/public/content/home/_index.md` — Home section overview redirecting to the Quick Start tutorial.
- `docs/public/content/home/quickstart.md` — Install Elodin CLI/SDK, run the three-body example, Windows/WSL networking, ball plot analysis.
- `docs/public/content/home/3-body.md` — Tutorial: stable three-body orbits with gravity edges, `six_dof`, gizmos, Broucke configurations.
- `docs/public/content/home/bouncing-ball.md` — Tutorial: bouncing ball with gravity, `jax.lax.cond` bounce, wind drag, system pipelining, multi-file layout.
- `docs/public/content/home/apollo-lander.md` — Tutorial: Apollo 11 SITL Monte Carlo campaign with `post_step` bridge, hooks, scoring, calibration.
- `docs/public/content/home/aleph/_index.md` — Aleph docs section redirect to the setup guide.
- `docs/public/content/home/aleph/setup.md` — Aleph unboxing: power, serial/SSH access, sensor curl streams, Editor connection.
- `docs/public/content/home/aleph/reset.md` — Factory reset: carrier board via NixOS SD image; expansion-board RP2040 debugprobe flashing.
- `docs/public/content/home/aleph/betaflight.md` — Flash patched Betaflight firmware with probe-rs, configure via app.betaflight.com; reference quad BOM.
- `docs/public/content/home/aleph/fc_board_flash.md` — Flash STM32 expansion firmware three ways: probe-rs, `deploy.sh` service, or onboard `flash-mcu`.
- `docs/public/content/home/db/_index.md` — Elodin DB docs section redirect to overview.
- `docs/public/content/home/db/overview.md` — Elodin DB quick start: Lua REPL, SQL, replay, follow, export, trim, merge, offline tools.
- `docs/public/content/home/db/architecture.md` — DB internals: ECS/tensor data model, VTables, postcard messages, nanosecond timestamps, on-disk layout.
- `docs/public/content/home/db/python-client.md` — `elodin.db` Python client: writing/reading telemetry, streams, SQL, viewport integration via KDL.
- `docs/public/content/home/tao/_index.md` — "Tao of Elodin Sim" section redirect to introduction.
- `docs/public/content/home/tao/introduction.md` — Platform philosophy overview: ECS physics, Monte Carlo, 3D viewer, Impeller protocol.
- `docs/public/content/home/tao/ecs.md` — ECS history and the Cranelift/JAX-based configurable physics engine design rationale.
- `docs/public/content/home/tao/jax-nox.md` — NumPy/JAX/Nox backend lineage: JIT, GPU, and JAX sharp-bits for simulations.
- `docs/public/content/home/tao/data-flow.md` — Impeller protocol for entity/component telemetry exchange between simulation and flight software.
- `docs/public/content/home/tao/monte-carlo.md` — Monte Carlo testing motivation and Elodin's cloud parallel campaign approach.
- `docs/public/content/reference/_index.md` — Reference section redirect to overview.
- `docs/public/content/reference/overview.md` — Simulation architecture reference: entities, archetypes, systems, World, Impeller-linked FSW.
- `docs/public/content/reference/python-api.md` — Full Python SDK reference: World, 6DoF, systems, StepContext, Monte Carlo, `elodin.db`.
- `docs/public/content/reference/elodin-cli.md` — `elodin` and `elodin-db` CLI reference: every subcommand, flag, and Monte Carlo campaign option.
- `docs/public/content/reference/command-palette.md` — Editor command palette keys and all palette actions (panels, viewport, skybox, time, schematic, presets).
- `docs/public/content/reference/schematic.md` — KDL schematic reference: panels, viewport, graphs, `object_3d`, EQL, OBS video integration.
- `docs/public/content/reference/coords.md` — Coordinate conventions: ENU/NED/ECEF world frames, body frame; ECI/GCRF listed as not yet supported.
- `docs/public/content/reference/color-schemes.md` — Built-in and custom editor color schemes, JSON preset format, persistence.
- `docs/public/content/reference/db-asset-server.md` — DB asset server: persisting schematic assets into the DB, port N+1 HTTP serving, follow-mode sync.
- `docs/public/content/reference/replays.md` — Legacy replay directory layout (`metadata.json`, `assets.bin`, Parquet) for editor replay.
- `docs/public/content/reference/migration/_index.md` — Migration guides section redirect.
- `docs/public/content/reference/migration/to-0.15.md` — v0.14→v0.15 migration: Python UI panels/shapes to KDL schematics; removed APIs.
- `docs/public/content/releases/_index.md` — Releases section redirect to changelog.
- `docs/public/content/releases/changelog.md` — Public changelog (v0.3–v0.17) generated from the root `CHANGELOG.md`.

## Examples (`examples/`)

- `examples/apollo-lander/README.md` — Apollo 11 LM powered-descent SITL with Rust guidance controller, truth-replay ghost, Monte Carlo scoring/calibration, editor scenes.
- `examples/apollo-lander/WHITEPAPER.md` — Educational walkthrough of descent physics, reference-trajectory guidance, UDP SITL bridge, KDL scaling, calibration against real telemetry.
- `examples/ball/README.md` — Bouncing steel ball with gravity, drag, and wind; `elodin run` workflow plus optional Matplotlib trajectory plotting.
- `examples/betaflight-sitl/README.md` — 8 kHz quadcopter SITL running real Betaflight firmware via UDP lockstep, multi-rate sensors, portable DB replay, Aleph HITL pattern.
- `examples/crazyflie-edu/README.md` — UC Berkeley-style Crazyflie labs: portable C `user_code.c` runs in SITL, on hardware, and in `--hitl` mode with keyboard controls.
- `examples/crazyflie-edu/NOTES.md` — Research notes comparing course architectures and Crazyflie firmware integration options behind the SITL/HITL design.
- `examples/crazyflie-edu/labs/hwlab1.md` — Hardware lab 1: flash Crazyflie firmware from shared C code, HITL sensor visualization, propeller-off motor test.
- `examples/crazyflie-edu/labs/hwlab2.md` — Hardware lab 2: validate PWM→speed→force powertrain model with tachometer/thrust rigs; tethered hover.
- `examples/crazyflie-edu/labs/simlab1.md` — Sim lab 1: build the SITL binary, run the editor, drive motors from C code, analyze gyro/accel with and without noise.
- `examples/crazyflie-edu/labs/simlab2.md` — Sim lab 2: identify affine PWM→speed and quadratic speed→force models with virtual tachometer/force rig.
- `examples/cube-sat/README.md` — LEO CubeSat attitude-control simulation (MEKF, LQR, reaction wheels, EGM08) launched in the editor.
- `examples/cube-sat-pysim/README.md` — Python-only CubeSat variant using `World.to_jax` with Matplotlib output, run headless without the editor.
- `examples/db-client/README.md` — Standalone `elodin.db` Python client streaming a synthetic Crazyflie figure-8 into an embedded DB with a live editor schematic.
- `examples/drone/README.md` — Quadcopter/quadplane 6-DOF sim in ENU/FLU with cascaded PID, INDI gain-estimation notebook, `--telemetry` CSV export.
- `examples/ellipsoid/README.md` — Sensor-camera frustum vs ellipsoid intersection demo: dual viewports, coverage %, 2D far-plane projection, inspector toggles.
- `examples/frames/README.md` — Automated regression tests for gravity direction, inertial-frame equivalence, and energy conservation across ENU/NED/ECEF/ECI/GCRF.
- `examples/linalg/README.md` — Kalman-filter bench validating LAPACK-backed linear algebra via `elodin run` or bench mode with backend overrides.
- `examples/logstream/README.md` — C++ log client ingesting postcard-encoded FSW log messages into the DB for the editor log-viewer panel.
- `examples/monte-carlo/README.md` — Minimal Monte Carlo SITL campaign: UDP external controller, LHS sampling, port planning, scaling/memory profiling, hooks.
- `examples/n-body/README.md` — Solar-system N-body gravity with CSV truth overlays, RK4, backend benchmark matrix, accuracy reports.
- `examples/rc-jet/README.md` — BDX RC jet 6-DOF: polynomial aero, turbine/servo dynamics, Death Valley terrain, scripted flight plan, Rust gamepad/keyboard controller.
- `examples/rc-jet/BDX_Simulation_Whitepaper.md` — Technical design of the BDX jet model: stability-derivative aero, spool/thrust, actuators, ISA atmosphere, module architecture.
- `examples/rc-jet/sources/references.md` — Bibliography and integration plan for BDX modeling from JSBSim, XFLR5, UAV literature, turbojet identification.
- `examples/rocket/README.md` — 6-DOF rocket with Mach/AoA lookup-table aero, thrust curve, wind, pitch PID, EQL-derived plots, external fin trim control.
- `examples/rocket-barrowman/README.md` — Streamlit app + Elodin integration for Barrowman rocket design: ThrustCurve motors, AI builder, 3D trajectory visualization.
- `examples/rocket-barrowman/docs/AI_BUILDER_README.md` — Natural-language rocket designer: parses altitude/payload constraints, sizes components, selects motors, sizes parachutes.
- `examples/rocket-barrowman/docs/API_INTEGRATION.md` — ThrustCurve.org REST API and OpenAI API integration replacing web scraping for motor search and NLP parsing.
- `examples/rocket-barrowman/docs/ATMOSPHERIC_MODELS.md` — Atmospheric model options: ISA, NRLMSISE-00, NetCDF weather data, and hybrid low/high-altitude profiles.
- `examples/rocket-barrowman/docs/WEATHER_DATA.md` — Automatic ERA5/GFS weather fetch via Open-Meteo from lat/lon/datetime for wind profiles.
- `examples/rocket-barrowman/docs/WHITEPAPER.md` — Barrowman 6-DOF flight solver: RK4 integration, parachute triggers, AI optimizer phases, flight analysis suite.
- `examples/sensor-camera/README.md` — Entity-mounted synthetic RGB/thermal cameras via a headless GPU render server, read with `ctx.read_msg`, shown in `sensor_view` panels.
- `examples/three-body/README.md` — Stable periodic three-body orbit with custom gravity edges and graph queries; trail/velocity-vector overlays.
- `examples/video-stream/README.md` — Rolling-ball sim plus GStreamer test pattern, OBS SRT, and RTSP H.264 ingestion with editor tabs, replay, MP4 export.
- `examples/video-stream/HISTORY.md` — Architecture history of H.264 streaming via `elodinsink`, `MsgWithTimestamp`, fixed-rate playback, editor decoding (PR #67).
- `examples/voyager/README.md` — WIP Voyager 1/2 heliocentric simulation with SPICE truth trajectories; simulated probes don't yet reach Saturn.

## Flight Software (`fsw/`)

- `fsw/c-blinky/README.md` — Bare-metal STM32 LED-blink firmware: local build/flash, Aleph deploy service, COBS UART logs into `aleph.stm32.log`.
- `fsw/gstreamer/README.md` — `elodinsink` GStreamer plugin streaming Annex-B H.264 NAL units to Elodin-DB from test/file/webcam/GenICam sources.
- `fsw/mekf/README.md` — Aleph sensor pipeline spec: dual BMI270 + mag/baro/GPS over COBS/EL UART, serial-bridge DB components, MEKF attitude service.
- `fsw/msp-osd/README.md` — MSP DisplayPort OSD service mapping Elodin-DB telemetry to FPV goggles; terminal debug and serial VTX backends, Walksnail support.
- `fsw/roci/README.md` — Roci reactive flight-software framework: composable `System` trait, drivers, Impeller2 TCP, CSV logging, roci-adcs algorithms.
- `fsw/sensor-fw/README.md` — STM32H747 Aleph expansion firmware: high-rate IMU/mag/baro sampling, SD blackbox, UART streaming; probe-rs/openocd flashing.
- `fsw/udp_component_broadcast/README.md` — Python UDP bridge broadcasting Elodin-DB components between machines with protobuf, rename/filter, timestamp modes.
- `fsw/video-streamer/README.md` — FFmpeg-based utility re-encoding video files to AV1 OBUs and streaming them into Elodin-DB.

## Aleph & Apps

- `aleph/README.md` — Aleph NixOS guide: WiFi/USB SSH access, `deploy.sh` config deployment, STM32 firmware flashing, SD-image recovery, MCU logs.
- `apps/elodin/README.md` — Elodin Editor app: release install, local dev against nox-py, `ELODIN_ASSETS`/`ELODIN_KDL_DIR`, Blockade skybox generation.
- `apps/inscriber/README.md` — Cross-platform CLI to flash Aleph NixOS images to USB/SD with zstd decompression, interactive drive selection, progress UI.

## Libraries (`libs/`)

- `libs/bbqueue/README.md` — Vendored bbqueue 0.7 SPSC ring buffer bridging Impeller2 TCP I/O to the editor, patched for `usize` frame headers (8 MiB frames).
- `libs/bevy_geo_frames/README.md` — Bevy crate providing geographical coordinate frame types via `bevy_geo_frames::prelude`.
- `libs/bevy_mat3_material/README.md` — Bevy `MaterialExtension` applying 3×3 shear/scale vertex transforms with correct normal handling.
- `libs/bevy_world_mesh/README.md` — Large-scale terrain renderer: planar regions and spherical Earth from public DEM/imagery, fly camera, debug overlays.
- `libs/bevy_world_mesh/ARCHITECTURE.md` — UDLOD + chunked clipmap pipeline: tile fetching, GPU preprocessing, runtime paging, `big_space` precision, shaders.
- `libs/bevy_world_mesh/CHANGELOG.md` — world_mesh crate history: single-crate refactor, feature flags, Bevy 0.14–0.18 migrations.
- `libs/cranelift-mlir/README.md` — StableHLO MLIR→Cranelift JIT backend replacing IREE: regression benchmarks, op-add workflow, checkpoint debugging.
- `libs/cranelift-mlir/ARCHITECTURE.md` — cranelift-mlir design: compilation pipeline, dual ABI, SIMD, tensor runtime, LAPACK, gather, checkpoint bisection, op catalog.
- `libs/cranelift-mlir/PERFORMANCE.md` — Profiling via `ELODIN_CRANELIFT_DEBUG_DIR`: stderr reports, JSON profiles, waveform plots, Tracy, validation checklist.
- `libs/db/README.md` — Elodin-DB usage: run/config, C/C++ clients, Lua REPL, editor connection, follow-mode replication, merge/trim, C++ header generation.
- `libs/db/eql/README.md` — EQL time-series query language: hierarchical components, time windows, formulas; compiles to SQL.
- `libs/db/eql/src/formulas/README.md` — Guide to implementing EQL formulas: `EqlFormula` trait, SQL-primitive vs DataFusion UDF registration patterns.
- `libs/db/examples/README.md` — C/C++ DB client examples: batched vs per-component patterns, `db.hpp` wire protocol notes, benchmarking guidance.
- `libs/db/examples/rust_client/README.md` — Rust DB client example: component auto-discovery, schema retrieval, real-time TUI dashboard, bidirectional external control.
- `libs/db/examples/rust_client/CHANGELOG.md` — Rust client version history: host:port CLI, external-control dirty-flag fix, deprecated future-timestamp workaround.
- `libs/elodin-editor/src/plugins/README.md` — Index of editor Bevy plugins with first-appearance dates, active/legacy status, doc links.
- `libs/elodin-editor/src/plugins/asset_cache/README.md` — `AssetCache` trait with ETag storage; `FsCache` (native) and `NoCache` (WASM) factories for HTTP asset caching.
- `libs/elodin-editor/src/plugins/camera_anchor/README.md` — `camera_anchor_from_transform` computes safe view-to-origin anchors, returning `None` on invalid transforms.
- `libs/elodin-editor/src/plugins/editor_cam_touch/README.md` — Touch gestures for `EditorCam`: one-finger orbit, two-finger pan/pinch zoom, gated to the active viewport.
- `libs/elodin-editor/src/plugins/env_asset_source/README.md` — Registers the default Bevy asset source from `ELODIN_ASSETS` with fallback to `./assets` and path warnings.
- `libs/elodin-editor/src/plugins/frustum/README.md` — KDL-driven viewport frustum overlays with per-source color/thickness, clipping, cross-viewport visibility.
- `libs/elodin-editor/src/plugins/frustum_intersection/README.md` — Frustum∩ellipsoid volume coverage (%) and 2D far-plane projection meshes, gated by inspector prerequisites.
- `libs/elodin-editor/src/plugins/gizmos/README.md` — Vector arrows, body axes, and label rendering on `GIZMO_RENDER_LAYER`; navigation-cube UX moved to `view_cube`.
- `libs/elodin-editor/src/plugins/kdl_asset_source/README.md` — Registers the `kdl` Bevy `AssetSource` from `ELODIN_KDL_DIR` with optional `FileWatcher` hot-reload.
- `libs/elodin-editor/src/plugins/kdl_document/README.md` — KDL schematic lifecycle: load/save/reload messages, multi-window document tree, allowed schema nodes.
- `libs/elodin-editor/src/plugins/logical_key/README.md` — Tracks logical keyboard keys across frames in `pressed`/`just_pressed`/`just_released` sets.
- `libs/elodin-editor/src/plugins/navigation_gizmo/README.md` — Legacy navigation gizmo; render-layer allocation and camera sync still used by the Cube-Viewer.
- `libs/elodin-editor/src/plugins/view_cube/README.md` — CAD-style Cube-Viewer: face/edge/corner snapping, rotation arrows, reset/zoom buttons, ENU axes overlay.
- `libs/elodin-editor/src/plugins/web_asset/README.md` — Registers `http`/`https` Bevy asset sources with ETag-aware download caching.
- `libs/elodin-macros/README.md` — Derive macros wiring Rust structs into nox/nox-py: `Component`, `Archetype`, `ComponentGroup`, `IntoOp`/`FromOp`, `FromBuilder`, `ReprMonad`.
- `libs/hamann-chen-line/README.md` — Hamann–Chen curvature-based polyline simplification for 2D/3D/time-series; editor `CurveCompressSettings` integration.
- `libs/impeller2/README.md` — Impeller2 pub-sub telemetry protocol: hierarchical components, Table/Message/TimeSeries packets, VTables, sub-crates, transports.
- `libs/impeller2/kdl/README.md` — KDL serdes for `Schematic`: coordinate frames, panel nodes, viewport/graph/query UI, 3D scene nodes, serialization defaults.
- `libs/nox/README.md` — Core Nox tensor engine: tensor types, symbolic IR, differentiable primitives, Cranelift/JAX backends, ecosystem map.
- `libs/nox/array/README.md` — Zero-copy `ArrayView` for n-dimensional tensors in `no_std`, plus NumPy/JAX-style dynamic broadcasting rules.
- `libs/nox/src/noxpr/README.md` — noxpr subsystem for building typed tensor compute graphs in Rust and lowering them to JAX/StableHLO.
- `libs/nox-frames/README.md` — Compile-time-safe coordinate frames, poses, and Earth transforms (ECI/ECEF/NED) with time-aware composition.
- `libs/nox-py/README.md` — Python SDK reference: ECS simulation API, execution modes, profiling, DB/HITL integration, gravity models, examples.
- `libs/nox-py/python/elodin/FSW Workshop 2025 abstract.md` — Workshop abstract on vectorizing the EGM2008 gravity model via JAX for real-time execution.
- `libs/postcard-c/README.md` — Postcard wire-format C/C++ codegen and header-only runtime for Rust↔C telemetry, HITL, Impeller2 integration.
- `libs/postcard-c/codegen/examples/README.md` — Example codegen workflow: generate C++ bindings from RON, build with C++23, encode/decode round-trip.
- `libs/s10/README.md` — S10 TOML recipe orchestrator: sim/cargo/process/group recipes, watch mode, readiness probes, editor/Python integration.
- `libs/stellarator/README.md` — Deterministic single-threaded async runtime for flight software: io_uring/polling I/O, serial, structured concurrency.
- `libs/stellarator/maitake/README.md` — Modular `no_std` async runtime construction kit: tasks, schedulers, timer wheel, sync primitives, custom storage.
- `libs/stellarator/maitake/sync/README.md` — Async `no_std` synchronization primitives: Mutex, RwLock, Semaphore, WaitCell/Queue/Map, blocking locks.
- `libs/stellarator/maitake/sync/CHANGELOG.md` — maitake-sync release history: crate split, spin RwLock, lock helpers, `wait_for`/`is_closed` additions.
- `libs/stellarator/maitake/util/README.md` — Mycelium utility types for kernel use: `CachePadded` and feature flags.
- `libs/video-toolbox/README.md` — Apple VideoToolbox H.264 decoder wrapper for live camera streams from Aleph via Elodin DB to editor display.
- `libs/wmm/README.md` — NOAA WMM 2020 Rust wrapper: geodetic magnetic field, declination/inclination/intensity, ADCS/MEKF integration.

## Coverage Notes (directories without markdown docs)

These components exist in the tree but have no markdown documentation; their behavior is documented indirectly (skills, sibling READMEs) or not at all — worth noting for the coverage effort:

- `examples/`: `geo-frames`, `stablehlo`, `linalg-iree` have no README.
- `fsw/`: `aleph-setup`, `aleph-status`, `blackbox`, `lqr`, `openocd`, `rtsp-streamer`, `serial-bridge`, `tegrastats-bridge` have no README (serial-bridge and blackbox are partially covered by `fsw/mekf/README.md` and `fsw/sensor-fw/README.md`).
- `libs/`: `bevy_ai_skybox`, `build-common`, `monte-carlo` (campaign runner crate), `rtsp-ingest`, and the top-level `elodin-editor` crate have no README (monte-carlo is covered by CLI/docs-site reference; rtsp-ingest by `examples/video-stream/README.md`).
