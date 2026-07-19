# Elodin Stack Feature Catalog

> Itemized catalog of every aspect, workflow, and use case currently supported in the Elodin stack, derived from all 163 markdown documents in the repository (see [`markdown-inventory.md`](markdown-inventory.md) for the source list).
> Generated 2026-07-10 at `822eb89a9` (main). Purpose: foundation for a QA test plan targeting 100% feature coverage.
> Section 20 lists historical/deprecated/WIP items that should be *excluded* from active test coverage.

---

## 1. Development Environment & Toolchain

- **Nix development shell (`nix develop`)** — Reproducible unified dev environment (Rust, Python, C/C++, GStreamer, docs tooling, git-lfs) used for all development; also `nix develop .#run` (run-focused shell) and one-off `nix develop --command ...` invocations.
- **Determinate Systems Nix install** — Recommended Nix installation path for dev machines and Aleph development.
- **Elodin S3 binary cache** — `elodin-nix-cache.s3.us-west-2.amazonaws.com` substituter with `trusted-users` configuration to speed up builds; misconfiguration produces the documented "ignoring untrusted substituter" FAQ symptom.
- **Just task runner** — `just install [py|editor|db|tracy|all]` builds and installs the Python SDK wheel (uv + maturin 1.12.6), `elodin` editor binary, and `elodin-db` binary into the cargo bin path; `just install tracy` builds all with Tracy profiling enabled (Linux only).
- **uv-based Python management** — All Python work uses `uv` inside the Nix shell; `just install py` creates `.venv` (Python 3.13) and examples run via `.venv/bin/python`.
- **git-lfs assets** — Large binary assets (GLBs, terrain) are LFS-tracked; `git lfs install` required at clone time.
- **Manual (non-Nix) setups** — Documented but discouraged alternatives: macOS via Homebrew (gstreamer, gfortran, openblas, uv, rust) and Ubuntu via apt (just, git-lfs, libasound2-dev, cmake, gfortran, patchelf), followed by the same `just install` / `uvx maturin develop` steps.
- **Cargo aliases** — `cargo elodin ...` / `cargo elodin-db ...` run the binaries from source.
- **Nix flake packages** — `elodin-py`, `elodin-cli`, `elodin-db`, `elodinsink` buildable from the root flake.
- **OrbStack NixOS VM remote builds** — macOS builds Linux binaries via an SSH build machine in `/etc/nix/machines`; includes the disk-backed `build-dir` workaround for Qt tmpfs exhaustion.
- **Version verification** — `elodin --version` and `elodin-db --version` report versions with git-hash suffix.
- **Supported platforms** — macOS, Linux glibc 2.35+, NixOS 21.11+; Windows runs the Editor natively (MSI) with the simulation in WSL (mirrored networking); `elodin run` and s10 `sim` recipes are not available on Windows.
- **End-user install scripts** — `curl`-able `elodin-installer.sh` and `elodin-db-installer.sh`, plus Windows MSI from the releases page.
- **Shell niceties** — Oh My Zsh + Powerlevel10k support in the Nix shell (`p10k configure`).

## 2. Build, CI & Quality Gates

- **Rust CI checks** — `cargo fmt`, `cargo test`, `cargo clippy -- -Dwarnings` across the workspace; `cargo test -p elodin-db` requires `CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 RAYON_NUM_THREADS=1` to avoid crashes.
- **Python CI checks** — `ruff format --check` and `ruff check --fix`.
- **Nix formatting** — `alejandra` (check mode `alejandra --check .`).
- **Buildkite CI infrastructure** — Self-hosted EC2 agents (overlayroot tmpfs, deploy keys) on `nixos-x86-aws` and `nixos-arm-aws` queues; post-build hook signs and uploads store paths to the S3 Nix cache.
- **Simulation regression gate** — `scripts/ci/regress.sh [--all|--update]` benchmarks baselined examples (ball, drone, rocket, three-body, cube-sat), exports DB telemetry to CSV, and compares against baselines via `compare_baseline_csv.py` (tolerance-based, ignores time column) and `compare_profile_metrics.py` (build_time_ms, real_time_factor, ticks) with per-example `baseline/tolerances.json`.
- **Apollo Monte Carlo CI** — Buildkite step running `scripts/test-apollo-monte-carlo.sh`: truncated-tick, fixed-sample campaign that validates MC infrastructure (not landing quality).
- **nox-py test suite** — `pytest libs/nox-py/tests/` unit/integration tests.
- **cranelift-mlir test suite** — 220+ unit tests, per-op golden tests (scalar + pointer ABI), full-example e2e MLIR tests, checkpoint verifier, and external customer-MLIR compile validation.
- **Windows cross-build** — `just build-windows-gnu` builds the editor for x86_64-pc-windows-gnu with mingw.

## 3. `elodin` CLI

- **`elodin editor <target>`** — Launch the editor against: a Python sim file, a DB address (`127.0.0.1:2240`, `[fde1:2240:a1ef::1]:2240`), an `s10.toml`, a directory containing `main.py`/`s10.toml`, or a legacy replay directory; optional `--kdl <file>` loads a schematic at startup.
- **`elodin run <target>`** — Headless simulation runner (same target types); keeps serving the DB after `max_ticks` (never exits on its own — QA must use bench mode for terminating runs).
- **`elodin monte-carlo` subcommands** — `quickstart` (scaffold spec/campaign/hooks from the sim's `params_spec()`), `template`, `sample` (regenerate LHS plan CSV), `run` (`--campaign`, `--spec`, `--out`, `--plan`, `--workers`, `--scratch-dir`, `--cache-dir`, `--clean`, `--memory-probe`, `--keep-existing`), `resume` (re-run missing/failed runs), `report` (rebuild reports).
- **CLI environment variables** — `BLOCKADE_API_KEY` (skybox AI generation), `ELODIN_ASSETS` (asset root incl. skyboxes), `ELODIN_KDL_DIR` (schematic search + hot-reload), `ELODIN_DB_PATH` (DB location override), `ELODIN_BACKEND` (execution backend).
- **Python sim file CLI** — Every sim script supports `python sim.py run` (default), `bench --ticks N [--profile] [--detail]` (terminating benchmark with timing summary), `components` (JSON dump of components/entities/shapes/metadata without running), `params` (JSON schema of MC parameters), and `plan <dir>` (generate s10 config).

## 4. Python SDK (nox-py) — World & Execution

- **`el.World()` container** — Holds all entities, components, systems, assets, and schematics for a simulation.
- **Entity spawning** — `w.spawn(archetypes, name=...)` with hierarchical dot-notation component naming (`drone.world_pos`); `w.insert()` for additional archetypes; `w.insert_asset()`, `w.shape()`, `w.glb()` for visual assets.
- **`w.run(system, ...)` flags** — `simulation_rate` (Hz, default 120), `telemetry_rate` (DB sync rate), `generate_real_time` (wall-clock pacing), `max_ticks`, `interactive` (keep serving after max_ticks), `db_path` / `db_addr` (embedded vs external DB), `start_timestamp` (signed µs; pre-1970 supported for historical missions), `default_playback_speed`, `optimize`, `log_level`, `is_canceled`, `pre_step`/`post_step`, `backend`.
- **`w.build(system)` → `Exec`** — Compile without running; `exec.run(ticks)`, `exec.history([components])` (joined on time), `exec.save_archive(path, format="arrow"|"parquet"|"csv")`.
- **`w.to_jax(system)`** — Pure-JAX simulation with `step`/`get_state`/`set_state`, compatible with `jax.vmap`/`jax.jit`; enables RL workflows (JAXMarl, Gym wrappers in `jaxsim.py`).
- **Execution backends** — `cranelift` (default; pure-Rust JIT, no GIL per tick), `jax-cpu` (full JAX op compatibility, XLA reference), `jax-gpu` (GPU offload for large entity counts); selected via `backend=` or `ELODIN_BACKEND`.
- **Embedded DB & source snapshot** — Default run creates a temp Elodin DB; persistent `db_path`/`ELODIN_DB_PATH` recordings capture project `.py` files under `{db}/simulation_source/` with a manifest.
- **Simulation profiler** — `bench --profile` adds complexity-weighted FLOP analysis, HLO breakdown, and Graphviz DOT graphs under `profile_output/graphs/`.
- **World frame selection** — `el.World(frame=...)` supports ENU (default), NED, ECEF, ECI, GCRF world frames.

## 5. Python SDK — Components, Systems & Physics

- **Custom components** — `ty.Annotated[jax.Array, el.Component(name, type, metadata)]` with `ComponentType`/`PrimitiveType` (F64, U64, ...); metadata keys: `element_names` (inspector/export labels), `private` (excluded from export unless `--include-private`), `external_control` (writable by external clients; sim never overwrites).
- **Archetypes** — `@el.dataclass class X(el.Archetype)` bundles components for spawning; built-in `el.Body` provides `WorldPos`, `WorldVel`, `Inertia`, `Force`, `WorldAccel`; archetypes without `el.Body` create kinematic (non-integrated) entities for ghosts/terrain/markers.
- **System decorators** — `@el.map` (vectorized per-entity), `@el.map_seq` (sequential; preserves `jax.lax.cond` short-circuit for conditional logic like bounces), `@el.system` (multi-query systems with `el.Query` inputs/outputs and `query.map`).
- **System composition** — Pipe operator `|` chains systems in execution order (effectors before `six_dof`, kinematic maps after).
- **Graph queries** — `el.GraphQuery` + `el.Edge` + `edge_fold` accumulate over entity-pair relationships (N-body gravity, springs, constraints).
- **6DOF integration** — `el.six_dof(time_step, effectors, integrator=el.Integrator.Rk4 | el.Integrator.SemiImplicit)` rigid-body dynamics; body-frame inertia, world-frame kinematics.
- **Spatial algebra** — `SpatialTransform`, `SpatialMotion`, `SpatialForce`, `SpatialInertia` (Featherstone), `el.Quaternion` (`identity`, `from_axis_angle`, `integrate_body`, `@` rotation).
- **Gravity models** — `elodin.j2.J2` (oblate Earth) and `elodin.egm08.EGM08(max_degree=N)` (spherical harmonics; coefficients auto-download on first use; documented latency tiers: degree 10 ≈ 0.1 ms up to degree 250 ≈ 2.5 ms).
- **Tick/time components** — `SimulationTick`, `SimulationTimeStep` available to systems (used e.g. by truth playback).

## 6. Python SDK — StepContext & SITL/HITL Integration

- **`pre_step` / `post_step` callbacks** — Plain-Python hooks each tick receiving `el.StepContext` for lockstep external I/O.
- **StepContext API** — `tick`, `timestamp`, `read_component` (with historical timestamp; floor/sample-and-hold semantics), `write_component` (custom timestamps; out-of-order writes raise `TimeTravel`), `component_batch_operation` (batch reads/writes under one lock), `read_msg(name, timestamp=...)` (e.g. latency-modeled camera frames), `truncate()` (clear DB, reset tick), `stop_recipes()` (graceful SIGTERM to s10 processes).
- **External control flow** — Components flagged `external_control` are written by FSW/clients over the DB (throttle, attitude setpoints, motor commands, fin trim) and never overwritten by the sim.
- **SITL connection patterns** — In-process `post_step` bridge (UDP fixed-layout f64 packets), s10-managed external process (`world.recipe()`), or networked Impeller2 client.
- **Lockstep SITL** — Physics blocks on flight-controller responses (e.g. Betaflight `SIMULATOR_GYROPID_SYNC` at 8 kHz).
- **HITL mode** — Real hardware in the loop (e.g. Crazyflie via cflib/Crazyradio with physics disabled; Aleph vehicle DB recordings).

## 7. Python SDK — Sensor Cameras

- **`world.sensor_camera()`** — Entity-mounted virtual camera with `width`, `height`, `fov`, mount position/orientation offsets, `fps`, and GPU post-effects (`normal`, `thermal` iron-bow, `night_vision`, `depth`).
- **Headless render server** — Separate GPU render process pushes RGBA frames into the DB as messages; verification env vars `ELODIN_SENSOR_CAMERA_DB`, `ELODIN_SENSOR_CAMERA_MAX_TICKS`.
- **Latency modeling** — `ctx.read_msg(name, timestamp=ctx.timestamp - latency_us)` retrieves historical frames for realistic vision pipelines.
- **Editor display** — `sensor_view` KDL panel shows live frames by `"entity.camera"` name; combined vision + FCU SITL in a single `post_step` is a documented pattern.

## 8. Python SDK — `elodin.db` Client

- **Embedded server** — `edb.Server.start(path, addr)` spins up a DB from plain Python (tests/notebooks).
- **Connection & discovery** — `edb.Client.connect(addr)`, `client.components`, `client.earliest_timestamp`, `client.latest` (latest-value subscription).
- **Batched writing** — `client.table_writer(schema)` with `Field` DSL (`f64[7].labeled(...)`, `f32`, `bool_`, `i32`); blocking `write` and non-blocking `write_nowait` with `queue=drop-oldest|drop-newest`, `maxlen`, and a `dropped` counter; `timestamp="ns"` source declaration (stored as µs); `client.send` convenience single-value write.
- **Reading** — `client.time_series` (paginated history → numpy), `client.stream` (live or fixed-rate replay: `rate_hz`, `start=earliest|latest|µs`), `client.sql` (DataFusion → pyarrow.Table, `sql_table_name()` mapping).
- **Message logs** — `send_msg` / `get_msgs` / `msg_stream` for variable-length payloads (bytes/str/JSON).
- **Diagnostics** — `ELODIN_DB_LOG=debug` tracing filter for embedded server/client.

## 9. Process Orchestration (s10)

- **TOML recipes** — `s10.toml` defines `sim` (Python sims, uv/python3 auto-detect; not on Windows), `cargo` (build+run with features/env), `process` (arbitrary commands), and `group` (parallel orchestration) recipe types; config resolution: `-c` flag → `S10_CONFIG` → `./s10.toml` → `/etc/elodin/s10.toml`.
- **Watch mode** — `--watch` auto-restarts on file changes (200 ms debounce, respects `.gitignore`); per-recipe `no_watch`; `--release` for cargo recipes.
- **Lifecycle control** — Restart policies (`instant`/`never`), readiness probes (`delay`, `tcp`, `unix`, `file`, `log`), `depends_on` ordering, `${VAR}`/`${VAR:-default}` env expansion, colored multiplexed output, sim liveness heartbeat (kills hung sims), Linux cgroup-v2 cleanup.
- **Python integration** — `world.recipe(el.s10.PyRecipe.cargo(...)/process(...))` registers sidecars (controllers, GStreamer pipelines, log clients, render servers); `python main.py plan <dir>` generates s10 config; `--no-s10` prevents recursion; Monte Carlo injects `ELODIN_MC_PORT_*` / `ELODIN_MONTE_CARLO_RUN_DIR` placeholders.
- **Editor integration** — `elodin editor main.py` auto-generates s10 recipes via the `plan` mechanism.

## 10. Monte Carlo Campaigns & Truth-Data Calibration

- **Parameter declaration** — `el.monte_carlo.params_spec()` / `Param` (defaults + bounds) in the sim; `el.monte_carlo.params()` reads the campaign row (`run_id`, `seed`, `db_path`, `db_addr`, `cache_dir`, `run_dir`, `slots()`); `el.monte_carlo.result()` writes scalar outputs to `result.json`; `el.monte_carlo.port("name", default)` allocates per-worker ports.
- **Sampling** — `spec.toml` parameter ranges with Latin Hypercube Sampling (`method = "lhs"`), reproducible plan CSVs (`sample` regenerates).
- **Campaign configuration (`campaign.toml`)** — `workers` (bounded concurrency; defaults to logical cores), `[[build]]` one-time pre-worker steps (e.g. cargo-build the controller), `[env]` injection with machine-sympathy thread defaults (OMP/XLA), `[resources.ports]` (static base+stride or `"auto"`; db_port+1 reserved for asset server), `scratch_dir = "auto"` (tmpfs per-run IO), `[retention]` (`keep_run_db = always|never|on-fail`, `prune_on_pass`/`prune_on_fail`, `compact_run_db`), `[quality]` gates (`max_behind_deadline_frac`, `max_real_time_factor`, `fail_on_degraded`), `[params_delivery]` (write sampled params to file with env placeholders).
- **Hooks & scoring** — `post_run` and `post_campaign` Python hooks; tri-state scoring (`valid`, `pass`, metrics) — e.g. soft-landing checks, capture radius, trajectory/pitch RMSE vs truth.
- **Campaign outputs** — `results.csv` (with `failure_reason`), `perf.csv`, `resources.csv`, `summary.json`, `campaign_summary.txt`, per-run DBs, per-run `sim_summary.json` (also via `ELODIN_SIM_SUMMARY_JSON`); `--memory-probe` writes PSS evidence to `memory.json`.
- **Process hygiene** — Campaigns reap stale `elodin`/`elodin-db` processes by default (`--keep-existing` opts out); Linux cgroup teardown (auto `systemd-run --user --scope` when needed).
- **Truth-data workflow** — Vendor raw + SI-derived experimental telemetry with provenance and `sanity_check()`; stdlib-only reference module (resample, despike, smooth, interpolate); physics-based reconstruction of missing channels; kinematic truth-ghost entity (non-`el.Body` archetype replayed on `SimulationTick`); truth-vs-sim overlays in the editor.
- **Calibration loop** — Narrow `spec.toml` from best-fit runs manually or via `calibrate.py --rounds --samples`; diagnose worst runs via `elodin-db export --join --flatten`.
- **Scale story** — Cloud parallelism up to ~100k runs (docs); local scaling/memory ablation harness in `examples/monte-carlo` (`S10_MAX_INFLIGHT`, scaling.csv, shared-constant PSS evidence).

## 11. Elodin DB — Server, Storage Model & Protocol

- **Server** — `elodin-db run <addr> <path>` (default `[::]:2240`, default dir `~/.local/share/elodin/db`); flags: `--config <lua>` (pre-register vtables/metadata/msgs), `--log-level`, `--start-timestamp`, `--http-addr` (HTTP API only if given), `--follows`, `--follow-packet-size`, `--assets <dir>` (ingest asset root, used on Aleph).
- **Data model** — Hierarchical dot-notation components (no entity IDs; FNV-1a `ComponentId`), fixed-shape dense tensors stored as timestamp+value time series (µs, i64; nanosecond ingest sources auto-divided), postcard-encoded variable-length message logs (video NALs/OBUs, logs, JSON events) with Umbra-style string offsets.
- **Impeller2 protocol** — Packet types Table / Message / TimeSeries / MsgWithTimestamp with 4-byte header (type, id, request-id); VTables describe zero-copy tensor layouts (field offsets must respect PrimType alignment); request-reply correlation; pub-sub `Stream` subscriptions (selective, batched); optional zstd compression; transports: TCP, UDP, serial (COBS framing), shared memory (bbqueue), WebSockets.
- **Impeller2 sub-crates** — `impeller2-bevy` (editor ECS sync), `impeller2-stellar` (async TCP/UDP client/server), `impeller2-bbq` (lock-free IPC), `impeller2-frame` (COBS serial), `impeller2-kdl` (schematic serdes), `impeller2-wkt` (well-known types: `Stream`, `SetStreamState`, `DumpMetadata`, ...).
- **Stream types** — RealTime, RealTimeBatched, FixedRate, VTableStream (incl. aggregators like mean), FollowStream; `UdpVTableStream` / `udp_vtable_stream()` Lua for static UDP unicast mirroring.
- **Lua REPL & scripting** — `elodin-db lua [script] [--db PATH]`: `connect(addr)`, `Client:sql`, `get_time_series`, `stream`, `get_msgs`, `send_msg(s)`, `save_archive(..., "csv")`, `dump_metadata`, `VTableBuilder`, `wmm()` helper; REPL commands `:sql`, `:help`, `:exit`.
- **SQL engine** — DataFusion SQL over component tables (Arrow IPC results) with FFT/FFTFREQ functions; exposed via Lua `:sql`, Python `client.sql`, editor SQL panes, and offline `elodin-db query`.

## 12. Elodin DB — EQL Query Language

- **Core language** — Hierarchical component references with automatic time-based joins; field access (`a.world_pos.x`) and 0-based array indexing (`a.world_pos[0]`); arithmetic `+ - * /`; time windows `.last("PT5S")` / `.first("5m")` (ISO-8601 durations); format strings `text ${expr}`; compiles source-to-source to SQL.
- **Formulas** — `fft()`, `fftfreq()`, `norm()`, `atan2(y,x)`, `degrees()`, `clip(min,max)`, `cast(type)` (e.g. `.cast(f32)`), `sqrt()`, `abs()`, `arccos()`, `sign()`; registered in a registry powering parser autocompletion; extension paths: SQL-primitive formulas vs DataFusion UDFs.
- **Usage surfaces** — Editor graphs, `query_table`/`query_plot` panels, viewport camera/object expressions (with `rotate_*`, `rotate_world_*`, `translate_*`, `translate_world_*`, `look_at` transform formulas), object scale expressions, GLB joint animation, offline `elodin-db query --eql`.

## 13. Elodin DB — Replication, Replay & Asset Server

- **Follow mode (`--follows <addr>`)** — TCP replication: schema/metadata sync on connect, full historical backfill, then live streaming; follower still accepts local writers (dual-source pattern, e.g. remote sim + local video); duplicate-component write warnings; `--follow-packet-size` tuning (default ~1500, jumbo 9000).
- **Replay mode (`elodin editor <db> --replay`)** — Editor-side flag: play a recorded DB as if live with progressive timeline reveal (not an `elodin-db run` flag).
- **Asset server** — During `run`, HTTP server on port N+1 serves `{db}/assets/`; local schematic asset paths (GLB, PNG icons, skybox cubemaps) rewritten to `db:` scheme at record time; followers mirror assets via `GET /__index__`; enables fully portable recordings (Aleph HITL DBs replay anywhere).
- **Legacy replay directories** — `metadata.json`, `assets.bin` (internal format, not for external consumption), per-archetype Parquet files; opened via `elodin editor <dir>`; well-known archetypes `body`, `shape`, `asset_handle_panel`, `asset_handle_entity_metadata`.

## 14. Elodin DB — Data Management CLI

- **Inspection** — `info` (recording state/metadata), `list-components`.
- **Offline query** — `query [--eql|sql]` with `--offset`, `--limit` (rows or duration), `-f table|csv|arrow-ipc|parquet`, `--flatten`, `--time-format`, `-p` precision, `--row-index`.
- **Export** — `export` (parallel; parquet/arrow-ipc/csv; `--pattern`, `--join`, `--flatten`, `--csv-fast-floats`, `--mono-ns`/`--mono-us`, `--include-private`); `export-videos` (H.264 and `sensor_camera` RGBA streams → MP4; `--pattern`, `--fps`).
- **Surgery** — `merge` (combine two DBs; `--prefix1/2`, `--align1/2` µs, `--from-playback-start`, `--dry-run`), `trim` (`--from-start`/`--from-end` µs, in-place or `--output`, `--dry-run`), `time-align` (`--all`/`--component`, `--timestamp`), `truncate` (clear data, keep schemas), `drop` (fuzzy name/`--pattern` glob/`--all`), `prune` (remove empty components), `compact` (shrink 8 GB sparse preallocations), `fix-timestamps` (normalize mixed clock sources; `--reference wall-clock|monotonic`).
- **Codegen** — `gen-cpp` emits the single-header C++20 `db.hpp` client library.

## 15. Elodin DB — Client Libraries & Examples

- **C client** — Minimal `client.c` streaming fake sensor data over TCP.
- **C++ clients (`db.hpp`)** — Header-only C++20 library (postcard serialization, VTable builder, `timestamp_ns`, packet encoding); examples: `client-batched.cpp` (recommended: 1 connection/1 VTable/1 packet-per-tick), `client-per-component.cpp`, `rocket-client.cpp`, `log-client.cpp` (structured FSW logs), subscribe example; self-compiling shebang scripts + Makefile.
- **Rust client** — Full-featured example: component auto-discovery, schema retrieval, real-time TUI telemetry dashboard, bidirectional external control (writes `fin_control_trim` at 60 Hz), `-H host:port` CLI.
- **Python** — First-class `elodin.db` client (section 8).
- **Benchmark** — `elodin-db-bench` throughput scenarios (batched vs per-component, customer scenario).

## 16. Elodin Editor

### 16.1 Startup & Connectivity
- **Startup screen** — Connect to IP address, or run a simulation from file.
- **Connection targets** — Live sim (spawns via s10), running DB (local, remote, Aleph over IPv6), recorded DB, legacy replay dir; `--kdl` schematic preload.
- **Editor dev loop** — `cargo run -p elodin -- editor ...`, `cargo watch` hot-reload, cargo features `big_space`, `inspector`, `debug`, `tracy`.

### 16.2 Viewport & 3D Rendering
- **3D viewport** — Bevy 0.19 + big_space floating-origin rendering; per-viewport `fov`, `near`/`far`, fixed `aspect`, `hdr`, `bloom`, grid toggle, EQL-driven camera `pos`/`look_at`/`up` (follow/chase cameras).
- **Camera interaction** — Mouse orbit/pan/zoom; touch gestures (one-finger orbit, two-finger pan + pinch zoom, gated to active viewport); camera anchoring/tracking of entities; safe anchor computation guards against NaN transforms.
- **View cube (Cube-Viewer)** — CAD-style orientation widget: face/edge/corner snapping (incl. hidden-face border groups), rotation arrows (roll/yaw/pitch steps), reset and zoom-out buttons, hover highlighting, camera sync, ENU axis labels.
- **Gizmos & annotations** — `vector_arrow` (EQL vectors; `scale`, `normalize`, `body_frame`, `arrow_thickness`, `label_position`, emissivity, text labels), body axes, `line_3d` trails (played/future color split, `perspective`, `line_width`).
- **3D objects** — `object_3d` bound to EQL pose expressions with `frame`/`frame_orientation`/`orientation=relative|absolute`; meshes: `glb`, `sphere`, `box`, `cylinder`, `plane`, `ellipsoid` with `emissivity`, `glow`, `visibility_range`; `animate joint` drives GLB joints from telemetry (e.g. spinning rotors).
- **Icons & effects** — `icon` (built-in Material Icons or custom PNG, distance-fade); `thruster` GPU exhaust particles (scalar/vector intensity, `plume`/`cold_gas`) — experimental, native-only.
- **Terrain & globes** — `world_mesh` regions (`death_valley`, `globe`) with `lod_count`, `translate`, `visible`; backed by bevy_world_mesh UDLOD/clipmap engine streaming AWS Terrain Tiles + EOX Sentinel-2 imagery with on-disk tile cache.
- **Frustum overlays** — Per-viewport `create_frustum` publishing camera frusta rendered in other viewports (`show_frustums`), with per-source color/thickness, persistence across hidden tabs; frustum∩ellipsoid volume coverage (writes `{ellipsoid}.frustum_coverage` F32 component queryable via EQL) and 2D far-plane projection meshes, gated by inspector toggles.
- **Skybox** — Blockade Labs AI skybox generation (needs `BLOCKADE_API_KEY`), cached presets in manifest, clear/revert; skybox copied into DB on record.
- **Video in viewport** — Live H.264 (and AV1) stream tiles; loss-of-signal overlay and auto-reconnect.
- **Global toggles** — Wireframe, HDR, grid (command palette).

### 16.3 Panels & KDL Schematics
- **Document lifecycle** — Load KDL from path/inline content, save, Save As, Save To DB, clear; hot-reload via `ELODIN_KDL_DIR` file watcher (300 ms debounce, symlink-safe); embedded schematic fallback from Python (`world.schematic()`); multi-window document trees (`window path= title= screen= rect=`) with persisted geometry.
- **Layout containers** — `tabs`, `hsplit`, `vsplit` with `share` weights; tab context menu (inspector, rename).
- **Panels** — `viewport`, `graph`, `component_monitor` (live scalar values), `query_table` (EQL/SQL), `query_plot` (EQL/SQL; `timeseries|xy`; `refresh_interval`, `auto_refresh`), `action_pane` (named Lua actions, e.g. `send_msg` presets), `video_stream`, `sensor_view`, `log_stream` (level filters TRACE–ERROR, auto-scroll), `data_overview` (all-component sparklines), `schematic_tree`, `dashboard` (Bevy-UI flex/grid layouts with text/colors), `inspector` and `hierarchy` sidebars.
- **Root config nodes** — `coordinate` (ENU/NED/ECEF + geodetic origin lat/lon/alt; per-element frame overrides), `theme` (scheme + dark/light mode), `timeline` (`played_color`, `future_color`, `follow_latest`), `skybox`.

### 16.4 Graphs & Data
- **Telemetry graphs** — GPU-accelerated time-series plots from live streams; `type=line|point|bar`; per-channel color children; `auto_y_range` / `y_min`-`y_max`; axis locks persisted across sessions; Ctrl/Shift-constrained pan/zoom; touch support; Alt/Option hidden zoom button.
- **Query plots/tables** — SQL and EQL sourced (incl. FFT spectral plots); XY mode.
- **Line compression** — Hamann–Chen curvature-based downsampling of graph lines and 3D trails (`CurveCompressSettings`: thresholds, target points, keep-recent fraction); GPU index-buffer cap handling for `plot_3d`.

### 16.5 Playback, Timeline & Recording
- **Timeline** — Scrub, pause, step (hold-to-multi-step), rewind; decoupled from simulation; live auto-follow (`follow_latest`) with Jump-to-Latest; replay-speed display; playback speed presets (default 1x, per-sim `default_playback_speed`).
- **Navigation** — Goto Tick (pauses), Fix Current Time Range, Set Time Range (`+5m`, `-10s`, `=ISO8601`).
- **Recording** — Toggle Recording start/stop on the connected DB; time-travel warning names the edited component.

### 16.6 Command Palette
- **Invocation** — `Cmd/Ctrl+P` (fresh), `Cmd/Ctrl+Shift+P` (resume nested state); type-to-filter, Enter/Escape/Backspace navigation.
- **Actions** — Create Window / Viewport / Graph / Monitor / Query Table / Query Plot / Action / Video Stream / Schematic Tree / Data Overview / 3D Object; Toggle Wireframe/HDR/Grid; Reset Cameras (all or one); Skybox submenu (Generate/Clear/Revert/preset); Toggle Recording; Set Playback Speed; Goto Tick; Fix/Set Time Range; Save/Save As/Save To DB/Load/Clear Schematic; Set Color Scheme/Mode; open Documentation and Release Notes.

### 16.7 Theming & Assets
- **Color schemes** — Built-ins `default`, `eggplant`, `catppuccini-macchiato/mocha/latte`, `matrix`; custom JSON presets in `color_schemes/` (assets or app-data dir, user overrides built-ins); persisted in `color_scheme.json`; dark/light modes; KDL `theme` node.
- **Asset sources** — `ELODIN_ASSETS` default source (fallback `./assets`, invalid-path warnings); `kdl` asset source from `ELODIN_KDL_DIR`; `http`/`https` remote assets with ETag-aware filesystem caching (304 reuse); remote GLB loading (`glb path="https://..."`).
- **GLB pipeline constraint** — Bevy 0.19 loads plain glTF 2.0 only (no Draco/meshopt/quantization/basisu); `scripts/optimize-glb.sh <file> <keep-ratio>` decimation workflow for LFS size control with visual verification.

## 17. Video Streaming & Decoding

- **`elodinsink` GStreamer element** — Streams Annex-B H.264 NAL units to the DB (`db-address`, `msg-name`/`msg-id`); keyframe guidance (~12-frame interval, SPS/PPS with every IDR via `h264parse config-interval=-1`); documented pipelines: `videotestsrc`, file playback, V4L2 webcams (JPEG/raw + `nvv4l2h264enc`), macOS `avfvideosrc ! vtenc_h264_hw`, GenICam via `aravissrc`; `GST_PLUGIN_PATH` for local testing.
- **OBS ingestion** — SRT listener pipeline (port 9000, OBS in caller mode; recommended H.264 CBR settings) or obs-gstreamer direct-to-elodinsink; H.264 only (no HEVC).
- **RTSP ingestion** — `rtsp-streamer` standalone Rust producer (`RTSP_URL`, auto-reconnect, Nix-packaged).
- **File streaming** — `video-streamer` FFmpeg utility re-encodes files to AV1 OBUs into the DB (bitrate/keyframe/speed-preset options); ffmpeg-based live-or-preserved timestamp modes.
- **Decoding** — Editor decodes H.264 via Apple VideoToolbox (hardware, macOS/iOS) or OpenH264 (Linux/Windows); automatic SPS/PPS handling; adaptive frame scaling; RGBA output.
- **Synchronized playback & export** — Fixed-rate message streams keep video synced with telemetry scrubbing; `elodin-db export-videos` muxes recordings to MP4.

## 18. Aleph Flight Computer & Flight Software

### 18.1 AlephOS (NixOS on Jetson Orin)
- **Access** — Serial console over left USB-C (115200, `root:root`); SSH over USB-C Ethernet gadget (`fde1:2240:a1ef::1`); WiFi via first-boot `aleph-setup` wizard (`iwctl`), then `aleph-XXXX.local` mDNS SSH; sensor HTTP streams (`curl localhost:2248/component/stream/accel/1`); ES01 hardware datasheet.
- **Deployment** — `./deploy.sh` (build, copy store paths, activate, bootloader entry) with custom `user@host`, SSH options, and `-c` named configurations (`base`, `c-blinky`, `sensor-fw`, `m10q`, `m9n`); `nix run .#deploy`; bootloader rollback via boot menu; user template flake (`aleph/template/`).
- **Recovery** — `nix build .#packages.aarch64-linux.sdimage`, USB boot (F11) + `aleph-installer` to internal SSD; **Inscriber** cross-platform flasher (zstd on-the-fly decompression, interactive drive selection or `--disk`, progress/ETA, macOS `diskutil` / Linux `lsblk`); RP2040 debugprobe flashing (drag `debugprobe.uf2`).
- **NixOS service modules** — `elodin-db` (port 2240; `dbUniqueOnBoot` timestamped-vs-persistent DBs; `--assets /var/lib/elodin/assets` for portable recordings), `serial-bridge`, `mekf`, `msp-osd` (with `autoRecord`), `elodinsink`/GStreamer video, `tegrastats-bridge` (1 Hz SoC telemetry), `udp-component-broadcast`/receive, `sensor-fw` (GPS model option), `c-blinky`, alternate kernel option.
- **MCU operations** — `[firmware]-flash` one-shot services (BOOT0/NRST GPIO + `stm32flash` on `/dev/ttyTHS1`, then start serial-bridge); onboard `reset-mcu` (app/bootloader/NRST/BOOT0) and `flash-mcu --bin/--elf`; STM32H747 UART bootloader timing and Tegra `ttyTHS*` `stty` quirks documented; probe-rs flashing from a host (`cargo rrb fw`, `probe-rs erase/reset`).
- **Betaflight on Aleph FC** — Flash patched Betaflight ELF via probe-rs (`--chip STM32H747IITx`), configure at app.betaflight.com; reference 7-inch quad BOM.

### 18.2 Flight Software Components (`fsw/`)
- **sensor-fw (STM32H747 firmware)** — Dual BMI270 IMUs (1600 Hz I2C/SPI), BMM350 mag (~400 Hz), BMP581 baro (~50 Hz), optional u-blox GPS (M10Q/M9N, 5 Hz NAV-PVT), optional QMC5883L compass; FRD body frame; coning/sculling pre-integration (2-sample Bortz); decimation to ~770 Hz; COBS + EL-frame UART at 1 Mbaud (Log/GpsRecord/CompassRecord/ImuRecord kinds + legacy kindless baro/voltage records); SD-card blackbox (`DATA.BIN`, converted to CSV by `fsw/blackbox`).
- **serial-bridge** — systemd service parsing UART COBS/EL frames into DB components: `imu` (~770 Hz accel/gyro/mag), `ublox` (5 Hz), `aleph` (10 Hz baro/power), `qmc5883l` (50 Hz); GPS-disciplined or wall-clock timestamping with monotonicity guarantee; also bridges MCU log lines into `aleph.stm32.log` for the editor `log_stream` panel.
- **mekf** — Multiplicative EKF attitude service on the Orin: subscribes to `imu`, outputs `q_hat`/`b_hat`/`gyro_est`/`world_pos`/`mag_cal` at ~770 Hz; Lua-configurable noise parameters and `wmm()` magnetic reference (`/root/mekf.lua`).
- **msp-osd** — MSP DisplayPort OSD for FPV goggles: config.toml component-to-input mappings, compass/altitude ladder/climb/speed/artificial horizon elements, debug-terminal and serial VTX backends (Walksnail Avatar on UART7), auto-DVR record, configurable grid; BDX sim test config.
- **udp_component_broadcast** — Python sender/receiver pair bridging DB components across machines over UDP protobuf (`--rename`, `--filter`, sender/local/monotonic timestamp modes, heartbeats, print-only mode); distributed chase-scenario demo.
- **c-blinky** — Bare-metal STM32 blink firmware (400 MHz PLL) with COBS UART logging; local `build.sh` + openocd, or Aleph deploy-time flash service.
- **roci framework** — Composable `System` trait with `pipe` combinators; drivers `Hz<N>`, `Interrupt`, `OsSleepDriver`, `LoopDriver`; `World` state containers with `Componentize`/`Decomponentize`/`AsVTable`/`Metadatatize` derives; functions-as-systems; Impeller2 `tcp_connect`/`tcp_listen`; `CSVLogger`; identical code in sim and on hardware.
- **roci-adcs algorithms** — TRIAD attitude determination, MEKF, UKF (configurable sigma points), MAG.I.CAL magnetometer calibration, MagKal UKF calibration, Yang quaternion LQR.
- **Undocumented fsw utilities** (no README; covered indirectly) — `aleph-setup`, `aleph-status`, `blackbox`, `lqr`, `openocd` configs, `rtsp-streamer`, `tegrastats-bridge`.

## 19. Example Simulations & Reference Use Cases

- **ball** — Atmospheric bouncing ball (gravity, drag, wind, `@el.map_seq` bounce); regression baseline; Matplotlib post-analysis (`plot.py`, Polars from Exec API).
- **three-body** — Stable periodic three-body orbits via gravity edges/graph queries; Broucke configurations; the canonical quickstart sim.
- **n-body** — Solar-system gravity with CSV truth-ghost overlays, RK4; canonical backend benchmark (`benchmark_backends.py`, `accuracy_report.py`, tick env overrides).
- **frames** — Automated frame-correctness regression: gravity sign/magnitude in ENU vs NED, ECI vs GCRF orbital equivalence, energy conservation across all world frames.
- **rocket** — 6-DOF rocket: Mach/AoA/fin lookup-table aero, thrust curve, wind, cascaded pitch PID with Mach-scaled fins, external `fin_control_trim`, EQL-derived plots (`atan2`, `degrees`, `norm`, `clip`).
- **rocket-barrowman** — Streamlit rocket-design suite: Barrowman CP/CG solver, ThrustCurve.org REST motor search with local cache, AI natural-language builder (OpenAI with regex fallback), multi-phase motor optimizer, parachute triggers, flight analysis, ISA/NRLMSISE-00/NetCDF/hybrid atmospheres, keyless Open-Meteo ERA5/GFS weather fetch with cache and ISA fallback, Plotly/trimesh rendering, Elodin editor launch, OpenRocket compatibility.
- **drone** — Quadcopter/quadplane 6-DOF (ENU world/FLU body) with cascaded PID, INDI gain-estimation Jupyter workflow, `--telemetry` CSV export; regression baseline.
- **betaflight-sitl** — Real Betaflight firmware in 8 kHz UDP lockstep (`SIMULATOR_GYROPID_SYNC`): FDM/RC/motor ports (9001–9004, CLI 5761), arming workflow (BOOTGRACE, AUX1, gyro cal, eeprom.bin), FLU↔FRD compensation, quad-X motor remapping, multi-rate sensor simulation with noise/bias models, portable DB replay, Aleph HITL pattern; git submodule build.
- **crazyflie-edu** — Educational SITL/hardware/HITL progression: portable C `user_code.c` (500 Hz control API) compiled into a SITL binary or Bitcraze app-layer firmware; wireless cfloader flashing; `--hitl` via Crazyradio with keyboard parameters; four structured labs (motor control, powertrain ID, firmware flash + HITL, tethered hover) with safety conventions.
- **cube-sat / cube-sat-pysim** — LEO attitude control (MEKF, LQR, reaction wheels, EGM08); pysim variant exercises `World.to_jax` + Matplotlib headless; regression baseline.
- **apollo-lander** — Flagship SITL + Monte Carlo + truth-data example: P63/P64/P66 powered descent, Rust LGC reference-trajectory guidance via UDP bridge, quaternion PD RCS, throttle lag/erosion band, truth-replay ghost from cleaned Apollo 11 telemetry, campaign scoring (soft landing, RMSE), calibration loop, GPU exhaust plumes, editor mission scenes.
- **monte-carlo** — Minimal MC campaign: saturated PD point-mass with UDP controller, LHS sampling, port planning, retention/quality knobs, scaling & memory-probe harnesses, hook-based reporting.
- **linalg** — Kalman-filter bench validating LAPACK-backed ops (SVD/LU/QR/Cholesky) across backends in CI.
- **ellipsoid** — Sensor-camera frustum∩ellipsoid coverage and 2D projection demo with dual viewports and inspector toggles.
- **sensor-camera** — Entity-mounted RGB/thermal/night-vision/depth cameras, headless render server, latency-modeled `ctx.read_msg`, `sensor_view` panels, headless DB verification mode.
- **db-client** — Pure-Python `elodin.db` writer/reader demo: synthetic Crazyflie figure-8, table_writer at mixed rates, derived write-back stream, JSON lap events, live editor schematic with animated rotors; `--no-editor --duration` headless mode.
- **logstream** — C++ postcard `LogEntry` ingestion + editor log-viewer panel with level filtering.
- **video-stream** — Rolling-ball sim + three concurrent video paths (test pattern, OBS SRT, RTSP) in editor tabs, replay from recorded DB, MP4 export.
- **voyager** — WIP Voyager 1/2 heliocentric sim with SPICE-kernel truth (download script); known accuracy gap.
- **Undocumented examples** (no README) — `geo-frames`, `stablehlo`, `linalg-iree`.

## 20. Core Libraries (Rust)

- **nox** — Rust JAX-like tensor engine: `Scalar`/`Vector`/`Matrix`/`Tensor` types, symbolic IR, differentiable primitives, Cranelift + JAX backends; `nox::Array` NumPy-style right-aligned broadcasting with fallible `try_*` variants; zero-copy `no_std` `ArrayView` (raw-byte views, NumPy-style printing); noxpr graph construction (`Noxpr`/`NoxprNode`, typed ops incl. dot_general/scan/cholesky, `JaxTracer` lowering to StableHLO, `PrettyPrintTracer`).
- **nox-frames** — Compile-time frame-safe vectors/poses (`Pose3<From,To>`, quaternions); Earth chains GCRF/ECI → ECEF/ITRF → NED with `hifitime` epochs and IERS corrections; `sun_vec(epoch)`; `no_std`.
- **cranelift-mlir** — StableHLO→Cranelift JIT backend (IREE replacement, 6x–8000x RTF gains): winnow MLIR parser, constant folding + DCE, dual scalar/pointer ABI with cross-ABI marshaling, f64x2 SIMD, parallel codegen, case-branch splitting, 2 GB arena, large-constant data section + content-addressed mmap interning (`ELODIN_CACHE_DIR`), zero-copy `TickFn`, ~40-function tensor runtime, faer-backed LAPACK, N-D gather, 256 MB tick worker stack, CPU-only; diagnostics: `ELODIN_CRANELIFT_DEBUG_DIR` checkpoints, `checkpoint_test` XLA-vs-Cranelift verifier, MLIR bisection, `catalog_ops.py`, profile JSON + waveform/diff/plot scripts, Tracy zones.
- **stellarator** — Deterministic single-threaded async runtime for FSW: thread-local executors, no steady-state allocation, structured concurrency (`struc_con`, `CancelToken`), io_uring (Linux) with polling fallback, zero-copy fs/TCP/UDP/serial/mmap I/O, timers; Miri-tested; embedded `no_std` build.
- **maitake / maitake-sync / mycelium-util** — `no_std` async runtime kit (tasks, schedulers, custom storage, timer wheel) and sync primitives (async Mutex/RwLock/Semaphore, WaitCell/WaitQueue/WaitMap with `wait_for`/`is_closed`, blocking + spin locks, custom RawMutex backends, `CachePadded`); panic="abort" requirement.
- **bbqueue (vendored)** — Lock-free SPSC ring buffer bridging Impeller2 TCP threads to the Bevy editor; patched `usize` frame headers (frames up to 8 MiB vs upstream 64 KiB); documented upstream-sync workflow.
- **elodin-macros** — Derives for Rust↔nox-py integration: `Component` (name + Impeller schema), `Archetype`, `ComponentGroup`, `IntoOp`/`FromOp`, `FromBuilder`, `ReprMonad`.
- **postcard-c** — Header-only C `postcard.h` (bounds-checked, zero-copy decode, varints) + RON-driven C++23 codegen (`encode_vec`/`decode`/`encoded_size`; primitives, vectors, maps, enums as `std::variant`, optionals, tuples, nesting); 100% Rust postcard wire-compatible; used for FSW telemetry encoding, ground command decoding, HITL exchange, Impeller2 payloads.
- **wmm** — NOAA WMM 2020 wrapper: B-field at geodetic coordinates with secular variation, declination/inclination/intensity, error bars, hifitime epochs; magnetometer-calibration workflow; used by mekf via Lua `wmm()`; coefficient-update procedure documented.
- **hamann-chen-line** — Curvature-based polyline simplification (2D, time-series, 3D, trajectory-time-norm shared indices); deterministic sorted-index output; archive-vs-view integration pattern; optional CSV CLI.
- **video-toolbox** — Apple VideoToolbox H.264 decode wrapper (SPS/PPS management, hardware decode, adaptive scaling, RGBA output).
- **bevy_world_mesh** — Terrain/globe engine (see 16.2) plus developer harnesses: region/globe render scripts, interactive fly camera (WASD/QE/mouse, speed-by-proximity on globe), 20+ keyboard debug overlays (wireframe, LOD coloring, morph/blend, frustum freeze, precision path), headless screenshot env vars, synthetic offline terrain generators.
- **bevy_geo_frames / bevy_mat3_material** — Geographic coordinate frames for Bevy; 3×3 shear/scale vertex-shader material with inverse-transpose normals (`sphere-mat3` example).
- **s10** — (section 9). **impeller2** — (section 11).

## 21. Profiling & Performance Diagnostics

- **Tracy profiling (Linux only)** — `just install tracy` builds editor/db/nox-py with Tracy; fixed ports per process (editor 8087, render server 8088, sim 8089, elodin-db 8090); `tracy-capture` + `tracy-csvexport` headless workflow; Bevy `trace_tracy` automatic system zones; Cranelift per-function JIT zones; DB spans (handle_conn, sink_table, follow_stream); `TRACY_NO_EXIT=1`; sudo for sampling; AutoNoVsync tip.
- **Cranelift profiling** — `ELODIN_CRANELIFT_DEBUG_DIR` stderr report (tick latency distribution, hot functions, SIMD utilization, marshal bytes), `profile.json` with per-tick waveform, `diff_profile.py`, `plot_tick_waveform.py`.
- **Sim bench/profile** — `bench --ticks N` timing summary (tick time, build time, real_time_factor); `--profile` FLOP/HLO analysis + DOT graphs.
- **DB benchmark** — `elodin-db-bench` scenario throughput.
- **MC observability** — `--memory-probe` PSS, `perf.csv`/`resources.csv`, campaign summaries.

## 22. QA & Release Engineering

- **Agentic QA plans** — `.cursor/skills/qa-test-plan/` skill + template: markdown test cases with exact shell steps and objective pass criteria; ID areas (SDK, SIM, DB, EDITOR, RUST, LINT, ALEPH), priorities P0–P2, modes `agent`/`agent+visual`/`manual`; plans instantiated under `.cursor/skills/qa-test-plan/<date>-<release>.md` with artifact directories; execution rules embedded in each plan.
- **Release workflow** — `release/v$VERSION` branch + PR, `just version`, `just tag`, `just wait-for-release`, `just promote` (PyPI wheel publish via `uv publish`), `just public-changelog` (generates docs changelog + version check), post-release alpha bump (`semver-cli`); pre-release tagging variant.
- **Docs site** — Zola static site (`docs/public/`, AdiDoks theme): local `zola serve`, GitHub Pages auto-deploy on main, PR preview URLs with auto-comment/cleanup, manual workflow dispatch, `just encode` H.264+AV1 video assets.
- **Learning resources** — VIDEOS.md YouTube index (dev intro, Tracy walkthrough); docs.elodin.systems; ES01 datasheet redirect.

## 23. Coordinate Frames & Conventions

- **World frames** — ENU (+X East, +Y North, +Z Up; gravity −Z; terrestrial default), NED (aviation; gravity +Z), ECEF (Earth-fixed geodetic); simulation additionally supports ECI and GCRF world frames (exercised by `examples/frames`), while the KDL `coordinate` node and docs list ECI/GCRF as not yet supported for visualization conventions.
- **Body frames** — Documented body conventions: +X nose, +Y left, +Z up (docs reference); FLU for drones, FRD for sensor-fw output and Betaflight compensation.
- **Geodetic anchoring** — KDL `coordinate` accepts `lat`/`lon`/`alt` origin; per-element `frame` overrides; positions may use NED/ECEF while orientations remain ENU (changelog note).
- **EQL pose transforms** — `rotate_x/y/z`, `rotate_world_*`, `translate_*`, `translate_world_*`, `look_at`, chainable, plus `.cast(f32)`.

## 24. Historical / Deprecated / Removed / WIP (exclude from active coverage)

- **Removed features** — IREE backend (→ cranelift-mlir, v0.17.1); Basilisk integration (v0.15.4); `elodin create` (v0.15.2); `exec.write_to_dir` (→ SaveArchive, v0.13); `python sim.py run --watch` (→ `elodin run`, v0.7); `time_step`/`output_time_step` params (→ `simulation_rate`/`default_playback_speed`); DB entity concept (v0.15; old saves incompatible); `el.Time`/`el.advance_time` (→ `SimulationTick`).
- **Removed Python viz APIs (v0.15)** — `el.VectorArrow`, `el.Line3d`, `el.BodyAxes`, `Panel.*`/`Shape` layouts → KDL schematic equivalents (`world.schematic()` migration path documented).
- **Deprecated / legacy** — Lua `downlink.lua` replication (→ `--follows`); legacy replay directory format (`assets.bin` not for external use); memserve docs server (→ Zola); navigation_gizmo plugin UX (→ view_cube); rust-client future-timestamp workaround (→ `external_control` metadata); ThrustCurve HTML scraping (→ REST API); muxide vendored fork (→ upstream); bbq2 fork (→ vendored bbqueue).
- **Known WIP / experimental** — Voyager sim accuracy (probes don't reach Saturn); GPU exhaust particles (experimental, native-only); crazyflie NOTES architecture options A/C (documented, not implemented); ECI/GCRF visualization conventions.
