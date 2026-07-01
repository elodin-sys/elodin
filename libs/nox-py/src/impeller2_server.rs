use crate::Error;
use bytemuck;
use elodin_db::{AtomicTimestampExt, DB, State, handle_conn};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{ComponentMetadata, EntityMetadata};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{self, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use stellarator::struc_con::{Joinable, Thread};
use stellarator::util::CancelToken;
use tracing::{info, warn};

use crate::World;
use crate::exec::WorldExec;

pub struct Server {
    db: elodin_db::Server,
    world: WorldExec,
}

impl Server {
    pub fn new(db: elodin_db::Server, world: WorldExec) -> Self {
        Self { db, world }
    }

    pub async fn run(self) -> Result<(), Error> {
        tracing::info!("running server");
        let cancel_token = CancelToken::new();
        self.run_with_cancellation(
            || false,
            |_, _, _, _, _| {},
            |_, _, _, _, _| {},
            false,
            None,
            cancel_token,
        )
        .await
    }

    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
        pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        interactive: bool,
        start_timestamp: Option<Timestamp>,
        cancel_token: CancelToken,
    ) -> Result<(), Error> {
        tracing::info!("running server with cancellation");
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = start_timestamp.unwrap_or_else(Timestamp::now);
        init_db(&db, world.world_mut(), start_time)?;
        let tick_db = db.clone();
        let tick_counter = Arc::new(AtomicU64::new(0));
        let stream: Thread<Option<Result<(), Error>>> =
            stellarator::struc_con::stellar(move || async move {
                loop {
                    let stream = listener.accept().await?;
                    let conn_db = db.clone();
                    stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
                }
            });
        let tick = stellarator::spawn(tick(
            tick_db,
            tick_counter,
            world,
            is_cancelled,
            pre_step,
            post_step,
            start_time,
            interactive,
            cancel_token,
        ));
        futures_lite::future::race(async { stream.join().await.unwrap().unwrap() }, async {
            tick.await
                .map_err(|_| stellarator::Error::JoinFailed)
                .map_err(Error::from)
        })
        .await
    }
}

/// Asset key under `{db}/assets/` for the active schematic the editor loads.
const ACTIVE_SCHEMATIC_KEY: &str = "schematics/main.kdl";

/// Ingest the source `assets/` tree into `{db}/assets/` once and register the
/// generated schematic as a stored asset (RFD #724).
///
/// The DB owns all KDL handling: `ingest_asset_dir` copies the whole tree and
/// rewrites local asset paths to `db:`, and `store_asset` rewrites the active
/// schematic the same way. This producer does no KDL parsing or rewriting and
/// never mutates `world.metadata.schematic`. Call once before starting the DB
/// Asset Server so early HTTP requests do not 404.
///
/// `entry` is the simulation entrypoint (e.g. the `main.py` path), used to
/// resolve the source `assets/` tree relative to the sim folder and its
/// ancestors when `ELODIN_ASSETS` is unset — so a run whose cwd is not the sim
/// directory still ingests the right tree instead of the wrong one or none.
pub fn prime_schematic_assets(
    db: &elodin_db::DB,
    world: &World,
    entry: Option<&Path>,
) -> Result<(), elodin_db::Error> {
    if let Some(source) = resolve_source_assets_root(entry) {
        match elodin_db::assets::ingest_asset_dir(&db.path, &source) {
            Ok(report) if report.skipped => {
                tracing::info!(source = %source.display(), "assets already ingested; skipping")
            }
            Ok(report) => tracing::info!(
                source = %source.display(),
                files = report.file_count,
                bytes = report.byte_count,
                "ingested assets into db"
            ),
            Err(err) => {
                tracing::warn!(?err, source = %source.display(), "failed to ingest assets into db")
            }
        }
    }

    // Register the generated schematic as a stored asset. `store_asset` rewrites
    // its asset paths to `db:` (unparsable KDL is stored verbatim), then
    // `set_active_schematic` points `schematic.active` at it. Failures are
    // logged, not fatal: a missing schematic asset must not abort the sim.
    let Some(content) = world.metadata.schematic.as_deref() else {
        return Ok(());
    };
    // Store the schematic as the active asset and point at it (RFD #724): the
    // bytes travel only as an asset, fetched over the Asset Server HTTP. If
    // either step fails, warn — there is no inline fallback.
    match db.store_asset(ACTIVE_SCHEMATIC_KEY, content.as_bytes()) {
        Ok(()) => {
            if let Err(err) = db.set_active_schematic(ACTIVE_SCHEMATIC_KEY) {
                tracing::warn!(?err, "failed to set active schematic pointer");
            }
        }
        Err(err) => {
            tracing::warn!(?err, "failed to store active schematic into db assets");
        }
    }
    Ok(())
}

pub fn init_db(
    db: &elodin_db::DB,
    world: &mut World,
    start_timestamp: Timestamp,
) -> Result<(), elodin_db::Error> {
    tracing::info!("initializing db");
    db.set_earliest_timestamp(start_timestamp)?;

    db.with_state_mut(|state| {
        for (component_id, (schema, component_metadata)) in world.metadata.component_map.iter() {
            let Some(column) = world.host.get(component_id) else {
                continue;
            };
            let size = schema.size();
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for (i, entity_id) in entity_ids.iter().enumerate() {
                let offset = i * size;
                let entity_id = impeller2::types::EntityId(*entity_id);
                let entity_metadata = world
                    .metadata
                    .entity_metadata
                    .entry(entity_id)
                    .or_insert_with(|| EntityMetadata {
                        entity_id,
                        name: format!("entity{}", entity_id),
                        metadata: Default::default(),
                    });
                let pair_name = format!("{}.{}", entity_metadata.name, component_metadata.name);
                let pair_id = ComponentId::new(&pair_name);
                let pair_metadata = ComponentMetadata {
                    component_id: pair_id,
                    name: pair_name,
                    metadata: component_metadata.metadata.clone(),
                };

                state.set_component_metadata(pair_metadata, &db.path)?;
                state.insert_component(pair_id, schema.clone(), &db.path)?;
                let component = state.get_component(pair_id).unwrap();
                let buf = &column.buffer[offset..offset + size];
                component.time_series.push_buf(start_timestamp, buf)?;
            }
        }
        if !world.metadata.sensor_cameras.is_empty() {
            match serde_json::to_string(&world.metadata.sensor_cameras) {
                Ok(json) => {
                    state
                        .db_config
                        .metadata
                        .insert("sensor_cameras".to_string(), json);
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize sensor_cameras metadata: {e}");
                }
            }
        }
        for entity_metadata in world.entity_metadata().values() {
            state.set_component_metadata(
                ComponentMetadata {
                    component_id: ComponentId::new(&entity_metadata.name),
                    name: entity_metadata.name.clone(),
                    metadata: entity_metadata.metadata.clone(),
                },
                &db.path,
            )?;
        }
        Ok::<_, elodin_db::Error>(())
    })?;

    const PLAYBACK_FREQUENCY: f64 = 60.0;
    let default_stream_time_step =
        Duration::from_secs_f64(world.metadata.default_playback_speed / PLAYBACK_FREQUENCY);
    db.default_stream_time_step.store(
        default_stream_time_step.as_nanos() as u64,
        atomic::Ordering::SeqCst,
    );
    db.with_state_mut(|state| {
        state.db_config.default_stream_time_step = default_stream_time_step;
    });
    let _ = db.save_db_state();

    Ok(())
}

/// Resolve the source `assets/` tree to ingest via [`elodin_db::assets::resolve_assets_root`],
/// passing the simulation `entry` so ingest can find `assets/` next to the sim
/// (or an ancestor) rather than only `$ELODIN_ASSETS` / cwd.
fn resolve_source_assets_root(entry: Option<&Path>) -> Option<PathBuf> {
    elodin_db::assets::resolve_assets_root(entry)
}

#[cfg(test)]
mod asset_tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Serializes tests that mutate `ELODIN_ASSETS*` env vars (process-global).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: env mutations are serialized by `ENV_LOCK` and restored on drop.
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            // SAFETY: env mutations are serialized by `ENV_LOCK` and restored on drop.
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: restoring the previous value is scoped to this test's `_guard`.
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn prime_stores_active_schematic_and_sets_pointer() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(assets.join("meshes")).unwrap();
        std::fs::write(assets.join("meshes/rocket.glb"), b"glb-bytes").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some(
            "object_3d \"rocket.world_pos\" {\n    glb path=\"meshes/rocket.glb\"\n}\n".to_string(),
        );

        let _guard = EnvVarGuard::set("ELODIN_ASSETS", assets.to_str().unwrap());
        prime_schematic_assets(&db, &world, None).unwrap();

        // The whole source tree is copied into the DB once.
        let stored = elodin_db::assets_http::assets_dir(&db.path);
        assert_eq!(
            std::fs::read(stored.join("meshes/rocket.glb")).unwrap(),
            b"glb-bytes".to_vec()
        );
        assert!(elodin_db::assets::assets_ingested(&db.path));

        // The DB stores the active schematic with its path rewritten to db: and
        // points schematic.active at it. The producer never touches
        // world.metadata.schematic.
        let active_kdl = std::fs::read_to_string(stored.join("schematics/main.kdl")).unwrap();
        assert!(
            active_kdl.contains("path=\"db:meshes/rocket.glb\""),
            "expected db: path in stored schematic, got:\n{active_kdl}"
        );
        let active = db.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(active.as_deref(), Some("schematics/main.kdl"));

        // The active schematic is read back from its asset file (single source).
        assert_eq!(
            db.read_active_schematic().as_deref(),
            Some(active_kdl.as_str())
        );
    }

    #[test]
    fn prime_ingest_is_copy_once() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("rocket.glb"), b"v1").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let world = World::default();

        let _guard = EnvVarGuard::set("ELODIN_ASSETS", assets.to_str().unwrap());
        prime_schematic_assets(&db, &world, None).unwrap();

        // A source change after ingest must not leak into the frozen DB record.
        std::fs::write(assets.join("rocket.glb"), b"v2").unwrap();
        prime_schematic_assets(&db, &world, None).unwrap();

        let stored = elodin_db::assets_http::assets_dir(&db.path);
        assert_eq!(
            std::fs::read(stored.join("rocket.glb")).unwrap(),
            b"v1".to_vec()
        );
    }

    #[test]
    fn prime_keeps_active_local_when_asset_absent() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        // No source assets resolved, so nothing is ingested into the DB.
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some(
            "object_3d \"rocket.world_pos\" {\n    glb path=\"meshes/rocket.glb\"\n}\n".to_string(),
        );

        prime_schematic_assets(&db, &world, None).unwrap();

        // The glb never reached the DB, so store_asset must NOT rewrite the path
        // to db: (which would make the editor chase a 404). The active pointer is
        // still set so the editor can load the (local-path) schematic.
        let stored = elodin_db::assets_http::assets_dir(&db.path);
        let active_kdl = std::fs::read_to_string(stored.join("schematics/main.kdl")).unwrap();
        assert!(active_kdl.contains("path=\"meshes/rocket.glb\""));
        assert!(!active_kdl.contains("db:meshes/rocket.glb"));
        let active = db.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(active.as_deref(), Some("schematics/main.kdl"));
    }

    #[test]
    fn prime_resolves_assets_relative_to_sim_entry() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // With `ELODIN_ASSETS` unset, ingest must find `assets/` next to the sim
        // entry even when the process cwd is elsewhere (Bugbot: the entry was
        // ignored, so a run outside the sim folder ingested the wrong tree/none).
        let _no_env = EnvVarGuard::unset("ELODIN_ASSETS");

        let dir = tempdir().unwrap();
        let sim = dir.path().join("sim");
        std::fs::create_dir_all(sim.join("assets/meshes")).unwrap();
        std::fs::write(sim.join("assets/meshes/rocket.glb"), b"glb-bytes").unwrap();
        let entry = sim.join("main.py");
        std::fs::write(&entry, b"# sim").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let world = World::default();
        prime_schematic_assets(&db, &world, Some(&entry)).unwrap();

        let stored = elodin_db::assets_http::assets_dir(&db.path);
        assert_eq!(
            std::fs::read(stored.join("meshes/rocket.glb")).unwrap(),
            b"glb-bytes".to_vec(),
            "ingest should copy assets found next to the sim entry"
        );
        assert!(elodin_db::assets::assets_ingested(&db.path));
    }
}

pub fn copy_db_to_world(state: &State, world: &mut WorldExec) {
    let world = world.world_mut();
    for (component_id, (schema, _)) in world.metadata.component_map.iter() {
        let Some(column) = world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();

        let mut component_changed = false;

        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let Some(entity_metadata) = world.metadata.entity_metadata.get(&entity_id) else {
                continue;
            };
            let Some((_, component_metadata)) = world.metadata.component_map.get(component_id)
            else {
                continue;
            };

            let pair_id = ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let Some((_, head)) = component.time_series.latest() else {
                continue;
            };

            let current_value = &column.buffer[offset..offset + size];
            if current_value != head {
                component_changed = true;
            }

            column.buffer[offset..offset + size].copy_from_slice(head);
        }

        if component_changed {
            world.dirty_components.insert(*component_id);
        }
    }
}

pub type PairId = ComponentId;

pub fn get_pair_ids(world: &WorldExec, components: &[ComponentId]) -> Result<Vec<PairId>, Error> {
    let w = world.world();
    let mut results = vec![];
    for component_id in components {
        if let Some((_schema, component_metadata)) = w.metadata.component_map.get(component_id) {
            let Some(column) = w.host.get(component_id) else {
                continue;
            };
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for entity_id in entity_ids.iter() {
                let entity_id = impeller2::types::EntityId(*entity_id);
                let Some(entity_metadata) = w.metadata.entity_metadata.get(&entity_id) else {
                    continue;
                };
                let pair_id =
                    ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
                results.push(pair_id);
            }
        }
    }
    Ok(results)
}

pub fn commit_world_head_unified(
    state: &State,
    exec: &mut crate::exec::WorldExec,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
) -> Result<(), Error> {
    commit_world_head_for_world(state, exec.world_mut(), timestamp, exclusions)
}

fn commit_world_head_for_world(
    state: &State,
    world: &mut crate::world::World,
    timestamp: Timestamp,
    exclusions: Option<&HashSet<ComponentId>>,
) -> Result<(), Error> {
    for (component_id, (schema, _)) in world.metadata.component_map.iter() {
        let Some(column) = world.host.get_mut(component_id) else {
            continue;
        };
        let entity_ids = bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
        let size = schema.size();
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let offset = i * size;
            let entity_id = impeller2::types::EntityId(*entity_id);
            let Some(entity_metadata) = world.metadata.entity_metadata.get(&entity_id) else {
                continue;
            };
            let Some((_schema, component_metadata)) =
                world.metadata.component_map.get(component_id)
            else {
                continue;
            };

            if let Some(exclusions) = exclusions
                && exclusions.contains(component_id)
            {
                continue;
            }

            let pair_id = ComponentId::from_pair(&entity_metadata.name, &component_metadata.name);
            let Some(component) = state.get_component(pair_id) else {
                continue;
            };
            let buf = &column.buffer[offset..offset + size];
            component.time_series.push_buf(timestamp, buf)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn tick(
    db: Arc<DB>,
    tick_counter: Arc<AtomicU64>,
    mut world: WorldExec,
    is_cancelled: impl Fn() -> bool + 'static,
    pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    start_timestamp: Timestamp,
    interactive: bool,
    cancel_token: CancelToken,
) {
    let external_controls: HashSet<ComponentId> = external_controls(&world).collect();
    let wait_for_write: Vec<ComponentId> = wait_for_write(&world).collect();
    let wait_for_write_pair_ids: Vec<PairId> = get_pair_ids(&world, &wait_for_write).unwrap();
    let mut wait_for_write_pair_ids = collect_timestamps(&db, &wait_for_write_pair_ids);
    let generate_real_time = world.world().metadata.generate_real_time;
    let configured_ticks_per_telemetry = world.world().ticks_per_telemetry();
    let time_step = world.world().sim_time_step().0;
    let detailed_timing = std::env::var("ELODIN_DETAILED_TIMING").is_ok();
    if detailed_timing {
        world.profiler_mut().detailed_timing = true;
        info!("detailed timing enabled (ELODIN_DETAILED_TIMING)");
    }
    // Real-time pacing lead buffer: aim to wake this far *before* each tick
    // deadline so the sim runs slightly ahead of wall-clock. A transient slow
    // cycle (e.g. a lockstep `post_step` round-trip that overruns one period)
    // then spends the buffer instead of crossing the deadline and logging
    // "cannot achieve real-time". The schedule stays anchored, so the sim is
    // bounded to at most `lead` ahead of real-time — data is available a few ms
    // early, which is harmless for telemetry/SITL. It is an absolute time
    // buffer (may span several ticks for high-rate sims, which is the point).
    // Default covers the observed lockstep tail; `ELODIN_PACING_LEAD_US`
    // overrides. Sanity-capped at 1s to swallow pathological env values.
    let pacing_lead: Duration = std::env::var("ELODIN_PACING_LEAD_US")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_micros)
        .unwrap_or(Duration::from_micros(2000))
        .min(Duration::from_secs(1));
    // Startup grace: suppress real-time warnings for this long after pacing
    // begins, so a slow SITL/controller boot (e.g. Betaflight's multi-second
    // init) doesn't fire a spurious "cannot achieve real-time". Overridable via
    // `ELODIN_PACING_GRACE_US`.
    let pacing_grace: Duration = std::env::var("ELODIN_PACING_GRACE_US")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_micros)
        .unwrap_or(Duration::from_secs(5));
    // Summary warmup: discard the first `warmup_ticks` ticks from the end-of-run
    // stats so the simulator/SITL spin-up (the first cycle can stall for seconds
    // while a flight controller boots) doesn't poison mean cycle / post_step /
    // lateness. Overridable via `ELODIN_SIM_WARMUP_TICKS`; 0 disables.
    let warmup_ticks: u64 = std::env::var("ELODIN_SIM_WARMUP_TICKS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100);
    let mut warmup_done = warmup_ticks == 0;
    #[rustfmt::skip]
    let should_cancel = || { is_cancelled() || cancel_token.is_cancelled() };
    let mut next_tick_deadline: Option<Instant> = None;
    let mut last_behind_warning: Option<Instant> = None;
    // Trustworthy-warning state. The honest "cannot achieve real-time" signal is
    // the *achieved cycle rate vs target* measured over a recent window — it is
    // immune to the per-cycle jitter, the lead buffer, and the drift-cap resets
    // that confound an instantaneous "now > deadline" check. We skip a startup
    // grace (the SITL boot stall) and then, each window, warn if the sim is
    // running materially slower than real-time.
    let mut first_pacing_at: Option<Instant> = None;
    let mut rate_win_start: Option<Instant> = None;
    let mut rate_win_start_cycles: u64 = 0;
    let mut pacing_cycles: u64 = 0;
    let mut last_timing_log: Option<Instant> = None;
    // Always-on per-phase timing summary. Dropping `metrics` prints
    // the one-block stdout summary on any exit (cancel, max_tick,
    // panic unwind). See `crate::tick_metrics` for the format.
    let mut metrics = crate::tick_metrics::TickMetrics::new();
    // Feed the summary header its rate context: simulation_rate is
    // 1 / sim_time_step; telemetry_rate is simulation_rate divided
    // by ticks_per_telemetry. The summary uses these to annotate
    // the steps line when rates differ.
    {
        let simulation_rate_hz = 1.0 / time_step.as_secs_f64();
        let telemetry_rate_hz = simulation_rate_hz / (configured_ticks_per_telemetry.max(1) as f64);
        metrics.set_rates(simulation_rate_hz, telemetry_rate_hz);
    }
    loop {
        if should_cancel() {
            metrics.mark_loop_end();
            return;
        }
        if !db.recording_cell.is_playing() {
            next_tick_deadline = None;
            last_behind_warning = None;
            first_pacing_at = None;
            rate_win_start = None;
            rate_win_start_cycles = 0;
            pacing_cycles = 0;
            let _ = futures_lite::future::race(db.recording_cell.wait(), async {
                cancel_token.wait().await;
                false
            })
            .await;
            if should_cancel() {
                metrics.mark_loop_end();
                return;
            }
            continue;
        }
        let start = Instant::now();
        if generate_real_time {
            // Deadline is set after effective_batch is known for this iteration.
        }
        let tick = tick_counter.load(Ordering::SeqCst);
        let tick_nanos = time_step.as_nanos() * (tick as u128);
        let tick_duration = Duration::from_nanos(tick_nanos.min(u64::MAX as u128) as u64);
        let timestamp = start_timestamp + tick_duration;
        let max_tick = world.world().max_tick();
        let effective_batch = configured_ticks_per_telemetry.min(max_tick.saturating_sub(tick));
        let effective_batch = effective_batch.max(1);
        let effective_batch_time_step =
            Duration::from_secs_f64(time_step.as_secs_f64() * effective_batch as f64);
        if generate_real_time {
            next_tick_deadline.get_or_insert(start + effective_batch_time_step);
        }
        let end_tick = tick.saturating_add(effective_batch.saturating_sub(1));
        let end_tick_nanos = time_step.as_nanos() * (end_tick as u128);
        let end_tick_duration = Duration::from_nanos(end_tick_nanos.min(u64::MAX as u128) as u64);
        let batch_end_timestamp = start_timestamp + end_tick_duration;
        if tick >= max_tick {
            db.recording_cell.set_playing(false);
            world.world_mut().metadata.max_tick = u64::MAX;
            if !interactive {
                metrics.mark_loop_end();
                return;
            } else {
                info!(
                    "Simulation stopped; it reached its max_tick {}.",
                    world.world().max_tick()
                );
            }
        }
        let pre_step_start = Instant::now();
        pre_step(tick, &db, &tick_counter, timestamp, start_timestamp);
        metrics.observe_pre_step(pre_step_start.elapsed());

        if tick_counter.load(Ordering::SeqCst) < tick {
            db.last_updated.store(Timestamp(i64::MIN));
            next_tick_deadline = None;
            continue;
        }

        // Once past warmup, discard the spin-up samples so the summary reflects
        // steady state. Wait until after pre_step/rollback handling so
        // StepContext::truncate() does not mark warmup complete for a run that is
        // about to restart from tick 0.
        if !warmup_done && tick >= warmup_ticks {
            metrics.reset_after_warmup(tick);
            warmup_done = true;
            last_behind_warning = None;
            first_pacing_at = None;
            rate_win_start = None;
            rate_win_start_cycles = 0;
            pacing_cycles = 0;
        }

        let copy_in_start = Instant::now();
        db.with_state(|state| copy_db_to_world(state, &mut world));
        let copy_in_elapsed = copy_in_start.elapsed();
        metrics.observe_copy_db_to_world(copy_in_elapsed);
        let copy_in_ms = if detailed_timing {
            copy_in_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        // Temporarily override so the kernel runs the right number of batched ticks.
        world.world_mut().metadata.ticks_per_telemetry = effective_batch;
        let run_start = Instant::now();
        if let Err(err) = world.run() {
            warn!(?err, "error ticking world");
        }
        let run_elapsed = run_start.elapsed();
        metrics.observe_world_run(run_elapsed);
        let run_ms = if detailed_timing {
            run_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        world.world_mut().metadata.ticks_per_telemetry = configured_ticks_per_telemetry;
        let commit_start = Instant::now();
        db.with_state(|state| {
            if let Err(err) = commit_world_head_unified(
                state,
                &mut world,
                batch_end_timestamp,
                Some(&external_controls),
            ) {
                warn!(?err, "error committing head");
            }
        });
        let commit_elapsed = commit_start.elapsed();
        metrics.observe_commit(commit_elapsed);
        let commit_ms = if detailed_timing {
            commit_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        db.last_updated.update_max(batch_end_timestamp);
        if detailed_timing {
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            tracing::trace!(wall_ms, copy_in_ms, run_ms, commit_ms, "server tick phases",);
        }
        let wait_for_write_start = Instant::now();
        while !wait_for_write_pair_ids.is_empty()
            && !timestamps_changed(&db, &mut wait_for_write_pair_ids).unwrap_or(false)
        {
            stellarator::sleep(Duration::from_millis(1)).await;
            if should_cancel() {
                metrics.observe_wait_for_write(wait_for_write_start.elapsed());
                metrics.mark_loop_end();
                return;
            }
        }
        metrics.observe_wait_for_write(wait_for_write_start.elapsed());
        if should_cancel() {
            metrics.mark_loop_end();
            return;
        }
        // Called with end_tick (not batch start) because the world state now
        // reflects the last tick of the batch.
        let post_step_start = Instant::now();
        post_step(
            end_tick,
            &db,
            &tick_counter,
            batch_end_timestamp,
            start_timestamp,
        );
        metrics.observe_post_step(post_step_start.elapsed());
        if generate_real_time && let Some(deadline) = next_tick_deadline.as_mut() {
            let pacing_start = Instant::now();
            let now = pacing_start;
            // Signed deadline error: negative = ahead of schedule (we will
            // sleep), positive = already behind (we will skip the sleep).
            let deadline_err_ns: i64 = if now >= *deadline {
                i64::try_from(now.duration_since(*deadline).as_nanos()).unwrap_or(i64::MAX)
            } else {
                -i64::try_from(deadline.duration_since(now).as_nanos()).unwrap_or(i64::MAX)
            };
            // Lateness vs the *true* deadline (the real-time violation). Warn
            // only when it is SUSTAINED (a run of late cycles persisting past
            // `warn_window`) and beyond a tolerance, so transient jitter and the
            // boot stall — which the lead buffer + a fast recovery absorb — do
            // not spam the log. The message reports the achieved-vs-target rate
            // and mean lateness over the run, not a single instantaneous miss.
            // Drift cap: re-anchor the schedule on a large one-time gap (the SITL
            // boot stall, a pause/resume, a render-bridge connection) so we don't
            // sprint a long burst of fast ticks to catch up. Transient slow cycles
            // below this stay on the fixed schedule (the lead buffer absorbs them).
            let was_reset = now > *deadline + effective_batch_time_step * 2;

            // Trustworthy warning: compare achieved cycle rate to the target over
            // a recent window, after a startup grace. This is the honest
            // real-time signal and ignores per-cycle jitter / resets / the lead.
            pacing_cycles += 1;
            if first_pacing_at.is_none() {
                first_pacing_at = Some(now);
            }
            let in_grace =
                !warmup_done || now.duration_since(first_pacing_at.unwrap_or(now)) < pacing_grace;
            if in_grace || rate_win_start.is_none() {
                // Pin the window to `now` during grace so it starts clean after.
                rate_win_start = Some(now);
                rate_win_start_cycles = pacing_cycles;
            } else if let Some(win_start) = rate_win_start {
                let win_elapsed = now.duration_since(win_start);
                // Evaluate over a 3 s window and only warn below 85% of target,
                // so a transient stall doesn't trip a "cannot achieve real-time"
                // — it fires only on a genuine, sustained deficit (a sim that is
                // basically keeping up, e.g. ~95%, stays quiet).
                if win_elapsed >= Duration::from_secs(3) {
                    let cycles_in_win = pacing_cycles - rate_win_start_cycles;
                    let achieved_hz = cycles_in_win as f64 / win_elapsed.as_secs_f64();
                    let target_hz = 1.0 / effective_batch_time_step.as_secs_f64();
                    if achieved_hz < 0.85 * target_hz {
                        let should_warn = last_behind_warning
                            .map(|last| now.duration_since(last) >= Duration::from_secs(1))
                            .unwrap_or(true);
                        if should_warn {
                            info!(
                                "simulation cannot achieve real-time; {:.0}/{:.0} cycles/s ({:.0}% of real-time)",
                                achieved_hz,
                                target_hz,
                                100.0 * achieved_hz / target_hz,
                            );
                            last_behind_warning = Some(now);
                        }
                    }
                    rate_win_start = Some(now);
                    rate_win_start_cycles = pacing_cycles;
                }
            }
            if was_reset {
                *deadline = now;
            }
            // Pace to `deadline - lead` (run slightly ahead) rather than to the
            // deadline itself, so a transient slow cycle spends the lead buffer
            // instead of crossing the deadline. `requested` is the sleep we ask
            // the runtime for; `actual` is what really elapsed (their difference
            // is the timer/OS oversleep we record as a diagnostic).
            let target = deadline.checked_sub(pacing_lead).unwrap_or(*deadline);
            let mut requested = Duration::ZERO;
            let mut actual = Duration::ZERO;
            if target > now {
                requested = target.duration_since(now);
                let sleep_start = Instant::now();
                stellarator::sleep(requested).await;
                actual = sleep_start.elapsed();
            }
            *deadline += effective_batch_time_step;
            metrics.observe_real_time_pacing(pacing_start.elapsed());
            metrics.observe_pacing(
                pacing_start.duration_since(start),
                requested,
                actual,
                deadline_err_ns,
                was_reset,
            );
        }
        if detailed_timing {
            let now = Instant::now();
            let should_log = last_timing_log
                .map(|last| now.duration_since(last) >= Duration::from_secs(2))
                .unwrap_or(true);
            if should_log {
                let profile = world.profile();
                let h2d = profile.get("h2d_upload").copied().unwrap_or(0.0);
                let kernel = profile.get("kernel_invoke").copied().unwrap_or(0.0);
                let d2h = profile.get("d2h_download").copied().unwrap_or(0.0);
                let history = profile.get("add_to_history").copied().unwrap_or(0.0);
                let tick_total = profile.get("tick").copied().unwrap_or(0.0);
                let rtf = profile.get("real_time_factor").copied().unwrap_or(0.0);
                info!(
                    "tick breakdown: total={:.3}ms (h2d={:.3}ms kernel={:.3}ms d2h={:.3}ms history={:.3}ms) RTF={:.2}x",
                    tick_total, h2d, kernel, d2h, history, rtf,
                );
                last_timing_log = Some(now);
            }
        }

        if tick_counter.load(Ordering::SeqCst) == tick {
            tick_counter.fetch_add(effective_batch, Ordering::SeqCst);
        }
        // One completed simulation cycle; `effective_batch` is the
        // number of sim steps `world.run()` executed inside it.
        metrics.cycles = metrics.cycles.saturating_add(1);
        metrics.steps = metrics.steps.saturating_add(effective_batch);
        // Record the full wall time of this loop iteration (pre_step
        // through real_time_pacing). Feeds the "mean cycle" value
        // in the summary header.
        metrics.observe_total_cycle(start.elapsed());
    }
}

pub fn external_controls(world: &WorldExec) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world()
        .metadata
        .component_map
        .iter()
        .filter(|(_, (_, component_metadata))| {
            component_metadata
                .metadata
                .get("external_control")
                .map(|s| s == "true")
                .unwrap_or(false)
        })
        .map(|(component_id, _)| *component_id)
}

pub fn wait_for_write(world: &WorldExec) -> impl Iterator<Item = ComponentId> + '_ {
    world
        .world()
        .metadata
        .component_map
        .iter()
        .filter(|(_component_id, (_schema, component_metadata))| {
            component_metadata
                .metadata
                .get("wait_for_write")
                .map(|s| s == "true")
                .unwrap_or(false)
        })
        .map(|(component_id, _)| *component_id)
}

pub fn collect_timestamps(db: &DB, components: &[ComponentId]) -> Vec<(ComponentId, Timestamp)> {
    db.with_state(|state| {
        let mut timestamps = vec![];
        for component_id in components.iter() {
            if let Some(component) = state.get_component(*component_id) {
                if let Some((timestamp, _)) = component.time_series.latest() {
                    timestamps.push((*component_id, *timestamp));
                } else {
                    timestamps.push((*component_id, Timestamp(i64::MIN)));
                }
            } else {
                timestamps.push((*component_id, Timestamp(i64::MIN)));
            }
        }
        timestamps
    })
}

pub fn timestamps_changed(db: &DB, components: &mut [(PairId, Timestamp)]) -> Option<bool> {
    if components.is_empty() {
        return None;
    }
    db.with_state(|state| {
        let mut changed = None;
        for (component_id, timestamp) in components.iter_mut() {
            if let Some(component) = state.get_component(*component_id)
                && let Some((curr_timestamp, _)) = component.time_series.latest()
            {
                if *timestamp != *curr_timestamp {
                    changed = Some(true);
                    *timestamp = *curr_timestamp;
                } else if changed.is_none() {
                    changed = Some(false);
                }
            }
        }
        changed
    })
}
