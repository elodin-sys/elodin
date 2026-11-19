use crate::Error;
use bytemuck;
use elodin_db::{AtomicTimestampExt, DB, State, WatchCursor, handle_conn};
use impeller::types::{ComponentId, Timestamp};
use impeller_wkt::{ComponentMetadata, EntityMetadata};
use std::{
    collections::HashSet,
    path::Path,
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
use crate::prepared::{CallbackCounters, DeferredWrite, HookWrites, PreCommit};
use crate::tick_metrics::Receipt;
use crate::witness::{Channels, Plan};

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
            false,
            Vec::new(),
            Default::default(),
            None,
            false,
            None,
            cancel_token,
        )
        .await
        .map(drop)
    }

    /// Returns the loop's receipt once it stops. `step_callbacks` says whether `pre_step` or
    /// `post_step` call user code, which reads the sim's commits back from the DB within the
    /// cycle. `synchronous_commits` says whether each commit must be in the DB before the next
    /// cycle begins. `witness` records the run's deterministic witness.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_with_cancellation(
        self,
        is_cancelled: impl Fn() -> bool + 'static,
        pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
        step_callbacks: bool,
        synchronous_commits: bool,
        pre_commit: Vec<Box<dyn PreCommit + Send>>,
        counters: Arc<CallbackCounters>,
        witness: Option<Plan>,
        interactive: bool,
        start_timestamp: Option<Timestamp>,
        cancel_token: CancelToken,
    ) -> Result<Option<Receipt>, Error> {
        tracing::info!("running server with cancellation");
        let Self { db, mut world } = self;
        let elodin_db::Server { listener, db } = db;
        let start_time = start_timestamp.unwrap_or_else(Timestamp::now);
        init_db(&db, world.world_mut(), start_time)?;
        if std::env::var_os(impeller_wkt::EDITOR_STARTUP_GATE_ENV).is_some() {
            hold_for_editor_startup(&db);
            info!("simulation held at tick zero until editor startup completes");
        }
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
            step_callbacks,
            synchronous_commits,
            pre_commit,
            counters,
            witness,
            start_time,
            interactive,
            cancel_token,
        ));
        futures_lite::future::race(
            async { stream.join().await.unwrap().unwrap().map(|()| None) },
            async {
                tick.await
                    .map(Some)
                    .map_err(|_| stellarator::Error::JoinFailed)
                    .map_err(Error::from)
            },
        )
        .await
    }
}

fn hold_for_editor_startup(db: &DB) {
    db.with_state_mut(|state| {
        state.db_config.recording = false;
        state.db_config.metadata.insert(
            impeller_wkt::EDITOR_STARTUP_GATE_METADATA_KEY.to_string(),
            "1".to_string(),
        );
    });
    db.recording_cell.set_playing(false);
}

/// Asset key under `{db}/assets/` for the active schematic the editor loads.
const ACTIVE_SCHEMATIC_KEY: &str = "schematics/main.kdl";

/// Ingest the source `assets/` trees into `{db}/assets/` once and register the
/// generated schematic as a stored asset (RFD #724).
///
/// The DB owns all KDL handling: `ingest_asset_dirs` copies the trees and
/// rewrites local asset paths to `db:`, and `store_asset` rewrites the active
/// schematic the same way. This producer does no KDL parsing or rewriting and
/// never mutates `world.metadata.schematic`. Call once before starting the DB
/// Asset Server so early HTTP requests do not 404.
///
/// The active schematic follows the same copy-once contract as the asset tree:
/// on a *fresh* DB the in-memory sim schematic seeds `schematics/main.kdl`, but
/// on a *reopen* the on-disk bytes are authoritative and left untouched — an
/// editor edit (or follower mirror) must not be clobbered by stale in-memory
/// metadata just because the sim was re-run against an existing DB (RFD #724).
///
/// `entry` is the simulation entrypoint (e.g. the `main.py` path). When
/// `ELODIN_ASSETS` is unset, the sim folder's `assets/` and the nearest
/// `assets/` above it are ingested after the cwd's, so a run from the repo
/// root still carries the sim's own assets.
pub fn prime_schematic_assets(
    db: &elodin_db::DB,
    world: &World,
    entry: Option<&Path>,
) -> Result<(), elodin_db::Error> {
    // Whether the DB already holds an active schematic asset *before* this call.
    // Captured pre-ingest so a fresh DB whose source tree happens to ship a
    // `schematics/main.kdl` still gets its freshly generated in-memory schematic,
    // while a reopen preserves whatever bytes are already there.
    let active_existed = elodin_db::assets_http::assets_dir(&db.path)
        .join(ACTIVE_SCHEMATIC_KEY)
        .is_file();

    let sources = elodin_db::assets::resolve_assets_roots(entry);
    if !sources.is_empty() {
        match elodin_db::assets::ingest_asset_dirs(&db.path, &sources) {
            Ok(report) if report.skipped => {
                tracing::info!(?sources, "assets already ingested; skipping")
            }
            Ok(report) => tracing::info!(
                ?sources,
                files = report.file_count,
                bytes = report.byte_count,
                "ingested assets into db"
            ),
            Err(err) => {
                tracing::warn!(?err, ?sources, "failed to ingest assets into db")
            }
        }
    }

    // Display-kernel sidecars are content-addressed (`schematics/kernels/<sha256>`),
    // so storing them on every prime is idempotent and still refreshes hashes
    // produced by an edited `@ui.kernel` even when the copy-once schematic KDL
    // is left untouched on a DB reopen.
    for (key, bytes) in &world.metadata.schematic_kernels {
        if let Err(err) = db.store_asset(key, bytes) {
            tracing::warn!(asset = %key, ?err, "failed to store display kernel sidecar");
        }
    }

    // Reopen: the on-disk active schematic is authoritative (copy-once, same as
    // the asset tree). Never overwrite editor-written bytes from stale in-memory
    // sim metadata. The pointer is equally authoritative: a `DB::open` reopen
    // (the sim serve path) preserves db_config, so an editor's "Save As"/"Open"
    // repoint to another schematic must survive a sim re-run.
    //
    // Detect a reopen from *either* signal, never just the default bytes on
    // disk: an editor "Save As" to a non-default key (e.g. `schematics/foo.kdl`)
    // repoints `schematic.active` and may leave no `schematics/main.kdl` at all.
    // Keying reopen detection on `main.kdl` alone would then miss it, take the
    // fresh-DB path, and clobber the saved pointer with a regenerated default.
    // So a set active pointer marks a reopen regardless of what it points at.
    let active_pointer_set = db.with_state(|s| s.db_config.schematic_active().is_some());
    if active_existed || active_pointer_set {
        // Only re-assert the default pointer when it is missing *and* the
        // default bytes exist on disk — a `DB::create` reopen resets db_config
        // and drops the pointer while the bytes survive. Never point at a
        // missing `main.kdl`, and never override an existing pointer.
        if !active_pointer_set
            && active_existed
            && let Err(err) = db.set_active_schematic(ACTIVE_SCHEMATIC_KEY)
        {
            tracing::warn!(?err, "failed to set active schematic pointer");
        }
        return Ok(());
    }

    // Fresh DB: register the generated schematic as a stored asset. `store_asset`
    // rewrites its asset paths to `db:` (unparsable KDL is stored verbatim), then
    // `set_active_schematic` points `schematic.active` at it. Failures are
    // logged, not fatal: a missing schematic asset must not abort the sim.
    let Some(content) = world.metadata.schematic.as_deref() else {
        // No in-memory schematic: if ingest just seeded the conventional
        // `schematics/main.kdl` and no pointer is set yet, point at it —
        // matching the CLI fallback in `elodin-db run` — so connected editors
        // still auto-load a schematic over HTTP.
        if !db.with_state(|s| s.db_config.schematic_active().is_some()) {
            match db.set_active_schematic(ACTIVE_SCHEMATIC_KEY) {
                Ok(()) => tracing::info!(
                    key = ACTIVE_SCHEMATIC_KEY,
                    "set active schematic from ingested assets"
                ),
                Err(err) => tracing::debug!(
                    ?err,
                    "no ingested {ACTIVE_SCHEMATIC_KEY} to set active; skipping"
                ),
            }
        }
        return Ok(());
    };
    // Pull window sub-schematics referenced outside the ingested tree (e.g.
    // `window path="examples/drone/panel.kdl"`) into `{db}/assets/schematics/`
    // and rewrite the references to `db:` keys, so the stored schematic is
    // self-contained on followers and other machines (RFD #724). Failures fall
    // back to storing the content as-is: a missing window file must not abort
    // the sim, it just stays a machine-local reference.
    let assets_dir = elodin_db::assets_http::assets_dir(&db.path);
    let search_dirs = elodin_db::assets::schematic_window_search_dirs(entry);
    let content =
        match elodin_db::assets::ingest_window_schematics(&assets_dir, content, &search_dirs) {
            Ok(Some(rewritten)) => std::borrow::Cow::Owned(rewritten),
            Ok(None) => std::borrow::Cow::Borrowed(content),
            Err(err) => {
                tracing::warn!(
                    ?err,
                    "failed to ingest window sub-schematics; storing as-is"
                );
                std::borrow::Cow::Borrowed(content)
            }
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
            if component_metadata.is_transient() {
                continue;
            }
            let Some(column) = world.host.get(component_id) else {
                continue;
            };
            let size = schema.size();
            let entity_ids =
                bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
            for (i, entity_id) in entity_ids.iter().enumerate() {
                let offset = i * size;
                let entity_id = impeller::types::EntityId(*entity_id);
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
        if !world.metadata.thermal_tags.is_empty() {
            match serde_json::to_string(&world.metadata.thermal_tags) {
                Ok(json) => {
                    state
                        .db_config
                        .metadata
                        .insert("thermal_tags".to_string(), json);
                }
                Err(e) => {
                    tracing::warn!("Failed to serialize thermal_tags metadata: {e}");
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
        state.db_config.metadata.insert(
            impeller_wkt::DbConfig::DEFAULT_PLAYBACK_SPEED_KEY.to_string(),
            world.metadata.default_playback_speed.to_string(),
        );
    });
    let _ = db.save_db_state();

    Ok(())
}

/// The DB component behind each (entity, component) pair of the world, resolved once after
/// `init_db`, with a copy of each pair's latest DB sample. A watch cursor on every pair says
/// which ones others appended to, so a pull reads only those, and a commit compares only the
/// columns the cycle can have changed.
pub struct DbSync {
    /// One entry per `world.host` column, in its order.
    columns: Vec<SyncColumn>,
    /// Marked by every append to a pair but this sync's own; rows take slots in column order.
    cursor: Arc<WatchCursor>,
    /// `DB::vtable_gen` when the shadows were last read; truncating the DB bumps it.
    vtable_gen: u64,
    /// Set by a full rescan so the next commit compares every column.
    compare_all: bool,
    /// Commits so far, which `Cadence::Every` counts.
    commits: u64,
    writer: Option<WriteBehind>,
    /// The slots the last commit recorded, once a witness asks for them.
    recorded: Option<Vec<usize>>,
    pub counters: SyncCounters,
}

/// What [`DbSync`] did, for the run receipt.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SyncCounters {
    /// `synchronous` when each commit is in the DB before the next cycle, else `write_behind`.
    pub commits: &'static str,
    /// (entity, component) pairs mirrored between the world and the DB.
    pub rows_registered: u64,
    /// Rows whose latest DB sample a pull or commit read.
    pub rows_scanned: u64,
    /// Pulls or commits that read every row, as the first does and those after a truncate.
    pub full_scans: u64,
    /// Rows a pull drained from the watch cursor, each appended to by another writer.
    pub rows_drained: u64,
    /// Pulls that found rows to drain.
    pub cursor_wakeups: u64,
    /// Appends to rows already awaiting a pull, which applies only the newest.
    pub writes_coalesced: u64,
    /// Rows a pull copied into the world.
    pub rows_pulled: u64,
    /// Rows a commit compared with their latest DB sample.
    pub rows_compared: u64,
    pub rows_committed: u64,
    /// Compared rows equal to their latest DB sample, so not appended.
    pub writes_suppressed: u64,
    pub bytes_pulled: u64,
    pub bytes_committed: u64,
    /// Present when a writer thread appends the commits.
    pub writer: Option<WriterStats>,
}

/// What the DB writer thread did with the commits it was handed.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct WriterStats {
    pub batches: u64,
    /// Time spent appending.
    pub write_ns: u64,
    pub max_write_ns: u64,
    /// From a commit queuing its rows to the writer finishing them.
    pub max_lag_ns: u64,
    /// Most commits queued or being written at once.
    pub queue_high_water: u64,
    /// Commits that found the queue full and waited for room.
    pub backpressure_events: u64,
    pub backpressure_ns: u64,
}

/// When a commit records a column, from its `recording` and `record_every_tick` metadata.
#[derive(Clone, Copy, PartialEq)]
enum Cadence {
    OnChange,
    EveryCommit,
    Every(u64),
    Manual,
    InitialOnly,
}

impl Cadence {
    fn of(metadata: &ComponentMetadata) -> Self {
        let recording = metadata.metadata.get("recording").map(String::as_str);
        match recording {
            Some("manual") => Self::Manual,
            Some("initial_only") => Self::InitialOnly,
            _ if metadata.record_every_tick() => Self::EveryCommit,
            _ => recording
                .and_then(|r| r.strip_prefix("every:")?.parse().ok())
                .map_or(Self::OnChange, Self::Every),
        }
    }
}

struct SyncColumn {
    component_id: ComponentId,
    size: usize,
    cadence: Cadence,
    excluded: bool,
    /// A system writes it, so a commit compares it every cycle.
    written: bool,
    /// The cursor slot of `rows[0]`.
    first_slot: usize,
    rows: Vec<SyncRow>,
    /// Each row's latest DB sample, laid out like the world column.
    shadow: Vec<u8>,
}

struct SyncRow {
    /// Byte offset of the entity's row in the column.
    offset: usize,
    component: elodin_db::Component,
    /// Whether the pair had a sample when its shadow was last read or written.
    has_sample: bool,
}

impl SyncRow {
    fn read(&mut self, shadow: &mut [u8]) {
        self.has_sample = match self.component.time_series.latest() {
            Some((_, head)) => {
                shadow.copy_from_slice(head);
                true
            }
            None => false,
        };
    }
}

impl DbSync {
    /// Transient components get no rows. Systems write the `written` components; `exclusions`
    /// are copied in but never committed.
    pub fn new(
        state: &State,
        world: &World,
        written: &[ComponentId],
        exclusions: &HashSet<ComponentId>,
    ) -> Self {
        let mut slots = 0;
        let columns = world
            .host
            .iter()
            .map(|(component_id, column)| {
                let mut sync = SyncColumn {
                    component_id: *component_id,
                    size: 0,
                    cadence: Cadence::OnChange,
                    excluded: exclusions.contains(component_id),
                    written: written.contains(component_id),
                    first_slot: slots,
                    rows: Vec::new(),
                    shadow: column.buffer.clone(),
                };
                let Some((schema, metadata)) = world.metadata.component_map.get(component_id)
                else {
                    return sync;
                };
                if metadata.is_transient() {
                    return sync;
                }
                let size = schema.size();
                let entity_ids =
                    bytemuck::try_cast_slice::<_, u64>(column.entity_ids.as_slice()).unwrap();
                sync.rows = entity_ids
                    .iter()
                    .enumerate()
                    .filter_map(|(i, entity_id)| {
                        let entity_id = impeller::types::EntityId(*entity_id);
                        let entity = world.metadata.entity_metadata.get(&entity_id)?;
                        let pair_id = ComponentId::from_pair(&entity.name, &metadata.name);
                        Some(SyncRow {
                            offset: i * size,
                            component: state.get_component(pair_id)?.clone(),
                            has_sample: false,
                        })
                    })
                    .collect();
                sync.size = size;
                sync.cadence = Cadence::of(metadata);
                slots += sync.rows.len();
                sync
            })
            .collect::<Vec<_>>();
        let cursor = Arc::new(WatchCursor::new(slots));
        for (slot, row) in columns.iter().flat_map(|c| &c.rows).enumerate() {
            row.component.time_series.watch(&cursor, slot);
        }
        let counters = SyncCounters {
            commits: "synchronous",
            rows_registered: slots as u64,
            ..Default::default()
        };
        Self {
            columns,
            cursor,
            vtable_gen: u64::MAX,
            compare_all: false,
            commits: 0,
            writer: None,
            recorded: None,
            counters,
        }
    }

    /// The cursor slot of row `row` of `component_id`'s column, if it has DB history.
    pub fn slot_of(&self, component_id: ComponentId, row: usize) -> Option<usize> {
        let sync = self
            .columns
            .iter()
            .find(|c| c.component_id == component_id)?;
        let index = sync.rows.iter().position(|r| r.offset == row * sync.size)?;
        Some(sync.first_slot + index)
    }

    /// Keeps the slots each commit records, for `recorded_slots`.
    pub fn track_recorded(&mut self) {
        self.recorded = Some(Vec::new());
    }

    pub fn recorded_slots(&self) -> &[usize] {
        self.recorded.as_deref().unwrap_or_default()
    }

    /// Hands commits to a writer thread from now on. Their rows reach the DB, and
    /// `last_updated` moves past them, up to a few cycles later.
    pub fn write_behind(&mut self, db: Arc<DB>) {
        let series = self.columns.iter().flat_map(|c| &c.rows);
        let series = series.map(|row| row.component.clone()).collect();
        self.writer = Some(WriteBehind::spawn(db, series, self.cursor.clone()));
        self.counters.writer = Some(WriterStats::default());
        self.counters.commits = "write_behind";
    }

    pub fn writes_behind(&self) -> bool {
        self.writer.is_some()
    }

    /// Waits for the writer thread, if any, to append every queued commit, and keeps it running.
    pub fn wait_for_writes(&mut self) {
        if let (Some(writer), Some(stats)) = (&mut self.writer, &mut self.counters.writer) {
            writer.wait(stats);
        }
    }

    /// Waits for the writer thread to append every queued commit.
    pub fn finish_writes(&mut self) {
        if let (Some(writer), Some(stats)) = (self.writer.take(), &mut self.counters.writer) {
            writer.finish(stats);
        }
    }

    /// Restarts the counters, as the loop does when warmup ends.
    pub fn reset_counters(&mut self) {
        self.counters = SyncCounters {
            commits: self.counters.commits,
            rows_registered: self.counters.rows_registered,
            writer: self
                .counters
                .writer
                .as_ref()
                .map(|_| WriterStats::default()),
            ..Default::default()
        };
    }

    /// Rereads every row when `vtable_gen` moved, since a truncate can leave a pair with as
    /// many samples as before but different values; the first call always does.
    fn rescan(&mut self, vtable_gen: u64) -> bool {
        if vtable_gen == self.vtable_gen {
            return false;
        }
        self.vtable_gen = vtable_gen;
        self.compare_all = true;
        self.cursor.drain(|_| {});
        for sync in &mut self.columns {
            for row in &mut sync.rows {
                row.read(&mut sync.shadow[row.offset..row.offset + sync.size]);
            }
            self.counters.rows_scanned += sync.rows.len() as u64;
        }
        self.counters.full_scans += 1;
        true
    }

    /// Copies into the world the latest DB sample of each pair another writer appended to.
    /// Excluded columns that systems write are restored every pull, as commits never record
    /// them.
    pub fn copy_db_to_world(&mut self, world: &mut World, vtable_gen: u64) {
        let full = self.rescan(vtable_gen);
        let Self {
            columns,
            cursor,
            counters,
            ..
        } = self;
        if !full {
            let drained = cursor.drain(|slot| {
                let column = columns.partition_point(|c| c.first_slot <= slot) - 1;
                let sync = &mut columns[column];
                let row = &mut sync.rows[slot - sync.first_slot];
                let range = row.offset..row.offset + sync.size;
                row.read(&mut sync.shadow[range.clone()]);
                counters.rows_scanned += 1;
                if row.has_sample {
                    let column = world.host.get_mut(&sync.component_id).unwrap();
                    column.buffer[range.clone()].copy_from_slice(&sync.shadow[range]);
                    counters.rows_pulled += 1;
                    counters.bytes_pulled += sync.size as u64;
                }
            });
            counters.rows_drained += drained as u64;
            counters.cursor_wakeups += u64::from(drained > 0);
            counters.writes_coalesced += cursor.take_coalesced();
        }
        for ((id, column), sync) in world.host.iter_mut().zip(columns.iter()) {
            debug_assert_eq!(*id, sync.component_id);
            if !full && !(sync.excluded && sync.written) {
                continue;
            }
            for row in sync.rows.iter().filter(|row| row.has_sample) {
                let range = row.offset..row.offset + sync.size;
                column.buffer[range.clone()].copy_from_slice(&sync.shadow[range]);
                counters.rows_pulled += 1;
                counters.bytes_pulled += sync.size as u64;
            }
        }
    }

    /// Appends the pairs each column's cadence says are due. On-change columns are due when
    /// systems or the `pre_commit` hook wrote them, or after a rescan, and append the pairs whose
    /// world value differs from their latest DB sample. Excluded columns are skipped unless the
    /// hook wrote them.
    pub fn commit_world(
        &mut self,
        world: &World,
        timestamp: Timestamp,
        vtable_gen: u64,
        hook_writes: &HookWrites,
    ) -> Result<(), Error> {
        self.rescan(vtable_gen);
        let compare_all = std::mem::take(&mut self.compare_all);
        let commit = self.commits;
        self.commits += 1;
        let Self {
            columns,
            cursor,
            writer,
            recorded,
            counters,
            ..
        } = self;
        if let Some(recorded) = recorded {
            recorded.clear();
        }
        for ((id, column), sync) in world.host.iter().zip(columns) {
            debug_assert_eq!(*id, sync.component_id);
            let hook_wrote = hook_writes.written.contains(id);
            let due = match sync.cadence {
                Cadence::OnChange => compare_all || hook_wrote || sync.written,
                Cadence::EveryCommit => true,
                Cadence::Every(n) => commit.is_multiple_of(n),
                Cadence::Manual => hook_writes.recorded.contains(id),
                Cadence::InitialOnly => compare_all,
            };
            if !due || (sync.excluded && !hook_wrote) {
                continue;
            }
            for (slot, row) in (sync.first_slot..).zip(&mut sync.rows) {
                let range = row.offset..row.offset + sync.size;
                // Another writer appended since the pull; compare with its value.
                if cursor.is_marked(slot) {
                    row.read(&mut sync.shadow[range.clone()]);
                    counters.rows_scanned += 1;
                }
                let buf = &column.buffer[range.clone()];
                let latest = row.has_sample.then_some(&sync.shadow[range.clone()]);
                counters.rows_compared += 1;
                if should_record(sync.cadence, latest, buf) {
                    match writer {
                        Some(writer) => writer.batch.push(slot, buf),
                        None => {
                            row.component
                                .time_series
                                .push_buf_from(Some(cursor), timestamp, buf)?
                        }
                    }
                    sync.shadow[range].copy_from_slice(buf);
                    row.has_sample = true;
                    if let Some(recorded) = recorded {
                        recorded.push(slot);
                    }
                    counters.rows_committed += 1;
                    counters.bytes_committed += sync.size as u64;
                } else {
                    counters.writes_suppressed += 1;
                }
            }
        }
        if let (Some(writer), Some(stats)) = (writer, &mut counters.writer) {
            writer.send(timestamp, self.vtable_gen, stats)?;
        }
        Ok(())
    }
}

/// The rows one commit recorded, for the writer thread to append.
#[derive(Default)]
struct CommitBatch {
    timestamp: Timestamp,
    vtable_gen: u64,
    /// Cursor slot of each row and the end of its value in `bytes`.
    rows: Vec<(usize, usize)>,
    bytes: Vec<u8>,
    queued: Option<Instant>,
    /// Set by the writer: how long the appends took, and when they finished.
    written: Option<(Duration, Instant)>,
}

impl CommitBatch {
    fn push(&mut self, slot: usize, value: &[u8]) {
        self.bytes.extend_from_slice(value);
        self.rows.push((slot, self.bytes.len()));
    }

    fn account(&self, stats: &mut WriterStats) {
        if let (Some(queued), Some((took, end))) = (self.queued, self.written) {
            stats.batches += 1;
            stats.write_ns += took.as_nanos() as u64;
            stats.max_write_ns = stats.max_write_ns.max(took.as_nanos() as u64);
            stats.max_lag_ns = stats.max_lag_ns.max((end - queued).as_nanos() as u64);
        }
    }
}

/// Appends commits on a thread of its own, so DB stalls such as page faults and writeback
/// throttling stay off the sim thread. Written batches come back for reuse.
struct WriteBehind {
    queue: std::sync::mpsc::SyncSender<CommitBatch>,
    /// Batches the thread finished, and the signal it gives for each.
    written: Arc<(std::sync::Mutex<Vec<CommitBatch>>, std::sync::Condvar)>,
    thread: std::thread::JoinHandle<()>,
    batch: CommitBatch,
    spare: Vec<CommitBatch>,
    in_flight: u64,
}

impl WriteBehind {
    /// Commits that can wait before the sim thread blocks on the writer: enough at 1 kHz to
    /// ride out writeback stalls of hundreds of milliseconds.
    const QUEUE: usize = 1024;

    /// `series[slot]` is the pair behind each cursor slot.
    fn spawn(db: Arc<DB>, series: Vec<elodin_db::Component>, cursor: Arc<WatchCursor>) -> Self {
        let (queue, queued) = std::sync::mpsc::sync_channel::<CommitBatch>(Self::QUEUE);
        let written = Arc::new((std::sync::Mutex::new(Vec::new()), std::sync::Condvar::new()));
        let written_by_thread = written.clone();
        let thread = std::thread::Builder::new()
            .name("elodin-db-writer".into())
            .spawn(move || {
                for mut batch in queued {
                    let start = Instant::now();
                    // A truncate since the commit discards it, as it would have a written one.
                    if batch.vtable_gen == db.vtable_gen.latest() {
                        db.with_state(|_| {
                            let mut from = 0;
                            for &(slot, to) in &batch.rows {
                                let value = &batch.bytes[from..to];
                                let time_series = &series[slot].time_series;
                                if let Err(err) =
                                    time_series.push_buf_from(Some(&cursor), batch.timestamp, value)
                                {
                                    warn!(?err, "error committing head");
                                    break;
                                }
                                from = to;
                            }
                        });
                        db.last_updated.update_max(batch.timestamp);
                    }
                    let end = Instant::now();
                    batch.written = Some((end - start, end));
                    let (written, signal) = &*written_by_thread;
                    written.lock().unwrap().push(batch);
                    signal.notify_one();
                }
            })
            .expect("failed to spawn the DB writer thread");
        Self {
            queue,
            written,
            thread,
            batch: CommitBatch::default(),
            spare: Vec::new(),
            in_flight: 0,
        }
    }

    /// Queues the rows pushed since the last call, as a commit at `timestamp`.
    fn send(
        &mut self,
        timestamp: Timestamp,
        vtable_gen: u64,
        stats: &mut WriterStats,
    ) -> Result<(), Error> {
        self.reclaim(stats);
        let next = self.spare.pop().unwrap_or_default();
        let mut batch = std::mem::replace(&mut self.batch, next);
        batch.timestamp = timestamp;
        batch.vtable_gen = vtable_gen;
        batch.queued = Some(Instant::now());
        self.in_flight += 1;
        stats.queue_high_water = stats.queue_high_water.max(self.in_flight);
        let exited = || std::io::Error::other("the DB writer thread exited");
        match self.queue.try_send(batch) {
            Ok(()) => Ok(()),
            Err(std::sync::mpsc::TrySendError::Full(batch)) => {
                let wait = Instant::now();
                self.queue.send(batch).map_err(|_| exited())?;
                stats.backpressure_events += 1;
                stats.backpressure_ns += wait.elapsed().as_nanos() as u64;
                Ok(())
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => Err(exited().into()),
        }
    }

    /// Takes back the batches the thread finished, for reuse.
    fn reclaim(&mut self, stats: &mut WriterStats) {
        for mut batch in self.written.0.lock().unwrap().drain(..) {
            self.in_flight -= 1;
            batch.account(stats);
            batch.rows.clear();
            batch.bytes.clear();
            self.spare.push(batch);
        }
    }

    /// Waits until the thread has appended every queued commit.
    fn wait(&mut self, stats: &mut WriterStats) {
        let (written, signal) = &*self.written;
        let finished = written.lock().unwrap();
        let in_flight = self.in_flight as usize;
        drop(signal.wait_while(finished, |finished| finished.len() < in_flight));
        self.reclaim(stats);
    }

    fn finish(self, stats: &mut WriterStats) {
        drop(self.queue);
        let _ = self.thread.join();
        for batch in self.written.0.lock().unwrap().iter() {
            batch.account(stats);
        }
    }
}

/// A cursor every append to `pairs` marks, slot by slot.
fn watch_pairs(state: &State, pairs: &[PairId]) -> Arc<WatchCursor> {
    let cursor = Arc::new(WatchCursor::new(pairs.len()));
    for (slot, pair) in pairs.iter().enumerate() {
        if let Some(component) = state.get_component(*pair) {
            component.time_series.watch(&cursor, slot);
        }
    }
    cursor
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
                let entity_id = impeller::types::EntityId(*entity_id);
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

/// Whether a due row is appended: on-change rows only when changed, initial-only rows only when
/// the DB lost their sample.
fn should_record(cadence: Cadence, latest: Option<&[u8]>, value: &[u8]) -> bool {
    match cadence {
        Cadence::OnChange => latest != Some(value),
        Cadence::InitialOnly => latest.is_none(),
        Cadence::EveryCommit | Cadence::Every(_) | Cadence::Manual => true,
    }
}

fn reanchored_deadline(now: Instant, pacing_lead: Duration) -> Instant {
    now.checked_add(pacing_lead).unwrap_or(now)
}

fn initial_deadline(now: Instant, pacing_lead: Duration, period: Duration) -> Instant {
    reanchored_deadline(now, pacing_lead)
        .checked_add(period)
        .unwrap_or(now)
}

#[allow(clippy::too_many_arguments)]
async fn tick(
    db: Arc<DB>,
    tick_counter: Arc<AtomicU64>,
    mut world: WorldExec,
    is_cancelled: impl Fn() -> bool + 'static,
    pre_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    post_step: impl Fn(u64, &Arc<DB>, &Arc<AtomicU64>, Timestamp, Timestamp) + 'static,
    step_callbacks: bool,
    synchronous_commits: bool,
    mut pre_commit: Vec<Box<dyn PreCommit + Send>>,
    counters: Arc<CallbackCounters>,
    witness: Option<Plan>,
    start_timestamp: Timestamp,
    interactive: bool,
    cancel_token: CancelToken,
) -> Receipt {
    let external_controls: HashSet<ComponentId> = external_controls(&world).collect();
    let mut sync = db.with_state(|state| {
        DbSync::new(
            state,
            world.world(),
            &world.metadata().ret_ids,
            &external_controls,
        )
    });
    if witness.is_some() {
        sync.track_recorded();
    }
    let mut witness =
        witness.map(|plan| plan.start(start_timestamp, |id, row| sync.slot_of(id, row)));
    let mut no_channels = Channels::default();
    let wait_for_write: Vec<ComponentId> = wait_for_write(&world).collect();
    let wait_for_write_pair_ids: Vec<PairId> = get_pair_ids(&world, &wait_for_write).unwrap();
    let write_barrier = (!wait_for_write_pair_ids.is_empty())
        .then(|| db.with_state(|state| watch_pairs(state, &wait_for_write_pair_ids)));
    // Callbacks and barrier writers read the sim's commits from the DB, so those wait for them.
    if !step_callbacks && write_barrier.is_none() {
        sync.write_behind(db.clone());
    }
    // Synchronous commits then land while the loop paces: each cycle waits for the last one's.
    let wait_for_commits = synchronous_commits && sync.writes_behind();
    if synchronous_commits {
        sync.counters.commits = "synchronous";
    }
    // Deferred writes wait for their cycle's commit to land, with the tick they were made at.
    let mut pending: Option<(u64, Vec<(usize, DeferredWrite)>)> = None;
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
    let max_tick = world.world().max_tick();
    if max_tick != u64::MAX {
        metrics.requested_ticks = Some(max_tick);
        metrics.record_cycles(Some(max_tick.div_ceil(configured_ticks_per_telemetry)));
    } else {
        metrics.record_cycles(None);
    }
    metrics.backend = Some(world.backend_report());
    'cycles: loop {
        if should_cancel() {
            break;
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
                break;
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
            next_tick_deadline.get_or_insert(initial_deadline(
                start,
                pacing_lead,
                effective_batch_time_step,
            ));
        }
        let end_tick = tick.saturating_add(effective_batch.saturating_sub(1));
        let end_tick_nanos = time_step.as_nanos() * (end_tick as u128);
        let end_tick_duration = Duration::from_nanos(end_tick_nanos.min(u64::MAX as u128) as u64);
        let batch_end_timestamp = start_timestamp + end_tick_duration;
        if tick >= max_tick {
            db.recording_cell.set_playing(false);
            world.world_mut().metadata.max_tick = u64::MAX;
            if !interactive {
                break;
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
            sync.reset_counters();
            counters.reset_activity();
            pre_commit.iter_mut().for_each(|hook| hook.reset_report());
            if let Some(witness) = &mut witness {
                witness.reset_report();
            }
            if world.probed() {
                cranelift_mlir::profile::reset_timings();
            }
            warmup_done = true;
            last_behind_warning = None;
            first_pacing_at = None;
            rate_win_start = None;
            rate_win_start_cycles = 0;
            pacing_cycles = 0;
        }

        if world.world().metadata.workers > 1 {
            cranelift_mlir::pool::wake();
        }
        let copy_in_start = Instant::now();
        // Holding the state lock keeps rows that gRPC ingest applies under the write lock whole.
        db.with_state(|_| sync.copy_db_to_world(world.world_mut(), db.vtable_gen.latest()));
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
        if let Some(witness) = &mut witness {
            witness.after_systems(end_tick, batch_end_timestamp, world.world());
        }
        // The writer had the systems' step, and the pacing before it, to land the last commit.
        let mut commit_wait = Duration::ZERO;
        if wait_for_commits {
            let wait_start = Instant::now();
            sync.wait_for_writes();
            commit_wait = wait_start.elapsed();
            let landing = Instant::now();
            if let Some(cycle) = pending.take()
                && !land_deferred(&db, cycle, &counters, &pre_commit, &mut metrics)
            {
                break;
            }
            metrics.observe_deferred_writes(landing.elapsed());
        }
        let mut hook_writes = HookWrites::default();
        let mut deferred = Vec::new();
        if !pre_commit.is_empty() {
            let pre_commit_start = Instant::now();
            let step = pre_commit
                .iter_mut()
                .enumerate()
                .try_for_each(|(index, hook)| {
                    let channels = witness
                        .as_mut()
                        .map_or(&mut no_channels, |w| &mut w.channels);
                    let writes = hook
                        .step(end_tick, batch_end_timestamp, world.world_mut(), channels)
                        .map_err(|cause| (index, cause))?;
                    if let Some(witness) = &mut witness {
                        witness.after_hook(index, world.world(), &writes);
                    }
                    hook_writes.written.extend(writes.written);
                    hook_writes.recorded.extend(writes.recorded);
                    deferred.extend(writes.deferred.into_iter().map(|write| (index, write)));
                    Ok(())
                });
            metrics.observe_pre_commit(pre_commit_start.elapsed());
            if let Err((index, cause)) = step {
                let when = format!("at tick {end_tick}");
                metrics.fail(index, pre_commit[index].name(), &when, &cause, false);
                break;
            }
        }
        let commit_start = Instant::now();
        db.with_state(|_| {
            if let Err(err) = sync.commit_world(
                world.world(),
                batch_end_timestamp,
                db.vtable_gen.latest(),
                &hook_writes,
            ) {
                warn!(?err, "error committing head");
            }
        });
        let commit_elapsed = commit_start.elapsed();
        // A cycle's commit time includes waiting for the previous commit to land.
        metrics.observe_commit(commit_wait + commit_elapsed);
        let commit_ms = if detailed_timing {
            commit_elapsed.as_secs_f64() * 1000.0
        } else {
            0.0
        };
        if !sync.writes_behind() {
            db.last_updated.update_max(batch_end_timestamp);
        }
        let cycle = (end_tick, deferred);
        if wait_for_commits {
            pending = Some(cycle);
        } else {
            let landing = Instant::now();
            if !land_deferred(&db, cycle, &counters, &pre_commit, &mut metrics) {
                break;
            }
            metrics.observe_deferred_writes(landing.elapsed());
        }
        if let Some(witness) = &mut witness
            && (witness.after_commit(sync.recorded_slots()).is_err() || witness.should_stop())
        {
            break;
        }
        if detailed_timing {
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            tracing::trace!(wall_ms, copy_in_ms, run_ms, commit_ms, "server tick phases",);
        }
        let wait_for_write_start = Instant::now();
        while let Some(barrier) = &write_barrier
            && barrier.drain(|_| {}) == 0
        {
            // The timeout only bounds how long a cancel goes unnoticed.
            let timeout = stellarator::sleep(Duration::from_millis(1));
            futures_lite::future::race(barrier.changed(), timeout).await;
            if should_cancel() {
                metrics.observe_wait_for_write(wait_for_write_start.elapsed());
                break 'cycles;
            }
        }
        metrics.observe_wait_for_write(wait_for_write_start.elapsed());
        if should_cancel() {
            break;
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
                *deadline = reanchored_deadline(now, pacing_lead);
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
    sync.finish_writes();
    if let Some(cycle) = pending.take() {
        land_deferred(&db, cycle, &counters, &pre_commit, &mut metrics);
    }
    if let Some(witness) = witness {
        let report = witness.finish(metrics.failure.clone());
        if let Some(failure) = &report.failure {
            metrics.fail_witness(failure);
        }
        metrics.witness = Some(report);
    }
    metrics.sync = Some(sync.counters.clone());
    metrics.timing = world.world().metadata.timing;
    metrics.systems = world.probed().then(cranelift_mlir::profile::timings);
    metrics.callbacks = Some(crate::prepared::CallbackReport {
        hooks: pre_commit.iter().filter_map(|hook| hook.report()).collect(),
        ..counters.report()
    });
    metrics.mark_loop_end();
    metrics.receipt()
}

/// Appends the deferred writes made at `tick`, now that its commit landed. On a failure, records
/// in `metrics` which hook made the write, and returns false.
fn land_deferred(
    db: &DB,
    (tick, writes): (u64, Vec<(usize, DeferredWrite)>),
    counters: &CallbackCounters,
    hooks: &[Box<dyn PreCommit + Send>],
    metrics: &mut crate::tick_metrics::TickMetrics,
) -> bool {
    let Err((index, cause)) = apply_deferred(db, &writes, counters) else {
        return true;
    };
    let when = format!("after the commit at tick {tick}");
    metrics.fail(index, hooks[index].name(), &when, &cause, true);
    false
}

/// Appends the cycle's deferred writes, each tagged with the hook that made it, in order.
/// Appending does not move `last_updated`. Fails with the hook of the write that failed.
fn apply_deferred(
    db: &DB,
    deferred: &[(usize, DeferredWrite)],
    counters: &CallbackCounters,
) -> Result<(), (usize, String)> {
    if deferred.is_empty() {
        return Ok(());
    }
    db.with_state(|state| {
        for (hook, write) in deferred {
            for (name, id, value) in &write.rows {
                let appended = match state.get_component(*id) {
                    Some(component) => component.time_series.push_buf(write.timestamp, value),
                    None => Err(elodin_db::Error::ComponentNotFound(*id)),
                };
                if let Err(err) = appended {
                    counters.deferred_failures.fetch_add(1, Ordering::Relaxed);
                    let at = write.timestamp.0;
                    return Err((*hook, format!("cannot append {name:?} at {at}: {err}")));
                }
                counters.deferred_rows.fetch_add(1, Ordering::Relaxed);
                counters
                    .deferred_bytes
                    .fetch_add(value.len() as u64, Ordering::Relaxed);
            }
            counters.deferred_batches.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    })
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

#[cfg(test)]
mod pacing_tests {
    use super::*;

    #[test]
    fn drift_reset_restores_the_configured_lead() {
        let now = Instant::now();
        let lead = Duration::from_millis(20);
        assert_eq!(reanchored_deadline(now, lead), now + lead);
    }

    #[test]
    fn initial_pacing_preserves_the_lead_without_sprinting() {
        let now = Instant::now();
        let lead = Duration::from_millis(20);
        let period = Duration::from_millis(1);
        assert_eq!(initial_deadline(now, lead, period) - lead, now + period);
    }
}

#[cfg(test)]
mod commit_tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn sparse_by_default_with_dense_opt_in() {
        let value = [1, 2, 3];
        let changed = [1, 2, 4];
        assert!(!should_record(Cadence::OnChange, Some(&value), &value));
        assert!(should_record(Cadence::OnChange, Some(&value), &changed));
        assert!(should_record(Cadence::OnChange, None, &value));
        assert!(should_record(Cadence::EveryCommit, Some(&value), &value));
    }

    #[test]
    fn db_sync_records_each_cadence_when_due() {
        let temp = tempdir().unwrap();
        let names = ["every", "manual", "initial"];
        let components = names.map(|name| (name, None));
        let (db, mut world) = scalar_world(&temp.path().join("db"), &components);
        for (name, recording) in names.into_iter().zip(["every:2", "manual", "initial_only"]) {
            let (_, metadata) = world
                .metadata
                .component_map
                .get_mut(&ComponentId::new(name))
                .unwrap();
            metadata
                .metadata
                .insert("recording".into(), recording.into());
        }
        let written = names.map(ComponentId::new);
        let mut sync = db.with_state(|state| DbSync::new(state, &world, &written, &HashSet::new()));
        for commit in 0..4 {
            for name in names {
                set(&mut world, name, commit as f64 + 1.0);
            }
            let recorded = HashSet::from_iter((commit == 3).then(|| ComponentId::new("manual")));
            let hook = HookWrites {
                recorded,
                ..Default::default()
            };
            let at = Timestamp(1000 * (commit + 1));
            sync.commit_world(&world, at, db.vtable_gen.latest(), &hook)
                .unwrap();
        }
        // Each has init_db's sample; `every` adds commits 0 and 2, `manual` the one it asked for.
        let samples = names.map(|name| series(&db, name).time_series.sample_count());
        assert_eq!(samples, [3, 2, 1]);
    }

    #[test]
    fn transient_component_is_not_copied_from_db() {
        let temp = tempdir().unwrap();
        let db = DB::create(temp.path().join("db")).unwrap();
        let component_id = ComponentId::new("scratch");
        let pair_id = ComponentId::from_pair("e1", "scratch");
        let schema = elodin_db::ComponentSchema {
            prim_type: impeller::types::PrimType::F64,
            dim: [1].into_iter().collect(),
        };

        db.with_state_mut(|state| {
            state.set_component_metadata(
                ComponentMetadata {
                    component_id: pair_id,
                    name: "e1.scratch".into(),
                    metadata: HashMap::new(),
                },
                &db.path,
            )?;
            state.insert_component(pair_id, schema.clone(), &db.path)?;
            state
                .get_component(pair_id)
                .unwrap()
                .time_series
                .push_buf(Timestamp(0), &99.0f64.to_le_bytes())
        })
        .unwrap();

        let entity_id = impeller::types::EntityId(0);
        let mut world = World::default();
        world.metadata.entity_metadata.insert(
            entity_id,
            EntityMetadata {
                entity_id,
                name: "e1".into(),
                metadata: HashMap::new(),
            },
        );
        world.metadata.component_map.insert(
            component_id,
            (
                schema,
                ComponentMetadata {
                    component_id,
                    name: "scratch".into(),
                    metadata: HashMap::from([("transient".into(), "true".into())]),
                },
            ),
        );
        world.host.insert(
            component_id,
            crate::world::Column {
                buffer: 1.0f64.to_le_bytes().to_vec(),
                entity_ids: entity_id.0.to_le_bytes().to_vec(),
            },
        );

        db.with_state(|state| {
            DbSync::new(state, &world, &[], &HashSet::new()).copy_db_to_world(&mut world, 0)
        });

        assert_eq!(
            world.host.get(&component_id).unwrap().buffer,
            1.0f64.to_le_bytes()
        );
    }

    /// A one-entity world of f64 components `(name, metadata flag)`, recorded into a new DB.
    fn scalar_world(path: &Path, components: &[(&str, Option<&str>)]) -> (DB, World) {
        let db = DB::create(path.to_path_buf()).unwrap();
        let mut world = World::default();
        let entity_id = impeller::types::EntityId(0);
        world.metadata.entity_metadata.insert(
            entity_id,
            EntityMetadata {
                entity_id,
                name: "e".into(),
                metadata: HashMap::new(),
            },
        );
        for (name, flag) in components {
            let component_id = ComponentId::new(name);
            let metadata = flag
                .map(|flag| HashMap::from([(flag.to_string(), "true".to_string())]))
                .unwrap_or_default();
            world.metadata.component_map.insert(
                component_id,
                (
                    elodin_db::ComponentSchema {
                        prim_type: impeller::types::PrimType::F64,
                        dim: [1].into_iter().collect(),
                    },
                    ComponentMetadata {
                        component_id,
                        name: name.to_string(),
                        metadata,
                    },
                ),
            );
            world.host.insert(
                component_id,
                crate::world::Column {
                    buffer: 0.0f64.to_le_bytes().to_vec(),
                    entity_ids: entity_id.0.to_le_bytes().to_vec(),
                },
            );
        }
        init_db(&db, &mut world, Timestamp(0)).unwrap();
        (db, world)
    }

    fn series(db: &DB, name: &str) -> elodin_db::Component {
        db.with_state(|state| {
            state
                .get_component(ComponentId::from_pair("e", name))
                .unwrap()
                .clone()
        })
    }

    fn push(db: &DB, name: &str, at: i64, value: f64) {
        series(db, name)
            .time_series
            .push_buf(Timestamp(at), &value.to_le_bytes())
            .unwrap();
    }

    fn latest(db: &DB, name: &str) -> f64 {
        let component = series(db, name);
        f64::from_le_bytes(
            component
                .time_series
                .latest()
                .unwrap()
                .1
                .try_into()
                .unwrap(),
        )
    }

    fn value(world: &World, name: &str) -> f64 {
        f64::from_le_bytes(
            world.host[&ComponentId::new(name)].buffer[..]
                .try_into()
                .unwrap(),
        )
    }

    fn set(world: &mut World, name: &str, value: f64) {
        let buffer = &mut world.host.get_mut(&ComponentId::new(name)).unwrap().buffer;
        buffer.copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn db_sync_commit_sees_writes_after_the_copy() {
        let temp = tempdir().unwrap();
        let (db, mut world) = scalar_world(
            &temp.path().join("db"),
            &[("a", None), ("b", Some("external_control"))],
        );
        let written = [ComponentId::new("a"), ComponentId::new("b")];
        let exclusions = HashSet::from([ComponentId::new("b")]);
        let mut sync = db.with_state(|state| DbSync::new(state, &world, &written, &exclusions));

        push(&db, "a", 500, 5.0);
        sync.copy_db_to_world(&mut world, db.vtable_gen.latest());
        assert_eq!(value(&world, "a"), 5.0);

        // A write landing after the copy is what the world's unchanged value is compared to.
        push(&db, "a", 800, 7.0);
        set(&mut world, "b", 3.0);
        sync.commit_world(
            &world,
            Timestamp(1000),
            db.vtable_gen.latest(),
            &HookWrites::default(),
        )
        .unwrap();
        assert_eq!(latest(&db, "a"), 5.0);
        assert_eq!(series(&db, "b").time_series.sample_count(), 1);

        sync.copy_db_to_world(&mut world, db.vtable_gen.latest());
        assert_eq!(value(&world, "b"), 0.0);
        sync.commit_world(
            &world,
            Timestamp(2000),
            db.vtable_gen.latest(),
            &HookWrites::default(),
        )
        .unwrap();
        assert_eq!(series(&db, "a").time_series.sample_count(), 4);
    }

    #[test]
    fn deferred_writes_append_in_order_without_moving_last_updated() {
        let temp = tempdir().unwrap();
        let (db, _) = scalar_world(&temp.path().join("db"), &[("x", Some("external_control"))]);
        db.last_updated.store(Timestamp(1000));
        let write = |at: i64, value: f64| DeferredWrite {
            timestamp: Timestamp(at),
            rows: vec![(
                "e.x".into(),
                ComponentId::from_pair("e", "x"),
                value.to_le_bytes().to_vec(),
            )],
        };
        let counters = CallbackCounters::default();

        apply_deferred(
            &db,
            &[(0, write(1001, 1.0)), (1, write(1001, 2.0))],
            &counters,
        )
        .unwrap();
        let series = series(&db, "x");
        let (timestamps, values) = series
            .time_series
            .get_range(&(Timestamp(1001)..Timestamp(1001)))
            .unwrap();
        assert_eq!(timestamps, [Timestamp(1001), Timestamp(1001)]);
        assert_eq!(
            values,
            [1.0f64.to_le_bytes(), 2.0f64.to_le_bytes()].concat()
        );
        assert_eq!(db.last_updated.latest(), Timestamp(1000));

        let (hook, err) = apply_deferred(&db, &[(1, write(500, 3.0))], &counters).unwrap_err();
        assert_eq!(hook, 1);
        assert!(err.contains("time travel"), "{err}");
        assert_eq!(counters.deferred_batches.load(Ordering::Relaxed), 2);
        assert_eq!(counters.deferred_failures.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn db_sync_rereads_every_pair_after_truncate() {
        let temp = tempdir().unwrap();
        let (db, mut world) = scalar_world(&temp.path().join("db"), &[("a", None), ("c", None)]);
        let mut sync = db.with_state(|state| DbSync::new(state, &world, &[], &HashSet::new()));
        sync.copy_db_to_world(&mut world, db.vtable_gen.latest());

        // As many samples as before the truncate, but a different value.
        db.truncate();
        push(&db, "a", 500, 9.0);
        set(&mut world, "c", 2.0);
        sync.copy_db_to_world(&mut world, db.vtable_gen.latest());
        assert_eq!(value(&world, "a"), 9.0);
        assert_eq!(value(&world, "c"), 2.0);

        sync.commit_world(
            &world,
            Timestamp(1000),
            db.vtable_gen.latest(),
            &HookWrites::default(),
        )
        .unwrap();
        assert_eq!(series(&db, "a").time_series.sample_count(), 1);
        assert_eq!(latest(&db, "c"), 2.0);
    }

    #[test]
    fn db_sync_reads_only_rows_others_appended_to() {
        let temp = tempdir().unwrap();
        let (db, mut world) = scalar_world(&temp.path().join("db"), &[("a", None), ("c", None)]);
        let written = [ComponentId::new("c")];
        let mut sync = db.with_state(|state| DbSync::new(state, &world, &written, &HashSet::new()));
        let vtable_gen = db.vtable_gen.latest();
        sync.copy_db_to_world(&mut world, vtable_gen);
        sync.reset_counters();

        set(&mut world, "c", 2.0);
        sync.commit_world(&world, Timestamp(1000), vtable_gen, &HookWrites::default())
            .unwrap();
        sync.copy_db_to_world(&mut world, vtable_gen);
        push(&db, "a", 1500, 4.0);
        push(&db, "a", 1600, 5.0);
        sync.copy_db_to_world(&mut world, vtable_gen);
        sync.commit_world(&world, Timestamp(2000), vtable_gen, &HookWrites::default())
            .unwrap();

        assert_eq!(value(&world, "a"), 5.0);
        let c = &sync.counters;
        assert_eq!((c.full_scans, c.rows_scanned, c.rows_drained), (0, 1, 1));
        assert_eq!((c.cursor_wakeups, c.writes_coalesced), (1, 1));
        // The first commit after the initial scan compares every row; later ones only `c`.
        assert_eq!(c.rows_compared, c.rows_registered + 1);
        assert_eq!(c.rows_committed, 1);
    }

    /// Times both DB sync phases on a two-entity world of 276 components, a fifth of which systems
    /// write each cycle, with two external writes per cycle. Prints one `DBBENCH {json}` line per cache
    /// mode, with 32 MB written between phases in the cold mode.
    #[test]
    #[ignore = "benchmark"]
    fn bench_db_sync() {
        const COMPONENTS: usize = 276;
        const CYCLES: usize = 3000;
        let temp = tempdir().unwrap();
        let db = DB::create(temp.path().join("db")).unwrap();
        let mut world = World::default();
        for e in 0..2u64 {
            let entity_id = impeller::types::EntityId(e);
            world.metadata.entity_metadata.insert(
                entity_id,
                EntityMetadata {
                    entity_id,
                    name: format!("vehicle_{e}"),
                    metadata: HashMap::new(),
                },
            );
        }
        let sizes = [1usize, 3, 4, 6, 7, 36];
        for c in 0..COMPONENTS {
            let name = format!("component_{c}");
            let component_id = ComponentId::new(&name);
            let n = sizes[c % sizes.len()];
            let schema = elodin_db::ComponentSchema {
                prim_type: impeller::types::PrimType::F64,
                dim: [n].into_iter().collect(),
            };
            world.metadata.component_map.insert(
                component_id,
                (
                    schema,
                    ComponentMetadata {
                        component_id,
                        name,
                        metadata: HashMap::new(),
                    },
                ),
            );
            world.host.insert(
                component_id,
                crate::world::Column {
                    buffer: vec![0u8; 2 * n * 8],
                    entity_ids: [0u64, 1].iter().flat_map(|e| e.to_le_bytes()).collect(),
                },
            );
        }
        init_db(&db, &mut world, Timestamp(0)).unwrap();
        let external: Vec<ComponentId> = (0..COMPONENTS)
            .map(|c| ComponentId::from_pair("vehicle_1", &format!("component_{c}")))
            .collect();
        let written: Vec<ComponentId> = world.host.keys().copied().step_by(5).collect();
        let mut sync = db.with_state(|state| DbSync::new(state, &world, &written, &HashSet::new()));

        let mut cycle = 0i64;
        for thrash_mb in [0usize, 32] {
            let mut thrash = vec![0u8; thrash_mb << 20];
            let evict = |thrash: &mut Vec<u8>| {
                for line in thrash.chunks_mut(64) {
                    line[0] = line[0].wrapping_add(1);
                }
                std::hint::black_box(&thrash);
            };
            let (mut copy_ns, mut commit_ns) = (Vec::new(), Vec::new());
            for _ in 0..CYCLES {
                cycle += 1;
                evict(&mut thrash);
                let t0 = Instant::now();
                db.with_state(|_| sync.copy_db_to_world(&mut world, db.vtable_gen.latest()));
                copy_ns.push(t0.elapsed().as_nanos() as u64);

                for id in &written {
                    let buffer = &mut world.host.get_mut(id).unwrap().buffer;
                    buffer[0] = buffer[0].wrapping_add(1);
                }
                evict(&mut thrash);
                let t0 = Instant::now();
                db.with_state(|_| {
                    sync.commit_world(
                        &world,
                        Timestamp(cycle * 1000),
                        db.vtable_gen.latest(),
                        &HookWrites::default(),
                    )
                })
                .unwrap();
                commit_ns.push(t0.elapsed().as_nanos() as u64);

                db.with_state(|state| {
                    for k in 0..2 {
                        let pair = external[(cycle as usize * 7 + k) % COMPONENTS];
                        let component = state.get_component(pair).unwrap();
                        let mut value = component.time_series.latest().unwrap().1.to_vec();
                        value[1] = value[1].wrapping_add(1);
                        component
                            .time_series
                            .push_buf(Timestamp(cycle * 1000 + 500), &value)
                            .unwrap();
                    }
                });
            }
            copy_ns.sort_unstable();
            commit_ns.sort_unstable();
            println!(
                "DBBENCH {{\"thrash_mb\":{thrash_mb},\"copy_ns\":{},\"commit_ns\":{}}}",
                copy_ns[CYCLES / 2],
                commit_ns[CYCLES / 2]
            );
        }
    }
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
    fn prime_stores_display_kernel_sidecars() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db_path = dir.path().join("db");
        let db = elodin_db::DB::create(db_path.clone()).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some("viewport {\n}\n".to_string());
        let key = format!(
            "{}{}",
            impeller_wkt::DISPLAY_KERNEL_ASSET_PREFIX,
            "deadbeef"
        );
        let sidecar = b"{\"version\":1,\"hash\":\"deadbeef\"}".to_vec();
        world
            .metadata
            .schematic_kernels
            .insert(key.clone(), sidecar.clone());

        prime_schematic_assets(&db, &world, None).unwrap();

        let stored = elodin_db::assets_http::assets_dir(&db.path);
        assert_eq!(
            std::fs::read(stored.join(&key)).unwrap(),
            sidecar,
            "kernel sidecar must be stored verbatim"
        );

        // Reopen: copy-once schematic is preserved, but a new kernel hash still
        // lands because sidecars are content-addressed.
        let reopened = elodin_db::DB::open(db_path).unwrap();
        let new_key = format!(
            "{}{}",
            impeller_wkt::DISPLAY_KERNEL_ASSET_PREFIX,
            "cafebabe"
        );
        let new_sidecar = b"{\"version\":1,\"hash\":\"cafebabe\"}".to_vec();
        world
            .metadata
            .schematic_kernels
            .insert(new_key.clone(), new_sidecar.clone());
        prime_schematic_assets(&reopened, &world, None).unwrap();
        assert_eq!(std::fs::read(stored.join(&new_key)).unwrap(), new_sidecar);
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
    fn prime_preserves_edited_active_schematic_on_reopen() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        // No source tree: isolate the active-schematic path from tree ingest.
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db_path = dir.path().join("db");
        let db = elodin_db::DB::create(db_path.clone()).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some("viewport {\n}\n".to_string());

        // First run seeds the active schematic from in-memory metadata.
        prime_schematic_assets(&db, &world, None).unwrap();
        let stored = elodin_db::assets_http::assets_dir(&db.path);
        let seeded = std::fs::read_to_string(stored.join(ACTIVE_SCHEMATIC_KEY)).unwrap();

        // The editor writes new active-schematic bytes into the DB. Capture the
        // stored form (store_asset re-serializes KDL) to compare against later.
        db.store_asset(ACTIVE_SCHEMATIC_KEY, b"graph {\n}\n")
            .unwrap();
        let edited = std::fs::read_to_string(stored.join(ACTIVE_SCHEMATIC_KEY)).unwrap();
        assert_ne!(edited, seeded, "editor edit should differ from the seed");

        // Re-running the sim (in-memory metadata still the old schematic) against
        // the existing DB must NOT clobber the editor's bytes — copy-once, like
        // the asset tree. Simulate a `DB::open` reopen so the flow matches
        // `Server::from_listener`.
        let reopened = elodin_db::DB::open(db_path).unwrap();
        prime_schematic_assets(&reopened, &world, None).unwrap();

        assert_eq!(
            std::fs::read_to_string(stored.join(ACTIVE_SCHEMATIC_KEY)).unwrap(),
            edited,
            "editor-written active schematic must survive a sim re-run"
        );
        // The pointer still targets the (preserved) active schematic.
        let active =
            reopened.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(active.as_deref(), Some(ACTIVE_SCHEMATIC_KEY));
    }

    #[test]
    fn prime_preserves_repointed_active_schematic_on_reopen() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db_path = dir.path().join("db");
        let db = elodin_db::DB::create(db_path.clone()).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some("viewport {\n}\n".to_string());
        prime_schematic_assets(&db, &world, None).unwrap();

        // The editor does a "Save As": stores another schematic and repoints
        // `schematic.active` at it.
        db.store_asset("schematics/essai.kdl", b"graph {\n}\n")
            .unwrap();
        db.set_active_schematic("schematics/essai.kdl").unwrap();

        // Re-running the sim against the existing DB (`DB::open` preserves
        // db_config) must not clobber the user's pointer back to main.kdl.
        let reopened = elodin_db::DB::open(db_path).unwrap();
        prime_schematic_assets(&reopened, &world, None).unwrap();

        let active =
            reopened.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(
            active.as_deref(),
            Some("schematics/essai.kdl"),
            "a sim re-run must not repoint schematic.active away from the user's choice"
        );
    }

    #[test]
    fn prime_preserves_repointed_active_when_main_kdl_removed() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db_path = dir.path().join("db");
        let db = elodin_db::DB::create(db_path.clone()).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some("viewport {\n}\n".to_string());
        prime_schematic_assets(&db, &world, None).unwrap();

        // The editor does a "Save As" to another key, repoints `schematic.active`
        // at it, and drops the default `main.kdl` from the tree entirely.
        db.store_asset("schematics/essai.kdl", b"graph {\n}\n")
            .unwrap();
        db.set_active_schematic("schematics/essai.kdl").unwrap();
        let stored = elodin_db::assets_http::assets_dir(&db.path);
        std::fs::remove_file(stored.join(ACTIVE_SCHEMATIC_KEY)).unwrap();

        // Re-running the sim must not treat the absent `main.kdl` as a fresh DB:
        // the surviving `schematic.active` pointer marks a reopen, so the saved
        // pointer stays put and no regenerated default is stored.
        let reopened = elodin_db::DB::open(db_path).unwrap();
        prime_schematic_assets(&reopened, &world, None).unwrap();

        let active =
            reopened.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(
            active.as_deref(),
            Some("schematics/essai.kdl"),
            "a missing main.kdl must not repoint schematic.active away from the user's choice"
        );
        assert!(
            !stored.join(ACTIVE_SCHEMATIC_KEY).is_file(),
            "the deleted default schematic must not be regenerated on reopen"
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
    fn prime_falls_back_to_ingested_main_kdl_without_sim_schematic() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        // The source tree ships the conventional active schematic, but the sim
        // itself carries no in-memory schematic metadata.
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(assets.join("schematics")).unwrap();
        std::fs::write(assets.join("schematics/main.kdl"), b"viewport {\n}\n").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let world = World::default();
        assert!(world.metadata.schematic.is_none());

        let _guard = EnvVarGuard::set("ELODIN_ASSETS", assets.to_str().unwrap());
        prime_schematic_assets(&db, &world, None).unwrap();

        // The ingested schematics/main.kdl becomes the active schematic, same
        // as the CLI fallback in `elodin-db run`, so editors auto-load it.
        let active = db.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(active.as_deref(), Some(ACTIVE_SCHEMATIC_KEY));
    }

    #[test]
    fn prime_leaves_pointer_unset_without_schematic_or_ingest() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempdir().unwrap();
        let _no_assets =
            EnvVarGuard::set("ELODIN_ASSETS", dir.path().join("nope").to_str().unwrap());

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let world = World::default();
        prime_schematic_assets(&db, &world, None).unwrap();

        // Nothing to point at: the fallback must not set a dangling pointer.
        let active = db.with_state(|state| state.db_config.schematic_active().map(str::to_owned));
        assert_eq!(active, None);
    }

    #[test]
    fn prime_ingests_window_schematics_outside_assets_tree() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _no_env = EnvVarGuard::unset("ELODIN_ASSETS");

        // The drone layout: the sim entry lives in `examples/drone/`, the
        // window KDL sits next to it (outside `assets/`), and the schematic
        // references it repo-root-relative.
        let dir = tempdir().unwrap();
        let sim = dir.path().join("examples/drone");
        std::fs::create_dir_all(sim.join("assets")).unwrap();
        std::fs::write(sim.join("assets/talon.glb"), b"glb-bytes").unwrap();
        std::fs::write(
            sim.join("motor-panel.kdl"),
            b"tabs {\n    graph \"drone.motor_input\"\n}\n",
        )
        .unwrap();
        let entry = sim.join("main.py");
        std::fs::write(&entry, b"# sim").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some(
            "object_3d \"drone.world_pos\" {\n    glb path=\"talon.glb\"\n}\nwindow title=\"motors\" path=\"examples/drone/motor-panel.kdl\"\n"
                .to_string(),
        );

        prime_schematic_assets(&db, &world, Some(&entry)).unwrap();

        // The window file was pulled into the DB under schematics/ and the
        // stored active schematic references it as a db: key — no machine-local
        // path survives, so a follower or another machine loads it over HTTP.
        let stored = elodin_db::assets_http::assets_dir(&db.path);
        let window = std::fs::read_to_string(stored.join("schematics/motor-panel.kdl")).unwrap();
        assert!(window.contains("drone.motor_input"));

        let active_kdl = std::fs::read_to_string(stored.join("schematics/main.kdl")).unwrap();
        assert!(
            active_kdl.contains("db:schematics/motor-panel.kdl"),
            "window reference should be a db: key, got:\n{active_kdl}"
        );
        assert!(
            !active_kdl.contains("examples/drone/motor-panel.kdl"),
            "no machine-local window path should survive, got:\n{active_kdl}"
        );
        assert!(
            active_kdl.contains("db:talon.glb"),
            "the glb rewrite must still apply after window ingest, got:\n{active_kdl}"
        );
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

    #[test]
    fn prime_merges_sim_assets_with_parent_assets() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _no_env = EnvVarGuard::unset("ELODIN_ASSETS");

        // Falcon 9 layout: shared meshes in the repo's `assets/`, map overlays
        // in the sim's own `assets/`.
        let dir = tempdir().unwrap();
        let repo = dir.path().join("repo");
        let sim = repo.join("examples/falcon9");
        std::fs::create_dir_all(repo.join("assets")).unwrap();
        std::fs::write(repo.join("assets/falcon9.glb"), b"glb-bytes").unwrap();
        std::fs::create_dir_all(sim.join("assets/overlays")).unwrap();
        std::fs::write(sim.join("assets/overlays/sites.json"), b"{}").unwrap();
        let entry = sim.join("main.py");
        std::fs::write(&entry, b"# sim").unwrap();

        let db = elodin_db::DB::create(dir.path().join("db")).unwrap();
        let mut world = World::default();
        world.metadata.schematic = Some(
            "object_3d \"booster.world_pos\" {\n    glb path=\"falcon9.glb\"\n}\nmap_overlay path=\"overlays/sites.json\"\n"
                .to_string(),
        );
        prime_schematic_assets(&db, &world, Some(&entry)).unwrap();

        let stored = elodin_db::assets_http::assets_dir(&db.path);
        let active_kdl = std::fs::read_to_string(stored.join("schematics/main.kdl")).unwrap();
        assert!(
            active_kdl.contains("db:falcon9.glb"),
            "the parent assets mesh should be stored, got:\n{active_kdl}"
        );
        assert!(
            active_kdl.contains("db:overlays/sites.json"),
            "the sim's own overlay should be stored, got:\n{active_kdl}"
        );
    }
}
