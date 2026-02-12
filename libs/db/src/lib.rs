use convert_case::Casing;
use datafusion::common::HashSet;
use futures_lite::StreamExt;
use impeller2::registry::VTableRegistry;
use impeller2::types::{PacketHeader, PacketTy};
use impeller2::vtable::builder::{
    OpBuilder, component, raw_field, raw_table, schema, timestamp, vtable,
};
use impeller2::vtable::{Op, RealizedField, builder};
use impeller2::{
    com_de::Decomponentize,
    registry,
    schema::Schema,
    types::{
        ComponentId, ComponentView, IntoLenPacket, LenPacket, Msg, OwnedPacket as Packet, PacketId,
        PrimType, RequestId, Timestamp,
    },
    vtable::VTable,
};
use impeller2_stellar::{PacketSink, PacketStream};
use impeller2_wkt::*;
use msg_log::MsgLog;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::OsStr,
    fs::{self, File},
    io::{self},
    net::{SocketAddr, ToSocketAddrs},
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex as StdMutex, RwLock,
        atomic::{self, AtomicBool, AtomicI64, AtomicU64},
    },
    time::Duration,
};
use stellarator::{
    buf::Slice,
    io::{AsyncRead, AsyncWrite, OwnedReader, OwnedWriter, SplitExt},
    net::{TcpListener, TcpStream, UdpSocket},
    rent,
    struc_con::Joinable,
    sync::{Mutex, WaitQueue},
    util::AtomicCell,
};
use time_series::TimeSeries;
use tracing::{debug, info, trace, warn};
use vtable_stream::handle_vtable_stream;
use zerocopy::IntoBytes;

pub use error::Error;

pub mod append_log;
mod arrow;
pub mod axum;
pub mod cancellation;
pub mod drop;
mod error;
pub mod export;
#[cfg(feature = "video-export")]
pub mod export_videos;
pub mod fix_timestamps;
pub mod merge;
mod msg_log;
pub mod prune;
pub mod time_align;
pub(crate) mod time_series;
pub mod truncate;
pub mod utils;
mod vtable_stream;

/// Analyzes a VTable to find byte ranges that are used as timestamp sources.
/// Returns a vector of (offset, end) tuples representing the byte ranges.
fn find_timestamp_source_ranges<Ops, Data, Fields>(
    vtable: &impeller2::vtable::VTable<Ops, Data, Fields>,
) -> Vec<(usize, usize)>
where
    Ops: impeller2::buf::Buf<Op>,
    Data: impeller2::buf::Buf<u8>,
    Fields: impeller2::buf::Buf<impeller2::vtable::Field>,
{
    let mut ranges = Vec::new();
    for (op_idx, op) in vtable.ops.as_slice().iter().enumerate() {
        if let Op::Timestamp { source, .. } = op {
            debug!(op_idx, "found timestamp operation");
            // Resolve source to get the actual Table op
            match vtable.realize(*source, None) {
                Ok(resolved) => {
                    if let Some(range) = resolved.as_table_range() {
                        debug!(
                            op_idx,
                            range_start = range.start,
                            range_end = range.end,
                            "timestamp source range resolved"
                        );
                        ranges.push((range.start, range.end));
                    } else {
                        warn!(
                            op_idx,
                            "timestamp operation source resolved but has no table range"
                        );
                    }
                }
                Err(e) => {
                    warn!(?e, op_idx, "failed to resolve timestamp operation source");
                }
            }
        }
    }
    if ranges.is_empty() {
        debug!("no timestamp source ranges found in vtable");
    } else {
        debug!(range_count = ranges.len(), "found timestamp source ranges");
    }
    ranges
}

/// Checks if a field's byte range overlaps with any timestamp source range.
fn field_overlaps_timestamp_source(
    field_offset: usize,
    field_len: usize,
    timestamp_source_ranges: &[(usize, usize)],
) -> bool {
    let field_end = field_offset + field_len;
    for &(ts_start, ts_end) in timestamp_source_ranges {
        // Check for overlap: ranges overlap if they are not disjoint
        // Disjoint means: field_end <= ts_start OR ts_end <= field_offset
        if !(field_end <= ts_start || ts_end <= field_offset) {
            return true;
        }
    }
    false
}

/// Fallback detection: checks if a component name suggests it's a timestamp source.
/// This is used as a last resort when range-based detection fails.
/// Only matches clear patterns to avoid false positives.
fn looks_like_timestamp_source_by_name(component_name: &str) -> bool {
    let name_upper = component_name.to_uppercase();
    // Match common timestamp source patterns
    name_upper.contains("TIME_MONOTONIC")
        || name_upper.contains("TIMESTAMP_SOURCE")
        || (name_upper.contains("TIMESTAMP") && name_upper.ends_with("_SOURCE"))
        || (name_upper.contains("TIME") && name_upper.contains("MONOTONIC"))
}

pub struct SnapshotBarrier {
    state: StdMutex<SnapshotState>,
    cv: Condvar,
}

#[derive(Default)]
struct SnapshotState {
    active_writers: usize,
    snapshot_active: bool,
}

pub struct SnapshotWriterGuard<'a> {
    barrier: &'a SnapshotBarrier,
    released: bool,
}

pub struct SnapshotGuard<'a> {
    barrier: &'a SnapshotBarrier,
    released: bool,
}

impl SnapshotBarrier {
    pub fn new() -> Self {
        Self {
            state: StdMutex::new(SnapshotState::default()),
            cv: Condvar::new(),
        }
    }

    pub fn enter_writer(&self) -> SnapshotWriterGuard<'_> {
        let mut state = self.state.lock().expect("snapshot barrier mutex poisoned");
        while state.snapshot_active {
            state = self.cv.wait(state).unwrap();
        }
        state.active_writers += 1;
        SnapshotWriterGuard {
            barrier: self,
            released: false,
        }
    }

    pub fn begin_snapshot(&self) -> SnapshotGuard<'_> {
        let mut state = self.state.lock().unwrap();
        while state.snapshot_active {
            state = self.cv.wait(state).unwrap();
        }
        state.snapshot_active = true;
        while state.active_writers > 0 {
            state = self.cv.wait(state).unwrap();
        }
        SnapshotGuard {
            barrier: self,
            released: false,
        }
    }

    fn release_writer(&self) {
        let mut state = self.state.lock().unwrap();
        debug_assert!(state.active_writers > 0);
        if state.active_writers == 0 {
            return;
        }
        state.active_writers -= 1;
        let should_notify = state.active_writers == 0;
        drop(state);
        if should_notify {
            self.cv.notify_all();
        }
    }

    fn finish_snapshot(&self) {
        let mut state = self.state.lock().unwrap();
        state.snapshot_active = false;
        drop(state);
        self.cv.notify_all();
    }
}

impl Default for SnapshotBarrier {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SnapshotWriterGuard<'a> {
    pub fn release(mut self) {
        if self.released {
            return;
        }
        self.barrier.release_writer();
        self.released = true;
    }
}

impl<'a> Drop for SnapshotWriterGuard<'a> {
    fn drop(&mut self) {
        if !self.released {
            self.barrier.release_writer();
            self.released = true;
        }
    }
}

impl<'a> SnapshotGuard<'a> {
    pub fn release(mut self) {
        if self.released {
            return;
        }
        self.barrier.finish_snapshot();
        self.released = true;
    }
}

impl<'a> Drop for SnapshotGuard<'a> {
    fn drop(&mut self) {
        if !self.released {
            self.barrier.finish_snapshot();
            self.released = true;
        }
    }
}

pub(crate) fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(target_family = "unix")]
    {
        let dir = File::open(path)?;
        dir.sync_all()
    }
    #[cfg(not(target_family = "unix"))]
    {
        let _ = path;
        Ok(())
    }
}

pub(crate) fn copy_file_native(src: &Path, dst: &Path) -> Result<(), Error> {
    let metadata = fs::metadata(src)?;
    reflink_copy::reflink_or_copy(src, dst)?;
    fs::set_permissions(dst, metadata.permissions())?;
    let file = File::open(dst)?;
    file.sync_all()?;
    if let Some(parent) = dst.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

pub(crate) fn copy_dir_native(src: &Path, dst: &Path) -> Result<(), Error> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_native(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            copy_file_native(&src_path, &dst_path)?;
        } else {
            // skip non-regular entries
            warn!(
                "Skipping irregular file for db copy {:?}",
                src_path.display()
            );
        }
    }
    let metadata = fs::metadata(src)?;
    fs::set_permissions(dst, metadata.permissions())?;
    let _ = sync_dir(dst);
    Ok(())
}

pub struct DB {
    pub vtable_gen: AtomicCell<u64>,
    state: RwLock<State>,
    snapshot_barrier: SnapshotBarrier,
    pub recording_cell: PlayingCell,

    // metadata
    pub path: PathBuf,
    pub default_stream_time_step: AtomicU64,
    pub last_updated: AtomicCell<Timestamp>,
    pub earliest_timestamp: AtomicCell<Timestamp>,
    // Wall-clock timestamp at the moment the DB start anchor was set.
    db_start_wall_clock: AtomicCell<Timestamp>,
}

#[derive(Default)]
pub struct State {
    components: HashMap<ComponentId, Component>,
    component_metadata: HashMap<ComponentId, ComponentMetadata>,

    msg_logs: HashMap<PacketId, MsgLog>,

    vtable_registry: registry::HashMapRegistry,
    streams: HashMap<StreamId, Arc<FixedRateStreamState>>,

    udp_vtable_streams: HashSet<(SocketAddr, [u8; 2])>,

    pub db_config: DbConfig,
}

impl DB {
    pub fn create(path: PathBuf) -> Result<Self, Error> {
        Self::with_time_step(path, Duration::from_secs_f64(1.0 / 120.0))
    }

    pub fn with_time_step(path: PathBuf, time_step: Duration) -> Result<Self, Error> {
        info!(?path, "created db");

        std::fs::create_dir_all(&path)?;
        let now = Timestamp::now();
        let default_stream_time_step = AtomicU64::new(time_step.as_nanos() as u64);
        let mut db_config = DbConfig {
            default_stream_time_step: time_step,
            ..Default::default()
        };
        db_config.set_time_start_timestamp_micros(now.0);
        db_config.set_version_created(env!("CARGO_PKG_VERSION"));
        db_config.set_version_last_opened(env!("CARGO_PKG_VERSION"));
        let state = State {
            db_config,
            ..Default::default()
        };
        let db = DB {
            state: RwLock::new(state),
            snapshot_barrier: SnapshotBarrier::new(),
            recording_cell: PlayingCell::new(true),
            path,
            vtable_gen: AtomicCell::new(0),
            default_stream_time_step,
            last_updated: AtomicCell::new(Timestamp(i64::MIN)),
            earliest_timestamp: AtomicCell::new(now),
            db_start_wall_clock: AtomicCell::new(now),
        };
        db.save_db_state()?;
        Ok(db)
    }

    pub fn with_state<O, F: FnOnce(&State) -> O>(&self, f: F) -> O {
        let state = self.state.read().unwrap();
        f(&state)
    }

    pub fn with_state_mut<O, F: FnOnce(&mut State) -> O>(&self, f: F) -> O {
        let mut state = self.state.write().unwrap();
        f(&mut state)
    }

    fn db_config(&self) -> DbConfig {
        self.with_state(|db| db.db_config.clone())
    }

    pub fn save_db_state(&self) -> Result<(), Error> {
        let db_state = self.db_config();
        db_state.write(self.path.join("db_state"))
    }

    pub fn begin_snapshot(&self) -> SnapshotGuard<'_> {
        self.snapshot_barrier.begin_snapshot()
    }

    pub fn flush_all(&self) -> Result<(), Error> {
        self.with_state(|state| -> Result<(), Error> {
            // Ensure time-series data is fully flushed
            for component in state.components.values() {
                component.sync_all()?;
            }
            // Ensure message logs are fully flushed
            for msg_log in state.msg_logs.values() {
                msg_log.sync_all()?;
            }
            // Additionally, make metadata and schema durable for each component
            for component_id in state.components.keys() {
                let comp_dir = self.path.join(component_id.to_string());
                let schema_path = comp_dir.join("schema");
                if schema_path.exists() {
                    let file = File::open(&schema_path)?;
                    file.sync_all()?;
                }
                let metadata_path = comp_dir.join("metadata");
                if metadata_path.exists() {
                    let file = File::open(&metadata_path)?;
                    file.sync_all()?;
                }
                if comp_dir.exists() {
                    // Best-effort sync of the component directory entry
                    let _ = sync_dir(&comp_dir);
                }
            }
            Ok(())
        })?;

        let db_state_path = self.path.join("db_state");
        if db_state_path.exists() {
            File::open(&db_state_path)?.sync_all()?;
        }
        File::open(&self.path)?.sync_all()?;
        Ok(())
    }

    /// Truncate all component data and message logs, clearing all data while preserving schemas and metadata.
    ///
    /// This effectively resets the database to an empty state, ready for fresh data.
    /// The vtable generation is incremented to signal that clients should refresh their views.
    pub fn truncate(&self) {
        self.with_state(|state| {
            state.truncate_all();
        });
        self.last_updated.store(Timestamp(i64::MIN));
        self.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
    }

    /// Set the earliest timestamp for this database.
    ///
    /// This is typically called during initialization to set the starting timestamp
    /// for the simulation. If not called, defaults to the current system time.
    pub fn set_earliest_timestamp(&self, timestamp: Timestamp) -> Result<(), Error> {
        self.with_state_mut(|state| {
            state.db_config.set_time_start_timestamp_micros(timestamp.0);
        });
        self.earliest_timestamp.store(timestamp);
        self.db_start_wall_clock.store(Timestamp::now());
        self.save_db_state()
    }

    pub fn apply_implicit_timestamp(&self) -> Timestamp {
        let start = self.earliest_timestamp.latest();
        let base = self.db_start_wall_clock.latest();
        let now = Timestamp::now();
        let delta = now.0.saturating_sub(base.0);
        Timestamp(start.0.saturating_add(delta))
    }

    pub fn copy_native(&self, target_db_path: impl AsRef<Path>) -> Result<PathBuf, Error> {
        let final_db_dir = target_db_path.as_ref().to_path_buf();
        let parent_dir = final_db_dir
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        if self.path.starts_with(&final_db_dir) || final_db_dir.starts_with(&self.path) {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "target directory overlaps database path",
            )));
        }

        if final_db_dir.exists() {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "target directory already exists",
            )));
        }

        fs::create_dir_all(&parent_dir)?;
        let tmp_db_dir = {
            let name = final_db_dir
                .file_name()
                .unwrap_or(OsStr::new("db"))
                .to_string_lossy();
            parent_dir.join(format!("{}.tmp", name))
        };
        if tmp_db_dir.exists() {
            fs::remove_dir_all(&tmp_db_dir)?;
        }
        fs::create_dir_all(&tmp_db_dir)?;
        copy_dir_native(&self.path, &tmp_db_dir)?;
        sync_dir(&tmp_db_dir)?;
        fs::rename(&tmp_db_dir, &final_db_dir)?;
        sync_dir(&parent_dir)?;
        Ok(final_db_dir)
    }

    pub fn open(path: PathBuf) -> Result<Self, Error> {
        let mut component_metadata = HashMap::new();
        let mut components = HashMap::new();
        let mut msg_logs = HashMap::new();
        let mut last_updated = i64::MIN;
        let mut start_timestamp = i64::MAX;

        info!("Opening database: {}", path.display());
        let db_state_path = path.join("db_state");
        if !db_state_path.exists() {
            return Err(Error::MissingDbState(db_state_path));
        }
        for elem in std::fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            if !path.is_dir() || path.file_name() == Some(OsStr::new("msgs")) {
                trace!("Skipping non-component directory: {}", path.display());
                continue;
            }

            let component_id = ComponentId(
                path.file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidComponentId)?,
            );

            let metadata_path = path.join("metadata");
            if metadata_path.exists() {
                let metadata = ComponentMetadata::read(metadata_path)?;
                trace!("Read component metadata for {}", metadata.name);
                component_metadata.insert(component_id, metadata);
            }

            let schema_path = path.join("schema");
            if !schema_path.exists() {
                warn!(
                    component.id = ?component_id.0,
                    component.dir = %path.display(),
                    "Skipping component without schema while opening database"
                );
                continue;
            }

            trace!("Opening component file {}", path.display());
            let schema = ComponentSchema::read(schema_path)?;
            let name = component_metadata
                .get(&component_id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| component_id.to_string());
            let component = Component::open(&path, component_id, name, schema.clone())?;

            // Check if this component is a timestamp source - if so, exclude from time calculations
            let is_timestamp_source = component_metadata
                .get(&component_id)
                .map(|m| m.is_timestamp_source())
                .unwrap_or(false);

            if !is_timestamp_source {
                if let Some((timestamp, _)) = component.time_series.latest() {
                    last_updated = timestamp.0.max(last_updated);
                };
                start_timestamp = start_timestamp.min(component.time_series.start_timestamp().0);
            } else {
                trace!(
                    component.id = ?component_id.0,
                    "Excluding timestamp source component from time range calculations"
                );
            }
            components.insert(component_id, component);
        }
        if let Ok(msgs_dir) = std::fs::read_dir(path.join("msgs")) {
            for elem in msgs_dir {
                let Ok(elem) = elem else { continue };

                let path = elem.path();
                let msg_id: u16 = path
                    .file_name()
                    .and_then(|p| p.to_str())
                    .and_then(|p| p.parse().ok())
                    .ok_or(Error::InvalidMsgId)?;
                let msg_log = MsgLog::open(path)?;
                if let Some(first_timestamp) = msg_log.timestamps().first() {
                    start_timestamp = start_timestamp.min(first_timestamp.0);
                }

                if let Some((timestamp, _)) = msg_log.latest() {
                    last_updated = timestamp.0.max(last_updated);
                };
                msg_logs.insert(msg_id.to_le_bytes(), msg_log);
            }
        }

        info!(db.path = ?path, "opened db");
        let mut db_state = DbConfig::read(db_state_path)?;
        // Update the version that last opened this database
        db_state.set_version_last_opened(env!("CARGO_PKG_VERSION"));
        let now = Timestamp::now();
        let state = State {
            components,
            component_metadata,
            msg_logs,
            db_config: db_state.clone(),
            ..Default::default()
        };
        let earliest_timestamp = db_state
            .time_start_timestamp_micros()
            .map(Timestamp)
            .unwrap_or_else(|| {
                if start_timestamp == i64::MAX {
                    now
                } else {
                    Timestamp(start_timestamp)
                }
            });
        let db = DB {
            state: RwLock::new(state),
            snapshot_barrier: SnapshotBarrier::new(),
            path,
            vtable_gen: AtomicCell::new(0),
            recording_cell: PlayingCell::new(db_state.recording),
            default_stream_time_step: AtomicU64::new(
                db_state.default_stream_time_step.as_nanos() as u64
            ),
            last_updated: AtomicCell::new(Timestamp(last_updated)),
            earliest_timestamp: AtomicCell::new(earliest_timestamp),
            db_start_wall_clock: AtomicCell::new(now),
        };
        // Save updated version info
        db.save_db_state()?;
        Ok(db)
    }

    pub fn insert_vtable(&self, vtable: VTableMsg) -> Result<(), Error> {
        info!(id = ?vtable.id, "inserting vtable");
        let _snapshot_guard = self.snapshot_barrier.enter_writer();

        // Find byte ranges that are used as timestamp sources
        let timestamp_source_ranges = find_timestamp_source_ranges(&vtable.vtable);
        debug!(
            vtable_id = ?vtable.id,
            timestamp_range_count = timestamp_source_ranges.len(),
            ranges = ?timestamp_source_ranges,
            "timestamp source ranges found"
        );

        self.with_state_mut(|state| {
            // We need to iterate over fields to get offset/len for timestamp source detection
            let fields: Vec<_> = vtable.vtable.fields.as_slice().to_vec();
            debug!(
                vtable_id = ?vtable.id,
                field_count = fields.len(),
                "processing vtable fields"
            );

            // Track component IDs that are referenced by timestamp operations
            let mut timestamp_source_component_ids = std::collections::HashSet::new();

            // First pass: identify components referenced by timestamp operations
            for op in vtable.vtable.ops.as_slice() {
                if let Op::Timestamp { source, arg } = op {
                    // Try to resolve the arg to get the component ID
                    if let Ok(resolved_arg) = vtable.vtable.realize(*arg, None)
                        && let Some(comp_id) = resolved_arg.as_component_id()
                    {
                        debug!(
                            component_id = ?comp_id.0,
                            "timestamp operation references component"
                        );
                        // Also check if the source itself resolves to a component
                        if let Ok(resolved_source) = vtable.vtable.realize(*source, None)
                            && let Some(source_comp_id) = resolved_source.as_component_id()
                        {
                            debug!(
                                component_id = ?source_comp_id.0,
                                "timestamp source is itself a component"
                            );
                            timestamp_source_component_ids.insert(source_comp_id);
                        }
                    }
                }
            }

            for (field_idx, (field, res)) in fields
                .iter()
                .zip(vtable.vtable.realize_fields(None))
                .enumerate()
            {
                let RealizedField {
                    component_id,
                    shape,
                    ty,
                    ..
                } = res?;

                // Get component name for logging and fallback detection
                let component_name = state
                    .component_metadata
                    .get(&component_id)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| component_id.to_string());

                // Check if this field overlaps with any timestamp source range
                let is_timestamp_source_by_range = field_overlaps_timestamp_source(
                    field.offset.to_index(),
                    field.len as usize,
                    &timestamp_source_ranges,
                );

                // Also check if this component ID is directly referenced as a timestamp source
                let is_timestamp_source_by_id =
                    timestamp_source_component_ids.contains(&component_id);

                // Fallback: check component name pattern (only if other methods didn't detect it)
                let is_timestamp_source_by_name =
                    if !is_timestamp_source_by_range && !is_timestamp_source_by_id {
                        let looks_like = looks_like_timestamp_source_by_name(&component_name);
                        if looks_like {
                            debug!(
                                component_id = ?component_id.0,
                                component_name = ?component_name,
                                "fallback detection: component name suggests timestamp source"
                            );
                        }
                        looks_like
                    } else {
                        false
                    };

                let is_timestamp_source = is_timestamp_source_by_range
                    || is_timestamp_source_by_id
                    || is_timestamp_source_by_name;

                debug!(
                    vtable_id = ?vtable.id,
                    field_idx,
                    component_id = ?component_id.0,
                    component_name = ?component_name,
                    field_offset = field.offset.to_index(),
                    field_len = field.len,
                    is_timestamp_source_by_range,
                    is_timestamp_source_by_id,
                    is_timestamp_source_by_name,
                    is_timestamp_source,
                    "processing field"
                );

                let component_schema = ComponentSchema::new(ty, shape);
                state.insert_component_with_timestamp_source_flag(
                    component_id,
                    component_schema,
                    is_timestamp_source,
                    &self.path,
                )?;
                self.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
            }
            state.vtable_registry.map.insert(vtable.id, vtable.vtable);
            Ok::<_, Error>(())
        })?;
        Ok(())
    }

    pub fn push_msg(&self, timestamp: Timestamp, id: PacketId, msg: &[u8]) -> Result<(), Error> {
        let _snapshot_guard = self.snapshot_barrier.enter_writer();
        let exists = self.with_state(|s| {
            if let Some(msg_log) = s.msg_logs.get(&id) {
                msg_log.push(timestamp, msg)?;
                Ok::<_, Error>(true)
            } else {
                Ok(false)
            }
        })?;
        if !exists {
            self.with_state_mut(move |s| {
                let msg_log = s.get_or_insert_msg_log(id, &self.path)?;
                msg_log.push(timestamp, msg)?;
                Ok::<_, Error>(())
            })?;
        }
        self.last_updated.update_max(timestamp);
        Ok(())
    }

    pub fn get_or_insert_fixed_rate_state(
        &self,
        stream_id: StreamId,
        behavior: FixedRateBehavior,
    ) -> Arc<FixedRateStreamState> {
        let mut state = self.state.write().expect("poisoned lock");
        let stream_state = state
            .streams
            .entry(stream_id)
            .or_insert_with(|| {
                Arc::new(FixedRateStreamState::new(
                    stream_id,
                    Duration::from_nanos(behavior.timestep),
                    match behavior.initial_timestamp {
                        InitialTimestamp::Earliest => self.earliest_timestamp.latest(),
                        InitialTimestamp::Latest => self.last_updated.latest(),
                        InitialTimestamp::Manual(timestamp) => timestamp,
                    },
                    behavior.frequency,
                ))
            })
            .clone();
        // Spawn a dedicated tick driver if one hasn't been spawned yet
        if !stream_state
            .driver_spawned
            .swap(true, atomic::Ordering::SeqCst)
        {
            stellarator::spawn(run_tick_driver(stream_state.clone()));
        }
        stream_state
    }
}

impl State {
    pub fn insert_component(
        &mut self,
        component_id: ComponentId,
        schema: ComponentSchema,
        db_path: &Path,
    ) -> Result<(), Error> {
        self.insert_component_with_timestamp_source_flag(component_id, schema, false, db_path)
    }

    /// Inserts a component, optionally marking it as a timestamp source.
    /// Timestamp source components contain raw clock values used as timestamps
    /// for other components, and should be excluded from time range calculations.
    pub fn insert_component_with_timestamp_source_flag(
        &mut self,
        component_id: ComponentId,
        schema: ComponentSchema,
        is_timestamp_source: bool,
        db_path: &Path,
    ) -> Result<(), Error> {
        if let Some(existing_component) = self.components.get(&component_id) {
            if existing_component.schema != schema {
                warn!( ?existing_component.schema, new_component.schema = ?schema,
                       ?existing_component.component_id,
                      "schema mismatch");
                return Err(Error::SchemaMismatch);
            }
            // If this component is a timestamp source, update the metadata
            if is_timestamp_source
                && let Some(existing_meta) = self.component_metadata.get_mut(&component_id)
                && !existing_meta.is_timestamp_source()
            {
                existing_meta.set_timestamp_source(true);
                // Re-save the metadata - ensure directory exists first
                let component_metadata_dir = db_path.join(component_id.to_string());
                if let Err(err) = std::fs::create_dir_all(&component_metadata_dir) {
                    warn!(
                        ?err,
                        ?component_id,
                        "failed to create component metadata directory"
                    );
                }
                let metadata_path = component_metadata_dir.join("metadata");
                if let Err(err) = existing_meta.write(&metadata_path) {
                    warn!(
                        ?err,
                        ?component_id,
                        "failed to update timestamp source metadata"
                    );
                }
            }
            return Ok(());
        }
        info!(component.id = ?component_id.0, is_timestamp_source, "inserting");
        // Check if custom metadata was previously set (e.g., via SetComponentMetadata)
        // If so, use the existing name; otherwise fall back to the component ID
        let name = self
            .component_metadata
            .get(&component_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| component_id.to_string());
        let mut component_metadata = ComponentMetadata {
            component_id,
            name: name.clone(),
            metadata: Default::default(),
        };
        if is_timestamp_source {
            component_metadata.set_timestamp_source(true);
        }
        let component = Component::create(db_path, component_id, name, schema, Timestamp::now())?;
        // Always update metadata if this is a timestamp source, or if metadata doesn't exist yet
        // This ensures the timestamp source flag is preserved even if metadata was set earlier
        if is_timestamp_source || !self.component_metadata.contains_key(&component_id) {
            self.set_component_metadata(component_metadata, db_path)?;
        }
        self.components.insert(component_id, component);
        Ok(())
    }

    pub fn get_component_metadata(&self, component_id: ComponentId) -> Option<&ComponentMetadata> {
        self.component_metadata.get(&component_id)
    }

    pub fn get_component(&self, component_id: ComponentId) -> Option<&Component> {
        self.components.get(&component_id)
    }

    pub fn set_component_metadata(
        &mut self,
        mut metadata: ComponentMetadata,
        db_path: &Path,
    ) -> Result<(), Error> {
        let component_metadata_path = db_path.join(metadata.component_id.to_string());
        std::fs::create_dir_all(&component_metadata_path)?;
        let component_metadata_path = component_metadata_path.join("metadata");

        // Preserve existing metadata flags, especially _is_timestamp_source
        // This is critical because SetComponentMetadata may be called after
        // insert_component_with_timestamp_source_flag has already set the flag
        if let Some(existing_metadata) = self.component_metadata.get(&metadata.component_id) {
            // Preserve the timestamp source flag if it was already set
            if existing_metadata.is_timestamp_source() {
                metadata.set_timestamp_source(true);
            }
            // Merge other metadata flags from existing metadata
            for (key, value) in existing_metadata.metadata.iter() {
                // Only preserve internal flags (starting with _) that aren't being overwritten
                if key.starts_with('_') && !metadata.metadata.contains_key(key) {
                    metadata.metadata.insert(key.clone(), value.clone());
                }
            }
        }

        if component_metadata_path.exists()
            && ComponentMetadata::read(&component_metadata_path)? == metadata
        {
            return Ok(());
        }
        info!(component.name= ?metadata.name, component.id = ?metadata.component_id.0, is_timestamp_source = metadata.is_timestamp_source(), "setting component metadata");
        metadata.write(&component_metadata_path)?;
        // Sync the name to the Component for better warning messages
        if let Some(component) = self.components.get(&metadata.component_id) {
            component.set_name(metadata.name.clone());
        }
        self.component_metadata
            .insert(metadata.component_id, metadata);
        Ok(())
    }

    pub fn get_or_insert_msg_log(
        &mut self,
        id: PacketId,
        db_path: &Path,
    ) -> Result<&mut MsgLog, Error> {
        Ok(match self.msg_logs.entry(id) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let msg_log = MsgLog::create(
                    db_path
                        .join("msgs")
                        .join(u16::from_le_bytes(id).to_string()),
                )?;
                entry.insert(msg_log)
            }
        })
    }

    pub fn set_msg_metadata(
        &mut self,
        id: PacketId,
        metadata: MsgMetadata,
        db_path: &Path,
    ) -> Result<(), Error> {
        let msg_log = self.get_or_insert_msg_log(id, db_path)?;
        msg_log.set_metadata(metadata)?;
        Ok(())
    }

    /// Truncate all components and message logs, clearing all data while preserving schemas and metadata.
    pub fn truncate_all(&self) {
        for component in self.components.values() {
            component.truncate();
        }
        for msg_log in self.msg_logs.values() {
            msg_log.truncate();
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentSchema {
    pub prim_type: PrimType,
    pub dim: SmallVec<[usize; 4]>,
}

impl Serialize for ComponentSchema {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let schema = self.to_schema();
        schema.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ComponentSchema {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let schema = Schema::<Vec<u64>>::deserialize(deserializer)?;
        Ok(schema.into())
    }
}

impl ComponentSchema {
    pub fn new(prim_type: PrimType, shape: &[usize]) -> Self {
        let dim = shape.into();
        ComponentSchema { prim_type, dim }
    }

    pub fn size(&self) -> usize {
        self.dim.iter().product::<usize>() * self.prim_type.size()
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Ok(postcard::from_bytes(&data)?)
    }

    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let data = postcard::to_allocvec(&self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn to_schema(&self) -> Schema<Vec<u64>> {
        Schema::new(self.prim_type, self.shape()).expect("failed to create shape")
    }

    pub fn shape(&self) -> SmallVec<[u64; 4]> {
        self.dim.iter().map(|&x| x as u64).collect()
    }

    pub fn parse_value<'a>(&'a self, buf: &'a [u8]) -> Result<(usize, ComponentView<'a>), Error> {
        let size = self.size();
        let buf = buf
            .get(..size)
            .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?;
        let dim = &self.dim;
        let view = match self.prim_type {
            PrimType::U8 => ComponentView::U8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U16 => ComponentView::U16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U32 => ComponentView::U32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::U64 => ComponentView::U64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I8 => ComponentView::I8(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I16 => ComponentView::I16(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I32 => ComponentView::I32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::I64 => ComponentView::I64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::Bool => ComponentView::Bool(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F32 => ComponentView::F32(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
            PrimType::F64 => ComponentView::F64(
                nox::ArrayView::from_bytes_shape_unchecked(buf, dim)
                    .ok_or(Error::Impeller(impeller2::error::Error::BufferOverflow))?,
            ),
        };
        Ok((size, view))
    }
}

impl From<Schema<Vec<u64>>> for ComponentSchema {
    fn from(value: Schema<Vec<u64>>) -> Self {
        let prim_type = value.prim_type();
        ComponentSchema {
            prim_type,
            dim: value.shape().into(),
        }
    }
}

impl From<ComponentSchema> for Schema<Vec<u64>> {
    fn from(value: ComponentSchema) -> Self {
        value.to_schema()
    }
}

pub trait MetadataExt: Sized + Serialize + DeserializeOwned {
    fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let data = std::fs::read(path)?;
        Ok(postcard::from_bytes(&data)?)
    }
    fn write(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let data = postcard::to_allocvec(&self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

impl MetadataExt for EntityMetadata {}
impl MetadataExt for ComponentMetadata {}
impl MetadataExt for MsgMetadata {}

#[derive(Clone)]
pub struct Component {
    pub component_id: ComponentId,
    pub time_series: TimeSeries,
    pub schema: ComponentSchema,
}

impl Component {
    pub fn create(
        db_path: &Path,
        component_id: ComponentId,
        name: String,
        schema: ComponentSchema,
        start_timestamp: Timestamp,
    ) -> Result<Self, Error> {
        let component_path = db_path.join(component_id.to_string());
        std::fs::create_dir_all(&component_path)?;
        let component_schema_path = component_path.join("schema");
        if !component_schema_path.exists() {
            schema.write(component_schema_path)?;
        }
        let time_series = TimeSeries::create(
            component_path.clone(),
            name,
            start_timestamp,
            schema.size() as u64,
        )?;
        Ok(Component {
            component_id,
            time_series,
            schema,
        })
    }

    pub fn open(
        path: impl AsRef<Path>,
        component_id: ComponentId,
        name: String,
        schema: ComponentSchema,
    ) -> Result<Self, Error> {
        let time_series = TimeSeries::open(path, name)?;
        Ok(Component {
            component_id,
            time_series,
            schema,
        })
    }

    fn as_vtable_op(&self) -> Arc<OpBuilder> {
        schema(
            self.schema.prim_type,
            &self.schema.shape(),
            component(self.component_id),
        )
    }

    fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &[u8])> {
        self.time_series.get_nearest(timestamp)
    }

    fn get_range(&self, range: &Range<Timestamp>) -> Option<(&[Timestamp], &[u8])> {
        self.time_series.get_range(range)
    }

    fn sync_all(&self) -> Result<(), Error> {
        self.time_series.sync_all()
    }

    /// Truncate the component, clearing all time-series data while preserving the schema.
    pub fn truncate(&self) {
        self.time_series.truncate();
    }

    /// Update the human-readable name for this component.
    ///
    /// This is used to provide better context in warning messages.
    pub fn set_name(&self, name: String) {
        self.time_series.set_name(name);
    }
}

struct DBSink<'a> {
    components: &'a HashMap<ComponentId, Component>,
    snapshot_barrier: &'a SnapshotBarrier,
    last_updated: &'a AtomicCell<Timestamp>,
    sunk_new_time_series: bool,
    table_received: Timestamp,
}

impl Decomponentize for DBSink<'_> {
    type Error = Error;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: impeller2::types::ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Error> {
        let _snapshot_guard = self.snapshot_barrier.enter_writer();
        let mut timestamp = timestamp.unwrap_or(self.table_received);
        let value_buf = value.as_bytes();
        let Some(component) = self.components.get(&component_id) else {
            return Err(Error::ComponentNotFound(component_id));
        };
        let time_series_empty = component.time_series.index().is_empty();
        // When timestamps are auto-generated (no explicit timestamp provided), concurrent writers
        // may occasionally observe a slightly newer last timestamp and reject with
        // TimeTravel. In that case, clamp the timestamp to last+1 and retry.
        if let Err(err) = component.time_series.push_buf(timestamp, value_buf) {
            match err {
                Error::TimeTravel if timestamp == self.table_received => {
                    // Retry with a monotonic bump based on the latest sample seen.
                    let mut attempts = 0u8;
                    loop {
                        if let Some((last_ts, _)) = component.time_series.latest() {
                            // ensure strictly non-decreasing order
                            timestamp = Timestamp(last_ts.0.saturating_add(1));
                        } else {
                            timestamp = self.table_received;
                        }
                        match component.time_series.push_buf(timestamp, value_buf) {
                            Ok(()) => break,
                            Err(Error::TimeTravel) if attempts < 8 => {
                                attempts = attempts.saturating_add(1);
                                continue;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
                other => return Err(other),
            }
        }
        if time_series_empty {
            debug!("sunk new time series for component {}", component_id);
            self.sunk_new_time_series = true;
        }
        self.last_updated.update_max(timestamp);
        Ok(())
    }
}

pub struct Server {
    pub listener: TcpListener,
    pub db: Arc<DB>,
}

impl Server {
    pub fn new(path: impl AsRef<Path>, addr: SocketAddr) -> Result<Server, Error> {
        let path = path.as_ref();
        info!(?addr, "listening with path {:?}", path.display());
        let listener = TcpListener::bind(addr)?;
        Server::from_listener(listener, path)
    }

    pub fn from_listener(listener: TcpListener, path: impl AsRef<Path>) -> Result<Server, Error> {
        let path = path.as_ref().to_path_buf();
        let db = if path.exists() {
            DB::open(path)?
        } else {
            DB::create(path)?
        };
        let db = Arc::new(db);
        Ok(Server { listener, db })
    }

    pub async fn run(self) -> Result<(), Error> {
        let Self { listener, db } = self;
        let addr = listener.local_addr()?;
        let udp_db = db.clone();
        stellarator::struc_con::stellar(move || Self::handle_udp(addr, udp_db));
        loop {
            let stream = listener.accept().await?;
            let conn_db = db.clone();
            stellarator::struc_con::stellar(move || handle_conn(stream, conn_db));
        }
    }

    pub async fn handle_udp(addr: SocketAddr, db: Arc<DB>) -> Result<(), Error> {
        let socket = UdpSocket::bind(addr)?;
        let (rx, tx) = socket.split();
        let rx = PacketStream::new(rx);
        let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
        handle_conn_inner(tx, rx, db).await?;
        Ok(())
    }
}

pub async fn handle_conn(stream: TcpStream, db: Arc<DB>) {
    let (rx, tx) = stream.split();
    let rx = PacketStream::new(rx);
    let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
    match handle_conn_inner(tx, rx, db).await {
        Ok(_) => {}
        Err(err) if err.is_stream_closed() => {}
        Err(err) => {
            warn!(?err, "error handling stream")
        }
    }
}

async fn handle_conn_inner<A: AsyncRead + AsyncWrite + 'static>(
    mut tx: Arc<Mutex<PacketSink<OwnedWriter<A>>>>,
    mut rx: PacketStream<OwnedReader<A>>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut resp_pkt = LenPacket::new(PacketTy::Msg, [0, 0], 8 * 1024 * 1024);
    loop {
        let pkt = rx.next(buf).await?;
        let req_id = pkt.req_id();
        let mut pkt_tx = PacketTx {
            req_id,
            tx,
            pkt: Some(resp_pkt),
        };
        let result = handle_packet(&pkt, &db, &mut pkt_tx).await;
        buf = pkt.into_buf().into_inner();
        match result {
            Ok(_) => {}
            Err(err) if err.is_stream_closed() => {}
            Err(err) => {
                debug!(?err, "error handling packet");
                if let Err(err) = pkt_tx
                    .send_msg(&ErrorResponse {
                        description: err.to_string(),
                    })
                    .await
                {
                    warn!(?err, "error sending err resp");
                }
            }
        }
        resp_pkt = pkt_tx.pkt.expect("len pkt taken and not given back");
        tx = pkt_tx.tx;
    }
}

pub struct PacketTx<A: AsyncWrite + 'static> {
    req_id: RequestId,
    tx: Arc<Mutex<PacketSink<OwnedWriter<A>>>>,
    pkt: Option<LenPacket>,
}

impl<A: AsyncWrite + 'static> Clone for PacketTx<A> {
    fn clone(&self) -> Self {
        Self {
            req_id: self.req_id,
            tx: self.tx.clone(),
            pkt: self.pkt.clone(),
        }
    }
}

impl<A: AsyncWrite + 'static> PacketTx<A> {
    pub async fn send_msg<M: Msg>(&mut self, msg: &M) -> Result<(), Error> {
        let req_id = self.req_id;
        self.send_with_builder(|pkt| {
            let header = PacketHeader {
                packet_ty: impeller2::types::PacketTy::Msg,
                id: M::ID,
                req_id,
            };
            pkt.as_mut_packet().header = header;
            pkt.clear();
            postcard::serialize_with_flavor(&msg, pkt).map_err(Error::from)
        })
        .await
    }

    pub async fn send_with_builder(
        &mut self,
        builder: impl FnOnce(&mut LenPacket) -> Result<(), Error> + '_,
    ) -> Result<(), Error> {
        let mut pkt = self.pkt.take().expect("missing len pkt");
        pkt.clear();
        if let Err(err) = builder(&mut pkt) {
            self.pkt = Some(pkt);
            return Err(err);
        }
        pkt.as_mut_packet().header.req_id = self.req_id;
        let tx = self.tx.lock().await;
        let res = rent!(tx.send(pkt).await, pkt);
        self.pkt = Some(pkt);
        res.map_err(Error::from)
    }

    pub async fn send_time_series(
        &mut self,
        id: PacketId,
        timestamps: &[Timestamp],
        data: &[u8],
    ) -> Result<(), Error> {
        let req_id = self.req_id;
        self.send_with_builder(|pkt| {
            let header = PacketHeader {
                packet_ty: impeller2::types::PacketTy::TimeSeries,
                id,
                req_id,
            };
            pkt.as_mut_packet().header = header;
            pkt.clear();
            pkt.extend_from_slice(&(timestamps.len() as u64).to_le_bytes());
            pkt.extend_from_slice(timestamps.as_bytes());
            pkt.extend_from_slice(data);
            Ok(())
        })
        .await
    }
}

async fn handle_packet<A: AsyncWrite + 'static>(
    pkt: &Packet<Slice<Vec<u8>>>,
    db: &Arc<DB>,
    tx: &mut PacketTx<A>,
) -> Result<(), Error> {
    trace!(?pkt, "handling pkt");

    match &pkt {
        Packet::Msg(m) if m.id == VTableMsg::ID => {
            let vtable = m.parse::<VTableMsg>()?;
            db.insert_vtable(vtable)?;
        }
        Packet::Msg(m) if m.id == UdpUnicast::ID => {
            let udp_broadcast = m.parse::<UdpUnicast>()?;
            let db = db.clone();
            let addr = udp_broadcast
                .addr
                .to_socket_addrs()?
                .next()
                .ok_or_else(|| {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "missing socket ip",
                    ))
                })?;
            handle_unicast_stream(addr, udp_broadcast.stream, db);
        }
        Packet::Msg(m) if m.id == Stream::ID => {
            let stream = m.parse::<Stream>()?;
            let tx = tx.tx.clone();
            let db = db.clone();
            handle_stream(tx, stream, db, m.req_id);
        }
        Packet::Msg(m) if m.id == SetStreamState::ID => {
            let set_stream_state = m.parse::<SetStreamState>()?;
            let stream_id = set_stream_state.id;
            db.with_state(|s| {
                let Some(state) = s.streams.get(&stream_id) else {
                    return Err(Error::StreamNotFound(stream_id));
                };
                debug!(msg = ?set_stream_state, "set_stream_state received");
                if let Some(playing) = set_stream_state.playing {
                    state.set_playing(playing);
                }
                if let Some(timestamp) = set_stream_state.timestamp {
                    state.set_timestamp(timestamp);
                }
                if let Some(frequency) = set_stream_state.frequency {
                    state.set_frequency(frequency);
                }
                if let Some(time_step) = set_stream_state.time_step {
                    state.set_time_step(time_step);
                }
                Ok(())
            })?;
        }
        Packet::Msg(m) if m.id == GetSchema::ID => {
            let get_schema = m.parse::<GetSchema>()?;
            let schema = db.with_state(|state| {
                state
                    .components
                    .iter()
                    .filter(|(component_id, _)| *component_id == &get_schema.component_id)
                    .map(|(_, component)| component.schema.to_schema())
                    .next()
                    .ok_or(Error::ComponentNotFound(get_schema.component_id))
            })?;
            tx.send_msg(&SchemaMsg(schema)).await?;
        }
        Packet::Msg(m) if m.id == GetTimeSeries::ID => {
            let get_time_series = m.parse::<GetTimeSeries>()?;
            let GetTimeSeries {
                component_id,
                range,
                limit,
                id,
            } = get_time_series;
            let component = db.with_state(|state| {
                let Some(component) = state.components.get(&component_id) else {
                    return Err(Error::ComponentNotFound(component_id));
                };
                Ok(component.clone())
            })?;
            let Some((timestamps, data)) = component.get_range(&range) else {
                return Err(Error::TimeRangeOutOfBounds {
                    range,
                    component_id: component.component_id,
                    latest: component.time_series.latest().map(|x| *x.0),
                });
            };
            let size = component.schema.size();
            let (timestamps, data) = if let Some(limit) = limit {
                let len = timestamps.len().min(limit);
                (&timestamps[..len], &data[..len * size])
            } else {
                (timestamps, data)
            };
            tx.send_time_series(id, timestamps, data).await?;
        }
        Packet::Msg(m) if m.id == SetComponentMetadata::ID => {
            let _snapshot_guard = db.snapshot_barrier.enter_writer();
            let SetComponentMetadata(metadata) = m.parse::<SetComponentMetadata>()?;
            db.with_state_mut(|state| state.set_component_metadata(metadata, &db.path))?;
        }
        Packet::Msg(m) if m.id == GetComponentMetadata::ID => {
            let GetComponentMetadata { component_id } = m.parse::<GetComponentMetadata>()?;

            tx.send_with_builder(|pkt| {
                let header = PacketHeader {
                    packet_ty: impeller2::types::PacketTy::Msg,
                    id: ComponentMetadata::ID,
                    req_id: m.req_id,
                };
                pkt.as_mut_packet().header = header;
                pkt.clear();
                db.with_state(|state| {
                    let Some(metadata) = state.component_metadata.get(&component_id) else {
                        return Err(Error::ComponentNotFound(component_id));
                    };
                    postcard::serialize_with_flavor(&metadata, pkt).map_err(Error::from)
                })
            })
            .await?;
        }

        Packet::Msg(m) if m.id == DumpMetadata::ID => {
            let msg = db.with_state(|state| {
                let component_metadata = state.component_metadata.values().cloned().collect();

                let msg_metadata = state
                    .msg_logs
                    .values()
                    .flat_map(|m| m.metadata())
                    .cloned()
                    .collect();
                DumpMetadataResp {
                    component_metadata,
                    msg_metadata,
                    db_config: state.db_config.clone(),
                }
            });
            tx.send_msg(&msg).await?;
        }
        Packet::Msg(m) if m.id == DumpSchema::ID => {
            let msg = db.with_state(|state| {
                let schemas = state
                    .components
                    .values()
                    .map(|c| (c.component_id, c.schema.to_schema()))
                    .collect();
                DumpSchemaResp { schemas }
            });

            tx.send_msg(&msg).await?;
        }

        Packet::Msg(m) if m.id == SubscribeLastUpdated::ID => {
            let mut tx = tx.clone();
            let db = db.clone();
            stellarator::spawn(async move {
                loop {
                    let last_updated = db.last_updated.latest();
                    {
                        match tx.send_msg(&LastUpdated(last_updated)).await {
                            Err(err) if err.is_stream_closed() => return,
                            Err(err) => {
                                warn!(?err, "failed to send packet");
                                return;
                            }
                            _ => (),
                        }
                    }
                    db.last_updated.wait_for(|time| time > last_updated).await;
                }
            });
        }
        Packet::Msg(m) if m.id == SetDbConfig::ID => {
            let _snapshot_guard = db.snapshot_barrier.enter_writer();
            let SetDbConfig {
                recording,
                metadata,
            } = m.parse::<SetDbConfig>()?;
            if let Some(recording) = recording {
                db.with_state_mut(|s| {
                    s.db_config.recording = recording;
                });
                db.recording_cell.set_playing(recording);
            }
            db.with_state_mut(|s| {
                s.db_config.metadata.extend(metadata);
            });
            db.save_db_state()?;
            drop(_snapshot_guard);
            tx.send_msg(&db.db_config()).await?;
        }
        Packet::Msg(m) if m.id == GetEarliestTimestamp::ID => {
            tx.send_msg(&EarliestTimestamp(db.earliest_timestamp.latest()))
                .await?;
        }
        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let settings = db.db_config();
            tx.send_msg(&settings).await?;
        }
        Packet::Table(table) => {
            trace!(table.len = table.buf.len(), "sinking table");
            db.with_state(|state| {
                let mut sink = DBSink {
                    components: &state.components,
                    snapshot_barrier: &db.snapshot_barrier,
                    last_updated: &db.last_updated,
                    sunk_new_time_series: false,
                    table_received: db.apply_implicit_timestamp(),
                };
                table.sink(&state.vtable_registry, &mut sink)??;
                if sink.sunk_new_time_series {
                    db.vtable_gen.fetch_add(1, atomic::Ordering::SeqCst);
                }
                Ok::<_, Error>(())
            })?;
        }

        Packet::Msg(m) if m.id == GetDbSettings::ID => {
            let settings = db.db_config();
            tx.send_msg(&settings).await?;
        }
        Packet::Msg(m) if m.id == SQLQuery::ID => {
            let SQLQuery(query) = m.parse::<SQLQuery>()?;
            let db = db.clone();
            let (tokio_tx, rx) = thingbuf::mpsc::channel::<Vec<u8>>(4);
            let res = stellarator::struc_con::tokio(move |_| async move {
                let mut ctx = db.as_session_context()?;
                db.insert_views(&mut ctx).await?;
                let df = ctx.sql(&query).await?;
                let mut stream = df.execute_stream().await?;

                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    let mut buf = vec![];
                    let mut writer =
                        ::arrow::ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
                    writer.write(&batch)?;
                    writer.finish()?;
                    let _ = tokio_tx.send(buf).await;
                }
                Ok::<_, Error>(())
            })
            .join();
            while let Some(batch) = rx.recv().await {
                tx.send_msg(&ArrowIPC {
                    batch: Some(Cow::Owned(batch)),
                })
                .await?;
            }
            res.await??;
            tx.send_msg(&ArrowIPC { batch: None }).await?;
        }
        Packet::Msg(m) if m.id == SparklineQuery::ID => {
            let query = m.parse::<SparklineQuery>()?;
            let table_name = query.table_name;
            let max_points = query.max_points as usize;

            // Find the component matching this table name
            let result = db.with_state(|state| {
                for component in state.components.values() {
                    // Skip components without metadata instead of returning None,
                    // which would abort the search and miss other valid components
                    let Some(component_metadata) =
                        state.component_metadata.get(&component.component_id)
                    else {
                        continue;
                    };
                    let component_name = crate::arrow::sanitize_sql_table_name(
                        &component_metadata.name.to_case(convert_case::Case::Snake),
                    );

                    if component_name == table_name {
                        // Get the raw data as byte slices
                        let time_bytes = component.time_series.index().data();
                        let value_bytes = component.time_series.data().data();

                        // Get the number of data points
                        let timestamp_size = std::mem::size_of::<Timestamp>();
                        let num_points = time_bytes.len() / timestamp_size;

                        if num_points == 0 {
                            return Some((vec![], vec![]));
                        }

                        // Convert timestamps to i64 array
                        let times: Vec<i64> = (0..num_points)
                            .map(|i| {
                                let offset = i * timestamp_size;
                                let bytes: [u8; 8] = time_bytes[offset..offset + 8]
                                    .try_into()
                                    .unwrap_or([0u8; 8]);
                                i64::from_le_bytes(bytes)
                            })
                            .collect();

                        // Extract scalar values from the data column
                        // For vector types, we take just the first element
                        let element_size = component.time_series.element_size();
                        let dim: usize = component.schema.dim.iter().product();
                        let scalar_size = element_size / dim.max(1);

                        let prim_type = component.schema.prim_type;
                        let values: Vec<f64> = (0..num_points)
                            .map(|i| {
                                let offset = i * element_size;
                                if offset + scalar_size > value_bytes.len() {
                                    return f64::NAN;
                                }
                                crate::utils::read_prim_as_f64(prim_type, value_bytes, offset)
                            })
                            .collect();

                        return Some((times, values));
                    }
                }
                None
            });

            match result {
                Some((times, values)) => {
                    // Apply LTTB downsampling
                    let (ds_times, ds_values) =
                        crate::arrow::lttb::lttb_downsample_arrays(&times, &values, max_points);

                    // Build Arrow RecordBatch
                    use ::arrow::array::{Float64Array, TimestampMicrosecondArray};
                    use ::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
                    use ::arrow::record_batch::RecordBatch;

                    let time_array = TimestampMicrosecondArray::from(ds_times);
                    let value_array = Float64Array::from(ds_values);

                    let schema = Arc::new(Schema::new(vec![
                        Field::new(
                            "time",
                            DataType::Timestamp(TimeUnit::Microsecond, None),
                            false,
                        ),
                        Field::new("value", DataType::Float64, true),
                    ]));

                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(time_array), Arc::new(value_array)],
                    )
                    .map_err(Error::Arrow)?;

                    // Serialize to Arrow IPC
                    let mut buf = vec![];
                    let mut writer =
                        ::arrow::ipc::writer::StreamWriter::try_new(&mut buf, &schema)?;
                    writer.write(&batch)?;
                    writer.finish()?;

                    tx.send_msg(&ArrowIPC {
                        batch: Some(Cow::Owned(buf)),
                    })
                    .await?;
                }
                None => {
                    // Table not found - send empty result
                    use ::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
                    use ::arrow::record_batch::RecordBatch;

                    let schema = Arc::new(Schema::new(vec![
                        Field::new(
                            "time",
                            DataType::Timestamp(TimeUnit::Microsecond, None),
                            false,
                        ),
                        Field::new("value", DataType::Float64, true),
                    ]));

                    let batch = RecordBatch::new_empty(schema.clone());

                    let mut buf = vec![];
                    let mut writer =
                        ::arrow::ipc::writer::StreamWriter::try_new(&mut buf, &schema)?;
                    writer.write(&batch)?;
                    writer.finish()?;

                    tx.send_msg(&ArrowIPC {
                        batch: Some(Cow::Owned(buf)),
                    })
                    .await?;
                }
            }

            // Send completion marker
            tx.send_msg(&ArrowIPC { batch: None }).await?;
        }
        Packet::Msg(m) if m.id == PlotOverviewQuery::ID => {
            let query = m.parse::<PlotOverviewQuery>()?;
            let PlotOverviewQuery {
                id: request_id,
                component_id,
                range,
                max_points,
                element_index,
            } = query;

            let component = db.with_state(|state| {
                let Some(component) = state.components.get(&component_id) else {
                    return Err(Error::ComponentNotFound(component_id));
                };
                Ok(component.clone())
            })?;

            // Get the data for the requested range
            // The range is already in Timestamp format from the query
            let (timestamps, data) = match component.get_range(&range) {
                Some(result) => result,
                None => {
                    // Range out of bounds - send empty result
                    tx.send_time_series(request_id, &[], &[]).await?;
                    return Ok(());
                }
            };

            let num_points = timestamps.len();
            if num_points == 0 {
                tx.send_time_series(request_id, &[], &[]).await?;
                return Ok(());
            }

            // Extract values for the specified element_index
            let element_size = component.time_series.element_size();
            let dim: usize = component.schema.dim.iter().product();

            // Validate element_index is within bounds
            if element_index >= dim {
                // Invalid element_index - send empty result rather than reading wrong data
                tracing::warn!(
                    component_id = ?component_id,
                    element_index = element_index,
                    dim = dim,
                    "PlotOverviewQuery element_index out of bounds"
                );
                tx.send_time_series(request_id, &[], &[]).await?;
                return Ok(());
            }

            let scalar_size = element_size / dim.max(1);
            let element_offset = element_index * scalar_size;

            // Convert timestamps to i64 for LTTB (Timestamp wraps i64)
            let times: Vec<i64> = timestamps.iter().map(|t| t.0).collect();

            // Extract values as f64 for the specified element
            let prim_type = component.schema.prim_type;
            let values: Vec<f64> = (0..num_points)
                .map(|i| {
                    let offset = i * element_size + element_offset;
                    if offset + scalar_size > data.len() {
                        return f64::NAN;
                    }
                    crate::utils::read_prim_as_f64(prim_type, data, offset)
                })
                .collect();

            // Apply LTTB downsampling
            let (ds_times, ds_values) =
                crate::arrow::lttb::lttb_downsample_arrays(&times, &values, max_points as usize);

            // Convert back to the format expected by the client
            let ds_timestamps: Vec<Timestamp> = ds_times.iter().map(|t| Timestamp(*t)).collect();

            // Convert f64 values back to the original type's byte representation
            // For simplicity, we'll send as f32 since that's what the plot panel uses internally
            let ds_data: Vec<u8> = ds_values
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect();

            tx.send_time_series(request_id, &ds_timestamps, &ds_data)
                .await?;
        }
        Packet::Msg(m) if m.id == SetMsgMetadata::ID => {
            let _snapshot_guard = db.snapshot_barrier.enter_writer();
            let SetMsgMetadata { id, metadata } = m.parse::<SetMsgMetadata>()?;
            info!(
                msg.name = %metadata.name,
                msg.id = ?id,
                "setting msg metadata"
            );
            db.with_state_mut(|s| s.set_msg_metadata(id, metadata, &db.path))?;
            drop(_snapshot_guard);
        }
        Packet::Msg(m) if m.id == MsgStream::ID => {
            let _snapshot_guard = db.snapshot_barrier.enter_writer();
            let MsgStream { msg_id } = m.parse::<MsgStream>()?;
            let msg_log =
                db.with_state_mut(|s| s.get_or_insert_msg_log(msg_id, &db.path).cloned())?;
            drop(_snapshot_guard);
            let req_id = m.req_id;
            stellarator::spawn(handle_msg_stream(msg_id, req_id, msg_log, tx.tx.clone()));
        }
        Packet::Msg(m) if m.id == FixedRateMsgStream::ID => {
            let _snapshot_guard = db.snapshot_barrier.enter_writer();
            let FixedRateMsgStream { msg_id, fixed_rate } = m.parse::<FixedRateMsgStream>()?;
            let msg_log =
                db.with_state_mut(|s| s.get_or_insert_msg_log(msg_id, &db.path).cloned())?;
            let stream_state =
                db.get_or_insert_fixed_rate_state(fixed_rate.stream_id, fixed_rate.behavior);
            drop(_snapshot_guard);
            stellarator::spawn(handle_fixed_rate_msg_stream(
                msg_id,
                m.req_id,
                msg_log,
                tx.tx.clone(),
                stream_state,
            ));
        }
        Packet::Msg(m) if m.id == GetMsgMetadata::ID => {
            let GetMsgMetadata { msg_id } = m.parse::<GetMsgMetadata>()?;
            let Some(metadata) =
                db.with_state(|s| s.msg_logs.get(&msg_id).and_then(|m| m.metadata().cloned()))
            else {
                return Err(Error::MsgNotFound(msg_id));
            };
            tx.send_msg(&metadata).await?;
        }
        Packet::Msg(m) if m.id == GetMsgs::ID => {
            let GetMsgs {
                msg_id,
                range,
                limit,
            } = m.parse::<GetMsgs>()?;
            let msg_log = db.with_state_mut(|s| {
                s.msg_logs
                    .get(&msg_id)
                    .ok_or(Error::MsgNotFound(msg_id))
                    .cloned()
            })?;
            let iter = msg_log.get_range(&range).map(|(t, b)| (t, b.to_vec()));
            let data = if let Some(limit) = limit {
                iter.take(limit).collect()
            } else {
                iter.collect()
            };
            tx.send_msg(&MsgBatch { data }).await?;
        }
        Packet::Msg(m) if m.id == SaveArchive::ID => {
            let SaveArchive { path, format } = m.parse()?;
            db.save_archive(&path, format)?;
            tx.send_msg(&ArchiveSaved { path }).await?;
        }
        Packet::Msg(m) if m.id == VTableStream::ID => {
            let VTableStream { id } = m.parse::<VTableStream>()?;
            let vtable = db
                .with_state(|state| state.vtable_registry.get(&id).cloned())
                .ok_or(Error::InvalidMsgId)?;
            stellarator::spawn(handle_vtable_stream(
                id,
                vtable,
                db.clone(),
                tx.tx.clone(),
                m.req_id,
            ));
        }
        Packet::Msg(m) if m.id == UdpVTableStream::ID => {
            let UdpVTableStream { id, addr } = m.parse::<UdpVTableStream>()?;
            let addr = addr.to_socket_addrs()?.next().ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "missing socket ip",
                ))
            })?;
            db.with_state_mut(|state| {
                // ensure idempotency
                if state.udp_vtable_streams.contains(&(addr, id)) {
                    return Ok(());
                }
                let vtable = state
                    .vtable_registry
                    .get(&id)
                    .ok_or(Error::InvalidMsgId)?
                    .clone();
                let db = db.clone();
                let req_id = m.req_id;
                stellarator::struc_con::stellar(move || async move {
                    info!("VTable streaming to udp://{}", addr);
                    let mut socket = UdpSocket::ephemeral()?;
                    socket.connect(addr);
                    let (_, tx) = socket.split();
                    let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
                    handle_vtable_stream(id, vtable, db, tx, req_id).await
                });
                state.udp_vtable_streams.insert((addr, id));
                Ok::<_, Error>(())
            })?;
        }
        Packet::Msg(m) => {
            let timestamp = m.timestamp.unwrap_or_else(|| db.apply_implicit_timestamp());
            db.push_msg(timestamp, m.id, &m.buf)?
        }
        _ => {}
    }
    Ok(())
}

pub async fn handle_msg_stream<A: AsyncWrite>(
    msg_id: PacketId,
    req_id: RequestId,
    msg_log: MsgLog,
    tx: Arc<Mutex<PacketSink<A>>>,
) -> Result<(), Error> {
    let mut pkt = LenPacket::msg(msg_id, 64).with_request_id(req_id);
    loop {
        msg_log.wait().await;
        let Some((_timestamp, msg)) = msg_log.latest() else {
            continue;
        };
        let tx = tx.lock().await;
        pkt.clear();
        pkt.extend_from_slice(msg);
        rent!(tx.send(pkt).await, pkt)?;
    }
}

pub async fn handle_fixed_rate_msg_stream<A: AsyncWrite>(
    msg_id: PacketId,
    req_id: RequestId,
    msg_log: MsgLog,
    tx: Arc<Mutex<PacketSink<A>>>,
    stream_state: Arc<FixedRateStreamState>,
) -> Result<(), Error> {
    let mut pkt = LenPacket::msg_with_timestamp(msg_id, Timestamp(0), 64).with_request_id(req_id);
    let mut last_sent_timestamp: Option<Timestamp> = None;
    let mut current_timestamp = stream_state.current_timestamp();
    loop {
        if !stream_state.wait_for_playing().await {
            return Ok(());
        }
        let Some((msg_timestamp, msg)) = msg_log.get_nearest(current_timestamp) else {
            // Wait for data to arrive in the msg_log.
            // This yields to the runtime (preventing scheduler starvation) without
            // advancing the stream timestamp, so we don't skip past incoming video frames.
            msg_log.wait().await;
            continue;
        };

        if Some(msg_timestamp) == last_sent_timestamp {
            current_timestamp = stream_state.wait_for_next_tick(current_timestamp).await;
            continue;
        }

        {
            let tx = tx.lock().await;
            pkt.clear();
            pkt.extend_from_slice(msg_timestamp.as_bytes());
            pkt.extend_from_slice(msg);
            rent!(tx.send(pkt).await, pkt)?;
        }
        last_sent_timestamp = Some(msg_timestamp);

        current_timestamp = stream_state.wait_for_next_tick(current_timestamp).await;
    }
}

pub struct FixedRateStreamState {
    stream_id: StreamId,
    time_step: AtomicU64,
    frequency: AtomicU64,
    is_scrubbed: AtomicBool,
    current_tick: AtomicI64,
    playing_cell: PlayingCell,
    tick_notify: WaitQueue,
    driver_spawned: AtomicBool,
}

impl FixedRateStreamState {
    fn new(
        stream_id: StreamId,
        time_step: Duration,
        current_tick: Timestamp,
        frequency: u64,
    ) -> FixedRateStreamState {
        FixedRateStreamState {
            stream_id,
            time_step: AtomicU64::new(time_step.as_nanos() as u64),
            is_scrubbed: AtomicBool::new(false),
            current_tick: AtomicI64::new(current_tick.0),
            playing_cell: PlayingCell::new(true),
            frequency: AtomicU64::new(frequency),
            tick_notify: WaitQueue::new(),
            driver_spawned: AtomicBool::new(false),
        }
    }

    fn is_scrubbed(&self) -> bool {
        self.is_scrubbed.swap(false, atomic::Ordering::Relaxed)
    }

    fn set_timestamp(&self, timestamp: Timestamp) {
        self.is_scrubbed.store(true, atomic::Ordering::SeqCst);
        self.current_tick
            .store(timestamp.0, atomic::Ordering::SeqCst);
        self.playing_cell.wait_cell.wake_all();
        self.tick_notify.wake_all();
    }

    fn set_frequency(&self, frequency: u64) {
        self.frequency.store(frequency, atomic::Ordering::SeqCst);
    }

    fn set_time_step(&self, time_step: Duration) {
        self.time_step
            .store(time_step.as_nanos() as u64, atomic::Ordering::SeqCst);
    }

    fn sleep_time(&self) -> Duration {
        Duration::from_micros(1_000_000 / self.frequency.load(atomic::Ordering::Relaxed))
    }

    fn current_timestamp(&self) -> Timestamp {
        Timestamp(self.current_tick.load(atomic::Ordering::Relaxed))
    }

    fn advance_tick(&self) {
        let time_step = self.time_step.load(atomic::Ordering::Relaxed) as i64;
        self.current_tick
            .fetch_add(time_step, atomic::Ordering::Release);
        self.tick_notify.wake_all();
    }

    async fn wait_for_next_tick(&self, last_seen: Timestamp) -> Timestamp {
        let _ = self
            .tick_notify
            .wait_for(|| self.current_timestamp() != last_seen)
            .await;
        self.current_timestamp()
    }

    fn is_playing(&self) -> bool {
        self.playing_cell.is_playing()
    }

    fn set_playing(&self, playing: bool) {
        self.playing_cell.set_playing(playing)
    }

    async fn wait_for_playing(&self) -> bool {
        self.playing_cell
            .wait_cell
            .wait_for(|| self.is_playing() || self.is_scrubbed())
            .await
            .is_ok()
    }
}

async fn run_tick_driver(state: Arc<FixedRateStreamState>) {
    loop {
        // Respect play/pause
        if !state.wait_for_playing().await {
            return;
        }
        let sleep_time = state.sleep_time();
        // Sleep for one tick period, interruptible by play/pause changes
        futures_lite::future::race(
            async {
                state.playing_cell.wait_for_change().await;
            },
            stellarator::sleep(sleep_time),
        )
        .await;
        // Only advance if still playing
        if state.is_playing() {
            state.advance_tick();
        }
    }
}

fn handle_unicast_stream(addr: SocketAddr, stream: Stream, db: Arc<DB>) {
    stellarator::struc_con::stellar(move || async move {
        info!("UDP unicasting to {}", addr);
        let mut socket = UdpSocket::ephemeral()?;
        socket.connect(addr);
        let (_, tx) = socket.split();
        let tx = Arc::new(Mutex::new(PacketSink::new(tx)));
        handle_stream(tx.clone(), stream.clone(), db.clone(), 0)
            .await
            .unwrap();
        Ok::<_, Error>(())
    });
}

fn handle_stream<A: AsyncWrite + 'static>(
    tx: Arc<Mutex<PacketSink<A>>>,
    stream: Stream,
    db: Arc<DB>,
    req_id: RequestId,
) -> stellarator::JoinHandle<()> {
    match stream.behavior {
        StreamBehavior::RealTime => stellarator::spawn(async move {
            match handle_real_time_stream(tx, req_id, db).await {
                Ok(_) => {}
                Err(err) if err.is_stream_closed() => {}
                Err(err) => {
                    warn!(?err, "error streaming data");
                }
            }
        }),
        StreamBehavior::FixedRate(fixed_rate) => {
            let state = Arc::new(FixedRateStreamState::new(
                stream.id,
                Duration::from_nanos(fixed_rate.timestep),
                match fixed_rate.initial_timestamp {
                    InitialTimestamp::Earliest => db.earliest_timestamp.latest(),
                    InitialTimestamp::Latest => db.last_updated.latest(),
                    InitialTimestamp::Manual(timestamp) => timestamp,
                },
                fixed_rate.frequency,
            ));
            debug!(stream.id = ?stream.id, "inserting stream");
            db.with_state_mut(|s| s.streams.insert(stream.id, state.clone()));
            // Spawn a dedicated tick driver for this stream state
            if !state.driver_spawned.swap(true, atomic::Ordering::SeqCst) {
                stellarator::spawn(run_tick_driver(state.clone()));
            }
            stellarator::spawn(async move {
                match handle_fixed_stream(tx, req_id, state, db).await {
                    Ok(_) => {}
                    Err(err) if err.is_stream_closed() => {}
                    Err(err) => {
                        warn!(?err, "error streaming data");
                    }
                }
            })
        }
    }
}

async fn handle_real_time_stream<A: AsyncWrite + 'static>(
    sink: Arc<Mutex<PacketSink<A>>>,
    req_id: RequestId,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut visited_ids = HashSet::new();
    loop {
        // Collect new components along with their metadata and schema
        let new_components: Vec<(Component, Option<ComponentMetadata>, Schema<Vec<u64>>)> = db
            .with_state(|state| {
                let mut new_comps = Vec::new();
                for component in state.components.values() {
                    if visited_ids.contains(&component.component_id) {
                        continue;
                    }
                    let metadata = state
                        .get_component_metadata(component.component_id)
                        .cloned();
                    let schema = component.schema.to_schema();
                    new_comps.push((component.clone(), metadata, schema));
                }
                new_comps
            });

        // Send metadata and schema for each new component, then spawn data handlers
        for (component, metadata, schema) in new_components {
            visited_ids.insert(component.component_id);

            // Send ComponentMetadata so the editor knows about this component
            if let Some(metadata) = metadata {
                let stream = sink.lock().await;
                if let Err(err) = stream
                    .send(metadata.into_len_packet().with_request_id(req_id))
                    .await
                    .0
                {
                    debug!(%err, "error sending component metadata");
                }
            }

            // Send schema for this component
            let schema_msg = DumpSchemaResp {
                schemas: [(component.component_id, schema)].into_iter().collect(),
            };
            {
                let stream = sink.lock().await;
                if let Err(err) = stream
                    .send(schema_msg.into_len_packet().with_request_id(req_id))
                    .await
                    .0
                {
                    debug!(%err, "error sending component schema");
                }
            }

            // Spawn the data handler for this component
            let sink_clone = sink.clone();
            let component_clone = component.clone();
            stellarator::spawn(handle_real_time_component(
                sink_clone,
                component_clone,
                req_id,
            ));
        }

        db.vtable_gen.wait().await;
    }
}

async fn handle_real_time_component<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    component: Component,
    req_id: RequestId,
) -> Result<(), Error> {
    let timestamp_loc = raw_table(0, size_of::<Timestamp>() as u16);
    let prim_type = component.schema.prim_type;
    let vtable = vtable([raw_field(
        (prim_type.padding(8) + size_of::<Timestamp>()) as u16,
        component.schema.size() as u16,
        timestamp(timestamp_loc, component.as_vtable_op()),
    )]);
    let waiter = component.time_series.waiter();
    let vtable_id: PacketId = fastrand::u16(..).to_le_bytes();
    {
        let stream = stream.lock().await;
        stream
            .send(
                VTableMsg {
                    id: vtable_id,
                    vtable,
                }
                .with_request_id(req_id),
            )
            .await
            .0?;
    }

    let mut table = LenPacket::table(vtable_id, 2048 - 16);
    loop {
        let _ = waiter.wait().await;
        let Some((&timestamp, buf)) = component.time_series.latest() else {
            continue;
        };
        table.push_aligned(timestamp);
        table.pad_for_type(prim_type);
        table.extend_from_slice(buf);
        {
            let stream = stream.lock().await;
            if let Err(err) = rent!(stream.send(table.with_request_id(req_id)).await, table) {
                debug!(%err, "error sending table");
                if Error::from(err).is_stream_closed() {
                    return Ok(());
                }
            }
        }
        table.clear();
    }
}

async fn handle_fixed_stream<A: AsyncWrite>(
    stream: Arc<Mutex<PacketSink<A>>>,
    req_id: RequestId,
    state: Arc<FixedRateStreamState>,
    db: Arc<DB>,
) -> Result<(), Error> {
    let mut current_gen = u64::MAX;
    let mut table = LenPacket::table([0; 2], 2048 - 16);
    let mut components = db.with_state(|state| state.components.clone());
    let mut current_timestamp = state.current_timestamp();
    loop {
        if !state.wait_for_playing().await {
            return Ok(());
        }
        let vtable_gen = db.vtable_gen.latest();
        if vtable_gen != current_gen {
            components = db.with_state(|state| state.components.clone());
            let stream = stream.lock().await;
            let id: PacketId = state.stream_id.to_le_bytes()[..2].try_into().unwrap();
            table = LenPacket::table(id, 2048 - 16);
            let vtable = DBVisitor.vtable(&components)?;
            let msg = VTableMsg { id, vtable };
            stream.send(msg.with_request_id(req_id)).await.0?;
            current_gen = vtable_gen;
        }
        table.clear();
        if let Err(err) = DBVisitor.populate_table(&components, &mut table, current_timestamp) {
            warn!(?err, "failed to populate table");
        }
        {
            let stream = stream.lock().await;
            stream
                .send(
                    StreamTimestamp {
                        timestamp: current_timestamp,
                        stream_id: state.stream_id,
                    }
                    .with_request_id(req_id),
                )
                .await
                .0?;
            rent!(stream.send(table.with_request_id(req_id)).await, table)?;
        }
        current_timestamp = state.wait_for_next_tick(current_timestamp).await;
    }
}

struct DBVisitor;

impl DBVisitor {
    fn vtable(&self, components: &HashMap<ComponentId, Component>) -> Result<VTable, Error> {
        let mut fields = vec![];
        let mut offset = 0;
        self.visit(components, |entity| {
            if !entity.time_series.index().is_empty() {
                offset += PrimType::U64.padding(offset);
                let op = timestamp(builder::raw_table(offset as u16, 8), entity.as_vtable_op());
                offset += size_of::<Timestamp>();
                offset += entity.schema.prim_type.padding(offset);
                let len = entity.schema.size();
                let size = len as u16;
                fields.push(raw_field(offset as u16, size, op));
                offset += len;
            }
            Ok(())
        })?;
        Ok(vtable(fields))
    }

    fn populate_table(
        &self,
        components: &HashMap<ComponentId, Component>,
        table: &mut LenPacket,
        timestamp: Timestamp,
    ) -> Result<(), Error> {
        self.visit(components, |entity| {
            let tick = entity.time_series.start_timestamp().max(timestamp);
            let Some((timestamp, buf)) = entity.get_nearest(tick) else {
                return Ok(());
            };
            table.push_aligned(timestamp);
            table.pad_for_type(entity.schema.prim_type);
            table.extend_from_slice(buf);
            Ok(())
        })
    }

    fn visit(
        &self,
        components: &HashMap<ComponentId, Component>,
        mut f: impl for<'a> FnMut(&'a Component) -> Result<(), Error>,
    ) -> Result<(), Error> {
        components
            .iter()
            .try_for_each(|(_, component)| f(component))
    }
}

pub struct PlayingCell {
    is_playing: AtomicBool,
    wait_cell: WaitQueue,
}

impl PlayingCell {
    fn new(is_playing: bool) -> Self {
        Self {
            is_playing: AtomicBool::new(is_playing),
            wait_cell: WaitQueue::new(),
        }
    }

    pub fn set_playing(&self, playing: bool) {
        self.is_playing.store(playing, atomic::Ordering::SeqCst);
        self.wait_cell.wake_all();
    }

    pub fn is_playing(&self) -> bool {
        self.is_playing.load(atomic::Ordering::Relaxed)
    }

    pub async fn wait(&self) -> bool {
        self.wait_cell.wait_for(|| self.is_playing()).await.is_ok()
    }

    async fn wait_for_change(&self) {
        let _ = self.wait_cell.wait().await;
    }
}

impl MetadataExt for DbConfig {}

pub trait AtomicTimestampExt {
    fn update_max(&self, val: Timestamp);
}

impl AtomicTimestampExt for AtomicCell<Timestamp> {
    fn update_max(&self, val: Timestamp) {
        self.value.fetch_max(val.0, atomic::Ordering::AcqRel);
        self.wait_queue.wake_all();
    }
}
