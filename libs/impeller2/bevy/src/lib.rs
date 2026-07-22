use bevy::{
    app::Plugin,
    ecs::{
        hierarchy::ChildOf,
        system::{EntityCommands, SystemId},
    },
    prelude::{Command, In, InRef, IntoSystem, Message, Mut, Single, System, With},
};
use bevy::{ecs::system::SystemParam, prelude::World};
use bevy::{
    ecs::system::SystemState,
    prelude::{Commands, Component, Deref, DerefMut, Entity, Name, Query, ResMut, Resource, debug},
};
use impeller2::types::IntoLenPacket;
use impeller2::types::RequestId;
use impeller2::{
    com_de::Decomponentize,
    registry::HashMapRegistry,
    types::{OwnedPacket, PacketId, Request},
};
use impeller2::{
    schema::Schema,
    types::{ComponentId, ComponentView, LenPacket, Msg, Timestamp},
};
use impeller2_bbq::{AsyncArcQueueRx, RxExt};
use impeller2_wkt::{
    ComponentMetadata, CurrentTimestamp, DbConfig, DumpMetadata, DumpMetadataResp, DumpSchema,
    DumpSchemaResp, EarliestTimestamp, ErrorResponse, GetDbSettings, GetEarliestTimestamp,
    GetTimeSeries, IsRecording, LastUpdated, Stream, StreamBehavior, StreamId, StreamTimestamp,
    SubscribeLastUpdated, VTableMsg, WorldPos,
};
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    convert::Infallible,
    marker::PhantomData,
};
use stellarator_buf::Slice;

pub use impeller2_bbq::PacketGrantR;
pub use impeller2_wkt::ComponentValue;
pub use impeller2_wkt::ElementValueMut;
pub use impeller2_wkt::{ComponentPart, ComponentPath};

#[cfg(feature = "tcp")]
mod tcp;
#[cfg(feature = "tcp")]
pub use tcp::*;

/// Size of the BBQ queue for incoming packets.
/// Increased from 64MB to 256MB to handle large Arrow IPC responses from SQL queries.
pub const QUEUE_LEN: usize = 256 * 1024 * 1024;

#[derive(Resource)]
pub struct PacketRx(AsyncArcQueueRx);

impl From<AsyncArcQueueRx> for PacketRx {
    fn from(rx: AsyncArcQueueRx) -> Self {
        Self(rx)
    }
}

impl PacketRx {
    #[inline]
    pub fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        self.0.try_recv_pkt()
    }
}

#[derive(Resource)]
pub struct PacketTx(pub thingbuf::mpsc::Sender<Option<LenPacket>>);

impl PacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.into_len_packet();
        let _ = self.0.try_send(Some(pkt));
    }
}

#[derive(Resource)]
pub struct MsgPacketRx(AsyncArcQueueRx);

impl From<AsyncArcQueueRx> for MsgPacketRx {
    fn from(rx: AsyncArcQueueRx) -> Self {
        Self(rx)
    }
}

impl MsgPacketRx {
    #[inline]
    pub fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        self.0.try_recv_pkt()
    }
}

#[derive(Resource)]
pub struct MsgPacketTx(pub thingbuf::mpsc::Sender<Option<LenPacket>>);

impl MsgPacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.into_len_packet();
        let _ = self.0.try_send(Some(pkt));
    }
}

#[derive(Debug, Message)]
pub enum DbMessage {
    UpdateConfig,
}

/// Per-component time-series cache. Stores raw component values keyed by
/// timestamp so the Editor can display data at any `CurrentTimestamp` without
/// a DB round-trip.
///
/// Populated by allowlisted backfill + live table stream for **subscribed**
/// components only ([`SeriesFetchPriority`]). Unsubscribed IDs are not stored.
/// Plots project visible windows from this store into GPU LineTrees.
#[derive(Resource, Default)]
pub struct TelemetryCache {
    components: HashMap<ComponentId, BTreeMap<Timestamp, ComponentValue>>,
    /// Merged half-open coverage intervals `[start, end)` per component (micros).
    coverage: HashMap<ComponentId, Vec<(i64, i64)>>,
    generation: u64,
}

/// Alias used by the telemetry-cache / SeriesStore work.
pub type SeriesStore = TelemetryCache;

/// Progressive backfill progress — playback never waits on `complete`.
#[derive(Resource, Default, Debug, Clone)]
pub struct SeriesStoreLoadState {
    pub components_started: usize,
    pub components_complete: usize,
    pub samples_loaded: u64,
    pub complete: bool,
}

impl TelemetryCache {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn total_sample_count(&self) -> u64 {
        self.components.values().map(|s| s.len() as u64).sum()
    }

    pub fn insert(&mut self, component_id: ComponentId, ts: Timestamp, value: ComponentValue) {
        let series = self.components.entry(component_id).or_default();
        // Keep the first value seen for a timestamp. In mixed backfill+live
        // streaming, the same timestamp can be replayed; replacing it can make
        // rendered poses jump between two states for the same tick.
        if let std::collections::btree_map::Entry::Vacant(entry) = series.entry(ts) {
            entry.insert(value);
            self.generation = self.generation.wrapping_add(1);
        }
    }

    pub fn get_at_or_before(
        &self,
        component_id: &ComponentId,
        ts: Timestamp,
    ) -> Option<&ComponentValue> {
        let series = self.components.get(component_id)?;
        series.range(..=ts).next_back().map(|(_, v)| v)
    }

    /// True when any samples have been cached for this component.
    pub fn has_series(&self, component_id: &ComponentId) -> bool {
        self.components.contains_key(component_id)
    }

    pub fn component_ids(&self) -> impl Iterator<Item = &ComponentId> {
        self.components.keys()
    }

    pub fn series(
        &self,
        component_id: &ComponentId,
    ) -> Option<&BTreeMap<Timestamp, ComponentValue>> {
        self.components.get(component_id)
    }

    /// Drop all samples and coverage for a component (unsubscribe / reclaim RAM).
    pub fn remove_series(&mut self, component_id: &ComponentId) {
        let removed_components = self.components.remove(component_id).is_some();
        let removed_coverage = self.coverage.remove(component_id).is_some();
        if removed_components || removed_coverage {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// Record that historical fetch has covered `[start, end)` (micros half-open).
    pub fn mark_covered(&mut self, component_id: ComponentId, start: Timestamp, end: Timestamp) {
        if end.0 <= start.0 {
            return;
        }
        // Refuse bogus "cover everything" marks.
        if end.0 == i64::MAX {
            bevy::log::warn!(
                target: "elodin_series_store",
                component = %component_id,
                start = start.0,
                "refusing mark_covered to i64::MAX"
            );
            return;
        }
        let intervals = self.coverage.entry(component_id).or_default();
        intervals.push((start.0, end.0));
        merge_intervals(intervals);
    }

    /// True when `range` is fully contained in recorded coverage for this component.
    pub fn is_covered(
        &self,
        component_id: &ComponentId,
        range: &std::ops::Range<Timestamp>,
    ) -> bool {
        if range.end.0 <= range.start.0 {
            return true;
        }
        let Some(intervals) = self.coverage.get(component_id) else {
            return false;
        };
        let mut cursor = range.start.0;
        let end = range.end.0;
        for &(a, b) in intervals {
            if b <= cursor {
                continue;
            }
            if a > cursor {
                return false;
            }
            cursor = cursor.max(b);
            if cursor >= end {
                return true;
            }
        }
        cursor >= end
    }

    /// Sample count in `[range.start, range.end)`.
    pub fn sample_count_in_range(
        &self,
        component_id: &ComponentId,
        range: &std::ops::Range<Timestamp>,
    ) -> usize {
        self.components
            .get(component_id)
            .map(|s| s.range(range.start..range.end).count())
            .unwrap_or(0)
    }

    /// First/last sample timestamps in range, if any.
    pub fn sample_span_in_range(
        &self,
        component_id: &ComponentId,
        range: &std::ops::Range<Timestamp>,
    ) -> Option<(Timestamp, Timestamp)> {
        let series = self.components.get(component_id)?;
        let mut iter = series.range(range.start..range.end);
        let (first, _) = iter.next()?;
        let (last, _) = series.range(range.start..range.end).next_back()?;
        Some((*first, *last))
    }
}

fn merge_intervals(intervals: &mut Vec<(i64, i64)>) {
    if intervals.is_empty() {
        return;
    }
    intervals.sort_by_key(|&(a, _)| a);
    let mut out = Vec::with_capacity(intervals.len());
    let mut cur = intervals[0];
    for &(a, b) in intervals.iter().skip(1) {
        if a <= cur.1 {
            cur.1 = cur.1.max(b);
        } else {
            out.push(cur);
            cur = (a, b);
        }
    }
    out.push(cur);
    *intervals = out;
}

/// Allowlist of component IDs that may be stored in SeriesStore / TelemetryCache
/// (enabled plots, Line3d, viewport adapters). Empty ⇒ no backfill / no live cache inserts.
#[derive(Resource, Default)]
pub struct SeriesFetchPriority {
    pub high: std::collections::HashSet<ComponentId>,
}

/// Single vtable pass for incoming tables: append allowlisted samples to the
/// pending cache buffer and run [`WorldSink`] entity/path setup in the same traversal.
struct TableCacheAndWorldSink<'a, 'w, 's> {
    pending_cache: &'a mut Vec<(ComponentId, Timestamp, ComponentValue)>,
    /// When `None`, treat as empty allowlist (cache nothing). When `Some`, only
    /// push samples whose id is in the set.
    allowlist: Option<&'a std::collections::HashSet<ComponentId>>,
    world: &'a mut WorldSink<'w, 's>,
}

impl Decomponentize for TableCacheAndWorldSink<'_, '_, '_> {
    type Error = Infallible;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        view: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Infallible> {
        if let Some(ts) = timestamp {
            let allowed = self
                .allowlist
                .is_some_and(|set| set.contains(&component_id));
            if allowed {
                let value = ComponentValue::from_view(view);
                self.pending_cache.push((component_id, ts, value));
            }
        }
        self.world.apply_value(component_id, view, timestamp)?;
        Ok(())
    }
}

/// Bevy system that reads the TelemetryCache at `CurrentTimestamp` and
/// overwrites entity `ComponentValue` components and adapted components
/// (like `WorldPos`) with the cached data. This allows the viewport to
/// display data at any timestamp the user scrubs to, without waiting for
/// the DB stream to deliver it.
#[allow(clippy::too_many_arguments)]
pub fn apply_cached_data(
    current_ts: bevy::prelude::Res<CurrentTimestamp>,
    cache: bevy::prelude::Res<TelemetryCache>,
    mut entity_map: ResMut<EntityMap>,
    mut query: Query<&mut ComponentValue>,
    adapters: bevy::prelude::Res<ComponentAdapters>,
    path_reg: bevy::prelude::Res<ComponentPathRegistry>,
    mut commands: Commands,
    mut last_applied: bevy::prelude::Local<(Timestamp, u64)>,
) {
    let ts = current_ts.0;
    let cache_gen = cache.generation;
    if ts == last_applied.0 && cache_gen == last_applied.1 {
        return;
    }
    *last_applied = (ts, cache_gen);
    for component_id in cache.component_ids().copied().collect::<Vec<_>>() {
        let Some(value) = cache.get_at_or_before(&component_id, ts) else {
            continue;
        };
        let Some(&entity) = entity_map.get(&component_id) else {
            continue;
        };
        if let Ok(mut cv) = query.get_mut(entity) {
            *cv = value.clone();
        } else {
            commands.entity(entity).insert(value.clone());
        }
        // Adapters are keyed by type leaf (`world_pos`), while live/DB series use
        // pair IDs (`ball_1.world_pos`). Resolve via path tail.
        let adapter_key = path_reg
            .0
            .get(&component_id)
            .map(|path| path.tail().id)
            .unwrap_or(component_id);
        if let Some(adapter) = adapters.get(&adapter_key) {
            let view = value.as_view();
            adapter.insert(&mut commands, &mut entity_map, component_id, view);
        }
    }
}

/// Tracks which components have had their backfill requests sent.
#[derive(Resource, Default)]
pub struct BackfillState {
    requested: std::collections::HashSet<ComponentId>,
    complete: std::collections::HashSet<ComponentId>,
}

impl BackfillState {
    /// Clear backfill bookkeeping for a component so it can be re-fetched if
    /// re-subscribed.
    pub fn clear_component(&mut self, component_id: ComponentId) {
        self.requested.remove(&component_id);
        self.complete.remove(&component_id);
    }
}

const BACKFILL_CHUNK_SIZE: usize = 4096;
/// Aggressive concurrent historical pages — does not block playback.
const BACKFILL_MAX_CONCURRENT: usize = 96;

/// Component IDs that should receive begin→end SeriesStore backfill: the
/// allowlist intersected with components that have a known schema.
pub fn series_store_backfill_candidates(
    allowlist: &std::collections::HashSet<ComponentId>,
    schema_reg: &ComponentSchemaRegistry,
) -> Vec<ComponentId> {
    allowlist
        .iter()
        .copied()
        .filter(|id| schema_reg.0.contains_key(id))
        .collect()
}

/// Bevy system that sends paginated `GetTimeSeries` requests for each
/// **allowlisted** component to populate the `TelemetryCache` with historical data.
/// Starts from the beginning of each series so early timeline scrubbing works
/// as soon as the first pages land. Playback never waits for
/// `SeriesStoreLoadState.complete`. Empty allowlist ⇒ no backfill.
pub fn backfill_cache(
    schema_reg: bevy::prelude::Res<ComponentSchemaRegistry>,
    msg_request_handlers: bevy::prelude::Res<MsgRequestIdHandlers>,
    mut backfill: ResMut<BackfillState>,
    priority: bevy::prelude::Res<SeriesFetchPriority>,
    mut load_state: ResMut<SeriesStoreLoadState>,
    mut commands: Commands,
) {
    if msg_request_handlers.len() >= BACKFILL_MAX_CONCURRENT {
        return;
    }
    let mut dispatched = 0usize;
    let available = BACKFILL_MAX_CONCURRENT - msg_request_handlers.len();

    let ordered = series_store_backfill_candidates(&priority.high, &schema_reg);
    let eligible = ordered.len();
    load_state.components_started = backfill.requested.len().max(load_state.components_started);
    load_state.components_complete = backfill.complete.len();
    load_state.complete =
        eligible > 0 && backfill.complete.len() >= eligible && msg_request_handlers.is_empty();

    for component_id in ordered {
        if dispatched >= available {
            break;
        }
        if backfill.requested.contains(&component_id) {
            continue;
        }
        backfill.requested.insert(component_id);
        load_state.components_started = backfill.requested.len();
        // Always stream from the series start so jump-to-beginning / early
        // windows become usable as soon as the first pages arrive.
        send_backfill_page(&mut commands, component_id, Timestamp(i64::MIN));
        dispatched += 1;
    }
}

fn send_backfill_page(commands: &mut Commands, component_id: ComponentId, start: Timestamp) {
    let page_start = start;
    let msg = GetTimeSeries {
        id: PacketId::default(),
        range: start..Timestamp(i64::MAX),
        component_id,
        limit: Some(BACKFILL_CHUNK_SIZE),
    };

    commands.send_msg_req_reply_raw::<_, GetTimeSeries, _>(
        msg,
        move |pkt: bevy::prelude::InRef<OwnedPacket<PacketGrantR>>,
              mut cache: ResMut<TelemetryCache>,
              schema_reg: bevy::prelude::Res<ComponentSchemaRegistry>,
              priority: bevy::prelude::Res<SeriesFetchPriority>,
              mut backfill: ResMut<BackfillState>,
              mut load_state: ResMut<SeriesStoreLoadState>,
              mut cmds: Commands| {
            // Component may have left the allowlist while this page was in flight.
            if !priority.high.contains(&component_id) {
                return true;
            }
            let OwnedPacket::TimeSeries(ts) = &*pkt else {
                return true;
            };
            let Ok(timestamps) = ts.timestamps() else {
                return true;
            };
            let Ok(data) = ts.data() else {
                return true;
            };

            let Some(schema) = schema_reg.0.get(&component_id) else {
                return true;
            };

            let elem_size = schema.size();
            let count = timestamps.len();
            let mut last_ts = start;

            for (i, &timestamp) in timestamps.iter().enumerate() {
                let offset = i * elem_size;
                if offset + elem_size > data.len() {
                    break;
                }
                let bytes = &data[offset..offset + elem_size];
                if let Ok(view) = impeller2::types::ComponentView::try_from_bytes_shape(
                    bytes,
                    schema.shape(),
                    schema.prim_type(),
                ) {
                    let value = ComponentValue::from_view(view);
                    cache.insert(component_id, timestamp, value);
                    last_ts = timestamp;
                }
            }

            if count > 0 {
                let cover_start = timestamps.first().copied().unwrap_or(page_start);
                let cover_end = Timestamp(last_ts.0.saturating_add(1));
                cache.mark_covered(component_id, cover_start, cover_end);
                load_state.samples_loaded = load_state.samples_loaded.saturating_add(count as u64);
            }

            // Re-check after inserts: allowlist may have changed mid-handler.
            if !priority.high.contains(&component_id) {
                cache.remove_series(&component_id);
                return true;
            }

            if count >= BACKFILL_CHUNK_SIZE {
                // Continue from the beginning forward — usable incrementally.
                send_backfill_page(&mut cmds, component_id, Timestamp(last_ts.0 + 1));
            } else {
                backfill.complete.insert(component_id);
                load_state.components_complete = backfill.complete.len();
            }
            true
        },
    );
}

fn sink_inner(
    world: &mut World,
    packet_rx: &mut PacketRx,
    vtable_registry: &mut HashMapRegistry,
    packet_handlers: &mut PacketHandlers,
    world_sink_state: &mut SystemState<WorldSink>,
) -> Result<(), impeller2::error::Error> {
    let mut count = 0;
    let sink_deadline = std::time::Instant::now() + std::time::Duration::from_millis(8);
    let mut pending_cache_entries: Vec<(ComponentId, Timestamp, ComponentValue)> = Vec::new();
    let allowlist = world
        .get_resource::<SeriesFetchPriority>()
        .map(|p| p.high.clone());
    loop {
        if !pending_cache_entries.is_empty()
            && let Some(mut cache) = world.get_resource_mut::<TelemetryCache>()
        {
            for (cid, ts, val) in pending_cache_entries.drain(..) {
                cache.insert(cid, ts, val);
            }
        }
        if count > 2048 || (count >= 16 && std::time::Instant::now() > sink_deadline) {
            break;
        }
        let Some(pkt) = packet_rx.try_recv_pkt() else {
            break;
        };
        count += 1;
        {
            let pkt_id = match &pkt {
                OwnedPacket::Msg(m) => m.id,
                OwnedPacket::Table(table) => table.id,
                OwnedPacket::TimeSeries(time_series) => time_series.id,
            };
            let handler = world
                .get_resource_mut::<PacketIdHandlers>()
                .and_then(|mut handlers| handlers.remove(&pkt_id));
            if let Some(handler) = handler {
                if let Err(err) = world.run_system_with(handler, &pkt) {
                    bevy::log::error!(?err, "packet id handler error");
                }
                if let Err(err) = world.unregister_system(handler) {
                    bevy::log::error!(?err, "unregister packet handler error");
                }
            }
        }

        {
            let req_id = match &pkt {
                OwnedPacket::Msg(m) => m.req_id,
                OwnedPacket::Table(table) => table.req_id,
                OwnedPacket::TimeSeries(time_series) => time_series.req_id,
            };

            let handler = world
                .get_resource_mut::<RequestIdHandlers>()
                .and_then(|mut handlers| handlers.remove(&req_id));
            if let Some(handler) = handler {
                match world.run_system_with(handler, &pkt) {
                    Ok(completed) => {
                        if !completed {
                            world
                                .get_resource_mut::<RequestIdHandlers>()
                                .and_then(|mut handlers| handlers.insert(req_id, handler));
                        } else if let Err(err) = world.unregister_system(handler) {
                            bevy::log::error!(?err, "unregister request id handler error");
                        }
                    }
                    Err(err) => {
                        bevy::log::error!(?err, "packet id handler error");
                        if let Err(unreg) = world.unregister_system(handler) {
                            bevy::log::error!(?unreg, "unregister request id handler error");
                        }
                    }
                }
            }
        }

        for handler in packet_handlers.0.iter() {
            if let Err(err) = world.run_system_with(*handler, (&pkt, vtable_registry)) {
                bevy::log::error!(?err, "packet handler error");
            }
        }
        let mut world_sink = world_sink_state
            .get_mut(world)
            .expect("WorldSink params invalid");
        match &pkt {
            OwnedPacket::Msg(m) if m.id == VTableMsg::ID => {
                let vtable = m.parse::<VTableMsg>()?;
                vtable_registry.map.insert(vtable.id, vtable.vtable);
            }
            OwnedPacket::Msg(m) if m.id == ComponentMetadata::ID => {
                let metadata = m.parse::<ComponentMetadata>()?;
                // Create the full path hierarchy so the component appears in UI
                // under DbComponentsRoot (not just an orphaned leaf entity).
                let path = ComponentPath::from_name(&metadata.name);
                ensure_component_path_hierarchy(
                    &mut world_sink.entity_map,
                    &mut world_sink.metadata_reg,
                    &mut world_sink.commands,
                    &path,
                    *world_sink.db_components_root,
                );
                world_sink.path_reg.0.insert(metadata.component_id, path);
                world_sink
                    .metadata_reg
                    .insert(metadata.component_id, metadata);
            }
            OwnedPacket::Msg(m) if m.id == DumpMetadataResp::ID => {
                let metadata = m.parse::<DumpMetadataResp>()?;
                for metadata in metadata.component_metadata.into_iter() {
                    let path = ComponentPath::from_name(&metadata.name);
                    ensure_component_path_hierarchy(
                        &mut world_sink.entity_map,
                        &mut world_sink.metadata_reg,
                        &mut world_sink.commands,
                        &path,
                        *world_sink.db_components_root,
                    );
                    world_sink.path_reg.0.insert(metadata.component_id, path);
                    world_sink
                        .metadata_reg
                        .insert(metadata.component_id, metadata);
                }
                *world_sink.db_config = metadata.db_config.clone();
                world_sink.commands.write_message(DbMessage::UpdateConfig);
            }
            OwnedPacket::Msg(m) if m.id == LastUpdated::ID => {
                let m = m.parse::<LastUpdated>()?;
                // Keep LastUpdated monotonic on the client. In mixed/reconnect
                // conditions, out-of-order packets can otherwise move the
                // playback clock backward and cause visible pose flicker.
                if m.0 > world_sink.max_tick.0 {
                    *world_sink.max_tick = m;
                }
            }
            OwnedPacket::Msg(m) if m.id == DbConfig::ID => {
                let config = m.parse::<DbConfig>()?;
                world_sink.recording.0 = config.recording;
                *world_sink.db_config = config;
                world_sink.commands.write_message(DbMessage::UpdateConfig);
            }
            OwnedPacket::Msg(m) if m.id == DumpSchemaResp::ID => {
                let dump_schema = m.parse::<DumpSchemaResp>()?;
                world_sink.schema_reg.0.extend(dump_schema.schemas);
            }
            OwnedPacket::Table(table) => {
                let _span = tracing::info_span!("impeller2_table_sink").entered();
                let mut combined = TableCacheAndWorldSink {
                    pending_cache: &mut pending_cache_entries,
                    allowlist: allowlist.as_ref(),
                    world: &mut world_sink,
                };
                let _ = table.sink(vtable_registry, &mut combined)?;
            }
            OwnedPacket::Msg(m) if m.id == EarliestTimestamp::ID => {
                let new_earliest = m.parse::<EarliestTimestamp>()?;
                apply_earliest_timestamp(
                    &mut world_sink.earliest_timestamp,
                    &mut world_sink.current_timestamp,
                    new_earliest,
                );
            }
            OwnedPacket::Msg(m) if m.id == StreamTimestamp::ID => {
                let _ = m;
            }
            OwnedPacket::Msg(_) => {}
            OwnedPacket::TimeSeries(_) => {}
        }
        world_sink_state.apply(world);
    }
    if !pending_cache_entries.is_empty()
        && let Some(mut cache) = world.get_resource_mut::<TelemetryCache>()
    {
        for (cid, ts, val) in pending_cache_entries.drain(..) {
            cache.insert(cid, ts, val);
        }
    }
    Ok(())
}

/// Apply a server `EarliestTimestamp` update.
///
/// On the first bound after reset (`earliest == MAX`), adopt the new earliest.
/// Snap the playhead to that earliest only when it is still uninitialized
/// (`CurrentTimestamp == EPOCH`) — soft reconnect preserves a non-EPOCH playhead.
pub(crate) fn apply_earliest_timestamp(
    earliest: &mut EarliestTimestamp,
    current: &mut CurrentTimestamp,
    new_earliest: EarliestTimestamp,
) {
    let is_first = earliest.0 == Timestamp(i64::MAX);
    if is_first {
        *earliest = new_earliest;
    } else if new_earliest.0 < earliest.0 {
        // Keep EarliestTimestamp monotonic (min) to avoid narrowing
        // the clamp window from stale/out-of-order updates.
        *earliest = new_earliest;
    }
    if is_first && current.0 == Timestamp::EPOCH {
        current.0 = new_earliest.0;
    }
}

pub fn sink(world: &mut World, world_sink_state: &mut SystemState<WorldSink>) {
    world.resource_scope(|world, mut packet_rx: Mut<PacketRx>| {
        world.resource_scope(|world, mut vtable_reg: Mut<HashMapRegistry>| {
            world.resource_scope(|world, mut packet_handlers: Mut<PacketHandlers>| {
                if let Err(err) = sink_inner(
                    world,
                    &mut packet_rx,
                    &mut vtable_reg,
                    &mut packet_handlers,
                    world_sink_state,
                ) {
                    bevy::log::error!(?err, "sink failed")
                }
            })
        })
    })
}

#[derive(SystemParam)]
pub struct MsgSinkState<'w> {
    msg_rx: ResMut<'w, MsgPacketRx>,
}

pub fn msg_sink(world: &mut World, msg_sink_state: &mut SystemState<MsgSinkState>) {
    let sink_deadline = std::time::Instant::now() + std::time::Duration::from_millis(8);
    let mut count = 0;
    loop {
        if count > 2048 || (count >= 16 && std::time::Instant::now() > sink_deadline) {
            return;
        }
        let pkt = {
            let MsgSinkState { mut msg_rx } = msg_sink_state
                .get_mut(world)
                .expect("MsgSinkState params invalid");
            msg_rx.try_recv_pkt()
        };
        let Some(pkt) = pkt else {
            break;
        };
        count += 1;
        let pkt_id = match &pkt {
            OwnedPacket::Msg(m) => m.id,
            OwnedPacket::Table(table) => table.id,
            OwnedPacket::TimeSeries(time_series) => time_series.id,
        };
        let handler = world
            .get_resource_mut::<PacketIdHandlers>()
            .and_then(|mut handlers| handlers.remove(&pkt_id));
        if let Some(handler) = handler {
            if let Err(err) = world.run_system_with(handler, &pkt) {
                bevy::log::error!(?err, "msg packet id handler error");
            }
            if let Err(err) = world.unregister_system(handler) {
                bevy::log::error!(?err, "unregister packet handler error");
            }
        }
        let req_id = match &pkt {
            OwnedPacket::Msg(m) => m.req_id,
            OwnedPacket::Table(table) => table.req_id,
            OwnedPacket::TimeSeries(time_series) => time_series.req_id,
        };
        let handler = world
            .get_resource_mut::<MsgRequestIdHandlers>()
            .and_then(|mut handlers| handlers.remove(&req_id));
        if let Some(handler) = handler {
            match world.run_system_with(handler, &pkt) {
                Ok(completed) => {
                    if !completed {
                        world
                            .get_resource_mut::<MsgRequestIdHandlers>()
                            .and_then(|mut handlers| handlers.insert(req_id, handler));
                    } else if let Err(err) = world.unregister_system(handler) {
                        bevy::log::error!(?err, "unregister msg request id handler error");
                    }
                }
                Err(err) => {
                    bevy::log::error!(?err, "msg request id handler error");
                    if let Err(unreg) = world.unregister_system(handler) {
                        bevy::log::error!(?unreg, "unregister msg request id handler error");
                    }
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketIdHandlers(
    pub HashMap<PacketId, SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, ()>>,
);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketHandlers(pub Vec<SystemId<PacketHandlerInput<'static>, ()>>);

pub struct PacketHandlerInput<'a> {
    pub packet: &'a OwnedPacket<PacketGrantR>,
    pub registry: &'a HashMapRegistry,
}

impl bevy::prelude::SystemInput for PacketHandlerInput<'_> {
    type Param<'i> = PacketHandlerInput<'i>;

    type Inner<'i> = (&'i OwnedPacket<PacketGrantR>, &'i HashMapRegistry);

    fn wrap((packet, registry): Self::Inner<'_>) -> Self::Param<'_> {
        PacketHandlerInput { packet, registry }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RequestIdHandlers(
    pub HashMap<RequestId, SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, bool>>,
);

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct MsgRequestIdHandlers(
    pub HashMap<RequestId, SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, bool>>,
);

type MsgReplySystemId = SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, bool>;

#[derive(Resource, Default, Deref, DerefMut)]
pub struct MsgRequestQueue(pub VecDeque<(LenPacket, MsgReplySystemId)>);

#[derive(Resource)]
pub struct MsgRequestIdAlloc(RequestId);

impl Default for MsgRequestIdAlloc {
    fn default() -> Self {
        Self(1)
    }
}

impl MsgRequestIdAlloc {
    pub fn alloc_next_id_avoiding(
        &mut self,
        occupied: &std::collections::HashSet<RequestId>,
    ) -> Option<RequestId> {
        for _ in 0..255 {
            self.0 = self.0.wrapping_add(1);
            if self.0 == 0 {
                self.0 = 1;
            }
            if !occupied.contains(&self.0) {
                return Some(self.0);
            }
        }
        None
    }
}

#[derive(Component, Default, DerefMut, Deref)]
pub struct ComponentValueMap(pub BTreeMap<ComponentId, ComponentValue>);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct EntityMap(pub HashMap<ComponentId, Entity>);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentMetadataRegistry(pub HashMap<ComponentId, ComponentMetadata>);

impl ComponentMetadataRegistry {
    #[inline(always)]
    pub fn get_metadata(&self, component_id: &ComponentId) -> Option<&ComponentMetadata> {
        self.get(component_id)
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentSchemaRegistry(pub HashMap<ComponentId, Schema<Vec<u64>>>);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentPathRegistry(pub HashMap<ComponentId, ComponentPath>);

#[derive(Component)]
pub struct DbComponentsRoot;

fn ensure_db_components_root(mut commands: Commands) {
    commands.spawn((DbComponentsRoot, Name::new("db components")));
}

#[derive(SystemParam)]
pub struct WorldSink<'w, 's> {
    commands: Commands<'w, 's>,
    entity_map: ResMut<'w, EntityMap>,
    metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    max_tick: ResMut<'w, LastUpdated>,
    earliest_timestamp: ResMut<'w, EarliestTimestamp>,
    recording: ResMut<'w, IsRecording>,
    current_timestamp: ResMut<'w, CurrentTimestamp>,
    schema_reg: ResMut<'w, ComponentSchemaRegistry>,
    path_reg: ResMut<'w, ComponentPathRegistry>,
    db_config: ResMut<'w, DbConfig>,
    db_components_root: Single<'w, 's, Entity, With<DbComponentsRoot>>,
}

#[allow(clippy::needless_lifetimes)] // removing these lifetimes causes an internal compiler error, so here we are
fn try_insert_entity<'a, 'w, 's>(
    entity_map: &mut EntityMap,
    metadata_reg: &mut ComponentMetadataRegistry,
    commands: &'a mut Commands<'w, 's>,
    component_path: &ComponentPart,
) -> Option<(EntityCommands<'a>, bool)> {
    let component_id = component_path.id;
    if let Some(entity) = entity_map.get(&component_id) {
        let Ok(e) = commands.get_entity(*entity) else {
            return None;
        };
        Some((e, false))
    } else {
        let metadata = metadata_reg
            .entry(component_id)
            .or_insert_with(|| ComponentMetadata {
                component_id,
                name: component_path.name.to_string(),
                metadata: Default::default(),
            })
            .clone();
        let e = commands.spawn((
            component_id,
            ComponentValueMap::default(),
            metadata.clone(),
            Name::new(metadata.name.clone()),
        ));
        entity_map.insert(component_id, e.id());
        Some((e, true))
    }
}

/// Creates every segment of `path` and parents newly created ones under
/// `db_components_root` (or the previous segment).
///
/// Hierarchy is wired only for `newly_created` entities so later value packets
/// do not re-insert `ChildOf` and undo editor reparenting (e.g. GridCell under
/// BigSpaceRoot). Call this from metadata handlers so leaves are not orphaned:
/// metadata often creates the leaf before `apply_value` runs, which would make
/// `newly_created == false` in a value-only path loop.
fn ensure_component_path_hierarchy(
    entity_map: &mut EntityMap,
    metadata_reg: &mut ComponentMetadataRegistry,
    commands: &mut Commands,
    path: &ComponentPath,
    db_components_root: Entity,
) {
    let mut last_entity: Option<Entity> = None;
    for part in path.path.iter() {
        let Some((mut e, newly_created)) =
            try_insert_entity(entity_map, metadata_reg, commands, part)
        else {
            continue;
        };
        if newly_created {
            if let Some(parent) = last_entity {
                e.insert(ChildOf(parent));
            } else {
                e.insert(ChildOf(db_components_root));
            }
        }
        last_entity = Some(e.id());
    }
}

impl Decomponentize for WorldSink<'_, '_> {
    type Error = core::convert::Infallible;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        _view: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Infallible> {
        let Some(path) = self.path_reg.get(&component_id).cloned() else {
            return Ok(());
        };

        ensure_component_path_hierarchy(
            &mut self.entity_map,
            &mut self.metadata_reg,
            &mut self.commands,
            &path,
            *self.db_components_root,
        );

        // ComponentValue and adapter writes (WorldPos, etc.) are handled
        // exclusively by apply_cached_data from the TelemetryCache.
        Ok(())
    }
}

#[derive(Resource, Debug, Deref, DerefMut)]
pub struct CurrentStreamId(pub StreamId);

impl CurrentStreamId {
    pub fn rand() -> CurrentStreamId {
        CurrentStreamId(fastrand::u64(..))
    }

    pub fn packet_id(&self) -> PacketId {
        self.0.to_le_bytes()[..2].try_into().unwrap()
    }
}

pub trait ComponentAdapter: Send + Sync {
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        component_id: ComponentId,
        value: ComponentView<'_>,
    );
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct ComponentAdapters(HashMap<ComponentId, Box<dyn ComponentAdapter>>);

pub struct StaticComponentAdapter<C>(PhantomData<C>);
impl<C> Default for StaticComponentAdapter<C> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<C> ComponentAdapter for StaticComponentAdapter<C>
where
    C: impeller2::component::Component + Decomponentize + Default + Component,
{
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        component_id: ComponentId,
        value: ComponentView<'_>,
    ) {
        let mut val = C::default();
        let _ = val.apply_value(C::COMPONENT_ID, value, None);
        let mut e = if let Some(entity) = entity_map.0.get(&component_id) {
            let Ok(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            return;
        };
        e.insert(val);
    }
}

/// Adapter for DB components that mirror into a Bevy [`Resource`] instead of
/// an entity component (e.g. `SimulationTimeStep`). Since Bevy 0.19 resources
/// are unique components on dedicated entities, inserting a resource type onto
/// a regular entity would despawn the resource; write the resource directly.
pub struct ResourceComponentAdapter<R>(PhantomData<R>);
impl<R> Default for ResourceComponentAdapter<R> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<R> ComponentAdapter for ResourceComponentAdapter<R>
where
    R: impeller2::component::Component + Decomponentize + Default + Resource,
{
    fn insert(
        &self,
        commands: &mut Commands,
        _entity_map: &mut EntityMap,
        _component_id: ComponentId,
        value: ComponentView<'_>,
    ) {
        let mut val = R::default();
        let _ = val.apply_value(R::COMPONENT_ID, value, None);
        commands.insert_resource(val);
    }
}

pub trait AppExt {
    fn add_impeller_component<C>(&mut self) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component;

    fn add_impeller_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter>,
    ) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component;
}

impl AppExt for bevy::app::App {
    fn add_impeller_component<C>(&mut self) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component,
    {
        self.add_impeller_component_with_adapter::<C>(Box::<StaticComponentAdapter<C>>::default())
    }

    fn add_impeller_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter>,
    ) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component,
    {
        let mut map = self
            .world_mut()
            .get_resource_or_insert_with(ComponentAdapters::default);

        map.0.insert(C::COMPONENT_ID, adapter);
        self
    }
}

#[derive(Resource)]
pub struct RequestIdAlloc(RequestId);

impl Default for RequestIdAlloc {
    fn default() -> Self {
        // Start at 1 to avoid request ID 0, which is reserved for streaming
        RequestIdAlloc(1)
    }
}

impl RequestIdAlloc {
    pub fn alloc_next_id(&mut self) -> RequestId {
        self.0 = self.0.wrapping_add(1);
        if self.0 == 0 {
            self.0 = 1;
        }
        self.0
    }

    pub fn alloc_next_id_avoiding(
        &mut self,
        occupied: &std::collections::HashSet<RequestId>,
    ) -> Option<RequestId> {
        for _ in 0..255 {
            self.0 = self.0.wrapping_add(1);
            if self.0 == 0 {
                self.0 = 1;
            }
            if !occupied.contains(&self.0) {
                return Some(self.0);
            }
        }
        None
    }
}

pub struct DefaultAdaptersPlugin;

impl Plugin for DefaultAdaptersPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_impeller_component::<WorldPos>();
        // `SimulationTimeStep` mirrors into the Bevy resource of the same
        // type rather than an entity component (resources are unique
        // components in Bevy 0.19, so inserting one on a regular entity
        // would despawn the resource entity).
        app.add_impeller_component_with_adapter::<impeller2_wkt::SimulationTimeStep>(Box::<
            ResourceComponentAdapter<impeller2_wkt::SimulationTimeStep>,
        >::default(
        ));
    }
}

pub struct Impeller2Plugin;

impl Plugin for Impeller2Plugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_message::<DbMessage>()
            .add_plugins(DefaultAdaptersPlugin)
            .add_systems(bevy::prelude::Startup, ensure_db_components_root)
            .add_systems(bevy::prelude::Update, flush_msg_request_queue)
            .insert_resource(impeller2_wkt::SimulationTimeStep(0.001))
            .insert_resource(impeller2_wkt::CurrentTimestamp(Timestamp::EPOCH))
            .insert_resource(impeller2_wkt::LastUpdated(Timestamp(i64::MIN)))
            .insert_resource(impeller2_wkt::EarliestTimestamp(Timestamp(i64::MAX)))
            .init_resource::<IsRecording>()
            .init_resource::<EntityMap>()
            .init_resource::<ComponentMetadataRegistry>()
            .init_resource::<ComponentSchemaRegistry>()
            .init_resource::<ComponentPathRegistry>()
            .init_resource::<HashMapRegistry>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<PacketHandlers>()
            .init_resource::<RequestIdHandlers>()
            .init_resource::<RequestIdAlloc>()
            .init_resource::<MsgRequestIdHandlers>()
            .init_resource::<MsgRequestIdAlloc>()
            .init_resource::<MsgRequestQueue>()
            .init_resource::<DbConfig>()
            .init_resource::<TelemetryCache>()
            .init_resource::<BackfillState>()
            .init_resource::<SeriesStoreLoadState>()
            .init_resource::<SeriesFetchPriority>();
    }
}

pub struct ReqHandlerCommand<S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = ()>> {
    request: LenPacket,
    packet_id: PacketId,
    system: S,
}

impl<S> Command for ReqHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = ()>,
{
    type Out = ();

    fn apply(self, world: &mut World) {
        let system_id = world.register_system(self.system);
        debug!("registered req handler {system_id:?}");
        let mut handlers = world
            .get_resource_mut::<PacketIdHandlers>()
            .expect("missing packet handlers");
        handlers.insert(self.packet_id, system_id);
        let tx = world
            .get_resource_mut::<PacketTx>()
            .expect("missing packet handlers");
        if let Err(err) = tx.0.try_send(Some(self.request)) {
            let mut handlers = world
                .get_resource_mut::<PacketIdHandlers>()
                .expect("missing packet handlers");
            handlers.remove(&self.packet_id);
            if let Err(unreg) = world.unregister_system(system_id) {
                bevy::log::error!(?unreg, "failed to unregister system after send failure");
            }
            bevy::log::warn!(?err, "failed to send msg");
        }
    }
}

pub struct ReplyHandlerCommand<
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
> {
    request: LenPacket,
    system: S,
}

impl<S> Command for ReplyHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
{
    type Out = ();

    fn apply(self, world: &mut World) {
        let system_id = world.register_system(self.system);

        debug!("registered reply handler {system_id:?}");
        let req_id = {
            let handlers = world.resource::<RequestIdHandlers>();
            let occupied: std::collections::HashSet<RequestId> = handlers.keys().copied().collect();
            let mut alloc = world.resource_mut::<RequestIdAlloc>();
            alloc.alloc_next_id_avoiding(&occupied)
        };

        let Some(req_id) = req_id else {
            bevy::log::warn!(
                "RequestId space exhausted — all 255 IDs are in use, dropping request"
            );
            if let Err(err) = world.unregister_system(system_id) {
                bevy::log::error!(
                    ?err,
                    "failed to unregister system after RequestId exhaustion"
                );
            }
            return;
        };

        if let Some(old) = world
            .resource_mut::<RequestIdHandlers>()
            .insert(req_id, system_id)
        {
            bevy::log::warn!(req_id, "RequestId collision — overwriting existing handler");
            if let Err(err) = world.unregister_system(old) {
                bevy::log::error!(?err, "failed to unregister collided system");
            }
        }

        let tx = world
            .get_resource_mut::<PacketTx>()
            .expect("missing packet handlers");
        if let Err(err) = tx.0.try_send(Some(self.request.with_request_id(req_id))) {
            let mut handlers = world
                .get_resource_mut::<RequestIdHandlers>()
                .expect("missing packet handlers");
            handlers.remove(&req_id);
            if let Err(unreg) = world.unregister_system(system_id) {
                bevy::log::error!(?unreg, "failed to unregister system after send failure");
            }
            bevy::log::warn!(?err, "failed to send msg");
        }
    }
}

pub struct MsgReplyHandlerCommand<
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
> {
    request: LenPacket,
    system: S,
}

impl<S> Command for MsgReplyHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
{
    type Out = ();

    fn apply(self, world: &mut World) {
        let system_id = world.register_system(self.system);

        // TODO: This runs fairly often, and it registers a system each time. We
        // should look for a different way to do this.
        debug!("registered msg reply handler {system_id:?}");
        let req_id = {
            let handlers = world.resource::<MsgRequestIdHandlers>();
            let occupied: std::collections::HashSet<RequestId> = handlers.keys().copied().collect();
            let mut alloc = world.resource_mut::<MsgRequestIdAlloc>();
            alloc.alloc_next_id_avoiding(&occupied)
        };

        let Some(req_id) = req_id else {
            world
                .resource_mut::<MsgRequestQueue>()
                .push_back((self.request, system_id));
            return;
        };

        if let Some(old) = world
            .resource_mut::<MsgRequestIdHandlers>()
            .insert(req_id, system_id)
        {
            bevy::log::warn!(req_id, "RequestId collision — overwriting existing handler");
            if let Err(err) = world.unregister_system(old) {
                bevy::log::error!(?err, "failed to unregister collided system");
            }
        }

        let request = self.request.with_request_id(req_id);
        let sent = world
            .get_resource_mut::<MsgPacketTx>()
            .and_then(|msg_tx| msg_tx.0.try_send(Some(request)).ok())
            .is_some();
        if !sent {
            let mut handlers = world
                .get_resource_mut::<MsgRequestIdHandlers>()
                .expect("missing MsgRequestIdHandlers");
            handlers.remove(&req_id);
            if let Err(err) = world.unregister_system(system_id) {
                bevy::log::error!(?err, "failed to unregister msg reply handler");
            }
            bevy::log::error!(
                "msg connection not available — cannot send request; ensure the editor/run process is connected to Elodin DB so sensor camera and msg backfill work"
            );
        }
    }
}

pub fn flush_msg_request_queue(world: &mut World) {
    while let Some((request, system_id)) = world
        .get_resource_mut::<MsgRequestQueue>()
        .and_then(|mut q| q.pop_front())
    {
        let req_id = {
            let handlers = world.resource::<MsgRequestIdHandlers>();
            let occupied: std::collections::HashSet<RequestId> = handlers.keys().copied().collect();
            let mut alloc = world.resource_mut::<MsgRequestIdAlloc>();
            alloc.alloc_next_id_avoiding(&occupied)
        };

        let Some(req_id) = req_id else {
            world
                .resource_mut::<MsgRequestQueue>()
                .push_front((request, system_id));
            break;
        };

        if let Some(old) = world
            .resource_mut::<MsgRequestIdHandlers>()
            .insert(req_id, system_id)
        {
            bevy::log::warn!(req_id, "RequestId collision — overwriting existing handler");
            if let Err(err) = world.unregister_system(old) {
                bevy::log::error!(?err, "failed to unregister collided system");
            }
        }

        let request = request.with_request_id(req_id);
        let sent = world
            .get_resource_mut::<MsgPacketTx>()
            .and_then(|msg_tx| msg_tx.0.try_send(Some(request)).ok())
            .is_some();
        if !sent {
            let mut handlers = world
                .get_resource_mut::<MsgRequestIdHandlers>()
                .expect("missing MsgRequestIdHandlers");
            handlers.remove(&req_id);
            if let Err(err) = world.unregister_system(system_id) {
                bevy::log::error!(?err, "failed to unregister msg reply handler");
            }
            bevy::log::error!(
                "msg connection not available — cannot send request; ensure the editor/run process is connected to Elodin DB so sensor camera and msg backfill work"
            );
        }
    }
}

pub trait CommandsExt {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, (), M>;

    fn send_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: Msg + DeserializeOwned + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>;

    fn send_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>;

    fn send_msg_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>;

    fn send_msg_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: Msg + DeserializeOwned + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>;
}

impl CommandsExt for Commands<'_, '_> {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, (), M>,
    {
        let system = S::into_system(handler);
        let cmd = ReqHandlerCommand {
            request: msg.into_len_packet(),
            packet_id,
            system,
        };
        self.queue(cmd);
    }

    fn send_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: DeserializeOwned + Msg + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>,
    {
        fn adapter<R: Msg + DeserializeOwned>(
            msg: InRef<OwnedPacket<PacketGrantR>>,
        ) -> Result<R, ErrorResponse> {
            match &*msg {
                OwnedPacket::Msg(m) if m.id == ErrorResponse::ID => {
                    let Ok(m) = m.parse::<ErrorResponse>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Err(m)
                }
                OwnedPacket::Msg(m) if m.id == R::ID => {
                    let Ok(m) = m.parse::<R>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Ok(m)
                }
                other => {
                    let desc = match other {
                        OwnedPacket::Msg(m) => {
                            format!("wrong msg type: got id={:?}, expected id={:?}", m.id, R::ID)
                        }
                        OwnedPacket::Table(_) => "wrong msg type: got Table".to_string(),
                        OwnedPacket::TimeSeries(_) => "wrong msg type: got TimeSeries".to_string(),
                    };
                    Err(ErrorResponse { description: desc })
                }
            }
        }
        let system = adapter::<M::Reply<Slice<Vec<u8>>>>.pipe(handler);
        let system = IntoSystem::into_system(system);

        let cmd = ReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }

    fn send_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>,
    {
        let system = IntoSystem::into_system(handler);

        let cmd = ReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }

    fn send_msg_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>,
    {
        let system = IntoSystem::into_system(handler);
        let cmd = MsgReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }

    fn send_msg_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: DeserializeOwned + Msg + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>,
    {
        fn adapter<R: Msg + DeserializeOwned>(
            msg: InRef<OwnedPacket<PacketGrantR>>,
        ) -> Result<R, ErrorResponse> {
            match &*msg {
                OwnedPacket::Msg(m) if m.id == ErrorResponse::ID => {
                    let Ok(m) = m.parse::<ErrorResponse>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Err(m)
                }
                OwnedPacket::Msg(m) if m.id == R::ID => {
                    let Ok(m) = m.parse::<R>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Ok(m)
                }
                other => {
                    let desc = match other {
                        OwnedPacket::Msg(m) => {
                            format!("wrong msg type: got id={:?}, expected id={:?}", m.id, R::ID)
                        }
                        OwnedPacket::Table(_) => "wrong msg type: got Table".to_string(),
                        OwnedPacket::TimeSeries(_) => "wrong msg type: got TimeSeries".to_string(),
                    };
                    Err(ErrorResponse { description: desc })
                }
            }
        }
        let system = adapter::<M::Reply<Slice<Vec<u8>>>>.pipe(handler);
        let system = IntoSystem::into_system(system);
        let cmd = MsgReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }
}

pub fn new_connection_packets(stream_id: StreamId) -> impl Iterator<Item = LenPacket> {
    [
        // RealTimeBatched delivers all component data whenever new data
        // arrives (batched per last_updated change).  For recorded DBs this
        // sends one table then blocks — historical data is loaded via
        // GetTimeSeries backfill (triggered after DumpMetadata).
        Stream {
            behavior: StreamBehavior::RealTimeBatched,
            id: stream_id,
        }
        .into_len_packet(),
        GetEarliestTimestamp.into_len_packet(),
        DumpMetadata.into_len_packet(),
        GetDbSettings.into_len_packet(),
        SubscribeLastUpdated.into_len_packet(),
        DumpSchema.into_len_packet(),
    ]
    .into_iter()
}

/// Initial packets for the msg TCP connection. Returns empty; main connection
/// handles DumpMetadata, Stream, and subscriptions.
pub fn msg_connection_packets(_stream_id: StreamId) -> impl Iterator<Item = LenPacket> {
    std::iter::empty()
}

pub trait ComponentValueExt {
    fn indexed_iter_mut<'i>(
        &'i mut self,
    ) -> Box<dyn Iterator<Item = (&'i smallvec::SmallVec<[usize; 4]>, ElementValueMut<'i>)> + 'i>;
}
impl ComponentValueExt for ComponentValue {
    fn indexed_iter_mut<'i>(
        &'i mut self,
    ) -> Box<dyn Iterator<Item = (&'i smallvec::SmallVec<[usize; 4]>, ElementValueMut<'i>)> + 'i>
    {
        match self {
            ComponentValue::U8(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U8(x))),
            ),
            ComponentValue::U16(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U16(x))),
            ),
            ComponentValue::U32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U32(x))),
            ),
            ComponentValue::U64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U64(x))),
            ),
            ComponentValue::I8(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I8(x))),
            ),
            ComponentValue::I16(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I16(x))),
            ),
            ComponentValue::I32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I32(x))),
            ),
            ComponentValue::I64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I64(x))),
            ),
            ComponentValue::Bool(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::Bool(x))),
            ),
            ComponentValue::F32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F32(x))),
            ),
            ComponentValue::F64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F64(x))),
            ),
        }
    }
}

#[cfg(test)]
mod earliest_timestamp_tests {
    use super::*;

    #[test]
    fn first_earliest_snaps_epoch_playhead() {
        let mut earliest = EarliestTimestamp(Timestamp(i64::MAX));
        let mut current = CurrentTimestamp(Timestamp::EPOCH);
        apply_earliest_timestamp(
            &mut earliest,
            &mut current,
            EarliestTimestamp(Timestamp(1_000)),
        );
        assert_eq!(earliest.0, Timestamp(1_000));
        assert_eq!(current.0, Timestamp(1_000));
    }

    #[test]
    fn first_earliest_preserves_non_epoch_playhead() {
        let mut earliest = EarliestTimestamp(Timestamp(i64::MAX));
        let mut current = CurrentTimestamp(Timestamp(5_000));
        apply_earliest_timestamp(
            &mut earliest,
            &mut current,
            EarliestTimestamp(Timestamp(1_000)),
        );
        assert_eq!(earliest.0, Timestamp(1_000));
        assert_eq!(current.0, Timestamp(5_000));
    }

    #[test]
    fn later_earliest_is_monotonic_min_without_moving_playhead() {
        let mut earliest = EarliestTimestamp(Timestamp(2_000));
        let mut current = CurrentTimestamp(Timestamp(5_000));
        apply_earliest_timestamp(
            &mut earliest,
            &mut current,
            EarliestTimestamp(Timestamp(1_500)),
        );
        assert_eq!(earliest.0, Timestamp(1_500));
        assert_eq!(current.0, Timestamp(5_000));
        apply_earliest_timestamp(
            &mut earliest,
            &mut current,
            EarliestTimestamp(Timestamp(3_000)),
        );
        assert_eq!(earliest.0, Timestamp(1_500));
        assert_eq!(current.0, Timestamp(5_000));
    }
}

#[cfg(test)]
mod series_store_allowlist_tests {
    use super::*;
    use impeller2::types::PrimType;
    use std::collections::HashSet;

    #[test]
    fn backfill_candidates_only_allowlisted_with_schema() {
        let allow: HashSet<ComponentId> = [ComponentId(1), ComponentId(2)].into_iter().collect();
        let mut schema_reg = ComponentSchemaRegistry::default();
        schema_reg.0.insert(
            ComponentId(1),
            Schema::new(PrimType::F64, [1usize]).expect("schema"),
        );
        // ComponentId(2) allowlisted but no schema — excluded.
        // ComponentId(3) has schema but not allowlisted — excluded.
        schema_reg.0.insert(
            ComponentId(3),
            Schema::new(PrimType::F64, [1usize]).expect("schema"),
        );

        let candidates = series_store_backfill_candidates(&allow, &schema_reg);
        assert_eq!(candidates, vec![ComponentId(1)]);
    }

    #[test]
    fn empty_allowlist_yields_no_backfill_candidates() {
        let allow = HashSet::new();
        let mut schema_reg = ComponentSchemaRegistry::default();
        schema_reg.0.insert(
            ComponentId(1),
            Schema::new(PrimType::F64, [1usize]).expect("schema"),
        );
        assert!(series_store_backfill_candidates(&allow, &schema_reg).is_empty());
    }

    #[test]
    fn remove_series_drops_samples_and_coverage() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId(42);
        cache.insert(
            id,
            Timestamp(1),
            ComponentValue::F64(nox::array![1.0f64].to_dyn()),
        );
        cache.mark_covered(id, Timestamp(1), Timestamp(2));
        assert!(cache.has_series(&id));
        cache.remove_series(&id);
        assert!(!cache.has_series(&id));
        assert!(!cache.is_covered(&id, &(Timestamp(1)..Timestamp(2))));
    }

    #[test]
    fn live_insert_allowed_only_when_in_allowlist() {
        let allow: HashSet<ComponentId> = [ComponentId(1)].into_iter().collect();
        let mut pending = Vec::new();
        for (cid, ts) in [
            (ComponentId(1), Timestamp(10)),
            (ComponentId(2), Timestamp(11)),
        ] {
            if allow.contains(&cid) {
                pending.push((cid, ts, ComponentValue::F64(nox::array![0.0f64].to_dyn())));
            }
        }
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, ComponentId(1));
    }
}

#[cfg(test)]
mod db_components_hierarchy_tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::App;

    fn parent_of(world: &World, entity: Entity) -> Option<Entity> {
        world.get::<ChildOf>(entity).map(|c| c.0)
    }

    #[test]
    fn single_segment_path_parents_under_db_root() {
        let mut app = App::new();
        let root = app
            .world_mut()
            .spawn((DbComponentsRoot, Name::new("db components")))
            .id();
        app.world_mut().insert_resource(EntityMap::default());
        app.world_mut()
            .insert_resource(ComponentMetadataRegistry::default());

        let path = ComponentPath::from_name("position");
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    ensure_component_path_hierarchy(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &path,
                        root,
                    );
                },
            )
            .unwrap();

        let leaf_id = ComponentPath::from_name("position").id;
        let leaf = *app.world().resource::<EntityMap>().get(&leaf_id).unwrap();
        assert_eq!(parent_of(app.world(), leaf), Some(root));
    }

    #[test]
    fn nested_path_wires_full_chain_under_db_root() {
        let mut app = App::new();
        let root = app
            .world_mut()
            .spawn((DbComponentsRoot, Name::new("db components")))
            .id();
        app.world_mut().insert_resource(EntityMap::default());
        app.world_mut()
            .insert_resource(ComponentMetadataRegistry::default());

        let path = ComponentPath::from_name("a.b.c");
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    ensure_component_path_hierarchy(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &path,
                        root,
                    );
                },
            )
            .unwrap();

        let a = *app
            .world()
            .resource::<EntityMap>()
            .get(&ComponentPart::new("a").id)
            .unwrap();
        let ab = *app
            .world()
            .resource::<EntityMap>()
            .get(&ComponentPart::new("a.b").id)
            .unwrap();
        let abc = *app
            .world()
            .resource::<EntityMap>()
            .get(&ComponentPart::new("a.b.c").id)
            .unwrap();

        assert_eq!(parent_of(app.world(), a), Some(root));
        assert_eq!(parent_of(app.world(), ab), Some(a));
        assert_eq!(parent_of(app.world(), abc), Some(ab));
    }

    #[test]
    fn second_call_does_not_reparent_existing_entities() {
        let mut app = App::new();
        let root = app
            .world_mut()
            .spawn((DbComponentsRoot, Name::new("db components")))
            .id();
        let other_parent = app.world_mut().spawn_empty().id();
        app.world_mut().insert_resource(EntityMap::default());
        app.world_mut()
            .insert_resource(ComponentMetadataRegistry::default());

        let path = ComponentPath::from_name("solo");
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    ensure_component_path_hierarchy(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &path,
                        root,
                    );
                },
            )
            .unwrap();

        let leaf_id = ComponentPath::from_name("solo").id;
        let leaf = *app.world().resource::<EntityMap>().get(&leaf_id).unwrap();
        app.world_mut()
            .entity_mut(leaf)
            .insert(ChildOf(other_parent));

        let path = ComponentPath::from_name("solo");
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    ensure_component_path_hierarchy(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &path,
                        root,
                    );
                },
            )
            .unwrap();

        assert_eq!(parent_of(app.world(), leaf), Some(other_parent));
    }

    #[test]
    fn leaf_created_before_hierarchy_stays_orphaned_without_metadata_helper() {
        // Documents the bug: creating only the leaf first, then running the
        // hierarchy helper, leaves the leaf without ChildOf because it is not
        // newly_created. Metadata handlers must call ensure_component_path_hierarchy
        // (full path) instead of try_insert_entity(leaf) alone.
        let mut app = App::new();
        let root = app
            .world_mut()
            .spawn((DbComponentsRoot, Name::new("db components")))
            .id();
        app.world_mut().insert_resource(EntityMap::default());
        app.world_mut()
            .insert_resource(ComponentMetadataRegistry::default());

        let path = ComponentPath::from_name("x.y");
        let leaf_part = path.path.last().unwrap().clone();
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    let _ = try_insert_entity(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &leaf_part,
                    );
                },
            )
            .unwrap();

        let path = ComponentPath::from_name("x.y");
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands,
                      mut entity_map: ResMut<EntityMap>,
                      mut metadata_reg: ResMut<ComponentMetadataRegistry>| {
                    ensure_component_path_hierarchy(
                        &mut entity_map,
                        &mut metadata_reg,
                        &mut commands,
                        &path,
                        root,
                    );
                },
            )
            .unwrap();

        let y = *app
            .world()
            .resource::<EntityMap>()
            .get(&ComponentPart::new("x.y").id)
            .unwrap();
        let x = *app
            .world()
            .resource::<EntityMap>()
            .get(&ComponentPart::new("x").id)
            .unwrap();
        assert_eq!(parent_of(app.world(), x), Some(root));
        // Leaf was pre-created, so hierarchy helper must not invent a parent.
        assert_eq!(parent_of(app.world(), y), None);
    }
}
