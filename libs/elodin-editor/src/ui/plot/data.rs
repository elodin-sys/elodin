use bevy::asset::Asset;
use bevy::log::warn_once;
use bevy::prelude::{InRef, Res, ResMut};
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::{Commands, Query},
    prelude::Resource,
};
use bevy_render::render_resource::{Buffer, BufferDescriptor, BufferSlice, BufferUsages};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, Timestamp};
use impeller2_bevy::{
    BackfillState, CommandsExt, ComponentAdapters, ComponentPathRegistry, ComponentSchemaRegistry,
    PacketGrantR, PacketHandlerInput, PacketHandlers, SeriesFetchPriority, TelemetryCache,
};
use impeller2_wkt::{
    ComponentValue, CurrentTimestamp, EarliestTimestamp, GetTimeSeries, Line3d, VectorArrow3d,
};
use itertools::{Itertools, MinMaxResult};
use nodit::NoditMap;
use nodit::interval::ii;
use roaring::bitmap::RoaringBitmap;
use zerocopy::{Immutable, IntoBytes};

use std::any::type_name;
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicBool};
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::object_3d::Object3DState;
use crate::sensor_camera::SensorCameraConfigs;
use crate::ui::inspector::viewport::Viewport;
use crate::ui::monitor::MonitorData;
use crate::ui::plot::gpu::INDEX_BUFFER_LEN;
use crate::ui::plot::state::GraphState;
use crate::ui::schematic::EqlExt;
use crate::{EqlContext, SelectedTimeRange};
use hamann_chen_line::{select_polyline3_indices, select_time_value_indices};

use super::PlotBounds;

/// Maximum points to request for overview data (LTTB downsampled)
/// Must be <= CHUNK_LEN to fit within a single GPU buffer shard
pub const OVERVIEW_MAX_POINTS: usize = CHUNK_LEN;

/// Tuning for **Hamann–Chen** downsampling of live [`LineTree`] telemetry.
///
/// The simplifier lives in the `hamann-chen-line` workspace crate; its steps follow Shane Celis’s
/// C# reference [`PiecewiseLinearCurveApproximation.cs`](https://gist.github.com/shanecelis/2e0ffd790e31507fba04dd56f806667a)
/// (Hamann–Chen style curvature sampling). Integration notes: `libs/hamann-chen-line/README.md`.
///
/// After graph ingest, [`maybe_compress_all_graph_lines`] walks components and may rewrite
/// [`Line`] assets so total stored points stay bounded. Scalar graphs use
/// [`LineTree::compress_time_value_hamann`]; exactly **three** lines per component with identical
/// timestamps use joint 3D polyline simplification so `line_3d` trails stay coherent.
#[derive(Resource, Clone, Debug)]
pub struct CurveCompressSettings {
    /// When `false`, no Hamann–Chen pass runs (series grow until other limits apply).
    pub enabled: bool,
    /// Run a compression pass when [`LineTree::total_points`] **exceeds** this threshold.
    pub compress_after_total_points: usize,
    /// Target vertex count **`m`** passed to `hamann-chen-line` after a pass (per line, or shared
    /// across the three joint XYZ lines).
    pub compress_to_points: usize,
    /// Fraction of the **last** samples (by time order) left uncompressed after a pass.
    ///
    /// - `0.0` — compress the whole series (still capped by `compress_to_points`).
    /// - `0.2` — keep roughly the last 20% at full resolution; Hamann–Chen runs on the leading
    ///   prefix only.
    pub keep_recent_fraction: f32,
    /// Non-destructive **archive + view** architecture. When `true`,
    /// [`PlotDataComponent::push_value`] appends each accepted live sample to
    /// [`LineTree::raw_slice`], and [`LineTree::compress_time_value_hamann`] reads from that
    /// archive instead of the decimated view. Switching this off at runtime reverts to the
    /// historical destructive pipeline (no archive, HC reads and rewrites the view).
    pub archive_enabled: bool,
    /// Throttle: recompute the view as soon as the archive has grown by this many samples since
    /// the last HC pass. Set small (relative to [`Self::compress_after_total_points`]) to keep
    /// the UI responsive; too small wastes CPU recomputing the view too often.
    pub view_recompute_min_samples: usize,
    /// Throttle: if the archive hasn't grown enough for [`Self::view_recompute_min_samples`],
    /// still recompute when at least this many milliseconds have passed since the last pass.
    /// Caps worst-case staleness for low-rate telemetry.
    pub view_recompute_min_interval_ms: u64,
}

impl Default for CurveCompressSettings {
    fn default() -> Self {
        let cap = CHUNK_COUNT.saturating_mul(CHUNK_LEN);
        let threshold = cap.saturating_mul(3) / 4;
        Self {
            enabled: true,
            compress_after_total_points: threshold,
            compress_to_points: cap / 2,
            keep_recent_fraction: 0.0,
            archive_enabled: true,
            view_recompute_min_samples: (threshold / 20).max(1),
            view_recompute_min_interval_ms: 250,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub element_names: Vec<String>,
    pub lines: BTreeMap<usize, Handle<Line>>,
    /// Last time this component's LineTrees were synced from SeriesStore.
    last_query_attempt: Option<Instant>,
}

impl PlotDataComponent {
    pub fn new(component_label: impl ToString, element_names: Vec<String>) -> Self {
        Self {
            label: component_label.to_string(),
            element_names,
            lines: BTreeMap::new(),
            last_query_attempt: None,
        }
    }

    pub fn push_value(
        &mut self,
        component_view: ComponentView<'_>,
        assets: &mut Assets<Line>,
        timestamp: Timestamp,
        earliest_timestamp: Timestamp,
        archive_enabled: bool,
    ) {
        let element_names = self
            .element_names
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| Some(s.as_str()))
            .chain(std::iter::repeat(None));
        for (i, (new_value, name)) in component_view.iter().zip(element_names).enumerate() {
            let new_value = new_value.as_f32();
            let line = self.lines.entry(i).or_insert_with(|| {
                let label = name.map(str::to_string).unwrap_or_else(|| format!("[{i}]"));
                assets.add(Line {
                    label,
                    ..Default::default()
                })
            });
            let mut line = assets.get_mut(line.id()).expect("missing line asset");
            // Only accept data at timestamps beyond all existing data for this
            // line. Live FixedRate snapshots can repeat the playhead timestamp;
            // skipping already-covered times keeps the tip monotonic.
            let mut accepted = false;
            if let Some(last) = line.data.last() {
                if timestamp <= last.summary.end_timestamp {
                    continue;
                }
                if last.timestamps.len() < CHUNK_LEN {
                    line.data.update_last(|c| {
                        c.push(timestamp, earliest_timestamp, new_value);
                    });
                    accepted = true;
                }
            }
            if !accepted {
                let new_chunk = Chunk::from_initial_value(timestamp, earliest_timestamp, new_value);
                line.data.insert(new_chunk);
            }
            // Archive mirrors exactly the sequence accepted by the view. Monotonicity of raw is
            // enforced by `push_raw` itself, so re-ingestion of historical timestamps (already
            // rejected above) is doubly safe.
            line.data.push_raw(archive_enabled, timestamp, new_value);
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct CollectedGraphData {
    pub components: BTreeMap<ComponentId, PlotDataComponent>,
}

impl CollectedGraphData {
    pub fn get_component(&self, component_id: &ComponentId) -> Option<&PlotDataComponent> {
        self.components.get(component_id)
    }
    pub fn get_component_mut(
        &mut self,
        component_id: &ComponentId,
    ) -> Option<&mut PlotDataComponent> {
        self.components.get_mut(component_id)
    }

    pub fn get_line(&self, component_id: &ComponentId, index: usize) -> Option<&Handle<Line>> {
        self.components
            .get(component_id)
            .and_then(|component| component.lines.get(&index))
    }
}

pub fn pkt_handler(
    PacketHandlerInput { packet, registry }: PacketHandlerInput,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<Line>>,
    tick: Res<CurrentTimestamp>,
    earliest_timestamp: Res<EarliestTimestamp>,
    curve_compress: Res<CurveCompressSettings>,
    selected_range: Res<SelectedTimeRange>,
) {
    let mut tick = *tick;
    if let OwnedPacket::Table(table) = packet {
        if let Err(err) = table.sink(registry, &mut tick) {
            warn_once!(?err, "tick sink failed");
        }
        if let Err(err) = table.sink(
            registry,
            &mut |component_id, view: ComponentView<'_>, timestamp: Option<Timestamp>| {
                let Some(plot_data) = collected_graph_data.get_component_mut(&component_id) else {
                    return;
                };
                if let Some(timestamp) = timestamp {
                    plot_data.push_value(
                        view,
                        &mut lines,
                        timestamp,
                        earliest_timestamp.0,
                        curve_compress.archive_enabled,
                    );
                }
            },
        ) {
            warn_once!(?err, "graph sink failed");
        }
        // Short windows draw every sample — do not rewrite the CPU LineTree via HC.
        if !crate::is_short_accuracy_window(&selected_range.0) {
            maybe_compress_all_graph_lines(
                &mut collected_graph_data,
                &mut lines,
                earliest_timestamp.0,
                &curve_compress,
            );
        }
    }
}

/// Optionally rewrites oversized [`Line`] / [`LineTree`] series using **Hamann–Chen** sampling
/// (`hamann-chen-line`, algorithm structure from Shane Celis’s C# gist linked on
/// [`CurveCompressSettings`]).
///
/// Does nothing when [`CurveCompressSettings::enabled`] is false. Callers should also skip this
/// for short accuracy windows (≤30 s) so live ingest cannot rewrite truth. Otherwise, for each
/// plot component: if there are **three** lines and timestamps match across them, tries joint 3D
/// compression; else compresses each line independently. GPU index-buffer sizing is handled
/// separately in the plot render path.
pub fn maybe_compress_all_graph_lines(
    graph_data: &mut CollectedGraphData,
    lines: &mut Assets<Line>,
    earliest: Timestamp,
    settings: &CurveCompressSettings,
) {
    if !settings.enabled {
        return;
    }
    for component in graph_data.components.values_mut() {
        let handles: Vec<Handle<Line>> = component.lines.values().cloned().collect();
        if handles.len() == 3 && try_joint_triline_compress(lines, &handles, earliest, settings) {
            continue;
        }
        for line_handle in &handles {
            let Some(mut line) = lines.get_mut(line_handle) else {
                continue;
            };
            line.maybe_compress_live(earliest, settings);
        }
    }
    if settings.archive_enabled {
        // Safety net: flag outsized archives so multi-hour aerospace runs don't silently blow up
        // RAM. V1 has no enforced horizon; a user report on this warning is the trigger to add
        // `archive_horizon_seconds` in a follow-up.
        const RAW_ARCHIVE_WARN_BYTES: usize = 100 * 1024 * 1024;
        for component in graph_data.components.values() {
            for line_handle in component.lines.values() {
                if let Some(line) = lines.get(line_handle)
                    && line.data.raw_archive_bytes() > RAW_ARCHIVE_WARN_BYTES
                {
                    let mb = line.data.raw_archive_bytes() / (1024 * 1024);
                    warn_once!(
                        raw_mb = mb,
                        line_label = %line.label,
                        "curve_compress: raw archive exceeds 100 MB on a single line; consider lowering run length or adding archive_horizon_seconds"
                    );
                }
            }
        }
    }
}

fn try_joint_triline_compress(
    lines: &mut Assets<Line>,
    handles: &[Handle<Line>],
    earliest: Timestamp,
    settings: &CurveCompressSettings,
) -> bool {
    if handles.len() != 3 {
        return false;
    }
    let (hx, hy, hz) = (&handles[0], &handles[1], &handles[2]);
    let (lx, ly, lz) = match (lines.get(hx), lines.get(hy), lines.get(hz)) {
        (Some(x), Some(y), Some(z)) => (x, y, z),
        _ => return false,
    };
    if settings.archive_enabled {
        if !lx.data.should_recompress(settings) {
            return false;
        }
    } else if lx.data.total_points() <= settings.compress_after_total_points {
        return false;
    }
    // Own the sequence: archive slices borrow from `lines`, which conflicts with the later
    // `lines.get_mut` on the same assets.
    let (tsx, vx): (Vec<Timestamp>, Vec<f32>) = if settings.archive_enabled {
        let (t, v) = lx.data.raw_slice();
        (t.to_vec(), v.to_vec())
    } else {
        lx.data.flattened_time_series()
    };
    let (tsy, vy): (Vec<Timestamp>, Vec<f32>) = if settings.archive_enabled {
        let (t, v) = ly.data.raw_slice();
        (t.to_vec(), v.to_vec())
    } else {
        ly.data.flattened_time_series()
    };
    let (tsz, vz): (Vec<Timestamp>, Vec<f32>) = if settings.archive_enabled {
        let (t, v) = lz.data.raw_slice();
        (t.to_vec(), v.to_vec())
    } else {
        lz.data.flattened_time_series()
    };
    if tsx.len() != tsy.len() || tsx.len() != tsz.len() {
        return false;
    }
    if tsx
        .iter()
        .zip(&tsy)
        .zip(&tsz)
        .any(|((&a, &b), &c)| a != b || a != c)
    {
        return false;
    }
    if tsx.len() < 3 {
        return false;
    }
    let n = tsx.len();
    let pos: Vec<bevy::prelude::Vec3> = vx
        .iter()
        .zip(vy.iter())
        .zip(vz.iter())
        .map(|((&x, &y), &z)| bevy::prelude::Vec3::new(x, y, z))
        .collect();

    let keep = recent_tail_keep_count(n, settings.keep_recent_fraction);
    let split = n.saturating_sub(keep);

    let (new_ts, new_x, new_y, new_z) = if settings.keep_recent_fraction > 0.0 && split >= 3 {
        let budget = settings.compress_to_points.saturating_sub(keep).max(2);
        let target_old = budget.min(split).max(2);
        let pos_old = &pos[..split];
        let idx = select_polyline3_indices(pos_old, target_old);
        if idx.len() < 2 {
            return false;
        }
        let mut new_ts: Vec<Timestamp> = idx.iter().map(|&i| tsx[i]).collect();
        let mut new_x: Vec<f32> = idx.iter().map(|&i| vx[i]).collect();
        let mut new_y: Vec<f32> = idx.iter().map(|&i| vy[i]).collect();
        let mut new_z: Vec<f32> = idx.iter().map(|&i| vz[i]).collect();
        if let (Some(tl), Some(tr)) = (new_ts.last(), tsx.get(split))
            && *tl == *tr
        {
            new_ts.pop();
            new_x.pop();
            new_y.pop();
            new_z.pop();
        }
        new_ts.extend_from_slice(&tsx[split..]);
        new_x.extend_from_slice(&vx[split..]);
        new_y.extend_from_slice(&vy[split..]);
        new_z.extend_from_slice(&vz[split..]);
        (new_ts, new_x, new_y, new_z)
    } else {
        let target = settings.compress_to_points.max(2).min(n);
        let idx = select_polyline3_indices(&pos, target);
        if idx.len() < 2 {
            return false;
        }
        let new_ts: Vec<Timestamp> = idx.iter().map(|&i| tsx[i]).collect();
        let new_x: Vec<f32> = idx.iter().map(|&i| vx[i]).collect();
        let new_y: Vec<f32> = idx.iter().map(|&i| vy[i]).collect();
        let new_z: Vec<f32> = idx.iter().map(|&i| vz[i]).collect();
        (new_ts, new_x, new_y, new_z)
    };
    // Scope each `AssetMut` guard so consecutive `get_mut` calls don't
    // overlap mutable borrows of the `Assets` collection.
    {
        let Some(mut xl) = lines.get_mut(hx) else {
            return false;
        };
        xl.data
            .rebuild_from_time_value_pairs(earliest, &new_ts, &new_x);
        xl.data.mark_compressed();
    }
    {
        let Some(mut yl) = lines.get_mut(hy) else {
            return false;
        };
        yl.data
            .rebuild_from_time_value_pairs(earliest, &new_ts, &new_y);
        yl.data.mark_compressed();
    }
    {
        let Some(mut zl) = lines.get_mut(hz) else {
            return false;
        };
        zl.data
            .rebuild_from_time_value_pairs(earliest, &new_ts, &new_z);
        zl.data.mark_compressed();
    }
    true
}

pub fn setup_pkt_handler(mut packet_handlers: ResMut<PacketHandlers>, mut commands: Commands) {
    let sys = commands.register_system(pkt_handler);
    packet_handlers.0.push(sys);
}

#[allow(clippy::too_many_arguments)]
pub fn queue_timestamp_read(
    selected_range: Res<SelectedTimeRange>,
    behavior: Res<crate::TimeRangeBehavior>,
    mut graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<Line>>,
    earliest_timestamp: Res<EarliestTimestamp>,
    graph_states: Query<&GraphState>,
    line_3ds: Query<&Line3d>,
    object_3ds: Query<&Object3DState>,
    eql_ctx: Res<EqlContext>,
    series_store: Res<TelemetryCache>,
    mut sync_state: ResMut<PlotSyncState>,
    mut prefetch: ResMut<VisiblePrefetchState>,
    schema_reg: Res<ComponentSchemaRegistry>,
    mut commands: Commands,
) {
    let sync_range = visible_sync_range(&selected_range, &behavior);
    let Some((range_key, sync_range)) = sync_range else {
        return;
    };

    // Fill the visible window ASAP for enabled plots while full begin→end
    // backfill continues in the background.
    prefetch_visible_window(
        &sync_range,
        range_key,
        &graph_states,
        &line_3ds,
        &object_3ds,
        &eql_ctx,
        &series_store,
        &schema_reg,
        &mut prefetch,
        &mut commands,
    );

    sync_plot_lines_from_series_store(
        range_key,
        &sync_range,
        &mut graph_data,
        &mut lines,
        earliest_timestamp.0,
        &graph_states,
        &line_3ds,
        &object_3ds,
        &eql_ctx,
        &series_store,
        &mut sync_state,
    );
}

fn visible_sync_range(
    selected_range: &SelectedTimeRange,
    behavior: &crate::TimeRangeBehavior,
) -> Option<((i64, i64), Range<Timestamp>)> {
    let selected = selected_range.0.clone();
    if selected.start.0 == i64::MIN || selected.end.0 == i64::MAX || selected.start >= selected.end
    {
        return None;
    }
    let range_key = if behavior.is_trailing_window() {
        let start = floor_ts_quantum(selected.start, REQUEST_KEY_QUANTUM_MICROS);
        let end = Timestamp(
            selected
                .end
                .0
                .div_euclid(REQUEST_KEY_QUANTUM_MICROS)
                .saturating_add(1)
                .saturating_mul(REQUEST_KEY_QUANTUM_MICROS),
        );
        (start.0, end.0.max(start.0.saturating_add(1)))
    } else {
        (selected.start.0, selected.end.0)
    };
    Some((range_key, Timestamp(range_key.0)..Timestamp(range_key.1)))
}

/// Dedupes in-flight visible-window GetTimeSeries requests.
#[derive(Resource, Default)]
pub struct VisiblePrefetchState {
    pub(crate) in_flight: HashSet<(ComponentId, i64, i64)>,
}

impl VisiblePrefetchState {
    pub fn clear_in_flight(&mut self) {
        self.in_flight.clear();
    }
}

const VISIBLE_PREFETCH_LIMIT: usize = 8192;

#[allow(clippy::too_many_arguments)]
fn prefetch_visible_window(
    sync_range: &Range<Timestamp>,
    range_key: (i64, i64),
    graph_states: &Query<&GraphState>,
    line_3ds: &Query<&Line3d>,
    object_3ds: &Query<&Object3DState>,
    eql_ctx: &EqlContext,
    series_store: &TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
    prefetch: &mut VisiblePrefetchState,
    commands: &mut Commands,
) {
    // Prefetch only plot/3D consumers; monitors/viewport/arrows are allowlisted
    // in `update_series_fetch_priority` and filled by live + begin→end backfill.
    let fetch_ids = plot_fetch_component_ids(graph_states, line_3ds, object_3ds, eql_ctx);
    if fetch_ids.is_empty() {
        return;
    }
    const MAX_VISIBLE_PREFETCH_IN_FLIGHT: usize = 32;
    for component_id in fetch_ids {
        if prefetch.in_flight.len() >= MAX_VISIBLE_PREFETCH_IN_FLIGHT {
            break;
        }
        if !schema_reg.0.contains_key(&component_id) {
            continue;
        }
        if series_store.sample_count_in_range(&component_id, sync_range) > 0 {
            prefetch
                .in_flight
                .remove(&(component_id, range_key.0, range_key.1));
            continue;
        }
        let key = (component_id, range_key.0, range_key.1);
        if prefetch.in_flight.contains(&key) {
            continue;
        }
        prefetch.in_flight.insert(key);
        let start = sync_range.start;
        let end = sync_range.end;
        let packet_id = fastrand::u16(..).to_le_bytes();
        let msg = GetTimeSeries {
            id: packet_id,
            range: start..end,
            component_id,
            limit: Some(VISIBLE_PREFETCH_LIMIT),
        };
        commands.send_req_with_handler(
            msg,
            packet_id,
            move |pkt: InRef<OwnedPacket<PacketGrantR>>,
                  mut series_store: ResMut<TelemetryCache>,
                  schema_reg: Res<ComponentSchemaRegistry>,
                  priority: Res<SeriesFetchPriority>,
                  mut prefetch: ResMut<VisiblePrefetchState>,
                  mut commands: Commands| {
                apply_visible_prefetch_page(
                    &pkt,
                    component_id,
                    start,
                    end,
                    range_key,
                    &mut series_store,
                    &schema_reg,
                    &priority,
                    &mut prefetch,
                    &mut commands,
                );
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_visible_prefetch_page(
    pkt: &OwnedPacket<PacketGrantR>,
    component_id: ComponentId,
    req_start: Timestamp,
    req_end: Timestamp,
    range_key: (i64, i64),
    series_store: &mut TelemetryCache,
    schema_reg: &ComponentSchemaRegistry,
    priority: &SeriesFetchPriority,
    prefetch: &mut VisiblePrefetchState,
    commands: &mut Commands,
) {
    let drop_in_flight = |prefetch: &mut VisiblePrefetchState| {
        prefetch
            .in_flight
            .remove(&(component_id, range_key.0, range_key.1));
    };
    // Component may have left the allowlist while this page was in flight.
    if !priority.high.contains(&component_id) {
        drop_in_flight(prefetch);
        return;
    }
    let OwnedPacket::TimeSeries(time_series) = pkt else {
        drop_in_flight(prefetch);
        return;
    };
    let (Ok(timestamps), Ok(buf)) = (time_series.timestamps(), time_series.data()) else {
        drop_in_flight(prefetch);
        return;
    };
    let Some(schema) = schema_reg.0.get(&component_id) else {
        drop_in_flight(prefetch);
        return;
    };
    let elem_size = schema.size();
    let mut last_ts = req_start;
    for (i, &timestamp) in timestamps.iter().enumerate() {
        let offset = i * elem_size;
        if offset + elem_size > buf.len() {
            break;
        }
        if let Ok(view) = impeller2::types::ComponentView::try_from_bytes_shape(
            &buf[offset..offset + elem_size],
            schema.shape(),
            schema.prim_type(),
        ) {
            series_store.insert(component_id, timestamp, ComponentValue::from_view(view));
            last_ts = timestamp;
        }
    }
    if let Some(&first) = timestamps.first() {
        series_store.mark_covered(component_id, first, Timestamp(last_ts.0.saturating_add(1)));
    }
    if !priority.high.contains(&component_id) {
        series_store.remove_series(&component_id);
        drop_in_flight(prefetch);
        return;
    }
    // Continue paging until the visible window is filled or the DB has no more.
    if !timestamps.is_empty()
        && timestamps.len() >= VISIBLE_PREFETCH_LIMIT
        && last_ts.0.saturating_add(1) < req_end.0
    {
        let next_start = Timestamp(last_ts.0.saturating_add(1));
        let packet_id = fastrand::u16(..).to_le_bytes();
        let msg = GetTimeSeries {
            id: packet_id,
            range: next_start..req_end,
            component_id,
            limit: Some(VISIBLE_PREFETCH_LIMIT),
        };
        commands.send_req_with_handler(
            msg,
            packet_id,
            move |pkt: InRef<OwnedPacket<PacketGrantR>>,
                  mut series_store: ResMut<TelemetryCache>,
                  schema_reg: Res<ComponentSchemaRegistry>,
                  priority: Res<SeriesFetchPriority>,
                  mut prefetch: ResMut<VisiblePrefetchState>,
                  mut commands: Commands| {
                apply_visible_prefetch_page(
                    &pkt,
                    component_id,
                    next_start,
                    req_end,
                    range_key,
                    &mut series_store,
                    &schema_reg,
                    &priority,
                    &mut prefetch,
                    &mut commands,
                );
            },
        );
    } else {
        drop_in_flight(prefetch);
    }
}

/// Rebuild enabled plot LineTrees from the full SeriesStore for the visible window.
/// Playback is never blocked: whatever samples are already in the store are shown;
/// as backfill / visible prefetch streams in, subsequent syncs fill gaps.
#[allow(clippy::too_many_arguments)]
pub fn sync_plot_lines_from_series_store(
    range_key: (i64, i64),
    sync_range: &Range<Timestamp>,
    graph_data: &mut CollectedGraphData,
    lines: &mut Assets<Line>,
    earliest: Timestamp,
    graph_states: &Query<&GraphState>,
    line_3ds: &Query<&Line3d>,
    object_3ds: &Query<&Object3DState>,
    eql_ctx: &EqlContext,
    series_store: &TelemetryCache,
    sync_state: &mut PlotSyncState,
) {
    let fetch_ids = plot_fetch_component_ids(graph_states, line_3ds, object_3ds, eql_ctx);
    let range_changed = sync_state.last_range != Some(range_key);
    let enabled_changed = sync_state.last_enabled != fetch_ids;
    let gen_changed = series_store.generation() != sync_state.last_generation;
    let due = sync_state
        .last_rebuild
        .map(|t| t.elapsed() >= Duration::from_millis(100))
        .unwrap_or(true);

    if !(range_changed || enabled_changed || (gen_changed && due)) {
        return;
    }

    sync_state.last_range = Some(range_key);
    sync_state.last_generation = series_store.generation();
    sync_state.last_enabled = fetch_ids.clone();
    sync_state.last_rebuild = Some(Instant::now());

    const TEN_MINUTES_MICROS: i64 = 600_000_000;
    let range_duration = sync_range.end.0.saturating_sub(sync_range.start.0);
    let max_points = if range_duration > TEN_MINUTES_MICROS {
        Some(OVERVIEW_MAX_POINTS)
    } else {
        None
    };

    for (&component_id, component) in graph_data.components.iter_mut() {
        if !fetch_ids.contains(&component_id) {
            continue;
        }
        let samples_in_window = series_store.sample_count_in_range(&component_id, sync_range);
        let num_elements = component.element_names.len().max(1);
        for element_index in 0..num_elements {
            let handle = component.lines.entry(element_index).or_insert_with(|| {
                let label = component
                    .element_names
                    .get(element_index)
                    .filter(|s| !s.is_empty())
                    .cloned()
                    .unwrap_or_else(|| format!("[{element_index}]"));
                lines.add(Line {
                    label,
                    ..Default::default()
                })
            });
            let Some(mut line) = lines.get_mut(handle) else {
                continue;
            };
            if samples_in_window == 0 {
                // Don't wipe live tip / prior draw when the store hasn't caught
                // up to this window yet — unless the camera moved (show empty).
                if range_changed {
                    line.data.clear();
                }
                continue;
            }
            line.data.clear();
            project_series_element_to_line(
                series_store,
                component_id,
                element_index,
                sync_range,
                earliest,
                &mut line.data,
                max_points,
            );
            line.last_queried = Some(Instant::now());
        }
        component.last_query_attempt = Some(Instant::now());
    }
}

/// Tracks when plot LineTrees were last rebuilt from SeriesStore.
#[derive(Resource, Default)]
pub struct PlotSyncState {
    last_range: Option<(i64, i64)>,
    last_generation: u64,
    last_enabled: HashSet<ComponentId>,
    last_rebuild: Option<Instant>,
}

/// Request-key quantum (100 ms) — used when quantizing trailing sync windows.
pub(crate) const REQUEST_KEY_QUANTUM_MICROS: i64 = 100_000;

fn floor_ts_quantum(ts: Timestamp, quantum_micros: i64) -> Timestamp {
    if quantum_micros <= 0 {
        return ts;
    }
    Timestamp(
        ts.0.div_euclid(quantum_micros)
            .saturating_mul(quantum_micros),
    )
}

/// Parse an EQL string and insert every referenced component ID into `out`.
fn collect_eql_component_ids(eql: &str, eql_ctx: &EqlContext, out: &mut HashSet<ComponentId>) {
    if eql.trim().is_empty() {
        return;
    }
    let Ok(parsed) = eql_ctx.0.parse_str(eql) else {
        return;
    };
    for (c, _) in parsed.to_graph_components() {
        out.insert(c.id);
    }
}

/// Component IDs for plot LineTree sync / visible-window prefetch
/// (graphs + Line3d + object_3d only).
fn plot_fetch_component_ids(
    graph_states: &Query<&GraphState>,
    line_3ds: &Query<&Line3d>,
    object_3ds: &Query<&Object3DState>,
    eql_ctx: &EqlContext,
) -> HashSet<ComponentId> {
    let mut ids = HashSet::new();
    for gs in graph_states.iter() {
        for (path, _) in gs.enabled_lines.keys() {
            ids.insert(path.id);
        }
    }
    for line in line_3ds.iter() {
        collect_eql_component_ids(&line.eql, eql_ctx, &mut ids);
    }
    for obj in object_3ds.iter() {
        collect_eql_component_ids(&obj.data.eql, eql_ctx, &mut ids);
        // Thruster particle intensity EQL (plume / cold_gas / motor smoke).
        for thruster in &obj.data.thrusters {
            collect_eql_component_ids(&thruster.intensity, eql_ctx, &mut ids);
        }
    }
    ids
}

/// Full SeriesStore consumer set: plots/3D plus monitors, viewport cameras,
/// and vector arrows.
fn enabled_fetch_component_ids(
    graph_states: &Query<&GraphState>,
    line_3ds: &Query<&Line3d>,
    object_3ds: &Query<&Object3DState>,
    monitors: &Query<&MonitorData>,
    viewports: &Query<&Viewport>,
    vector_arrows: &Query<&VectorArrow3d>,
    eql_ctx: &EqlContext,
) -> HashSet<ComponentId> {
    let mut ids = plot_fetch_component_ids(graph_states, line_3ds, object_3ds, eql_ctx);
    for monitor in monitors.iter() {
        if !monitor.component_name.trim().is_empty() {
            ids.insert(ComponentId::new(&monitor.component_name));
        }
    }
    for viewport in viewports.iter() {
        collect_eql_component_ids(&viewport.pos.eql, eql_ctx, &mut ids);
        collect_eql_component_ids(&viewport.look_at.eql, eql_ctx, &mut ids);
        collect_eql_component_ids(&viewport.up.eql, eql_ctx, &mut ids);
    }
    for arrow in vector_arrows.iter() {
        collect_eql_component_ids(&arrow.vector, eql_ctx, &mut ids);
        if let Some(origin) = &arrow.origin {
            collect_eql_component_ids(origin, eql_ctx, &mut ids);
        }
    }
    ids
}

/// SeriesStore allowlist: UI consumers plus viewport adapter pair IDs
/// (e.g. `ball_1.world_pos`, not only the static `world_pos` type id).
pub(crate) fn build_series_store_allowlist(
    consumer_ids: HashSet<ComponentId>,
    extra_ids: impl IntoIterator<Item = ComponentId>,
) -> HashSet<ComponentId> {
    let mut ids = consumer_ids;
    ids.extend(extra_ids);
    ids
}

/// All `ComponentPathRegistry` IDs whose leaf name matches a registered adapter
/// type (`WorldPos::COMPONENT_ID` = hash("world_pos"), etc.), plus the static
/// adapter keys themselves.
pub(crate) fn viewport_adapter_component_ids(
    path_reg: &ComponentPathRegistry,
    adapter_leaf_ids: &HashSet<ComponentId>,
) -> HashSet<ComponentId> {
    let mut ids = adapter_leaf_ids.clone();
    for (&id, path) in path_reg.0.iter() {
        if adapter_leaf_ids.contains(&path.tail().id) {
            ids.insert(id);
        }
    }
    ids
}

/// `{entity}.world_pos` for each configured sensor camera parent entity.
pub(crate) fn sensor_camera_world_pos_ids(configs: &SensorCameraConfigs) -> HashSet<ComponentId> {
    configs
        .0
        .iter()
        .map(|c| ComponentId::new(&format!("{}.world_pos", c.entity_name)))
        .collect()
}

/// Keep SeriesStore allowlist in sync with enabled plot / 3D / UI consumers.
/// Reclaims RAM when IDs leave the allowlist so re-subscribe re-fetches cleanly.
#[allow(clippy::too_many_arguments)]
pub fn update_series_fetch_priority(
    graph_states: Query<&GraphState>,
    line_3ds: Query<&Line3d>,
    object_3ds: Query<&Object3DState>,
    monitors: Query<&MonitorData>,
    viewports: Query<&Viewport>,
    vector_arrows: Query<&VectorArrow3d>,
    eql_ctx: Res<EqlContext>,
    path_reg: Res<ComponentPathRegistry>,
    adapters: Res<ComponentAdapters>,
    sensor_cameras: Res<SensorCameraConfigs>,
    mut priority: ResMut<SeriesFetchPriority>,
    mut cache: ResMut<TelemetryCache>,
    mut backfill: ResMut<BackfillState>,
    // Present in the interactive editor; absent in headless render-server.
    mut prefetch: Option<ResMut<VisiblePrefetchState>>,
) {
    let adapter_leaves: HashSet<ComponentId> = adapters.keys().copied().collect();
    let mut extras = viewport_adapter_component_ids(&path_reg, &adapter_leaves);
    extras.extend(sensor_camera_world_pos_ids(&sensor_cameras));
    let next = build_series_store_allowlist(
        enabled_fetch_component_ids(
            &graph_states,
            &line_3ds,
            &object_3ds,
            &monitors,
            &viewports,
            &vector_arrows,
            &eql_ctx,
        ),
        extras,
    );
    for id in priority.high.difference(&next).copied().collect::<Vec<_>>() {
        cache.remove_series(&id);
        backfill.clear_component(id);
        if let Some(ref mut prefetch) = prefetch {
            prefetch.in_flight.retain(|(cid, _, _)| *cid != id);
        }
    }
    priority.high = next;
}

/// Project SeriesStore samples into a plot `LineTree` for one element.
/// When `max_points` is set, stride-downsample so long windows stay GPU-friendly.
pub(crate) fn project_series_element_to_line(
    cache: &TelemetryCache,
    component_id: ComponentId,
    element_index: usize,
    range: &Range<Timestamp>,
    earliest: Timestamp,
    line: &mut LineTree<f32>,
    max_points: Option<usize>,
) -> usize {
    let Some(series) = cache.series(&component_id) else {
        return 0;
    };
    let count = series.range(range.start..range.end).count();
    if count == 0 {
        return 0;
    }
    let stride = max_points
        .map(|max| count.div_ceil(max).max(1))
        .unwrap_or(1);

    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut total = 0usize;
    for (i, (ts, val)) in series.range(range.start..range.end).enumerate() {
        if i % stride != 0 && i + 1 != count {
            continue;
        }
        let Some(elem) = val.get(element_index) else {
            continue;
        };
        timestamps.push(*ts);
        values.push(elem.as_f32());
        if timestamps.len() >= CHUNK_LEN {
            let n = timestamps.len();
            if let Some(chunk) = Chunk::from_iter(&timestamps, earliest, values.iter().copied()) {
                line.insert(chunk);
                total += n;
            }
            timestamps.clear();
            values.clear();
        }
    }
    if !timestamps.is_empty() {
        let n = timestamps.len();
        if let Some(chunk) = Chunk::from_iter(&timestamps, earliest, values.into_iter()) {
            line.insert(chunk);
            total += n;
        }
    }
    total
}

/// Fraction of `a`'s duration that overlaps `b` (0.0–1.0).
#[cfg(test)]
pub(crate) fn range_overlap_ratio(a: &Range<Timestamp>, b: &Range<Timestamp>) -> f64 {
    let overlap_start = a.start.0.max(b.start.0);
    let overlap_end = a.end.0.min(b.end.0);
    let overlap = overlap_end.saturating_sub(overlap_start).max(0);
    let a_len = a.end.0.saturating_sub(a.start.0).max(1);
    overlap as f64 / a_len as f64
}

/// Expand a selected window with prefetch margin for trailing / non-replay fetches.
#[cfg(test)]
pub(crate) fn expand_query_range_with_margin(
    selected_range: Range<Timestamp>,
    earliest: Timestamp,
    latest: Timestamp,
    trailing: bool,
) -> Range<Timestamp> {
    const MIN_PREFETCH_MARGIN: Duration = Duration::from_secs(2);
    if !trailing {
        return selected_range.start.max(earliest)..selected_range.end.min(latest);
    }
    let span = selected_range
        .end
        .0
        .saturating_sub(selected_range.start.0)
        .max(0);
    let margin = span.max(MIN_PREFETCH_MARGIN.as_micros() as i64);
    let start = Timestamp(
        selected_range
            .start
            .0
            .saturating_sub(margin)
            .max(earliest.0),
    );
    let end = Timestamp(selected_range.end.0.saturating_add(margin).min(latest.0));
    if start < end {
        start..end
    } else {
        selected_range.start.max(earliest)..selected_range.end.min(latest)
    }
}

#[cfg(test)]
fn data_query_range(
    selected_range: Range<Timestamp>,
    earliest: Timestamp,
    latest: Timestamp,
    replay_mode: bool,
    trailing: bool,
) -> Range<Timestamp> {
    // Trailing and replay: fetch selected ± margin (not the entire DB).
    // Non-trailing recorded: clamp selected to DB bounds only.
    let use_margin = trailing || replay_mode;
    expand_query_range_with_margin(selected_range, earliest, latest, use_margin)
}

#[cfg(test)]
fn next_range(
    mut current_range: Range<Timestamp>,
    component: &PlotDataComponent,
    lines: &Assets<Line>,
) -> Range<Timestamp> {
    if let Some((_, line)) = component.lines.first_key_value()
        && let Some(line) = lines.get(line)
        && let Some(chunk) = line.data.range_iter(current_range.clone()).next()
        && chunk.summary.start_timestamp <= current_range.start
    {
        current_range.start = next_timestamp(chunk.summary.end_timestamp);
    }
    current_range
}

#[cfg(test)]
fn next_timestamp(timestamp: Timestamp) -> Timestamp {
    Timestamp(timestamp.0.saturating_add(1))
}

#[derive(Asset, TypePath, Default)]
pub struct Line {
    pub label: String,
    pub data: LineTree<f32>,
    pub last_queried: Option<Instant>,
}

impl Line {
    fn maybe_compress_live(&mut self, earliest: Timestamp, settings: &CurveCompressSettings) {
        let should = if settings.archive_enabled {
            self.data.should_recompress(settings)
        } else {
            self.data.total_points() > settings.compress_after_total_points
        };
        if should {
            self.data.compress_time_value_hamann(earliest, settings);
        }
    }
}

#[derive(Asset, TypePath, Default)]
pub struct XYLine {
    pub label: String,
    pub x_shard_alloc: Option<BufferShardAlloc>,
    pub y_shard_alloc: Option<BufferShardAlloc>,
    pub x_values: Vec<SharedBuffer<f32, CHUNK_LEN>>,
    pub y_values: Vec<SharedBuffer<f32, CHUNK_LEN>>,
}

impl XYLine {
    pub fn point_count(&self) -> usize {
        self.x_values.iter().map(|c| c.cpu().len()).sum()
    }

    pub fn queue_load(&mut self, render_queue: &RenderQueue, render_device: &RenderDevice) {
        let x_shard_alloc = self.x_shard_alloc.get_or_insert_with(|| {
            BufferShardAlloc::with_nan_chunk(CHUNK_COUNT, CHUNK_LEN, render_device, render_queue)
        });
        let y_shard_alloc = self.y_shard_alloc.get_or_insert_with(|| {
            BufferShardAlloc::with_nan_chunk(CHUNK_COUNT, CHUNK_LEN, render_device, render_queue)
        });
        for buf in &mut self.x_values {
            buf.queue_load(render_queue, x_shard_alloc);
        }
        for buf in &mut self.y_values {
            buf.queue_load(render_queue, y_shard_alloc);
        }
    }

    pub fn write_to_index_buffer(
        &mut self,
        index_buffer: &Buffer,
        render_queue: &RenderQueue,
        pixel_width: usize,
    ) -> u32 {
        // Decimate to respect the fixed index buffer size (same pattern as timeseries)
        let desired_index_len = INDEX_BUFFER_LEN.min(pixel_width.max(1) * 4);
        let total_points: usize = self.x_values.iter().map(|c| c.cpu().len()).sum();
        if total_points == 0 {
            return 0;
        }
        let step = total_points.div_ceil(desired_index_len.max(1)).max(1);

        let mut view = render_queue
            .write_buffer_with(
                index_buffer,
                0,
                NonZeroU64::new((INDEX_BUFFER_LEN * 4) as u64).unwrap(),
            )
            .expect("no write buf");
        let mut view = view.slice(..);
        let mut written_u32s: u32 = 0;
        let mut global_index = 0usize;
        for buf in &mut self.x_values {
            let gpu = buf.gpu.lock();
            let Some(gpu) = gpu.as_ref() else {
                return 0;
            };
            let chunk = gpu.as_index_chunk::<f32>(buf.cpu().len());
            for (i, index) in chunk.into_index_iter().enumerate() {
                let absolute = global_index + i;
                if absolute.is_multiple_of(step) || absolute + 1 == total_points {
                    let Some(v) = try_append_u32(view, index) else {
                        return written_u32s;
                    };
                    view = v;
                    written_u32s += 1;
                }
            }
            global_index += buf.cpu().len();
        }

        written_u32s
    }

    pub fn plot_bounds(&self) -> PlotBounds {
        let (min_x, max_x) = match self.x_values.iter().flat_map(|c| c.cpu()).minmax() {
            MinMaxResult::NoElements => (0.0, 1.0),
            MinMaxResult::OneElement(x) => (*x - 1.0, *x + 1.0),
            MinMaxResult::MinMax(min, max) => (*min, *max),
        };
        let (min_y, max_y) = match self.y_values.iter().flat_map(|c| c.cpu()).minmax() {
            MinMaxResult::NoElements => (0.0, 1.0),
            MinMaxResult::OneElement(y) => (*y - 1.0, *y + 1.0),
            MinMaxResult::MinMax(min, max) => (*min, *max),
        };
        PlotBounds::new(min_x as f64, min_y as f64, max_x as f64, max_y as f64)
    }

    pub fn push_x_value(&mut self, value: f32) {
        if let Some(buf) = self.x_values.last_mut()
            && buf.cpu().len() < CHUNK_LEN
        {
            buf.push(value);
            return;
        }
        let mut buf = SharedBuffer::default();
        buf.push(value);
        self.x_values.push(buf);
    }

    pub fn push_y_value(&mut self, value: f32) {
        if let Some(buf) = self.y_values.last_mut()
            && buf.cpu().len() < CHUNK_LEN
        {
            buf.push(value);
            return;
        }
        let mut buf = SharedBuffer::default();
        buf.push(value);
        self.y_values.push(buf);
    }
}

#[derive(Debug, Clone)]
pub struct SharedBuffer<T, const N: usize> {
    cpu: Vec<T>,
    gpu: Arc<spin::Mutex<Option<BufferShard>>>,
    gpu_dirty: Arc<AtomicBool>,
}

impl<T, const N: usize> Default for SharedBuffer<T, N> {
    fn default() -> Self {
        Self {
            cpu: Vec::with_capacity(N),
            gpu: Default::default(),
            gpu_dirty: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl<T: IntoBytes + Immutable + Debug + Clone, const N: usize> SharedBuffer<T, N> {
    pub fn queue_load(&self, render_queue: &RenderQueue, buffer_alloc: &mut BufferShardAlloc) {
        if !self.gpu_dirty.load(atomic::Ordering::SeqCst) {
            return;
        }
        let mut gpu = self.gpu.lock();
        let gpu = if let Some(gpu) = &mut *gpu {
            gpu
        } else {
            let Some(buffer) = buffer_alloc.alloc() else {
                return;
            };
            gpu.insert(buffer)
        };
        render_queue.write_buffer_shard(gpu, self.cpu.as_bytes());
        self.gpu_dirty.store(false, atomic::Ordering::SeqCst);
    }

    pub fn push(&mut self, value: T) -> Option<()> {
        self.cpu.push(value);
        self.gpu_dirty.store(true, atomic::Ordering::SeqCst);
        Some(())
    }
}

impl<T, const N: usize> SharedBuffer<T, N> {
    pub fn cpu(&self) -> &[T] {
        &self.cpu
    }
}

#[derive(Clone, Debug)]
pub struct Chunk<D> {
    timestamps: Vec<Timestamp>,
    timestamps_float: SharedBuffer<f32, CHUNK_LEN>,
    data: SharedBuffer<D, CHUNK_LEN>,
    summary: ChunkSummary<D>,
}

impl<D: Immutable + IntoBytes + BoundOrd + Clone + Debug> Chunk<D> {
    pub fn from_iter(
        timestamps: &'_ [Timestamp],
        earliest_timestamp: Timestamp,
        values: impl Iterator<Item = D>,
    ) -> Option<Self> {
        let data = SharedBuffer {
            cpu: values.collect(),
            gpu: Default::default(),
            gpu_dirty: Arc::new(AtomicBool::new(true)),
        };
        let (min, max) = data
            .cpu()
            .iter()
            .fold((None::<D>, None::<D>), |(min_acc, max_acc), x| {
                let min_acc = Some(match min_acc {
                    Some(m) => m.min(x.clone()),
                    None => x.clone(),
                });
                let max_acc = Some(match max_acc {
                    Some(m) => m.max(x.clone()),
                    None => x.clone(),
                });
                (min_acc, max_acc)
            });

        let summary = ChunkSummary {
            len: timestamps.len(),
            start_timestamp: *timestamps.first()?,
            end_timestamp: *timestamps.last()?,
            min,
            max,
        };
        let timestamps_float = SharedBuffer {
            cpu: Vec::from_iter(
                timestamps
                    .iter()
                    .map(|&x| (x.0 - earliest_timestamp.0) as f32),
            ),
            gpu: Default::default(),
            gpu_dirty: Arc::new(AtomicBool::new(true)),
        };

        let timestamps = timestamps.to_vec();
        Some(Chunk {
            timestamps,
            timestamps_float,
            data,
            summary,
        })
    }

    pub fn from_initial_value(
        timestamp: Timestamp,
        earliest_timestamp: Timestamp,
        value: D,
    ) -> Self {
        let mut chunk = Chunk {
            timestamps: vec![],
            timestamps_float: SharedBuffer::default(),
            data: SharedBuffer::default(),
            summary: ChunkSummary {
                len: 0,
                start_timestamp: timestamp,
                end_timestamp: timestamp,
                min: Some(value.clone()),
                max: Some(value.clone()),
            },
        };
        chunk.push(timestamp, earliest_timestamp, value);
        chunk
    }

    pub fn queue_load(
        &self,
        render_queue: &RenderQueue,
        data_buffer_alloc: &mut BufferShardAlloc,
        timestamp_buffer_alloc: &mut BufferShardAlloc,
    ) {
        self.timestamps_float
            .queue_load(render_queue, timestamp_buffer_alloc);
        self.data.queue_load(render_queue, data_buffer_alloc);
    }

    pub fn push(
        &mut self,
        timestamp: Timestamp,
        earliest_timestamp: Timestamp,
        value: D,
    ) -> Option<()> {
        self.timestamps.push(timestamp);
        self.timestamps_float
            .push((timestamp.0 - earliest_timestamp.0) as f32)?;
        self.data.push(value.clone())?;
        self.summary.len += 1;
        self.summary.start_timestamp = self.summary.start_timestamp.min(timestamp);
        self.summary.end_timestamp = self.summary.end_timestamp.max(timestamp);
        let min = self.summary.min.get_or_insert_with(|| value.clone());
        *min = min.clone().min(value.clone());
        let max = self.summary.max.get_or_insert_with(|| value.clone());
        *max = max.clone().max(value.clone());

        Some(())
    }
}

#[derive(Clone, Debug)]
pub struct ChunkSummary<D> {
    pub len: usize,
    pub start_timestamp: Timestamp,
    pub end_timestamp: Timestamp,
    pub min: Option<D>,
    pub max: Option<D>,
}

impl<D> Default for ChunkSummary<D> {
    fn default() -> Self {
        ChunkSummary {
            len: 0,
            start_timestamp: Timestamp(i64::MAX),
            end_timestamp: Timestamp(i64::MIN),
            min: None,
            max: None,
        }
    }
}

impl<D: Clone + BoundOrd> ChunkSummary<D> {
    pub fn range(&self) -> RangeInclusive<Timestamp> {
        self.start_timestamp..=self.end_timestamp
    }
}

impl<D: Clone + BoundOrd> ChunkSummary<D> {
    fn add_summary(&mut self, summary: &Self) {
        self.start_timestamp = self.start_timestamp.min(summary.start_timestamp);
        self.end_timestamp = self.end_timestamp.max(summary.end_timestamp);
        self.len += summary.len;
        self.min = match (self.min.clone(), summary.min.clone()) {
            (None, None) => None,
            (None, Some(m)) | (Some(m), None) => Some(m),
            (Some(a), Some(b)) => Some(a.min(b)),
        };
        self.max = match (self.max.clone(), summary.max.clone()) {
            (None, None) => None,
            (None, Some(m)) | (Some(m), None) => Some(m),
            (Some(a), Some(b)) => Some(a.max(b)),
        };
    }
}

pub struct LineTree<D: Clone + BoundOrd> {
    tree: NoditMap<i64, nodit::Interval<i64>, Chunk<D>>,
    data_buffer_shard_alloc: Option<BufferShardAlloc>,
    timestamp_buffer_shard_alloc: Option<BufferShardAlloc>,
    /// Append-only archive of raw `(timestamp, value)` samples ingested live.
    ///
    /// When [`CurveCompressSettings::archive_enabled`] is true, [`PlotDataComponent::push_value`]
    /// pushes every accepted sample here in addition to the view (`tree`). The Hamann-Chen pass
    /// then reads from this archive and rebuilds the view, leaving the archive untouched. Because
    /// the archive is never decimated, running HC multiple times produces identical output
    /// (monotone quality) and parameter changes can be re-applied without loss.
    ///
    /// Populated only for live streaming via [`PlotDataComponent::push_value`].
    /// Historical samples projected from SeriesStore fill the view (`tree`) directly
    /// without updating this archive.
    raw_timestamps: Vec<Timestamp>,
    raw_values: Vec<D>,
    /// Archive length at the end of the last HC pass, for throttling decisions.
    last_hc_archive_len: usize,
    /// Wall-clock instant of the last HC pass, for time-based throttling.
    last_hc_instant: Option<std::time::Instant>,
    /// Bumped on [`Self::clear`] / view rebuild so GPU index caches can invalidate
    /// when LineTree contents change without a visible-range change.
    content_gen: u64,
}

impl<D: Clone + BoundOrd> Default for LineTree<D> {
    fn default() -> Self {
        Self {
            tree: Default::default(),
            data_buffer_shard_alloc: None,
            timestamp_buffer_shard_alloc: None,
            raw_timestamps: Vec::new(),
            raw_values: Vec::new(),
            last_hc_archive_len: 0,
            last_hc_instant: None,
            content_gen: 0,
        }
    }
}

pub const CHUNK_COUNT: usize = 0x400;
pub const CHUNK_LEN: usize = 0xC00;

impl<D: Clone + BoundOrd + Immutable + IntoBytes + Debug> LineTree<D> {
    pub fn range_iter(&self, range: Range<Timestamp>) -> impl Iterator<Item = &Chunk<D>> {
        self.tree
            .overlapping(ii(range.start.0, range.end.0))
            .map(|(_, v)| v)
    }

    pub fn update_last(&mut self, f: impl FnOnce(&mut Chunk<D>)) {
        if let Some(last) = self.tree.last_entry() {
            let (_, mut last) = last.remove_entry();
            f(&mut last);
            self.insert(last);
        }
    }

    pub fn insert(&mut self, chunk: Chunk<D>) {
        let _ = self.tree.insert_overwrite(
            ii(
                chunk.summary.start_timestamp.0,
                chunk.summary.end_timestamp.0,
            ),
            chunk,
        );
    }

    pub fn total_points(&self) -> usize {
        self.tree.iter().map(|(_, c)| c.summary.len).sum()
    }

    /// Identity for GPU index-cache invalidation when the view is rebuilt in place.
    pub fn content_gen(&self) -> u64 {
        self.content_gen
    }

    pub fn chunk_count(&self) -> usize {
        self.tree.iter().count()
    }

    /// Append `(ts, value)` to the raw archive. Rejects out-of-order timestamps
    /// (`ts <= last`) so the archive stays monotonic by construction. No-op if
    /// `enabled` is false — the caller gates on [`CurveCompressSettings::archive_enabled`].
    pub fn push_raw(&mut self, enabled: bool, ts: Timestamp, value: D) {
        if !enabled {
            return;
        }
        if let Some(last) = self.raw_timestamps.last()
            && ts <= *last
        {
            return;
        }
        self.raw_timestamps.push(ts);
        self.raw_values.push(value);
    }

    pub fn raw_len(&self) -> usize {
        self.raw_timestamps.len()
    }

    pub fn raw_slice(&self) -> (&[Timestamp], &[D]) {
        (&self.raw_timestamps, &self.raw_values)
    }

    pub fn clear_raw(&mut self) {
        self.raw_timestamps.clear();
        self.raw_values.clear();
        self.last_hc_archive_len = 0;
        self.last_hc_instant = None;
    }

    /// True when the archive has grown enough (samples or time) since the last HC pass to justify
    /// recomputing the view. Always false below [`CurveCompressSettings::compress_after_total_points`].
    pub fn should_recompress(&self, settings: &CurveCompressSettings) -> bool {
        if self.raw_len() <= settings.compress_after_total_points {
            return false;
        }
        let grew = self.raw_len().saturating_sub(self.last_hc_archive_len)
            >= settings.view_recompute_min_samples;
        if grew {
            return true;
        }
        match self.last_hc_instant {
            None => true,
            Some(t) => {
                t.elapsed()
                    >= std::time::Duration::from_millis(settings.view_recompute_min_interval_ms)
            }
        }
    }

    /// Record that an HC pass has just run (updates throttle bookkeeping).
    pub fn mark_compressed(&mut self) {
        self.last_hc_archive_len = self.raw_len();
        self.last_hc_instant = Some(std::time::Instant::now());
    }

    /// Approximate memory cost of the raw archive in bytes (timestamps + values).
    pub fn raw_archive_bytes(&self) -> usize {
        self.raw_timestamps.len() * (std::mem::size_of::<Timestamp>() + std::mem::size_of::<D>())
    }

    pub fn range_summary(&self, range: Range<Timestamp>) -> ChunkSummary<D> {
        self.range_iter(range)
            .fold(ChunkSummary::default(), |mut xs, x| {
                xs.add_summary(&x.summary);
                xs
            })
    }

    /// Compute robust percentile-based bounds that filter out extreme outliers.
    /// Returns (p1, p99) percentile values from the data, which excludes the most extreme 1% on each end.
    pub fn percentile_bounds(
        &self,
        range: Range<Timestamp>,
        p_low: f32,
        p_high: f32,
    ) -> Option<(D, D)>
    where
        D: PartialOrd + Copy,
    {
        // Collect all values from chunks in range
        let mut values: Vec<D> = self
            .range_iter(range)
            .flat_map(|chunk| chunk.data.cpu().iter().copied())
            .filter(|v| {
                // Filter out non-finite f32 values if D is f32
                // This is a bit of a hack but works for our use case
                let bytes = std::mem::size_of::<D>();
                if bytes == 4 {
                    // Likely f32
                    let v_f32: f32 = unsafe { std::mem::transmute_copy(v) };
                    v_f32.is_finite()
                } else {
                    true
                }
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        // Sort for percentile calculation
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = values.len();
        let low_idx = ((p_low / 100.0) * len as f32) as usize;
        let high_idx = ((p_high / 100.0) * len as f32) as usize;
        let high_idx = high_idx.min(len - 1);

        Some((values[low_idx], values[high_idx]))
    }

    pub fn last(&self) -> Option<&Chunk<D>> {
        self.tree.last_key_value().map(|(_k, v)| v)
    }

    pub fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &D)> {
        let (_, chunk) = self.tree.overlapping(ii(timestamp.0, timestamp.0)).next()?;
        let index = match chunk.timestamps.binary_search(&timestamp) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        Some((*chunk.timestamps.get(index)?, chunk.data.cpu().get(index)?))
    }

    /// Returns the latest sample timestamp **strictly less than** `target`.
    ///
    /// Used by `line_3d` rendering to snap the played/future split onto the
    /// previous sample boundary. Without this, the future segment can collapse
    /// to a single index per frame when `current_timestamp` falls between
    /// discrete sim ticks, which disappears from the shader (NaN-only draw)
    /// and shows up as a blink near the current position.
    pub fn last_timestamp_strictly_before(&self, target: Timestamp) -> Option<Timestamp> {
        let upper = target.0.checked_sub(1)?;
        let (_, chunk) = self.tree.overlapping(ii(i64::MIN, upper)).last()?;
        let probe = Timestamp(upper);
        let idx = match chunk.timestamps.binary_search(&probe) {
            Ok(i) => i,
            Err(i) => i.checked_sub(1)?,
        };
        chunk.timestamps.get(idx).copied()
    }

    /// Timestamp of the most recent sample stored in the view, or `None` when empty.
    /// Used by `line_3d` to tell apart "playhead between ticks" (snap the future
    /// segment back one sample to keep ≥ 2 indices and avoid a single-sample
    /// blink) from "playhead on the latest sample" (live follow, future must
    /// stay empty so no white tail overdraws the yellow trail).
    pub fn latest_sample_timestamp(&self) -> Option<Timestamp> {
        self.tree
            .last_key_value()
            .map(|(_, c)| c.summary.end_timestamp)
    }

    pub fn data_buffer_shard_alloc(&self) -> Option<&BufferShardAlloc> {
        self.data_buffer_shard_alloc.as_ref()
    }

    pub fn timestamp_buffer_shard_alloc(&self) -> Option<&BufferShardAlloc> {
        self.timestamp_buffer_shard_alloc.as_ref()
    }

    pub fn queue_load_range(
        &mut self,
        range: Range<Timestamp>,
        render_queue: &RenderQueue,
        render_device: &RenderDevice,
    ) {
        let data_buffer_alloc = self.data_buffer_shard_alloc.get_or_insert_with(|| {
            BufferShardAlloc::with_nan_chunk(CHUNK_COUNT, CHUNK_LEN, render_device, render_queue)
        });
        let timestamp_buffer_alloc = self.timestamp_buffer_shard_alloc.get_or_insert_with(|| {
            BufferShardAlloc::with_nan_chunk(CHUNK_COUNT, CHUNK_LEN, render_device, render_queue)
        });
        for (_, chunk) in self.tree.overlapping_mut(ii(range.start.0, range.end.0)) {
            chunk.queue_load(render_queue, data_buffer_alloc, timestamp_buffer_alloc);
        }
    }

    pub fn draw_index_count(&self, range: Range<Timestamp>) -> (usize, usize) {
        self.range_index_stats(range)
    }

    pub fn range_index_stats(&self, range: Range<Timestamp>) -> (usize, usize) {
        let mut chunks = 0;
        let mut total_count = 0;
        for chunk in self.range_iter(range.clone()) {
            let Some((start_offset, end_offset)) = chunk_visible_offsets(&chunk.timestamps, &range)
            else {
                continue;
            };
            chunks += 1;
            total_count += end_offset.saturating_sub(start_offset) + 6;
        }
        (chunks, total_count)
    }

    pub fn draw_index_chunk_iter(
        &self,
        range: Range<Timestamp>,
    ) -> impl Iterator<Item = IndexChunk> + '_ {
        self.range_iter(range.clone()).filter_map(move |c| {
            let gpu = c.data.gpu.lock();
            let gpu = gpu.as_ref()?;

            let mut index_chunk = gpu.as_index_chunk::<f32>(c.summary.len);

            // Clip to the visible timestamp interval within each chunk.
            let (start_offset, end_offset) = chunk_visible_offsets(&c.timestamps, &range)?;
            index_chunk.range.start = index_chunk.range.start.saturating_add(start_offset as u32);
            index_chunk.len = end_offset.saturating_sub(start_offset);

            #[cfg(debug_assertions)]
            {
                let timestamp = c.timestamps_float.gpu.lock();
                let timestamp = timestamp.as_ref()?;
                let mut timestamp_chunk = timestamp.as_index_chunk::<f32>(c.summary.len);
                timestamp_chunk.range.start = timestamp_chunk
                    .range
                    .start
                    .saturating_add(start_offset as u32);
                timestamp_chunk.len = end_offset.saturating_sub(start_offset);

                assert_eq!(index_chunk, timestamp_chunk);
            }

            Some(index_chunk)
        })
    }

    pub fn write_to_index_buffer(
        &self,
        index_buffer: &Buffer,
        render_queue: &RenderQueue,
        line_visible_range: Range<Timestamp>,
        pixel_width: usize,
    ) -> u32 {
        // No selected-span context: treat as a long window (pixel-faithful on clip).
        self.write_to_index_buffer_with_sampling_range(
            index_buffer,
            render_queue,
            line_visible_range,
            i64::MAX,
            pixel_width,
        )
    }

    pub fn write_to_index_buffer_with_sampling_range(
        &self,
        index_buffer: &Buffer,
        render_queue: &RenderQueue,
        line_visible_range: Range<Timestamp>,
        selected_span_micros: i64,
        pixel_width: usize,
    ) -> u32 {
        let (chunk_count, index_count) = self.range_index_stats(line_visible_range.clone());
        let step = index_sampling_step_for_selection(
            selected_span_micros,
            chunk_count,
            index_count,
            pixel_width,
        );
        self.write_to_index_buffer_with_step(index_buffer, render_queue, line_visible_range, step)
    }

    /// Count of `u32` indices written by [`Self::write_to_index_buffer_with_step`] for this range
    /// and step (must stay in sync with that loop, including when `step == 1`).
    ///
    /// Uses the same visibility clipping as [`Self::draw_index_chunk_iter`] but does **not**
    /// require GPU-resident chunks (counts from CPU timestamps + visible length only).
    pub fn count_strip_index_u32s(&self, line_visible_range: Range<Timestamp>, step: usize) -> u32 {
        let step = step.max(1);
        let mut n: u32 = 0;
        for c in self.range_iter(line_visible_range.clone()) {
            let Some((start_offset, end_offset)) =
                chunk_visible_offsets(&c.timestamps, &line_visible_range)
            else {
                continue;
            };
            let vis_len = end_offset.saturating_sub(start_offset);
            if vis_len == 0 {
                continue;
            }
            // `into_index_iter` length depends only on `len`; absolute indices match GPU path
            // after clip, but counts are identical for any `range.start` with sufficient span.
            let chunk = IndexChunk {
                range: 0..u32::MAX,
                len: vis_len,
            };
            n = n.saturating_add(1);
            let end = chunk.clone().into_index_iter().last();
            let mut index_iter = chunk.into_index_iter();
            let mut last_written: Option<u32> = None;
            if let Some(index) = index_iter.next() {
                n = n.saturating_add(1);
                last_written = Some(index);
            }
            for index in index_iter.step_by(step) {
                n = n.saturating_add(1);
                last_written = Some(index);
            }
            if let Some(end) = end
                && last_written != Some(end)
            {
                n = n.saturating_add(1);
            }
            n = n.saturating_add(1);
        }
        n
    }

    pub fn write_to_index_buffer_with_step(
        &self,
        index_buffer: &Buffer,
        render_queue: &RenderQueue,
        line_visible_range: Range<Timestamp>,
        step: usize,
    ) -> u32 {
        let mut view = render_queue
            .write_buffer_with(
                index_buffer,
                0,
                NonZeroU64::new((INDEX_BUFFER_LEN * 4) as u64).unwrap(),
            )
            .expect("no write buf");
        let mut view = view.slice(..);
        let mut written_u32s: u32 = 0;
        'chunks: for chunk in self.draw_index_chunk_iter(line_visible_range) {
            let Some(v) = try_append_u32(view, 0) else {
                break 'chunks;
            };
            view = v;
            written_u32s += 1;
            let end = chunk.clone().into_index_iter().last();
            let mut index_iter = chunk.into_index_iter();
            let mut last_written: Option<u32> = None;
            if let Some(index) = index_iter.next() {
                let Some(v) = try_append_u32(view, index) else {
                    break 'chunks;
                };
                view = v;
                written_u32s += 1;
                last_written = Some(index);
            }
            for index in index_iter.step_by(step) {
                let Some(v) = try_append_u32(view, index) else {
                    break 'chunks;
                };
                view = v;
                written_u32s += 1;
                last_written = Some(index);
            }
            if let Some(end) = end
                && last_written != Some(end)
            {
                let Some(v) = try_append_u32(view, end) else {
                    break 'chunks;
                };
                view = v;
                written_u32s += 1;
            }
            let Some(v) = try_append_u32(view, 0) else {
                break 'chunks;
            };
            view = v;
            written_u32s += 1;
        }
        written_u32s
    }

    pub fn garbage_collect(&mut self, line_visible_range: Range<Timestamp>) {
        let first_half =
            nodit::interval::ii(i64::MIN, line_visible_range.start.0.saturating_sub(1));
        let second_half = nodit::interval::ii(line_visible_range.end.0.saturating_add(1), i64::MAX);
        for interval in [first_half, second_half] {
            for (_range, chunk) in self.tree.remove_overlapping(interval) {
                if let Some(alloc) = &mut self.data_buffer_shard_alloc
                    && let Some(gpu) = chunk.data.gpu.lock().take()
                {
                    alloc.dealloc(gpu);
                }
                if let Some(alloc) = &mut self.timestamp_buffer_shard_alloc
                    && let Some(gpu) = chunk.timestamps_float.gpu.lock().take()
                {
                    alloc.dealloc(gpu);
                }
            }
        }
    }

    /// Drop all CPU/GPU chunks (full rebuild from SeriesStore).
    pub fn clear(&mut self) {
        self.content_gen = self.content_gen.wrapping_add(1);
        let full = nodit::interval::ii(i64::MIN, i64::MAX);
        for (_range, chunk) in self.tree.remove_overlapping(full) {
            if let Some(alloc) = &mut self.data_buffer_shard_alloc
                && let Some(gpu) = chunk.data.gpu.lock().take()
            {
                alloc.dealloc(gpu);
            }
            if let Some(alloc) = &mut self.timestamp_buffer_shard_alloc
                && let Some(gpu) = chunk.timestamps_float.gpu.lock().take()
            {
                alloc.dealloc(gpu);
            }
        }
        self.clear_raw();
    }
}

fn release_line_chunk_gpu(
    chunk: &mut Chunk<f32>,
    data_alloc: &mut Option<BufferShardAlloc>,
    ts_alloc: &mut Option<BufferShardAlloc>,
) {
    if let Some(alloc) = data_alloc
        && let Some(gpu) = chunk.data.gpu.lock().take()
    {
        alloc.dealloc(gpu);
        chunk.data.gpu_dirty.store(true, atomic::Ordering::SeqCst);
    }
    if let Some(alloc) = ts_alloc
        && let Some(gpu) = chunk.timestamps_float.gpu.lock().take()
    {
        alloc.dealloc(gpu);
        chunk
            .timestamps_float
            .gpu_dirty
            .store(true, atomic::Ordering::SeqCst);
    }
}

impl LineTree<f32> {
    pub fn flattened_time_series(&self) -> (Vec<Timestamp>, Vec<f32>) {
        let mut ts = Vec::new();
        let mut vs = Vec::new();
        for (_, chunk) in self.tree.iter() {
            for (&t, &v) in chunk.timestamps.iter().zip(chunk.data.cpu().iter()) {
                ts.push(t);
                vs.push(v);
            }
        }
        (ts, vs)
    }

    /// Rebuild the tree from `new_ts` / `new_v`, atomically, in `CHUNK_LEN` shards.
    ///
    /// Uses **build-then-swap**: a fresh tree is constructed off-band, swapped in
    /// with `mem::replace`, and only then are the old GPU shards released. Render
    /// frames therefore observe either the previous tree in full or the new tree
    /// in full — never an empty intermediate state. This is what kills the trail
    /// blink at high tick counts when [`compress_time_value_hamann`] fires.
    pub fn rebuild_from_time_value_pairs(
        &mut self,
        earliest: Timestamp,
        new_ts: &[Timestamp],
        new_v: &[f32],
    ) {
        let n = new_ts.len().min(new_v.len());

        if n == self.total_points()
            && n > 0
            && self
                .tree
                .iter()
                .next()
                .map(|(_, c)| c.summary.start_timestamp)
                == new_ts.first().copied()
            && self
                .tree
                .last_key_value()
                .map(|(_, c)| c.summary.end_timestamp)
                == new_ts.last().copied()
        {
            return;
        }

        self.content_gen = self.content_gen.wrapping_add(1);

        let mut new_tree: NoditMap<i64, nodit::Interval<i64>, Chunk<f32>> = NoditMap::default();
        let mut offset = 0usize;
        while offset < n {
            let end = (offset + CHUNK_LEN).min(n);
            if let Some(chunk) = Chunk::from_iter(
                &new_ts[offset..end],
                earliest,
                new_v[offset..end].iter().copied(),
            ) {
                let _ = new_tree.insert_overwrite(
                    ii(
                        chunk.summary.start_timestamp.0,
                        chunk.summary.end_timestamp.0,
                    ),
                    chunk,
                );
            }
            offset = end;
        }

        let old_tree = std::mem::replace(&mut self.tree, new_tree);
        for (_, mut chunk) in old_tree {
            release_line_chunk_gpu(
                &mut chunk,
                &mut self.data_buffer_shard_alloc,
                &mut self.timestamp_buffer_shard_alloc,
            );
        }
    }

    /// Hamann–Chen in `(t, y)`; optional [`CurveCompressSettings::keep_recent_fraction`] leaves a
    /// suffix of recent samples uncompressed.
    ///
    /// Source of truth depends on [`CurveCompressSettings::archive_enabled`]:
    /// - `true` (non-destructive) — reads from [`LineTree::raw_slice`]. Repeated passes produce
    ///   identical output (monotone quality); archive is never touched.
    /// - `false` (legacy destructive) — reads from [`Self::flattened_time_series`] (the already
    ///   decimated view), so successive passes compound decimation. Kept as rollback switch.
    pub fn compress_time_value_hamann(
        &mut self,
        earliest: Timestamp,
        settings: &CurveCompressSettings,
    ) {
        // Own the source slice: the archive path needs a borrow of `self` that conflicts with the
        // later `&mut self` in `rebuild_from_time_value_pairs`. The clone of ~1-10 MB is cheap
        // compared to the HC pass itself (50-100 ms).
        let (ts_all, v_all): (Vec<Timestamp>, Vec<f32>) = if settings.archive_enabled {
            let (t, v) = self.raw_slice();
            (t.to_vec(), v.to_vec())
        } else {
            self.flattened_time_series()
        };
        let n = ts_all.len();
        if n < 3 {
            return;
        }

        let keep = recent_tail_keep_count(n, settings.keep_recent_fraction);
        let split = n.saturating_sub(keep);

        if settings.keep_recent_fraction > 0.0 && split >= 3 {
            let old_ts = &ts_all[..split];
            let old_v = &v_all[..split];
            let recent_ts = &ts_all[split..];
            let recent_v = &v_all[split..];
            let budget = settings.compress_to_points.saturating_sub(keep).max(2);
            let target_old = budget.min(split).max(2);
            let t_rel: Vec<f32> = old_ts.iter().map(|t| (t.0 - earliest.0) as f32).collect();
            let idx = select_time_value_indices(&t_rel, old_v, target_old);
            if idx.len() < 2 {
                return;
            }
            let mut new_ts: Vec<Timestamp> = idx.iter().map(|&i| old_ts[i]).collect();
            let mut new_v: Vec<f32> = idx.iter().map(|&i| old_v[i]).collect();
            if let (Some(tl), Some(tr)) = (new_ts.last(), recent_ts.first())
                && *tl == *tr
            {
                new_ts.pop();
                new_v.pop();
            }
            new_ts.extend_from_slice(recent_ts);
            new_v.extend_from_slice(recent_v);
            self.rebuild_from_time_value_pairs(earliest, &new_ts, &new_v);
            self.mark_compressed();
            return;
        }

        let target = settings.compress_to_points.max(2).min(n);
        let t_rel: Vec<f32> = ts_all.iter().map(|t| (t.0 - earliest.0) as f32).collect();
        let idx = select_time_value_indices(&t_rel, &v_all, target);
        if idx.len() < 2 {
            return;
        }
        let new_ts: Vec<Timestamp> = idx.iter().map(|&i| ts_all[i]).collect();
        let new_v: Vec<f32> = idx.iter().map(|&i| v_all[i]).collect();
        self.rebuild_from_time_value_pairs(earliest, &new_ts, &new_v);
        self.mark_compressed();
    }
}

fn recent_tail_keep_count(n: usize, keep_recent_fraction: f32) -> usize {
    if n == 0 || keep_recent_fraction <= 0.0 {
        return 0;
    }
    let f = keep_recent_fraction.clamp(0.0_f32, 0.999_f32);
    let keep = (n as f32 * f).ceil() as usize;
    keep.min(n.saturating_sub(1))
}

fn chunk_visible_offsets(
    timestamps: &[Timestamp],
    range: &Range<Timestamp>,
) -> Option<(usize, usize)> {
    let start_offset = timestamps.partition_point(|&t| t < range.start);
    let end_offset = timestamps.partition_point(|&t| t <= range.end);
    (end_offset > start_offset).then_some((start_offset, end_offset))
}

pub fn index_sampling_step(chunk_count: usize, index_count: usize, pixel_width: usize) -> usize {
    let desired_index_len = INDEX_BUFFER_LEN.min(pixel_width.max(1) * 4);
    // Per-chunk index overhead: leading sentinel, first point, strided interior, last point,
    // trailing sentinel (see `write_to_index_buffer_with_step`).
    const PER_CHUNK_OVERHEAD: usize = 6;
    let overhead = PER_CHUNK_OVERHEAD.saturating_mul(chunk_count.max(1));
    let divisor = desired_index_len.saturating_sub(overhead);
    if divisor > 0 {
        return index_count.div_ceil(divisor).max(1);
    }
    // Many chunks: `2 * chunk_count` alone can exceed the index budget — never fall back to step 1.
    index_count.div_ceil(desired_index_len.max(1)).max(1)
}

/// For short accuracy windows (≤30 s), always draw every sample (`step = 1`).
/// Longer windows use a pixel-faithful stride on the visible clip.
pub fn index_sampling_step_for_selection(
    selected_span_micros: i64,
    zoomed_chunk_count: usize,
    zoomed_index_count: usize,
    pixel_width: usize,
) -> usize {
    if selected_span_micros <= crate::SHORT_WINDOW_ACCURACY_MICROS {
        1
    } else {
        index_sampling_step(zoomed_chunk_count, zoomed_index_count, pixel_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_wkt::ComponentValue;

    #[test]
    fn next_timestamp_advances_by_one_microsecond() {
        assert_eq!(next_timestamp(Timestamp(41)), Timestamp(42));
    }

    #[test]
    fn next_timestamp_saturates_at_i64_max() {
        assert_eq!(next_timestamp(Timestamp(i64::MAX)), Timestamp(i64::MAX));
    }

    #[test]
    fn next_range_skips_the_last_timestamp_of_the_existing_chunk() {
        let mut lines = Assets::<Line>::default();
        let handle = lines.add(Line::default());
        let chunk = Chunk::from_iter(
            &[Timestamp(10), Timestamp(20)],
            Timestamp(0),
            [1.0_f32, 2.0_f32].into_iter(),
        )
        .expect("chunk should exist");
        {
            let mut line = lines.get_mut(&handle).expect("line asset should exist");
            line.data.insert(chunk);
        }

        let mut component = PlotDataComponent::new("test", vec![]);
        component.lines.insert(0, handle);

        let next = next_range(Timestamp(10)..Timestamp(30), &component, &lines);
        assert_eq!(next.start, Timestamp(21));
        assert_eq!(next.end, Timestamp(30));
    }

    #[test]
    fn replay_query_range_uses_selected_plus_margin() {
        let range = data_query_range(
            Timestamp(5_000_000)..Timestamp(5_000_100),
            Timestamp(0),
            Timestamp(20_000_000),
            true,
            false,
        );
        // span 100µs → margin max(100, 2s) = 2s
        assert_eq!(range, Timestamp(3_000_000)..Timestamp(7_000_100));
    }

    #[test]
    fn non_replay_query_range_stays_visible_window_when_not_trailing() {
        let range = data_query_range(
            Timestamp(50)..Timestamp(75),
            Timestamp(10),
            Timestamp(100),
            false,
            false,
        );
        assert_eq!(range, Timestamp(50)..Timestamp(75));
    }

    #[test]
    fn trailing_query_range_expands_with_margin_clamped_to_db() {
        // 5s window → margin max(5s, 2s) = 5s
        let selected = Timestamp(10_000_000)..Timestamp(15_000_000);
        let range =
            expand_query_range_with_margin(selected, Timestamp(0), Timestamp(20_000_000), true);
        assert_eq!(range.start, Timestamp(5_000_000));
        assert_eq!(range.end, Timestamp(20_000_000));
    }

    #[test]
    fn request_key_quantum_collides_sub_100ms_starts() {
        let a = floor_ts_quantum(Timestamp(40_050_000), REQUEST_KEY_QUANTUM_MICROS);
        let b = floor_ts_quantum(Timestamp(40_090_000), REQUEST_KEY_QUANTUM_MICROS);
        assert_eq!(a, b);
        assert_eq!(a, Timestamp(40_000_000));
    }

    #[test]
    fn range_overlap_ratio_full_overlap_is_one() {
        let a = Timestamp(0)..Timestamp(1_000_000);
        let b = Timestamp(0)..Timestamp(1_000_000);
        assert!((range_overlap_ratio(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn range_overlap_ratio_half_slide_is_below_half() {
        // Previous overview: [0, 20min]. Current: [15min, 35min] → 25% of prev remains.
        let prev = Timestamp(0)..Timestamp(1_200_000_000);
        let curr = Timestamp(900_000_000)..Timestamp(2_100_000_000);
        let ratio = range_overlap_ratio(&prev, &curr);
        assert!(ratio < 0.5, "ratio={ratio}");
        assert!(ratio > 0.2, "ratio={ratio}");
    }

    #[test]
    fn range_overlap_ratio_no_overlap_is_zero() {
        let a = Timestamp(0)..Timestamp(100);
        let b = Timestamp(200)..Timestamp(300);
        assert_eq!(range_overlap_ratio(&a, &b), 0.0);
    }

    #[test]
    fn range_index_stats_uses_cpu_timestamps() {
        let mut tree = LineTree::<f32>::default();
        let chunk = Chunk::from_iter(
            &[Timestamp(10), Timestamp(20), Timestamp(30), Timestamp(40)],
            Timestamp(0),
            [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32].into_iter(),
        )
        .expect("chunk should exist");
        tree.insert(chunk);

        let (chunk_count, index_count) = tree.range_index_stats(Timestamp(15)..Timestamp(35));
        assert_eq!(chunk_count, 1);
        assert_eq!(index_count, 8);
    }

    #[test]
    fn index_sampling_step_matches_index_budget() {
        assert_eq!(index_sampling_step(1, 100, 100), 1);
        assert_eq!(index_sampling_step(1, 1_000, 100), 3);
    }

    #[test]
    fn series_store_allowlist_unions_plots_and_adapters() {
        let plot = ComponentId(10);
        let adapter = ComponentId(20);
        let ids = build_series_store_allowlist([plot].into_iter().collect(), [adapter]);
        assert!(ids.contains(&plot));
        assert!(ids.contains(&adapter));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn series_store_allowlist_adapters_alone_still_subscribe() {
        let adapter = ComponentId(7);
        let ids = build_series_store_allowlist(HashSet::new(), [adapter]);
        assert_eq!(ids, [adapter].into_iter().collect());
    }

    #[test]
    fn viewport_adapter_ids_expand_pair_paths_by_leaf() {
        use impeller2_wkt::ComponentPath;
        let leaf = ComponentId::new("world_pos");
        let pair = ComponentId::new("ball_1.world_pos");
        let mut path_reg = ComponentPathRegistry::default();
        path_reg
            .0
            .insert(pair, ComponentPath::from_name("ball_1.world_pos"));
        // Unrelated component must not be pulled in.
        path_reg.0.insert(
            ComponentId::new("ball_1.world_vel"),
            ComponentPath::from_name("ball_1.world_vel"),
        );
        let adapter_leaves = [leaf].into_iter().collect();
        let ids = viewport_adapter_component_ids(&path_reg, &adapter_leaves);
        assert!(ids.contains(&leaf));
        assert!(ids.contains(&pair));
        assert!(!ids.contains(&ComponentId::new("ball_1.world_vel")));
    }

    #[test]
    fn line_tree_content_gen_bumps_on_clear() {
        let mut tree = LineTree::<f32>::default();
        assert_eq!(tree.content_gen(), 0);
        let chunk =
            Chunk::from_iter(&[Timestamp(1)], Timestamp(0), [1.0f32].into_iter()).expect("chunk");
        tree.insert(chunk);
        tree.clear();
        assert_eq!(tree.content_gen(), 1);
        tree.clear();
        assert_eq!(tree.content_gen(), 2);
    }

    #[test]
    fn monitor_component_name_maps_to_component_id() {
        let name = "PCDUMESSAGE.ADC_12V";
        assert_eq!(
            ComponentId::new(name),
            ComponentId::new("PCDUMESSAGE.ADC_12V")
        );
    }

    #[test]
    fn sensor_camera_world_pos_ids_from_configs() {
        use crate::sensor_camera::SensorCameraConfigs;
        use impeller2_wkt::SensorCameraConfig;
        let configs = SensorCameraConfigs(vec![SensorCameraConfig {
            entity_name: "cam_ball_a".into(),
            camera_name: "scene_cam".into(),
            width: 64,
            height: 64,
            fov_degrees: 90.0,
            near: 0.1,
            far: 100.0,
            pos_offset: [0.0, 0.0, 0.0],
            rot_offset: [0.0, 0.0, 0.0],
            format: "rgba8".into(),
            effect: String::new(),
            effect_params: Default::default(),
            create_frustum: false,
            show_ellipsoids: false,
            frustums_color: Default::default(),
            projection_color: Default::default(),
            frustums_thickness: 0.01,
            fps: 30.0,
        }]);
        let ids = sensor_camera_world_pos_ids(&configs);
        assert!(ids.contains(&ComponentId::new("cam_ball_a.world_pos")));
    }

    #[test]
    fn allowlist_unions_consumers_and_extras() {
        let consumer = ComponentId::new("CONTROLMESSAGE.ACC_CMD_BODY");
        let extra = ComponentId::new("ball_1.world_pos");
        let ids = build_series_store_allowlist([consumer].into_iter().collect(), [extra]);
        assert!(ids.contains(&consumer));
        assert!(ids.contains(&extra));
    }

    #[test]
    fn short_window_sampling_step_is_always_one() {
        // Truth mode: ≤30 s never strides, even with many points / narrow widgets.
        assert_eq!(
            index_sampling_step_for_selection(5_000_000, 2, 8_000, 400),
            1
        );
        assert_eq!(
            index_sampling_step_for_selection(30_000_000, 10, 30_000, 200),
            1
        );
        // Zoomed sub-range still step 1 under a short selection.
        assert_eq!(index_sampling_step_for_selection(5_000_000, 1, 200, 400), 1);
    }

    #[test]
    fn long_window_sampling_step_uses_visible_clip() {
        let zoomed_step = index_sampling_step(1, 200, 400);
        let long = index_sampling_step_for_selection(60_000_000, 1, 200, 400);
        assert_eq!(long, zoomed_step);
        let dense = index_sampling_step_for_selection(60_000_000, 2, 8_000, 400);
        assert!(dense > 1, "dense={dense}");
    }

    #[test]
    fn short_window_1khz_5s_fits_index_buffer_at_step_one() {
        // Synthetic 1 kHz × 5 s (~5000 samples), matching SITL IMU rate.
        let mut tree = LineTree::<f32>::default();
        let n = 5_000;
        let ts: Vec<Timestamp> = (0..n).map(|i| Timestamp(i as i64 * 1_000)).collect();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        // Insert in CHUNK_LEN-sized pieces like production.
        let mut offset = 0;
        while offset < n {
            let end = (offset + CHUNK_LEN).min(n);
            let chunk = Chunk::from_iter(
                &ts[offset..end],
                Timestamp(0),
                vals[offset..end].iter().copied(),
            )
            .expect("chunk");
            tree.insert(chunk);
            offset = end;
        }
        let full = Timestamp(0)..Timestamp((n as i64 - 1) * 1_000);
        let zoomed = Timestamp(1_000_000)..Timestamp(2_000_000); // 1 s sub-range
        let full_count = tree.count_strip_index_u32s(full.clone(), 1);
        let zoom_count = tree.count_strip_index_u32s(zoomed.clone(), 1);
        assert!(
            (full_count as usize) <= INDEX_BUFFER_LEN,
            "full_count={full_count} INDEX_BUFFER_LEN={INDEX_BUFFER_LEN}"
        );
        assert!(zoom_count < full_count);
        assert_eq!(index_sampling_step_for_selection(5_000_000, 2, n, 400), 1);
        assert_eq!(
            index_sampling_step_for_selection(5_000_000, 1, 1_000, 400),
            1
        );
    }

    #[test]
    fn series_store_coverage_contains_marked_range() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("test.cov");
        cache.mark_covered(id, Timestamp(100), Timestamp(500));
        assert!(cache.is_covered(&id, &(Timestamp(100)..Timestamp(500))));
        assert!(cache.is_covered(&id, &(Timestamp(200)..Timestamp(400))));
        assert!(!cache.is_covered(&id, &(Timestamp(50)..Timestamp(150))));
        cache.mark_covered(id, Timestamp(500), Timestamp(800));
        assert!(cache.is_covered(&id, &(Timestamp(100)..Timestamp(800))));
    }

    #[test]
    fn series_store_refuses_cover_to_i64_max() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("test.cov.max");
        cache.mark_covered(id, Timestamp(100), Timestamp(i64::MAX));
        assert!(!cache.is_covered(&id, &(Timestamp(100)..Timestamp(200))));
        assert_eq!(
            cache.sample_count_in_range(&id, &(Timestamp(0)..Timestamp(i64::MAX))),
            0
        );
    }

    #[test]
    fn series_store_sample_span_in_range() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("test.span");
        cache.insert(
            id,
            Timestamp(1_000_000),
            ComponentValue::F64(nox::array![1.0f64].to_dyn()),
        );
        cache.insert(
            id,
            Timestamp(2_000_000),
            ComponentValue::F64(nox::array![2.0f64].to_dyn()),
        );
        let span = cache
            .sample_span_in_range(&id, &(Timestamp(0)..Timestamp(3_000_000)))
            .expect("span");
        assert_eq!(span.0, Timestamp(1_000_000));
        assert_eq!(span.1, Timestamp(2_000_000));
        assert_eq!(
            cache.sample_count_in_range(&id, &(Timestamp(0)..Timestamp(3_000_000))),
            2
        );
    }

    #[test]
    fn project_from_store_rebuilds_visible_window_only() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("test.project");
        // 10s of 100 Hz data (0..10_000_000 micros)
        for i in 0..1000 {
            let ts = Timestamp(i * 10_000);
            cache.insert(id, ts, ComponentValue::F64(nox::array![i as f64].to_dyn()));
        }
        let earliest = Timestamp(0);
        let mut line = LineTree::<f32>::default();
        // Tip window: last 5s
        let tip = Timestamp(5_000_000)..Timestamp(10_000_000);
        let n = project_series_element_to_line(&cache, id, 0, &tip, earliest, &mut line, None);
        assert_eq!(n, 500);
        assert_eq!(line.total_points(), 500);

        // Jump to start: first 5s — clear and rebuild
        line.clear();
        let start = Timestamp(0)..Timestamp(5_000_000);
        let n2 = project_series_element_to_line(&cache, id, 0, &start, earliest, &mut line, None);
        assert_eq!(n2, 500);
        assert_eq!(line.total_points(), 500);
        // Store still holds full history
        assert_eq!(cache.total_sample_count(), 1000);
    }

    #[test]
    fn project_stride_caps_long_window() {
        let mut cache = TelemetryCache::default();
        let id = ComponentId::new("test.stride");
        for i in 0..10_000 {
            cache.insert(
                id,
                Timestamp(i),
                ComponentValue::F64(nox::array![i as f64].to_dyn()),
            );
        }
        let mut line = LineTree::<f32>::default();
        let range = Timestamp(0)..Timestamp(10_000);
        let n = project_series_element_to_line(
            &cache,
            id,
            0,
            &range,
            Timestamp(0),
            &mut line,
            Some(100),
        );
        assert!(n <= 101, "n={n}");
        assert!(n >= 100, "n={n}");
    }

    #[test]
    fn sync_skips_clear_when_store_empty_and_range_unchanged() {
        // Mirrors the tip-window case: live LineTree has points, SeriesStore has
        // not caught up yet — sync must not wipe the line.
        let mut line = LineTree::<f32>::default();
        let chunk = Chunk::from_iter(
            &[Timestamp(1_000_000), Timestamp(2_000_000)],
            Timestamp(0),
            [1.0_f32, 2.0_f32].into_iter(),
        )
        .expect("chunk");
        line.insert(chunk);
        assert_eq!(line.total_points(), 2);

        let cache = TelemetryCache::default();
        let id = ComponentId::new("empty.yet");
        let range = Timestamp(0)..Timestamp(5_000_000);
        let n =
            project_series_element_to_line(&cache, id, 0, &range, Timestamp(0), &mut line, None);
        assert_eq!(n, 0);
        // Caller must not clear on n==0 when range unchanged — points remain.
        assert_eq!(line.total_points(), 2);
    }

    #[test]
    fn index_sampling_step_many_chunks_never_returns_one() {
        let step = index_sampling_step(5000, 1_000_000, 1920);
        assert!(step > 1, "step={step}");
    }

    #[test]
    fn recent_tail_keep_count_zero_fraction_is_zero() {
        assert_eq!(recent_tail_keep_count(100, 0.0), 0);
    }

    #[test]
    fn recent_tail_keep_count_respects_fraction_and_caps_at_n_minus_one() {
        assert_eq!(recent_tail_keep_count(100, 0.2), 20);
        assert_eq!(recent_tail_keep_count(5, 0.2), 1);
        assert_eq!(recent_tail_keep_count(100, 1.0), 99);
    }

    #[test]
    fn count_strip_index_u32s_decreases_with_larger_step() {
        let mut tree = LineTree::<f32>::default();
        let ts: Vec<Timestamp> = (0i64..200).map(Timestamp).collect();
        let vals: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let chunk = Chunk::from_iter(&ts, Timestamp(0), vals.into_iter()).expect("chunk");
        tree.insert(chunk);
        let range = Timestamp(0)..Timestamp(199);
        let c1 = tree.count_strip_index_u32s(range.clone(), 1);
        let c8 = tree.count_strip_index_u32s(range.clone(), 8);
        assert!(c8 <= c1, "c1={c1} c8={c8}");
        assert!(c1 > 0);
    }

    #[test]
    fn count_strip_index_u32s_step_one_matches_sentinels_plus_vertices() {
        let mut tree = LineTree::<f32>::default();
        let ts: Vec<Timestamp> = (0i64..10).map(Timestamp).collect();
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunk = Chunk::from_iter(&ts, Timestamp(0), vals.into_iter()).expect("chunk");
        tree.insert(chunk);
        let c = tree.count_strip_index_u32s(Timestamp(0)..Timestamp(10), 1);
        assert_eq!(
            c, 12,
            "leading 0 + 10 indices + trailing 0 (no duplicate last)"
        );
    }

    // === Archive + view (non-destructive HC) tests ===

    fn archive_settings(
        archive_enabled: bool,
        threshold: usize,
        target: usize,
    ) -> CurveCompressSettings {
        CurveCompressSettings {
            enabled: true,
            compress_after_total_points: threshold,
            compress_to_points: target,
            keep_recent_fraction: 0.0,
            archive_enabled,
            view_recompute_min_samples: 1,
            view_recompute_min_interval_ms: 0,
        }
    }

    #[test]
    fn push_raw_enforces_monotonicity() {
        let mut tree = LineTree::<f32>::default();
        tree.push_raw(true, Timestamp(10), 1.0);
        tree.push_raw(true, Timestamp(20), 2.0);
        tree.push_raw(true, Timestamp(15), 99.0);
        tree.push_raw(true, Timestamp(20), 99.0);
        tree.push_raw(true, Timestamp(30), 3.0);
        let (ts, vs) = tree.raw_slice();
        assert_eq!(ts, &[Timestamp(10), Timestamp(20), Timestamp(30)]);
        assert_eq!(vs, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn push_raw_disabled_is_noop() {
        let mut tree = LineTree::<f32>::default();
        tree.push_raw(false, Timestamp(10), 1.0);
        tree.push_raw(false, Timestamp(20), 2.0);
        assert_eq!(tree.raw_len(), 0);
    }

    #[test]
    fn clear_raw_resets_state() {
        let mut tree = LineTree::<f32>::default();
        tree.push_raw(true, Timestamp(10), 1.0);
        tree.mark_compressed();
        assert_eq!(tree.raw_len(), 1);
        assert!(tree.last_hc_instant.is_some());
        tree.clear_raw();
        assert_eq!(tree.raw_len(), 0);
        assert_eq!(tree.last_hc_archive_len, 0);
        assert!(tree.last_hc_instant.is_none());
    }

    #[test]
    fn rebuild_from_time_value_pairs_preserves_raw_archive() {
        let mut tree = LineTree::<f32>::default();
        for i in 0..20 {
            tree.push_raw(true, Timestamp(i * 10), i as f32);
        }
        let raw_before = tree.raw_slice();
        let (ts_before, vs_before) = (raw_before.0.to_vec(), raw_before.1.to_vec());
        let new_ts = vec![Timestamp(0), Timestamp(100), Timestamp(190)];
        let new_v = vec![0.0, 10.0, 19.0];
        tree.rebuild_from_time_value_pairs(Timestamp(0), &new_ts, &new_v);
        let (ts_after, vs_after) = tree.raw_slice();
        assert_eq!(
            ts_after,
            ts_before.as_slice(),
            "archive timestamps must be untouched"
        );
        assert_eq!(
            vs_after,
            vs_before.as_slice(),
            "archive values must be untouched"
        );
    }

    #[test]
    fn should_recompress_below_threshold_is_false() {
        let mut tree = LineTree::<f32>::default();
        let settings = archive_settings(true, 1000, 100);
        tree.push_raw(true, Timestamp(1), 0.0);
        assert!(!tree.should_recompress(&settings));
    }

    #[test]
    fn should_recompress_above_threshold_triggers_once_no_prior_pass() {
        let mut tree = LineTree::<f32>::default();
        let settings = archive_settings(true, 5, 3);
        for i in 0..10 {
            tree.push_raw(true, Timestamp(i), i as f32);
        }
        assert!(tree.should_recompress(&settings));
    }

    #[test]
    fn should_recompress_throttled_by_sample_delta() {
        let mut tree = LineTree::<f32>::default();
        let settings = CurveCompressSettings {
            view_recompute_min_samples: 100,
            view_recompute_min_interval_ms: 60_000,
            ..archive_settings(true, 5, 3)
        };
        for i in 0..10 {
            tree.push_raw(true, Timestamp(i), i as f32);
        }
        tree.mark_compressed();
        tree.push_raw(true, Timestamp(11), 11.0);
        assert!(
            !tree.should_recompress(&settings),
            "1 new sample, 100 required + recent pass -> should skip"
        );
    }

    #[test]
    fn should_recompress_fires_when_interval_elapsed_even_without_enough_samples() {
        let mut tree = LineTree::<f32>::default();
        let settings = CurveCompressSettings {
            view_recompute_min_samples: 100,
            view_recompute_min_interval_ms: 0,
            ..archive_settings(true, 5, 3)
        };
        for i in 0..10 {
            tree.push_raw(true, Timestamp(i), i as f32);
        }
        tree.mark_compressed();
        tree.push_raw(true, Timestamp(11), 11.0);
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(
            tree.should_recompress(&settings),
            "interval=0 means elapsed check always fires"
        );
    }

    #[test]
    fn hc_pass_is_deterministic_on_stable_archive() {
        let mut tree = LineTree::<f32>::default();
        let settings = archive_settings(true, 5, 20);
        for i in 0..200 {
            let t = Timestamp(i);
            let v = (i as f32 * 0.05).sin();
            tree.push_raw(true, t, v);
        }
        tree.compress_time_value_hamann(Timestamp(0), &settings);
        let view_1 = tree.flattened_time_series();
        tree.compress_time_value_hamann(Timestamp(0), &settings);
        let view_2 = tree.flattened_time_series();
        assert_eq!(
            view_1.0, view_2.0,
            "two successive passes must pick identical timestamps"
        );
        assert_eq!(
            view_1.1, view_2.1,
            "two successive passes must pick identical values"
        );
    }

    #[test]
    fn hc_pass_is_monotone_across_many_passes() {
        let mut tree = LineTree::<f32>::default();
        let settings = archive_settings(true, 5, 20);
        for i in 0..200 {
            tree.push_raw(true, Timestamp(i), (i as f32 * 0.05).sin());
        }
        tree.compress_time_value_hamann(Timestamp(0), &settings);
        let reference = tree.flattened_time_series();
        for _ in 0..10 {
            tree.compress_time_value_hamann(Timestamp(0), &settings);
        }
        let after = tree.flattened_time_series();
        assert_eq!(
            reference.0, after.0,
            "10 passes with no new samples must not shift the view"
        );
        assert_eq!(reference.1, after.1);
    }

    #[test]
    fn hc_pass_disabled_archive_reads_from_view_like_before() {
        let mut tree = LineTree::<f32>::default();
        let settings = archive_settings(false, 5, 20);
        for i in 0..200 {
            let t = Timestamp(i);
            let v = (i as f32 * 0.05).sin();
            // Use insert path so the view gets populated without touching the archive
            let chunk = Chunk::from_iter(&[t], Timestamp(0), [v].into_iter()).expect("chunk");
            tree.insert(chunk);
        }
        assert_eq!(
            tree.raw_len(),
            0,
            "archive_enabled=false keeps archive empty"
        );
        tree.compress_time_value_hamann(Timestamp(0), &settings);
        let (ts, _) = tree.flattened_time_series();
        assert!(!ts.is_empty() && ts.len() <= settings.compress_to_points);
    }

    #[test]
    fn raw_archive_bytes_scales_with_sample_count() {
        let mut tree = LineTree::<f32>::default();
        assert_eq!(tree.raw_archive_bytes(), 0);
        for i in 0..1_000 {
            tree.push_raw(true, Timestamp(i), i as f32);
        }
        // Timestamp(i64) = 8 bytes + f32 = 4 bytes => 12 bytes per sample.
        assert_eq!(tree.raw_archive_bytes(), 1_000 * 12);
    }
}

fn try_append_u32(view: wgpu::WriteOnly<'_, [u8]>, val: u32) -> Option<wgpu::WriteOnly<'_, [u8]>> {
    if view.len() < size_of::<u32>() {
        return None;
    }
    let (mut head, rest) = view.split_at(size_of::<u32>());
    head.copy_from_slice(&val.to_le_bytes());
    Some(rest)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexChunk {
    range: Range<u32>,
    len: usize,
}

impl IndexChunk {
    pub fn into_index_iter(self) -> impl Iterator<Item = u32> {
        self.range.take(self.len)
    }
}

pub trait BoundOrd {
    fn min(self, val: Self) -> Self;
    fn max(self, val: Self) -> Self;
}

impl BoundOrd for f32 {
    fn min(self, val: f32) -> Self {
        self.min(val)
    }

    fn max(self, val: f32) -> Self {
        self.max(val)
    }
}

pub struct BufferShardAlloc {
    buffer: Buffer,
    chunk_size: usize,
    free_map: RoaringBitmap,
}

impl BufferShardAlloc {
    pub fn with_nan_chunk(
        chunks: usize,
        chunk_len: usize,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) -> Self {
        let mut this = Self::new::<f32>(chunks + 1, chunk_len, render_device);
        let shard = this.alloc().expect("couldn't alloc nan");
        render_queue.write_buffer_shard(&shard, &f32::NAN.to_le_bytes());
        this
    }

    pub fn new<T: Sized>(chunks: usize, chunk_len: usize, render_device: &RenderDevice) -> Self {
        let chunk_size = size_of::<T>() * chunk_len;
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Buffer<{}, {}>", type_name::<T>(), chunks)),
            size: (chunk_size * chunks) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut free_map = RoaringBitmap::new();
        for i in 0..chunks {
            free_map.insert(i as u32);
        }
        Self {
            buffer,
            free_map,
            chunk_size,
        }
    }

    pub fn alloc(&mut self) -> Option<BufferShard> {
        let i = self.free_map.iter().next()?;
        self.free_map.remove(i);
        let i = i as usize;
        let start = (i * self.chunk_size) as u64;
        let end = ((i + 1) * self.chunk_size) as u64;
        Some(BufferShard {
            buffer: self.buffer.clone(),
            range: start..end,
        })
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dealloc(&mut self, shard: BufferShard) {
        debug_assert_eq!(
            self.buffer.id(),
            shard.buffer.id(),
            "trying to dealloc shard from wrong buffer",
        );
        let i = shard.range.start as usize / self.chunk_size;
        self.free_map.insert(i as u32);
    }
}

#[derive(Clone, Debug)]
pub struct BufferShard {
    buffer: Buffer,
    range: Range<u64>,
}

impl BufferShard {
    pub fn as_slice(&self) -> BufferSlice<'_> {
        self.buffer.slice(self.range.clone())
    }

    pub fn as_index_chunk<D>(&self, len: usize) -> IndexChunk {
        let start = (self.range.start / size_of::<D>() as u64) as u32;
        let end = (self.range.end / size_of::<D>() as u64) as u32;

        IndexChunk {
            range: start..end,
            len,
        }
    }
}

pub trait RenderQueueExt {
    fn write_buffer_shard(&self, buffer_shard: &BufferShard, data: &[u8]);
}

impl RenderQueueExt for RenderQueue {
    fn write_buffer_shard(&self, buffer_shard: &BufferShard, data: &[u8]) {
        let len = data
            .len()
            .min((buffer_shard.range.end - buffer_shard.range.start) as usize);
        let data = &data[..len];
        self.write_buffer(&buffer_shard.buffer, buffer_shard.range.start, data);
    }
}
