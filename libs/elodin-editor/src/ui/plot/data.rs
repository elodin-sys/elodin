use bevy::asset::Asset;
use bevy::log::warn_once;
use bevy::prelude::{DetectChanges, InRef, Res, ResMut};
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::{Commands, Query},
    prelude::Resource,
};
use bevy_render::render_resource::{Buffer, BufferDescriptor, BufferSlice, BufferUsages};
use bevy_render::renderer::{RenderDevice, RenderQueue};
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, PrimType, Timestamp};
use impeller2_bevy::{
    CommandsExt, ComponentSchemaRegistry, ComponentValueMap, EntityMap, PacketGrantR,
    PacketHandlerInput, PacketHandlers,
};
use impeller2_wkt::{
    CurrentTimestamp, EarliestTimestamp, GetTimeSeries, LastUpdated, PlotOverviewQuery,
};
use itertools::{Itertools, MinMaxResult};
use nodit::NoditMap;
use nodit::interval::ii;
use roaring::bitmap::RoaringBitmap;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

use std::any::type_name;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicBool};
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::ui::plot::gpu::INDEX_BUFFER_LEN;
use crate::{SelectedTimeRange, TimeRangeBehavior};
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
}

impl Default for CurveCompressSettings {
    fn default() -> Self {
        let cap = CHUNK_COUNT.saturating_mul(CHUNK_LEN);
        Self {
            enabled: true,
            compress_after_total_points: cap.saturating_mul(3) / 4,
            compress_to_points: cap / 2,
            keep_recent_fraction: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub element_names: Vec<String>,
    pub lines: BTreeMap<usize, Handle<Line>>,
    request_states: HashMap<Timestamp, RequestState>,
    /// Whether the overview data has been requested for each line element
    overview_requested: HashMap<usize, bool>,
}

#[derive(Clone, Debug)]
enum RequestState {
    Requested(Instant),
    Returned {
        len: usize,
        last_timestamp: Option<Timestamp>,
    },
}

impl PlotDataComponent {
    pub fn new(component_label: impl ToString, element_names: Vec<String>) -> Self {
        Self {
            label: component_label.to_string(),
            element_names,
            lines: BTreeMap::new(),
            request_states: HashMap::new(),
            overview_requested: HashMap::new(),
        }
    }

    pub fn push_value(
        &mut self,
        component_view: ComponentView<'_>,
        assets: &mut Assets<Line>,
        timestamp: Timestamp,
        earliest_timestamp: Timestamp,
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
            let line = assets.get_mut(line.id()).expect("missing line asset");
            // Only accept data at timestamps beyond all existing data for this
            // line.  The FixedRate stream sends a snapshot at the current
            // playback position each frame; skipping timestamps already covered
            // prevents corrupting historical data loaded by GetTimeSeries.
            if let Some(last) = line.data.last() {
                if timestamp <= last.summary.end_timestamp {
                    continue;
                }
                if last.timestamps.len() < CHUNK_LEN {
                    line.data.update_last(|c| {
                        c.push(timestamp, earliest_timestamp, new_value);
                    });
                    continue;
                }
            }
            let new_chunk = Chunk::from_initial_value(timestamp, earliest_timestamp, new_value);
            line.data.insert(new_chunk);
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
                    plot_data.push_value(view, &mut lines, timestamp, earliest_timestamp.0);
                }
            },
        ) {
            warn_once!(?err, "graph sink failed");
        }
        maybe_compress_all_graph_lines(
            &mut collected_graph_data,
            &mut lines,
            earliest_timestamp.0,
            &curve_compress,
        );
    }
}

/// Optionally rewrites oversized [`Line`] / [`LineTree`] series using **Hamann–Chen** sampling
/// (`hamann-chen-line`, algorithm structure from Shane Celis’s C# gist linked on
/// [`CurveCompressSettings`]).
///
/// Does nothing when [`CurveCompressSettings::enabled`] is false. Otherwise, for each plot
/// component: if there are **three** lines and timestamps match across them, tries joint 3D
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
            let Some(line) = lines.get_mut(line_handle) else {
                continue;
            };
            line.maybe_compress_live(earliest, settings);
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
    if lx.data.total_points() <= settings.compress_after_total_points {
        return false;
    }
    let (tsx, vx) = lx.data.flattened_time_series();
    let (tsy, vy) = ly.data.flattened_time_series();
    let (tsz, vz) = lz.data.flattened_time_series();
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
    let Some(xl) = lines.get_mut(hx) else {
        return false;
    };
    xl.data
        .rebuild_from_time_value_pairs(earliest, &new_ts, &new_x);
    let Some(yl) = lines.get_mut(hy) else {
        return false;
    };
    yl.data
        .rebuild_from_time_value_pairs(earliest, &new_ts, &new_y);
    let Some(zl) = lines.get_mut(hz) else {
        return false;
    };
    zl.data
        .rebuild_from_time_value_pairs(earliest, &new_ts, &new_z);
    true
}

pub fn setup_pkt_handler(mut packet_handlers: ResMut<PacketHandlers>, mut commands: Commands) {
    let sys = commands.register_system(pkt_handler);
    packet_handlers.0.push(sys);
}

fn process_time_series<T>(
    time_series_buf: &[u8],
    timestamps: &[Timestamp],
    len: usize,
    plot_data: &mut PlotDataComponent,
    lines: &mut Assets<Line>,
    earliest_timestamp: Timestamp,
) where
    T: AsF32 + TryFromBytes + Immutable,
{
    let Ok(data) = <[T]>::try_ref_from_bytes(time_series_buf) else {
        return;
    };
    for i in 0..len {
        let line = plot_data.lines.entry(i).or_insert_with(|| {
            let label = format!("[{i}]");
            lines.add(Line {
                label,
                ..Default::default()
            })
        });
        let values = data.iter().skip(i).step_by(len).map(|v| v.as_f32());
        let Some(line) = lines.get_mut(line) else {
            continue;
        };
        let Some(chunk) = Chunk::from_iter(timestamps, earliest_timestamp, values) else {
            continue;
        };
        line.data.insert(chunk);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn handle_time_series(
    InRef(pkt): InRef<OwnedPacket<PacketGrantR>>,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    entity_map: Res<EntityMap>,
    component_values: Query<&ComponentValueMap>,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
    mut range: Range<Timestamp>,
    entity_id: ComponentId,
    component_id: ComponentId,
    earliest_timestamp: Res<EarliestTimestamp>,
    schema_reg: Res<ComponentSchemaRegistry>,
) {
    match pkt {
        OwnedPacket::Msg(_) => {}
        OwnedPacket::Table(_) => {}
        OwnedPacket::TimeSeries(time_series) => {
            let Some((len, prim_type)) = entity_map
                .get(&entity_id)
                .and_then(|entity| component_values.get(*entity).ok())
                .and_then(|component_value_map| component_value_map.get(&component_id))
                .map(|current_value| {
                    (
                        current_value.shape().iter().fold(1usize, |xs, &x| x * xs),
                        current_value.prim_type(),
                    )
                })
                .or_else(|| {
                    let schema = schema_reg.0.get(&component_id)?;
                    Some((
                        schema.shape().iter().fold(1usize, |xs, &x| x * xs),
                        schema.prim_type(),
                    ))
                })
            else {
                return;
            };
            let Ok(timestamps) = time_series.timestamps() else {
                return;
            };
            let Ok(buf) = time_series.data() else {
                return;
            };
            let Some(plot_data) = collected_graph_data.get_component_mut(&component_id) else {
                return;
            };
            match prim_type {
                PrimType::U8 => process_time_series::<u8>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::U16 => process_time_series::<u16>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::U32 => process_time_series::<u32>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::U64 => process_time_series::<u64>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::I8 => process_time_series::<i8>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::I16 => process_time_series::<i16>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::I32 => process_time_series::<i32>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::I64 => process_time_series::<i64>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::Bool => process_time_series::<bool>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::F32 => process_time_series::<f32>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
                PrimType::F64 => process_time_series::<f64>(
                    buf,
                    timestamps,
                    len,
                    plot_data,
                    &mut lines,
                    earliest_timestamp.0,
                ),
            }
            plot_data.request_states.insert(
                range.start,
                RequestState::Returned {
                    len: timestamps.len(),
                    last_timestamp: timestamps.last().copied(),
                },
            );
            let Some(last_timestamp) = timestamps.last() else {
                return;
            };
            if last_timestamp >= &range.end {
                return;
            }
            // GetTimeSeries returns the end bound inclusively, so reusing the
            // terminal timestamp overlaps the next page with the current chunk.
            range.start = next_timestamp(*last_timestamp);

            if range.start >= range.end {
                return;
            }

            if timestamps.len() < CHUNK_LEN {
                return;
            }
            let range = next_range(range, plot_data, &lines);

            let packet_id = fastrand::u16(..).to_le_bytes();
            let start = range.start;
            let end = range.end;
            let msg = GetTimeSeries {
                id: packet_id,
                range,
                component_id,
                limit: Some(CHUNK_LEN),
            };
            commands.send_req_with_handler(
                msg,
                packet_id,
                move |pkt: InRef<OwnedPacket<PacketGrantR>>,
                      collected_graph_data: ResMut<CollectedGraphData>,
                      entity_map: Res<EntityMap>,
                      component_values: Query<&ComponentValueMap>,
                      lines: ResMut<Assets<Line>>,
                      earliest_timestamp: Res<EarliestTimestamp>,
                      schema_reg: Res<ComponentSchemaRegistry>,
                      commands: Commands| {
                    handle_time_series(
                        pkt,
                        collected_graph_data,
                        entity_map,
                        component_values,
                        lines,
                        commands,
                        start..end,
                        entity_id,
                        component_id,
                        earliest_timestamp,
                        schema_reg,
                    );
                },
            );
        }
    }
}

pub fn queue_timestamp_read(
    selected_range: Res<SelectedTimeRange>,
    mut commands: Commands,
    mut graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<Line>>,
    earliest_timestamp: Res<EarliestTimestamp>,
    latest_timestamp: Res<LastUpdated>,
    replay_mode: Option<Res<crate::ReplayMode>>,
) {
    if selected_range.0.end.0 == i64::MIN
        || selected_range.0.start.0 == i64::MAX
        || selected_range.0.start.0 == i64::MIN
        || selected_range.0.end.0 == i64::MAX
    {
        return;
    }

    let query_range = data_query_range(
        selected_range.0.clone(),
        earliest_timestamp.0,
        latest_timestamp.0,
        replay_mode.is_some(),
    );
    if query_range.start >= query_range.end {
        return;
    }

    // Simple volume-based decision: use overview for large time ranges (> 10 minutes)
    // This covers two use cases:
    // 1. Live telemetry: data span is small (< 10 minutes) -> direct chunked loading
    // 2. Historical datasets: data span is large (> 10 minutes) -> use LTTB overview first
    const TEN_MINUTES_MICROS: i64 = 600_000_000; // 10 minutes in microseconds
    let range_duration = query_range.end.0.saturating_sub(query_range.start.0);
    let use_overview = range_duration > TEN_MINUTES_MICROS;

    for (&component_id, component) in graph_data.components.iter_mut() {
        let mut line = component
            .lines
            .first_key_value()
            .and_then(|(_k, v)| lines.get_mut(v));

        if let Some(last_queried) = line.as_ref().and_then(|l| l.last_queried.as_ref())
            && last_queried.elapsed() <= Duration::from_millis(250)
        {
            continue;
        }

        if use_overview {
            // Request overview for each element that hasn't been requested yet
            let num_elements = component.element_names.len().max(1);

            for element_index in 0..num_elements {
                if component
                    .overview_requested
                    .get(&element_index)
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }

                // Mark as requested
                component.overview_requested.insert(element_index, true);

                let packet_id = fastrand::u16(..).to_le_bytes();

                let query = PlotOverviewQuery {
                    id: packet_id,
                    component_id,
                    range: query_range.clone(),
                    max_points: OVERVIEW_MAX_POINTS as u32,
                    element_index,
                };
                let earliest = earliest_timestamp.0;

                commands.send_req_with_handler(
                    query,
                    packet_id,
                    move |pkt: InRef<OwnedPacket<PacketGrantR>>,
                          mut collected_graph_data: ResMut<CollectedGraphData>,
                          mut lines: ResMut<Assets<Line>>| {
                        handle_overview_response(
                            pkt,
                            &mut collected_graph_data,
                            &mut lines,
                            component_id,
                            element_index,
                            earliest,
                        );
                    },
                );
            }
            // When using overview mode, skip the regular gap-filling logic
            // Overview provides downsampled data for the full range at once
            continue;
        }

        let mut process_range = |mut range: Range<Timestamp>| {
            let packet_id = fastrand::u16(..).to_le_bytes();

            loop {
                match component.request_states.get(&range.start) {
                    // Rerequest chunks if we have not received a response for 10 seconds
                    Some(RequestState::Requested(time))
                        if time.elapsed() > Duration::from_secs(10) =>
                    {
                        break;
                    }

                    // When the visible range grows (replay mode), extend the tail
                    // from the last returned timestamp instead of replacing the
                    // whole partial chunk from its original start.
                    Some(RequestState::Returned {
                        len,
                        last_timestamp,
                    }) if *len < CHUNK_LEN
                        && last_timestamp
                            .map(|t| t < query_range.end)
                            .unwrap_or_default() =>
                    {
                        let Some(last_timestamp) = *last_timestamp else {
                            return;
                        };
                        range.start = next_timestamp(last_timestamp);
                        if range.start >= range.end {
                            return;
                        }
                    }
                    // Skip the chunk if it was already requested
                    Some(RequestState::Returned { .. }) | Some(RequestState::Requested(_)) => {
                        return;
                    }
                    None => break,
                }
            }

            component
                .request_states
                .insert(range.start, RequestState::Requested(Instant::now()));

            let start = range.start;
            let end = range.end;
            let msg = GetTimeSeries {
                id: packet_id,
                range: range.clone(),
                component_id,
                limit: Some(CHUNK_LEN),
            };
            commands.send_req_with_handler(
                msg,
                packet_id,
                move |pkt: InRef<OwnedPacket<PacketGrantR>>,
                      collected_graph_data: ResMut<CollectedGraphData>,
                      entity_map: Res<EntityMap>,
                      component_values: Query<&ComponentValueMap>,
                      lines: ResMut<Assets<Line>>,
                      earliest_timestamp: Res<EarliestTimestamp>,
                      schema_reg: Res<ComponentSchemaRegistry>,
                      commands: Commands| {
                    handle_time_series(
                        pkt,
                        collected_graph_data,
                        entity_map,
                        component_values,
                        lines,
                        commands,
                        start..end,
                        component_id,
                        component_id,
                        earliest_timestamp,
                        schema_reg,
                    );
                },
            );
        };
        if let Some(line) = line.as_mut() {
            line.last_queried = Some(Instant::now());
            line.data
                .tree
                .gaps_trimmed(nodit::interval::ie(query_range.start.0, query_range.end.0))
                .map(|i| Timestamp(i.start())..Timestamp(i.end()))
                .for_each(process_range)
        } else {
            process_range(query_range.clone());
        }
    }
}

fn data_query_range(
    selected_range: Range<Timestamp>,
    earliest: Timestamp,
    latest: Timestamp,
    replay_mode: bool,
) -> Range<Timestamp> {
    if replay_mode && earliest < latest {
        earliest..latest
    } else {
        selected_range
    }
}

/// Handle the response from a PlotOverviewQuery.
/// This inserts downsampled data into the LineTree for quick rendering.
fn handle_overview_response(
    pkt: InRef<OwnedPacket<PacketGrantR>>,
    collected_graph_data: &mut CollectedGraphData,
    lines: &mut Assets<Line>,
    component_id: ComponentId,
    element_index: usize,
    earliest_timestamp: Timestamp,
) {
    let OwnedPacket::TimeSeries(time_series) = &*pkt else {
        return;
    };

    let Ok(timestamps) = time_series.timestamps() else {
        return;
    };
    let Ok(buf) = time_series.data() else {
        return;
    };

    if timestamps.is_empty() {
        return;
    }

    let Some(plot_data) = collected_graph_data.get_component_mut(&component_id) else {
        return;
    };

    // Get or create the line for this element
    let line = plot_data.lines.entry(element_index).or_insert_with(|| {
        let label = plot_data
            .element_names
            .get(element_index)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("[{element_index}]"));
        lines.add(Line {
            label,
            ..Default::default()
        })
    });

    let Some(line) = lines.get_mut(line) else {
        return;
    };

    // The response contains f32 values
    let Ok(values) = <[f32]>::try_ref_from_bytes(buf) else {
        return;
    };

    // Create a chunk with the overview data
    if let Some(chunk) = Chunk::from_iter(timestamps, earliest_timestamp, values.iter().copied()) {
        line.data.insert(chunk);
    }
}

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
        if self.data.total_points() > settings.compress_after_total_points {
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
        let mut view = &mut view[..];
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

pub trait AsF32 {
    fn as_f32(&self) -> f32;
}
macro_rules! impl_as_f32 {
    ($($t:ty),*) => {
        $(
            impl AsF32 for $t {
                fn as_f32(&self) -> f32 { *self as f32 }
            }
        )*
    }
}

impl_as_f32!(u8, u16, u32, u64, i8, i16, i32, i64, f64);

impl AsF32 for f32 {
    fn as_f32(&self) -> f32 {
        *self
    }
}

impl AsF32 for bool {
    fn as_f32(&self) -> f32 {
        if *self { 1.0 } else { 0.0 }
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
}

impl<D: Clone + BoundOrd> Default for LineTree<D> {
    fn default() -> Self {
        Self {
            tree: Default::default(),
            data_buffer_shard_alloc: None,
            timestamp_buffer_shard_alloc: None,
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

    pub fn chunk_count(&self) -> usize {
        self.tree.iter().count()
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
        self.write_to_index_buffer_with_sampling_range(
            index_buffer,
            render_queue,
            line_visible_range.clone(),
            line_visible_range,
            pixel_width,
        )
    }

    pub fn write_to_index_buffer_with_sampling_range(
        &self,
        index_buffer: &Buffer,
        render_queue: &RenderQueue,
        line_visible_range: Range<Timestamp>,
        sampling_range: Range<Timestamp>,
        pixel_width: usize,
    ) -> u32 {
        let (chunk_count, index_count) = self.range_index_stats(sampling_range);
        let step = index_sampling_step(chunk_count, index_count, pixel_width);
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
        let mut view = &mut view[..];
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
        for (range, chunk) in self
            .tree
            .overlapping(first_half)
            .chain(self.tree.overlapping(second_half))
        {
            if line_visible_range.contains(&Timestamp(range.start()))
                || line_visible_range.contains(&Timestamp(range.end()))
            {
                continue;
            }
            if let Some(alloc) = &mut self.data_buffer_shard_alloc
                && let Some(gpu) = chunk.data.gpu.lock().take()
            {
                alloc.dealloc(gpu);
                chunk.data.gpu_dirty.store(true, atomic::Ordering::SeqCst);
            }
            if let Some(alloc) = &mut self.timestamp_buffer_shard_alloc
                && let Some(gpu) = chunk.timestamps_float.gpu.lock().take()
            {
                alloc.dealloc(gpu);
                chunk
                    .timestamps_float
                    .gpu_dirty
                    .store(true, atomic::Ordering::SeqCst);
            }
        }
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
    pub fn compress_time_value_hamann(
        &mut self,
        earliest: Timestamp,
        settings: &CurveCompressSettings,
    ) {
        let (ts_all, v_all) = self.flattened_time_series();
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let line = lines.get_mut(&handle).expect("line asset should exist");
        let chunk = Chunk::from_iter(
            &[Timestamp(10), Timestamp(20)],
            Timestamp(0),
            [1.0_f32, 2.0_f32].into_iter(),
        )
        .expect("chunk should exist");
        line.data.insert(chunk);

        let mut component = PlotDataComponent::new("test", vec![]);
        component.lines.insert(0, handle);

        let next = next_range(Timestamp(10)..Timestamp(30), &component, &lines);
        assert_eq!(next.start, Timestamp(21));
        assert_eq!(next.end, Timestamp(30));
    }

    #[test]
    fn replay_query_range_uses_full_available_extent() {
        let range = data_query_range(
            Timestamp(50)..Timestamp(75),
            Timestamp(10),
            Timestamp(100),
            true,
        );
        assert_eq!(range, Timestamp(10)..Timestamp(100));
    }

    #[test]
    fn non_replay_query_range_stays_visible_window() {
        let range = data_query_range(
            Timestamp(50)..Timestamp(75),
            Timestamp(10),
            Timestamp(100),
            false,
        );
        assert_eq!(range, Timestamp(50)..Timestamp(75));
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
}

fn try_append_u32(view: &mut [u8], val: u32) -> Option<&mut [u8]> {
    if view.len() < size_of::<u32>() {
        return None;
    }
    view[..size_of::<u32>()].copy_from_slice(&val.to_le_bytes());
    Some(&mut view[size_of::<u32>()..])
}

pub fn collect_garbage(
    behavior: Res<TimeRangeBehavior>,
    selected_range: Res<SelectedTimeRange>,
    graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<Line>>,
) {
    if !behavior.is_changed() {
        return;
    }
    for component in graph_data.components.values() {
        for line in component.lines.values() {
            let Some(line) = lines.get_mut(line) else {
                continue;
            };
            line.data.garbage_collect(selected_range.0.clone());
        }
    }
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
