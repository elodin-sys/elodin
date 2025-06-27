use bevy::asset::Asset;
use bevy::log::warn;
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
    CommandsExt, ComponentSchemaRegistry, ComponentValueMap, CurrentStreamId, EntityMap,
    PacketGrantR, PacketHandlerInput, PacketHandlers,
};
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp, GetTimeSeries};
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

use super::PlotBounds;

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub element_names: Vec<String>,
    pub lines: BTreeMap<usize, Handle<Line>>,
    request_states: HashMap<Timestamp, RequestState>,
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
            if let Some(last) = line.data.last() {
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
    current_stream_id: Res<CurrentStreamId>,
) {
    let mut tick = *tick;
    if let OwnedPacket::Table(table) = packet {
        if table.id == current_stream_id.packet_id() {
            return;
        }
        if let Err(err) = table.sink(registry, &mut tick) {
            warn!(?err, "tick sick failed");
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
            warn!(?err, "graph sink failed");
        }
    }
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
            plot_data.request_states.insert(
                range.start,
                RequestState::Returned {
                    len: timestamps.len(),
                    last_timestamp: timestamps.last().copied(),
                },
            );
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
            let Some(last_timestamp) = timestamps.last() else {
                return;
            };
            if last_timestamp >= &range.end {
                return;
            }
            range.start = *last_timestamp;

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
) {
    if selected_range.0.end.0 == i64::MIN || selected_range.0.start.0 == i64::MAX {
        return;
    }
    for (&component_id, component) in graph_data.components.iter_mut() {
        let mut line = component
            .lines
            .first_key_value()
            .and_then(|(_k, v)| lines.get_mut(v));

        if let Some(last_queried) = line.as_ref().and_then(|l| l.last_queried.as_ref()) {
            if last_queried.elapsed() <= Duration::from_millis(250) {
                continue;
            }
        }

        let mut process_range = |range: Range<Timestamp>| {
            let packet_id = fastrand::u16(..).to_le_bytes();

            match component.request_states.get(&range.start) {
                // Rerequest chunks if we have not received a response for 10 seconds
                Some(RequestState::Requested(time)) if time.elapsed() > Duration::from_secs(10) => {
                }

                // Recheck if we have a potentially incomplete chunk.
                // We first check if the chunk is not full, indicating that it existed at the end of available data
                // or at the end of the previously requested time range. We check if it was the chunk at the end of the previously requested
                // range by checking if the last time stamp is less than the end of the range
                Some(RequestState::Returned {
                    len,
                    last_timestamp,
                }) if *len < CHUNK_LEN
                    && last_timestamp
                        .map(|t| t < selected_range.0.end)
                        .unwrap_or_default() => {}
                // Skip the chunk if it was already requested
                Some(RequestState::Returned { .. }) | Some(RequestState::Requested(_)) => {
                    return;
                }
                None => {}
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
                .gaps_trimmed(nodit::interval::ie(
                    selected_range.0.start.0,
                    selected_range.0.end.0,
                ))
                .map(|i| Timestamp(i.start())..Timestamp(i.end()))
                .for_each(process_range)
        } else {
            process_range(selected_range.0.clone());
        }
    }
}

fn next_range(
    mut current_range: Range<Timestamp>,
    component: &PlotDataComponent,
    lines: &Assets<Line>,
) -> Range<Timestamp> {
    if let Some((_, line)) = component.lines.first_key_value() {
        if let Some(line) = lines.get(line) {
            if let Some(chunk) = line.data.range_iter(current_range.clone()).next() {
                if chunk.summary.start_timestamp <= current_range.start {
                    current_range.start = chunk.summary.end_timestamp;
                }
            }
        }
    }
    current_range
}

#[derive(Asset, TypePath, Default)]
pub struct Line {
    pub label: String,
    pub data: LineTree<f32>,
    pub last_queried: Option<Instant>,
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
    ) -> u32 {
        let mut count = 0;
        let mut view = render_queue
            .write_buffer_with(
                index_buffer,
                0,
                NonZeroU64::new((INDEX_BUFFER_LEN * 4) as u64).unwrap(),
            )
            .expect("no write buf");
        let mut view = &mut view[..];
        for buf in &mut self.x_values {
            let gpu = buf.gpu.lock();
            let Some(gpu) = gpu.as_ref() else { return 0 };
            let chunk = gpu.as_index_chunk::<f32>(buf.cpu().len());
            for index in chunk.into_index_iter() {
                view = append_u32(view, index);
            }
            count += buf.cpu().len() as u32;
        }
        count
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
        if let Some(buf) = self.x_values.last_mut() {
            if buf.cpu().len() < CHUNK_LEN {
                buf.push(value);
                return;
            }
        }
        let mut buf = SharedBuffer::default();
        buf.push(value);
        self.x_values.push(buf);
    }

    pub fn push_y_value(&mut self, value: f32) {
        if let Some(buf) = self.y_values.last_mut() {
            if buf.cpu().len() < CHUNK_LEN {
                buf.push(value);
                return;
            }
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
        let min = data.cpu().iter().fold(None, |xs: Option<D>, x| {
            if let Some(xs) = xs.clone() {
                Some(xs.min(x.clone()))
            } else {
                Some(x.clone())
            }
        });
        let max = data.cpu().iter().fold(None, |xs: Option<D>, x| {
            if let Some(xs) = xs.clone() {
                Some(xs.max(x.clone()))
            } else {
                Some(x.clone())
            }
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

    pub fn range_summary(&self, range: Range<Timestamp>) -> ChunkSummary<D> {
        self.range_iter(range)
            .fold(ChunkSummary::default(), |mut xs, x| {
                xs.add_summary(&x.summary);
                xs
            })
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
        let mut chunks = 0;
        let mut total_count = 0;
        for chunk in self.draw_index_chunk_iter(range) {
            chunks += 1;
            total_count += chunk.into_index_iter().count() + 6;
        }
        (chunks, total_count)
    }

    pub fn draw_index_chunk_iter(
        &self,
        range: Range<Timestamp>,
    ) -> impl Iterator<Item = IndexChunk> + '_ {
        self.range_iter(range).filter_map(|c| {
            let gpu = c.data.gpu.lock();
            let gpu = gpu.as_ref()?;

            let index_chunk = gpu.as_index_chunk::<f32>(c.summary.len);

            #[cfg(debug_assertions)]
            {
                let timestamp = c.timestamps_float.gpu.lock();
                let timestamp = timestamp.as_ref()?;

                assert_eq!(index_chunk, timestamp.as_index_chunk::<f32>(c.summary.len));
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
        let mut view = render_queue
            .write_buffer_with(
                index_buffer,
                0,
                NonZeroU64::new((INDEX_BUFFER_LEN * 4) as u64).unwrap(),
            )
            .expect("no write buf");
        let (chunk_count, index_count) = self.draw_index_count(line_visible_range.clone());
        let desired_index_len = INDEX_BUFFER_LEN.min(pixel_width * 4);
        let step = (index_count.div_ceil(desired_index_len - 2 * chunk_count)).max(1);
        let mut view = &mut view[..];
        let mut count = 0;
        for chunk in self.draw_index_chunk_iter(line_visible_range) {
            view = append_u32(view, 0);
            let end = chunk.clone().into_index_iter().last();
            let mut index_iter = chunk.into_index_iter();
            if let Some(index) = index_iter.next() {
                count += 1;
                view = append_u32(view, index);
            }
            for index in index_iter.step_by(step) {
                count += 1;
                view = append_u32(view, index);
            }
            if let Some(end) = end {
                count += 1;
                view = append_u32(view, end);
            }
            view = append_u32(view, 0);
            count += 2;
        }
        count
    }

    pub fn garbage_collect(&mut self, line_visible_range: Range<Timestamp>) {
        let first_half = nodit::interval::ee(i64::MIN, line_visible_range.start.0 - 1);
        let second_half = nodit::interval::ee(line_visible_range.end.0 + 1, i64::MAX);
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
            if let Some(alloc) = &mut self.data_buffer_shard_alloc {
                if let Some(gpu) = chunk.data.gpu.lock().take() {
                    alloc.dealloc(gpu);
                    chunk.data.gpu_dirty.store(true, atomic::Ordering::SeqCst);
                }
            }
            if let Some(alloc) = &mut self.timestamp_buffer_shard_alloc {
                if let Some(gpu) = chunk.timestamps_float.gpu.lock().take() {
                    alloc.dealloc(gpu);
                    chunk
                        .timestamps_float
                        .gpu_dirty
                        .store(true, atomic::Ordering::SeqCst);
                }
            }
        }
    }
}

fn append_u32(view: &mut [u8], val: u32) -> &mut [u8] {
    if view.len() < 4 {
        return view;
    }
    view[..size_of::<u32>()].copy_from_slice(&val.to_le_bytes());
    &mut view[size_of::<u32>()..]
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
