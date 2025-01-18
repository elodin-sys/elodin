use bevy::asset::Asset;
use bevy::log::warn;
use bevy::math::{DVec3, Vec3};
use bevy::prelude::{InRef, Res, ResMut};
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::{Commands, Query},
    prelude::Resource,
};
use big_space::{FloatingOriginSettings, GridCell};
use impeller2::types::PrimType;
use impeller2::types::{ComponentId, ComponentView, EntityId, OwnedPacket};
use impeller2_bevy::{ComponentValueMap, EntityMap, PacketHandlerInput, PacketHandlers};
use impeller2_wkt::{MaxTick, Tick};
use zerocopy::{Immutable, TryFromBytes};

use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::chunks::{Chunks, ShadowBuffer, Unloaded, CHUNK_SIZE};
pub fn collect_entity_data(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    max_tick: Res<MaxTick>,
) {
    collected_graph_data.tick_range = 0..max_tick.0;
}

pub fn pkt_handler(
    PacketHandlerInput { packet, registry }: PacketHandlerInput,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<LineData>>,
    tick: Res<Tick>,
    floating_origin: Res<FloatingOriginSettings>,
) {
    let mut tick = *tick;
    if let OwnedPacket::Table(table) = packet {
        if let Err(err) = table.sink(registry, &mut tick) {
            warn!(?err, "tick sink failed");
        }
        if let Err(err) = table.sink(
            registry,
            &mut |component_id, entity_id, view: ComponentView<'_>| {
                let Some(plot_data) =
                    collected_graph_data.get_component_mut(&entity_id, &component_id)
                else {
                    return;
                };
                plot_data.add_values(view, &mut lines, tick.0, &floating_origin);
            },
        ) {
            warn!(?err, "graph sink failed");
        }
    }
}

pub trait AsF64 {
    fn as_f64(&self) -> f64;
}

macro_rules! impl_as_f64 {
    ($($t:ty),*) => {
        $(
            impl AsF64 for $t {
                fn as_f64(&self) -> f64 { *self as f64 }
            }
        )*
    };
}

impl_as_f64!(u8, u16, u32, u64, i8, i16, i32, i64, f32);

impl AsF64 for f64 {
    fn as_f64(&self) -> f64 {
        *self
    }
}

impl AsF64 for bool {
    fn as_f64(&self) -> f64 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}

fn process_time_series<T>(
    time_series_buf: &[u8],
    len: usize,
    time_range: &Range<u64>,
    plot_data: &mut PlotDataComponent,
    lines: &mut Assets<LineData>,
    floating_origin: &FloatingOriginSettings,
) where
    T: AsF64 + TryFromBytes + Immutable,
{
    let Ok(data) = <[T]>::try_ref_from_bytes(time_series_buf) else {
        return;
    };
    let mut tick = time_range.start;
    for chunk in data.chunks(len) {
        let Some(x) = chunk.get(plot_data.indexes[0]) else {
            continue;
        };
        let Some(y) = chunk.get(plot_data.indexes[1]) else {
            continue;
        };
        let Some(z) = chunk.get(plot_data.indexes[2]) else {
            continue;
        };

        let value = DVec3::new(x.as_f64(), y.as_f64(), z.as_f64());

        if let Some(line) = &plot_data.line {
            let line = lines.get_mut(line.id()).expect("missing line asset");
            line.push(tick as usize, value, floating_origin);
        }
        tick += 1;
    }
}

#[allow(clippy::type_complexity)]
pub fn time_series_handler(
    entity_id: EntityId,
    component_id: ComponentId,
    time_range: Range<u64>,
) -> impl Fn(
    InRef<OwnedPacket<Vec<u8>>>,
    ResMut<CollectedGraphData>,
    Res<EntityMap>,
    Query<&ComponentValueMap>,
    ResMut<Assets<LineData>>,
    Res<FloatingOriginSettings>,
) {
    move |InRef(pkt),
          mut collected_graph_data: ResMut<CollectedGraphData>,
          entity_map: Res<EntityMap>,
          component_values,
          mut lines,
          floating_origin| match pkt {
        OwnedPacket::Msg(_) => {}
        OwnedPacket::Table(_) => {}
        OwnedPacket::TimeSeries(time_series) => {
            let Some(entity) = entity_map.get(&entity_id) else {
                return;
            };
            let Ok(component_value_map) = component_values.get(*entity) else {
                return;
            };
            let Some(current_value) = component_value_map.get(&component_id) else {
                return;
            };
            let Some(plot_data) = collected_graph_data.get_component_mut(&entity_id, &component_id)
            else {
                return;
            };
            let len = current_value.shape().iter().fold(1usize, |xs, &x| x * xs);
            let buf = &time_series.buf;
            match current_value.prim_type() {
                PrimType::U8 => process_time_series::<u8>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::U16 => process_time_series::<u16>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::U32 => process_time_series::<u32>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::U64 => process_time_series::<u64>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::I8 => process_time_series::<i8>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::I16 => process_time_series::<i16>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::I32 => process_time_series::<i32>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::I64 => process_time_series::<i64>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::Bool => process_time_series::<bool>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::F32 => process_time_series::<f32>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
                PrimType::F64 => process_time_series::<f64>(
                    buf,
                    len,
                    &time_range,
                    plot_data,
                    &mut lines,
                    &floating_origin,
                ),
            }
        }
    }
}
pub fn setup_pkt_handler(mut packet_handlers: ResMut<PacketHandlers>, mut commands: Commands) {
    let sys = commands.register_system(pkt_handler);
    packet_handlers.0.push(sys);
}

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub next_tick: u64,
    pub indexes: [usize; 3],
    pub line: Option<Handle<LineData>>,
}

impl PlotDataComponent {
    pub fn new(component_label: impl ToString, indexes: Option<[usize; 3]>) -> Self {
        Self {
            label: component_label.to_string(),
            next_tick: 0,
            indexes: indexes.unwrap_or([0, 1, 2]),
            line: None,
        }
    }

    pub fn add_values(
        &mut self,
        component_value: ComponentView<'_>,
        assets: &mut Assets<LineData>,
        tick: u64,
        floating_origin: &FloatingOriginSettings,
    ) {
        self.next_tick += 1;
        let [x, y, z] = self.indexes;
        let Some(x) = component_value.get(x) else {
            return;
        };
        let Some(y) = component_value.get(y) else {
            return;
        };
        let Some(z) = component_value.get(z) else {
            return;
        };
        let new_value = DVec3::new(x.as_f64(), y.as_f64(), z.as_f64());
        let line = self
            .line
            .get_or_insert_with(|| assets.add(LineData::default()));

        let line = assets.get_mut(line.id()).expect("missing line asset");
        line.push(tick as usize, new_value, floating_origin);
    }
}

#[derive(Clone, Debug)]
pub struct PlotDataEntity {
    pub label: String,
    pub components: BTreeMap<ComponentId, PlotDataComponent>,
}

#[derive(Resource, Default, Clone)]
pub struct CollectedGraphData {
    pub tick_range: Range<u64>,
    pub entities: BTreeMap<EntityId, PlotDataEntity>,
}

impl CollectedGraphData {
    pub fn get_entity(&self, entity_id: &EntityId) -> Option<&PlotDataEntity> {
        self.entities.get(entity_id)
    }
    pub fn get_component(
        &self,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) -> Option<&PlotDataComponent> {
        self.entities
            .get(entity_id)
            .and_then(|entity| entity.components.get(component_id))
    }
    pub fn get_component_mut(
        &mut self,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) -> Option<&mut PlotDataComponent> {
        self.entities
            .get_mut(entity_id)
            .and_then(|entity| entity.components.get_mut(component_id))
    }
    pub fn get_line(
        &self,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) -> Option<&Handle<LineData>> {
        self.entities
            .get(entity_id)
            .and_then(|entity| entity.components.get(component_id))
            .and_then(|component| component.line.as_ref())
    }
}

#[derive(Asset, Clone, TypePath)]
pub struct LineData {
    pub data: Chunks<DVec3>,
    pub processed_data: ShadowBuffer<Vec3>,
    pub range: Range<usize>,
    pub invalidated_range: Option<Range<usize>>,
    pub grid_cell: GridCell<i128>,
    pub grid_cell_length: f64,
}

fn local_value(grid_cell: &GridCell<i128>, value: DVec3, grid_edge_length: f64) -> Vec3 {
    let x = value.x - grid_cell.x as f64 * grid_edge_length;
    let y = value.y - grid_cell.y as f64 * grid_edge_length;
    let z = value.z - grid_cell.z as f64 * grid_edge_length;
    Vec3::new(x as f32, y as f32, z as f32)
}

impl Default for LineData {
    fn default() -> Self {
        Self {
            data: Default::default(),
            processed_data: Default::default(),
            range: 0..0,
            invalidated_range: None,
            grid_cell: GridCell::new(0, 0, 0),
            grid_cell_length: 0.0,
        }
    }
}

impl LineData {
    pub fn push(&mut self, tick: usize, value: DVec3, floating_origin: &FloatingOriginSettings) {
        let value = DVec3::new(value.x, value.z, -value.y);
        self.data.push(tick, value);
        self.grid_cell_length = floating_origin.grid_edge_length() as f64;
        let (new_grid_cell, translation) = floating_origin.translation_to_grid(value);
        if new_grid_cell == self.grid_cell && self.processed_data.push(tick, translation) {
            self.invalidated_range = None;
        } else {
            self.grid_cell = new_grid_cell;
            self.recalc_local_values();
        }
    }

    pub fn mark_unfetched(&mut self) {
        let start = self.range.start / CHUNK_SIZE;
        let end = self.range.end / CHUNK_SIZE;

        let range = start..end;
        for chunk in self.data.chunks_range(range) {
            let start = self.range.start.max(chunk.range.start) - chunk.range.start;
            let end = self.range.end.min(chunk.range.end) - chunk.range.start;
            for (i, x) in chunk.data.iter().enumerate().skip(start).take(end) {
                if x.is_unloaded() {
                    chunk.unfetched.insert((i + chunk.range.start) as u32);
                }
            }
        }
    }

    pub fn recalc_local_values(&mut self) {
        let grid_cell = self.grid_cell;
        self.mark_unfetched();

        let buf = self
            .data
            .range(self.range.clone())
            .map(|c| local_value(&grid_cell, *c, self.grid_cell_length))
            .collect();
        self.processed_data = ShadowBuffer::new(buf, self.range.clone());
        self.invalidated_range = None;
    }
}
