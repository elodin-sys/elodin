use bevy::asset::Asset;
use bevy::log::warn;
use bevy::prelude::{InRef, Res, ResMut};
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::{Commands, Query},
    prelude::Resource,
};
use impeller2::types::{ComponentId, ComponentView, EntityId, OwnedPacket, PrimType};
use impeller2_bevy::{ComponentValueMap, EntityMap, PacketHandlerInput, PacketHandlers};
use impeller2_wkt::{MaxTick, Tick};
use zerocopy::{Immutable, TryFromBytes};

use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::chunks::{Chunk, Chunks};

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub element_names: Vec<String>,
    pub next_tick: u64,
    pub lines: BTreeMap<usize, Handle<Line>>,
}

impl PlotDataComponent {
    pub fn new(component_label: impl ToString, element_names: Vec<String>) -> Self {
        Self {
            label: component_label.to_string(),
            element_names,
            next_tick: 0,
            lines: BTreeMap::new(),
        }
    }

    pub fn add_values(
        &mut self,
        component_value: ComponentView<'_>,
        assets: &mut Assets<Line>,
        tick: u64,
    ) {
        self.next_tick += 1;
        let element_names = self
            .element_names
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| Some(s.as_str()))
            .chain(std::iter::repeat(None));
        for (i, (new_value, name)) in component_value.iter().zip(element_names).enumerate() {
            let new_value = new_value.as_f32();
            let label = name.map(str::to_string).unwrap_or_else(|| format!("[{i}]"));
            let line = self.lines.entry(i).or_insert_with(|| {
                assets.add(Line {
                    label,
                    min: new_value,
                    max: new_value,
                    ..Default::default()
                })
            });
            let line = assets.get_mut(line.id()).expect("missing line asset");
            line.data.push(tick as usize, new_value);
            if line.min > new_value {
                line.min = new_value;
            }
            if line.max < new_value {
                line.max = new_value;
            }
        }
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
        index: usize,
    ) -> Option<&Handle<Line>> {
        self.entities
            .get(entity_id)
            .and_then(|entity| entity.components.get(component_id))
            .and_then(|component| component.lines.get(&index))
    }
}

pub fn collect_entity_data(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    max_tick: Res<MaxTick>,
) {
    collected_graph_data.tick_range = 0..max_tick.0;
}

pub fn pkt_handler(
    PacketHandlerInput { packet, registry }: PacketHandlerInput,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut lines: ResMut<Assets<Line>>,
    tick: Res<Tick>,
) {
    let mut tick = *tick;
    if let OwnedPacket::Table(table) = packet {
        if let Err(err) = table.sink(registry, &mut tick) {
            warn!(?err, "tick sick failed");
        }
        if let Err(err) = table.sink(
            registry,
            &mut |component_id, entity_id, view: ComponentView<'_>| {
                let Some(plot_data) =
                    collected_graph_data.get_component_mut(&entity_id, &component_id)
                else {
                    return;
                };
                plot_data.add_values(view, &mut lines, tick.0);
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
    len: usize,
    time_range: &Range<u64>,
    plot_data: &mut PlotDataComponent,
    lines: &mut Assets<Line>,
) where
    T: AsF32 + TryFromBytes + Immutable,
{
    let Ok(data) = <[T]>::try_ref_from_bytes(time_series_buf) else {
        return;
    };
    let mut tick = time_range.start;
    for chunk in data.chunks(len) {
        for (i, val) in chunk.iter().enumerate() {
            let val = val.as_f32();
            if let Some(line) = plot_data.lines.get(&i) {
                let line = lines.get_mut(line.id()).expect("missing line asset");
                line.data.push(tick as usize, val);
                line.min = line.min.min(val);
                line.max = line.max.max(val);
            }
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

    ResMut<Assets<Line>>,
) {
    move |InRef(pkt),
          mut collected_graph_data: ResMut<CollectedGraphData>,
          entity_map: Res<EntityMap>,
          component_values,
          mut lines| match pkt {
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
                PrimType::U8 => {
                    process_time_series::<u8>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::U16 => {
                    process_time_series::<u16>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::U32 => {
                    process_time_series::<u32>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::U64 => {
                    process_time_series::<u64>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::I8 => {
                    process_time_series::<i8>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::I16 => {
                    process_time_series::<i16>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::I32 => {
                    process_time_series::<i32>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::I64 => {
                    process_time_series::<i64>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::Bool => {
                    process_time_series::<bool>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::F32 => {
                    process_time_series::<f32>(buf, len, &time_range, plot_data, &mut lines)
                }
                PrimType::F64 => {
                    process_time_series::<f64>(buf, len, &time_range, plot_data, &mut lines)
                }
            }
        }
    }
}

#[derive(Debug, Asset, Clone, TypePath, Default)]
pub struct Line {
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub data: LineData,
}

#[derive(Debug, Clone)]
pub struct LineData {
    pub data: Chunks<f32>,
    pub invalidated_range: Option<Range<usize>>,
    pub current_range: Range<usize>,
}

impl Default for LineData {
    fn default() -> Self {
        Self {
            data: Default::default(),
            current_range: 0..0,
            invalidated_range: None,
        }
    }
}

impl LineData {
    pub fn push(&mut self, tick: usize, value: f32) {
        self.data.push(tick, value);
        self.expand_invalidation_range(tick..tick + 1);
    }

    pub fn expand_invalidation_range(&mut self, new_range: Range<usize>) {
        if let Some(ref mut range) = &mut self.invalidated_range {
            self.invalidated_range =
                Some(range.start.min(new_range.start)..range.end.max(new_range.end))
        } else {
            self.invalidated_range = Some(new_range)
        }
    }

    pub fn get(&self, tick: usize) -> Option<&f32> {
        self.data.get(tick)
    }

    pub fn range(&mut self) -> impl Iterator<Item = &mut Chunk<f32>> {
        self.data.chunks_range(self.current_range.clone())
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
    };
}

impl_as_f32!(u8, u16, u32, u64, i8, i16, i32, i64, f64);

impl AsF32 for f32 {
    fn as_f32(&self) -> f32 {
        *self
    }
}

impl AsF32 for bool {
    fn as_f32(&self) -> f32 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}
