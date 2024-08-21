use bevy::asset::Asset;
use bevy::ecs::query::Without;
use bevy::ecs::system::Commands;
use bevy::math::{DVec3, Vec3};
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::Query,
    log::warn,
    prelude::Resource,
};
use big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};
use impeller::bytes::Bytes;
use impeller::{
    client::Msg, ser_de::ColumnValue, ComponentId, ComponentValue, ControlMsg, EntityId,
};

use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::chunks::{Chunks, ShadowBuffer, Unloaded, CHUNK_SIZE};
use crate::ui::widgets::entity_data::Plot3dData;

use super::gpu::LineBundle;

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
        component_value: &ComponentValue,
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

#[allow(clippy::too_many_arguments)]
pub fn collect_entity_data(
    msg: &Msg<Bytes>,
    collected_graph_data: &mut CollectedGraphData,
    graphs: &mut Query<Plot3dData, Without<FloatingOrigin>>,
    lines: &mut Assets<LineData>,
    floating_origin: &FloatingOriginSettings,
    origin: &Query<(&GridCell<i128>, &FloatingOrigin)>,
    commands: &mut Commands,
) {
    match &msg {
        Msg::Control(ControlMsg::StartSim { .. }) => {
            collected_graph_data.entities.clear();
            collected_graph_data.tick_range = 0..0;
            for graph in graphs.iter_mut() {
                commands.entity(graph.0).remove::<LineBundle>();
            }
        }
        Msg::Control(ControlMsg::Tick {
            tick,
            max_tick: _,
            simulating: _,
        }) => {
            if collected_graph_data.tick_range.end < *tick {
                collected_graph_data.tick_range.end = *tick;
            }
        }
        Msg::Column(col) => {
            let component_id = col.metadata.component_id();
            for res in col.iter() {
                let Ok(ColumnValue { entity_id, value }) = res else {
                    warn!("error parsing column value");
                    continue;
                };
                let Some(entity) = collected_graph_data.entities.get_mut(&entity_id) else {
                    continue;
                };
                let Some(component) = entity.components.get_mut(&component_id) else {
                    continue;
                };
                component.add_values(&value, lines, col.payload.time, floating_origin);
            }
        }
        _ => {}
    }
    let (origin_cell, _) = origin.single();

    for (_, line, mut grid_cell, transform, mut global) in graphs {
        let Some(line) = lines.get_mut(line) else {
            continue;
        };
        *grid_cell = line.grid_cell;
        let grid_cell_delta = *grid_cell - *origin_cell;
        *global = transform
            .with_translation(floating_origin.grid_position(&grid_cell_delta, transform))
            .into();
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
