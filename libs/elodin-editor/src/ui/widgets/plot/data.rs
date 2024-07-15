use bevy::asset::Asset;
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::system::{Commands, Query},
    log::warn,
    prelude::Resource,
};
use conduit::bytes::Bytes;
use conduit::{
    client::Msg, ser_de::ColumnValue, ComponentId, ComponentValue, ControlMsg, EntityId,
};
use itertools::Itertools;

use std::{collections::BTreeMap, fmt::Debug, ops::Range};

use crate::chunks::{Chunk, Chunks};

use super::GraphState;

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
        component_value: &ComponentValue,
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
            let new_value = new_value.as_f64();
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

#[allow(clippy::too_many_arguments)]
pub fn collect_entity_data(
    msg: &Msg<Bytes>,
    collected_graph_data: &mut CollectedGraphData,
    graphs: &mut Query<&mut GraphState>,
    lines: &mut Assets<Line>,
    commands: &mut Commands,
) {
    // while let Ok(msg) = reader.try_recv() {
    //     collect_entity_data_line_3d(
    //         &msg,
    //         &mut collected_graph_data_3d,
    //         &mut plots_3d,
    //         &mut lines_3d,
    //         &floating_origin,
    //     );
    match msg {
        Msg::Control(ControlMsg::StartSim { .. }) => {
            collected_graph_data.entities.clear();
            collected_graph_data.tick_range = 0..0;
            for mut graph in graphs.iter_mut() {
                for (_, (entity, _)) in graph.enabled_lines.iter() {
                    commands.entity(*entity).despawn();
                }
                graph.enabled_lines.clear();
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
                component.add_values(&value, lines, col.payload.time);
            }
        }
        _ => {}
    }
    //}
}

#[derive(Debug, Asset, Clone, TypePath, Default)]
pub struct Line {
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub data: LineData,
}

#[derive(Debug, Clone)]
pub struct LineData {
    pub data: Chunks<f64>,
    pub averaged_data: Vec<f32>,
    pub averaged_count: Vec<u16>,
    pub invalidated_range: Option<Range<usize>>,
    pub current_range: Range<usize>,
    pub averaged_range: Range<usize>,
    pub chunk_size: usize,
    pub max_count: usize,
}

impl Default for LineData {
    fn default() -> Self {
        Self {
            data: Default::default(),
            averaged_data: Default::default(),
            averaged_count: Default::default(),
            chunk_size: 1,
            max_count: 1_000,
            current_range: 0..0,
            averaged_range: 0..0,
            invalidated_range: None,
        }
    }
}

impl LineData {
    pub fn push(&mut self, tick: usize, value: f64) {
        self.data.push(tick, value);
        if self.recalculate_avg_data() {
            self.push_avg(tick, value);
        }
    }

    pub fn recalculate_avg_data(&mut self) -> bool {
        let data_len = self.current_range.end - self.current_range.start;
        let new_chunk_size = (data_len / self.max_count).max(1);
        let range_changed = self.averaged_range.start != self.current_range.start
            || self
                .current_range
                .end
                .overflowing_sub(self.averaged_range.end)
                .0
                > 200;
        if new_chunk_size == self.chunk_size && !range_changed {
            return true;
        }
        self.chunk_size = new_chunk_size;
        self.averaged_data = self
            .data
            .range(self.current_range.clone())
            .map(|x| *x)
            .chunks(new_chunk_size)
            .into_iter()
            .map(|chunk| (chunk.sum::<f64>() / new_chunk_size as f64) as f32)
            .collect();
        self.averaged_count = self
            .averaged_data
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if (*f).is_nan() {
                    let index = self.current_range.start + i * self.chunk_size;
                    self.data.set_unfetched(index..index + self.chunk_size);
                    0
                } else {
                    self.chunk_size as u16
                }
            })
            .collect();
        self.averaged_range = self.current_range.clone();
        self.invalidated_range = Some(0..self.averaged_data.len());
        false
    }

    fn push_avg(&mut self, tick: usize, value: f64) {
        if tick >= self.averaged_range.end + 200 || tick <= self.current_range.start {
            return;
        }
        let Some(tick) = tick.checked_sub(self.current_range.start) else {
            return;
        };
        let tick = tick / self.chunk_size;

        if let Some(ref mut range) = self.invalidated_range {
            range.start = range.start.min(tick);
            range.end = range.end.max(tick + 1);
        } else {
            self.invalidated_range = Some(tick..tick + 1)
        }
        if tick >= self.averaged_data.len() {
            if self.averaged_data.len() != tick {
                self.data.set_unfetched(0..tick.saturating_sub(1));
            }
            for i in self.averaged_data.len()..=tick {
                // TODO(sphw): this papers over a bug where we are missing certain ticks in the averaged value
                // When we fix that bug `value as f32` should be replaced with NAN.
                self.averaged_data.insert(i, value as f32);
                self.averaged_count.insert(i, 0);
            }
        }
        let averaged_count = self.averaged_count[tick] as usize;
        if averaged_count == 0 {
            self.averaged_data[tick] = value as f32;
            self.averaged_count[tick] = 1;
        } else if averaged_count < self.chunk_size {
            let new_count = self.averaged_count[tick] + 1;
            self.averaged_data[tick] = (self.averaged_data[tick] * averaged_count as f32
                + value as f32)
                / new_count as f32;
            self.averaged_count[tick] = new_count;
        }
        self.averaged_range = self.current_range.clone();
    }

    pub fn range(&mut self) -> impl Iterator<Item = &mut Chunk<f64>> {
        self.data.chunks_range(self.current_range.clone())
    }
}
