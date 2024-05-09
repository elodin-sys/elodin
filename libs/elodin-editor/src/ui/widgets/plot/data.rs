use bevy::asset::Asset;
use bevy::reflect::TypePath;
use bevy::{
    asset::{Assets, Handle},
    ecs::{
        event::EventReader,
        system::{Commands, Query, Res},
    },
    log::warn,
    prelude::{ResMut, Resource},
};
use conduit::bevy::EntityMap;
use conduit::{
    client::Msg, query::MetadataStore, ser_de::ColumnValue, well_known::EntityMetadata,
    ComponentId, ComponentValue, ControlMsg, EntityId,
};
use itertools::Itertools;

use std::{collections::BTreeMap, fmt::Debug, ops::Range};

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
        if tick < self.next_tick {
            return;
        }
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
            let line = assets.get_mut(line.clone()).expect("missing line asset");
            line.push(new_value);
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
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut graphs: Query<&mut GraphState>,
    mut reader: EventReader<Msg>,
    metadata_store: Res<MetadataStore>,
    entity_metadata: Query<&EntityMetadata>,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
    entity_map: Res<EntityMap>,
) {
    for msg in reader.read() {
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
            Msg::Control(ControlMsg::Tick { tick, max_tick: _ }) => {
                if collected_graph_data.tick_range.end < *tick {
                    collected_graph_data.tick_range.end = *tick;
                }
            }
            Msg::Column(col) => {
                let component_id = col.metadata.component_id();
                let Some(component_metadata) = metadata_store.get_metadata(&component_id) else {
                    return;
                };
                let component_label = component_metadata.component_name();
                let element_names = component_metadata.element_names();
                for res in col.iter() {
                    let Ok(ColumnValue { entity_id, value }) = res else {
                        warn!("error parsing column value");
                        continue;
                    };
                    collected_graph_data
                        .entities
                        .entry(entity_id)
                        .or_insert_with(|| {
                            let label = entity_map
                                .get(&entity_id)
                                .and_then(|id| entity_metadata.get(*id).ok())
                                .map_or(format!("E[{}]", entity_id.0), |metadata| {
                                    metadata.name.to_owned()
                                });
                            PlotDataEntity {
                                label,
                                components: Default::default(),
                            }
                        })
                        .components
                        .entry(component_id)
                        .or_insert_with(|| {
                            PlotDataComponent::new(
                                component_label,
                                element_names
                                    .split(',')
                                    .filter(|s| !s.is_empty())
                                    .map(str::to_string)
                                    .collect(),
                            )
                        })
                        .add_values(&value, &mut lines, col.payload.time);
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Asset, Clone, TypePath, Default)]
pub struct Line {
    pub label: String,
    pub min: f64,
    pub max: f64,
    pub data: LineData,
}

impl Line {
    pub fn push(&mut self, value: f64) {
        self.data.push(value)
    }

    pub fn recalculate_chunk_size(&mut self) -> bool {
        self.data.recalculate_chunk_size()
    }
}

#[derive(Debug, Clone)]
pub struct LineData {
    pub data: Vec<f64>,
    pub averaged_data: Vec<f32>,
    pub chunk_size: usize,
    pub max_count: usize,
    pub mean_state: MeanState,
}

#[derive(Debug, Copy, Clone, Default)]
pub enum MeanState {
    #[default]
    Pending,
    Averaging {
        count: usize,
        sum: f64,
    },
}

impl Default for LineData {
    fn default() -> Self {
        Self {
            data: Default::default(),
            averaged_data: Default::default(),
            chunk_size: 1,
            max_count: 1_000,
            mean_state: Default::default(),
        }
    }
}

impl LineData {
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        if self.recalculate_chunk_size() {
            self.push_raw(value)
        }
    }

    pub fn recalculate_chunk_size(&mut self) -> bool {
        let new_chunk_size = (self.data.len() / self.max_count).max(1);
        if new_chunk_size == self.chunk_size {
            return true;
        }
        self.chunk_size = new_chunk_size;
        self.averaged_data = self
            .data
            .iter()
            .chunks(new_chunk_size)
            .into_iter()
            .map(|chunk| (chunk.sum::<f64>() / new_chunk_size as f64) as f32)
            .collect();
        false
    }

    fn push_raw(&mut self, value: f64) {
        let (count, sum) = match self.mean_state {
            MeanState::Pending => (1, value),
            MeanState::Averaging { count, sum } => {
                let count = count + 1;
                let sum = sum + value;
                (count, sum)
            }
        };
        if count >= self.chunk_size {
            self.mean_state = MeanState::Pending;
            let datum = sum / count as f64;
            self.averaged_data.push(datum as f32);
        } else {
            self.mean_state = MeanState::Averaging { count, sum }
        }
    }
}
