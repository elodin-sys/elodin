use bevy::{
    asset::{Assets, Handle},
    ecs::{
        entity::Entity,
        event::EventReader,
        system::{Commands, Query, Res},
    },
    log::warn,
    prelude::{ResMut, Resource},
};
use conduit::{
    client::Msg, query::MetadataStore, ser_de::ColumnValue, well_known::EntityMetadata,
    ComponentId, ComponentValue, ControlMsg, EntityId,
};

use crate::ui::{widgets::plot::gpu::Line, widgets::plot::GraphsState};
use std::{collections::BTreeMap, fmt::Debug, ops::Range};

#[derive(Debug, Clone)]
pub struct PlotDataLine {
    pub label: String,
    pub values: Handle<Line>,
    pub line_entity: Option<Entity>,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug)]
pub struct PlotDataComponent {
    pub label: String,
    pub element_names: String,
    pub next_tick: u64,
    pub lines: BTreeMap<usize, PlotDataLine>,
}

impl PlotDataComponent {
    pub fn new(component_label: impl ToString, element_names: String) -> Self {
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
            .split(',')
            .filter(|s| !s.is_empty())
            .map(Option::Some)
            .chain(std::iter::repeat(None));
        for (i, (new_value, name)) in component_value.iter().zip(element_names).enumerate() {
            let new_value = new_value.as_f64();
            let label = name.map(str::to_string).unwrap_or_else(|| format!("[{i}]"));
            let line = self.lines.entry(i).or_insert_with(|| PlotDataLine {
                label,
                values: assets.add(Line::default()),
                min: new_value,
                max: new_value,
                line_entity: None,
            });
            let values = assets
                .get_mut(line.values.clone())
                .expect("missing line asset");
            values.push(new_value);
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
    ) -> Option<&PlotDataLine> {
        self.entities
            .get(entity_id)
            .and_then(|entity| entity.components.get(component_id))
            .and_then(|component| component.lines.get(&index))
    }
}

pub fn collect_entity_data(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut graph_states: ResMut<GraphsState>,
    mut reader: EventReader<Msg>,
    metadata_store: Res<MetadataStore>,
    entity_metadata: Query<(&EntityId, &EntityMetadata)>,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
) {
    let entity_metadata = entity_metadata
        .iter()
        .collect::<BTreeMap<&EntityId, &EntityMetadata>>();

    for msg in reader.read() {
        match msg {
            Msg::Control(ControlMsg::StartSim { .. }) => {
                collected_graph_data.entities.clear();
                collected_graph_data.tick_range = 0..0;
                for graph in graph_states.0.values_mut() {
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
                        .or_insert_with(|| PlotDataEntity {
                            label: entity_metadata
                                .get(&entity_id)
                                .map_or(format!("E[{}]", entity_id.0), |metadata| {
                                    metadata.name.to_owned()
                                }),
                            components: Default::default(),
                        })
                        .components
                        .entry(component_id)
                        .or_insert_with(|| {
                            PlotDataComponent::new(component_label, element_names.to_string())
                        })
                        .add_values(&value, &mut lines, col.payload.time);
                }
            }
            _ => {}
        }
    }
}
