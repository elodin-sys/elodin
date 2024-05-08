use std::collections::BTreeMap;

use bevy::core_pipeline::tonemapping::Tonemapping;

use bevy::render::camera::ScalingMode;

use bevy::prelude::*;
use bevy::render::view::RenderLayers;
use bevy_egui::egui::{self, Color32};

use conduit::ComponentValue;
use conduit::{ComponentId, EntityId, GraphId};

use crate::plugins::navigation_gizmo::RenderLayerAlloc;
use crate::ui::widgets::timeline::tagged_range::TaggedRangeId;
use crate::ui::{colors, ViewportRect};
use crate::MainCamera;

pub type GraphStateComponent = Vec<(bool, egui::Color32)>;
pub type GraphStateEntity = BTreeMap<ComponentId, GraphStateComponent>;

#[derive(Clone, Debug)]
pub struct GraphState {
    pub entities: BTreeMap<EntityId, GraphStateEntity>,
    pub range_id: Option<TaggedRangeId>,
    pub enabled_lines: BTreeMap<(EntityId, ComponentId, usize), (Entity, Color32)>,
    pub render_layers: RenderLayers,
    pub camera: Entity,
    pub line_width: f32,
}

impl GraphState {
    pub fn spawn(
        commands: &mut Commands,
        render_layer_alloc: &mut RenderLayerAlloc,
        entities: BTreeMap<EntityId, GraphStateEntity>,
    ) -> Self {
        let Some(layer) = render_layer_alloc.alloc() else {
            todo!("ran out of layers")
        };
        let render_layers = RenderLayers::layer(layer as u8);
        let camera = commands
            .spawn((
                Camera3dBundle {
                    camera: Camera {
                        order: 1,
                        hdr: false,
                        ..Default::default()
                    },
                    tonemapping: Tonemapping::None,
                    projection: Projection::Orthographic(OrthographicProjection {
                        near: 0.0,
                        far: 1000.0,
                        viewport_origin: Vec2::new(0.0, 0.0),
                        scaling_mode: ScalingMode::Fixed {
                            width: 1000.0,
                            height: 1.0,
                        },
                        scale: 1.0,
                        area: Rect::new(0., 0., 500., 1.),
                    }),
                    ..Default::default()
                },
                ViewportRect(None),
                MainCamera,
                render_layers,
            ))
            .id();
        GraphState {
            entities,
            range_id: None,
            enabled_lines: BTreeMap::new(),
            render_layers,
            camera,
            line_width: 2.0,
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct GraphsState(pub BTreeMap<GraphId, GraphState>);

impl GraphsState {
    pub fn get(&self, graph_id: &GraphId) -> Option<&GraphState> {
        self.0.get(graph_id)
    }

    pub fn get_mut(&mut self, graph_id: &GraphId) -> Option<&mut GraphState> {
        self.0.get_mut(graph_id)
    }

    pub fn push_graph_state(&mut self, graph_state: GraphState) -> GraphId {
        let new_graph_id = self
            .0
            .keys()
            .max()
            .map_or(GraphId(0), |lk| GraphId(lk.0 + 1));
        self.0.insert(new_graph_id, graph_state);

        new_graph_id
    }

    pub fn default_component_values(
        entity_id: &EntityId,
        component_id: &ComponentId,
        component_value: &ComponentValue,
    ) -> GraphStateComponent {
        component_value
            .iter()
            .enumerate()
            .map(|(i, _)| (entity_id.0 + component_id.0) as usize + i)
            .map(|i| (true, colors::get_color_by_index_all(i)))
            .collect::<Vec<(bool, egui::Color32)>>()
    }

    pub fn insert_component(
        &mut self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
        component_values: Vec<(bool, egui::Color32)>,
    ) {
        let Some(graph) = self.0.get_mut(graph_id) else {
            println!("graph not found");
            return;
        };

        let entity = graph
            .entities
            .entry(*entity_id)
            .or_insert_with(BTreeMap::new);

        entity.insert(*component_id, component_values);
    }

    pub fn remove_graph(&mut self, graph_id: &GraphId) {
        self.0.remove(graph_id);
    }

    pub fn remove_component(
        &mut self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) {
        let Some(graph) = self.0.get_mut(graph_id) else {
            return;
        };

        let Some(components) = graph.entities.get_mut(entity_id) else {
            return;
        };

        components.remove(component_id);

        if components.is_empty() {
            graph.entities.remove(entity_id);
        }
    }

    pub fn contains_graph(&self, graph_id: &GraphId) -> bool {
        self.0.contains_key(graph_id)
    }

    pub fn contains_component(
        &self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) -> bool {
        if let Some(graph) = self.0.get(graph_id) {
            if let Some(entity) = graph.entities.get(entity_id) {
                return entity.contains_key(component_id);
            }
        }

        false
    }
}
