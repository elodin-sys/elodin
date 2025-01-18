use std::collections::BTreeMap;

use bevy::core_pipeline::tonemapping::Tonemapping;

use bevy::render::camera::ScalingMode;

use bevy::prelude::*;
use bevy::render::view::RenderLayers;
use bevy_egui::egui::{self, Color32};

use impeller2::types::{ComponentId, EntityId};
use impeller2_bevy::ComponentValue;

use crate::plugins::navigation_gizmo::RenderLayerAlloc;
use crate::ui::widgets::timeline::timeline_ranges::TimelineRangeId;
use crate::ui::{colors, ViewportRect};
use crate::MainCamera;

pub type GraphStateComponent = Vec<(bool, egui::Color32)>;
pub type GraphStateEntity = BTreeMap<ComponentId, GraphStateComponent>;

#[derive(Bundle)]
pub struct GraphBundle {
    pub graph_state: GraphState,
    pub camera: Camera,
    pub camera_2d: Camera2d,
    pub projection: OrthographicProjection,
    pub tonemapping: Tonemapping,
    pub viewport_rect: ViewportRect,
    pub render_layers: RenderLayers,
    pub main_camera: MainCamera,
}

#[derive(Clone, Debug, Component)]
pub struct GraphState {
    pub entities: BTreeMap<EntityId, GraphStateEntity>,
    pub range_id: Option<TimelineRangeId>,
    pub enabled_lines: BTreeMap<(EntityId, ComponentId, usize), (Entity, Color32)>,
    pub render_layers: RenderLayers,
    pub line_width: f32,
    pub zoom_factor: Vec2,
    pub pan_offset: Vec2,
}

impl GraphBundle {
    pub fn new(
        //commands: &mut Commands,
        render_layer_alloc: &mut RenderLayerAlloc,
        entities: BTreeMap<EntityId, GraphStateEntity>,
        range_id: Option<TimelineRangeId>,
    ) -> Self {
        let Some(layer) = render_layer_alloc.alloc() else {
            todo!("ran out of layers")
        };
        let render_layers = RenderLayers::layer(layer);
        let graph_state = GraphState {
            entities,
            range_id,
            enabled_lines: BTreeMap::new(),
            render_layers: render_layers.clone(),
            line_width: 2.0,
            zoom_factor: Vec2::ONE,
            pan_offset: Vec2::ZERO,
        };
        GraphBundle {
            camera: Camera {
                order: 2,
                hdr: false,
                ..Default::default()
            },
            tonemapping: Tonemapping::None,
            projection: OrthographicProjection {
                near: 0.0,
                far: 1000.0,
                viewport_origin: Vec2::new(0.0, 0.0),
                scaling_mode: ScalingMode::Fixed {
                    width: 1000.0,
                    height: 1.0,
                },
                scale: 1.0,
                area: Rect::new(0., 0., 500., 1.),
            },
            graph_state,
            camera_2d: Camera2d,
            viewport_rect: ViewportRect(None),
            main_camera: MainCamera,
            render_layers,
        }
    }
}

impl GraphState {
    pub fn remove_component(&mut self, entity_id: &EntityId, component_id: &ComponentId) {
        let Some(components) = self.entities.get_mut(entity_id) else {
            return;
        };

        components.remove(component_id);

        if components.is_empty() {
            self.entities.remove(entity_id);
        }
    }

    pub fn insert_component(
        &mut self,
        entity_id: &EntityId,
        component_id: &ComponentId,
        component_values: Vec<(bool, egui::Color32)>,
    ) {
        let entity = self.entities.entry(*entity_id).or_default();

        entity.insert(*component_id, component_values);
    }
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
