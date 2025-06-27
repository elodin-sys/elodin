use std::collections::BTreeMap;
use std::ops::Range;

use bevy::core_pipeline::tonemapping::Tonemapping;

use bevy::render::camera::ScalingMode;

use bevy::prelude::*;
use bevy::render::view::RenderLayers;
use bevy_egui::egui::{self, Color32};

use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{ComponentPath, ComponentValue};
use impeller2_wkt::GraphType;

use super::gpu::LineVisibleRange;
use crate::MainCamera;
use crate::plugins::navigation_gizmo::RenderLayerAlloc;
use crate::ui::{ViewportRect, colors};

pub type GraphStateComponent = Vec<(bool, egui::Color32)>;
pub type GraphStateEntity = BTreeMap<ComponentId, GraphStateComponent>;

#[derive(Bundle, Clone)]
pub struct GraphBundle {
    pub graph_state: GraphState,
    pub camera: Camera,
    pub camera_2d: Camera2d,
    pub projection: Projection,
    pub tonemapping: Tonemapping,
    pub viewport_rect: ViewportRect,
    pub render_layers: RenderLayers,
    pub main_camera: MainCamera,
}

#[derive(Clone, Debug, Component)]
pub struct GraphState {
    pub components: BTreeMap<ComponentPath, GraphStateComponent>,
    pub enabled_lines: BTreeMap<(ComponentPath, usize), (Entity, Color32)>,
    pub render_layers: RenderLayers,
    pub line_width: f32,
    pub zoom_factor: Vec2,
    pub pan_offset: Vec2,
    pub graph_type: GraphType,
    pub label: String,
    pub auto_y_range: bool,
    pub y_range: Range<f64>,
    pub auto_x_range: bool,
    pub x_range: Range<f64>,
    pub widget_width: f64,
    pub visible_range: LineVisibleRange,
}

impl GraphBundle {
    pub fn new(
        render_layer_alloc: &mut RenderLayerAlloc,
        components: BTreeMap<ComponentPath, GraphStateComponent>,
        label: String,
    ) -> Self {
        let Some(layer) = render_layer_alloc.alloc() else {
            todo!("ran out of layers")
        };
        let render_layers = RenderLayers::layer(layer);
        let graph_state = GraphState {
            components,
            enabled_lines: BTreeMap::new(),
            render_layers: render_layers.clone(),
            line_width: 2.0,
            zoom_factor: Vec2::ONE,
            pan_offset: Vec2::ZERO,
            graph_type: GraphType::Line,
            label,
            y_range: 0.0..1.0,
            x_range: 0.0..1.0,
            auto_y_range: true,
            auto_x_range: true,
            widget_width: 1920.0,
            visible_range: LineVisibleRange(Timestamp(i64::MIN)..Timestamp(i64::MAX)),
        };
        GraphBundle {
            camera: Camera {
                order: 2,
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
            graph_state,
            camera_2d: Camera2d,
            viewport_rect: ViewportRect(None),
            main_camera: MainCamera,
            render_layers,
        }
    }
}

impl GraphState {
    pub fn remove_component(&mut self, component_path: &ComponentPath) {
        self.components.remove(component_path);

        // Also remove from enabled_lines
        self.enabled_lines
            .retain(|(path, _), _| path != component_path);
    }

    pub fn insert_component(
        &mut self,
        component_path: ComponentPath,
        component_values: Vec<(bool, egui::Color32)>,
    ) {
        self.components.insert(component_path, component_values);
    }
}

pub fn default_component_values(
    component_id: &ComponentId,
    component_value: &ComponentValue,
) -> GraphStateComponent {
    component_value
        .iter()
        .enumerate()
        .map(|(i, _)| (component_id.0) as usize + i)
        .map(|i| (true, colors::get_color_by_index_all(i)))
        .collect::<Vec<(bool, egui::Color32)>>()
}
