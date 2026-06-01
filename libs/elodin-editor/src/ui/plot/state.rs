use std::collections::BTreeMap;
use std::ops::Range;

use bevy::camera::ScalingMode;
use bevy::camera::visibility::RenderLayers;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy_egui::egui::{self, Color32};

use impeller2::schema::Schema;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::ComponentPath;
use impeller2_wkt::{ComponentMetadata, GraphType};

use super::gpu::LineVisibleRange;
use crate::MainCamera;
use crate::plugins::render_layer_alloc::{RenderLayerAllocator, RenderLayerLease};
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
    pub allocated_layer: RenderLayerLease,
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
    pub locked: bool,

    // peer-ready metadata (not used until widget.rs switches to peers)
    pub x_rev: u64,
    pub x_dirty: bool,
}

impl GraphBundle {
    pub fn try_new(
        render_layer_alloc: &mut RenderLayerAllocator,
        components: BTreeMap<ComponentPath, GraphStateComponent>,
        label: String,
    ) -> Option<Self> {
        let allocated_layer = render_layer_alloc.alloc()?;
        let render_layers = allocated_layer.render_layers();
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
            locked: false,
            x_rev: 0,
            x_dirty: false,
        };
        Some(GraphBundle {
            camera: Camera {
                order: 2,
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
            allocated_layer,
        })
    }

    pub fn new(
        render_layer_alloc: &mut RenderLayerAllocator,
        components: BTreeMap<ComponentPath, GraphStateComponent>,
        label: String,
    ) -> Self {
        Self::try_new(render_layer_alloc, components, label).expect("ran out of render layers")
    }
}

impl GraphState {
    pub fn remove_component(&mut self, component_path: &ComponentPath) {
        self.components.remove(component_path);
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

pub fn element_names_for_graph(
    schema: &Schema<Vec<u64>>,
    metadata: &ComponentMetadata,
) -> Vec<String> {
    let from_metadata: Vec<String> = metadata
        .element_names()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if !from_metadata.is_empty() {
        return from_metadata;
    }
    let len = schema.shape().iter().copied().product::<usize>().max(1);
    if len == 1 {
        return vec![metadata.name.clone()];
    }
    eql::Component::new(metadata.name.clone(), metadata.component_id, schema.clone()).element_names
}

pub fn graph_lines_from_component(
    component_path: &ComponentPath,
    schema: &Schema<Vec<u64>>,
    _metadata: &ComponentMetadata,
) -> GraphStateComponent {
    let len = schema.shape().iter().copied().product::<usize>().max(1);
    let color_base = component_path.id.0 as usize;
    (0..len)
        .map(|i| (true, colors::get_color_by_index_all(color_base + i)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2::types::PrimType;
    use std::collections::HashMap;

    fn test_schema(shape: &[u64]) -> Schema<Vec<u64>> {
        Schema::new(PrimType::F64, shape).unwrap()
    }

    fn test_metadata(name: &str, element_names: &str) -> ComponentMetadata {
        let mut metadata = HashMap::new();
        if !element_names.is_empty() {
            metadata.insert("element_names".to_string(), element_names.to_string());
        }
        ComponentMetadata {
            component_id: ComponentId::new(name),
            name: name.to_string(),
            metadata,
        }
    }

    #[test]
    fn graph_lines_scalar_enables_one_line() {
        let path = ComponentPath::from_name("rocket.mach");
        let schema = test_schema(&[1]);
        let metadata = test_metadata("rocket.mach", "");
        let lines = graph_lines_from_component(&path, &schema, &metadata);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].0);
    }

    #[test]
    fn graph_lines_vector_enables_all_elements() {
        let path = ComponentPath::from_name("CONTROLMESSAGE.FIN_DEFLECTION_DEG");
        let schema = test_schema(&[4]);
        let metadata = test_metadata("CONTROLMESSAGE.FIN_DEFLECTION_DEG", "");
        let lines = graph_lines_from_component(&path, &schema, &metadata);
        assert_eq!(lines.len(), 4);
        assert!(lines.iter().all(|(enabled, _)| *enabled));
    }

    #[test]
    fn element_names_prefers_metadata() {
        let schema = test_schema(&[3]);
        let metadata = test_metadata("foo.bar", "a,b,c");
        assert_eq!(
            element_names_for_graph(&schema, &metadata),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn element_names_falls_back_to_schema_defaults() {
        let schema = test_schema(&[4]);
        let metadata = test_metadata("foo.bar", "");
        assert_eq!(
            element_names_for_graph(&schema, &metadata),
            vec!["x", "y", "z", "w"]
        );
    }

    #[test]
    fn element_names_scalar_shape_one_uses_component_name() {
        let schema = test_schema(&[1]);
        let metadata = test_metadata("STATEMACHINEOUTPUT.MISSILE_MODE", "");
        assert_eq!(
            element_names_for_graph(&schema, &metadata),
            vec!["STATEMACHINEOUTPUT.MISSILE_MODE"]
        );
    }

    #[test]
    fn element_names_scalar_empty_shape_uses_component_name() {
        let schema = test_schema(&[]);
        let metadata = test_metadata("STATEMACHINEOUTPUT.MISSILE_MODE", "");
        assert_eq!(
            element_names_for_graph(&schema, &metadata),
            vec!["STATEMACHINEOUTPUT.MISSILE_MODE"]
        );
    }
}
