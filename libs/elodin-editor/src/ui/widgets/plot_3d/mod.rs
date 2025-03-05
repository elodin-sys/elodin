#![allow(warnings)]

use std::collections::BTreeMap;

use crate::ui::widgets::plot::CollectedGraphData;
use bevy::{
    app::{Startup, Update},
    asset::Assets,
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    math::{Mat4, Vec4},
    render::view::RenderLayers,
};
use big_space::GridCell;
use impeller2_bevy::{CommandsExt, ComponentMetadataRegistry};
use impeller2_wkt::LastUpdated;
use impeller2_wkt::{EntityMetadata, GetTimeSeries, Line3d};

use gpu::LineBundle;
use gpu::{LineConfig, LineUniform};

use super::plot::{Line, PlotDataComponent, PlotDataEntity, gpu::LineHandle};

pub mod gpu;

pub fn sync_line_plot_3d(
    mut collected_graph_data: ResMut<CollectedGraphData>,
    line_plot_3d_query: Query<(Entity, &EntityMetadata, &Line3d), Without<LineHandle>>,
    metadata_store: Res<ComponentMetadataRegistry>,
    mut uniforms: Query<(&Line3d, &mut LineUniform), With<LineHandle>>,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
) {
    for (entity, metadata, line_plot) in line_plot_3d_query.iter() {
        let line = collected_graph_data
            .entities
            .entry(line_plot.entity)
            .or_insert_with(|| PlotDataEntity {
                label: metadata.name.clone(),
                components: BTreeMap::new(),
            });
        let Some(metadata) = metadata_store.get_metadata(&line_plot.component_id) else {
            continue;
        };

        let data_component = line
            .components
            .entry(line_plot.component_id)
            .or_insert_with(|| {
                PlotDataComponent::new(
                    metadata.name.clone(),
                    metadata
                        .element_names()
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect(),
                )
            });
        let [x, y, z] = line_plot.index;
        let Some(x) = data_component.lines.get(&x).cloned() else {
            continue;
        };
        let Some(y) = data_component.lines.get(&y).cloned() else {
            continue;
        };
        let Some(z) = data_component.lines.get(&z).cloned() else {
            continue;
        };

        if let Some(mut entity) = commands.get_entity(entity) {
            entity.try_insert(LineBundle {
                line: gpu::LineHandles([x, y, z]),
                uniform: LineUniform {
                    line_width: line_plot.line_width,
                    color: Vec4::new(line_plot.color.r, line_plot.color.g, line_plot.color.b, 1.0),
                    depth_bias: 0.0,
                    model: Mat4::IDENTITY,
                    perspective: if line_plot.perspective { 1 } else { 0 },
                    #[cfg(target_arch = "wasm32")]
                    _padding: Default::default(),
                },
                config: LineConfig {
                    render_layers: RenderLayers::default(),
                },
                global_transform: Default::default(),
                transform: Default::default(),
                grid_cell: GridCell::default(),
            });
        }
    }
    for (line_plot, mut uniform) in uniforms.iter_mut() {
        uniform.color = Vec4::new(line_plot.color.r, line_plot.color.g, line_plot.color.b, 1.0);
        uniform.line_width = line_plot.line_width;
        uniform.perspective = if line_plot.perspective { 1 } else { 0 };
    }
}

pub struct LinePlot3dPlugin;

impl bevy::app::Plugin for LinePlot3dPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .add_plugins(gpu::Plot3dGpuPlugin)
            .add_systems(Update, sync_line_plot_3d);
    }
}
