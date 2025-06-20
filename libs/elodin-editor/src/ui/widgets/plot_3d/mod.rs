#![allow(warnings)]

use std::collections::BTreeMap;

use bevy::{
    animation::graph,
    app::{Startup, Update},
    asset::{Assets, Handle},
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    math::{Mat4, Vec4},
    render::view::RenderLayers,
};
use big_space::GridCell;
use eql;
use impeller2_bevy::{CommandsExt, ComponentMetadataRegistry, EntityMap};
use impeller2_wkt::LastUpdated;
use impeller2_wkt::{ComponentValue, EntityMetadata, GetTimeSeries, Line3d};

use gpu::LineBundle;
use gpu::{LineConfig, LineUniform};

use super::plot::{CollectedGraphData, Line, PlotDataComponent, gpu::LineHandle};
use crate::{
    EqlContext,
    object_3d::{CompiledExpr, EditableEQL, compile_eql_expr},
    ui::preset::EqlExt,
};

pub mod gpu;

pub fn sync_line_plot_3d(
    line_plot_3d_query: Query<(Entity, &Line3d), Without<gpu::LineHandles>>,
    mut uniforms: Query<(&Line3d, &mut LineUniform), With<LineHandle>>,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
    eql_ctx: Res<EqlContext>,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    metadata_store: Res<ComponentMetadataRegistry>,
) {
    for (entity, line_plot) in line_plot_3d_query.iter() {
        // Parse and compile the EQL expression
        let parsed = match eql_ctx.0.parse_str(&line_plot.eql) {
            Ok(expr) => expr,
            Err(e) => {
                println!(
                    "Failed to parse Line3D EQL expression '{}': {}",
                    line_plot.eql, e
                );
                continue;
            }
        };
        let graph_components = parsed.to_graph_components();
        let skip = if graph_components.len() == 7 { 4 } else { 0 };
        let mut handles: [Option<Handle<Line>>; 3] = [None, None, None];
        for (i, (c, index)) in graph_components.iter().skip(skip).take(3).enumerate() {
            let Some(metadata) = metadata_store.get_metadata(&c.id) else {
                continue;
            };
            let data = collected_graph_data
                .components
                .entry(c.id)
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
            handles[i] = data.lines.get(index).cloned();
        }
        let [Some(x), Some(y), Some(z)] = handles else {
            continue;
        };

        if let Ok(mut entity) = commands.get_entity(entity) {
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
