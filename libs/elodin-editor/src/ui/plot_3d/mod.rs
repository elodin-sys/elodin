#![allow(warnings)]

use std::collections::BTreeMap;

use bevy::{
    animation::graph,
    app::{Startup, Update},
    asset::{Assets, Handle},
    camera::visibility::RenderLayers,
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    math::{DQuat, Mat4, Vec4},
    prelude::{Color, Transform},
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation};
use eql;
use impeller2_bevy::{CommandsExt, ComponentMetadataRegistry, EntityMap};
use impeller2_wkt::LastUpdated;
use impeller2_wkt::{ComponentValue, EntityMetadata, GetTimeSeries, Line3d};

use gpu::{LineConfig, LineUniform};

use super::plot::{CollectedGraphData, Line, PlotDataComponent};
use crate::{
    EqlContext,
    object_3d::{CompiledExpr, EditableEQL, compile_eql_expr},
    ui::schematic::EqlExt,
};

pub mod gpu;

/// Convert a schematic (sRGB) color into the linear RGB the line pipeline renders,
/// keeping it consistent with meshes/gizmos. Alpha is forced opaque.
fn line_color_linear(color: &impeller2_wkt::Color) -> Vec4 {
    let linear = Color::srgba(color.r, color.g, color.b, color.a).to_linear();
    Vec4::new(linear.red, linear.green, linear.blue, 1.0)
}

/// Resolve a `line_3d` color: an explicit KDL color uses the line's own color,
/// otherwise the timeline played/future trail colors are used at render time.
fn line_color_and_source(color: &Option<impeller2_wkt::Color>) -> (Vec4, gpu::LineColorSource) {
    match color {
        Some(color) => (line_color_linear(color), gpu::LineColorSource::Explicit),
        None => (Vec4::ZERO, gpu::LineColorSource::Timeline),
    }
}

pub fn sync_line_plot_3d(
    line_plot_3d_query: Query<(Entity, &Line3d), Without<gpu::LineHandles>>,
    mut uniforms: Query<
        (&Line3d, &mut LineUniform, &mut gpu::LineColorSource),
        With<gpu::LineHandles>,
    >,
    mut lines: ResMut<Assets<Line>>,
    mut commands: Commands,
    eql_ctx: Res<EqlContext>,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    metadata_store: Res<ComponentMetadataRegistry>,
    geo_ctx: Option<Res<GeoContext>>,
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

        let (color, color_source) = line_color_and_source(&line_plot.color);
        if let Ok(mut entity) = commands.get_entity(entity) {
            entity.try_insert((
                gpu::LineHandles([x, y, z]),
                LineUniform {
                    line_width: line_plot.line_width,
                    color,
                    depth_bias: 0.0,
                    model: Mat4::IDENTITY,
                    perspective: if line_plot.perspective { 1 } else { 0 },
                    #[cfg(target_arch = "wasm32")]
                    _padding: Default::default(),
                },
                color_source,
                LineConfig {
                    render_layers: RenderLayers::layer(crate::plugins::gizmos::GIZMO_RENDER_LAYER),
                },
                Transform::default(),
                #[cfg(feature = "big_space")]
                crate::spatial::GridCell::default(),
            ));
            if let Some(frame) = line_plot.frame {
                // Absolute: the line's vertex data is raw frame coordinates, so
                // its transform must carry the frame -> Bevy basis change.
                entity.try_insert(GeoRotation::absolute(frame, DQuat::IDENTITY));
            }
        }
    }
    for (line_plot, mut uniform, mut color_source) in uniforms.iter_mut() {
        let (color, source) = line_color_and_source(&line_plot.color);
        uniform.color = color;
        uniform.line_width = line_plot.line_width;
        uniform.perspective = if line_plot.perspective { 1 } else { 0 };
        *color_source = source;
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
