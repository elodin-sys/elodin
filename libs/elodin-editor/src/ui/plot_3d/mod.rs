use bevy::{
    app::{PostUpdate, Update},
    asset::{Assets, Handle},
    camera::visibility::RenderLayers,
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    math::{DVec3, Vec4},
    prelude::{Color, GlobalTransform, IntoScheduleConfigs, Transform, warn_once},
    transform::TransformSystems,
};
use bevy_geo_frames::GeoPosition;
use impeller2_bevy::ComponentMetadataRegistry;
use impeller2_wkt::Line3d;

use gpu::{LineConfig, LineUniform};

use super::plot::{CollectedGraphData, Line, PlotDataComponent, queue_timestamp_read};
use crate::{EqlContext, ui::schematic::EqlExt};

pub mod gpu;

/// Convert a schematic (sRGB) color into the linear RGBA the line pipeline
/// renders, keeping it consistent with meshes/gizmos. Alpha is preserved so a
/// KDL `color`/`future_color` can set per-line opacity. An explicit
/// `future_color` alpha is used as-is; only fallback futures get the default
/// fade (see `LineTrailColors::resolve`).
fn line_color_linear(color: &impeller2_wkt::Color) -> Vec4 {
    let linear = Color::srgba(color.r, color.g, color.b, color.a).to_linear();
    Vec4::new(linear.red, linear.green, linear.blue, linear.alpha)
}

/// Resolve a `line_3d`'s played/future trail colors from its KDL `color`/
/// `future_color`. `None` entries fall back to the timeline colors at render
/// time (see `extract_lines`).
fn line_trail_colors(line_plot: &Line3d) -> gpu::LineTrailColors {
    gpu::LineTrailColors {
        played: line_plot.color.as_ref().map(line_color_linear),
        future: line_plot.future_color.as_ref().map(line_color_linear),
    }
}

/// Keep `GeoPosition` aligned with the LineTree's first sample (frame coords).
/// GPU vertices are `p - first`; the entity pose must use the same first point
/// so the trail lands in world space.
///
/// Short selected-time windows rebuild each LineTree to the visible range, so
/// that first sample slides as the window rolls. The pose must track it —
/// writing once at handle insert strands the trail at the original start while
/// vertices re-anchor to the window (trail appears to vanish near the craft).
/// Full-range recordings keep a stable first sample, so this is a no-op write
/// after the initial sync and does not reintroduce ECEF jitter.
///
/// Scheduled after [`queue_timestamp_read`] so a rolling-window rebuild and the
/// pose update land in the same frame (before PostUpdate geo→Transform).
fn sync_line_3d_anchor(
    mut lines: Query<(&gpu::LineHandles, &mut GeoPosition), With<Line3d>>,
    line_assets: Res<Assets<Line>>,
) {
    for (handles, mut geo) in &mut lines {
        let Some(x) = line_assets
            .get(&handles.0[0])
            .and_then(|l| l.data.first_sample())
        else {
            continue;
        };
        let Some(y) = line_assets
            .get(&handles.0[1])
            .and_then(|l| l.data.first_sample())
        else {
            continue;
        };
        let Some(z) = line_assets
            .get(&handles.0[2])
            .and_then(|l| l.data.first_sample())
        else {
            continue;
        };
        let first = DVec3::new(x as f64, y as f64, z as f64);
        if geo.1 != first {
            geo.1 = first;
        }
    }
}

pub fn sync_line_plot_3d(
    line_plot_3d_query: Query<(Entity, &Line3d), Without<gpu::LineHandles>>,
    mut uniforms: Query<
        (
            Entity,
            &Line3d,
            &mut LineUniform,
            Option<&mut gpu::LineTrailColors>,
        ),
        With<gpu::LineHandles>,
    >,
    mut commands: Commands,
    eql_ctx: Res<EqlContext>,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    metadata_store: Res<ComponentMetadataRegistry>,
    #[cfg(feature = "big_space")] root: Option<Res<crate::spatial::BigSpaceRootEntity>>,
) {
    for (entity, line_plot) in line_plot_3d_query.iter() {
        // Parse and compile the EQL expression
        let parsed = match eql_ctx.0.parse_str(&line_plot.eql) {
            Ok(expr) => expr,
            Err(e) => {
                // TODO: Consider changing this to a warn once per error value.
                warn_once!(
                    "Failed to parse Line3D EQL expression '{}': {}",
                    line_plot.eql,
                    e
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

        let trail = line_trail_colors(line_plot);
        if let Ok(mut entity) = commands.get_entity(entity) {
            // Pose comes from GeoPosition(first sample) + GeoRotation::absolute;
            // vertices are frame-relative to that first point.
            entity.try_insert((
                gpu::LineHandles([x, y, z]),
                LineUniform {
                    line_width: line_plot.line_width,
                    color: trail.played.unwrap_or(Vec4::ZERO),
                    depth_bias: 0.0,
                    model: bevy::math::Mat4::IDENTITY,
                    perspective: if line_plot.perspective { 1 } else { 0 },
                    #[cfg(target_arch = "wasm32")]
                    _padding: Default::default(),
                },
                trail,
                LineConfig {
                    render_layers: RenderLayers::layer(crate::plugins::gizmos::GIZMO_RENDER_LAYER),
                },
                Transform::default(),
                GlobalTransform::default(),
                #[cfg(feature = "big_space")]
                crate::spatial::GridCell::default(),
            ));
            #[cfg(feature = "big_space")]
            crate::spatial::parent_under_big_space(&mut entity, root.as_deref());
        }
    }
    for (entity, line_plot, mut uniform, trail) in uniforms.iter_mut() {
        let next = line_trail_colors(line_plot);
        uniform.color = next.played.unwrap_or(Vec4::ZERO);
        uniform.line_width = line_plot.line_width;
        uniform.perspective = if line_plot.perspective { 1 } else { 0 };
        // Entities that have handles but lost their trail colors (e.g. an older
        // build) still get width/perspective/color re-applied; re-attach the
        // trail colors so rendering doesn't silently fall back to defaults.
        match trail {
            Some(mut trail) => *trail = next,
            None => {
                if let Ok(mut entity) = commands.get_entity(entity) {
                    entity.try_insert(next);
                }
            }
        }
    }
}

pub struct LinePlot3dPlugin;

impl bevy::app::Plugin for LinePlot3dPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .add_plugins(gpu::Plot3dGpuPlugin)
            .add_systems(Update, sync_line_plot_3d)
            // After SeriesStore→LineTree projection so rolling windows update
            // the anchor in the same frame the tree's first sample slides.
            .add_systems(
                Update,
                sync_line_3d_anchor
                    .after(queue_timestamp_read)
                    .after(sync_line_plot_3d),
            )
            // Re-apply geo→Transform after Update may have moved the anchor;
            // `update_uniform_model` (after Propagate) then copies a matching model.
            .add_systems(
                PostUpdate,
                (
                    #[cfg(not(feature = "big_space"))]
                    bevy_geo_frames::apply_transforms,
                    bevy_geo_frames::apply_geo_rotation,
                    #[cfg(feature = "big_space")]
                    crate::spatial::apply_big_translation,
                )
                    .chain()
                    .before(TransformSystems::Propagate),
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_color_linear_preserves_alpha() {
        // A KDL color/future_color alpha must survive into the line uniform
        // (sRGB->linear leaves alpha untouched). An explicit future_color keeps
        // this alpha as-is; fallback futures get the default fade in `resolve`.
        let color = impeller2_wkt::Color::rgba(1.0, 1.0, 1.0, 0.25);
        assert_eq!(line_color_linear(&color).w, 0.25);
    }

    #[test]
    fn rolling_window_anchor_must_match_entity_pose() {
        // GPU vertices are `p - first_visible`. Entity pose must use that same
        // first point; a stale recording-start pose offsets the trail by
        // (start - first_visible) and the path leaves the craft.
        let recording_start = DVec3::new(0.0, 0.0, 0.0);
        let window_first = DVec3::new(100.0, 0.0, 50.0);
        let tip = DVec3::new(120.0, 0.0, 60.0);
        let tip_local = tip - window_first;

        let stale_placed = recording_start + tip_local;
        assert!((stale_placed - tip).length() > 10.0);

        let synced_placed = window_first + tip_local;
        assert!((synced_placed - tip).length() < 1e-9);
    }
}
