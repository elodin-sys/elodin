//! `point_trails`: N point trajectories from one flat `3 × N` f64 component.
//!
//! Trails go through the `line_3d` pipeline as one strip per point, separated
//! by the NaN slot, so each color group is a single draw. Heads are children
//! that share one mesh and one material per group, which lets Bevy batch them.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Bound, Range};

use bevy::{
    camera::visibility::RenderLayers,
    ecs::system::SystemState,
    math::{DVec3, Vec4},
    prelude::*,
    render::{
        ExtractSchedule, MainWorld, RenderApp,
        renderer::RenderDevice,
        sync_world::{MainEntity, TemporaryRenderEntity},
    },
};
use bevy_geo_frames::GeoPosition;
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::TelemetryCache;
use impeller2_wkt::{ComponentValue, CurrentTimestamp, PointTrails, PointTrailsHeadShape};
use nox::ArrayBuf;

use super::gpu::{
    GpuLine, LineConfig, LineIndexLayout, LineUniform, LineValuesLayout, build_gpu_line,
};
use crate::{
    BevyExt, SelectedTimeRange,
    ui::{
        plot::gpu::INDEX_BUFFER_LEN,
        timeline::{LatestFollow, TimelineSettings},
        widgets::SystemStateExt,
    },
};

/// Position window, status sample, and geometry settings behind the strips.
type SyncKey = (Timestamp, Timestamp, usize, Option<Timestamp>, Option<u32>);

#[derive(Clone, Copy, PartialEq)]
struct PointTrailsStyle {
    head_size: f32,
    head_shape: PointTrailsHeadShape,
    line_width: f32,
    color: impeller2_wkt::Color,
    hit_color: impeller2_wkt::Color,
}

fn point_trails_style(trails: &PointTrails, timeline: &TimelineSettings) -> PointTrailsStyle {
    PointTrailsStyle {
        head_size: trails.head_size,
        head_shape: trails.head_shape,
        line_width: trails.line_width,
        color: trails.color.unwrap_or(timeline.played_color),
        hit_color: trails.hit_color.unwrap_or(impeller2_wkt::Color::RED),
    }
}

fn head_mesh_index(shape: PointTrailsHeadShape) -> usize {
    match shape {
        PointTrailsHeadShape::Cube => 0,
        PointTrailsHeadShape::Sphere => 1,
    }
}

#[derive(Component)]
struct PointTrailsState {
    component_id: ComponentId,
    status_id: Option<ComponentId>,
    meshes: [Handle<Mesh>; 2],
    /// Normal and hit head materials.
    materials: [Handle<StandardMaterial>; 2],
    style: PointTrailsStyle,
    heads: Vec<Entity>,
    key: Option<SyncKey>,
}

/// One strip per point, trail-major.
#[derive(Default)]
struct StripGroup {
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    strip_ends: Vec<usize>,
}

/// Trail samples handed to the render world: normal and hit groups.
#[derive(Component, Default)]
struct PointTrailsStrips {
    groups: [StripGroup; 2],
    anchor: DVec3,
    hit_color: Vec4,
    generation: u64,
}

/// Samples per trail such that every trail plus its NaN break fits one index buffer.
fn trail_sample_budget(n_points: usize) -> usize {
    ((INDEX_BUFFER_LEN - 1) / n_points.max(1))
        .saturating_sub(1)
        .max(2)
}

/// Evenly spaced sample indices that always keep the first and last sample.
fn downsample(count: usize, budget: usize) -> Vec<usize> {
    if count <= budget {
        return (0..count).collect();
    }
    let last = count - 1;
    (0..budget).map(|k| k * last / (budget - 1)).collect()
}

/// Samples in `(start, end]` plus the last one at or before `start`, so sparse
/// (record-on-change) series still draw from the window start.
fn window_samples<'a>(
    series: &'a BTreeMap<Timestamp, ComponentValue>,
    range: &Range<Timestamp>,
) -> Vec<(Timestamp, &'a [f64])> {
    series
        .range(..=range.start)
        .next_back()
        .into_iter()
        .chain(series.range((Bound::Excluded(range.start), Bound::Included(range.end))))
        .filter_map(|(ts, value)| match value {
            ComponentValue::F64(array) => Some((*ts, array.buf.as_buf())),
            _ => None,
        })
        .collect()
}

fn point(values: &[f64], n: usize, i: usize) -> DVec3 {
    DVec3::new(values[i], values[n + i], values[2 * n + i])
}

fn trail_points(
    samples: &[(Timestamp, &[f64])],
    n: usize,
    index: usize,
    max_length: Option<f64>,
    budget: usize,
) -> Vec<DVec3> {
    let Some((_, latest)) = samples.last() else {
        return Vec::new();
    };
    let mut points = vec![point(latest, n, index)];
    let mut newer = points[0];
    let mut remaining = max_length.unwrap_or(f64::INFINITY);

    for (_, values) in samples[..samples.len() - 1].iter().rev() {
        let older = point(values, n, index);
        let delta = older - newer;
        let segment = delta.length();
        if segment > remaining {
            points.push(newer + delta * (remaining / segment));
            break;
        }
        points.push(older);
        remaining -= segment;
        newer = older;
        if remaining <= 0.0 {
            break;
        }
    }
    points.reverse();

    downsample(points.len(), budget)
        .into_iter()
        .map(|i| points[i])
        .collect()
}

/// Group index per point: 1 where the status element is nonzero.
fn hit_groups(status: Option<&ComponentValue>, n: usize) -> Vec<usize> {
    let mut groups = vec![0; n];
    if let Some(status) = status {
        for (group, element) in groups.iter_mut().zip(status.iter()) {
            *group = usize::from(element.as_f64() != 0.0);
        }
    }
    groups
}

fn init_point_trails(
    trails: Query<(Entity, &PointTrails), Without<PointTrailsState>>,
    timeline: Res<TimelineSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for (entity, trails) in &trails {
        let style = point_trails_style(trails, &timeline);
        let mut material =
            |color| materials.add(impeller2_wkt::Material::with_color(color).into_bevy());
        commands.entity(entity).insert((
            PointTrailsState {
                component_id: ComponentId::new(trails.component.trim()),
                status_id: trails.status.as_deref().map(|s| ComponentId::new(s.trim())),
                meshes: [
                    meshes.add(Cuboid::from_length(1.0)),
                    meshes.add(Sphere::new(0.5)),
                ],
                materials: [material(style.color), material(style.hit_color)],
                style,
                heads: Vec::new(),
                key: None,
            },
            PointTrailsStrips {
                hit_color: super::line_color_linear(&style.hit_color),
                ..default()
            },
            LineUniform::new(
                style.line_width,
                Color::srgba(style.color.r, style.color.g, style.color.b, style.color.a),
            ),
            LineConfig {
                render_layers: RenderLayers::layer(crate::plugins::gizmos::GIZMO_RENDER_LAYER),
            },
        ));
    }
}

/// Visible history up to the playhead, matching `extract_lines`' played range.
fn played_range(
    selected: &SelectedTimeRange,
    current: &CurrentTimestamp,
    live_follow: &LatestFollow,
) -> Option<Range<Timestamp>> {
    let selected = &selected.0;
    if selected.start.0 == i64::MIN || selected.end.0 == i64::MAX {
        return None;
    }
    let end = if live_follow.0 {
        selected.end
    } else {
        selected.end.min(current.0)
    };
    (selected.start <= end).then_some(selected.start..end)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn sync_point_trails(
    mut trails: Query<(
        Entity,
        &PointTrails,
        &mut PointTrailsState,
        &mut PointTrailsStrips,
        &mut GeoPosition,
        &mut Visibility,
        &mut LineUniform,
    )>,
    mut heads: Query<
        (
            &mut Transform,
            &mut Mesh3d,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        Without<PointTrails>,
    >,
    cache: Res<TelemetryCache>,
    selected: Res<SelectedTimeRange>,
    current: Res<CurrentTimestamp>,
    live_follow: Res<LatestFollow>,
    timeline: Res<TimelineSettings>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    let range = played_range(&selected, &current, &live_follow);
    for (entity, trails, mut state, mut strips, mut geo, mut visibility, mut uniform) in &mut trails
    {
        let style = point_trails_style(trails, &timeline);
        let style_changed = state.style != style;
        if style_changed {
            state.style = style;
            for (handle, color) in state.materials.iter().zip([style.color, style.hit_color]) {
                if let Some(mut material) = materials.get_mut(handle) {
                    *material = impeller2_wkt::Material::with_color(color).into_bevy();
                }
            }
            uniform.line_width = style.line_width;
            uniform.color = super::line_color_linear(&style.color);
            strips.hit_color = super::line_color_linear(&style.hit_color);
        }
        let samples = match (&range, cache.series(&state.component_id)) {
            (Some(range), Some(series)) => window_samples(series, range),
            _ => Vec::new(),
        };
        let n = samples.last().map_or(0, |(_, values)| values.len() / 3);
        let (Some(range), true) = (&range, n > 0) else {
            visibility.set_if_neq(Visibility::Hidden);
            if strips.groups.iter().any(|group| !group.xs.is_empty()) {
                strips.groups = default();
                strips.generation = strips.generation.wrapping_add(1);
            }
            state.key = None;
            continue;
        };
        visibility.set_if_neq(Visibility::Inherited);
        let status = state
            .status_id
            .and_then(|id| cache.series(&id))
            .and_then(|series| series.range(..=range.end).next_back());
        let key = (
            samples[0].0,
            samples[samples.len() - 1].0,
            samples.len(),
            status.map(|(ts, _)| *ts),
            trails.max_length.map(f32::to_bits),
        );
        if state.key == Some(key) && state.heads.len() == n && !style_changed {
            continue;
        }
        state.key = Some(key);
        let group_of = hit_groups(status.map(|(_, value)| value), n);

        let budget = trail_sample_budget(n);
        let max_length = trails.max_length.map(f64::from);
        let first_trail = trail_points(&samples, n, 0, max_length, budget);
        let anchor = first_trail[0];
        if geo.1 != anchor {
            geo.1 = anchor;
        }

        let latest = samples[samples.len() - 1].1;
        let head_offset = |i: usize| (point(latest, n, i) - anchor).as_vec3();
        let mesh = state.meshes[head_mesh_index(style.head_shape)].clone();
        if state.heads.len() == n {
            for (i, &head) in state.heads.iter().enumerate() {
                if let Ok((mut transform, mut head_mesh, mut material)) = heads.get_mut(head) {
                    transform.translation = head_offset(i);
                    transform.scale = Vec3::splat(style.head_size);
                    if head_mesh.0 != mesh {
                        head_mesh.0 = mesh.clone();
                    }
                    if material.0 != state.materials[group_of[i]] {
                        material.0 = state.materials[group_of[i]].clone();
                    }
                }
            }
        } else {
            for head in state.heads.drain(..) {
                commands.entity(head).despawn();
            }
            let scale = Vec3::splat(style.head_size);
            state.heads = (0..n)
                .map(|i| {
                    commands
                        .spawn((
                            Mesh3d(mesh.clone()),
                            MeshMaterial3d(state.materials[group_of[i]].clone()),
                            Transform::from_translation(head_offset(i)).with_scale(scale),
                            ChildOf(entity),
                        ))
                        .id()
                })
                .collect();
        }

        let strips = &mut *strips;
        strips.groups = default();
        for (i, &group) in group_of.iter().enumerate() {
            let group = &mut strips.groups[group];
            for p in trail_points(&samples, n, i, max_length, budget) {
                group.xs.push(p.x);
                group.ys.push(p.y);
                group.zs.push(p.z);
            }
            group.strip_ends.push(group.xs.len());
        }
        strips.anchor = anchor;
        strips.generation = strips.generation.wrapping_add(1);
    }
}

type ExtractQuery = Query<
    'static,
    'static,
    (
        Entity,
        &'static PointTrails,
        &'static PointTrailsStrips,
        &'static LineUniform,
        &'static LineConfig,
    ),
>;

#[derive(Resource)]
struct PointTrailsExtractState(SystemState<ExtractQuery>);

impl FromWorld for PointTrailsExtractState {
    fn from_world(world: &mut World) -> Self {
        Self(SystemState::new(world))
    }
}

/// Uploaded strips per main-world entity and group, rebuilt only when the
/// strips change.
#[derive(Resource, Default)]
struct PointTrailsGpuCache(HashMap<(Entity, usize), (u64, GpuLine)>);

fn extract_point_trails(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    mut cache: ResMut<PointTrailsGpuCache>,
    render_device: Res<RenderDevice>,
    values_layout: Res<LineValuesLayout>,
    index_layout: Res<LineIndexLayout>,
) {
    main_world.resource_scope(|world, mut state: Mut<PointTrailsExtractState>| {
        let query = state.0.params(world);
        cache.0.retain(|(entity, _), _| query.contains(*entity));
        for (entity, trails, strips, uniform, config) in &query {
            for (index, group) in strips.groups.iter().enumerate() {
                let key = (entity, index);
                let cached = cache
                    .0
                    .get(&key)
                    .filter(|(generation, _)| *generation == strips.generation);
                let gpu_line = match cached {
                    Some((_, gpu_line)) => gpu_line.clone(),
                    None => {
                        let Some((gpu_line, residual_too_large)) = build_gpu_line(
                            [&group.xs, &group.ys, &group.zs],
                            &group.strip_ends,
                            strips.anchor,
                            &render_device,
                            &values_layout,
                            &index_layout,
                        ) else {
                            cache.0.remove(&key);
                            continue;
                        };
                        if residual_too_large {
                            warn_once!(
                                "point_trails anchor residual is large enough that f32 ULP is visible: {}",
                                trails.component
                            );
                        }
                        cache
                            .0
                            .insert(key, (strips.generation, gpu_line.clone()));
                        gpu_line
                    }
                };
                let mut uniform = *uniform;
                if index == 1 {
                    uniform.color = strips.hit_color;
                }
                commands.spawn((
                    MainEntity::from(entity),
                    config.clone(),
                    uniform,
                    gpu_line,
                    TemporaryRenderEntity,
                ));
            }
        }
    });
}

pub struct PointTrailsPlugin;

impl Plugin for PointTrailsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PointTrailsExtractState>().add_systems(
            Update,
            (init_point_trails, sync_point_trails)
                .chain()
                .after(crate::ui::plot::queue_timestamp_read),
        );
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<PointTrailsGpuCache>()
            .add_systems(ExtractSchedule, extract_point_trails);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_tracks_timeline_color_until_overridden() {
        let mut trails = PointTrails {
            component: "points".into(),
            status: None,
            head_size: 0.1,
            head_shape: PointTrailsHeadShape::Sphere,
            line_width: 2.0,
            max_length: None,
            color: None,
            hit_color: None,
            frame: None,
            node_id: default(),
        };
        let mut timeline = TimelineSettings {
            played_color: impeller2_wkt::Color::GREEN,
            ..default()
        };
        assert_eq!(
            point_trails_style(&trails, &timeline).color,
            impeller2_wkt::Color::GREEN
        );

        trails.color = Some(impeller2_wkt::Color::BLUE);
        trails.hit_color = Some(impeller2_wkt::Color::YELLOW);
        trails.head_size = 0.2;
        trails.head_shape = PointTrailsHeadShape::Cube;
        trails.line_width = 3.0;
        timeline.played_color = impeller2_wkt::Color::RED;
        let style = point_trails_style(&trails, &timeline);
        assert_eq!(style.color, impeller2_wkt::Color::BLUE);
        assert_eq!(style.hit_color, impeller2_wkt::Color::YELLOW);
        assert_eq!(style.head_size, 0.2);
        assert_eq!(style.head_shape, PointTrailsHeadShape::Cube);
        assert_eq!(style.line_width, 3.0);
    }

    #[test]
    fn full_census_fits_one_index_buffer() {
        let n = 695;
        let budget = trail_sample_budget(n);
        assert_eq!(budget, 187);
        // Leading NaN slot + (samples + break) per trail.
        assert!(1 + n * (budget + 1) <= INDEX_BUFFER_LEN);
    }

    #[test]
    fn downsample_keeps_endpoints_and_budget() {
        assert_eq!(downsample(3, 200), vec![0, 1, 2]);
        let picked = downsample(7_500, 200);
        assert_eq!(picked.len(), 200);
        assert_eq!(picked.first(), Some(&0));
        assert_eq!(picked.last(), Some(&7_499));
        assert!(picked.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn max_length_interpolates_inside_a_sparse_segment() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [10.0, 0.0, 0.0];
        let p2 = [20.0, 0.0, 0.0];
        let samples = [
            (Timestamp(0), p0.as_slice()),
            (Timestamp(10), p1.as_slice()),
            (Timestamp(20), p2.as_slice()),
        ];
        assert_eq!(
            trail_points(&samples, 1, 0, Some(3.0), 100),
            [DVec3::new(17.0, 0.0, 0.0), DVec3::new(20.0, 0.0, 0.0)]
        );
    }

    #[test]
    fn max_length_uses_cumulative_curved_path() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [2.0, 0.0, 0.0];
        let p2 = [2.0, 2.0, 0.0];
        let samples = [
            (Timestamp(0), p0.as_slice()),
            (Timestamp(10), p1.as_slice()),
            (Timestamp(20), p2.as_slice()),
        ];
        assert_eq!(
            trail_points(&samples, 1, 0, Some(3.0), 100),
            [
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(2.0, 2.0, 0.0)
            ]
        );
    }

    #[test]
    fn nonzero_status_moves_points_to_the_hit_group() {
        let status = ComponentValue::U64(nox::array![0u64, 3010, 0, 3020].to_dyn());
        assert_eq!(hit_groups(Some(&status), 4), vec![0, 1, 0, 1]);
        assert_eq!(hit_groups(None, 3), vec![0, 0, 0]);
    }

    #[test]
    fn sparse_series_carries_last_sample_into_window() {
        let value = |x: f64| ComponentValue::F64(nox::array![x, 0.0, 0.0].to_dyn());
        let series: BTreeMap<_, _> = [
            (Timestamp(0), value(1.0)),
            (Timestamp(50), value(2.0)),
            (Timestamp(90), value(3.0)),
        ]
        .into_iter()
        .collect();
        let samples = window_samples(&series, &(Timestamp(10)..Timestamp(60)));
        let xs: Vec<f64> = samples.iter().map(|(_, v)| v[0]).collect();
        assert_eq!(xs, vec![1.0, 2.0]);
    }
}
