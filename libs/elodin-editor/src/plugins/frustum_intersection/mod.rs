//! Frustum∩ellipsoid intersection: volume coverage, 2D projection, and tint overlay.
//!
//! See `frustum_intersection/README.md` for feature overview and Inspector controls.

mod coverage_db;
mod frustum_tint_material;
mod projection;
mod volume;

use super::frustum_common::{MainViewportQueryItem, color_component_to_u8, frustum_local_points};
use crate::{
    MainCamera,
    object_3d::{Object3DMeshChild, Object3DState},
};
use bevy::asset::embedded_asset;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::pbr::MaterialPlugin;
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use bevy_mat3_material::Mat3Material;
use coverage_db::write_coverage_to_db;
use frustum_tint_material::{FrustumTintExt, FrustumTintMaterial, FrustumTintParams};
use projection::build_projection_mesh;
use std::collections::HashMap;
use volume::{
    FrustumVolume, SURFACE_EPS, aabb_overlap, compute_intersection_volume, ellipsoid_world_extent,
    frustum_planes, plane_to_vec4, points_aabb,
};

#[derive(Clone)]
struct EllipsoidVolume {
    entity: Entity,
    mesh_child: Entity,
    mesh_handle: Handle<Mesh>,
    center: Vec3,
    rotation: Quat,
    radii: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
    base_color: impeller2_wkt::Color,
    material_target: Option<EllipsoidMaterialTarget>,
}

/// How we apply tint to an ellipsoid: Standard/Mat3 materials get runtime edits;
/// FrustumTintSwapped means the mesh uses FrustumTintMaterial (GPU tint) instead.
#[derive(Clone)]
enum EllipsoidMaterialTarget {
    Standard(Handle<StandardMaterial>),
    /// Mesh was swapped to FrustumTintMaterial; tint is handled by the shader, not CPU.
    FrustumTintSwapped,
    Mat3(Handle<Mat3Material>),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct IntersectionMaterialKey {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Resource, Default)]
struct IntersectionMaterialCache {
    materials: HashMap<IntersectionMaterialKey, Handle<StandardMaterial>>,
}

#[derive(Component, Clone, Debug)]
struct IntersectionProjectionCache(PerspectiveProjection);

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FrustumEllipsoidIntersectionVisual {
    source: Entity,
    target: Entity,
    ellipsoid: Entity,
}

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FrustumEllipsoidIntersectionLight {
    key: FrustumEllipsoidIntersectionVisual,
}

#[derive(Component, Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FrustumEllipsoidTintOverlay {
    ellipsoid: Entity,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct IntersectionRatio {
    pub ellipsoid: Entity,
    pub ratio: f32,
}

#[derive(Resource, Default)]
pub struct IntersectionRatios(pub Vec<IntersectionRatio>);

struct DesiredIntersection {
    key: FrustumEllipsoidIntersectionVisual,
    mesh: Mesh,
    render_layers: RenderLayers,
    material: MeshMaterial3d<StandardMaterial>,
}

struct DesiredIntersectionLight {
    key: FrustumEllipsoidIntersectionVisual,
    render_layers: RenderLayers,
    light: PointLight,
    transform: Transform,
}

struct DesiredTintOverlay {
    ellipsoid: Entity,
    mesh_child: Entity,
    mesh: Handle<Mesh>,
    material: MeshMaterial3d<FrustumTintMaterial>,
    render_layers: RenderLayers,
}

struct FrustumSource {
    volume: FrustumVolume,
    color: impeller2_wkt::Color,
}

#[derive(SystemParam)]
struct FrustumIntersectionParams<'w, 's> {
    main_viewports: Query<'w, 's, MainViewportQueryItem, With<MainCamera>>,
    ellipsoids: Query<'w, 's, (Entity, &'static GlobalTransform, &'static Object3DState)>,
    children: Query<'w, 's, &'static Children>,
    mesh_children: Query<'w, 's, (), With<Object3DMeshChild>>,
    mesh_handles: Query<'w, 's, &'static Mesh3d>,
    transforms: Query<'w, 's, &'static Transform>,
    mesh_materials: Query<'w, 's, &'static MeshMaterial3d<StandardMaterial>>,
    mesh_frustum_tint_materials: Query<'w, 's, &'static MeshMaterial3d<FrustumTintMaterial>>,
    mesh_mat3_materials: Query<'w, 's, &'static MeshMaterial3d<Mat3Material>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    frustum_tint_materials: ResMut<'w, Assets<FrustumTintMaterial>>,
    mat3_materials: ResMut<'w, Assets<Mat3Material>>,
    material_cache: ResMut<'w, IntersectionMaterialCache>,
    projection_cache: Query<'w, 's, &'static IntersectionProjectionCache>,
    existing_visuals: Query<
        'w,
        's,
        (
            Entity,
            &'static FrustumEllipsoidIntersectionVisual,
            &'static Mesh3d,
        ),
    >,
    existing_lights:
        Query<'w, 's, (Entity, &'static FrustumEllipsoidIntersectionLight), With<PointLight>>,
    existing_tint_overlays: Query<'w, 's, (Entity, &'static FrustumEllipsoidTintOverlay)>,
    swapped: ResMut<'w, EllipsoidTintSwapped>,
}

pub struct FrustumIntersectionPlugin;

/// Tracks mesh children that were swapped to FrustumTintMaterial; maps to original StandardMaterial handle.
#[derive(Resource, Default)]
struct EllipsoidTintSwapped(HashMap<Entity, Handle<StandardMaterial>>);

impl Plugin for FrustumIntersectionPlugin {
    fn build(&self, app: &mut App) {
        use impeller2_bevy::AppExt;
        app.init_resource::<IntersectionMaterialCache>()
            .init_resource::<IntersectionRatios>()
            .init_resource::<EllipsoidTintSwapped>()
            .add_plugins(MaterialPlugin::<FrustumTintMaterial>::default())
            .add_impeller_component::<impeller2_wkt::FrustumCoverage>()
            .add_systems(
                PostUpdate,
                (
                    draw_frustum_ellipsoid_intersections.after(TransformSystems::Propagate),
                    write_coverage_to_db.after(draw_frustum_ellipsoid_intersections),
                ),
            );
        embedded_asset!(app, "frustum_tint.wgsl");
    }
}

/// Alpha for the 2D projection mesh overlay on the far plane.
const PROJECTION_ALPHA: u8 = 230;
/// Emissive multiplier for projection material visibility.
const PROJECTION_EMISSIVE_STRENGTH: f32 = 1.4;
/// Depth bias so the projection mesh renders in front of far-plane geometry.
const PROJECTION_DEPTH_BIAS: f32 = -8.0;
/// Intensity of the point light placed at each intersection for visual emphasis.
const INTERSECTION_LIGHT_INTENSITY: f32 = 1500.0;
/// Ellipsoid tint: min blend toward warm when partially inside frustum.
const ELLIPSOID_TINT_MIN_BLEND: f32 = 0.20;
/// Ellipsoid tint: max blend toward green when fully inside frustum.
const ELLIPSOID_TINT_MAX_BLEND: f32 = 0.65;
/// Emissive scale for tinted ellipsoids (increases with coverage ratio).
const ELLIPSOID_TINT_EMISSIVE_SCALE: f32 = 0.12;

fn projection_material_for_color(
    color: impeller2_wkt::Color,
    materials: &mut Assets<StandardMaterial>,
    cache: &mut IntersectionMaterialCache,
) -> Handle<StandardMaterial> {
    let key = IntersectionMaterialKey {
        r: color_component_to_u8(color.r),
        g: color_component_to_u8(color.g),
        b: color_component_to_u8(color.b),
        a: PROJECTION_ALPHA,
    };
    if let Some(handle) = cache.materials.get(&key) {
        return handle.clone();
    }

    let material = materials.add(StandardMaterial {
        base_color: Color::srgba_u8(key.r, key.g, key.b, PROJECTION_ALPHA),
        emissive: Color::srgba(
            color.r * PROJECTION_EMISSIVE_STRENGTH,
            color.g * PROJECTION_EMISSIVE_STRENGTH,
            color.b * PROJECTION_EMISSIVE_STRENGTH,
            1.0,
        )
        .into(),
        cull_mode: None,
        unlit: true,
        double_sided: true,
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.3,
        depth_bias: PROJECTION_DEPTH_BIAS,
        ..Default::default()
    });
    cache.materials.insert(key, material.clone());
    material
}

/// Place a point light near the ellipsoid surface, toward the camera, to emphasize the intersection.
fn intersection_light_for(
    frustum: &FrustumVolume,
    ellipsoid: &EllipsoidVolume,
    color: impeller2_wkt::Color,
) -> (PointLight, Transform) {
    let radius = ellipsoid.radii.max_element().max(0.05);
    let to_camera = frustum.camera_pos - ellipsoid.center;
    let direction = if to_camera.length_squared() > SURFACE_EPS {
        to_camera.normalize()
    } else {
        Vec3::Z
    };
    let translation = ellipsoid.center + direction * radius * 0.8;
    let light = PointLight {
        color: Color::srgb(color.r, color.g, color.b),
        intensity: INTERSECTION_LIGHT_INTENSITY,
        range: (radius * 8.0).max(1.0),
        shadows_enabled: false,
        ..Default::default()
    };
    (light, Transform::from_translation(translation))
}

fn ellipsoid_tinted_color(base: impeller2_wkt::Color, ratio: f32) -> (Color, Color) {
    let ratio = ratio.clamp(0.0, 1.0);
    if ratio <= SURFACE_EPS {
        return (
            Color::srgba(base.r, base.g, base.b, base.a),
            Color::srgba(0.0, 0.0, 0.0, 1.0),
        );
    }

    // Partial overlap trends warm, near-full overlap trends green.
    let partial = Vec3::new(1.0, 0.58, 0.12);
    let inside = Vec3::new(0.18, 0.92, 0.34);
    let target = partial.lerp(inside, ratio);
    let blend =
        ELLIPSOID_TINT_MIN_BLEND + (ELLIPSOID_TINT_MAX_BLEND - ELLIPSOID_TINT_MIN_BLEND) * ratio;

    let base_rgb = Vec3::new(base.r, base.g, base.b);
    let rgb = base_rgb.lerp(target, blend);
    let alpha = base.a.max(0.35);

    let tinted = Color::srgba(rgb.x, rgb.y, rgb.z, alpha);
    let emissive = Color::srgba(
        rgb.x * ratio * ELLIPSOID_TINT_EMISSIVE_SCALE,
        rgb.y * ratio * ELLIPSOID_TINT_EMISSIVE_SCALE,
        rgb.z * ratio * ELLIPSOID_TINT_EMISSIVE_SCALE,
        1.0,
    );
    (tinted, emissive)
}

fn apply_ellipsoid_tint(
    target: &EllipsoidMaterialTarget,
    base: impeller2_wkt::Color,
    ratio: f32,
    standard_materials: &mut Assets<StandardMaterial>,
    mat3_materials: &mut Assets<Mat3Material>,
) {
    let (tinted, emissive) = ellipsoid_tinted_color(base, ratio);
    match target {
        EllipsoidMaterialTarget::FrustumTintSwapped => {
            // Tint is applied by FrustumTintMaterial shader; no CPU-side edit.
        }
        EllipsoidMaterialTarget::Standard(handle) => {
            if let Some(material) = standard_materials.get_mut(handle) {
                material.base_color = tinted;
                material.emissive = emissive.into();
            }
        }
        EllipsoidMaterialTarget::Mat3(handle) => {
            if let Some(material) = mat3_materials.get_mut(handle) {
                material.base.base_color = tinted;
                material.base.emissive = emissive.into();
            }
        }
    }
}

fn cleanup_all(
    params: &FrustumIntersectionParams<'_, '_>,
    ratios: &mut ResMut<IntersectionRatios>,
    commands: &mut Commands,
) {
    for (entity, _, _) in params.existing_visuals.iter() {
        commands.entity(entity).despawn();
    }
    for (entity, _) in params.existing_lights.iter() {
        commands.entity(entity).despawn();
    }
    for (entity, _) in params.existing_tint_overlays.iter() {
        commands.entity(entity).despawn();
    }
    ratios.0.clear();
}

fn reset_ellipsoid_tints(params: &mut FrustumIntersectionParams<'_, '_>, commands: &mut Commands) {
    for (mesh_child, original) in params.swapped.0.drain() {
        commands
            .entity(mesh_child)
            .remove::<MeshMaterial3d<FrustumTintMaterial>>()
            .insert(MeshMaterial3d(original));
    }
    for (entity, _global_transform, object_state) in params.ellipsoids.iter() {
        let impeller2_wkt::Object3DMesh::Ellipsoid { color, .. } = object_state.data.mesh else {
            continue;
        };
        let Ok(children) = params.children.get(entity) else {
            continue;
        };
        let Some(mesh_child_entity) = children
            .iter()
            .find(|child| params.mesh_children.contains(*child))
        else {
            continue;
        };

        let material_target = if let Ok(mat) = params.mesh_materials.get(mesh_child_entity) {
            Some(EllipsoidMaterialTarget::Standard(mat.0.clone()))
        } else if let Ok(mat) = params.mesh_mat3_materials.get(mesh_child_entity) {
            Some(EllipsoidMaterialTarget::Mat3(mat.0.clone()))
        } else {
            None
        };

        if let Some(material_target) = material_target {
            apply_ellipsoid_tint(
                &material_target,
                color,
                0.0,
                &mut params.materials,
                &mut params.mat3_materials,
            );
        }
    }
}

fn source_frustum_perspective(
    camera_is_active: bool,
    perspective: &PerspectiveProjection,
    config_aspect: Option<f32>,
    viewport_aspect: Option<f32>,
    projection_cache: Option<&IntersectionProjectionCache>,
) -> PerspectiveProjection {
    if camera_is_active {
        return perspective.clone();
    }

    if let Some(perspective) = projection_cache {
        return perspective.0.clone();
    }

    let mut perspective = perspective.clone();
    if let Some(aspect) = config_aspect.or(viewport_aspect) {
        perspective.aspect_ratio = aspect;
    }
    perspective
}

fn camera_viewport_aspect(camera: &Camera) -> Option<f32> {
    let size = camera.viewport.as_ref()?.physical_size.as_vec2();
    if size.x > 0.0 && size.y > 0.0 {
        Some(size.x / size.y)
    } else {
        None
    }
}

fn draw_frustum_ellipsoid_intersections(
    mut ratios: ResMut<IntersectionRatios>,
    mut params: FrustumIntersectionParams<'_, '_>,
    mut commands: Commands,
) {
    let mut sources = Vec::new();
    let mut targets = Vec::new();

    for (camera_entity, camera, projection, global_transform, config, render_layer_lease) in
        params.main_viewports.iter()
    {
        let Some(config) = config else {
            continue;
        };

        if camera.is_active
            && config.show_frustums
            && (config.show_coverage_in_viewport || config.show_projection_2d)
            && let Some(render_layer_lease) = render_layer_lease
        {
            targets.push((
                camera_entity,
                render_layer_lease.render_layers(),
                config.show_coverage_in_viewport,
                config.show_projection_2d,
            ));
        }

        if !config.create_frustum {
            commands
                .entity(camera_entity)
                .remove::<IntersectionProjectionCache>();
            continue;
        }

        let Projection::Perspective(perspective) = projection else {
            commands
                .entity(camera_entity)
                .remove::<IntersectionProjectionCache>();
            continue;
        };

        if camera.is_active {
            commands
                .entity(camera_entity)
                .insert(IntersectionProjectionCache(perspective.clone()));
        }

        let source_perspective = source_frustum_perspective(
            camera.is_active,
            perspective,
            config.aspect,
            camera_viewport_aspect(camera),
            params.projection_cache.get(camera_entity).ok(),
        );
        let Some(local_points) = frustum_local_points(&source_perspective) else {
            continue;
        };
        let world_points = local_points.map(|point| global_transform.transform_point(point));
        let Some(planes) = frustum_planes(&world_points) else {
            continue;
        };
        let (aabb_min, aabb_max) = points_aabb(&world_points);
        sources.push(FrustumSource {
            volume: FrustumVolume {
                source: camera_entity,
                camera_pos: global_transform.translation(),
                far_corners: [
                    world_points[4],
                    world_points[5],
                    world_points[6],
                    world_points[7],
                ],
                planes,
                aabb_min,
                aabb_max,
            },
            color: config.projection_color,
        });
    }

    if sources.is_empty() || targets.is_empty() {
        reset_ellipsoid_tints(&mut params, &mut commands);
        cleanup_all(&params, &mut ratios, &mut commands);
        return;
    }

    let mut ellipsoids = Vec::new();
    for (entity, global_transform, object_state) in params.ellipsoids.iter() {
        let impeller2_wkt::Object3DMesh::Ellipsoid { color, .. } = object_state.data.mesh else {
            continue;
        };

        let Ok(children) = params.children.get(entity) else {
            continue;
        };
        let Some(mesh_child_entity) = children
            .iter()
            .find(|child| params.mesh_children.contains(*child))
        else {
            continue;
        };
        let Ok(mesh_handle) = params.mesh_handles.get(mesh_child_entity) else {
            continue;
        };
        let Ok(child_transform) = params.transforms.get(mesh_child_entity) else {
            continue;
        };

        let material_target = if let Ok(mat) = params.mesh_materials.get(mesh_child_entity) {
            Some(EllipsoidMaterialTarget::Standard(mat.0.clone()))
        } else if params
            .mesh_frustum_tint_materials
            .get(mesh_child_entity)
            .is_ok()
        {
            Some(EllipsoidMaterialTarget::FrustumTintSwapped)
        } else if let Ok(mat) = params.mesh_mat3_materials.get(mesh_child_entity) {
            Some(EllipsoidMaterialTarget::Mat3(mat.0.clone()))
        } else {
            None
        };

        let radii = child_transform.scale.abs().max(Vec3::splat(0.001));
        let (_, rotation, center) = global_transform.to_scale_rotation_translation();
        let extent = ellipsoid_world_extent(rotation, radii);
        ellipsoids.push(EllipsoidVolume {
            entity,
            mesh_child: mesh_child_entity,
            mesh_handle: mesh_handle.0.clone(),
            center,
            rotation,
            radii,
            aabb_min: center - extent,
            aabb_max: center + extent,
            base_color: color,
            material_target,
        });
    }

    if ellipsoids.is_empty() {
        cleanup_all(&params, &mut ratios, &mut commands);
        return;
    }

    ratios.0.clear();
    let mut desired = Vec::new();
    let mut desired_lights = Vec::new();
    let mut desired_tint_overlays = Vec::new();
    let mut coverage_layers = RenderLayers::none();
    for (_, render_layers, show_coverage, _) in &targets {
        if *show_coverage {
            coverage_layers = coverage_layers.union(render_layers);
        }
    }
    let any_target_wants_coverage = coverage_layers.iter().next().is_some();
    let any_target_wants_projection = targets
        .iter()
        .any(|(_, _, _, show_projection)| *show_projection);
    // Per-ellipsoid max coverage ratio across all frustums. Empty when coverage is disabled.
    let mut ellipsoid_max_ratio: HashMap<Entity, f32> = if any_target_wants_coverage {
        ellipsoids
            .iter()
            .map(|ellipsoid| (ellipsoid.entity, 0.0_f32))
            .collect::<HashMap<Entity, f32>>()
    } else {
        HashMap::<Entity, f32>::new()
    };

    for source in &sources {
        let frustum = &source.volume;
        for ellipsoid in &ellipsoids {
            if !aabb_overlap(
                frustum.aabb_min,
                frustum.aabb_max,
                ellipsoid.aabb_min,
                ellipsoid.aabb_max,
            ) {
                continue;
            }

            let bounds_min = frustum.aabb_min.max(ellipsoid.aabb_min);
            let bounds_max = frustum.aabb_max.min(ellipsoid.aabb_max);
            let maybe_ratio = if any_target_wants_coverage {
                compute_intersection_volume(bounds_min, bounds_max, frustum, ellipsoid)
            } else {
                None
            };

            if let Some(ratio) = maybe_ratio {
                let entry = ellipsoid_max_ratio.entry(ellipsoid.entity).or_insert(0.0);
                *entry = (*entry).max(ratio);
            }

            if !any_target_wants_projection {
                continue;
            }
            let Some(mesh) = build_projection_mesh(frustum, ellipsoid) else {
                continue;
            };
            for (target_camera, render_layers, _show_coverage, target_show_projection) in &targets {
                if frustum.source == *target_camera || !target_show_projection {
                    continue;
                }
                let (light, light_transform) =
                    intersection_light_for(frustum, ellipsoid, source.color);
                let material = MeshMaterial3d(projection_material_for_color(
                    source.color,
                    &mut params.materials,
                    &mut params.material_cache,
                ));
                let key = FrustumEllipsoidIntersectionVisual {
                    source: frustum.source,
                    target: *target_camera,
                    ellipsoid: ellipsoid.entity,
                };
                desired.push(DesiredIntersection {
                    key,
                    mesh: mesh.clone(),
                    render_layers: render_layers.clone(),
                    material: material.clone(),
                });
                desired_lights.push(DesiredIntersectionLight {
                    key,
                    render_layers: render_layers.clone(),
                    light,
                    transform: light_transform,
                });
            }
        }
    }
    if any_target_wants_coverage {
        for (&ellipsoid, &ratio) in &ellipsoid_max_ratio {
            ratios.0.push(IntersectionRatio { ellipsoid, ratio });
        }
    }

    for ellipsoid in &ellipsoids {
        let ratio = ellipsoid_max_ratio
            .get(&ellipsoid.entity)
            .copied()
            .unwrap_or(0.0);

        if ratio > SURFACE_EPS && any_target_wants_coverage {
            let first_frustum = sources.iter().find(|source| {
                let frustum = &source.volume;
                aabb_overlap(
                    frustum.aabb_min,
                    frustum.aabb_max,
                    ellipsoid.aabb_min,
                    ellipsoid.aabb_max,
                )
            });
            if let Some(source) = first_frustum {
                let frustum = &source.volume;
                let mut base = match &ellipsoid.material_target {
                    Some(EllipsoidMaterialTarget::Standard(handle)) => {
                        params.materials.get(handle).cloned().unwrap_or_default()
                    }
                    Some(EllipsoidMaterialTarget::Mat3(handle)) => params
                        .mat3_materials
                        .get(handle)
                        .map(|m| m.base.clone())
                        .unwrap_or_default(),
                    _ => StandardMaterial::default(),
                };
                base.alpha_mode = AlphaMode::Blend;
                base.cull_mode = None;
                base.double_sided = true;
                base.depth_bias = PROJECTION_DEPTH_BIAS;

                let planes: [Vec4; 6] = std::array::from_fn(|i| plane_to_vec4(&frustum.planes[i]));
                let (inside_rgb, _) = ellipsoid_tinted_color(ellipsoid.base_color, 1.0);
                let (outside_rgb, _) = ellipsoid_tinted_color(ellipsoid.base_color, 0.01);
                let alpha = ellipsoid.base_color.a.max(0.35);
                let tint_material = FrustumTintMaterial {
                    base,
                    extension: FrustumTintExt {
                        params: FrustumTintParams {
                            planes,
                            inside_color: Vec4::new(
                                inside_rgb.to_srgba().red,
                                inside_rgb.to_srgba().green,
                                inside_rgb.to_srgba().blue,
                                alpha,
                            ),
                            outside_color: Vec4::new(
                                outside_rgb.to_srgba().red,
                                outside_rgb.to_srgba().green,
                                outside_rgb.to_srgba().blue,
                                alpha,
                            ),
                            enabled: 1,
                        },
                    },
                };
                desired_tint_overlays.push(DesiredTintOverlay {
                    ellipsoid: ellipsoid.entity,
                    mesh_child: ellipsoid.mesh_child,
                    mesh: ellipsoid.mesh_handle.clone(),
                    material: MeshMaterial3d(params.frustum_tint_materials.add(tint_material)),
                    render_layers: coverage_layers.clone(),
                });
            }
        }

        match &ellipsoid.material_target {
            Some(EllipsoidMaterialTarget::FrustumTintSwapped) => {
                // Avoid runtime material-type swaps on mesh entities: they can desync
                // Bevy's specialization bookkeeping and panic in `specialize_material_meshes`.
                if let Some(original) = params.swapped.0.remove(&ellipsoid.mesh_child) {
                    commands
                        .entity(ellipsoid.mesh_child)
                        .remove::<MeshMaterial3d<FrustumTintMaterial>>()
                        .insert(MeshMaterial3d(original));
                }
            }
            Some(material_target) => {
                apply_ellipsoid_tint(
                    material_target,
                    ellipsoid.base_color,
                    ratio,
                    &mut params.materials,
                    &mut params.mat3_materials,
                );
            }
            None => {}
        }
    }

    let mut existing_by_key: HashMap<FrustumEllipsoidIntersectionVisual, (Entity, Handle<Mesh>)> =
        HashMap::new();
    for (entity, key, mesh3d) in params.existing_visuals.iter() {
        existing_by_key.insert(*key, (entity, mesh3d.0.clone()));
    }
    let mut existing_lights_by_key: HashMap<FrustumEllipsoidIntersectionVisual, Entity> =
        HashMap::new();
    for (entity, marker) in params.existing_lights.iter() {
        existing_lights_by_key.insert(marker.key, entity);
    }
    let mut existing_tint_overlays_by_ellipsoid: HashMap<Entity, Entity> = HashMap::new();
    for (entity, marker) in params.existing_tint_overlays.iter() {
        existing_tint_overlays_by_ellipsoid.insert(marker.ellipsoid, entity);
    }

    for visual in desired {
        if let Some((entity, mesh_handle)) = existing_by_key.remove(&visual.key) {
            if let Some(mesh_asset) = params.meshes.get_mut(&mesh_handle) {
                *mesh_asset = visual.mesh;
            } else {
                let new_mesh = params.meshes.add(visual.mesh);
                commands.entity(entity).insert(Mesh3d(new_mesh));
            }
            commands
                .entity(entity)
                .insert((visual.render_layers, visual.material));
            commands
                .entity(entity)
                .remove::<Wireframe>()
                .remove::<WireframeColor>();
            continue;
        }

        let mesh_handle = params.meshes.add(visual.mesh);
        commands.spawn((
            Mesh3d(mesh_handle),
            visual.material,
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            Visibility::default(),
            InheritedVisibility::default(),
            ViewVisibility::default(),
            visual.render_layers,
            NoFrustumCulling,
            visual.key,
            Name::new("frustum_ellipsoid_intersection"),
        ));
    }

    for light in desired_lights {
        if let Some(entity) = existing_lights_by_key.remove(&light.key) {
            commands
                .entity(entity)
                .insert((light.render_layers, light.light, light.transform));
            continue;
        }
        commands.spawn((
            light.light,
            light.transform,
            light.render_layers,
            FrustumEllipsoidIntersectionLight { key: light.key },
            Name::new("frustum_ellipsoid_intersection_light"),
        ));
    }

    for overlay in desired_tint_overlays {
        if let Some(entity) = existing_tint_overlays_by_ellipsoid.remove(&overlay.ellipsoid) {
            commands.entity(entity).insert((
                Mesh3d(overlay.mesh),
                overlay.material,
                overlay.render_layers,
            ));
            commands.entity(overlay.mesh_child).add_child(entity);
            continue;
        }
        let overlay_entity = commands
            .spawn((
                Mesh3d(overlay.mesh),
                overlay.material,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                Visibility::default(),
                InheritedVisibility::default(),
                ViewVisibility::default(),
                overlay.render_layers,
                NoFrustumCulling,
                bevy::light::NotShadowCaster,
                bevy::light::NotShadowReceiver,
                FrustumEllipsoidTintOverlay {
                    ellipsoid: overlay.ellipsoid,
                },
                Name::new("frustum_ellipsoid_tint_overlay"),
            ))
            .id();
        commands
            .entity(overlay.mesh_child)
            .add_child(overlay_entity);
    }

    for (entity, _) in existing_by_key.into_values() {
        commands.entity(entity).despawn();
    }
    for entity in existing_lights_by_key.into_values() {
        commands.entity(entity).despawn();
    }
    for entity in existing_tint_overlays_by_ellipsoid.into_values() {
        commands.entity(entity).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perspective_with_aspect(aspect_ratio: f32) -> PerspectiveProjection {
        PerspectiveProjection {
            fov: 1.0,
            aspect_ratio,
            near: 0.1,
            far: 10.0,
        }
    }

    #[test]
    fn active_source_uses_current_projection() {
        let perspective = perspective_with_aspect(16.0 / 9.0);

        let source = source_frustum_perspective(true, &perspective, None, None, None);

        assert_eq!(source.aspect_ratio, 16.0 / 9.0);
    }

    #[test]
    fn hidden_source_reuses_component_cache_instead_of_stale_viewport_aspect() {
        let cache = IntersectionProjectionCache(perspective_with_aspect(16.0 / 9.0));
        let inactive_perspective = perspective_with_aspect(1.0);

        let source = source_frustum_perspective(
            false,
            &inactive_perspective,
            Some(4.0 / 3.0),
            Some(1.0),
            Some(&cache),
        );

        assert_eq!(source.aspect_ratio, 16.0 / 9.0);
    }

    #[test]
    fn inactive_source_without_cache_uses_config_aspect() {
        let inactive_perspective = perspective_with_aspect(1.0);

        let source = source_frustum_perspective(
            false,
            &inactive_perspective,
            Some(4.0 / 3.0),
            Some(16.0 / 9.0),
            None,
        );

        assert_eq!(source.aspect_ratio, 4.0 / 3.0);
    }

    #[test]
    fn inactive_source_without_cache_uses_viewport_aspect() {
        let inactive_perspective = perspective_with_aspect(1.0);

        let source =
            source_frustum_perspective(false, &inactive_perspective, None, Some(16.0 / 9.0), None);

        assert_eq!(source.aspect_ratio, 16.0 / 9.0);
    }
}
