use super::frustum_common::{MainViewportQueryItem, color_component_to_u8, frustum_local_points};
use crate::{
    MainCamera,
    object_3d::{Object3DMeshChild, Object3DState},
};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::transform::TransformSystems;
use bevy_mat3_material::Mat3Material;
use impeller2::types::ComponentId;
use impeller2_bevy::{ComponentMetadataRegistry, ComponentValue, EntityMap};
use impeller2_wkt::ComponentMetadata;
use std::collections::HashMap;

/// Marching grid resolution for volume sampling. Higher = better quality, higher CPU cost per frame.
const INTERSECTION_GRID: UVec3 = UVec3::new(32, 32, 32);
/// Epsilon for signed-distance and plane tests.
const SURFACE_EPS: f32 = 1.0e-5;

#[derive(Clone, Copy)]
struct Plane {
    normal: Vec3,
    d: f32,
}

#[derive(Clone, Copy)]
struct FrustumVolume {
    source: Entity,
    camera_pos: Vec3,
    far_corners: [Vec3; 4],
    planes: [Plane; 6],
    aabb_min: Vec3,
    aabb_max: Vec3,
}

#[derive(Clone)]
struct EllipsoidVolume {
    entity: Entity,
    center: Vec3,
    rotation: Quat,
    radii: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
    base_color: impeller2_wkt::Color,
    material_target: Option<EllipsoidMaterialTarget>,
}

#[derive(Clone)]
enum EllipsoidMaterialTarget {
    Standard(Handle<StandardMaterial>),
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

#[derive(Component, Clone, Copy, Debug)]
pub struct IntersectionRatio {
    pub source: Entity,
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

#[derive(SystemParam)]
struct FrustumIntersectionParams<'w, 's> {
    main_viewports: Query<'w, 's, MainViewportQueryItem, With<MainCamera>>,
    ellipsoids: Query<'w, 's, (Entity, &'static GlobalTransform, &'static Object3DState)>,
    children: Query<'w, 's, &'static Children>,
    mesh_children: Query<'w, 's, (), With<Object3DMeshChild>>,
    transforms: Query<'w, 's, &'static Transform>,
    mesh_materials: Query<'w, 's, &'static MeshMaterial3d<StandardMaterial>>,
    mesh_mat3_materials: Query<'w, 's, &'static MeshMaterial3d<Mat3Material>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    mat3_materials: ResMut<'w, Assets<Mat3Material>>,
    material_cache: ResMut<'w, IntersectionMaterialCache>,
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
}

pub struct FrustumIntersectionPlugin;

impl Plugin for FrustumIntersectionPlugin {
    fn build(&self, app: &mut App) {
        use impeller2_bevy::AppExt;
        app.init_resource::<IntersectionMaterialCache>()
            .init_resource::<IntersectionRatios>()
            .add_impeller_component::<impeller2_wkt::FrustumCoverage>()
            .add_systems(
                PostUpdate,
                (
                    draw_frustum_ellipsoid_intersections.after(TransformSystems::Propagate),
                    write_coverage_to_db.after(draw_frustum_ellipsoid_intersections),
                ),
            );
    }
}

const PROJECTION_ALPHA: u8 = 230;
const PROJECTION_EMISSIVE_STRENGTH: f32 = 1.4;
const PROJECTION_DEPTH_BIAS: f32 = -8.0;
const INTERSECTION_LIGHT_INTENSITY: f32 = 1500.0;
/// Ellipsoid tint: min blend toward warm when partially inside frustum.
const ELLIPSOID_TINT_MIN_BLEND: f32 = 0.20;
/// Ellipsoid tint: max blend toward green when fully inside frustum.
const ELLIPSOID_TINT_MAX_BLEND: f32 = 0.65;
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

fn plane_from_points(a: Vec3, b: Vec3, c: Vec3, inside: Vec3) -> Option<Plane> {
    let normal = (b - a).cross(c - a);
    let len = normal.length();
    if len <= SURFACE_EPS {
        return None;
    }
    let mut n = normal / len;
    let mut d = -n.dot(a);
    if n.dot(inside) + d > 0.0 {
        n = -n;
        d = -d;
    }
    Some(Plane { normal: n, d })
}

fn frustum_planes(points: &[Vec3; 8]) -> Option<[Plane; 6]> {
    let mut center = Vec3::ZERO;
    for point in points {
        center += *point;
    }
    center /= 8.0;

    Some([
        plane_from_points(points[0], points[1], points[2], center)?,
        plane_from_points(points[4], points[7], points[6], center)?,
        plane_from_points(points[0], points[3], points[7], center)?,
        plane_from_points(points[1], points[6], points[5], center)?,
        plane_from_points(points[0], points[4], points[5], center)?,
        plane_from_points(points[3], points[2], points[6], center)?,
    ])
}

fn points_aabb(points: &[Vec3; 8]) -> (Vec3, Vec3) {
    let mut min = points[0];
    let mut max = points[0];
    for point in &points[1..] {
        min = min.min(*point);
        max = max.max(*point);
    }
    (min, max)
}

fn aabb_overlap(min_a: Vec3, max_a: Vec3, min_b: Vec3, max_b: Vec3) -> bool {
    min_a.x <= max_b.x
        && max_a.x >= min_b.x
        && min_a.y <= max_b.y
        && max_a.y >= min_b.y
        && min_a.z <= max_b.z
        && max_a.z >= min_b.z
}

fn ellipsoid_world_extent(rotation: Quat, radii: Vec3) -> Vec3 {
    let rot = Mat3::from_quat(rotation);
    let ex = rot.x_axis * radii.x;
    let ey = rot.y_axis * radii.y;
    let ez = rot.z_axis * radii.z;
    Vec3::new(
        ex.x.abs() + ey.x.abs() + ez.x.abs(),
        ex.y.abs() + ey.y.abs() + ez.y.abs(),
        ex.z.abs() + ey.z.abs() + ez.z.abs(),
    )
}

fn frustum_signed_distance(p: Vec3, planes: &[Plane; 6]) -> f32 {
    let mut max_distance = f32::NEG_INFINITY;
    for plane in planes {
        max_distance = max_distance.max(plane.normal.dot(p) + plane.d);
    }
    max_distance
}

// Signed distance approximation from IQ's ellipsoid formulation.
fn ellipsoid_signed_distance(p: Vec3, ellipsoid: &EllipsoidVolume) -> f32 {
    let local = ellipsoid.rotation.inverse() * (p - ellipsoid.center);
    let r = ellipsoid.radii.max(Vec3::splat(SURFACE_EPS));
    let k0 = (local / r).length();
    let rr = r * r;
    let k1 = (local / rr).length();
    if k1 <= SURFACE_EPS {
        return k0 - 1.0;
    }
    k0 * (k0 - 1.0) / k1
}

/// Compute frustum∩ellipsoid volume ratio (intersection_volume / ellipsoid_volume) without building mesh.
fn compute_intersection_volume(
    bounds_min: Vec3,
    bounds_max: Vec3,
    frustum: &FrustumVolume,
    ellipsoid: &EllipsoidVolume,
) -> Option<f32> {
    let size = bounds_max - bounds_min;
    if size.x <= SURFACE_EPS || size.y <= SURFACE_EPS || size.z <= SURFACE_EPS {
        return None;
    }

    let nx = INTERSECTION_GRID.x as usize;
    let ny = INTERSECTION_GRID.y as usize;
    let nz = INTERSECTION_GRID.z as usize;
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }

    let points_x = nx + 1;
    let points_y = ny + 1;
    let points_z = nz + 1;
    let mut has_inside = false;
    let mut inside_intersection_count: u32 = 0;
    for k in 0..points_z {
        let tz = k as f32 / nz as f32;
        let z = bounds_min.z + size.z * tz;
        for j in 0..points_y {
            let ty = j as f32 / ny as f32;
            let y = bounds_min.y + size.y * ty;
            for i in 0..points_x {
                let tx = i as f32 / nx as f32;
                let x = bounds_min.x + size.x * tx;
                let p = Vec3::new(x, y, z);
                let d_ellipsoid = ellipsoid_signed_distance(p, ellipsoid);
                let d_frustum = frustum_signed_distance(p, &frustum.planes);
                let s = d_ellipsoid.max(d_frustum);
                if s <= 0.0 {
                    has_inside = true;
                    inside_intersection_count += 1;
                }
            }
        }
    }
    if !has_inside {
        return None;
    }

    let cell_volume = (size.x / nx as f32) * (size.y / ny as f32) * (size.z / nz as f32);
    let intersection_volume = inside_intersection_count as f32 * cell_volume;
    let r = ellipsoid.radii;
    let ellipsoid_volume = (4.0 / 3.0) * std::f32::consts::PI * r.x * r.y * r.z;
    let ratio = if ellipsoid_volume > SURFACE_EPS {
        (intersection_volume / ellipsoid_volume).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some(ratio)
}

/// Grid resolution for 2D projection mesh on far plane. Higher = finer projection boundary.
const PROJECTION_GRID: usize = 80;

fn ray_intersects_ellipsoid_in_frustum(
    origin: Vec3,
    dir: Vec3,
    far_point: Vec3,
    ellipsoid: &EllipsoidVolume,
) -> f32 {
    let inv_rot = ellipsoid.rotation.inverse();
    let local_o = inv_rot * (origin - ellipsoid.center);
    let local_d = inv_rot * dir;
    let r = ellipsoid.radii.max(Vec3::splat(SURFACE_EPS));
    let o_s = local_o / r;
    let d_s = local_d / r;
    let a = d_s.dot(d_s);
    let b = 2.0 * o_s.dot(d_s);
    let c = o_s.dot(o_s) - 1.0;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return -discriminant;
    }
    let sqrt_disc = discriminant.sqrt();
    let t_enter = (-b - sqrt_disc) / (2.0 * a);
    let t_exit = (-b + sqrt_disc) / (2.0 * a);
    let t_far = (far_point - origin).length();
    let overlap_enter = t_enter.max(0.0);
    let overlap_exit = t_exit.min(t_far);
    if overlap_enter < overlap_exit {
        -discriminant
    } else {
        discriminant
    }
}

fn build_projection_mesh(frustum: &FrustumVolume, ellipsoid: &EllipsoidVolume) -> Option<Mesh> {
    let [c0, c1, c2, c3] = frustum.far_corners;
    let cam = frustum.camera_pos;
    let n = PROJECTION_GRID;
    let np = n + 1;

    let bilinear = |u: f32, v: f32| -> Vec3 {
        let e03 = c0.lerp(c3, v);
        let e12 = c1.lerp(c2, v);
        e03.lerp(e12, u)
    };

    let mut scalar = vec![0.0_f32; np * np];
    let idx = |i: usize, j: usize| j * np + i;
    let mut has_inside = false;
    let mut has_outside = false;

    for j in 0..np {
        let v = j as f32 / n as f32;
        for i in 0..np {
            let u = i as f32 / n as f32;
            let p = bilinear(u, v);
            let dir = (p - cam).normalize_or_zero();
            let s = ray_intersects_ellipsoid_in_frustum(cam, dir, p, ellipsoid);
            scalar[idx(i, j)] = s;
            if s <= 0.0 {
                has_inside = true;
            } else {
                has_outside = true;
            }
        }
    }

    if !has_inside {
        return None;
    }

    let far_normal = (c1 - c0).cross(c3 - c0).normalize_or_zero();
    let nn = [far_normal.x, far_normal.y, far_normal.z];

    let mut positions = Vec::new();
    let mut normals = Vec::new();

    let interp_uv = |ua: f32, va: f32, sa: f32, ub: f32, vb: f32, sb: f32| -> (f32, f32) {
        let denom = sa - sb;
        if denom.abs() <= SURFACE_EPS {
            return (ua, va);
        }
        let t = (sa / denom).clamp(0.0, 1.0);
        (ua + t * (ub - ua), va + t * (vb - va))
    };

    for j in 0..n {
        for i in 0..n {
            let u0 = i as f32 / n as f32;
            let u1 = (i + 1) as f32 / n as f32;
            let v0 = j as f32 / n as f32;
            let v1 = (j + 1) as f32 / n as f32;

            let s00 = scalar[idx(i, j)];
            let s10 = scalar[idx(i + 1, j)];
            let s11 = scalar[idx(i + 1, j + 1)];
            let s01 = scalar[idx(i, j + 1)];

            let inside = [s00 <= 0.0, s10 <= 0.0, s11 <= 0.0, s01 <= 0.0];
            let count = inside.iter().filter(|&&x| x).count();
            if count == 0 {
                continue;
            }

            let uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
            let vals = [s00, s10, s11, s01];

            if !has_outside || count == 4 {
                let p00 = bilinear(u0, v0);
                let p10 = bilinear(u1, v0);
                let p11 = bilinear(u1, v1);
                let p01 = bilinear(u0, v1);
                positions.extend_from_slice(&[
                    [p00.x, p00.y, p00.z],
                    [p10.x, p10.y, p10.z],
                    [p11.x, p11.y, p11.z],
                    [p00.x, p00.y, p00.z],
                    [p11.x, p11.y, p11.z],
                    [p01.x, p01.y, p01.z],
                ]);
                normals.extend_from_slice(&[nn, nn, nn, nn, nn, nn]);
                continue;
            }

            let mut edge_verts = Vec::new();
            for e in 0..4 {
                let e_next = (e + 1) % 4;
                if inside[e] {
                    edge_verts.push(bilinear(uvs[e].0, uvs[e].1));
                }
                if inside[e] != inside[e_next] {
                    let (eu, ev) = interp_uv(
                        uvs[e].0,
                        uvs[e].1,
                        vals[e],
                        uvs[e_next].0,
                        uvs[e_next].1,
                        vals[e_next],
                    );
                    edge_verts.push(bilinear(eu, ev));
                }
            }

            if edge_verts.len() >= 3 {
                let center: Vec3 =
                    edge_verts.iter().copied().sum::<Vec3>() / edge_verts.len() as f32;
                for k in 0..edge_verts.len() {
                    let a = edge_verts[k];
                    let b = edge_verts[(k + 1) % edge_verts.len()];
                    positions.extend_from_slice(&[
                        [center.x, center.y, center.z],
                        [a.x, a.y, a.z],
                        [b.x, b.y, b.z],
                    ]);
                    normals.extend_from_slice(&[nn, nn, nn]);
                }
            }
        }
    }

    if positions.is_empty() {
        return None;
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    Some(mesh)
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
    ratios.0.clear();
}

fn reset_ellipsoid_tints(params: &mut FrustumIntersectionParams<'_, '_>) {
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

fn draw_frustum_ellipsoid_intersections(
    mut ratios: ResMut<IntersectionRatios>,
    mut params: FrustumIntersectionParams<'_, '_>,
    mut commands: Commands,
) {
    let mut sources = Vec::new();
    let mut targets = Vec::new();

    for (camera_entity, camera, projection, global_transform, config) in
        params.main_viewports.iter()
    {
        if !camera.is_active {
            continue;
        }

        let Some(config) = config else {
            continue;
        };
        let Some(viewport_layer) = config.viewport_layer else {
            continue;
        };

        if config.show_frustums && (config.show_coverage_in_viewport || config.show_projection_2d) {
            targets.push((
                camera_entity,
                RenderLayers::layer(viewport_layer),
                config.projection_color,
                config.show_coverage_in_viewport,
                config.show_projection_2d,
            ));
        }

        if !config.create_frustum {
            continue;
        }

        let Projection::Perspective(perspective) = projection else {
            continue;
        };
        let Some(local_points) = frustum_local_points(perspective) else {
            continue;
        };
        let world_points = local_points.map(|point| global_transform.transform_point(point));
        let Some(planes) = frustum_planes(&world_points) else {
            continue;
        };
        let (aabb_min, aabb_max) = points_aabb(&world_points);
        sources.push(FrustumVolume {
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
        });
    }

    if sources.is_empty() || targets.is_empty() {
        reset_ellipsoid_tints(&mut params);
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
        let Ok(child_transform) = params.transforms.get(mesh_child_entity) else {
            continue;
        };

        let material_target = if let Ok(mat) = params.mesh_materials.get(mesh_child_entity) {
            Some(EllipsoidMaterialTarget::Standard(mat.0.clone()))
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
    let mut ellipsoid_max_ratio: HashMap<Entity, f32> = HashMap::new();

    for frustum in &sources {
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
            let maybe_ratio =
                compute_intersection_volume(bounds_min, bounds_max, frustum, ellipsoid);

            let any_target_wants_coverage = targets.iter().any(|(_, _, _, sc, _)| *sc);
            if let Some(ratio) = maybe_ratio.filter(|_| any_target_wants_coverage) {
                let entry = ellipsoid_max_ratio.entry(ellipsoid.entity).or_insert(0.0);
                *entry = entry.max(ratio);
                ratios.0.push(IntersectionRatio {
                    source: frustum.source,
                    ellipsoid: ellipsoid.entity,
                    ratio,
                });
            }

            let Some(mesh) = build_projection_mesh(frustum, ellipsoid) else {
                continue;
            };
            for (
                target_camera,
                render_layers,
                target_projection_color,
                _show_coverage,
                target_show_projection,
            ) in &targets
            {
                if frustum.source == *target_camera || !target_show_projection {
                    continue;
                }
                let (light, light_transform) =
                    intersection_light_for(frustum, ellipsoid, *target_projection_color);
                let material = MeshMaterial3d(projection_material_for_color(
                    *target_projection_color,
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

    for ellipsoid in &ellipsoids {
        let ratio = ellipsoid_max_ratio
            .get(&ellipsoid.entity)
            .copied()
            .unwrap_or(0.0);
        if let Some(material_target) = &ellipsoid.material_target {
            apply_ellipsoid_tint(
                material_target,
                ellipsoid.base_color,
                ratio,
                &mut params.materials,
                &mut params.mat3_materials,
            );
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

    for (entity, _) in existing_by_key.into_values() {
        commands.entity(entity).despawn();
    }
    for entity in existing_lights_by_key.into_values() {
        commands.entity(entity).despawn();
    }
}

#[derive(SystemParam)]
struct CoverageDbParams<'w, 's> {
    ratios: Res<'w, IntersectionRatios>,
    entity_map: ResMut<'w, EntityMap>,
    metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    schema_reg: ResMut<'w, impeller2_bevy::ComponentSchemaRegistry>,
    path_reg: ResMut<'w, impeller2_bevy::ComponentPathRegistry>,
    values: Query<'w, 's, &'static mut ComponentValue>,
    names: Query<'w, 's, &'static Name>,
}

fn write_coverage_to_db(mut params: CoverageDbParams<'_, '_>, mut commands: Commands) {
    for ratio in params.ratios.0.iter() {
        let ellipsoid_name = params
            .names
            .get(ratio.ellipsoid)
            .map(|n| n.as_str())
            .unwrap_or("ellipsoid");
        let full_name = format!("{ellipsoid_name}.frustum_coverage");
        let cid = ComponentId::new(&full_name);

        let entity = if let Some(&e) = params.entity_map.get(&cid) {
            e
        } else {
            let metadata = params
                .metadata_reg
                .entry(cid)
                .or_insert_with(|| ComponentMetadata {
                    component_id: cid,
                    name: full_name.clone(),
                    metadata: Default::default(),
                })
                .clone();

            params.schema_reg.0.entry(cid).or_insert_with(|| {
                use impeller2::component::Component;
                impeller2_wkt::FrustumCoverage::schema()
            });

            params
                .path_reg
                .0
                .entry(cid)
                .or_insert_with(|| impeller2_bevy::ComponentPath::from_name(&full_name));

            let e = commands
                .spawn((cid, impeller2_bevy::ComponentValueMap::default(), metadata))
                .id();
            params.entity_map.insert(cid, e);
            e
        };

        if let Ok(mut value) = params.values.get_mut(entity) {
            if let ComponentValue::F32(arr) = &mut *value {
                let buf = nox::ArrayBuf::as_mut_buf(&mut arr.buf);
                if !buf.is_empty() {
                    buf[0] = ratio.ratio;
                }
            }
        } else {
            let mut arr = nox::Array::<f32, nox::Dyn>::zeroed(&[1]);
            nox::ArrayBuf::as_mut_buf(&mut arr.buf)[0] = ratio.ratio;
            commands.entity(entity).insert(ComponentValue::F32(arr));
        }
    }
}
