use crate::{
    MainCamera,
    object_3d::{EllipsoidVisual, Object3DState, WorldPosReceived},
    ui::{
        ViewportRect,
        tiles::{EllipsoidIntersectMode, ViewportConfig},
    },
};
use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use bevy::transform::TransformSystems;
use bevy_egui::{EguiContexts, egui};
use std::collections::HashMap;

/// Keep parity with the frustum line overlay behavior: skip source/target pairs
/// when cameras overlap at startup.
const MIN_FRUSTUM_CAMERA_DISTANCE_SQ: f32 = 0.01;
/// POC marching grid resolution (cells). Balance quality vs per-frame CPU cost.
const INTERSECTION_GRID: UVec3 = UVec3::new(32, 32, 32);
const SURFACE_EPS: f32 = 1.0e-5;

type MainViewportQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static ViewportConfig>,
);

#[derive(Clone, Copy)]
struct Plane {
    normal: Vec3,
    d: f32,
}

#[derive(Clone, Copy)]
struct FrustumVolume {
    source: Entity,
    mode: EllipsoidIntersectMode,
    camera_pos: Vec3,
    far_corners: [Vec3; 4],
    planes: [Plane; 6],
    aabb_min: Vec3,
    aabb_max: Vec3,
    color: impeller2_wkt::Color,
    projection_color: impeller2_wkt::Color,
}

#[derive(Clone, Copy)]
struct EllipsoidVolume {
    entity: Entity,
    center: Vec3,
    rotation: Quat,
    radii: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
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
    ratio: f32,
    render_layers: RenderLayers,
    material: MeshMaterial3d<StandardMaterial>,
}

#[derive(SystemParam)]
struct FrustumIntersectionParams<'w, 's> {
    main_viewports: Query<'w, 's, MainViewportQueryItem, With<MainCamera>>,
    ellipsoids: Query<
        'w,
        's,
        (
            Entity,
            &'static GlobalTransform,
            &'static EllipsoidVisual,
            &'static Object3DState,
        ),
        With<WorldPosReceived>,
    >,
    transforms: Query<'w, 's, &'static Transform>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
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
}

pub struct FrustumIntersectionPlugin;

impl Plugin for FrustumIntersectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<IntersectionMaterialCache>()
            .init_resource::<IntersectionRatios>()
            .add_systems(
                PostUpdate,
                (
                    draw_frustum_ellipsoid_intersections.after(TransformSystems::Propagate),
                    draw_intersection_ratio_overlay.after(draw_frustum_ellipsoid_intersections),
                ),
            );
    }
}

fn color_component_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

const INTERSECTION_ALPHA: u8 = 160;

fn intersection_material_for_color(
    color: impeller2_wkt::Color,
    materials: &mut Assets<StandardMaterial>,
    cache: &mut IntersectionMaterialCache,
) -> Handle<StandardMaterial> {
    let key = IntersectionMaterialKey {
        r: color_component_to_u8(color.r),
        g: color_component_to_u8(color.g),
        b: color_component_to_u8(color.b),
        a: INTERSECTION_ALPHA,
    };
    if let Some(handle) = cache.materials.get(&key) {
        return handle.clone();
    }

    let emissive_strength = 0.3;
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba_u8(key.r, key.g, key.b, INTERSECTION_ALPHA),
        emissive: Color::srgba(
            color.r * emissive_strength,
            color.g * emissive_strength,
            color.b * emissive_strength,
            1.0,
        )
        .into(),
        cull_mode: None,
        unlit: false,
        double_sided: true,
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.4,
        depth_bias: -2.0,
        ..Default::default()
    });
    cache.materials.insert(key, material.clone());
    material
}

const PROJECTION_ALPHA: u8 = 180;

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

    let emissive_strength = 0.5;
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba_u8(key.r, key.g, key.b, PROJECTION_ALPHA),
        emissive: Color::srgba(
            color.r * emissive_strength,
            color.g * emissive_strength,
            color.b * emissive_strength,
            1.0,
        )
        .into(),
        cull_mode: None,
        unlit: false,
        double_sided: true,
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.3,
        depth_bias: -3.0,
        ..Default::default()
    });
    cache.materials.insert(key, material.clone());
    material
}

fn frustum_local_points(perspective: &PerspectiveProjection) -> Option<[Vec3; 8]> {
    let near = perspective.near;
    let far = perspective.far;
    let fov = perspective.fov;
    let aspect = perspective.aspect_ratio;
    if !(near > 0.0 && far > near && fov > 0.0 && aspect > 0.0) {
        return None;
    }

    let tan_half = (fov * 0.5).tan();
    let near_half_height = tan_half * near;
    let near_half_width = near_half_height * aspect;
    let far_half_height = tan_half * far;
    let far_half_width = far_half_height * aspect;

    Some([
        Vec3::new(-near_half_width, near_half_height, -near),
        Vec3::new(near_half_width, near_half_height, -near),
        Vec3::new(near_half_width, -near_half_height, -near),
        Vec3::new(-near_half_width, -near_half_height, -near),
        Vec3::new(-far_half_width, far_half_height, -far),
        Vec3::new(far_half_width, far_half_height, -far),
        Vec3::new(far_half_width, -far_half_height, -far),
        Vec3::new(-far_half_width, -far_half_height, -far),
    ])
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

fn intersection_sdf(p: Vec3, frustum: &FrustumVolume, ellipsoid: &EllipsoidVolume) -> f32 {
    let d_ellipsoid = ellipsoid_signed_distance(p, ellipsoid);
    let d_frustum = frustum_signed_distance(p, &frustum.planes);
    d_ellipsoid.max(d_frustum)
}

const SDF_GRADIENT_H: f32 = 0.001;

fn sdf_gradient(p: Vec3, frustum: &FrustumVolume, ellipsoid: &EllipsoidVolume) -> Vec3 {
    let dx = intersection_sdf(p + Vec3::X * SDF_GRADIENT_H, frustum, ellipsoid)
        - intersection_sdf(p - Vec3::X * SDF_GRADIENT_H, frustum, ellipsoid);
    let dy = intersection_sdf(p + Vec3::Y * SDF_GRADIENT_H, frustum, ellipsoid)
        - intersection_sdf(p - Vec3::Y * SDF_GRADIENT_H, frustum, ellipsoid);
    let dz = intersection_sdf(p + Vec3::Z * SDF_GRADIENT_H, frustum, ellipsoid)
        - intersection_sdf(p - Vec3::Z * SDF_GRADIENT_H, frustum, ellipsoid);
    Vec3::new(dx, dy, dz).normalize_or_zero()
}

fn push_triangle(
    verts: [(Vec3, Vec3); 3],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
) {
    let face = (verts[1].0 - verts[0].0).cross(verts[2].0 - verts[0].0);
    if face.length_squared() <= SURFACE_EPS * SURFACE_EPS {
        return;
    }
    for (p, n) in &verts {
        positions.push([p.x, p.y, p.z]);
        normals.push([n.x, n.y, n.z]);
    }
}

fn polygonize_tetra(
    p: [Vec3; 4],
    v: [f32; 4],
    frustum: &FrustumVolume,
    ellipsoid: &EllipsoidVolume,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
) {
    let inside = [v[0] <= 0.0, v[1] <= 0.0, v[2] <= 0.0, v[3] <= 0.0];
    let inside_count = inside.iter().filter(|&&x| x).count();
    if inside_count == 0 || inside_count == 4 {
        return;
    }

    let interp = |a: usize, b: usize| -> Vec3 {
        let va = v[a];
        let vb = v[b];
        let denom = va - vb;
        if denom.abs() <= SURFACE_EPS {
            return p[a];
        }
        let t = (va / denom).clamp(0.0, 1.0);
        p[a].lerp(p[b], t)
    };

    let grad = |pt: Vec3| sdf_gradient(pt, frustum, ellipsoid);

    if inside_count == 1 {
        let in_idx = inside.iter().position(|&x| x).unwrap_or(0);
        let mut outs = [0_usize; 3];
        let mut cursor = 0;
        for (idx, is_inside) in inside.iter().enumerate() {
            if !*is_inside {
                outs[cursor] = idx;
                cursor += 1;
            }
        }
        let a = interp(in_idx, outs[0]);
        let b = interp(in_idx, outs[1]);
        let c = interp(in_idx, outs[2]);
        push_triangle(
            [(a, grad(a)), (b, grad(b)), (c, grad(c))],
            positions,
            normals,
        );
        return;
    }

    if inside_count == 3 {
        let out_idx = inside.iter().position(|&x| !x).unwrap_or(0);
        let mut ins = [0_usize; 3];
        let mut cursor = 0;
        for (idx, is_inside) in inside.iter().enumerate() {
            if *is_inside {
                ins[cursor] = idx;
                cursor += 1;
            }
        }
        let a = interp(out_idx, ins[0]);
        let b = interp(out_idx, ins[2]);
        let c = interp(out_idx, ins[1]);
        push_triangle(
            [(a, grad(a)), (b, grad(b)), (c, grad(c))],
            positions,
            normals,
        );
        return;
    }

    let mut ins = [0_usize; 2];
    let mut outs = [0_usize; 2];
    let mut in_cursor = 0;
    let mut out_cursor = 0;
    for (idx, is_inside) in inside.iter().enumerate() {
        if *is_inside {
            ins[in_cursor] = idx;
            in_cursor += 1;
        } else {
            outs[out_cursor] = idx;
            out_cursor += 1;
        }
    }

    let p0 = interp(ins[0], outs[0]);
    let p1 = interp(ins[0], outs[1]);
    let p2 = interp(ins[1], outs[0]);
    let p3 = interp(ins[1], outs[1]);
    let n0 = grad(p0);
    let n1 = grad(p1);
    let n2 = grad(p2);
    let n3 = grad(p3);
    push_triangle([(p0, n0), (p1, n1), (p2, n2)], positions, normals);
    push_triangle([(p2, n2), (p1, n1), (p3, n3)], positions, normals);
}

struct IntersectionResult {
    mesh: Mesh,
    ratio: f32,
}

fn build_intersection_mesh(
    bounds_min: Vec3,
    bounds_max: Vec3,
    frustum: &FrustumVolume,
    ellipsoid: &EllipsoidVolume,
) -> Option<IntersectionResult> {
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
    let mut scalar = vec![0.0_f32; points_x * points_y * points_z];
    let idx = |i: usize, j: usize, k: usize| -> usize { (k * points_y + j) * points_x + i };

    let mut has_inside = false;
    let mut has_outside = false;
    let mut inside_ellipsoid_count: u32 = 0;
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
                if d_ellipsoid <= 0.0 {
                    inside_ellipsoid_count += 1;
                }
                if s <= 0.0 {
                    has_inside = true;
                    inside_intersection_count += 1;
                } else {
                    has_outside = true;
                }
                scalar[idx(i, j, k)] = s;
            }
        }
    }
    if !(has_inside && has_outside) {
        return None;
    }

    let ratio = if inside_ellipsoid_count > 0 {
        inside_intersection_count as f32 / inside_ellipsoid_count as f32
    } else {
        0.0
    };

    let mut positions = Vec::new();
    let mut normals = Vec::new();

    const CORNERS: [(usize, usize, usize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    const TETS: [[usize; 4]; 6] = [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ];

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let mut cube_points = [Vec3::ZERO; 8];
                let mut cube_values = [0.0_f32; 8];
                for (corner_idx, (ox, oy, oz)) in CORNERS.iter().enumerate() {
                    let gx = i + ox;
                    let gy = j + oy;
                    let gz = k + oz;

                    let tx = gx as f32 / nx as f32;
                    let ty = gy as f32 / ny as f32;
                    let tz = gz as f32 / nz as f32;
                    cube_points[corner_idx] = Vec3::new(
                        bounds_min.x + size.x * tx,
                        bounds_min.y + size.y * ty,
                        bounds_min.z + size.z * tz,
                    );
                    cube_values[corner_idx] = scalar[idx(gx, gy, gz)];
                }

                for tetra in TETS {
                    let tetra_points = [
                        cube_points[tetra[0]],
                        cube_points[tetra[1]],
                        cube_points[tetra[2]],
                        cube_points[tetra[3]],
                    ];
                    let tetra_values = [
                        cube_values[tetra[0]],
                        cube_values[tetra[1]],
                        cube_values[tetra[2]],
                        cube_values[tetra[3]],
                    ];
                    polygonize_tetra(
                        tetra_points,
                        tetra_values,
                        frustum,
                        ellipsoid,
                        &mut positions,
                        &mut normals,
                    );
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
    Some(IntersectionResult { mesh, ratio })
}

const PROJECTION_GRID: usize = 80;

fn ray_intersects_ellipsoid_in_frustum(
    origin: Vec3,
    dir: Vec3,
    frustum: &FrustumVolume,
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
        return discriminant;
    }
    let sqrt_disc = discriminant.sqrt();
    let t_enter = ((-b - sqrt_disc) / (2.0 * a)).max(0.0);
    let t_exit = ((-b + sqrt_disc) / (2.0 * a)).max(0.0);
    let samples: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    for &frac in &samples {
        let t = t_enter + frac * (t_exit - t_enter);
        let p = origin + dir * t;
        if frustum_signed_distance(p, &frustum.planes) <= 0.0 {
            return -discriminant;
        }
    }
    discriminant
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
            let s = ray_intersects_ellipsoid_in_frustum(cam, dir, frustum, ellipsoid);
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
    ratios.0.clear();
}

fn draw_frustum_ellipsoid_intersections(
    mut ratios: ResMut<IntersectionRatios>,
    mut params: FrustumIntersectionParams<'_, '_>,
    mut commands: Commands,
) {
    let mut camera_positions: HashMap<Entity, Vec3> = HashMap::new();
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

        camera_positions.insert(camera_entity, global_transform.translation());
        if config.show_frustums || config.ellipsoid_intersect_mode != EllipsoidIntersectMode::Off {
            targets.push((camera_entity, RenderLayers::layer(viewport_layer)));
        }

        if !config.create_frustum || config.ellipsoid_intersect_mode == EllipsoidIntersectMode::Off
        {
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
            mode: config.ellipsoid_intersect_mode,
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
            color: config.intersect_3d_color,
            projection_color: config.projection_color,
        });
    }

    if sources.is_empty() || targets.is_empty() {
        cleanup_all(&params, &mut ratios, &mut commands);
        return;
    }

    let mut ellipsoids = Vec::new();
    for (entity, global_transform, ellipse_visual, object_state) in params.ellipsoids.iter() {
        if !matches!(
            object_state.data.mesh,
            impeller2_wkt::Object3DMesh::Ellipsoid { .. }
        ) {
            continue;
        }

        let Ok(child_transform) = params.transforms.get(ellipse_visual.child) else {
            continue;
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
        });
    }

    if ellipsoids.is_empty() {
        cleanup_all(&params, &mut ratios, &mut commands);
        return;
    }

    let mut desired = Vec::new();

    for frustum in &sources {
        let material_handle = match frustum.mode {
            EllipsoidIntersectMode::Projection2D => projection_material_for_color(
                frustum.projection_color,
                &mut params.materials,
                &mut params.material_cache,
            ),
            _ => intersection_material_for_color(
                frustum.color,
                &mut params.materials,
                &mut params.material_cache,
            ),
        };
        let material = MeshMaterial3d(material_handle);
        for (target_camera, render_layers) in &targets {
            if frustum.source == *target_camera {
                continue;
            }
            if let (Some(&src_pos), Some(&tgt_pos)) = (
                camera_positions.get(&frustum.source),
                camera_positions.get(target_camera),
            ) && (src_pos - tgt_pos).length_squared() < MIN_FRUSTUM_CAMERA_DISTANCE_SQ
            {
                continue;
            }

            for ellipsoid in &ellipsoids {
                if !aabb_overlap(
                    frustum.aabb_min,
                    frustum.aabb_max,
                    ellipsoid.aabb_min,
                    ellipsoid.aabb_max,
                ) {
                    continue;
                }

                let (mesh, ratio) = match frustum.mode {
                    EllipsoidIntersectMode::Mesh3D => {
                        let bounds_min = frustum.aabb_min.max(ellipsoid.aabb_min);
                        let bounds_max = frustum.aabb_max.min(ellipsoid.aabb_max);
                        match build_intersection_mesh(bounds_min, bounds_max, frustum, ellipsoid) {
                            Some(result) => (result.mesh, Some(result.ratio)),
                            None => continue,
                        }
                    }
                    EllipsoidIntersectMode::Projection2D => {
                        match build_projection_mesh(frustum, ellipsoid) {
                            Some(m) => (m, None),
                            None => continue,
                        }
                    }
                    EllipsoidIntersectMode::Off => continue,
                };

                desired.push(DesiredIntersection {
                    key: FrustumEllipsoidIntersectionVisual {
                        source: frustum.source,
                        target: *target_camera,
                        ellipsoid: ellipsoid.entity,
                    },
                    mesh,
                    ratio: ratio.unwrap_or(0.0),
                    render_layers: render_layers.clone(),
                    material: material.clone(),
                });
            }
        }
    }

    ratios.0.clear();
    for visual in &desired {
        if visual.ratio > 0.0 {
            ratios.0.push(IntersectionRatio {
                source: visual.key.source,
                ellipsoid: visual.key.ellipsoid,
                ratio: visual.ratio,
            });
        }
    }

    let mut existing_by_key: HashMap<FrustumEllipsoidIntersectionVisual, (Entity, Handle<Mesh>)> =
        HashMap::new();
    for (entity, key, mesh3d) in params.existing_visuals.iter() {
        existing_by_key.insert(*key, (entity, mesh3d.0.clone()));
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

    for (entity, _) in existing_by_key.into_values() {
        commands.entity(entity).despawn();
    }
}

fn draw_intersection_ratio_overlay(
    ratios: Res<IntersectionRatios>,
    viewports: Query<(Entity, &ViewportConfig, &ViewportRect), With<MainCamera>>,
    mut contexts: EguiContexts,
) {
    if ratios.0.is_empty() {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    for (cam_entity, config, viewport_rect) in viewports.iter() {
        let Some(rect) = viewport_rect.0 else {
            continue;
        };

        let is_source = config.create_frustum
            && config.ellipsoid_intersect_mode == EllipsoidIntersectMode::Mesh3D;
        let is_target =
            config.show_frustums || config.ellipsoid_intersect_mode != EllipsoidIntersectMode::Off;

        let relevant: Vec<&IntersectionRatio> = ratios
            .0
            .iter()
            .filter(|r| {
                (is_source && r.source == cam_entity) || (is_target && r.source != cam_entity)
            })
            .collect();

        if relevant.is_empty() {
            continue;
        }

        let area_id = egui::Id::new("intersection_ratio").with(cam_entity);
        let row_height = 16.0;
        let overlay_height = relevant.len() as f32 * row_height + 8.0;
        egui::Area::new(area_id)
            .fixed_pos(egui::pos2(
                rect.min.x + 8.0,
                rect.max.y - overlay_height - 4.0,
            ))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_black_alpha(140))
                    .corner_radius(egui::CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(6, 3))
                    .show(ui, |ui| {
                        for r in &relevant {
                            let pct = (r.ratio * 100.0).clamp(0.0, 100.0);
                            ui.label(
                                egui::RichText::new(format!("{pct:.1}%"))
                                    .monospace()
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(220, 220, 220)),
                            );
                        }
                    });
            });
    }
}
