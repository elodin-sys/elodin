//! 2D projection of frustum∩ellipsoid on the far plane.
//!
//! Casts rays from the camera through a grid on the far plane; for each cell we determine
//! whether the ray hits the ellipsoid before the far plane. A marching-squares-style
//! contour extraction builds the projection mesh from the scalar field.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

use super::EllipsoidVolume;
use super::volume::{FrustumVolume, SURFACE_EPS};

/// Grid resolution for 2D projection mesh on far plane. Higher = finer projection boundary.
const PROJECTION_GRID: usize = 80;

/// Ray-ellipsoid intersection test for projection mesh generation.
/// Returns a scalar: **negative** if the ray hits the ellipsoid between origin and far_point
/// (i.e. "inside" the projected silhouette), **positive** otherwise. Used for marching-squares
/// contour extraction on the far-plane grid.
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

/// Build a triangle mesh of the frustum∩ellipsoid silhouette projected onto the far plane.
/// Samples a grid via bilinear interpolation of far-plane corners; extracts contours
/// where the ray-ellipsoid hit scalar crosses zero.
pub(super) fn build_projection_mesh(
    frustum: &FrustumVolume,
    ellipsoid: &EllipsoidVolume,
) -> Option<Mesh> {
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
