//! Frustum∩ellipsoid volume computation via SDF grid sampling.
//!
//! Uses a fixed-resolution grid to sample the intersection of frustum and ellipsoid
//! signed-distance fields. Points with both SDFs ≤ 0 are counted as inside the intersection.

use bevy::prelude::*;

use super::EllipsoidVolume;

/// Epsilon for signed-distance and plane tests. Used to avoid division by zero and
/// to classify points as inside/outside surfaces.
pub(super) const SURFACE_EPS: f32 = 1.0e-5;

/// Marching grid resolution for volume sampling. Higher = better quality, higher CPU cost per frame.
const INTERSECTION_GRID: UVec3 = UVec3::new(32, 32, 32);

/// Half-space plane: points satisfying `normal·p + d ≤ 0` are inside the frustum.
#[derive(Clone, Copy)]
pub(super) struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

#[derive(Clone, Copy)]
pub(super) struct FrustumVolume {
    pub source: Entity,
    pub camera_pos: Vec3,
    pub far_corners: [Vec3; 4],
    pub planes: [Plane; 6],
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}

/// Build a plane from three points (a, b, c). The normal points toward the half-space
/// containing `inside`, so that points on the same side as `inside` have negative distance.
pub(super) fn plane_from_points(a: Vec3, b: Vec3, c: Vec3, inside: Vec3) -> Option<Plane> {
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

/// Build the six frustum planes from the eight corner points. Order: near, far, left, right, top, bottom.
pub(super) fn frustum_planes(points: &[Vec3; 8]) -> Option<[Plane; 6]> {
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

/// Axis-aligned bounding box of the eight frustum corner points.
pub(super) fn points_aabb(points: &[Vec3; 8]) -> (Vec3, Vec3) {
    let mut min = points[0];
    let mut max = points[0];
    for point in &points[1..] {
        min = min.min(*point);
        max = max.max(*point);
    }
    (min, max)
}

/// True if the two axis-aligned boxes overlap.
pub(super) fn aabb_overlap(min_a: Vec3, max_a: Vec3, min_b: Vec3, max_b: Vec3) -> bool {
    min_a.x <= max_b.x
        && max_a.x >= min_b.x
        && min_a.y <= max_b.y
        && max_a.y >= min_b.y
        && min_a.z <= max_b.z
        && max_a.z >= min_b.z
}

/// Convert plane to Vec4 (xyz = normal, w = d) for GPU uniform upload.
pub(super) fn plane_to_vec4(plane: &Plane) -> Vec4 {
    Vec4::new(plane.normal.x, plane.normal.y, plane.normal.z, plane.d)
}

/// Half-extent of the ellipsoid AABB in world space (axis-aligned bounding box half-sizes).
pub(super) fn ellipsoid_world_extent(rotation: Quat, radii: Vec3) -> Vec3 {
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

/// Signed distance to frustum: max over all planes. Negative = inside, positive = outside.
pub(super) fn frustum_signed_distance(p: Vec3, planes: &[Plane; 6]) -> f32 {
    let mut max_distance = f32::NEG_INFINITY;
    for plane in planes {
        max_distance = max_distance.max(plane.normal.dot(p) + plane.d);
    }
    max_distance
}

/// Approximate signed distance to ellipsoid surface (IQ's formulation).
/// Not an exact SDF but sufficient for inside/outside classification. Negative = inside.
pub(super) fn ellipsoid_signed_distance(p: Vec3, ellipsoid: &EllipsoidVolume) -> f32 {
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
/// Samples the intersection AABB with a regular grid; cells inside both SDFs contribute to the count.
pub(super) fn compute_intersection_volume(
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

    let mut has_inside = false;
    let mut inside_intersection_count: u32 = 0;
    for k in 0..nz {
        let tz = (k as f32 + 0.5) / nz as f32;
        let z = bounds_min.z + size.z * tz;
        for j in 0..ny {
            let ty = (j as f32 + 0.5) / ny as f32;
            let y = bounds_min.y + size.y * ty;
            for i in 0..nx {
                let tx = (i as f32 + 0.5) / nx as f32;
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
