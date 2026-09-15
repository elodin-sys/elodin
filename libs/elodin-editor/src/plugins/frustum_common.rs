use crate::plugins::render_layer_alloc::RenderLayerLease;
use crate::sensor_camera::SensorCameraFrustumSource;
use crate::ui::tiles::{DEFAULT_VIEWPORT_FAR, ViewportConfig};
use bevy::prelude::*;
use impeller2_wkt::FrustumUpMarker;

pub type MainViewportQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static ViewportConfig>,
    Option<&'static RenderLayerLease>,
);

pub type SensorCameraFrustumQueryItem = (
    Entity,
    &'static Projection,
    &'static GlobalTransform,
    &'static SensorCameraFrustumSource,
);

pub fn presentation_far(near: f32, configured_far: Option<f32>) -> f32 {
    match configured_far {
        Some(far) if far > near => far,
        _ => DEFAULT_VIEWPORT_FAR.max(near * 2.0),
    }
}

pub fn presentation_perspective(
    live: &PerspectiveProjection,
    config: Option<&ViewportConfig>,
) -> PerspectiveProjection {
    let near = config
        .and_then(|config| config.configured_near)
        .unwrap_or(live.near);
    let far = presentation_far(near, config.and_then(|config| config.configured_far));
    PerspectiveProjection {
        near,
        far,
        near_clip_plane: near_clip_plane(near),
        ..live.clone()
    }
}

pub fn frustum_local_points(perspective: &PerspectiveProjection) -> Option<[Vec3; 8]> {
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

/// Edge-width multiplier for up-marker geometry, so both markers read at the
/// same weight against the plain frustum edges.
pub const FRUSTUM_UP_MARKER_SCALE: f32 = 3.0;

/// Radius of the image-origin ball, relative to the frustum edge radius.
pub const FRUSTUM_IMAGE_ORIGIN_SCALE: f32 = 9.0;

/// Index of the far-plane top edge within [`frustum_segments`].
const FAR_TOP_SEGMENT: usize = 4;

/// Up-triangle dimensions, as fractions of the far-plane half extents.
const UP_TRIANGLE_HALF_BASE: f32 = 0.22;
const UP_TRIANGLE_GAP: f32 = 0.06;
const UP_TRIANGLE_HEIGHT: f32 = 0.24;

/// Frustum edges in camera-local space as `(start, end, thickness)`, including
/// the extra edges drawn for `up_marker`.
pub fn frustum_segments(
    points: [Vec3; 8],
    thickness: f32,
    up_marker: FrustumUpMarker,
) -> Vec<(Vec3, Vec3, f32)> {
    let mut segments: Vec<(Vec3, Vec3, f32)> = [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[3]),
        (points[3], points[0]),
        (points[4], points[5]),
        (points[5], points[6]),
        (points[6], points[7]),
        (points[7], points[4]),
        (points[0], points[4]),
        (points[1], points[5]),
        (points[2], points[6]),
        (points[3], points[7]),
    ]
    .into_iter()
    .map(|(start, end)| (start, end, thickness))
    .collect();

    match up_marker {
        FrustumUpMarker::None => {}
        FrustumUpMarker::Highlight => {
            segments[FAR_TOP_SEGMENT].2 = thickness * FRUSTUM_UP_MARKER_SCALE;
        }
        FrustumUpMarker::Triangle => {
            segments.extend(
                frustum_up_triangle(&points)
                    .into_iter()
                    .map(|(start, end)| (start, end, thickness * FRUSTUM_UP_MARKER_SCALE)),
            );
        }
    }

    segments
}

/// Ball marking the image origin — the far-plane corner holding pixel (0, 0),
/// i.e. top-left as seen through the camera — as `(center, radius)` in
/// camera-local space. Together with the thickened top edge it tells which way
/// up the image is, and which end of that edge it starts from.
pub fn frustum_image_origin_marker(
    points: &[Vec3; 8],
    thickness: f32,
    up_marker: FrustumUpMarker,
) -> Option<(Vec3, f32)> {
    matches!(up_marker, FrustumUpMarker::Highlight)
        .then(|| (points[4], thickness * FRUSTUM_IMAGE_ORIGIN_SCALE))
}

/// Outline of the triangle standing on the middle of the far-plane top edge,
/// pointing towards the camera's up direction.
pub fn frustum_up_triangle(points: &[Vec3; 8]) -> [(Vec3, Vec3); 3] {
    let far_half_width = (points[5].x - points[4].x) * 0.5;
    let far_half_height = (points[4].y - points[7].y) * 0.5;
    let z = points[4].z;

    let half_base = far_half_width * UP_TRIANGLE_HALF_BASE;
    let base_y = points[4].y + far_half_height * UP_TRIANGLE_GAP;
    let apex_y = base_y + far_half_height * UP_TRIANGLE_HEIGHT;

    let left = Vec3::new(-half_base, base_y, z);
    let right = Vec3::new(half_base, base_y, z);
    let apex = Vec3::new(0.0, apex_y, z);
    [(left, right), (right, apex), (apex, left)]
}

pub fn color_component_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Canonical `PerspectiveProjection::near_clip_plane` value for a given `near`
/// distance. Used to keep the two fields in sync at every construction site.
pub fn near_clip_plane(near: f32) -> Vec4 {
    Vec4::new(0.0, 0.0, -1.0, -near)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perspective(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> PerspectiveProjection {
        PerspectiveProjection {
            fov,
            aspect_ratio,
            near,
            far,
            near_clip_plane: near_clip_plane(near),
        }
    }

    fn viewport_config(near: Option<f32>, far: Option<f32>) -> ViewportConfig {
        ViewportConfig {
            aspect: None,
            configured_near: near,
            configured_far: far,
            show_arrows: true,
            create_frustum: true,
            show_frustums: false,
            show_coverage_in_viewport: false,
            show_projection_2d: false,
            frustums_color: default(),
            projection_color: default(),
            frustums_thickness: 0.006,
            frustums_up_marker: FrustumUpMarker::None,
            cinematic: false,
            bloom: None,
        }
    }

    #[test]
    fn presentation_perspective_uses_configured_distances() {
        let live = perspective(1.2, 16.0 / 9.0, 9.0, 1.0e16);
        let config = viewport_config(Some(0.2), Some(500.0));
        let presentation = presentation_perspective(&live, Some(&config));

        assert_eq!(presentation.fov, live.fov);
        assert_eq!(presentation.aspect_ratio, live.aspect_ratio);
        assert_eq!(presentation.near, 0.2);
        assert_eq!(presentation.far, 500.0);
        assert_eq!(presentation.near_clip_plane, near_clip_plane(0.2));
    }

    #[test]
    fn presentation_perspective_uses_live_near_when_unconfigured() {
        let live = perspective(1.2, 16.0 / 9.0, 9.0, 1.0e16);
        let presentation = presentation_perspective(&live, None);

        assert_eq!(presentation.near, 9.0);
        assert_eq!(presentation.far, 18.0);
        assert_eq!(presentation.near_clip_plane, near_clip_plane(9.0));
        assert!(frustum_local_points(&presentation).is_some());
    }

    #[test]
    fn presentation_perspective_derives_far_beyond_large_near() {
        let live = perspective(1.2, 16.0 / 9.0, 9.0, 1.0e16);
        let config = viewport_config(Some(1_000_000.0), None);
        let presentation = presentation_perspective(&live, Some(&config));

        assert_eq!(presentation.near, 1_000_000.0);
        assert_eq!(presentation.far, 2_000_000.0);
        assert!(frustum_local_points(&presentation).is_some());
    }

    #[test]
    fn frustum_local_points_basic() {
        let persp = perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let pts = frustum_local_points(&persp).unwrap();
        for p in &pts[0..4] {
            assert!((p.z - (-0.1)).abs() < 1e-5, "near plane z should be -near");
        }
        for p in &pts[4..8] {
            assert!((p.z - (-100.0)).abs() < 1e-3, "far plane z should be -far");
        }
    }

    #[test]
    fn frustum_local_points_rejects_degenerate() {
        let bad_near = perspective(1.0, 1.0, 0.0, 10.0);
        assert!(frustum_local_points(&bad_near).is_none());

        let bad_far = perspective(1.0, 1.0, 10.0, 5.0);
        assert!(frustum_local_points(&bad_far).is_none());

        let bad_fov = perspective(0.0, 1.0, 0.1, 10.0);
        assert!(frustum_local_points(&bad_fov).is_none());

        let bad_aspect = perspective(1.0, 0.0, 0.1, 10.0);
        assert!(frustum_local_points(&bad_aspect).is_none());
    }

    #[test]
    fn frustum_local_points_aspect_ratio() {
        let persp = perspective(std::f32::consts::FRAC_PI_2, 2.0, 1.0, 10.0);
        let pts = frustum_local_points(&persp).unwrap();
        let near_width = (pts[1].x - pts[0].x).abs();
        let near_height = (pts[0].y - pts[3].y).abs();
        assert!(
            ((near_width / near_height) - 2.0).abs() < 1e-5,
            "aspect ratio should be reflected in near plane dimensions"
        );
    }

    #[test]
    fn frustum_segments_marker_none_leaves_edges_untouched() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let segments = frustum_segments(points, 0.01, FrustumUpMarker::None);
        assert_eq!(segments.len(), 12);
        assert!(segments.iter().all(|(_, _, thickness)| *thickness == 0.01));
    }

    #[test]
    fn frustum_segments_highlight_thickens_far_top_edge() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let segments = frustum_segments(points, 0.01, FrustumUpMarker::Highlight);
        assert_eq!(segments.len(), 12);

        let (start, end, thickness) = segments[4];
        assert_eq!(start, points[4]);
        assert_eq!(end, points[5]);
        assert_eq!(thickness, 0.01 * FRUSTUM_UP_MARKER_SCALE);
        for (idx, (_, _, thickness)) in segments.iter().enumerate() {
            if idx != 4 {
                assert_eq!(*thickness, 0.01);
            }
        }
    }

    #[test]
    fn frustum_image_origin_marker_only_for_highlight() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();

        let (center, radius) =
            frustum_image_origin_marker(&points, 0.01, FrustumUpMarker::Highlight).unwrap();
        assert_eq!(center, points[4], "ball sits on the far top-left corner");
        assert!(center.x < 0.0 && center.y > 0.0, "left of and above center");
        assert_eq!(radius, 0.01 * FRUSTUM_IMAGE_ORIGIN_SCALE);

        assert!(frustum_image_origin_marker(&points, 0.01, FrustumUpMarker::None).is_none());
        assert!(frustum_image_origin_marker(&points, 0.01, FrustumUpMarker::Triangle).is_none());
    }

    #[test]
    fn frustum_up_triangle_sits_above_far_top_edge() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let triangle = frustum_up_triangle(&points);

        let far_top_y = points[4].y;
        let far_z = points[4].z;
        for (start, end) in triangle {
            assert!(start.y > far_top_y && end.y > far_top_y);
            assert!((start.z - far_z).abs() < 1e-5 && (end.z - far_z).abs() < 1e-5);
        }

        let apex = triangle[1].1;
        assert!(apex.x.abs() < 1e-6, "apex should be horizontally centered");
        assert!(apex.y > triangle[0].0.y, "apex should be above the base");
    }

    #[test]
    fn frustum_segments_triangle_appends_outline() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let segments = frustum_segments(points, 0.01, FrustumUpMarker::Triangle);
        assert_eq!(segments.len(), 15);

        let (base, marker) = segments.split_at(12);
        assert!(base.iter().all(|(_, _, thickness)| *thickness == 0.01));
        assert!(
            marker
                .iter()
                .all(|(_, _, thickness)| *thickness == 0.01 * FRUSTUM_UP_MARKER_SCALE),
            "triangle reads at the same weight as the highlighted edge"
        );
    }

    #[test]
    fn color_component_to_u8_boundaries() {
        assert_eq!(color_component_to_u8(0.0), 0);
        assert_eq!(color_component_to_u8(1.0), 255);
        assert_eq!(color_component_to_u8(0.5), 128);
        assert_eq!(color_component_to_u8(-1.0), 0);
        assert_eq!(color_component_to_u8(2.0), 255);
    }
}
