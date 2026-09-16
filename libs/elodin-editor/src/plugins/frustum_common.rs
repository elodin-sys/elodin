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

/// Up-triangle dimensions, as fractions of [`up_triangle_reference`].
const UP_TRIANGLE_HALF_BASE: f32 = 0.13;
const UP_TRIANGLE_HEIGHT: f32 = 0.20;

/// Cap on the triangle's reference size, as a fraction of the far distance.
/// A wide-FOV frustum has a far plane that dwarfs the pyramid behind it, so
/// sizing the triangle off that plane alone makes it swallow the shape it is
/// supposed to annotate.
const UP_TRIANGLE_MAX_REFERENCE: f32 = 0.7;

/// A frustum edge in camera-local space.
pub struct FrustumEdge {
    pub start: Vec3,
    pub end: Vec3,
    pub thickness: f32,
    /// Part of the up marker, so drawn in [`frustum_up_marker_color`].
    pub is_up_marker: bool,
}

/// Frustum edges in camera-local space, including the extra edges drawn for
/// `up_marker`.
pub fn frustum_segments(
    points: [Vec3; 8],
    thickness: f32,
    up_marker: FrustumUpMarker,
) -> Vec<FrustumEdge> {
    let mut segments: Vec<FrustumEdge> = [
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
    .map(|(start, end)| FrustumEdge {
        start,
        end,
        thickness,
        is_up_marker: false,
    })
    .collect();

    match up_marker {
        FrustumUpMarker::None => {}
        FrustumUpMarker::Highlight => {
            segments[FAR_TOP_SEGMENT].thickness = thickness * FRUSTUM_UP_MARKER_SCALE;
            segments[FAR_TOP_SEGMENT].is_up_marker = true;
        }
        FrustumUpMarker::Triangle => {
            segments.extend(
                frustum_up_triangle(&points)
                    .into_iter()
                    .map(|(start, end)| FrustumEdge {
                        start,
                        end,
                        thickness: thickness * FRUSTUM_UP_MARKER_SCALE,
                        is_up_marker: true,
                    }),
            );
        }
    }

    segments
}

/// Channel floor above which a frustum color counts as near-white.
const FRUSTUM_NEAR_WHITE: f32 = 0.7;

/// Color for up-marker geometry. White separates the marker from the frustum
/// color, which the user picks freely; when that color is itself near-white a
/// white marker would vanish, so fall back to its complement. Always opaque, so
/// the marker still reads on a translucent or fully clear frustum.
pub fn frustum_up_marker_color(frustum_color: impeller2_wkt::Color) -> impeller2_wkt::Color {
    let darkest_channel = frustum_color
        .r
        .min(frustum_color.g)
        .min(frustum_color.b)
        .clamp(0.0, 1.0);
    if darkest_channel > FRUSTUM_NEAR_WHITE {
        impeller2_wkt::Color::rgba(
            1.0 - frustum_color.r.clamp(0.0, 1.0),
            1.0 - frustum_color.g.clamp(0.0, 1.0),
            1.0 - frustum_color.b.clamp(0.0, 1.0),
            1.0,
        )
    } else {
        impeller2_wkt::Color::rgba(1.0, 1.0, 1.0, 1.0)
    }
}

/// Balls belonging to the up marker, as `(center, radius)` in camera-local
/// space.
///
/// `Highlight` gets one on the far-plane corner holding pixel (0, 0) — top-left
/// as seen through the camera — which, with the thickened top edge, says which
/// way up the image is and which end of that edge it starts from.
///
/// `Triangle` gets one on each of its corners. Edges are flat-capped cylinders,
/// so the two sides meeting at an angle would otherwise leave a notch; a ball
/// of the same radius rounds the joint into a continuous line.
pub fn frustum_up_marker_balls(
    points: &[Vec3; 8],
    thickness: f32,
    up_marker: FrustumUpMarker,
) -> Vec<(Vec3, f32)> {
    match up_marker {
        FrustumUpMarker::None => Vec::new(),
        FrustumUpMarker::Highlight => {
            vec![(points[4], thickness * FRUSTUM_IMAGE_ORIGIN_SCALE)]
        }
        FrustumUpMarker::Triangle => {
            let radius = thickness * FRUSTUM_UP_MARKER_SCALE;
            let [(left, apex), (_, right)] = frustum_up_triangle(points);
            vec![(left, radius), (apex, radius), (right, radius)]
        }
    }
}

/// Size the up triangle is derived from: the far-plane half height, capped
/// against the far distance so a wide field of view cannot inflate it.
fn up_triangle_reference(points: &[Vec3; 8]) -> f32 {
    let far_half_height = (points[4].y - points[7].y) * 0.5;
    let far_distance = -points[4].z;
    far_half_height.min(far_distance * UP_TRIANGLE_MAX_REFERENCE)
}

/// The two sides of the triangle standing on the middle of the far-plane top
/// edge, pointing towards the camera's up direction. The edge itself closes the
/// shape, so drawing a base here would only double that line. Both axes scale
/// off the same reference, so the triangle keeps its shape whatever the frustum
/// aspect.
pub fn frustum_up_triangle(points: &[Vec3; 8]) -> [(Vec3, Vec3); 2] {
    let reference = up_triangle_reference(points);
    let z = points[4].z;
    let base_y = points[4].y;

    let half_base = reference * UP_TRIANGLE_HALF_BASE;
    let apex = Vec3::new(0.0, base_y + reference * UP_TRIANGLE_HEIGHT, z);
    let left = Vec3::new(-half_base, base_y, z);
    let right = Vec3::new(half_base, base_y, z);
    [(left, apex), (apex, right)]
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
        assert!(segments.iter().all(|edge| edge.thickness == 0.01));
        assert!(segments.iter().all(|edge| !edge.is_up_marker));
    }

    #[test]
    fn frustum_segments_highlight_thickens_far_top_edge() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let segments = frustum_segments(points, 0.01, FrustumUpMarker::Highlight);
        assert_eq!(segments.len(), 12);

        let top = &segments[4];
        assert_eq!(top.start, points[4]);
        assert_eq!(top.end, points[5]);
        assert_eq!(top.thickness, 0.01 * FRUSTUM_UP_MARKER_SCALE);
        assert!(top.is_up_marker);
        for (idx, edge) in segments.iter().enumerate() {
            if idx != 4 {
                assert_eq!(edge.thickness, 0.01);
                assert!(!edge.is_up_marker);
            }
        }
    }

    #[test]
    fn up_marker_color_is_white_against_ordinary_frustums() {
        use impeller2_wkt::Color as FrustumColor;

        let white = FrustumColor::rgba(1.0, 1.0, 1.0, 1.0);
        for color in [
            FrustumColor::YELLOW,
            FrustumColor::RED,
            FrustumColor::BLACK,
            FrustumColor::MINT,
            FrustumColor::PEACH,
        ] {
            assert_eq!(frustum_up_marker_color(color), white);
        }
    }

    #[test]
    fn up_marker_color_complements_near_white_frustums() {
        use impeller2_wkt::Color as FrustumColor;

        assert_eq!(
            frustum_up_marker_color(FrustumColor::WHITE),
            FrustumColor::rgba(0.0, 0.0, 0.0, 1.0)
        );

        let complement = frustum_up_marker_color(FrustumColor::rgba(0.9, 0.8, 1.0, 0.4));
        assert!((complement.r - 0.1).abs() < 1e-6);
        assert!((complement.g - 0.2).abs() < 1e-6);
        assert!(complement.b.abs() < 1e-6);
        assert_eq!(complement.a, 1.0, "marker stays opaque on a faint frustum");
    }

    #[test]
    fn up_marker_color_is_opaque_on_a_clear_frustum() {
        use impeller2_wkt::Color as FrustumColor;

        assert_eq!(
            frustum_up_marker_color(FrustumColor::rgba(0.2, 0.3, 0.4, 0.0)),
            FrustumColor::rgba(1.0, 1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn frustum_up_marker_balls_per_mode() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();

        assert!(frustum_up_marker_balls(&points, 0.01, FrustumUpMarker::None).is_empty());

        let highlight = frustum_up_marker_balls(&points, 0.01, FrustumUpMarker::Highlight);
        let [(center, radius)] = highlight[..] else {
            panic!("highlight should place a single ball, got {highlight:?}");
        };
        assert_eq!(center, points[4], "ball sits on the far top-left corner");
        assert!(center.x < 0.0 && center.y > 0.0, "left of and above center");
        assert_eq!(radius, 0.01 * FRUSTUM_IMAGE_ORIGIN_SCALE);

        // One rounded joint per triangle corner, matching the stroke radius so
        // the flat-capped sides join into a continuous line.
        let triangle_balls = frustum_up_marker_balls(&points, 0.01, FrustumUpMarker::Triangle);
        assert_eq!(triangle_balls.len(), 3);
        assert!(
            triangle_balls
                .iter()
                .all(|(_, radius)| *radius == 0.01 * FRUSTUM_UP_MARKER_SCALE)
        );

        let corners: Vec<Vec3> = triangle_balls.iter().map(|(center, _)| *center).collect();
        let triangle = frustum_up_triangle(&points);
        for (start, end) in triangle {
            assert!(corners.contains(&start) && corners.contains(&end));
        }
    }

    #[test]
    fn frustum_up_triangle_rests_on_the_far_top_edge() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let triangle = frustum_up_triangle(&points);

        let far_top_y = points[4].y;
        let far_z = points[4].z;
        for (start, end) in triangle {
            assert!(start.y >= far_top_y && end.y >= far_top_y);
            assert!((start.z - far_z).abs() < 1e-5 && (end.z - far_z).abs() < 1e-5);
        }

        let (left, apex) = triangle[0];
        let right = triangle[1].1;
        assert!(apex.x.abs() < 1e-6, "apex should be horizontally centered");
        assert!(apex.y > far_top_y, "apex should be above the edge");
        assert_eq!(left.y, far_top_y, "the edge itself serves as the base");
        assert_eq!(right.y, far_top_y);
        assert!(
            left.x < 0.0 && right.x > 0.0,
            "base corners straddle the middle of the edge"
        );
        assert!(
            !triangle
                .iter()
                .any(|(start, end)| start.y == far_top_y && end.y == far_top_y),
            "no base segment, which would double the far top edge"
        );
    }

    #[test]
    fn frustum_segments_triangle_appends_outline() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();
        let segments = frustum_segments(points, 0.01, FrustumUpMarker::Triangle);
        assert_eq!(segments.len(), 14);

        let (base, marker) = segments.split_at(12);
        assert!(base.iter().all(|edge| edge.thickness == 0.01));
        assert!(base.iter().all(|edge| !edge.is_up_marker));
        assert!(
            marker
                .iter()
                .all(|edge| edge.thickness == 0.01 * FRUSTUM_UP_MARKER_SCALE),
            "triangle reads at the same weight as the highlighted edge"
        );
        assert!(marker.iter().all(|edge| edge.is_up_marker));
    }

    #[test]
    fn up_triangle_shrinks_relative_to_a_wide_field_of_view() {
        // Height of the triangle as a fraction of the far-plane half height.
        let relative_height = |points: &[Vec3; 8]| {
            let (base, apex) = frustum_up_triangle(points)[0];
            (apex.y - base.y) / ((points[4].y - points[7].y) * 0.5)
        };

        let narrow = frustum_local_points(&perspective(0.6, 1.0, 0.1, 6.0)).unwrap();
        let wide = frustum_local_points(&perspective(2.0, 1.0, 0.1, 6.0)).unwrap();
        assert!(
            relative_height(&wide) < relative_height(&narrow) * 0.6,
            "a far plane that dwarfs the pyramid must not carry a proportional triangle"
        );
        assert!(
            relative_height(&narrow) < 0.25,
            "triangle stays a small annotation on the far plane"
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
