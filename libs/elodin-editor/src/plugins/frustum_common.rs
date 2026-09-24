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

/// Edge-width multiplier for the highlighted top edge, so it reads apart from
/// the plain frustum edges.
pub const FRUSTUM_UP_MARKER_SCALE: f32 = 3.0;

/// Radius of the image-origin ball, relative to the frustum edge radius.
pub const FRUSTUM_IMAGE_ORIGIN_SCALE: f32 = 9.0;

/// Index of the far-plane top edge within [`frustum_segments`].
const FAR_TOP_SEGMENT: usize = 4;

/// A frustum edge in camera-local space.
pub struct FrustumEdge {
    pub start: Vec3,
    pub end: Vec3,
    pub thickness: f32,
}

/// Frustum edges in camera-local space, thickening the top edge when
/// `up_marker` asks for it.
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
    })
    .collect();

    if up_marker == FrustumUpMarker::Highlight {
        segments[FAR_TOP_SEGMENT].thickness = thickness * FRUSTUM_UP_MARKER_SCALE;
    }

    segments
}

/// Channel floor above which a frustum color counts as near-white.
const FRUSTUM_NEAR_WHITE: f32 = 0.7;

/// Color for the image-origin ball. The thickened edge keeps the frustum's own
/// color, which is what tells several frustums apart; the ball is white so it
/// stays legible against that color, falling back to its complement when the
/// frustum is itself near-white. Always opaque, so the ball still reads on a
/// translucent or fully clear frustum.
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

/// Ball marking the image origin — the far-plane corner holding pixel (0, 0),
/// i.e. top-left as seen through the camera — as `(center, radius)` in
/// camera-local space. With the thickened top edge it says which way up the
/// image is, and which end of that edge it starts from.
pub fn frustum_image_origin_ball(
    points: &[Vec3; 8],
    thickness: f32,
    up_marker: FrustumUpMarker,
) -> Option<(Vec3, f32)> {
    (up_marker == FrustumUpMarker::Highlight)
        .then(|| (points[4], thickness * FRUSTUM_IMAGE_ORIGIN_SCALE))
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
            frustums_up_marker_overlay: false,
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
        for (idx, edge) in segments.iter().enumerate() {
            if idx != 4 {
                assert_eq!(edge.thickness, 0.01);
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
    fn frustum_image_origin_ball_only_for_highlight() {
        let points = frustum_local_points(&perspective(1.0, 1.6, 0.1, 10.0)).unwrap();

        assert!(frustum_image_origin_ball(&points, 0.01, FrustumUpMarker::None).is_none());

        let (center, radius) =
            frustum_image_origin_ball(&points, 0.01, FrustumUpMarker::Highlight).unwrap();
        assert_eq!(center, points[4], "ball sits on the far top-left corner");
        assert!(center.x < 0.0 && center.y > 0.0, "left of and above center");
        assert_eq!(radius, 0.01 * FRUSTUM_IMAGE_ORIGIN_SCALE);
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
