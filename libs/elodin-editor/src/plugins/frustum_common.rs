use crate::plugins::render_layer_alloc::RenderLayerLease;
use crate::sensor_camera::SensorCameraFrustumSource;
use crate::ui::tiles::{DEFAULT_VIEWPORT_FAR, ViewportConfig};
use bevy::prelude::*;

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
    fn color_component_to_u8_boundaries() {
        assert_eq!(color_component_to_u8(0.0), 0);
        assert_eq!(color_component_to_u8(1.0), 255);
        assert_eq!(color_component_to_u8(0.5), 128);
        assert_eq!(color_component_to_u8(-1.0), 0);
        assert_eq!(color_component_to_u8(2.0), 255);
    }
}
