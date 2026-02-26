use crate::ui::tiles::ViewportConfig;
use bevy::prelude::*;

pub type MainViewportQueryItem = (
    Entity,
    &'static Camera,
    &'static Projection,
    &'static GlobalTransform,
    Option<&'static ViewportConfig>,
);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_local_points_basic() {
        let persp = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_2,
            aspect_ratio: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let pts = frustum_local_points(&persp).unwrap();
        for p in &pts[0..4] {
            assert!((p.z - (-0.1)).abs() < 1e-5, "near plane z should be -near");
        }
        for p in &pts[4..8] {
            assert!(
                (p.z - (-100.0)).abs() < 1e-3,
                "far plane z should be -far"
            );
        }
    }

    #[test]
    fn frustum_local_points_rejects_degenerate() {
        let bad_near = PerspectiveProjection {
            fov: 1.0,
            aspect_ratio: 1.0,
            near: 0.0,
            far: 10.0,
        };
        assert!(frustum_local_points(&bad_near).is_none());

        let bad_far = PerspectiveProjection {
            fov: 1.0,
            aspect_ratio: 1.0,
            near: 10.0,
            far: 5.0,
        };
        assert!(frustum_local_points(&bad_far).is_none());

        let bad_fov = PerspectiveProjection {
            fov: 0.0,
            aspect_ratio: 1.0,
            near: 0.1,
            far: 10.0,
        };
        assert!(frustum_local_points(&bad_fov).is_none());

        let bad_aspect = PerspectiveProjection {
            fov: 1.0,
            aspect_ratio: 0.0,
            near: 0.1,
            far: 10.0,
        };
        assert!(frustum_local_points(&bad_aspect).is_none());
    }

    #[test]
    fn frustum_local_points_aspect_ratio() {
        let persp = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_2,
            aspect_ratio: 2.0,
            near: 1.0,
            far: 10.0,
        };
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
