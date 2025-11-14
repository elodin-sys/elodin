use bevy::log::warn;
use bevy::math::DVec3;
use bevy::transform::components::Transform;

/// Returns the camera's view-space anchor when looking at the world origin.
///
/// This falls back to `None` if the transform cannot be inverted (for example,
/// because it contains invalid scale values), which signals the caller to let
/// `EditorCam` reuse its last known depth instead of corrupting it with NaNs.
pub fn camera_anchor_from_transform(transform: &Transform) -> Option<DVec3> {
    let anchor = transform
        .compute_matrix()
        .as_dmat4()
        .inverse()
        .transform_point3(DVec3::ZERO);

    let anchor_is_valid = anchor.x.is_finite() && anchor.y.is_finite() && anchor.z.is_finite();
    if !anchor_is_valid {
        warn!(
            "camera_anchor_from_transform produced non-finite anchor: ({:.3}, {:.3}, {:.3})",
            anchor.x, anchor.y, anchor.z
        );
    }
    anchor_is_valid.then_some(anchor)
}
