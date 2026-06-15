//! Picking integration for the ViewCube overlay.

use std::collections::HashSet;

use bevy::picking::{PickingSystems, backend::ray::RayMap, mesh_picking::update_hits};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::plugins::navigation_gizmo::NavGizmoParent;
use crate::ui::input_owner::{PointerOwner, UiInputOwners};

use super::components::ViewCubeCamera;

/// Restrict mesh picking to the ViewCube overlay camera while the pointer is
/// over the overlay.
///
/// Picking generates one ray per active camera. Over the cube's screen corner
/// the main viewport (and any graph/UI cameras) also raycast the scene, so a
/// km-scale terrain GLB on the default render layer (e.g. the Apollo landing
/// tile) wins the hit and steals hover/click from the cube. Whitelisting only
/// the overlay camera's rays keeps cube interactions isolated.
pub fn filter_picking_rays_for_view_cube_overlay(
    mut ray_map: ResMut<RayMap>,
    input_owners: Res<UiInputOwners>,
    primary_window: Query<(Entity, &Window), With<PrimaryWindow>>,
    overlay_cameras: Query<(Entity, &Camera, &NavGizmoParent), With<ViewCubeCamera>>,
) {
    let Ok((window_entity, window)) = primary_window.single() else {
        return;
    };
    let allowed =
        overlay_cameras_for_cursor(window, &overlay_cameras, &input_owners, window_entity);
    if allowed.is_empty() {
        return;
    }

    let allowed: HashSet<Entity> = allowed.into_iter().collect();
    ray_map
        .map
        .retain(|ray_id, _| allowed.contains(&ray_id.camera));
}

/// Overlay cameras whose viewport currently contains the cursor (falling back to
/// the input-owner region, which is updated a frame later in `Update`).
fn overlay_cameras_for_cursor(
    window: &Window,
    overlay_cameras: &Query<(Entity, &Camera, &NavGizmoParent), With<ViewCubeCamera>>,
    input_owners: &UiInputOwners,
    window_entity: Entity,
) -> Vec<Entity> {
    let cursor = window.physical_cursor_position();
    let mut allowed = Vec::new();

    for (entity, camera, _) in overlay_cameras.iter() {
        if !camera.is_active {
            continue;
        }
        let Some(viewport) = camera.viewport.as_ref() else {
            continue;
        };
        if cursor_in_physical_viewport(cursor, viewport.physical_position, viewport.physical_size) {
            allowed.push(entity);
        }
    }

    if allowed.is_empty()
        && let PointerOwner::ViewCube {
            camera: main_camera,
        } = input_owners.owner_for_window(window_entity)
    {
        for (entity, _, parent) in overlay_cameras.iter() {
            if parent.main_camera == main_camera {
                allowed.push(entity);
            }
        }
    }

    allowed
}

fn cursor_in_physical_viewport(cursor: Option<Vec2>, position: UVec2, size: UVec2) -> bool {
    let Some(cursor) = cursor else {
        return false;
    };
    if size.x == 0 || size.y == 0 {
        return false;
    }
    let min = position.as_vec2();
    let max = min + size.as_vec2();
    cursor.x >= min.x && cursor.x < max.x && cursor.y >= min.y && cursor.y < max.y
}

pub(crate) fn plugin(app: &mut App) {
    app.add_systems(
        PreUpdate,
        filter_picking_rays_for_view_cube_overlay
            .before(update_hits)
            .in_set(PickingSystems::Backend),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_inside_and_outside_physical_viewport() {
        let pos = UVec2::new(700, 100);
        let size = UVec2::new(128, 128);
        assert!(cursor_in_physical_viewport(
            Some(Vec2::new(710.0, 150.0)),
            pos,
            size
        ));
        assert!(!cursor_in_physical_viewport(
            Some(Vec2::new(690.0, 150.0)),
            pos,
            size
        ));
        assert!(!cursor_in_physical_viewport(None, pos, size));
        assert!(!cursor_in_physical_viewport(
            Some(Vec2::new(710.0, 150.0)),
            pos,
            UVec2::ZERO
        ));
    }
}
