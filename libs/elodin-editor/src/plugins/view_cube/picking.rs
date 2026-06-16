//! Picking integration for the ViewCube overlay.

use std::collections::HashSet;

use bevy::picking::{PickingSystems, backend::ray::RayMap, mesh_picking::update_hits};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_egui::egui;

use crate::plugins::navigation_gizmo::NavGizmoParent;
use crate::ui::input_owner::UiInputOwners;

use super::components::ViewCubeCamera;

/// Stop a hovered ViewCube's main viewport camera from stealing the cube's hit.
///
/// Picking generates one ray per active camera, and `RayMap` is global across
/// every viewport/tile. Over the cube's screen corner the cube's main viewport
/// also raycasts the scene, so a km-scale terrain GLB on the default render
/// layer (e.g. the Apollo landing tile) wins the hit and steals hover/click
/// from the cube. We drop only the ray of each main camera whose cube is
/// currently hovered, leaving every other camera (other viewports, graphs, UI,
/// and the cube's own overlay camera) untouched so picking stays per-viewport.
pub fn filter_picking_rays_for_view_cube_overlay(
    mut ray_map: ResMut<RayMap>,
    input_owners: Res<UiInputOwners>,
    primary_window: Query<(Entity, &Window), With<PrimaryWindow>>,
    overlay_cameras: Query<(&Camera, &NavGizmoParent), With<ViewCubeCamera>>,
) {
    let Ok((window_entity, window)) = primary_window.single() else {
        return;
    };
    let blocked =
        blocked_main_cameras_for_cursor(window, &overlay_cameras, &input_owners, window_entity);
    if blocked.is_empty() {
        return;
    }

    ray_map
        .map
        .retain(|ray_id, _| !blocked.contains(&ray_id.camera));
}

/// Main viewport cameras whose ViewCube overlay is hovered, and whose scene
/// rays should therefore be suppressed.
///
/// Combines two independent signals, both evaluated against the *current*
/// cursor position so the restriction is dropped the same frame the pointer
/// leaves the cube (never lingering and suppressing main-viewport picking):
/// - the cursor is physically inside an overlay camera's viewport, or
/// - egui resolves the cube as the input owner at the cursor position.
///
/// The owner is resolved at the live cursor (`permits_view_cube_at`) rather
/// than read from the cached last-frame resolution (`owner_for_window`), which
/// would lag one frame behind this `PreUpdate` system. Taking the union keeps
/// the filter active on entry even when the overlay viewport rect and the egui
/// cube rect disagree momentarily — exactly when a km-scale terrain GLB would
/// otherwise steal the cube's hit — without keeping it active after the pointer
/// has left.
fn blocked_main_cameras_for_cursor(
    window: &Window,
    overlay_cameras: &Query<(&Camera, &NavGizmoParent), With<ViewCubeCamera>>,
    input_owners: &UiInputOwners,
    window_entity: Entity,
) -> HashSet<Entity> {
    let physical_cursor = window.physical_cursor_position();
    let cube_owner_pos = window.cursor_position().map(|pos| egui::pos2(pos.x, pos.y));
    let mut blocked = HashSet::new();

    for (camera, parent) in overlay_cameras.iter() {
        if !camera.is_active {
            continue;
        }
        let cursor_in_viewport = camera.viewport.as_ref().is_some_and(|viewport| {
            cursor_in_physical_viewport(
                physical_cursor,
                viewport.physical_position,
                viewport.physical_size,
            )
        });
        let owned_by_view_cube = cube_owner_pos.is_some_and(|pos| {
            input_owners.permits_view_cube_at(window_entity, parent.main_camera, pos)
        });
        if cursor_in_viewport || owned_by_view_cube {
            blocked.insert(parent.main_camera);
        }
    }

    blocked
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
