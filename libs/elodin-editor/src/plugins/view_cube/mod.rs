//! ViewCube plugin for CAD-style camera orientation in 3D viewports.

mod camera;
mod components;
mod config;
mod events;
mod interactions;
pub mod spawn;
mod theme;

pub use camera::{NeedsInitialSnap, ViewCubeTargetCamera};
pub use components::*;
pub use config::*;
pub use events::*;
pub use spawn::SpawnedViewCube;
pub use theme::ViewCubeColors;

use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy_fontmesh::prelude::*;

#[derive(Resource)]
struct CurrentColorMode {
    mode: String,
}

impl Default for CurrentColorMode {
    fn default() -> Self {
        Self {
            mode: crate::ui::colors::current_selection().mode,
        }
    }
}

#[derive(Default)]
pub struct ViewCubePlugin {
    pub config: ViewCubeConfig,
}

impl Plugin for ViewCubePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<MeshPickingPlugin>() {
            app.add_plugins(MeshPickingPlugin);
        }

        app.insert_resource(self.config.clone())
            .init_resource::<HoveredElement>()
            .init_resource::<OriginalMaterials>()
            .init_resource::<CurrentColorMode>()
            .add_message::<ViewCubeEvent>()
            .add_plugins(FontMeshPlugin)
            .add_systems(Update, interactions::setup_cube_elements)
            .add_systems(Update, update_theme_on_mode_change)
            .add_observer(interactions::on_cube_hover_start)
            .add_observer(interactions::on_cube_hover_end)
            .add_observer(interactions::on_cube_click)
            .add_observer(interactions::on_arrow_hover_start)
            .add_observer(interactions::on_arrow_hover_end)
            .add_observer(interactions::on_arrow_click);

        app.init_resource::<camera::ViewCubeArrowTargetCache>()
            .add_systems(Update, camera::handle_view_cube_editor)
            .add_systems(Update, camera::snap_initial_camera);

        if self.config.sync_with_camera {
            app.add_systems(PostUpdate, camera::sync_view_cube_rotation);
        }

        app.add_systems(Update, camera::apply_render_layers_to_scene);
    }
}

fn update_theme_on_mode_change(
    mut current_mode: ResMut<CurrentColorMode>,
    cube_elements: Query<(Entity, &CubeElement), With<ViewCubeSetup>>,
    children_query: Query<&Children>,
    material_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut original_materials: ResMut<OriginalMaterials>,
    arrows: Query<(Entity, &RotationArrow)>,
) {
    let new_mode = crate::ui::colors::current_selection().mode;
    if new_mode == current_mode.mode {
        return;
    }
    current_mode.mode = new_mode;

    let colors = ViewCubeColors::default();

    for (entity, element) in cube_elements.iter() {
        let new_color = colors.get_element_color(element);

        if let Ok(children) = children_query.get(entity) {
            for child in children.iter() {
                original_materials.colors.insert(child, new_color);
                if let Ok(mat_handle) = material_query.get(child)
                    && let Some(mat) = materials.get_mut(&mat_handle.0)
                {
                    mat.base_color = new_color;
                }
            }
        }

        original_materials.colors.insert(entity, new_color);
        if let Ok(mat_handle) = material_query.get(entity)
            && let Some(mat) = materials.get_mut(&mat_handle.0)
        {
            mat.base_color = new_color;
        }
    }

    for (entity, _) in arrows.iter() {
        if let Ok(mat_handle) = material_query.get(entity)
            && let Some(mat) = materials.get_mut(&mat_handle.0)
        {
            mat.base_color = colors.arrow_normal;
        }
    }
}
