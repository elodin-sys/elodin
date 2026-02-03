//! ViewCube Plugin - A CAD-style navigation widget for 3D viewports
//!
//! Provides a clickable cube with faces, edges, and corners for quick camera orientation,
//! plus rotation arrows for incremental adjustments.
//!
//! # Usage
//!
//! ## With auto_rotate (default)
//!
//! ```rust,ignore
//! use elodin_editor::plugins::view_cube::{
//!     ViewCubePlugin, ViewCubeConfig, ViewCubeTargetCamera, spawn::spawn_view_cube
//! };
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(ViewCubePlugin::default()) // auto_rotate = true
//!         .add_systems(Startup, setup)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands, ...) {
//!     // Add ViewCubeTargetCamera to the camera you want to control
//!     let camera = commands.spawn((Camera3d::default(), ViewCubeTargetCamera)).id();
//!     spawn_view_cube(&mut commands, ..., camera);
//! }
//! ```
//!
//! ## With manual event handling
//!
//! ```rust,ignore
//! let config = ViewCubeConfig { auto_rotate: false, ..default() };
//! app.add_plugins(ViewCubePlugin { config });
//!
//! // Then handle ViewCubeEvent in your own systems
//! fn handle_events(mut events: MessageReader<ViewCubeEvent>, ...) {
//!     for event in events.read() {
//!         // Handle camera rotation manually
//!     }
//! }
//! ```

mod camera;
mod components;
mod config;
mod events;
mod interactions;
pub mod spawn;
mod theme;

pub use camera::{CameraAnimation, ViewCubeTargetCamera};
pub use components::*;
pub use config::*;
pub use events::*;
pub use theme::ViewCubeColors;

use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy_fontmesh::prelude::*;

/// Main plugin for the ViewCube widget
#[derive(Default)]
pub struct ViewCubePlugin {
    pub config: ViewCubeConfig,
}

impl Plugin for ViewCubePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .init_resource::<HoveredElement>()
            .init_resource::<OriginalMaterials>()
            .add_message::<ViewCubeEvent>()
            .add_plugins(MeshPickingPlugin)
            .add_plugins(FontMeshPlugin)
            .add_systems(Update, interactions::setup_cube_elements)
            .add_observer(interactions::on_cube_hover_start)
            .add_observer(interactions::on_cube_hover_end)
            .add_observer(interactions::on_cube_click)
            .add_observer(interactions::on_arrow_hover_start)
            .add_observer(interactions::on_arrow_hover_end)
            .add_observer(interactions::on_arrow_click);

        // Add camera control systems when auto_rotate is enabled
        if self.config.auto_rotate {
            app.init_resource::<CameraAnimation>()
                .add_systems(Update, (camera::handle_view_cube_camera, camera::animate_camera));
        }
    }
}
