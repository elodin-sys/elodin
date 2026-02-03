//! ViewCube Plugin - A CAD-style navigation widget for 3D viewports
//!
//! Provides a clickable cube with faces, edges, and corners for quick camera orientation,
//! plus rotation arrows for incremental adjustments.
//!
//! # Usage
//!
//! ```rust,ignore
//! use elodin_editor::plugins::view_cube::{ViewCubePlugin, ViewCubeConfig, ViewCubeEvent};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins(DefaultPlugins)
//!         .add_plugins(ViewCubePlugin::default())
//!         .add_systems(Startup, setup)
//!         .add_systems(Update, handle_events)
//!         .run();
//! }
//!
//! fn setup(mut commands: Commands, ...) {
//!     let camera = commands.spawn(Camera3d::default()).id();
//!     view_cube::spawn::spawn_view_cube(&mut commands, ..., camera);
//! }
//!
//! fn handle_events(mut events: EventReader<ViewCubeEvent>, ...) {
//!     for event in events.read() {
//!         // Handle camera rotation
//!     }
//! }
//! ```

mod components;
mod config;
mod events;
mod interactions;
pub mod spawn;
mod theme;

pub use components::*;
pub use config::*;
pub use events::*;
pub use theme::ViewCubeColors;

use bevy::picking::prelude::*;
use bevy::prelude::*;

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
            .add_systems(Update, interactions::setup_cube_elements)
            .add_observer(interactions::on_cube_hover_start)
            .add_observer(interactions::on_cube_hover_end)
            .add_observer(interactions::on_cube_click)
            .add_observer(interactions::on_arrow_hover_start)
            .add_observer(interactions::on_arrow_hover_end)
            .add_observer(interactions::on_arrow_click);
    }
}
