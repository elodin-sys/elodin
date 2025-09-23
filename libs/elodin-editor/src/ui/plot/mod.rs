pub mod data;
pub use data::*;
pub mod gpu;
mod widget;
pub use widget::*;
mod state;
pub use state::*;

use bevy::{
    app::{Plugin, PostUpdate, Startup, Update},
    ecs::schedule::IntoScheduleConfigs, // for `.after(...)`
};

pub struct PlotPlugin;
impl Plugin for PlotPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app
            // Resources
            .init_resource::<CollectedGraphData>()
            .init_resource::<LockedGraphsLeader>()
            .init_resource::<LockTracker>() // Tracks per-entity lock state edges
            // Systems
            .add_systems(Startup, setup_pkt_handler)
            // Input/interaction updates
            .add_systems(Update, zoom_graph)
            .add_systems(Update, graph_touch)
            .add_systems(Update, pan_graph)
            .add_systems(Update, reset_graph)
            // Lock edge tracking must run after interaction updates
            .add_systems(
                Update,
                widget::track_lock_toggles
                    .after(zoom_graph)
                    .after(pan_graph)
                    .after(reset_graph),
            )
            // Leader-driven X-sync must run after lock edge handling
            .add_systems(
                Update,
                widget::sync_locked_graphs
                    .after(widget::track_lock_toggles)
                    .after(zoom_graph)
                    .after(pan_graph)
                    .after(reset_graph),
            )
            // Data lifecyle / GPU
            .add_systems(PostUpdate, queue_timestamp_read)
            .add_systems(Update, collect_garbage)
            .add_systems(Update, sync_graphs)
            .add_systems(Update, auto_y_bounds.after(sync_graphs))
            .add_plugins(gpu::PlotGpuPlugin);
    }
}
