pub mod data;
use bevy::{
    app::{Plugin, PostUpdate, Startup, Update},
    ecs::schedule::IntoScheduleConfigs,
};
pub use data::*;
pub mod gpu;
mod widget;
pub use widget::*;
mod state;
pub use state::*;

pub struct PlotPlugin;
impl Plugin for PlotPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .add_systems(Startup, setup_pkt_handler)
            .add_systems(Update, zoom_graph)
            .add_systems(Update, graph_touch)
            .add_systems(Update, pan_graph)
            .add_systems(Update, reset_graph)
            .add_systems(PostUpdate, queue_timestamp_read)
            .add_systems(Update, collect_garbage)
            .add_systems(Update, sync_graphs)
            .add_systems(Update, auto_y_bounds.after(sync_graphs))
            .add_plugins(gpu::PlotGpuPlugin);
    }
}
