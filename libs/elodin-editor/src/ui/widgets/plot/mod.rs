pub mod data;
use bevy::app::{Plugin, Startup, Update};
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
            .add_systems(Update, pan_graph)
            .add_systems(Update, reset_graph)
            .add_systems(Update, queue_timestamp_read)
            //.add_systems(Update, collect_entity_data)
            .add_plugins(gpu::PlotGpuPlugin);
    }
}
