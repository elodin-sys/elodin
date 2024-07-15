pub mod data;
use bevy::app::Plugin;
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
            .add_plugins(gpu::PlotGpuPlugin);
    }
}
