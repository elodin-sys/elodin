pub mod data;

pub use hamann_chen_line::{
    select_point_indices, select_polyline2_indices, select_polyline3_indices,
    select_time_value_indices, select_trajectory_time_norm_indices,
};

pub use data::{
    BufferShardAlloc, CHUNK_COUNT, CHUNK_LEN, CollectedGraphData, CurveCompressSettings, Line,
    OVERVIEW_MAX_POINTS, PlotDataComponent, XYLine, collect_garbage,
    maybe_compress_all_graph_lines, queue_timestamp_read, setup_pkt_handler,
};

pub mod gpu;
pub use gpu::{
    INDEX_BUFFER_LEN, INDEX_BUFFER_SIZE, LineBundle, LineConfig, LineHandle, LineUniform,
    LineWidgetWidth, PlotGpuPlugin, VALUE_BUFFER_SIZE,
};

mod widget;
pub use widget::{
    AXIS_LABEL_MARGIN, LockTracker, NOTCH_LENGTH, PlotBounds, PlotDataSource, PlotWidget,
    STEPS_X_WIDTH_DIVISOR, STEPS_Y_HEIGHT_DIVISOR, TimeseriesPlot, XSyncClock, auto_y_bounds,
    draw_borders, draw_y_axis, get_inner_rect, graph_touch, pan_graph, pretty_round, reset_graph,
    sync_graphs, sync_locked_graphs, track_lock_toggles, zoom_graph,
};

mod state;
pub use state::*;

use crate::ui::UiInputConsumerSet;
use bevy::{
    app::{Plugin, PostUpdate, Startup, Update},
    ecs::schedule::IntoScheduleConfigs,
};

pub struct PlotPlugin;
impl Plugin for PlotPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_resource::<CollectedGraphData>()
            .init_resource::<CurveCompressSettings>()
            .init_resource::<LockTracker>()
            .init_resource::<XSyncClock>()
            .add_systems(Startup, setup_pkt_handler)
            .add_systems(Update, zoom_graph.in_set(UiInputConsumerSet))
            .add_systems(Update, graph_touch.in_set(UiInputConsumerSet))
            .add_systems(Update, pan_graph.in_set(UiInputConsumerSet))
            .add_systems(Update, reset_graph.in_set(UiInputConsumerSet))
            .add_systems(
                Update,
                track_lock_toggles
                    .after(zoom_graph)
                    .after(pan_graph)
                    .after(reset_graph),
            )
            .add_systems(
                Update,
                sync_locked_graphs
                    .after(track_lock_toggles)
                    .after(zoom_graph)
                    .after(pan_graph)
                    .after(reset_graph),
            )
            .add_systems(PostUpdate, queue_timestamp_read)
            .add_systems(Update, collect_garbage)
            .add_systems(Update, sync_graphs)
            .add_systems(Update, auto_y_bounds.after(sync_graphs))
            .add_plugins(PlotGpuPlugin);
    }
}
