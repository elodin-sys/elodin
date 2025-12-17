use bevy::prelude::*;
use bevy_egui::egui;

use crate::ui::{monitor::MonitorPane, query_plot, query_table::QueryTablePane, video_stream};

#[derive(Clone, Debug)]
pub struct TileIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
    pub scrub: egui::TextureId,
    pub tile_3d_viewer: egui::TextureId,
    pub tile_graph: egui::TextureId,
    pub subtract: egui::TextureId,
    pub setting: egui::TextureId,
    pub search: egui::TextureId,
    pub chart: egui::TextureId,
    pub chevron: egui::TextureId,
    pub plot: egui::TextureId,
    pub viewport: egui::TextureId,
    pub container: egui::TextureId,
    pub entity: egui::TextureId,
}

#[derive(Clone, Debug)]
pub enum Pane {
    Viewport(ViewportPane),
    Graph(GraphPane),
    Monitor(MonitorPane),
    QueryTable(QueryTablePane),
    QueryPlot(query_plot::QueryPlotPane),
    ActionTile(ActionTilePane),
    VideoStream(video_stream::VideoStreamPane),
    Dashboard(DashboardPane),
    Hierarchy,
    Inspector,
    SchematicTree(TreePane),
}

#[derive(Default, Clone, Debug)]
pub struct ViewportPane {
    pub camera: Option<Entity>,
    pub nav_gizmo: Option<Entity>,
    pub nav_gizmo_camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
    pub label: String,
    pub grid_layer: Option<usize>,
    pub viewport_layer: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct GraphPane {
    pub id: Entity,
    pub label: String,
    pub rect: Option<egui::Rect>,
}

#[derive(Resource, Default)]
pub struct ViewportContainsPointer(pub bool);

#[derive(Clone, Debug)]
pub struct ActionTilePane {
    pub entity: Entity,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct TreePane {
    pub entity: Entity,
}

#[derive(Clone, Debug)]
pub struct DashboardPane {
    pub entity: Entity,
    pub label: String,
}
