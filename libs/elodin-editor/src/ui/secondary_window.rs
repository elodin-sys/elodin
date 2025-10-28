use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::multi_window::{SecondaryWindowHandle, WindowContent, SecondaryWindows};
use crate::ui::tiles::Pane;
use crate::ui::ViewportRect;

/// Plugin to handle secondary window UI rendering
pub struct SecondaryWindowUiPlugin;

impl Plugin for SecondaryWindowUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, render_secondary_window_ui)
            .add_systems(Update, update_secondary_viewport_rects);
    }
}

/// System to render UI in secondary windows
fn render_secondary_window_ui(
    _contexts: EguiContexts,
    _secondary_windows: Query<(&Window, &SecondaryWindowHandle, Entity)>,
) {
    // For now, we'll just skip the actual rendering - this needs
    // a different approach since bevy_egui doesn't directly support
    // multiple window contexts easily
    // TODO: Implement proper multi-window egui rendering
}

#[allow(dead_code)]
/// Render a single pane in a secondary window
fn render_pane_in_window(
    ctx: &mut egui::Context,
    world: &mut World,
    mut pane: Pane,
    _window_entity: Entity,
) {
    use crate::ui::{colors::get_scheme, widgets::WidgetSystemExt};
    use egui::Frame;
    
    // Create a central panel that fills the entire window
    egui::CentralPanel::default()
        .frame(Frame {
            fill: get_scheme().bg_primary,
            ..Default::default()
        })
        .show(ctx, |ui| {
            // Get or create tile icons (these are needed for some pane types)
            let icons = get_or_create_tile_icons(world, ui);
            
            // For viewport panes, we just need to set the viewport rect
            // The actual 3D rendering is handled by the camera system
            match &mut pane {
                Pane::Viewport(viewport_pane) => {
                    // Store the rect for the viewport
                    viewport_pane.rect = Some(ui.available_rect_before_wrap());
                    
                    // The 3D content will be rendered by the camera system
                    // We just need to ensure the viewport rect is updated
                },
                Pane::Graph(graph_pane) => {
                    graph_pane.rect = Some(ui.available_rect_before_wrap());
                    
                    // Render the graph widget
                    ui.add_widget_with::<crate::ui::plot::PlotWidget>(
                        world,
                        "graph_secondary",
                        (graph_pane.id, icons.scrub)
                    );
                },
                Pane::Monitor(monitor_pane) => {
                    ui.add_widget_with::<crate::ui::monitor::MonitorWidget>(
                        world,
                        "monitor_secondary",
                        monitor_pane.clone()
                    );
                },
                Pane::QueryTable(query_table_pane) => {
                    ui.add_widget_with::<crate::ui::query_table::QueryTableWidget>(
                        world,
                        "query_table_secondary",
                        query_table_pane.clone()
                    );
                },
                Pane::QueryPlot(query_plot_pane) => {
                    let mut pane_clone = query_plot_pane.clone();
                    pane_clone.rect = Some(ui.available_rect_before_wrap());
                    
                    ui.add_widget_with::<crate::ui::query_plot::QueryPlotWidget>(
                        world,
                        "query_plot_secondary",
                        pane_clone
                    );
                },
                Pane::Dashboard(dashboard_pane) => {
                    ui.add_widget_with::<crate::ui::dashboard::DashboardWidget>(
                        world,
                        "dashboard_secondary",
                        dashboard_pane.entity
                    );
                },
                Pane::Hierarchy => {
                    let hierarchy_icons = crate::ui::hierarchy::Hierarchy {
                        search: icons.search,
                        entity: icons.entity,
                        chevron: icons.chevron,
                    };
                    
                    ui.add_widget_with::<crate::ui::hierarchy::HierarchyContent>(
                        world,
                        "hierarchy_secondary",
                        hierarchy_icons
                    );
                },
                Pane::Inspector => {
                    let inspector_icons = crate::ui::inspector::InspectorIcons {
                        chart: icons.chart,
                        add: icons.add,
                        subtract: icons.subtract,
                        setting: icons.setting,
                        search: icons.search,
                    };
                    
                    ui.add_widget_with::<crate::ui::inspector::InspectorContent>(
                        world,
                        "inspector_secondary",
                        (inspector_icons, false) // false = not in tile system
                    );
                },
                Pane::SchematicTree(tree_pane) => {
                    let tree_icons = crate::ui::schematic::tree::TreeIcons {
                        chevron: icons.chevron,
                        search: icons.search,
                        container: icons.container,
                        plot: icons.plot,
                        viewport: icons.viewport,
                        add: icons.add,
                    };
                    
                    ui.add_widget_with::<crate::ui::schematic::tree::TreeWidget>(
                        world,
                        "schematic_tree_secondary",
                        (tree_icons, tree_pane.entity)
                    );
                },
                Pane::ActionTile(action_pane) => {
                    ui.add_widget_with::<crate::ui::actions::ActionTileWidget>(
                        world,
                        "action_tile_secondary",
                        action_pane.entity
                    );
                },
                Pane::VideoStream(video_pane) => {
                    ui.add_widget_with::<crate::ui::video_stream::VideoStreamWidget>(
                        world,
                        "video_stream_secondary",
                        video_pane.clone()
                    );
                },
            }
        });
}

#[allow(dead_code)]
/// Helper to get or create tile icons for rendering
fn get_or_create_tile_icons(_world: &mut World, _ui: &egui::Ui) -> crate::ui::tiles::TileIcons {
    use crate::ui::tiles::TileIcons;
    // Try to get existing icons from resources or create new ones
    
    // This is a simplified version - in production you'd want to properly
    // cache these icons in a resource
    TileIcons {
        add: egui::TextureId::default(),
        close: egui::TextureId::default(),
        scrub: egui::TextureId::default(),
        chart: egui::TextureId::default(),
        search: egui::TextureId::default(),
        entity: egui::TextureId::default(),
        chevron: egui::TextureId::default(),
        subtract: egui::TextureId::default(),
        setting: egui::TextureId::default(),
        container: egui::TextureId::default(),
        plot: egui::TextureId::default(),
        viewport: egui::TextureId::default(),
        tile_3d_viewer: egui::TextureId::default(),
        tile_graph: egui::TextureId::default(),
    }
}

/// Update viewport rects for cameras in secondary windows
fn update_secondary_viewport_rects(
    mut commands: Commands,
    _secondary_windows: Res<SecondaryWindows>,
    windows: Query<(&Window, &SecondaryWindowHandle)>,
) {
    for (_window, handle) in windows.iter() {
        if let WindowContent::Pane(Pane::Viewport(viewport_pane)) = &handle.content {
            if let Some(camera_entity) = handle.camera {
                if let Ok(mut entity_commands) = commands.get_entity(camera_entity) {
                    entity_commands.try_insert(ViewportRect(viewport_pane.rect));
                }
            }
        }
    }
}
