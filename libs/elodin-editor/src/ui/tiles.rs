use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Exposure, PhysicalCameraParameters};
use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    ecs::system::{SystemParam, SystemState},
    input::keyboard::Key,
    post_process::bloom::Bloom,
    prelude::*,
    window::{Monitor, PrimaryWindow, Window, WindowPosition},
};
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion, OrbitConstraint};
use bevy_egui::{
    EguiContexts, EguiTextureHandle,
    egui::{self, Color32, CornerRadius, Frame, Id, RichText, Stroke, Ui, Visuals, vec2},
};
use egui::UiBuilder;
use egui::response::Flags;
use egui_tiles::{Container, Tile, TileId, Tiles};
use impeller2_wkt::{Dashboard, Graph, Viewport, WindowRect};
use smallvec::{SmallVec, smallvec};
use std::collections::{BTreeMap, HashMap};
use std::{
    fmt::Write as _,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    monitor::MonitorHandle,
    window::Window as WinitWindow,
};

use super::{
    PaneName, SelectedObject, ViewportRect, WindowUiState,
    actions::{ActionTile, ActionTileWidget},
    button::{EImageButton, ETileButton},
    colors::{self, get_scheme, with_opacity},
    command_palette::{CommandPaletteState, palette_items},
    dashboard::{DashboardWidget, DashboardWidgetArgs, spawn_dashboard},
    data_overview::{DataOverviewPane, DataOverviewWidget},
    hierarchy::{Hierarchy, HierarchyContent},
    images,
    inspector::{InspectorContent, InspectorIcons},
    monitor::{MonitorPane, MonitorWidget},
    plot::{GraphBundle, GraphState, PlotWidget},
    query_plot::QueryPlotData,
    query_table::{QueryTableData, QueryTablePane, QueryTableWidget},
    schematic::{graph_label, viewport_label},
    video_stream::{IsTileVisible, VideoDecoderHandle, VideoStreamWidgetArgs},
    widgets::{RootWidgetSystem, WidgetSystem, WidgetSystemExt},
};
use crate::{
    EqlContext, GridHandle, MainCamera,
    object_3d::{EditableEQL, compile_eql_expr},
    plugins::{
        LogicalKeyState,
        gizmos::GIZMO_RENDER_LAYER,
        navigation_gizmo::{RenderLayerAlloc, spawn_gizmo},
    },
    ui::dashboard::NodeUpdaterParams,
};

pub(crate) mod sidebar;

use sidebar::{
    SidebarKind, SidebarMaskState, apply_share_updates, collect_sidebar_gutter_updates,
    fix_invalid_drops, tab_add_visible, tab_title_visible, tile_is_sidebar,
};

pub(crate) fn plugin(app: &mut App) {
    app.register_type::<WindowId>()
        .add_message::<WindowRelayout>()
        .add_systems(Startup, setup_primary_window_state);
}

fn setup_primary_window_state(
    primary_window: Query<Entity, With<PrimaryWindow>>,
    mut commands: Commands,
) {
    let Some(id) = primary_window.iter().next() else {
        warn!("No primary window to setup");
        return;
    };
    // TODO: Setup this descriptor with the path when its known.
    let descriptor = WindowDescriptor::default();
    let state = WindowState {
        descriptor,
        graph_entities: vec![],
        tile_state: TileState::new(egui::Id::new("main_tab_tree")),
        ui_state: WindowUiState::default(),
    };
    commands.entity(id).insert((state, WindowId(0)));
}

#[derive(Component)]
pub struct ViewportConfig {
    pub show_arrows: bool,
    pub viewport_layer: Option<usize>,
}

#[derive(Clone)]
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

const SIDEBAR_COLLAPSED_SHARE: f32 = 0.01;

#[derive(Clone)]
pub struct TileState {
    pub tree: egui_tiles::Tree<Pane>,
    pub tree_actions: SmallVec<[TreeAction; 4]>,
    pub graphs: HashMap<TileId, Entity>,
    pub container_titles: HashMap<TileId, String>,
    pub sidebar_state: SidebarMaskState,
    tree_id: Id,
}

#[derive(Clone, Debug, Default)]
pub struct WindowDescriptor {
    pub path: Option<PathBuf>,
    pub title: Option<String>,
    pub screen: Option<usize>,
    pub mode: Option<String>,
    pub screen_rect: Option<WindowRect>,
}

impl WindowDescriptor {
    pub fn wants_explicit_layout(&self) -> bool {
        self.screen.is_some() || self.screen_rect.is_some()
    }
}

/// Events dealing with window layout
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub enum WindowRelayout {
    /// Move window to given screen.
    Screen { window: Entity, screen: usize },
    /// Set window to given rect.
    Rect { window: Entity, rect: WindowRect },
    /// Update all window descriptors.
    UpdateDescriptors,
}

/// The primary window is 0; all other windows are secondary windows.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
pub struct WindowId(pub u32);

impl WindowId {
    /// Returns true if this is the primary window.
    pub fn is_primary(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WindowId {
    /// Return a new `WindowId` that's auto-incremented starting from 1.
    fn default() -> Self {
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        WindowId(id)
    }
}

#[derive(Component, Clone)]
pub struct WindowState {
    pub descriptor: WindowDescriptor,
    pub graph_entities: Vec<Entity>,
    pub tile_state: TileState,
    pub ui_state: WindowUiState,
}

impl WindowState {
    pub fn update_descriptor_from_window(
        &mut self,
        window: &Window,
        screens: &Query<(Entity, &Monitor)>,
    ) {
        let position = match window.position {
            WindowPosition::At(pos) => pos,
            WindowPosition::Centered(_) | WindowPosition::Automatic => IVec2::ZERO,
        };
        let size = window.resolution.physical_size();
        let center = IVec2::new(
            position.x + size.x as i32 / 2,
            position.y + size.y as i32 / 2,
        );

        let mut best: Option<(usize, i32)> = None;
        let mut best_bounds: Option<(IVec2, UVec2)> = None;
        for (index, (_, screen)) in screens.iter().enumerate() {
            let min = screen.physical_position;
            let size = screen.physical_size();
            let max = IVec2::new(min.x + size.x as i32, min.y + size.y as i32);
            if center.x >= min.x && center.x < max.x && center.y >= min.y && center.y < max.y {
                let distance = (center.x - min.x).abs() + (center.y - min.y).abs();
                if best.map(|(_, d)| distance < d).unwrap_or(true) {
                    best = Some((index, distance));
                    best_bounds = Some((min, size));
                }
            }
        }

        if let Some((index, _)) = best {
            self.descriptor.screen = Some(index);
            if let Some((screen_pos, screen_size)) = best_bounds
                && position_is_reliable_linux(position, (screen_pos.x, screen_pos.y))
                && let Some(rect) = rect_from_bounds(
                    (position.x, position.y),
                    (size.x, size.y),
                    (screen_pos.x, screen_pos.y),
                    (screen_size.x, screen_size.y),
                )
            {
                self.descriptor.screen_rect = Some(rect);
            }
        }
    }

    pub fn update_descriptor_from_winit_window(
        &mut self,
        window: &WinitWindow,
        screens: &[MonitorHandle],
    ) -> bool {
        let current_monitor = window.current_monitor();
        let outer_position = window.outer_position().ok();
        let outer_size = window.outer_size();
        let mut updated = false;

        let screen_index = current_monitor
            .as_ref()
            .and_then(|current| screens.iter().position(|screen| screen == current))
            .or_else(|| {
                outer_position
                    .as_ref()
                    .and_then(|position| screen_index_from_bounds(*position, outer_size, screens))
            });

        if let Some(index) = screen_index {
            self.descriptor.screen = Some(index);
            updated = true;
        }

        #[cfg(target_os = "macos")]
        {
            if let (Some(position), Some(screen_handle)) = (
                outer_position,
                screen_index
                    .and_then(|idx| screens.get(idx).cloned())
                    .or_else(|| current_monitor.clone()),
            ) {
                let scale = screen_handle.scale_factor().max(0.0001);
                let screen_pos = screen_handle.position();
                let screen_size = screen_handle.size();

                // Convert window bounds to logical space to stay consistent with placement.
                let window_pos_logical = (
                    (position.x as f64 / scale).round() as i32,
                    (position.y as f64 / scale).round() as i32,
                );
                let window_size_logical = (
                    (outer_size.width as f64 / scale).round() as u32,
                    (outer_size.height as f64 / scale).round() as u32,
                );
                let screen_pos_logical = (screen_pos.x, screen_pos.y);
                let screen_size_logical = (
                    (screen_size.width as f64 / scale).round() as u32,
                    (screen_size.height as f64 / scale).round() as u32,
                );

                if let Some(rect) = rect_from_bounds(
                    window_pos_logical,
                    window_size_logical,
                    screen_pos_logical,
                    screen_size_logical,
                ) {
                    self.descriptor.screen_rect = Some(rect);
                    updated = true;
                }
            }
            return updated;
        }

        #[cfg(not(target_os = "macos"))]
        {
            if let (Some(position), Some(screen_handle)) = (
                outer_position,
                screen_index
                    .and_then(|idx| screens.get(idx).cloned())
                    .or_else(|| current_monitor.clone()),
            ) {
                let screen_pos = screen_handle.position();
                if position_is_reliable_linux(
                    IVec2::new(position.x, position.y),
                    (screen_pos.x, screen_pos.y),
                ) && let Some(rect) = rect_from_bounds(
                    (position.x, position.y),
                    (outer_size.width, outer_size.height),
                    (screen_pos.x, screen_pos.y),
                    (screen_handle.size().width, screen_handle.size().height),
                ) {
                    self.descriptor.screen_rect = Some(rect);
                    updated = true;
                }
            }
            return updated;
        }

        #[allow(unreachable_code)]
        {
            updated
        }
    }
}

pub fn set_mode_all(mode: &str, windows_state: &mut Query<&mut WindowState>) {
    for mut state in windows_state.iter_mut() {
        state.descriptor.mode = Some(mode.to_string());
    }
}

pub(crate) fn screen_index_from_bounds(
    position: PhysicalPosition<i32>,
    size: PhysicalSize<u32>,
    screens: &[MonitorHandle],
) -> Option<usize> {
    let center_x = position.x + size.width as i32 / 2;
    let center_y = position.y + size.height as i32 / 2;

    screens
        .iter()
        .enumerate()
        .filter_map(|(index, screen)| {
            let screen_pos = screen.position();
            let screen_size = screen.size();
            let min_x = screen_pos.x;
            let max_x = screen_pos.x + screen_size.width as i32;
            let min_y = screen_pos.y;
            let max_y = screen_pos.y + screen_size.height as i32;

            if center_x >= min_x && center_x < max_x && center_y >= min_y && center_y < max_y {
                let distance = (center_x - screen_pos.x).abs() + (center_y - screen_pos.y).abs();
                Some((index, distance))
            } else {
                None
            }
        })
        .min_by_key(|(_, distance)| *distance)
        .map(|(index, _)| index)
}

pub(crate) fn rect_from_bounds(
    position: (i32, i32),
    size: (u32, u32),
    screen_position: (i32, i32),
    screen_size: (u32, u32),
) -> Option<WindowRect> {
    if screen_size.0 == 0 || screen_size.1 == 0 {
        return None;
    }

    let width_pct = (size.0 as f32 / screen_size.0 as f32) * 100.0;
    let height_pct = (size.1 as f32 / screen_size.1 as f32) * 100.0;

    let offset_x = position.0 - screen_position.0;
    let offset_y = position.1 - screen_position.1;

    let x_pct = (offset_x as f32 / screen_size.0 as f32) * 100.0;
    let y_pct = (offset_y as f32 / screen_size.1 as f32) * 100.0;

    Some(WindowRect {
        x: clamp_percent(x_pct),
        y: clamp_percent(y_pct),
        width: clamp_percent(width_pct),
        height: clamp_percent(height_pct),
    })
}

fn position_is_reliable_linux(position: IVec2, screen_position: (i32, i32)) -> bool {
    if !cfg!(target_os = "linux") {
        return true;
    }
    if position.x == 0 && position.y == 0 {
        screen_position.0 == 0 && screen_position.1 == 0
    } else {
        true
    }
}

pub(crate) fn clamp_percent(value: f32) -> u32 {
    value.round().clamp(0.0, 100.0) as u32
}

/// Creates a `WindowState`. Must be spawned with a `WindowId` to create an
/// actual window.
pub fn create_secondary_window(title: Option<String>) -> (WindowState, WindowId) {
    let id = WindowId::default();
    let cleaned_title = title.and_then(|t| {
        let trimmed = t.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    });
    let path = cleaned_title
        .as_deref()
        .map(|title| {
            let stem = super::schematic::sanitize_to_stem(title);
            if stem.is_empty() {
                format!("secondary-window-{}.kdl", id.0)
            } else {
                format!("{stem}.kdl")
            }
        })
        .unwrap_or_else(|| format!("secondary-window-{}.kdl", id.0));
    let descriptor = WindowDescriptor {
        path: Some(PathBuf::from(path)),
        title: cleaned_title.or_else(|| Some(format!("Window {}", id.0 + 1))),
        screen: None,
        mode: None,
        screen_rect: None,
    };
    let tile_state = TileState::new(Id::new(("secondary_tab_tree", id.0)));
    (
        WindowState {
            descriptor,
            tile_state,
            graph_entities: Vec::new(),
            ui_state: WindowUiState::default(),
        },
        id,
    )
}

#[derive(Resource, Default)]
pub struct ViewportContainsPointer(pub bool);
#[derive(Clone)]
pub struct ActionTilePane {
    pub entity: Entity,
    pub name: PaneName,
}

#[derive(Clone)]
pub struct TreePane {
    pub entity: Entity,
    pub name: PaneName,
}

#[derive(Clone)]
pub struct DashboardPane {
    pub entity: Entity,
    pub name: PaneName,
}

impl TileState {
    fn has_non_sidebar_content(&self) -> bool {
        fn visit(tree: &egui_tiles::Tree<Pane>, id: TileId) -> bool {
            match tree.tiles.get(id) {
                Some(Tile::Pane(pane)) if pane.is_sidebar() => false,
                Some(Tile::Pane(_)) => true,
                Some(Tile::Container(container)) => {
                    container.children().any(|child| visit(tree, *child))
                }
                None => false,
            }
        }

        self.tree
            .root()
            .map(|root| visit(&self.tree, root))
            .unwrap_or(false)
    }

    pub fn insert_tile(
        &mut self,
        tile: Tile<Pane>,
        parent_id: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        // Check if we're inserting a pane (need nested tabs handling) or a container
        let is_pane = matches!(tile, Tile::Pane(_));

        let parent_id = if let Some(id) = parent_id {
            id
        } else {
            // If there's no root, create a proper tabs container as root
            let root_id = match self.tree.root() {
                Some(id) => id,
                None => {
                    let tabs_container = Tile::Container(Container::new_tabs(vec![]));
                    let tabs_id = self.tree.tiles.insert_new(tabs_container);
                    self.tree.root = Some(tabs_id);
                    tabs_id
                }
            };

            // Check if the root is a container we can add to
            match self.tree.tiles.get(root_id) {
                Some(Tile::Container(Container::Linear(linear))) => {
                    // For Linear containers, find the center child
                    if let Some(center) = linear.children.get(linear.children.len() / 2) {
                        *center
                    } else {
                        root_id
                    }
                }
                Some(Tile::Container(Container::Tabs(tabs))) if is_pane => {
                    // Root is a Tabs container and we're inserting a PANE.
                    // egui_tiles only shows tab bars for NESTED Tabs containers, not the root.
                    // So we need to find or create a nested Tabs container to add the pane to.
                    //
                    // Look for an existing nested Tabs container among the children.
                    let nested_tabs_id = tabs.children.iter().find(|&&child_id| {
                        matches!(
                            self.tree.tiles.get(child_id),
                            Some(Tile::Container(Container::Tabs(_)))
                        )
                    });

                    if let Some(&nested_id) = nested_tabs_id {
                        // Found a nested Tabs container, use it
                        nested_id
                    } else if tabs.children.is_empty() {
                        // Root tabs is empty, create a nested tabs container
                        let nested_tabs = Tile::Container(Container::new_tabs(vec![]));
                        let nested_id = self.tree.tiles.insert_new(nested_tabs);
                        // Add the nested tabs to the root
                        if let Some(Tile::Container(Container::Tabs(root_tabs))) =
                            self.tree.tiles.get_mut(root_id)
                        {
                            root_tabs.add_child(nested_id);
                        }
                        nested_id
                    } else {
                        // Root tabs has children but none are Tabs containers.
                        // Wrap all existing children in a new nested Tabs container.
                        let existing_children: Vec<_> = tabs.children.clone();
                        let nested_tabs =
                            Tile::Container(Container::new_tabs(existing_children.clone()));
                        let nested_id = self.tree.tiles.insert_new(nested_tabs);
                        // Replace root's children with just the nested container
                        if let Some(Tile::Container(Container::Tabs(root_tabs))) =
                            self.tree.tiles.get_mut(root_id)
                        {
                            root_tabs.children.clear();
                            root_tabs.add_child(nested_id);
                        }
                        nested_id
                    }
                }
                Some(Tile::Container(_)) => {
                    // Root is a container (Tabs with Container being inserted, or Grid), use it directly
                    root_id
                }
                Some(Tile::Pane(_)) => {
                    // Root is a pane, not a container! This can happen if simplification
                    // pruned away the tabs container. We need to wrap it in a new tabs container.
                    let tabs_container = Tile::Container(Container::new_tabs(vec![root_id]));
                    let tabs_id = self.tree.tiles.insert_new(tabs_container);
                    self.tree.root = Some(tabs_id);
                    tabs_id
                }
                None => {
                    // Root tile doesn't exist, create a new tabs container
                    let tabs_container = Tile::Container(Container::new_tabs(vec![]));
                    let tabs_id = self.tree.tiles.insert_new(tabs_container);
                    self.tree.root = Some(tabs_id);
                    tabs_id
                }
            }
        };

        let tile_id = self.tree.tiles.insert_new(tile);
        let parent_tile = self.tree.tiles.get_mut(parent_id)?;

        let Tile::Container(container) = parent_tile else {
            return None;
        };

        container.add_child(tile_id);

        if active && let Container::Tabs(tabs) = container {
            tabs.set_active(tile_id);
        }

        Some(tile_id)
    }

    pub fn create_graph_tile(&mut self, parent_id: Option<TileId>, graph_state: GraphBundle) {
        self.tree_actions
            .push(TreeAction::AddGraph(parent_id, Box::new(Some(graph_state))));
    }

    pub fn create_graph_tile_empty(&mut self) {
        self.tree_actions
            .push(TreeAction::AddGraph(None, Box::new(None)));
    }

    pub fn create_viewport_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddViewport(tile_id));
    }

    pub fn create_viewport_tile_empty(&mut self) {
        self.tree_actions.push(TreeAction::AddViewport(None));
    }

    pub fn create_monitor_tile(&mut self, eql: String, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddMonitor(tile_id, eql));
    }

    pub fn create_action_tile(
        &mut self,
        button_name: String,
        lua_code: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddActionTile(tile_id, button_name, lua_code));
    }

    pub fn create_query_table_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddQueryTable(tile_id));
    }

    pub fn create_query_plot_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddQueryPlot(tile_id));
    }

    pub fn create_video_stream_tile(
        &mut self,
        msg_id: [u8; 2],
        name: PaneName,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddVideoStream(tile_id, msg_id, name));
    }

    pub fn create_dashboard_tile(
        &mut self,
        dashboard: impeller2_wkt::Dashboard,
        name: PaneName,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddDashboard(tile_id, Box::new(dashboard), name));
    }

    pub fn create_tree_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions
            .push(TreeAction::AddSchematicTree(tile_id));
    }

    pub fn create_data_overview_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddDataOverview(tile_id));
    }

    pub fn debug_dump(&self) -> String {
        fn visit(
            tree: &egui_tiles::Tree<Pane>,
            tile_id: egui_tiles::TileId,
            depth: usize,
            out: &mut String,
        ) {
            let indent = "  ".repeat(depth);
            if let Some(tile) = tree.tiles.get(tile_id) {
                match tile {
                    Tile::Container(container) => {
                        let _ = writeln!(out, "{}Container::{:?}", indent, container.kind());
                        for child in container.children() {
                            visit(tree, *child, depth + 1, out);
                        }
                    }
                    Tile::Pane(pane) => {
                        let (kind, label): (&str, &str) = match pane {
                            Pane::Viewport(viewport) => ("Viewport", viewport.name.as_str()),
                            Pane::Graph(graph) => ("Graph", graph.name.as_str()),
                            Pane::Monitor(monitor) => ("Monitor", monitor.name.as_str()),
                            Pane::QueryTable(table) => ("QueryTable", table.name.as_str()),
                            Pane::QueryPlot(_) => ("QueryPlot", "QueryPlot"),
                            Pane::ActionTile(action) => ("Action", action.name.as_str()),
                            Pane::VideoStream(video) => ("VideoStream", video.name.as_str()),
                            Pane::Dashboard(dashboard) => ("Dashboard", dashboard.name.as_str()),
                            Pane::Hierarchy => ("Hierarchy", "Hierarchy"),
                            Pane::Inspector => ("Inspector", "Inspector"),
                            Pane::SchematicTree(pane) => ("SchematicTree", pane.name.as_str()),
                            Pane::DataOverview(pane) => ("DataOverview", pane.name.as_str()),
                        };
                        let _ = writeln!(out, "{}Pane::{} ({})", indent, kind, label);
                    }
                }
            }
        }

        let mut out = String::new();
        if let Some(root) = self.tree.root() {
            visit(&self.tree, root, 0, &mut out);
        } else {
            let _ = writeln!(out, "<empty>");
        }
        out
    }

    pub fn collect_graph_entities(&self) -> Vec<Entity> {
        self.collect_render_targets().cameras
    }

    pub fn collect_render_targets(&self) -> PaneRenderTargets {
        fn visit(
            tree: &egui_tiles::Tree<Pane>,
            tile_id: egui_tiles::TileId,
            out: &mut PaneRenderTargets,
        ) {
            let Some(tile) = tree.tiles.get(tile_id) else {
                return;
            };
            match tile {
                Tile::Pane(pane) => pane.collect_render_targets(out),
                Tile::Container(container) => {
                    for child in container.children() {
                        visit(tree, *child, out);
                    }
                }
            }
        }

        let mut targets = PaneRenderTargets::default();
        if let Some(root) = self.tree.root() {
            visit(&self.tree, root, &mut targets);
        }
        targets
    }

    pub fn create_sidebars_layout(&mut self) {
        self.tree_actions.push(TreeAction::AddSidebars);
    }

    fn apply_sidebars_layout(&mut self) {
        let mut has_hierarchy = false;
        let mut has_inspector = false;
        let mut sidebar_ids = Vec::new();
        for (id, tile) in self.tree.tiles.iter() {
            match tile {
                Tile::Pane(Pane::Hierarchy) => {
                    has_hierarchy = true;
                    sidebar_ids.push(*id);
                }
                Tile::Pane(Pane::Inspector) => {
                    has_inspector = true;
                    sidebar_ids.push(*id);
                }
                _ => {}
            }
        }

        if has_hierarchy && has_inspector {
            return;
        }

        if !sidebar_ids.is_empty() {
            let container_ids: Vec<TileId> = self
                .tree
                .tiles
                .iter()
                .filter_map(|(id, tile)| matches!(tile, Tile::Container(_)).then_some(*id))
                .collect();

            for container_id in container_ids {
                if let Some(Tile::Container(container)) = self.tree.tiles.get_mut(container_id) {
                    match container {
                        Container::Tabs(tabs) => {
                            tabs.children.retain(|child| !sidebar_ids.contains(child));
                            if let Some(active) = tabs.active
                                && !tabs.children.contains(&active)
                            {
                                tabs.active = tabs.children.first().copied();
                            }
                        }
                        Container::Linear(linear) => {
                            linear.children.retain(|child| !sidebar_ids.contains(child));
                        }
                        Container::Grid(_) => {}
                    }
                }
            }
        }

        let root_id = self
            .tree
            .root()
            .and_then(|root_id| match self.tree.tiles.get(root_id) {
                Some(Tile::Pane(pane)) if pane.is_sidebar() => None,
                _ => Some(root_id),
            });

        let mut main_content = root_id.unwrap_or_else(|| {
            let tabs_container = Tile::Container(Container::new_tabs(vec![]));
            let tabs_id = self.tree.tiles.insert_new(tabs_container);
            self.tree.root = Some(tabs_id);
            tabs_id
        });

        if matches!(self.tree.tiles.get(main_content), Some(Tile::Pane(_))) {
            let mut tabs = egui_tiles::Tabs::new(vec![main_content]);
            tabs.set_active(main_content);
            main_content = self
                .tree
                .tiles
                .insert_new(Tile::Container(Container::Tabs(tabs)));
        }

        let hierarchy = self.tree.tiles.insert_new(Tile::Pane(Pane::Hierarchy));
        let inspector = self.tree.tiles.insert_new(Tile::Pane(Pane::Inspector));

        let hier_default = self.sidebar_state.last_hierarchy_share.get_or_insert(0.2);
        let insp_default = self.sidebar_state.last_inspector_share.get_or_insert(0.2);
        let hier_share = if self.sidebar_state.hierarchy_masked {
            SIDEBAR_COLLAPSED_SHARE
        } else {
            *hier_default
        };
        let insp_share = if self.sidebar_state.inspector_masked {
            SIDEBAR_COLLAPSED_SHARE
        } else {
            *insp_default
        };
        let mut center_share = 1.0 - (hier_share + insp_share);
        if center_share <= 0.0 {
            center_share = 0.1;
        }

        let mut linear = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![hierarchy, main_content, inspector],
        );
        linear.shares.set_share(hierarchy, hier_share);
        linear.shares.set_share(main_content, center_share);
        linear.shares.set_share(inspector, insp_share);

        let root = self
            .tree
            .tiles
            .insert_new(Tile::Container(Container::Linear(linear)));
        self.tree.root = Some(root);
    }

    pub fn is_empty(&self) -> bool {
        !self.has_non_sidebar_content()
    }

    pub fn clear(&mut self, commands: &mut Commands, render_layer_alloc: &mut RenderLayerAlloc) {
        for (tile_id, tile) in self.tree.tiles.iter() {
            match tile {
                Tile::Pane(Pane::Viewport(viewport)) => {
                    if let Some(layer) = viewport.viewport_layer {
                        render_layer_alloc.free(layer);
                    }
                    if let Some(layer) = viewport.grid_layer {
                        render_layer_alloc.free(layer);
                    }
                    if let Some(camera) = viewport.camera {
                        commands.entity(camera).despawn();
                    }
                    if let Some(nav_gizmo_camera) = viewport.nav_gizmo_camera {
                        commands.entity(nav_gizmo_camera).despawn();
                    }
                    if let Some(nav_gizmo) = viewport.nav_gizmo {
                        commands.entity(nav_gizmo).despawn();
                    }
                }
                Tile::Pane(Pane::Graph(graph)) => {
                    commands.entity(graph.id).despawn();
                    if let Some(graph_id) = self.graphs.get(tile_id) {
                        commands.entity(*graph_id).despawn();
                        self.graphs.remove(tile_id);
                    }
                }

                Tile::Pane(Pane::VideoStream(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::ActionTile(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::QueryTable(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::QueryPlot(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::SchematicTree(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::Dashboard(dashboard)) => {
                    commands.entity(dashboard.entity).despawn();
                }
                _ => {}
            }
        }

        if let Some(root_id) = self.tree.root()
            && let Some(Tile::Container(root)) = self.tree.tiles.get_mut(root_id)
        {
            root.retain(|_| false);
        };
        self.graphs.clear();
        self.container_titles.clear();
        self.reset_tree();
    }

    pub fn get_container_title(&self, id: TileId) -> Option<&str> {
        self.container_titles.get(&id).map(|s| s.as_str())
    }

    pub fn set_container_title(&mut self, id: TileId, title: impl Into<String>) {
        let title = title.into();
        self.container_titles.insert(id, title);
    }

    pub fn clear_container_title(&mut self, id: TileId) {
        self.container_titles.remove(&id);
    }

    pub fn container_title_or_default(
        &self,
        tiles: &egui_tiles::Tiles<Pane>,
        id: TileId,
    ) -> String {
        if let Some(t) = self.get_container_title(id) {
            return t.to_owned();
        }
        if let Some(egui_tiles::Tile::Container(c)) = tiles.get(id) {
            format!("{:?}", c.kind())
        } else {
            "Container".to_owned()
        }
    }
}

#[derive(Default)]
pub struct PaneRenderTargets {
    pub cameras: Vec<Entity>,
    pub ui_nodes: Vec<Entity>,
}

impl PaneRenderTargets {
    fn push_camera(&mut self, entity: Entity) {
        self.cameras.push(entity);
    }

    fn push_ui_node(&mut self, entity: Entity) {
        self.ui_nodes.push(entity);
    }
}

#[derive(Clone)]
pub enum Pane {
    Viewport(ViewportPane),
    Graph(GraphPane),
    Monitor(MonitorPane),
    QueryTable(QueryTablePane),
    QueryPlot(super::query_plot::QueryPlotPane),
    ActionTile(ActionTilePane),
    VideoStream(super::video_stream::VideoStreamPane),
    Dashboard(DashboardPane),
    Hierarchy,
    Inspector,
    SchematicTree(TreePane),
    DataOverview(DataOverviewPane),
}

impl Pane {
    pub(crate) fn is_sidebar(&self) -> bool {
        matches!(self, Pane::Hierarchy | Pane::Inspector)
    }

    pub(crate) fn sidebar_kind(&self) -> Option<SidebarKind> {
        match self {
            Pane::Hierarchy => Some(SidebarKind::Hierarchy),
            Pane::Inspector => Some(SidebarKind::Inspector),
            _ => None,
        }
    }

    fn collect_render_targets(&self, out: &mut PaneRenderTargets) {
        match self {
            Pane::Graph(pane) => out.push_camera(pane.id),
            Pane::QueryPlot(pane) => out.push_camera(pane.entity),
            Pane::Viewport(pane) => {
                if let Some(cam) = pane.camera {
                    out.push_camera(cam);
                }
                if let Some(nav_cam) = pane.nav_gizmo_camera {
                    out.push_camera(nav_cam);
                }
            }
            Pane::VideoStream(pane) => out.push_ui_node(pane.entity),
            Pane::Dashboard(pane) => out.push_ui_node(pane.entity),
            Pane::Monitor(_)
            | Pane::QueryTable(_)
            | Pane::ActionTile(_)
            | Pane::Hierarchy
            | Pane::Inspector
            | Pane::SchematicTree(_)
            | Pane::DataOverview(_) => {}
        }
    }

    fn title(
        &self,
        graph_states: &Query<&GraphState>,
        dashboards: &Query<&Dashboard<Entity>>,
    ) -> String {
        match self {
            Pane::Graph(pane) => {
                if let Ok(graph_state) = graph_states.get(pane.id) {
                    return graph_state.label.to_string();
                }
                pane.name.to_string()
            }
            Pane::Viewport(viewport) => viewport.name.to_string(),
            Pane::Monitor(monitor) => monitor.name.to_string(),
            Pane::QueryTable(table) => table.name.to_string(),
            Pane::QueryPlot(query_plot) => {
                if let Ok(graph_state) = graph_states.get(query_plot.entity) {
                    return graph_state.label.to_string();
                }
                "Query Plot".to_string()
            }
            Pane::ActionTile(action) => action.name.to_string(),
            Pane::VideoStream(video_stream) => video_stream.name.to_string(),
            Pane::Dashboard(dashboard) => {
                if let Ok(dash) = dashboards.get(dashboard.entity) {
                    return dash.root.name.as_deref().unwrap_or("Dashboard").to_string();
                }
                "Dashboard".to_string()
            }
            Pane::Hierarchy => "Entities".to_string(),
            Pane::Inspector => "Inspector".to_string(),
            Pane::SchematicTree(pane) => pane.name.to_string(),
            Pane::DataOverview(pane) => pane.name.to_string(),
        }
    }

    fn set_title(&mut self, title: &str) -> PaneTitleTargets {
        let mut targets = PaneTitleTargets::default();
        match self {
            Pane::Viewport(viewport) => {
                viewport.name = title.to_string();
            }
            Pane::Graph(graph) => {
                graph.name = title.to_string();
                targets.graph_id = Some(graph.id);
            }
            Pane::Monitor(monitor) => {
                monitor.name = title.to_string();
            }
            Pane::QueryTable(table) => {
                table.name = title.to_string();
                targets.query_table_id = Some(table.entity);
            }
            Pane::QueryPlot(plot) => {
                targets.graph_id = Some(plot.entity);
                targets.query_plot_id = Some(plot.entity);
            }
            Pane::ActionTile(action) => {
                action.name = title.to_string();
                targets.action_tile_id = Some(action.entity);
            }
            Pane::VideoStream(video) => {
                video.name = title.to_string();
            }
            Pane::Dashboard(dashboard) => {
                dashboard.name = title.to_string();
                targets.dashboard_id = Some(dashboard.entity);
            }
            Pane::SchematicTree(tree) => {
                tree.name = title.to_string();
            }
            Pane::DataOverview(pane) => {
                pane.name = title.to_string();
            }
            Pane::Hierarchy | Pane::Inspector => {}
        }
        targets
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        ui: &mut Ui,
        icons: &TileIcons,
        world: &mut World,
        tree_actions: &mut SmallVec<[TreeAction; 4]>,
        target_window: Entity,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                pane.rect = Some(content_rect);

                ui.add_widget_with::<PlotWidget>(
                    world,
                    "graph",
                    (pane.id, icons.scrub, target_window),
                );

                egui_tiles::UiResponse::None
            }
            Pane::Viewport(pane) => {
                pane.rect = Some(content_rect);
                egui_tiles::UiResponse::None
            }
            Pane::Monitor(pane) => {
                ui.add_widget_with::<MonitorWidget>(world, "monitor", pane.clone());
                egui_tiles::UiResponse::None
            }
            Pane::QueryTable(pane) => {
                ui.add_widget_with::<QueryTableWidget>(world, "sql", pane.clone());
                egui_tiles::UiResponse::None
            }
            Pane::QueryPlot(pane) => {
                pane.rect = Some(content_rect);
                let mut pane_with_icon = pane.clone();
                pane_with_icon.scrub_icon = Some(icons.scrub);
                ui.add_widget_with::<super::query_plot::QueryPlotWidget>(
                    world,
                    "query_plot",
                    (pane_with_icon, target_window),
                );
                egui_tiles::UiResponse::None
            }
            Pane::ActionTile(pane) => {
                ui.add_widget_with::<ActionTileWidget>(world, "action_tile", pane.entity);
                egui_tiles::UiResponse::None
            }
            Pane::VideoStream(pane) => {
                ui.add_widget_with::<super::video_stream::VideoStreamWidget>(
                    world,
                    "video_stream",
                    VideoStreamWidgetArgs {
                        entity: pane.entity,
                        window: target_window,
                    },
                );
                egui_tiles::UiResponse::None
            }
            Pane::Dashboard(pane) => {
                ui.add_widget_with::<DashboardWidget>(
                    world,
                    "dashboard",
                    DashboardWidgetArgs {
                        entity: pane.entity,
                        window: target_window,
                    },
                );
                egui_tiles::UiResponse::None
            }
            Pane::Hierarchy => {
                ui.add_widget_with::<HierarchyContent>(
                    world,
                    "hierarchy_content",
                    (
                        Hierarchy {
                            search: icons.search,
                            entity: icons.entity,
                            chevron: icons.chevron,
                        },
                        target_window,
                    ),
                );
                egui_tiles::UiResponse::None
            }
            Pane::Inspector => {
                let inspector_icons = InspectorIcons {
                    chart: icons.chart,
                    subtract: icons.subtract,
                    setting: icons.setting,
                    search: icons.search,
                };
                let actions = ui.add_widget_with::<InspectorContent>(
                    world,
                    "inspector_content",
                    (inspector_icons, true, target_window),
                );
                tree_actions.extend(actions);
                egui_tiles::UiResponse::None
            }
            Pane::SchematicTree(tree_pane) => {
                let tree_icons = super::schematic::tree::TreeIcons {
                    chevron: icons.chevron,
                    search: icons.search,
                    container: icons.container,
                    plot: icons.plot,
                    viewport: icons.viewport,
                    add: icons.add,
                };
                ui.add_widget_with::<super::schematic::tree::TreeWidget>(
                    world,
                    "tree",
                    (tree_icons, tree_pane.entity, target_window),
                );
                egui_tiles::UiResponse::None
            }
            Pane::DataOverview(pane) => {
                let updated_pane =
                    ui.add_widget_with::<DataOverviewWidget>(world, "data_overview", pane.clone());
                *pane = updated_pane;
                egui_tiles::UiResponse::None
            }
        }
    }
}

#[derive(Default)]
struct PaneTitleTargets {
    graph_id: Option<Entity>,
    query_plot_id: Option<Entity>,
    query_table_id: Option<Entity>,
    action_tile_id: Option<Entity>,
    dashboard_id: Option<Entity>,
}

fn apply_pane_title_updates(
    title: &str,
    targets: PaneTitleTargets,
    graph_states: &mut Query<'_, '_, &'static mut GraphState>,
    query_plots: &mut Query<'_, '_, &'static mut QueryPlotData>,
    query_tables: &mut Query<'_, '_, &'static mut QueryTableData>,
    action_tiles: &mut Query<'_, '_, &'static mut ActionTile>,
    dashboards: &mut Query<'_, '_, &'static mut Dashboard<Entity>>,
) {
    let title = title.to_string();

    if let Some(graph_id) = targets.graph_id
        && let Ok(mut graph_state) = graph_states.get_mut(graph_id)
    {
        graph_state.label = title.clone();
    }

    if let Some(query_plot_id) = targets.query_plot_id
        && let Ok(mut plot_data) = query_plots.get_mut(query_plot_id)
    {
        plot_data.data.name = title.clone();
    }

    if let Some(query_table_id) = targets.query_table_id
        && let Ok(mut table) = query_tables.get_mut(query_table_id)
    {
        table.data.name = Some(title.clone());
    }

    if let Some(action_tile_id) = targets.action_tile_id
        && let Ok(mut action_tile) = action_tiles.get_mut(action_tile_id)
    {
        action_tile.button_name = title.clone();
    }

    if let Some(dashboard_id) = targets.dashboard_id
        && let Ok(mut dashboard) = dashboards.get_mut(dashboard_id)
    {
        dashboard.root.name = Some(title);
    }
}

#[derive(Default, Clone)]
pub struct ViewportPane {
    pub camera: Option<Entity>,
    pub nav_gizmo: Option<Entity>,
    pub nav_gizmo_camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
    pub name: PaneName,
    pub grid_layer: Option<usize>,
    pub viewport_layer: Option<usize>,
}

impl ViewportPane {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        commands: &mut Commands,
        asset_server: &Res<AssetServer>,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
        eql_ctx: &eql::Context,
        viewport: &Viewport,
        name: PaneName,
    ) -> Self {
        let mut main_camera_layers = RenderLayers::default().with(GIZMO_RENDER_LAYER);
        let mut grid_layers = RenderLayers::none();
        let grid_layer = render_layer_alloc.alloc();
        if let Some(layer) = grid_layer {
            main_camera_layers = main_camera_layers.with(layer);
            grid_layers = grid_layers.with(layer);
        }
        let viewport_layer = render_layer_alloc.alloc();
        if let Some(layer) = viewport_layer {
            main_camera_layers = main_camera_layers.with(layer);
        }

        let grid_visibility = if viewport.show_grid {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let grid_id = commands
            .spawn((
                bevy_infinite_grid::InfiniteGridBundle {
                    settings: bevy_infinite_grid::InfiniteGridSettings {
                        minor_line_color: Color::srgba(1.0, 1.0, 1.0, 0.02),
                        major_line_color: Color::srgba(1.0, 1.0, 1.0, 0.05),
                        z_axis_color: crate::ui::colors::bevy::GREEN,
                        x_axis_color: crate::ui::colors::bevy::RED,
                        fadeout_distance: 50_000.0,
                        scale: 0.1,
                        ..Default::default()
                    },
                    visibility: grid_visibility,
                    ..Default::default()
                },
                grid_layers,
            ))
            .id();

        let parent = commands
            .spawn((
                GlobalTransform::default(),
                Transform::from_translation(Vec3::new(5.0, 5.0, 10.0))
                    .looking_at(Vec3::ZERO, Vec3::Y),
                impeller2_wkt::WorldPos::default(),
                Name::new("viewport"),
            ))
            .id();
        let pos = viewport
            .pos
            .as_ref()
            .map(|eql| {
                let compiled_expr = match eql_ctx.parse_str(eql) {
                    Ok(expr) => Some(compile_eql_expr(expr)),
                    Err(e) => {
                        bevy::log::error!(
                            "Failed to parse viewport pos expression '{}': {}",
                            eql,
                            e
                        );
                        None
                    }
                };
                EditableEQL {
                    eql: eql.to_string(),
                    compiled_expr,
                }
            })
            .unwrap_or_default();
        let look_at = viewport
            .look_at
            .as_ref()
            .map(|eql| {
                let compiled_expr = match eql_ctx.parse_str(eql) {
                    Ok(expr) => Some(compile_eql_expr(expr)),
                    Err(e) => {
                        bevy::log::error!(
                            "Failed to parse viewport look_at expression '{}': {}",
                            eql,
                            e
                        );
                        None
                    }
                };
                EditableEQL {
                    eql: eql.to_string(),
                    compiled_expr,
                }
            })
            .unwrap_or_default();

        let mut camera = commands.spawn((
            Transform::default(),
            Camera3d::default(),
            Camera {
                order: 1,
                ..Default::default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: viewport.fov.to_radians(),
                ..Default::default()
            }),
            Tonemapping::TonyMcMapface,
            Exposure::from_physical_camera(PhysicalCameraParameters {
                aperture_f_stops: 2.8,
                shutter_speed_s: 1.0 / 200.0,
                sensitivity_iso: 400.0,
                sensor_height: 24.0 / 1000.0,
            }),
            main_camera_layers,
            MainCamera,
            big_space::GridCell::<i128>::default(),
            EditorCam {
                orbit_constraint: OrbitConstraint::Fixed {
                    up: Vec3::Y,
                    can_pass_tdc: false,
                },
                last_anchor_depth: 2.0,
                ..Default::default()
            },
            GridHandle { grid: grid_id },
            ViewportConfig {
                show_arrows: viewport.show_arrows,
                viewport_layer,
            },
            crate::ui::inspector::viewport::Viewport::new(parent, pos, look_at),
            ChildOf(parent),
            Name::new("viewport camera3d"),
        ));

        camera.insert(Bloom { ..default() });
        camera.insert(EnvironmentMapLight {
            diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
            specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
            intensity: 2000.0,
            ..Default::default()
        });

        let camera = camera.id();

        let (nav_gizmo, nav_gizmo_camera) =
            spawn_gizmo(camera, commands, meshes, materials, render_layer_alloc);
        Self {
            camera: Some(camera),
            nav_gizmo,
            nav_gizmo_camera,
            rect: None,
            name,
            grid_layer,
            viewport_layer,
        }
    }
}

#[derive(Clone)]
pub struct GraphPane {
    pub id: Entity,
    pub name: PaneName,
    pub rect: Option<egui::Rect>,
}

impl GraphPane {
    pub fn new(graph_id: Entity, name: PaneName) -> Self {
        Self {
            id: graph_id,
            name,
            rect: None,
        }
    }
}

impl TileState {
    pub fn new(tree_id: Id) -> Self {
        Self {
            tree: egui_tiles::Tree::new_tabs(tree_id, vec![]),
            tree_actions: smallvec![TreeAction::AddSidebars],
            graphs: HashMap::new(),
            container_titles: HashMap::new(),
            sidebar_state: SidebarMaskState::default(),
            tree_id,
        }
    }

    fn reset_tree(&mut self) {
        self.tree = egui_tiles::Tree::new_tabs(self.tree_id, vec![]);
        self.tree_actions = smallvec![TreeAction::AddSidebars];
        self.sidebar_state = SidebarMaskState::default();
    }
}

impl Default for TileState {
    fn default() -> Self {
        Self::new(Id::new("main_tab_tree"))
    }
}

fn main_content_rect(tree: &egui_tiles::Tree<Pane>) -> Option<egui::Rect> {
    let root = tree.root()?;
    if let Some(Tile::Container(Container::Linear(linear))) = tree.tiles.get(root) {
        for child in &linear.children {
            if !tile_is_sidebar(&tree.tiles, *child) {
                return tree.tiles.rect(*child);
            }
        }
    }
    tree.tiles.rect(root)
}

struct TreeBehavior<'w> {
    icons: TileIcons,
    tree_actions: SmallVec<[TreeAction; 4]>,
    world: &'w mut World,
    container_titles: HashMap<TileId, String>,
    read_only: bool,
    target_window: Entity,
}

type ShareUpdate = (TileId, TileId, TileId, f32, f32);

#[derive(Clone)]
pub enum TreeAction {
    AddViewport(Option<TileId>),
    AddGraph(Option<TileId>, Box<Option<GraphBundle>>),
    AddMonitor(Option<TileId>, PaneName),
    AddQueryTable(Option<TileId>),
    AddQueryPlot(Option<TileId>),
    AddActionTile(Option<TileId>, PaneName, String),
    AddVideoStream(Option<TileId>, [u8; 2], PaneName),
    AddDashboard(Option<TileId>, Box<impeller2_wkt::Dashboard>, PaneName),
    AddSchematicTree(Option<TileId>),
    AddDataOverview(Option<TileId>),
    AddSidebars,
    DeleteTab(TileId),
    SelectTile(TileId),
    RenameContainer(TileId, String),
    RenamePane(TileId, String),
}

enum TabState {
    Selected,
    Inactive,
}

impl egui_tiles::Behavior<Pane> for TreeBehavior<'_> {
    fn on_edit(&mut self, _edit_action: egui_tiles::EditAction) {}

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        let mut query =
            SystemState::<(Query<&GraphState>, Query<&Dashboard<Entity>>)>::new(self.world);
        let (graphs, dashes) = query.get(self.world);
        pane.title(&graphs, &dashes).into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        pane.ui(
            ui,
            &self.icons,
            self.world,
            &mut self.tree_actions,
            self.target_window,
        )
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn tab_ui(
        &mut self,
        tiles: &mut Tiles<Pane>,
        ui: &mut Ui,
        id: egui::Id,
        tile_id: egui_tiles::TileId,
        state: &egui_tiles::TabState,
    ) -> egui::Response {
        if !tab_title_visible(tiles, tile_id) {
            let min_width = self.tab_title_spacing(ui.visuals()) * 2.0;
            let (_, rect) = ui.allocate_space(vec2(min_width, ui.available_height()));
            let response = ui.interact(rect, id, egui::Sense::click_and_drag());
            return self.on_tab_button(tiles, tile_id, response);
        }

        let tab_state = if state.active {
            TabState::Selected
        } else {
            TabState::Inactive
        };

        let is_container = matches!(tiles.get(tile_id), Some(egui_tiles::Tile::Container(_)));
        let persist_id = id.with(("rename_title", tile_id));
        let edit_flag_id = id.with(("rename_editing", tile_id));
        let edit_buf_id = id.with(("rename_buffer", tile_id));
        let mut is_editing = ui
            .ctx()
            .data(|d| d.get_temp::<bool>(edit_flag_id))
            .unwrap_or(false);

        let title_str: String = if is_container {
            if let Some(custom) = ui.ctx().data(|d| d.get_temp::<String>(persist_id)) {
                custom
            } else if let Some(t) = self.container_titles.get(&tile_id) {
                t.clone()
            } else {
                match tiles.get(tile_id) {
                    Some(egui_tiles::Tile::Container(Container::Tabs(_))) => {
                        // Hide Tabs containers without custom name (wrapper Tabs)
                        let min_width = self.tab_title_spacing(ui.visuals()) * 2.0;
                        let (_, rect) = ui.allocate_space(vec2(min_width, ui.available_height()));
                        let response = ui.interact(rect, id, egui::Sense::click_and_drag());
                        return self.on_tab_button(tiles, tile_id, response);
                    }
                    Some(egui_tiles::Tile::Container(c)) => format!("{:?}", c.kind()),
                    _ => "Container".to_owned(),
                }
            }
        } else {
            self.tab_title_for_tile(tiles, tile_id).text().to_string()
        };

        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 11.0;
        let mut galley = egui::WidgetText::from(title_str.clone()).into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            font_id.clone(),
        );

        let x_margin = self.tab_title_spacing(ui.visuals());
        let (_, rect) = ui.allocate_space(vec2(
            galley.size().x + x_margin * 4.0,
            ui.available_height(),
        ));
        let text_rect = rect
            .shrink2(vec2(x_margin * 4.0, 0.0))
            .translate(vec2(-3.0 * x_margin, 0.0));
        let response = {
            let mut resp = ui.interact(rect, id, egui::Sense::click_and_drag());
            let drag_distance = ui.input(|i| {
                let press = i.pointer.press_origin();
                let latest = i.pointer.latest_pos();
                press
                    .zip(latest)
                    .map(|(p, l)| p.distance(l))
                    .unwrap_or_default()
            });
            const DRAG_SLOP: f32 = 12.0;
            if drag_distance < DRAG_SLOP {
                resp.flags
                    .remove(Flags::DRAG_STARTED | Flags::DRAGGED | Flags::DRAG_STOPPED);
            }
            resp
        };

        if !self.read_only && state.active && response.double_clicked() && !is_editing {
            ui.ctx()
                .data_mut(|d| d.insert_temp(edit_buf_id, title_str.clone()));
            ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, true));
            is_editing = true;
        }

        if ui.is_rect_visible(rect) && !state.is_being_dragged {
            let scheme = get_scheme();
            let bg_color = match tab_state {
                TabState::Selected => scheme.text_primary,
                TabState::Inactive => scheme.bg_secondary,
            };

            let text_color = match tab_state {
                TabState::Selected => scheme.bg_secondary,
                TabState::Inactive => with_opacity(scheme.text_primary, 0.6),
            };

            ui.painter().rect_filled(rect, 0.0, bg_color);

            if !self.read_only && is_editing {
                let label_rect =
                    egui::Align2::LEFT_CENTER.align_size_within_rect(galley.size(), text_rect);
                let edit_rect = label_rect.expand(1.0);

                let mut buf = ui.ctx().data_mut(|d| {
                    d.get_temp_mut_or::<String>(edit_buf_id, String::new())
                        .clone()
                });

                let resp = ui
                    .scope(|ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::BLACK);
                        ui.put(
                            edit_rect,
                            egui::TextEdit::singleline(&mut buf)
                                .font(egui::TextStyle::Button)
                                .clip_text(true)
                                .desired_width(edit_rect.width())
                                .frame(false),
                        )
                    })
                    .inner;

                let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                let lost_focus = resp.lost_focus();

                if enter_pressed || lost_focus {
                    let trimmed = buf.trim();
                    let new_title = if trimmed.is_empty() {
                        title_str.clone()
                    } else {
                        trimmed.to_owned()
                    };

                    ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, false));
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(edit_buf_id, new_title.clone()));
                    ui.memory_mut(|m| m.surrender_focus(resp.id));

                    if !self.read_only {
                        if is_container {
                            ui.ctx()
                                .data_mut(|d| d.insert_temp(persist_id, new_title.clone()));
                            self.tree_actions
                                .push(TreeAction::RenameContainer(tile_id, new_title.clone()));
                        } else {
                            self.tree_actions
                                .push(TreeAction::RenamePane(tile_id, new_title.clone()));
                        }
                    }

                    galley = egui::WidgetText::from(new_title).into_galley(
                        ui,
                        Some(egui::TextWrapMode::Extend),
                        f32::INFINITY,
                        font_id.clone(),
                    );

                    ui.painter().galley(
                        egui::Align2::LEFT_CENTER
                            .align_size_within_rect(galley.size(), text_rect)
                            .min,
                        galley.clone(),
                        text_color,
                    );
                } else {
                    ui.ctx().data_mut(|d| d.insert_temp(edit_buf_id, buf));
                    if !resp.has_focus() {
                        resp.request_focus();
                    }
                }
            } else {
                ui.painter().galley(
                    egui::Align2::LEFT_CENTER
                        .align_size_within_rect(galley.size(), text_rect)
                        .min,
                    galley.clone(),
                    text_color,
                );
            }

            ui.add_space(-3.0 * x_margin);
            let close_response = ui.add(
                EImageButton::new(self.icons.close)
                    .scale(1.3, 1.3)
                    .image_tint(match tab_state {
                        TabState::Inactive => scheme.text_primary,
                        TabState::Selected => scheme.bg_primary,
                    })
                    .bg_color(colors::TRANSPARENT)
                    .hovered_bg_color(colors::TRANSPARENT),
            );
            if close_response.clicked() {
                self.tree_actions.push(TreeAction::DeleteTab(tile_id));
            }

            ui.painter().hline(
                rect.x_range(),
                rect.bottom(),
                egui::Stroke::new(1.0, scheme.border_primary),
            );

            ui.painter().vline(
                rect.right(),
                rect.y_range(),
                egui::Stroke::new(1.0, scheme.border_primary),
            );
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn on_tab_button(
        &mut self,
        _tiles: &Tiles<Pane>,
        tile_id: TileId,
        button_response: egui::Response,
    ) -> egui::Response {
        if button_response.middle_clicked() && !self.read_only {
            self.tree_actions.push(TreeAction::DeleteTab(tile_id));
        } else if button_response.clicked() {
            self.tree_actions.push(TreeAction::SelectTile(tile_id));
        }
        button_response
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        32.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        get_scheme().bg_secondary
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            prune_empty_tabs: false,
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
            prune_single_child_tabs: false, // Keep tabs container even with single child
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, get_scheme().text_primary)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::with_opacity(get_scheme().text_primary, 0.6)
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = get_scheme().text_primary;
        frame.corner_radius = CornerRadius::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = match tiles.get(tile_id) {
                Some(Tile::Container(_)) => {
                    if let Some(t) = self.container_titles.get(&tile_id) {
                        egui::WidgetText::from(t.clone())
                    } else {
                        self.tab_title_for_tile(tiles, tile_id)
                    }
                }
                _ => self.tab_title_for_tile(tiles, tile_id),
            };
            let text = text.text();
            ui.label(
                RichText::new(text)
                    .color(get_scheme().bg_secondary)
                    .size(11.0),
            );
        });
    }

    fn top_bar_right_ui(
        &mut self,
        tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.read_only {
            return;
        }
        if !tab_add_visible(tiles, tabs) {
            return;
        }
        let mut layout = SystemState::<TileLayout>::new(self.world);
        let mut layout = layout.get_mut(self.world);

        let top_bar_rect = ui.available_rect_before_wrap();
        ui.painter().hline(
            top_bar_rect.x_range(),
            top_bar_rect.bottom(),
            egui::Stroke::new(1.0, get_scheme().border_primary),
        );

        ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        ui.style_mut().visuals.widgets.active.bg_stroke = Stroke::NONE;
        ui.add_space(5.0);
        let resp = ui.add(EImageButton::new(self.icons.add).scale(1.4, 1.4));
        if resp.clicked() {
            layout
                .cmd_palette_state
                .open_page_for_window(Some(self.target_window), move || {
                    palette_items::create_tiles(tile_id)
                });
        }
    }
}

#[derive(SystemParam)]
pub struct TileSystem<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    window_states: Query<'w, 's, (Entity, &'static WindowId, &'static WindowState)>,
    primary_window: Single<'w, 's, Entity, With<PrimaryWindow>>,
}

impl<'w, 's> TileSystem<'w, 's> {
    fn prepare_panel_data(
        world: &mut World,
        state: &mut SystemState<Self>,
        target: Option<Entity>,
    ) -> Option<(TileIcons, bool, bool)> {
        let read_only = false;
        let params = state.get_mut(world);
        let mut contexts = params.contexts;
        let images = params.images;
        let target_id = target.unwrap_or_else(|| *params.primary_window);
        let is_empty_tile_tree = params
            .window_states
            .get(target_id)
            .map(|(_, _, s)| {
                let pending_non_sidebar = s
                    .tile_state
                    .tree_actions
                    .iter()
                    .any(|action| !matches!(action, TreeAction::AddSidebars));
                !pending_non_sidebar && !s.tile_state.has_non_sidebar_content()
            })
            .ok()?;

        let icons = TileIcons {
            add: contexts.add_image(EguiTextureHandle::Weak(images.icon_add.id())),
            close: contexts.add_image(EguiTextureHandle::Weak(images.icon_close.id())),
            scrub: contexts.add_image(EguiTextureHandle::Weak(images.icon_scrub.id())),
            tile_3d_viewer: contexts
                .add_image(EguiTextureHandle::Weak(images.icon_tile_3d_viewer.id())),
            tile_graph: contexts.add_image(EguiTextureHandle::Weak(images.icon_tile_graph.id())),
            subtract: contexts.add_image(EguiTextureHandle::Weak(images.icon_subtract.id())),
            chart: contexts.add_image(EguiTextureHandle::Weak(images.icon_chart.id())),
            setting: contexts.add_image(EguiTextureHandle::Weak(images.icon_setting.id())),
            search: contexts.add_image(EguiTextureHandle::Weak(images.icon_search.id())),
            chevron: contexts.add_image(EguiTextureHandle::Weak(images.icon_chevron_right.id())),
            plot: contexts.add_image(EguiTextureHandle::Weak(images.icon_plot.id())),
            viewport: contexts.add_image(EguiTextureHandle::Weak(images.icon_viewport.id())),
            container: contexts.add_image(EguiTextureHandle::Weak(images.icon_container.id())),
            entity: contexts.add_image(EguiTextureHandle::Weak(images.icon_entity.id())),
        };

        Some((icons, is_empty_tile_tree, read_only))
    }

    fn render_panel_contents(
        world: &mut World,
        ui: &mut egui::Ui,
        target: Option<Entity>,
        icons: TileIcons,
        is_empty_tile_tree: bool,
        read_only: bool,
    ) {
        let show_empty_overlay = is_empty_tile_tree && !read_only;
        ui.add_widget_with::<TileLayout>(
            world,
            "tile_layout",
            TileLayoutArgs {
                icons,
                window: target,
                read_only,
                show_empty_overlay,
            },
        );
    }
}

impl WidgetSystem for TileSystem<'_, '_> {
    type Args = Option<Entity>;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        target: Self::Args,
    ) {
        let Some((icons, is_empty_tile_tree, read_only)) =
            Self::prepare_panel_data(world, state, target)
        else {
            return;
        };

        let fill_color = if is_empty_tile_tree {
            get_scheme().bg_secondary
        } else {
            colors::TRANSPARENT
        };

        egui::CentralPanel::default()
            .frame(Frame {
                fill: fill_color,
                ..Default::default()
            })
            .show_inside(ui, |ui| {
                Self::render_panel_contents(
                    world,
                    ui,
                    target,
                    icons.clone(),
                    is_empty_tile_tree,
                    read_only,
                );
            });
    }
}

impl RootWidgetSystem for TileSystem<'_, '_> {
    type Args = Option<Entity>;
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        target: Self::Args,
    ) {
        let Some((icons, is_empty_tile_tree, read_only)) =
            Self::prepare_panel_data(world, state, target)
        else {
            return;
        };

        let fill_color = if is_empty_tile_tree {
            get_scheme().bg_secondary
        } else {
            colors::TRANSPARENT
        };

        #[cfg(target_os = "macos")]
        let frame = {
            let mut frame = Frame {
                fill: fill_color,
                ..Default::default()
            };
            if target.is_some() {
                // Leave room for the native titlebar controls on secondary windows.
                frame.inner_margin.top = 32;
            }
            frame
        };
        #[cfg(not(target_os = "macos"))]
        let frame = Frame {
            fill: fill_color,
            ..Default::default()
        };

        let central = egui::CentralPanel::default().frame(frame);

        central.show(ctx, |ui| {
            Self::render_panel_contents(
                world,
                ui,
                target,
                icons.clone(),
                is_empty_tile_tree,
                read_only,
            );
        });
    }
}

#[derive(SystemParam)]
pub struct TileLayoutEmpty<'w, 's> {
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
    primary_window: Query<'w, 's, Entity, With<PrimaryWindow>>,
}

#[derive(Clone)]
pub struct TileLayoutEmptyArgs {
    pub icons: TileIcons,
    pub window: Option<Entity>,
}

impl WidgetSystem for TileLayoutEmpty<'_, '_> {
    type Args = TileLayoutEmptyArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let TileLayoutEmptyArgs { icons, window } = args;

        let window_size = window
            .and_then(|window_entity| world.get::<Window>(window_entity))
            .map(|window| window.resolution.size());
        let window_rect = window_size.and_then(|size| {
            if size.x > 0.0 && size.y > 0.0 {
                Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(size.x, size.y),
                ))
            } else {
                None
            }
        });
        let max_rect = ui.max_rect();
        let layout_rect = match window_rect {
            Some(rect)
                if max_rect.width() > rect.width() * 1.5
                    || max_rect.height() > rect.height() * 1.5 =>
            {
                rect
            }
            _ => max_rect,
        };

        let button_height = 160.0;
        let base_button_width: f32 = 240.0;
        let base_button_spacing: f32 = 20.0;
        let button_spacing = base_button_spacing.min((layout_rect.width() / 6.0).max(0.0));
        let max_button_width = ((layout_rect.width() - 2.0 * button_spacing) / 3.0).max(0.0);
        let button_width = max_button_width.min(base_button_width);
        let desired_size = egui::vec2(button_width * 3.0 + button_spacing * 2.0, button_height);

        let mut state_mut = state.get_mut(world);
        let target_window = window.or_else(|| state_mut.primary_window.iter().next());

        ui.scope_builder(
            UiBuilder::new().max_rect(egui::Rect::from_center_size(
                layout_rect.center(),
                desired_size,
            )),
            |ui| {
                ui.horizontal(|ui| {
                    ui.style_mut().spacing.item_spacing = egui::vec2(button_spacing, 0.0);

                    let create_viewport_btn = ui.add(
                        ETileButton::new("Viewport", icons.add)
                            .description("3D Output")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_viewport_btn.clicked() {
                        state_mut
                            .cmd_palette_state
                            .open_for_window(target_window, palette_items::create_viewport(None));
                    }

                    let create_graph_btn = ui.add(
                        ETileButton::new("Graph", icons.add)
                            .description("Point Graph")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_graph_btn.clicked() {
                        state_mut
                            .cmd_palette_state
                            .open_for_window(target_window, palette_items::create_graph(None));
                    }

                    let create_monitor_btn = ui.add(
                        ETileButton::new("Monitor", icons.add)
                            .description("Monitor")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_monitor_btn.clicked() {
                        state_mut
                            .cmd_palette_state
                            .open_for_window(target_window, palette_items::create_monitor(None));
                    }
                });
            },
        );
    }
}

#[derive(SystemParam)]
pub struct TileLayout<'w, 's> {
    commands: Commands<'w, 's>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    viewport_contains_pointer: ResMut<'w, ViewportContainsPointer>,
    editor_cam: Query<'w, 's, &'static mut EditorCam, With<MainCamera>>,
    primary_window: Single<'w, 's, Entity, With<PrimaryWindow>>,
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
    eql_ctx: Res<'w, EqlContext>,
    node_updater_params: NodeUpdaterParams<'w, 's>,
    tile_param: crate::ui::command_palette::palette_items::TileParam<'w, 's>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
    query_plots: Query<'w, 's, &'static mut QueryPlotData>,
    query_tables: Query<'w, 's, &'static mut QueryTableData>,
    action_tiles: Query<'w, 's, &'static mut ActionTile>,
    dashboards: Query<'w, 's, &'static mut Dashboard<Entity>>,
}

#[derive(Clone)]
pub struct TileLayoutArgs {
    pub icons: TileIcons,
    pub window: Option<Entity>,
    pub read_only: bool,
    pub show_empty_overlay: bool,
}

impl WidgetSystem for TileLayout<'_, '_> {
    type Args = TileLayoutArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let TileLayoutArgs {
            icons,
            window,
            read_only,
            show_empty_overlay,
        } = args;

        let target_window = {
            let state_mut = state.get_mut(world);
            window.unwrap_or(*state_mut.primary_window)
        };

        let (
            tree,
            mut tree_actions,
            sidebar_state,
            share_updates,
            empty_overlay_rect,
            overlay_icons,
        ) = {
            let (tab_diffs, container_titles, mut tree, sidebar_state) = {
                let mut state_mut = state.get_mut(world);
                let Some(mut window_state) = state_mut.tile_param.target_state(Some(target_window))
                else {
                    return;
                };
                let tile_state = &mut window_state.tile_state;
                let empty_tree = egui_tiles::Tree::empty(tile_state.tree_id);
                (
                    std::mem::take(&mut tile_state.tree_actions),
                    tile_state.container_titles.clone(),
                    std::mem::replace(&mut tile_state.tree, empty_tree),
                    tile_state.sidebar_state,
                )
            };
            let overlay_icons = icons.clone();
            let mut behavior = TreeBehavior {
                icons,
                // This world here makes getting ui_state difficult.
                world,
                tree_actions: tab_diffs,
                container_titles,
                read_only,
                target_window,
            };
            tree.ui(&mut behavior, ui);

            // Fix any invalid drops (non-sidebar tiles dropped into sidebar containers)
            fix_invalid_drops(&mut tree);

            let empty_overlay_rect = if show_empty_overlay {
                main_content_rect(&tree).or_else(|| Some(ui.max_rect()))
            } else {
                None
            };
            let window_width = ui.ctx().content_rect().width();
            let gutter_width = (window_width * 0.02).max(12.0);
            let painter = ui.painter_at(ui.max_rect());
            let mut sidebar_state = sidebar_state;
            let share_updates = collect_sidebar_gutter_updates(
                &mut tree,
                ui,
                painter,
                gutter_width,
                &mut sidebar_state,
            );
            let TreeBehavior { tree_actions, .. } = behavior;
            (
                tree,
                tree_actions,
                sidebar_state,
                share_updates,
                empty_overlay_rect,
                overlay_icons,
            )
        };

        {
            let mut state_mut = state.get_mut(world);
            let Some(mut window_state) = state_mut.tile_param.target_state(Some(target_window))
            else {
                return;
            };
            let WindowState {
                tile_state,
                ui_state,
                ..
            } = &mut *window_state;
            let _ = std::mem::replace(&mut tile_state.tree, tree);
            apply_share_updates(&mut tile_state.tree, &share_updates);
            tile_state.sidebar_state = sidebar_state;
            state_mut.viewport_contains_pointer.0 = ui.ui_contains_pointer();

            for mut editor_cam in state_mut.editor_cam.iter_mut() {
                editor_cam.enabled_motion = EnabledMotion {
                    pan: state_mut.viewport_contains_pointer.0,
                    orbit: state_mut.viewport_contains_pointer.0,
                    zoom: state_mut.viewport_contains_pointer.0,
                }
            }

            for diff in tree_actions.drain(..) {
                if read_only && !matches!(diff, TreeAction::SelectTile(_)) {
                    continue;
                }
                match diff {
                    TreeAction::DeleteTab(tile_id) => {
                        if read_only {
                            continue;
                        }
                        let Some(tile) = tile_state.tree.tiles.get(tile_id) else {
                            continue;
                        };

                        if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
                            if let Some(layer) = viewport.viewport_layer {
                                state_mut.render_layer_alloc.free(layer);
                            }
                            if let Some(layer) = viewport.grid_layer {
                                state_mut.render_layer_alloc.free(layer);
                            }
                            if let Some(camera) = viewport.camera {
                                state_mut.commands.entity(camera).despawn();
                            }
                            if let Some(nav_gizmo_camera) = viewport.nav_gizmo_camera {
                                state_mut.commands.entity(nav_gizmo_camera).despawn();
                            }
                            if let Some(nav_gizmo) = viewport.nav_gizmo {
                                state_mut.commands.entity(nav_gizmo).despawn();
                            }
                        };

                        if let egui_tiles::Tile::Pane(Pane::Graph(graph)) = tile {
                            state_mut.commands.entity(graph.id).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::ActionTile(action)) = tile {
                            state_mut.commands.entity(action.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::VideoStream(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::QueryPlot(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::QueryTable(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::SchematicTree(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        tile_state.tree.remove_recursively(tile_id);

                        if let Some(graph_id) = tile_state.graphs.get(&tile_id) {
                            state_mut.commands.entity(*graph_id).despawn();
                            tile_state.graphs.remove(&tile_id);
                        }

                        if tile_state.has_non_sidebar_content() {
                            tile_state
                                .tree
                                .simplify(&egui_tiles::SimplificationOptions {
                                    prune_empty_tabs: true,
                                    prune_single_child_tabs: false,
                                    all_panes_must_have_tabs: true,
                                    join_nested_linear_containers: true,
                                    ..Default::default()
                                });
                        }
                    }
                    TreeAction::AddViewport(parent_tile_id) => {
                        if read_only {
                            continue;
                        }
                        let viewport = Viewport::default();
                        let label = viewport_label(&viewport);
                        let viewport_pane = ViewportPane::spawn(
                            &mut state_mut.commands,
                            &state_mut.asset_server,
                            &mut state_mut.meshes,
                            &mut state_mut.materials,
                            &mut state_mut.render_layer_alloc,
                            &state_mut.eql_ctx.0,
                            &viewport,
                            label,
                        );

                        if let Some(tile_id) = tile_state.insert_tile(
                            Tile::Pane(Pane::Viewport(viewport_pane)),
                            parent_tile_id,
                            true,
                        ) {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddGraph(parent_tile_id, graph_bundle) => {
                        if read_only {
                            continue;
                        }
                        let graph_label = graph_label(&Graph::default());

                        let graph_bundle = if let Some(graph_bundle) = *graph_bundle {
                            graph_bundle
                        } else {
                            GraphBundle::new(
                                &mut state_mut.render_layer_alloc,
                                BTreeMap::default(),
                                graph_label.clone(),
                            )
                        };
                        let graph_id = state_mut.commands.spawn(graph_bundle).id();
                        let graph = GraphPane::new(graph_id, graph_label.clone());
                        let pane = Pane::Graph(graph);

                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.selected_object = SelectedObject::Graph { graph_id };
                            tile_state.tree.make_active(|id, _| id == tile_id);
                            tile_state.graphs.insert(tile_id, graph_id);
                        }
                    }
                    TreeAction::AddMonitor(parent_tile_id, eql) => {
                        if read_only {
                            continue;
                        }
                        let monitor = MonitorPane::new(eql.clone(), eql);

                        let pane = Pane::Monitor(monitor);
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddVideoStream(parent_tile_id, msg_id, name) => {
                        if read_only {
                            continue;
                        }
                        let entity = state_mut
                            .commands
                            .spawn((
                                super::video_stream::VideoStream {
                                    msg_id,
                                    ..Default::default()
                                },
                                bevy::ui::Node {
                                    position_type: PositionType::Absolute,
                                    ..Default::default()
                                },
                                bevy::prelude::ImageNode {
                                    image_mode: NodeImageMode::Stretch,
                                    ..Default::default()
                                },
                                VideoDecoderHandle::default(),
                            ))
                            .id();
                        let pane = Pane::VideoStream(super::video_stream::VideoStreamPane {
                            entity,
                            name: name.clone(),
                        });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddDashboard(parent_tile_id, dashboard, name) => {
                        if read_only {
                            continue;
                        }
                        let entity = match spawn_dashboard(
                            &dashboard,
                            &state_mut.eql_ctx.0,
                            &mut state_mut.commands,
                            &state_mut.node_updater_params,
                        ) {
                            Ok(entity) => entity,
                            Err(_) => state_mut.commands.spawn(bevy::ui::Node::default()).id(),
                        };
                        let pane = Pane::Dashboard(DashboardPane { entity, name });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }

                    TreeAction::SelectTile(tile_id) => {
                        tile_state.tree.make_active(|id, _| id == tile_id);

                        if let Some(egui_tiles::Tile::Pane(pane)) =
                            tile_state.tree.tiles.get(tile_id)
                        {
                            match pane {
                                Pane::Graph(graph) => {
                                    ui_state.selected_object =
                                        SelectedObject::Graph { graph_id: graph.id };
                                }
                                Pane::QueryPlot(plot) => {
                                    ui_state.selected_object = SelectedObject::Graph {
                                        graph_id: plot.entity,
                                    };
                                }
                                Pane::Viewport(viewport) => {
                                    if let Some(camera) = viewport.camera {
                                        ui_state.selected_object =
                                            SelectedObject::Viewport { camera };
                                    }
                                }
                                _ => {}
                            }
                            if let Some(kind) = pane.sidebar_kind() {
                                unmask_sidebar_by_kind(tile_state, kind);
                            }
                            unmask_sidebar_by_kind(tile_state, SidebarKind::Inspector);
                        }
                    }
                    TreeAction::AddActionTile(parent_tile_id, button_name, lua_code) => {
                        let name = button_name.clone();
                        let entity = state_mut
                            .commands
                            .spawn(super::actions::ActionTile {
                                button_name,
                                lua: lua_code,
                                status: Default::default(),
                            })
                            .id();
                        let pane = Pane::ActionTile(ActionTilePane { entity, name });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddQueryTable(parent_tile_id) => {
                        let entity = state_mut.commands.spawn(QueryTableData::default()).id();
                        let pane = Pane::QueryTable(QueryTablePane {
                            entity,
                            name: "Query".to_string(),
                        });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddQueryPlot(parent_tile_id) => {
                        let graph_bundle = GraphBundle::new(
                            &mut state_mut.render_layer_alloc,
                            BTreeMap::default(),
                            "Query Plot".to_string(),
                        );
                        let entity = state_mut
                            .commands
                            .spawn(QueryPlotData::default())
                            .insert(graph_bundle)
                            .id();
                        let pane = Pane::QueryPlot(super::query_plot::QueryPlotPane {
                            entity,
                            rect: None,
                            scrub_icon: None,
                        });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.selected_object = SelectedObject::Graph { graph_id: entity };
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddSchematicTree(parent_tile_id) => {
                        if read_only {
                            continue;
                        }
                        let entity = state_mut
                            .commands
                            .spawn(super::schematic::tree::TreeWidgetState::default())
                            .id();
                        let pane = Pane::SchematicTree(TreePane {
                            entity,
                            name: "Tree".to_string(),
                        });
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddDataOverview(parent_tile_id) => {
                        if read_only {
                            continue;
                        }
                        let pane = Pane::DataOverview(DataOverviewPane::default());
                        if let Some(tile_id) =
                            tile_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            tile_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddSidebars => {
                        if read_only {
                            continue;
                        }
                        tile_state.apply_sidebars_layout();
                    }
                    TreeAction::RenameContainer(tile_id, title) => {
                        if read_only {
                            continue;
                        }
                        tile_state.set_container_title(tile_id, title);
                    }
                    TreeAction::RenamePane(tile_id, title) => {
                        if read_only {
                            continue;
                        }
                        let Some(tile) = tile_state.tree.tiles.get_mut(tile_id) else {
                            continue;
                        };
                        if let Tile::Pane(pane) = tile {
                            let targets = pane.set_title(&title);
                            apply_pane_title_updates(
                                &title,
                                targets,
                                &mut state_mut.graph_states,
                                &mut state_mut.query_plots,
                                &mut state_mut.query_tables,
                                &mut state_mut.action_tiles,
                                &mut state_mut.dashboards,
                            );
                        }
                    }
                }
            }
            let tiles = tile_state.tree.tiles.iter();
            let active_tiles = tile_state.tree.active_tiles();
            for (tile_id, tile) in tiles {
                let egui_tiles::Tile::Pane(pane) = tile else {
                    continue;
                };
                let visible = read_only || active_tiles.contains(tile_id);
                match pane {
                    Pane::Viewport(viewport) => {
                        let Some(cam) = viewport.camera else { continue };
                        if visible {
                            if let Ok(mut cam) = state_mut.commands.get_entity(cam) {
                                cam.try_insert(ViewportRect(viewport.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(cam) {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::Hierarchy => {}
                    Pane::Inspector => {}
                    Pane::SchematicTree(_) => {}
                    Pane::Graph(graph) => {
                        if visible {
                            if let Ok(mut cam) = state_mut.commands.get_entity(graph.id) {
                                cam.try_insert(ViewportRect(graph.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(graph.id) {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::Monitor(_) => {}
                    Pane::QueryTable(_) => {}
                    Pane::QueryPlot(query_plot) => {
                        if visible {
                            if let Ok(mut cam) = state_mut.commands.get_entity(query_plot.entity) {
                                cam.try_insert(ViewportRect(query_plot.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(query_plot.entity)
                        {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::ActionTile(_) => {}
                    Pane::VideoStream(stream) => {
                        if let Ok(mut stream) = state_mut.commands.get_entity(stream.entity) {
                            stream.try_insert(IsTileVisible(visible));
                        }
                    }

                    Pane::Dashboard(dash) => {
                        if let Ok(mut stream) = state_mut.commands.get_entity(dash.entity) {
                            stream.try_insert(IsTileVisible(visible));
                        }
                    }
                    Pane::DataOverview(_) => {}
                }
            }
        }

        if show_empty_overlay && let Some(rect) = empty_overlay_rect {
            ui.scope_builder(UiBuilder::new().max_rect(rect), |ui| {
                ui.add_widget_with::<TileLayoutEmpty>(
                    world,
                    "tile_layout_empty",
                    TileLayoutEmptyArgs {
                        icons: overlay_icons,
                        window: Some(target_window),
                    },
                );
            });
        }
    }
}

fn unmask_sidebar_on_select(ui_state: &mut TileState, sidebar_tile: TileId, kind: SidebarKind) {
    fn build_parent_map(tiles: &Tiles<Pane>) -> HashMap<TileId, TileId> {
        let mut parents = HashMap::new();
        for (id, tile) in tiles.iter() {
            if let Tile::Container(container) = tile {
                for child in container.children() {
                    parents.insert(*child, *id);
                }
            }
        }
        parents
    }

    fn rebalance_linear_shares(
        linear: &mut egui_tiles::Linear,
        target_child: TileId,
        target_frac: f32,
        min_share: f32,
    ) {
        let share_sum: f32 = linear
            .shares
            .iter()
            .map(|(_, s)| s)
            .sum::<f32>()
            .max(min_share);
        let old = linear.shares[target_child];
        let others_sum = (share_sum - old).max(0.0);
        let mut target = (share_sum * target_frac).max(min_share);
        let min_others = min_share * (linear.children.len().saturating_sub(1) as f32);
        if target > share_sum - min_others {
            target = (share_sum - min_others).max(min_share);
        }
        let factor = if others_sum > 0.0 {
            (share_sum - target) / others_sum
        } else {
            0.0
        };
        let children = linear.children.clone();
        for child in children {
            if child == target_child {
                linear.shares.set_share(child, target.max(min_share));
            } else {
                let new = (linear.shares[child] * factor).max(min_share);
                linear.shares.set_share(child, new);
            }
        }
    }

    let target_frac = ui_state.sidebar_state.last_share(kind).unwrap_or(0.2);
    let min_share = SIDEBAR_COLLAPSED_SHARE;
    let parents = build_parent_map(&ui_state.tree.tiles);
    let mut current = Some(sidebar_tile);
    while let Some(child) = current {
        let Some(parent) = parents.get(&child).copied() else {
            break;
        };
        if let Some(Tile::Container(Container::Linear(linear))) =
            ui_state.tree.tiles.get_mut(parent)
        {
            rebalance_linear_shares(linear, child, target_frac, min_share);
        }
        current = Some(parent);
    }

    ui_state.sidebar_state.set_masked(kind, false);
    ui_state
        .sidebar_state
        .set_last_share(kind, Some(target_frac));
}

fn unmask_sidebar_by_kind(ui_state: &mut TileState, kind: SidebarKind) {
    let target = ui_state
        .tree
        .tiles
        .iter()
        .find_map(|(id, tile)| match (kind, tile) {
            (SidebarKind::Hierarchy, Tile::Pane(Pane::Hierarchy)) => Some(*id),
            (SidebarKind::Inspector, Tile::Pane(Pane::Inspector)) => Some(*id),
            _ => None,
        });

    if let Some(tile_id) = target {
        unmask_sidebar_on_select(ui_state, tile_id, kind);
    }
}

pub fn shortcuts(
    key_state: Res<LogicalKeyState>,
    primary_window: Single<Entity, With<PrimaryWindow>>,
    mut window_state: Query<&mut WindowState>,
) {
    let Ok(window_state) = &mut window_state.get_mut(*primary_window) else {
        return;
    };
    let ui_state = &mut window_state.tile_state;
    if key_state.pressed(&Key::Control) && key_state.just_pressed(&Key::Tab) {
        let Some(tile_id) = ui_state.tree.root() else {
            return;
        };
        let Some(tile) = ui_state.tree.tiles.get_mut(tile_id) else {
            return;
        };
        let Tile::Container(container) = tile else {
            return;
        };

        let Container::Tabs(tabs) = container else {
            return;
        };
        let Some(active_id) = tabs.active else {
            return;
        };
        let Some(index) = tabs.children.iter().position(|x| *x == active_id) else {
            return;
        };
        let offset = if key_state.pressed(&Key::Shift) {
            -1
        } else {
            1
        };
        let new_index = if let Some(index) = index.checked_add_signed(offset) {
            index % tabs.children.len()
        } else {
            tabs.children.len() - 1
        };
        let Some(new_active_id) = tabs.children.get(new_index) else {
            return;
        };
        tabs.set_active(*new_active_id);
    }
}
