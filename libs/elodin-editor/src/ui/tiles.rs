use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    ecs::system::{SystemParam, SystemState},
    input::keyboard::Key,
    log::info,
    prelude::*,
    window::{Monitor, PrimaryWindow, Window, WindowPosition},
};
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion, OrbitConstraint};
use bevy_egui::{
    EguiContexts,
    egui::{self, Frame, Id},
};
use bevy_render::{
    camera::{Exposure, PhysicalCameraParameters},
    view::RenderLayers,
};
use egui::{Ui, UiBuilder};
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
    SelectedObject, ViewportRect,
    actions::{ActionTile, ActionTileWidget},
    button::{EImageButton, ETileButton},
    colors::{self, get_scheme},
    command_palette::{CommandPaletteState, palette_items},
    dashboard::{DashboardWidget, spawn_dashboard},
    hierarchy::{Hierarchy, HierarchyContent},
    images,
    inspector::{InspectorContent, InspectorIcons},
    monitor::{MonitorPane, MonitorWidget},
    plot::{GraphBundle, GraphState, PlotWidget},
    query_plot::QueryPlotData,
    query_table::{QueryTableData, QueryTablePane, QueryTableWidget},
    schematic::{graph_label, viewport_label},
    video_stream::{IsTileVisible, VideoDecoderHandle},
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

mod behavior;
mod render;
mod sidebar;
mod tile_actions;
mod types;
mod util;

use render::{apply_share_updates, render_tree_and_collect_updates};
use sidebar::SidebarMaskState;
use tile_actions::ActionContext;
pub use tile_actions::TreeAction;
pub use types::{
    ActionTilePane, DashboardPane, GraphPane, Pane, TileIcons, TreePane, ViewportContainsPointer,
    ViewportPane,
};
use util::{describe_tile, is_content_tile};

pub(crate) fn plugin(app: &mut App) {
    app.register_type::<WindowId>()
        .add_event::<WindowRelayout>()
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
    };
    commands.entity(id).insert((state, WindowId(0)));
}

#[derive(Component)]
pub struct ViewportConfig {
    pub show_arrows: bool,
    pub viewport_layer: Option<usize>,
}

#[derive(Clone)]
pub struct TileState {
    pub tree: egui_tiles::Tree<Pane>,
    pub tree_actions: smallvec::SmallVec<[TreeAction; 4]>,
    pub graphs: HashMap<TileId, Entity>,
    pub container_titles: HashMap<TileId, String>,
    pub has_non_sidebar: bool,
    pub hierarchy_masked: bool,
    pub inspector_masked: bool,
    pub last_hierarchy_share: Option<f32>,
    pub last_inspector_share: Option<f32>,
    tree_id: Id,
}

#[derive(Clone, Debug, Default)]
pub struct WindowDescriptor {
    pub path: Option<PathBuf>,
    pub title: Option<String>,
    pub screen: Option<usize>,
    pub screen_rect: Option<WindowRect>,
}

impl WindowDescriptor {
    pub fn wants_explicit_layout(&self) -> bool {
        self.screen.is_some() || self.screen_rect.is_some()
    }
}

/// Events dealing with window layout
#[derive(Event, Clone, Debug, PartialEq, Eq)]
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
            if let Some((screen_pos, screen_size)) = best_bounds {
                if position_is_reliable_linux(position, (screen_pos.x, screen_pos.y)) {
                    if let Some(rect) = rect_from_bounds(
                        (position.x, position.y),
                        (size.x, size.y),
                        (screen_pos.x, screen_pos.y),
                        (screen_size.x, screen_size.y),
                    ) {
                        self.descriptor.screen_rect = Some(rect);
                    }
                }
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
                ) {
                    if let Some(rect) = rect_from_bounds(
                        (position.x, position.y),
                        (outer_size.width, outer_size.height),
                        (screen_pos.x, screen_pos.y),
                        (screen_handle.size().width, screen_handle.size().height),
                    ) {
                        self.descriptor.screen_rect = Some(rect);
                        updated = true;
                    }
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
        screen_rect: None,
    };
    let tile_state = TileState::new(Id::new(("secondary_tab_tree", id.0)));
    info!(
        id = id.0,
        title = descriptor.title.as_deref().unwrap_or(""),
        path = ?descriptor.path,
        "Created secondary window"
    );
    (
        WindowState {
            descriptor,
            tile_state,
            graph_entities: Vec::new(),
        },
        id,
    )
}

impl TileState {
    fn recompute_has_non_sidebar(&mut self) {
        fn visit(tree: &egui_tiles::Tree<Pane>, id: TileId) -> bool {
            match tree.tiles.get(id) {
                Some(tile) if is_content_tile(tile) => true,
                Some(Tile::Container(container)) => {
                    container.children().any(|child| visit(tree, *child))
                }
                Some(Tile::Pane(_)) => false,
                None => false,
            }
        }
        self.has_non_sidebar = self
            .tree
            .root()
            .map(|root| visit(&self.tree, root))
            .unwrap_or(false);
    }

    // Track when a new content pane appears so we know the layout is not empty.
    fn update_has_non_sidebar_for(&mut self, tile_id: TileId) {
        if self.has_non_sidebar {
            return;
        }

        fn visit(tree: &egui_tiles::Tree<Pane>, id: TileId) -> bool {
            match tree.tiles.get(id) {
                Some(tile) if is_content_tile(tile) => true,
                Some(Tile::Container(container)) => {
                    container.children().any(|child| visit(tree, *child))
                }
                Some(Tile::Pane(_)) | None => false,
            }
        }

        self.has_non_sidebar = visit(&self.tree, tile_id);
    }

    fn reset_scaffold_state(&mut self) {
        self.container_titles.clear();
        self.tree_actions.clear();
        self.graphs.clear();
        self.hierarchy_masked = true;
        self.inspector_masked = true;
        self.last_hierarchy_share = Some(0.2);
        self.last_inspector_share = Some(0.2);
        self.has_non_sidebar = false;
    }

    fn apply_scaffold(&mut self, title: &str) -> Option<TileId> {
        let mut tree = egui_tiles::Tree::empty(self.tree_id);
        self.reset_scaffold_state();

        let tabs_left =
            tree.tiles
                .insert_new(Tile::Container(Container::Tabs(egui_tiles::Tabs::new(
                    vec![],
                ))));
        let tabs_right =
            tree.tiles
                .insert_new(Tile::Container(Container::Tabs(egui_tiles::Tabs::new(
                    vec![],
                ))));

        let tabs_inner_left =
            tree.tiles
                .insert_new(Tile::Container(Container::Tabs(egui_tiles::Tabs::new(
                    vec![],
                ))));
        let tabs_inner_center =
            tree.tiles
                .insert_new(Tile::Container(Container::Tabs(egui_tiles::Tabs::new(
                    vec![],
                ))));
        let tabs_inner_right =
            tree.tiles
                .insert_new(Tile::Container(Container::Tabs(egui_tiles::Tabs::new(
                    vec![],
                ))));

        let mut inner_linear = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![tabs_inner_left, tabs_inner_center, tabs_inner_right],
        );
        inner_linear.shares.set_share(tabs_inner_left, 0.001);
        inner_linear.shares.set_share(tabs_inner_center, 0.98);
        inner_linear.shares.set_share(tabs_inner_right, 0.001);
        let inner_linear_id = tree
            .tiles
            .insert_new(Tile::Container(Container::Linear(inner_linear)));

        let mut outer_linear = egui_tiles::Linear::new(egui_tiles::LinearDir::Horizontal, vec![]);
        outer_linear.children.push(tabs_left);
        outer_linear.children.push(inner_linear_id);
        outer_linear.children.push(tabs_right);
        outer_linear.shares.set_share(tabs_left, 0.001);
        outer_linear.shares.set_share(inner_linear_id, 0.98);
        outer_linear.shares.set_share(tabs_right, 0.001);
        let root_id = tree
            .tiles
            .insert_new(Tile::Container(Container::Linear(outer_linear)));
        tree.root = Some(root_id);
        tree.make_active(|id, _| id == tabs_inner_center);

        self.tree = tree;
        self.set_container_title(inner_linear_id, title);
        self.set_container_title(tabs_inner_center, title);

        Some(tabs_inner_center)
    }

    fn default_tabs_title_for_tile(tile: &Tile<Pane>) -> Option<&'static str> {
        match tile {
            Tile::Pane(Pane::Viewport(_)) => Some("Viewports"),
            Tile::Pane(Pane::Monitor(_)) => Some("Monitors"),
            Tile::Pane(Pane::Graph(_)) => Some("Graphs"),
            _ => None,
        }
    }

    pub fn insert_tile(
        &mut self,
        tile: Tile<Pane>,
        parent_id: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        let tabs_title_hint = Self::default_tabs_title_for_tile(&tile);
        let parent_id = parent_id.or_else(|| self.default_parent_center())?;

        let tile_id = self.tree.tiles.insert_new(tile);
        let is_tabs_container = {
            let parent_tile = self.tree.tiles.get_mut(parent_id)?;
            let Tile::Container(container) = parent_tile else {
                return None;
            };

            let is_tabs = matches!(container, Container::Tabs(_));
            container.add_child(tile_id);

            if active {
                if let Container::Tabs(tabs) = container {
                    tabs.set_active(tile_id);
                }
            }
            if is_tabs {
                if let Some(tile) = self.tree.tiles.get(tile_id) {
                    let (kind, label) = describe_tile(tile);
                    info!(
                        target: "tabs.insert",
                        ?parent_id,
                        tile_id = ?tile_id,
                        kind = %kind,
                        label = %label.unwrap_or_default(),
                        tabs_title_hint = tabs_title_hint.unwrap_or(""),
                        parent_title = %self.container_titles.get(&parent_id).map(String::as_str).unwrap_or(""),
                        "insert_tile: added child to Tabs"
                    );
                }
            }
            is_tabs
        };

        if is_tabs_container {
            if let Some(title) = tabs_title_hint {
                if !self.container_titles.contains_key(&parent_id) {
                    self.set_container_title(parent_id, title);
                }
            }
        }

        self.update_has_non_sidebar_for(tile_id);

        Some(tile_id)
    }

    pub fn insert_root_tile(&mut self, tile: Tile<Pane>) -> TileId {
        let tile_id = self.tree.tiles.insert_new(tile);
        self.tree.root = Some(tile_id);
        self.tree.make_active(|id, _| id == tile_id);
        tile_id
    }

    pub(crate) fn default_parent_center(&mut self) -> Option<TileId> {
        let root_id = self.tree.root().or_else(|| {
            self.reset_tree();
            self.tree.root()
        })?;

        if let Some(Tile::Container(Container::Linear(linear))) = self.tree.tiles.get(root_id) {
            let center = linear
                .children
                .get(linear.children.len() / 2)
                .copied()
                .unwrap_or(root_id);
            match self.tree.tiles.get(center) {
                Some(Tile::Container(Container::Tabs(_))) => Some(center),
                Some(Tile::Container(Container::Linear(inner))) => inner
                    .children
                    .get(inner.children.len() / 2)
                    .copied()
                    .or(Some(center)),
                _ => Some(center),
            }
        } else {
            Some(root_id)
        }
    }

    pub fn create_graph_tile(
        &mut self,
        parent_id: Option<TileId>,
        graph_state: GraphBundle,
        new_tab: bool,
    ) {
        self.tree_actions.push(TreeAction::AddGraph(
            parent_id,
            Box::new(Some(graph_state)),
            new_tab,
        ));
    }

    pub fn create_graph_tile_empty(&mut self, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddGraph(None, Box::new(None), new_tab));
    }

    pub fn create_viewport_tile(&mut self, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddViewport(tile_id, new_tab));
    }

    pub fn create_viewport_tile_empty(&mut self, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddViewport(None, new_tab));
    }

    pub fn create_monitor_tile(&mut self, eql: String, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddMonitor(tile_id, eql, new_tab));
    }

    pub fn create_action_tile(
        &mut self,
        button_name: String,
        lua_code: String,
        tile_id: Option<TileId>,
        new_tab: bool,
    ) {
        self.tree_actions.push(TreeAction::AddActionTile(
            tile_id,
            button_name,
            lua_code,
            new_tab,
        ));
    }

    pub fn create_query_table_tile(&mut self, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddQueryTable(tile_id, new_tab));
    }

    pub fn create_query_plot_tile(&mut self, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddQueryPlot(tile_id, new_tab));
    }

    pub fn create_video_stream_tile(
        &mut self,
        msg_id: [u8; 2],
        label: String,
        tile_id: Option<TileId>,
        new_tab: bool,
    ) {
        self.tree_actions
            .push(TreeAction::AddVideoStream(tile_id, msg_id, label, new_tab));
    }

    pub fn create_dashboard_tile(
        &mut self,
        dashboard: impeller2_wkt::Dashboard,
        label: String,
        tile_id: Option<TileId>,
        new_tab: bool,
    ) {
        self.tree_actions.push(TreeAction::AddDashboard(
            tile_id,
            Box::new(dashboard),
            label,
            new_tab,
        ));
    }

    pub fn create_hierarchy_tile(&mut self, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddHierarchy(tile_id, new_tab));
    }

    pub fn create_tree_tile(&mut self, tile_id: Option<TileId>, new_tab: bool) {
        self.tree_actions
            .push(TreeAction::AddSchematicTree(tile_id, new_tab));
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
                            Pane::Viewport(viewport) => ("Viewport", viewport.label.as_str()),
                            Pane::Graph(graph) => ("Graph", graph.label.as_str()),
                            Pane::Monitor(monitor) => ("Monitor", monitor.label.as_str()),
                            Pane::QueryTable(_) => ("QueryTable", "QueryTable"),
                            Pane::QueryPlot(_) => ("QueryPlot", "QueryPlot"),
                            Pane::ActionTile(action) => ("Action", action.label.as_str()),
                            Pane::VideoStream(video) => ("VideoStream", video.label.as_str()),
                            Pane::Dashboard(dashboard) => ("Dashboard", dashboard.label.as_str()),
                            Pane::Hierarchy => ("Hierarchy", "Hierarchy"),
                            Pane::Inspector => ("Inspector", "Inspector"),
                            Pane::SchematicTree(_) => ("SchematicTree", "SchematicTree"),
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
        fn visit(
            tree: &egui_tiles::Tree<Pane>,
            tile_id: egui_tiles::TileId,
            out: &mut Vec<Entity>,
        ) {
            let Some(tile) = tree.tiles.get(tile_id) else {
                return;
            };
            match tile {
                Tile::Pane(Pane::Graph(graph)) => out.push(graph.id),
                Tile::Pane(Pane::Viewport(viewport)) => {
                    if let Some(cam) = viewport.camera {
                        out.push(cam);
                    }
                    if let Some(nav_cam) = viewport.nav_gizmo_camera {
                        out.push(nav_cam);
                    }
                }
                Tile::Pane(_) => {}
                Tile::Container(container) => {
                    for child in container.children() {
                        visit(tree, *child, out);
                    }
                }
            }
        }

        let mut entities = Vec::new();
        if let Some(root) = self.tree.root() {
            visit(&self.tree, root, &mut entities);
        }
        entities
    }

    pub fn create_sidebars_layout(&mut self) {
        self.tree_actions.push(TreeAction::AddSidebars);
    }

    pub fn is_empty(&self) -> bool {
        self.tree.active_tiles().is_empty()
    }

    pub fn clear(
        &mut self,
        commands: &mut Commands,
        _selected_object: &mut SelectedObject,
        render_layer_alloc: &mut RenderLayerAlloc,
    ) {
        for (tile_id, tile) in self.tree.tiles.iter() {
            match tile {
                Tile::Pane(Pane::Viewport(viewport)) => {
                    bevy::log::info!(
                        grid_layer = ?viewport.grid_layer,
                        viewport_layer = ?viewport.viewport_layer,
                        camera = ?viewport.camera,
                        "clear: free viewport layers"
                    );
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

        if let Some(root_id) = self.tree.root() {
            if let Some(Tile::Container(root)) = self.tree.tiles.get_mut(root_id) {
                root.retain(|_| false);
            }
        };
        self.graphs.clear();
        self.container_titles.clear();
        self.tree_actions.clear();
        self.reset_tree();
        self.has_non_sidebar = false;
    }

    pub fn get_container_title(&self, id: TileId) -> Option<&str> {
        self.container_titles.get(&id).map(|s| s.as_str())
    }

    pub fn set_container_title(&mut self, id: TileId, title: impl Into<String>) {
        let title = title.into();
        bevy::log::info!(
            target: "tiles.title",
            ?id,
            title = %title,
            "set_container_title"
        );
        self.container_titles.insert(id, title);
    }

    pub fn clear_container_title(&mut self, id: TileId) {
        bevy::log::info!(target: "tiles.title", ?id, "clear_container_title");
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

impl Pane {
    fn fallback_label(label: &str, default: &str) -> String {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            default.to_string()
        } else {
            trimmed.to_string()
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
                    return Self::fallback_label(&graph_state.label, "Graph");
                }
                Self::fallback_label(&pane.label, "Graph")
            }
            Pane::Viewport(viewport) => Self::fallback_label(&viewport.label, "Viewport"),
            Pane::Monitor(monitor) => Self::fallback_label(&monitor.label, "Monitor"),
            Pane::QueryTable(..) => "Query".to_string(),
            Pane::QueryPlot(query_plot) => {
                if let Ok(graph_state) = graph_states.get(query_plot.entity) {
                    return Self::fallback_label(&graph_state.label, "Query Plot");
                }
                "Query Plot".to_string()
            }
            Pane::ActionTile(action) => Self::fallback_label(&action.label, "Action"),
            Pane::VideoStream(video_stream) => {
                Self::fallback_label(&video_stream.label, "Video Stream")
            }
            Pane::Dashboard(dashboard) => {
                if let Ok(dash) = dashboards.get(dashboard.entity) {
                    return Self::fallback_label(
                        dash.root.label.as_deref().unwrap_or(""),
                        "Dashboard",
                    );
                }
                "Dashboard".to_string()
            }
            Pane::Hierarchy => "Entities".to_string(),
            Pane::Inspector => "Inspector".to_string(),
            Pane::SchematicTree(_) => "Tree".to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        ui: &mut Ui,
        icons: &TileIcons,
        world: &mut World,
        tree_actions: &mut SmallVec<[TreeAction; 4]>,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                pane.rect = Some(content_rect);

                ui.add_widget_with::<PlotWidget>(world, "graph", (pane.id, icons.scrub));

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
                    pane_with_icon,
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
                    pane.clone(),
                );
                egui_tiles::UiResponse::None
            }
            Pane::Dashboard(pane) => {
                ui.add_widget_with::<DashboardWidget>(world, "dashboard", pane.entity);
                egui_tiles::UiResponse::None
            }
            Pane::Hierarchy => {
                ui.add_widget_with::<HierarchyContent>(
                    world,
                    "hierarchy_content",
                    Hierarchy {
                        search: icons.search,
                        entity: icons.entity,
                        chevron: icons.chevron,
                    },
                );
                egui_tiles::UiResponse::None
            }
            Pane::Inspector => {
                let inspector_icons = InspectorIcons {
                    chart: icons.chart,
                    add: icons.add,
                    subtract: icons.subtract,
                    setting: icons.setting,
                    search: icons.search,
                };
                let actions = ui.add_widget_with::<InspectorContent>(
                    world,
                    "inspector_content",
                    (inspector_icons, true),
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
                    (tree_icons, tree_pane.entity),
                );
                egui_tiles::UiResponse::None
            }
        }
    }
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
        label: String,
    ) -> Self {
        let mut main_camera_layers = RenderLayers::default().with(GIZMO_RENDER_LAYER);
        let grid_layer = render_layer_alloc.alloc();
        let mut grid_layers = RenderLayers::none();
        if let Some(layer) = grid_layer {
            main_camera_layers = main_camera_layers.with(layer);
            grid_layers = RenderLayers::layer(layer);
        } else {
            bevy::log::error!("grid layer allocation failed; grid will not render");
        }

        let viewport_layer = render_layer_alloc.alloc();
        if let Some(layer) = viewport_layer {
            main_camera_layers = main_camera_layers.with(layer);
        } else {
            bevy::log::warn!("viewport layer allocation failed; arrows will not render");
        }

        let grid_visibility = if viewport.show_grid {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let grid_settings = bevy_infinite_grid::InfiniteGridSettings {
            minor_line_color: Color::srgba(1.0, 1.0, 1.0, 0.02),
            major_line_color: Color::srgba(1.0, 1.0, 1.0, 0.05),
            z_axis_color: crate::ui::colors::bevy::GREEN,
            x_axis_color: crate::ui::colors::bevy::RED,
            fadeout_distance: 50_000.0,
            scale: 0.1,
            ..Default::default()
        };
        let grid_id = commands
            .spawn((
                bevy_infinite_grid::InfiniteGridBundle {
                    settings: grid_settings,
                    visibility: grid_visibility,
                    ..Default::default()
                },
                grid_layers,
            ))
            .id();
        let perspective = PerspectiveProjection {
            fov: viewport.fov.to_radians(),
            ..Default::default()
        };

        let parent_transform =
            Transform::from_translation(Vec3::new(5.0, 5.0, 10.0)).looking_at(Vec3::ZERO, Vec3::Y);
        let parent = commands
            .spawn((
                GlobalTransform::default(),
                parent_transform,
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
                hdr: viewport.hdr,
                clear_color: ClearColorConfig::Default,
                order: 1,
                ..Default::default()
            },
            Projection::Perspective(perspective),
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
            label,
            grid_layer,
            viewport_layer,
        }
    }
}

impl GraphPane {
    pub fn new(graph_id: Entity, label: String) -> Self {
        Self {
            id: graph_id,
            label,
            rect: None,
        }
    }
}

impl TileState {
    pub fn tree_id(&self) -> Id {
        self.tree_id
    }

    pub fn new(tree_id: Id) -> Self {
        Self {
            tree: egui_tiles::Tree::new_tabs(tree_id, vec![]),
            tree_actions: smallvec![TreeAction::AddSidebars],
            graphs: HashMap::new(),
            container_titles: HashMap::new(),
            has_non_sidebar: false,
            hierarchy_masked: true,
            inspector_masked: true,
            last_hierarchy_share: Some(0.2),
            last_inspector_share: Some(0.2),
            tree_id,
        }
    }

    fn reset_tree(&mut self) {
        self.tree = egui_tiles::Tree::new_tabs(self.tree_id, vec![]);
        self.tree_actions = smallvec![TreeAction::AddSidebars];
        self.hierarchy_masked = true;
        self.inspector_masked = true;
        self.last_hierarchy_share = Some(0.2);
        self.last_inspector_share = Some(0.2);
        self.has_non_sidebar = false;
    }

    /// Returns true when there is any user-visible content beyond the built-in sidebars.
    fn has_non_sidebar_content(&self) -> bool {
        self.has_non_sidebar
    }

    fn insert_pane_in_tabs(
        &mut self,
        pane: Pane,
        parent_tile_id: Option<TileId>,
        active: bool,
        window_entity: Option<Entity>,
        new_tab: bool,
    ) -> Option<TileId> {
        let parent_tabs_id = if let Some(id) = parent_tile_id {
            id
        } else if !self.has_non_sidebar_content() {
            self.apply_scaffold("New tab")?
        } else {
            self.default_parent_center()?
        };

        if new_tab {
            if let Some(Tile::Container(Container::Tabs(parent_tabs))) =
                self.tree.tiles.get(parent_tabs_id)
            {
                let child_kinds: SmallVec<[String; 8]> = parent_tabs
                    .children
                    .iter()
                    .map(|child| {
                        self.tree
                            .tiles
                            .get(*child)
                            .map(|tile| format!("{tile:?}"))
                            .unwrap_or_else(|| "Missing".to_string())
                    })
                    .collect();
                let all_panes = parent_tabs
                    .children
                    .iter()
                    .all(|child| matches!(self.tree.tiles.get(*child), Some(Tile::Pane(_))));
                if all_panes || parent_tabs.children.is_empty() {
                    let pane_id = self.tree.tiles.insert_new(Tile::Pane(pane));
                    if let Some(Tile::Container(Container::Tabs(parent_tabs_mut))) =
                        self.tree.tiles.get_mut(parent_tabs_id)
                    {
                        parent_tabs_mut.add_child(pane_id);
                        if active || parent_tabs_mut.active.is_none() {
                            parent_tabs_mut.set_active(pane_id);
                        }
                    }
                    self.update_has_non_sidebar_for(pane_id);
                    info!(
                        target: "tabs.insert_pane",
                        ?window_entity,
                        ?parent_tabs_id,
                        ?pane_id,
                        children_before = child_kinds.len(),
                        child_kinds = ?child_kinds,
                        pane_kind = "new_tab_direct",
                        "inserted pane as sibling tab"
                    );
                    return Some(pane_id);
                }
            }

            let inner_tabs_id = self.tree.tiles.insert_new(Tile::Container(Container::Tabs(
                egui_tiles::Tabs::new(vec![]),
            )));
            self.set_container_title(inner_tabs_id, "New tab");

            let Some(Tile::Container(Container::Tabs(parent_tabs))) =
                self.tree.tiles.get_mut(parent_tabs_id)
            else {
                warn!(
                    target: "tabs.insert_pane",
                    ?window_entity,
                    ?parent_tabs_id,
                    "new_tab: parent is not Tabs"
                );
                return None;
            };

            parent_tabs.add_child(inner_tabs_id);
            if active || parent_tabs.active.is_none() {
                parent_tabs.set_active(inner_tabs_id);
            }

            let pane_id = self
                .insert_tile(Tile::Pane(pane), Some(inner_tabs_id), true)
                .or_else(|| {
                    warn!(
                        target: "tabs.insert_pane",
                        ?window_entity,
                        ?parent_tabs_id,
                        ?inner_tabs_id,
                        "new_tab: failed to insert pane into inner tabs"
                    );
                    None
                })?;

            info!(
                target: "tabs.insert_pane",
                ?window_entity,
                ?parent_tabs_id,
                ?inner_tabs_id,
                ?pane_id,
                pane_kind = "new_tab",
                "inserted pane into freshly created tab"
            );
            return Some(pane_id);
        }

        let target_tabs = match self.tree.tiles.get(parent_tabs_id) {
            Some(Tile::Container(Container::Tabs(tabs)))
                if tabs
                    .children
                    .iter()
                    .all(|c| matches!(self.tree.tiles.get(*c), Some(Tile::Pane(_)))) =>
            {
                if let Some(active_child) = tabs.active.or_else(|| tabs.children.first().copied()) {
                    let pane_id = self.tree.tiles.insert_new(Tile::Pane(pane));
                    let mut linear = egui_tiles::Linear::new(
                        egui_tiles::LinearDir::Vertical,
                        vec![active_child, pane_id],
                    );
                    linear.shares.set_share(active_child, 1.0);
                    linear.shares.set_share(pane_id, 1.0);
                    let linear_id = self
                        .tree
                        .tiles
                        .insert_new(Tile::Container(Container::Linear(linear)));

                    if let Some(Tile::Container(Container::Tabs(tabs_mut))) =
                        self.tree.tiles.get_mut(parent_tabs_id)
                    {
                        for child in &mut tabs_mut.children {
                            if *child == active_child {
                                *child = linear_id;
                            }
                        }
                        if tabs_mut.active == Some(active_child) {
                            tabs_mut.active = Some(linear_id);
                        }
                    }
                    if active {
                        self.tree.make_active(|id, _| id == pane_id);
                    }
                    self.update_has_non_sidebar_for(pane_id);
                    return Some(pane_id);
                }
                Some(parent_tabs_id)
            }
            Some(Tile::Container(Container::Tabs(tabs))) => {
                if let Some(active_child) = tabs.active.or_else(|| tabs.children.first().copied()) {
                    match self.tree.tiles.get(active_child) {
                        Some(Tile::Container(Container::Tabs(_))) => {
                            info!(
                                target: "tabs.insert_pane",
                                ?window_entity,
                                ?parent_tabs_id,
                                active_child = ?active_child,
                                pane = %match pane {
                                    Pane::Viewport(_) => "Viewport",
                                    Pane::Graph(_) => "Graph",
                                    Pane::Monitor(_) => "Monitor",
                                    Pane::QueryTable(_) => "QueryTable",
                                    Pane::QueryPlot(_) => "QueryPlot",
                                    Pane::ActionTile(_) => "ActionTile",
                                    Pane::VideoStream(_) => "VideoStream",
                                    Pane::Dashboard(_) => "Dashboard",
                                    Pane::Hierarchy => "Hierarchy",
                                    Pane::Inspector => "Inspector",
                                    Pane::SchematicTree(_) => "SchematicTree",
                                },
                                "insert_pane_in_tabs: descending into nested Tabs"
                            );
                            return self.insert_pane_in_tabs(
                                pane,
                                Some(active_child),
                                active,
                                window_entity,
                                new_tab,
                            );
                        }
                        Some(Tile::Container(Container::Linear(_))) => {
                            info!(
                                target: "tabs.insert_pane",
                                ?window_entity,
                                ?parent_tabs_id,
                                active_child = ?active_child,
                                pane = %match pane {
                                    Pane::Viewport(_) => "Viewport",
                                    Pane::Graph(_) => "Graph",
                                    Pane::Monitor(_) => "Monitor",
                                    Pane::QueryTable(_) => "QueryTable",
                                    Pane::QueryPlot(_) => "QueryPlot",
                                    Pane::ActionTile(_) => "ActionTile",
                                    Pane::VideoStream(_) => "VideoStream",
                                    Pane::Dashboard(_) => "Dashboard",
                                    Pane::Hierarchy => "Hierarchy",
                                    Pane::Inspector => "Inspector",
                                    Pane::SchematicTree(_) => "SchematicTree",
                                },
                                "insert_pane_in_tabs: append to active Linear"
                            );
                            let pane_id = self.tree.tiles.insert_new(Tile::Pane(pane));
                            if let Some(Tile::Container(Container::Linear(linear_mut))) =
                                self.tree.tiles.get_mut(active_child)
                            {
                                linear_mut.dir = egui_tiles::LinearDir::Vertical;
                                linear_mut.add_child(pane_id);
                                linear_mut.shares.set_share(pane_id, 1.0);
                            }
                            if active {
                                self.tree.make_active(|id, _| id == pane_id);
                            }
                            self.update_has_non_sidebar_for(pane_id);
                            return Some(pane_id);
                        }
                        Some(Tile::Pane(_)) => {
                            info!(
                                target: "tabs.insert_pane",
                                ?window_entity,
                                ?parent_tabs_id,
                                active_child = ?active_child,
                                pane = %match pane {
                                    Pane::Viewport(_) => "Viewport",
                                    Pane::Graph(_) => "Graph",
                                    Pane::Monitor(_) => "Monitor",
                                    Pane::QueryTable(_) => "QueryTable",
                                    Pane::QueryPlot(_) => "QueryPlot",
                                    Pane::ActionTile(_) => "ActionTile",
                                    Pane::VideoStream(_) => "VideoStream",
                                    Pane::Dashboard(_) => "Dashboard",
                                    Pane::Hierarchy => "Hierarchy",
                                    Pane::Inspector => "Inspector",
                                    Pane::SchematicTree(_) => "SchematicTree",
                                },
                                "insert_pane_in_tabs: wrap active pane into Linear with new pane"
                            );
                            let pane_id = self.tree.tiles.insert_new(Tile::Pane(pane));
                            let mut linear = egui_tiles::Linear::new(
                                egui_tiles::LinearDir::Vertical,
                                vec![active_child, pane_id],
                            );
                            linear.shares.set_share(active_child, 1.0);
                            linear.shares.set_share(pane_id, 1.0);
                            let linear_id = self
                                .tree
                                .tiles
                                .insert_new(Tile::Container(Container::Linear(linear)));

                            if let Some(Tile::Container(Container::Tabs(tabs_mut))) =
                                self.tree.tiles.get_mut(parent_tabs_id)
                            {
                                if let Some(child) =
                                    tabs_mut.children.iter_mut().find(|c| **c == active_child)
                                {
                                    *child = linear_id;
                                }
                                if tabs_mut.active == Some(active_child) {
                                    tabs_mut.active = Some(linear_id);
                                }
                            }
                            if active {
                                self.tree.make_active(|id, _| id == pane_id);
                            }
                            self.update_has_non_sidebar_for(pane_id);
                            return Some(pane_id);
                        }
                        _ => {}
                    }
                }
                let inner_tabs = self.tree.tiles.insert_new(Tile::Container(Container::Tabs(
                    egui_tiles::Tabs::new(vec![]),
                )));
                self.set_container_title(inner_tabs, "New tab");

                let mut linear =
                    egui_tiles::Linear::new(egui_tiles::LinearDir::Horizontal, vec![inner_tabs]);
                linear.shares.set_share(inner_tabs, 1.0);
                let linear_id = self
                    .tree
                    .tiles
                    .insert_new(Tile::Container(Container::Linear(linear)));
                self.set_container_title(linear_id, "New tab");

                if let Some(Tile::Container(Container::Tabs(parent_tabs_mut))) =
                    self.tree.tiles.get_mut(parent_tabs_id)
                {
                    parent_tabs_mut.add_child(linear_id);
                    if active || parent_tabs_mut.active.is_none() {
                        parent_tabs_mut.set_active(linear_id);
                    }
                }

                Some(inner_tabs)
            }
            other => {
                warn!(
                    target: "tabs.insert_pane",
                    ?window_entity,
                    ?parent_tabs_id,
                    tile_kind = ?other.map(|t| format!("{t:?}")),
                    "insert_pane_in_tabs: parent is not Tabs"
                );
                None
            }
        }?;

        let inserted = self.insert_tile(Tile::Pane(pane), Some(target_tabs), active);
        if let Some(tile_id) = inserted {
            info!(
                target: "tabs.insert_pane",
                ?window_entity,
                ?parent_tabs_id,
                ?target_tabs,
                ?tile_id,
                pane_kind = match self.tree.tiles.get(tile_id) {
                    Some(Tile::Pane(p)) => match p {
                        Pane::Viewport(_) => "Viewport",
                        Pane::Graph(_) => "Graph",
                        Pane::Monitor(_) => "Monitor",
                        Pane::QueryTable(_) => "QueryTable",
                        Pane::QueryPlot(_) => "QueryPlot",
                        Pane::ActionTile(_) => "ActionTile",
                        Pane::VideoStream(_) => "VideoStream",
                        Pane::Dashboard(_) => "Dashboard",
                        Pane::Hierarchy => "Hierarchy",
                        Pane::Inspector => "Inspector",
                        Pane::SchematicTree(_) => "SchematicTree",
                    },
                    _ => "Unknown",
                },
                "inserted pane into tabs"
            );
        }
        inserted
    }
}

impl Default for TileState {
    fn default() -> Self {
        Self::new(Id::new("main_tab_tree"))
    }
}
#[derive(SystemParam)]
pub struct TileSystem<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    window_states: Query<'w, 's, (Entity, &'static WindowId, &'static WindowState)>,
    primary_window: Single<'w, Entity, With<PrimaryWindow>>,
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
        let (_window_entity, _window_id, window_state) =
            params.window_states.get(target_id).ok()?;
        let pending_non_sidebar = window_state
            .tile_state
            .tree_actions
            .iter()
            .any(|action| !matches!(action, TreeAction::AddSidebars));
        let is_empty_tile_tree =
            !pending_non_sidebar && !window_state.tile_state.has_non_sidebar_content();

        let icons = TileIcons {
            add: contexts.add_image(images.icon_add.clone_weak()),
            close: contexts.add_image(images.icon_close.clone_weak()),
            scrub: contexts.add_image(images.icon_scrub.clone_weak()),
            tile_3d_viewer: contexts.add_image(images.icon_tile_3d_viewer.clone_weak()),
            tile_graph: contexts.add_image(images.icon_tile_graph.clone_weak()),
            subtract: contexts.add_image(images.icon_subtract.clone_weak()),
            chart: contexts.add_image(images.icon_chart.clone_weak()),
            setting: contexts.add_image(images.icon_setting.clone_weak()),
            search: contexts.add_image(images.icon_search.clone_weak()),
            chevron: contexts.add_image(images.icon_chevron_right.clone_weak()),
            plot: contexts.add_image(images.icon_plot.clone_weak()),
            viewport: contexts.add_image(images.icon_viewport.clone_weak()),
            container: contexts.add_image(images.icon_container.clone_weak()),
            entity: contexts.add_image(images.icon_entity.clone_weak()),
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
        if is_empty_tile_tree && !read_only {
            ui.add_widget_with::<TileLayoutEmpty>(
                world,
                "tile_layout_empty",
                TileLayoutEmptyArgs {
                    icons: icons.clone(),
                    window: target,
                },
            );
            return;
        }

        ui.add_widget_with::<TileLayout>(
            world,
            "tile_layout",
            TileLayoutArgs {
                icons,
                window: target,
                read_only,
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

        let frame = Frame {
            fill: fill_color,
            inner_margin: egui::Margin {
                top: 32,
                ..Default::default()
            },
            ..Default::default()
        };

        egui::CentralPanel::default()
            .frame(frame)
            .show_inside(ui, |ui| {
                ui.set_clip_rect(ui.max_rect());
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

        let frame = Frame {
            fill: fill_color,
            inner_margin: egui::Margin {
                top: 32,
                ..Default::default()
            },
            ..Default::default()
        };

        let central = egui::CentralPanel::default().frame(frame);

        central.show(ctx, |ui| {
            ui.set_clip_rect(ui.max_rect());
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
pub struct TileLayoutEmpty<'w> {
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
}

#[derive(Clone)]
pub struct TileLayoutEmptyArgs {
    pub icons: TileIcons,
    pub window: Option<Entity>,
}

impl WidgetSystem for TileLayoutEmpty<'_> {
    type Args = TileLayoutEmptyArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let mut state_mut = state.get_mut(world);

        let TileLayoutEmptyArgs { icons, window } = args;

        let button_height = 160.0;
        let button_width = 240.0;
        let button_spacing = 20.0;

        let desired_size = egui::vec2(button_width * 3.0 + button_spacing, button_height);

        ui.allocate_new_ui(
            UiBuilder::new().max_rect(egui::Rect::from_center_size(
                ui.max_rect().center(),
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
                            .open_item(palette_items::create_viewport(None, false));
                        state_mut.cmd_palette_state.target_window = window;
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
                            .open_item(palette_items::create_graph(None, false));
                        state_mut.cmd_palette_state.target_window = window;
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
                            .open_item(palette_items::create_monitor(None, false));
                        state_mut.cmd_palette_state.target_window = window;
                    }
                });
            },
        );
    }
}

#[derive(SystemParam)]
pub struct TileLayout<'w, 's> {
    commands: Commands<'w, 's>,
    selected_object: ResMut<'w, SelectedObject>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    viewport_contains_pointer: ResMut<'w, ViewportContainsPointer>,
    editor_cam: Query<'w, 's, &'static mut EditorCam, With<MainCamera>>,
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
    eql_ctx: Res<'w, EqlContext>,
    node_updater_params: NodeUpdaterParams<'w, 's>,
    graphs: Query<'w, 's, &'static mut GraphState>,
    query_plots: Query<'w, 's, &'static mut QueryPlotData>,
    action_tiles: Query<'w, 's, &'static mut ActionTile>,
    tile_param: crate::ui::command_palette::palette_items::TileParam<'w, 's>,
}

#[derive(Clone)]
pub struct TileLayoutArgs {
    pub icons: TileIcons,
    pub window: Option<Entity>,
    pub read_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SidebarKind {
    Hierarchy,
    Inspector,
}

#[derive(Clone)]
struct UiFrameState {
    tree: egui_tiles::Tree<Pane>,
    container_titles: HashMap<TileId, String>,
    mask_state: SidebarMaskState,
    tree_actions: SmallVec<[TreeAction; 4]>,
}

pub(super) type ShareUpdate = (TileId, TileId, TileId, f32, f32);

// Pull the current tiles tree and mask flags out of ECS so the frame can mutate them.
fn take_ui_frame_state(
    world: &mut World,
    state: &mut SystemState<TileLayout>,
    window: Option<Entity>,
) -> Option<UiFrameState> {
    let result = {
        let mut state_mut = state.get_mut(world);
        let mut ui_state = state_mut.tile_param.target(window)?;
        let empty_tree = egui_tiles::Tree::empty(ui_state.tree_id);
        UiFrameState {
            tree_actions: std::mem::take(&mut ui_state.tree_actions),
            container_titles: ui_state.container_titles.clone(),
            tree: std::mem::replace(&mut ui_state.tree, empty_tree),
            mask_state: SidebarMaskState {
                hierarchy_masked: ui_state.hierarchy_masked,
                inspector_masked: ui_state.inspector_masked,
                last_hierarchy_share: ui_state.last_hierarchy_share,
                last_inspector_share: ui_state.last_inspector_share,
            },
        }
    };
    Some(result)
}

// Ensure the center lane of the root linear container is always tabs for consistent UX.
fn ensure_center_tabs(tree: &mut egui_tiles::Tree<Pane>) {
    if let Some(root_id) = tree.root() {
        if let Some(Tile::Container(Container::Linear(linear))) = tree.tiles.get(root_id) {
            if linear.children.len() == 3 {
                let center_id = linear.children[1];
                let is_tabs = matches!(
                    tree.tiles.get(center_id),
                    Some(Tile::Container(Container::Tabs(_)))
                );
                if !is_tabs {
                    let center_share = linear.shares[center_id];
                    let mut tabs = egui_tiles::Tabs::new(vec![center_id]);
                    tabs.set_active(center_id);
                    let tabs_id = tree
                        .tiles
                        .insert_new(Tile::Container(Container::Tabs(tabs)));
                    if let Some(Tile::Container(Container::Linear(linear_mut))) =
                        tree.tiles.get_mut(root_id)
                    {
                        linear_mut.children[1] = tabs_id;
                        linear_mut.shares.set_share(tabs_id, center_share);
                    }
                }
            }
        }
    }
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
        } = args;

        let Some(mut frame_state) = take_ui_frame_state(world, state, window) else {
            return;
        };

        ensure_center_tabs(&mut frame_state.tree);
        let (mut tree_actions, share_updates) = render_tree_and_collect_updates(
            world,
            ui,
            &mut frame_state.tree,
            icons,
            frame_state.container_titles,
            read_only,
            window,
            &mut frame_state.mask_state,
            frame_state.tree_actions,
        );

        let mut state_mut = state.get_mut(world);
        let Some(mut ui_state) = state_mut.tile_param.target(window) else {
            return;
        };
        let _ = std::mem::replace(&mut ui_state.tree, frame_state.tree);
        apply_share_updates(&mut ui_state.tree, &share_updates);
        ui_state.hierarchy_masked = frame_state.mask_state.hierarchy_masked;
        ui_state.inspector_masked = frame_state.mask_state.inspector_masked;
        ui_state.last_hierarchy_share = frame_state.mask_state.last_hierarchy_share;
        ui_state.last_inspector_share = frame_state.mask_state.last_inspector_share;
        state_mut.viewport_contains_pointer.0 = ui.ui_contains_pointer();

        for mut editor_cam in state_mut.editor_cam.iter_mut() {
            editor_cam.enabled_motion = EnabledMotion {
                pan: state_mut.viewport_contains_pointer.0,
                orbit: state_mut.viewport_contains_pointer.0,
                zoom: state_mut.viewport_contains_pointer.0,
            }
        }

        {
            let mut action_ctx = ActionContext {
                ui_state: &mut ui_state,
                commands: &mut state_mut.commands,
                selected_object: &mut state_mut.selected_object,
                asset_server: &state_mut.asset_server,
                meshes: &mut state_mut.meshes,
                materials: &mut state_mut.materials,
                render_layer_alloc: &mut state_mut.render_layer_alloc,
                eql_ctx: &state_mut.eql_ctx,
                node_updater_params: &state_mut.node_updater_params,
                graphs: &mut state_mut.graphs,
                query_plots: &mut state_mut.query_plots,
                action_tiles: &mut state_mut.action_tiles,
                window,
                read_only,
            };
            for diff in tree_actions.drain(..) {
                action_ctx.handle(diff);
            }
        }

        let tiles = ui_state.tree.tiles.iter();
        let active_tiles = ui_state.tree.active_tiles();
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
                    } else if let Ok(mut cam) = state_mut.commands.get_entity(query_plot.entity) {
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
            }
        }
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
