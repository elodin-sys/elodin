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
    egui::{self, Color32, CornerRadius, Frame, Id, RichText, Stroke, Ui, Visuals, vec2},
};
use bevy_render::{
    camera::{Exposure, PhysicalCameraParameters},
    view::RenderLayers,
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
    SelectedObject, ViewportRect,
    actions::ActionTileWidget,
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

impl TileState {
    fn recompute_has_non_sidebar(&mut self) {
        fn visit(tree: &egui_tiles::Tree<Pane>, id: TileId) -> bool {
            match tree.tiles.get(id) {
                Some(Tile::Pane(Pane::Hierarchy | Pane::Inspector)) => false,
                Some(Tile::Pane(_)) => true,
                Some(Tile::Container(container)) => {
                    container.children().any(|child| visit(tree, *child))
                }
                None => false,
            }
        }
        self.has_non_sidebar = self
            .tree
            .root()
            .map(|root| visit(&self.tree, root))
            .unwrap_or(false);
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

            if active && let Container::Tabs(tabs) = container {
                tabs.set_active(tile_id);
            }
            is_tabs
        };

        if is_tabs_container
            && !self.container_titles.contains_key(&parent_id)
            && let Some(title) = tabs_title_hint
        {
            self.set_container_title(parent_id, title);
        }

        if matches!(
            self.tree.tiles.get(tile_id),
            Some(Tile::Pane(
                Pane::Viewport(_)
                    | Pane::Graph(_)
                    | Pane::Monitor(_)
                    | Pane::QueryTable(_)
                    | Pane::QueryPlot(_)
                    | Pane::ActionTile(_)
                    | Pane::VideoStream(_)
                    | Pane::Dashboard(_)
                    | Pane::SchematicTree(_)
            ))
        ) {
            self.has_non_sidebar = true;
        }

        Some(tile_id)
    }

    fn default_parent_center(&mut self) -> Option<TileId> {
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
        label: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddVideoStream(tile_id, msg_id, label));
    }

    pub fn create_dashboard_tile(
        &mut self,
        dashboard: impeller2_wkt::Dashboard,
        label: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions.push(TreeAction::AddDashboard(
            tile_id,
            Box::new(dashboard),
            label,
        ));
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

        if let Some(root_id) = self.tree.root()
            && let Some(Tile::Container(root)) = self.tree.tiles.get_mut(root_id)
        {
            root.retain(|_| false);
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

#[derive(Clone, Debug)]
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
            Pane::Viewport(viewport) => Self::fallback_label(&viewport.label, "viewport"),
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
                hdr: viewport.hdr,
                clear_color: ClearColorConfig::Default,
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
            label,
            grid_layer,
            viewport_layer,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphPane {
    pub id: Entity,
    pub label: String,
    pub rect: Option<egui::Rect>,
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
}

impl Default for TileState {
    fn default() -> Self {
        Self::new(Id::new("main_tab_tree"))
    }
}

struct TreeBehavior<'w> {
    icons: TileIcons,
    tree_actions: SmallVec<[TreeAction; 4]>,
    world: &'w mut World,
    container_titles: HashMap<TileId, String>,
    read_only: bool,
    target_window: Option<Entity>,
}

#[derive(Clone)]
pub enum TreeAction {
    AddViewport(Option<TileId>),
    AddGraph(Option<TileId>, Box<Option<GraphBundle>>),
    AddMonitor(Option<TileId>, String),
    AddQueryTable(Option<TileId>),
    AddQueryPlot(Option<TileId>),
    AddActionTile(Option<TileId>, String, String),
    AddVideoStream(Option<TileId>, [u8; 2], String),
    AddDashboard(Option<TileId>, Box<impeller2_wkt::Dashboard>, String),
    AddSidebars,
    DeleteTab(TileId),
    SelectTile(TileId),
    RenameContainer(TileId, String),
}

enum TabState {
    Selected,
    Inactive,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TabRole {
    Super,
    Normal,
}

impl<'w> TreeBehavior<'w> {
    fn tab_role(&self, tiles: &Tiles<Pane>, tile_id: TileId) -> TabRole {
        // Hide chrome for the static sidebars (Hierarchy/Inspector).
        if let Some(Tile::Pane(pane)) = tiles.get(tile_id) {
            return match pane {
                Pane::Hierarchy | Pane::Inspector => TabRole::Super,
                _ => TabRole::Normal,
            };
        }
        TabRole::Normal
    }
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
        // Hide sidebar content completely when collapsed to a thin slice.
        if matches!(pane, Pane::Hierarchy | Pane::Inspector) && ui.available_size().x <= 20.0 {
            let size = ui.available_size();
            ui.allocate_space(size);
            return egui_tiles::UiResponse::None;
        }
        pane.ui(ui, &self.icons, self.world, &mut self.tree_actions)
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
        let tab_role = self.tab_role(tiles, tile_id);
        let _clip = ui.clip_rect();
        let _avail = ui.available_rect_before_wrap();
        if matches!(tab_role, TabRole::Super) {
            // Hide sidebar tabs completely: no title, no background, no "+" chrome.
            let (_, rect) = ui.allocate_space(vec2(0.0, ui.available_height()));
            return ui.interact(rect, id, egui::Sense::hover());
        }
        let hide_title = false;
        let show_close = true;

        let tab_state = if state.active {
            TabState::Selected
        } else {
            TabState::Inactive
        };

        let persist_id = id.with(("rename_title", tile_id));
        let edit_flag_id = id.with(("rename_editing", tile_id));
        let edit_buf_id = id.with(("rename_buffer", tile_id));
        let mut is_editing = ui
            .ctx()
            .data(|d| d.get_temp::<bool>(edit_flag_id))
            .unwrap_or(false);

        let title_str: String =
            if let Some(custom) = ui.ctx().data(|d| d.get_temp::<String>(persist_id)) {
                custom
            } else if let Some(t) = self.container_titles.get(&tile_id) {
                t.clone()
            } else {
                match tiles.get(tile_id) {
                    Some(egui_tiles::Tile::Container(c)) => format!("{:?}", c.kind()),
                    _ => self.tab_title_for_tile(tiles, tile_id).text().to_string(),
                }
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
        let tab_width = if hide_title {
            0.0
        } else {
            galley.size().x + x_margin * 4.0
        };
        let (_, rect) = ui.allocate_space(vec2(tab_width, ui.available_height()));
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

        if !self.read_only
            && !hide_title
            && state.active
            && response.double_clicked()
            && !is_editing
        {
            ui.ctx()
                .data_mut(|d| d.insert_temp(edit_buf_id, title_str.clone()));
            ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, true));
            is_editing = true;
        }

        if ui.is_rect_visible(rect) && !state.is_being_dragged {
            let scheme = get_scheme();
            let bg_color = match tab_state {
                TabState::Selected => scheme.text_primary,
                TabState::Inactive => Color32::from_rgb(0, 0, 0),
            };

            let text_color = match tab_state {
                TabState::Selected => scheme.bg_secondary,
                TabState::Inactive => Color32::from_rgb(230, 230, 230),
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
                    let new_title = buf.trim().to_owned();

                    ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, false));
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(edit_buf_id, new_title.clone()));
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(persist_id, new_title.clone()));
                    ui.memory_mut(|m| m.surrender_focus(resp.id));

                    if !self.read_only {
                        self.tree_actions
                            .push(TreeAction::RenameContainer(tile_id, new_title.clone()));
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

            if show_close {
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
        Color32::from_rgb(0, 0, 0)
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            // Keep tab bars visible (titles and "+") even with a single tab.
            prune_empty_tabs: false,
            prune_single_child_tabs: false,
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
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
            let text = if let Some(t) = self.container_titles.get(&tile_id) {
                egui::WidgetText::from(t.clone())
            } else {
                self.tab_title_for_tile(tiles, tile_id)
            };
            let text = text.text();
            ui.label(
                RichText::new(text)
                    .color(get_scheme().bg_secondary)
                    .size(11.0),
            );
        });
    }

    fn resize_stroke(
        &self,
        _style: &egui::Style,
        _resize_state: egui_tiles::ResizeState,
    ) -> Stroke {
        Stroke::NONE
    }

    fn top_bar_right_ui(
        &mut self,
        tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.read_only {
            return;
        }

        let is_sidebar_tabs =
            if let Some(Tile::Container(Container::Tabs(tabs))) = tiles.get(tile_id) {
                tabs.children.iter().all(|child| {
                    matches!(
                        tiles.get(*child),
                        Some(Tile::Pane(Pane::Hierarchy | Pane::Inspector))
                    )
                })
            } else {
                false
            };
        if is_sidebar_tabs {
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
                .open_page(move || palette_items::create_tiles(tile_id));
            layout.cmd_palette_state.target_window = self.target_window;
        }
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

fn unmask_sidebar_on_select(ui_state: &mut TileState, sidebar_tile: TileId, kind: SidebarKind) {
    let target_frac = 0.2;
    let min_share = 0.01;
    fn contains_child(tiles: &Tiles<Pane>, haystack: TileId, needle: TileId) -> bool {
        if haystack == needle {
            return true;
        }
        match tiles.get(haystack) {
            Some(Tile::Container(container)) => container
                .children()
                .any(|child| contains_child(tiles, *child, needle)),
            _ => false,
        }
    }

    let mut targets: Vec<(TileId, TileId)> = Vec::new();
    for (cid, tile) in ui_state.tree.tiles.iter() {
        if let Tile::Container(Container::Linear(linear)) = tile
            && let Some(child) = linear
                .children
                .iter()
                .find(|child| contains_child(&ui_state.tree.tiles, **child, sidebar_tile))
        {
            targets.push((*cid, *child));
        }
    }

    for (cid, target_child) in targets {
        if let Some(Tile::Container(Container::Linear(linear))) = ui_state.tree.tiles.get_mut(cid) {
            let share_sum: f32 = linear.shares.iter().map(|(_, s)| s).sum::<f32>().max(0.01);
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
    }

    match kind {
        SidebarKind::Hierarchy => {
            ui_state.hierarchy_masked = false;
            ui_state.last_hierarchy_share = Some(target_frac);
        }
        SidebarKind::Inspector => {
            ui_state.inspector_masked = false;
            ui_state.last_inspector_share = Some(target_frac);
        }
    }
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
                            .open_item(palette_items::create_viewport(None));
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
                            .open_item(palette_items::create_graph(None));
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
                            .open_item(palette_items::create_monitor(None));
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

#[derive(Clone, Copy, Debug)]
struct SidebarMaskState {
    hierarchy_masked: bool,
    inspector_masked: bool,
    last_hierarchy_share: Option<f32>,
    last_inspector_share: Option<f32>,
}

impl SidebarMaskState {
    fn masked(&self, kind: SidebarKind) -> bool {
        match kind {
            SidebarKind::Hierarchy => self.hierarchy_masked,
            SidebarKind::Inspector => self.inspector_masked,
        }
    }

    fn set_masked(&mut self, kind: SidebarKind, masked: bool) {
        match kind {
            SidebarKind::Hierarchy => self.hierarchy_masked = masked,
            SidebarKind::Inspector => self.inspector_masked = masked,
        }
    }

    fn last_share(&self, kind: SidebarKind) -> Option<f32> {
        match kind {
            SidebarKind::Hierarchy => self.last_hierarchy_share,
            SidebarKind::Inspector => self.last_inspector_share,
        }
    }

    fn set_last_share(&mut self, kind: SidebarKind, share: Option<f32>) {
        match kind {
            SidebarKind::Hierarchy => self.last_hierarchy_share = share,
            SidebarKind::Inspector => self.last_inspector_share = share,
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

        let (tree, mut tree_actions, share_updates, mask_state) = {
            let (tab_diffs, container_titles, mut tree, mut mask_state) = {
                let mut state_mut = state.get_mut(world);
                let Some(mut ui_state) = state_mut.tile_param.target(window) else {
                    return;
                };
                let log_dump = !ui_state.tree_actions.is_empty();
                let _ = log_dump;
                let empty_tree = egui_tiles::Tree::empty(ui_state.tree_id);
                (
                    std::mem::take(&mut ui_state.tree_actions),
                    ui_state.container_titles.clone(),
                    std::mem::replace(&mut ui_state.tree, empty_tree),
                    SidebarMaskState {
                        hierarchy_masked: ui_state.hierarchy_masked,
                        inspector_masked: ui_state.inspector_masked,
                        last_hierarchy_share: ui_state.last_hierarchy_share,
                        last_inspector_share: ui_state.last_inspector_share,
                    },
                )
            };
            let mut behavior = TreeBehavior {
                icons,
                // This world here makes getting ui_state difficult.
                world,
                tree_actions: tab_diffs,
                container_titles,
                read_only,
                target_window: window,
            };
            let _logged_diag = ui
                .ctx()
                .data(|d| d.get_temp::<bool>(egui::Id::new(("center_tabs_diag", window))))
                .unwrap_or(false);
            let _window_id = window.and_then(|w| behavior.world.get::<WindowId>(w));
            let _max_rect = ui.max_rect();
            let _clip = ui.clip_rect();
            if let Some(root_id) = tree.root()
                && let Some(Tile::Container(Container::Linear(linear))) = tree.tiles.get(root_id)
                && linear.children.len() == 3
            {
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
            tree.ui(&mut behavior, ui);
            let window_width = ui.ctx().screen_rect().width();
            let gutter_width: f32 = (window_width * 0.02).max(12.0);
            let painter = ui.painter_at(ui.max_rect());
            let mut share_updates: Vec<(TileId, TileId, TileId, f32, f32)> = Vec::new();

            fn sidebar_kind(tiles: &Tiles<Pane>, id: TileId) -> Option<SidebarKind> {
                match tiles.get(id) {
                    Some(Tile::Pane(Pane::Hierarchy)) => Some(SidebarKind::Hierarchy),
                    Some(Tile::Pane(Pane::Inspector)) => Some(SidebarKind::Inspector),
                    Some(Tile::Container(Container::Tabs(tabs))) => tabs
                        .children
                        .iter()
                        .find_map(|child| sidebar_kind(tiles, *child)),
                    Some(Tile::Container(Container::Linear(linear))) => linear
                        .children
                        .iter()
                        .find_map(|child| sidebar_kind(tiles, *child)),
                    Some(Tile::Container(Container::Grid(grid))) => grid
                        .children()
                        .find_map(|child| sidebar_kind(tiles, *child)),
                    _ => None,
                }
            }

            let linear_ids: Vec<_> = tree
                .tiles
                .iter()
                .filter_map(|(id, tile)| {
                    matches!(tile, Tile::Container(Container::Linear(_))).then_some(*id)
                })
                .collect();

            for container_id in linear_ids {
                let Some(parent_rect) = tree.tiles.rect(container_id) else {
                    continue;
                };

                let visible_children: Vec<TileId> = tree
                    .tiles
                    .get(container_id)
                    .and_then(|tile| match tile {
                        Tile::Container(Container::Linear(linear)) => Some(
                            linear
                                .children
                                .iter()
                                .copied()
                                .filter(|child| tree.tiles.is_visible(*child))
                                .collect(),
                        ),
                        _ => None,
                    })
                    .unwrap_or_default();

                if visible_children.len() < 2 {
                    continue;
                }

                let mut child_data = Vec::new();
                for &child in &visible_children {
                    if let Some(rect) = tree.tiles.rect(child) {
                        let sidebar_kind = sidebar_kind(&tree.tiles, child);
                        child_data.push((child, rect, sidebar_kind));
                    }
                }

                if child_data.len() < 2 {
                    continue;
                }

                for i in 0..child_data.len().saturating_sub(1) {
                    let (left_id, left_rect, left_kind) = child_data[i];
                    let (right_id, right_rect, right_kind) = child_data[i + 1];
                    let left_sidebar = left_kind.is_some();
                    let right_sidebar = right_kind.is_some();
                    if left_sidebar == right_sidebar {
                        continue;
                    }
                    let sidebar_kind = left_kind.or(right_kind).unwrap();
                    let sidebar_on_left = left_sidebar;
                    let pair_width = left_rect.width() + right_rect.width();
                    let Some((share_left, share_right)) =
                        tree.tiles.get(container_id).and_then(|tile| match tile {
                            Tile::Container(Container::Linear(linear)) => {
                                Some((linear.shares[left_id], linear.shares[right_id]))
                            }
                            _ => None,
                        })
                    else {
                        continue;
                    };
                    let pair_sum = share_left + share_right;
                    if pair_sum <= 0.0 {
                        continue;
                    }
                    let share_per_px = if pair_width > 0.0 {
                        pair_sum / pair_width
                    } else {
                        0.0
                    };
                    let min_sidebar_px = gutter_width;
                    let min_other_px = 32.0;
                    let min_sidebar_share = if share_per_px > 0.0 {
                        min_sidebar_px * share_per_px
                    } else {
                        pair_sum * 0.05
                    };
                    let min_other_share = if share_per_px > 0.0 {
                        min_other_px * share_per_px
                    } else {
                        pair_sum * 0.05
                    };

                    let gap = right_rect.min.x - left_rect.max.x;
                    let mut center_x = (left_rect.max.x + right_rect.min.x) * 0.5;
                    let local_width = gutter_width;
                    if gap < gutter_width {
                        let offset = (gutter_width - gap).max(0.0) * 0.5;
                        if left_sidebar {
                            center_x -= offset;
                        } else {
                            center_x += offset;
                        }
                    }
                    let half = local_width * 0.5;
                    center_x = center_x
                        .max(parent_rect.left() + half)
                        .min(parent_rect.right() - half);
                    let gutter_rect = egui::Rect::from_min_max(
                        egui::pos2(center_x - half, parent_rect.top()),
                        egui::pos2(center_x + half, parent_rect.bottom()),
                    );

                    let fill = Color32::from_rgb(0, 0, 0);
                    let stroke = Stroke::new(1.0, Color32::from_rgb(0, 0, 0));
                    painter.rect_filled(gutter_rect, 0.0, fill);
                    painter.rect_stroke(gutter_rect, 0.0, stroke, egui::StrokeKind::Inside);

                    let id = ui.id().with(("sidebar_gutter", container_id, i));
                    let hit_rect = gutter_rect;
                    #[derive(Clone, Copy, Default)]
                    struct DragState {
                        left_width: f32,
                        right_width: f32,
                        start_x: f32,
                        active: bool,
                    }
                    let mut drag_state = ui
                        .ctx()
                        .data(|d| d.get_temp::<DragState>(id))
                        .unwrap_or_default();

                    let response = ui
                        .interact(hit_rect, id, egui::Sense::click_and_drag())
                        .on_hover_cursor(egui::CursorIcon::PointingHand);

                    if response.hovered() {
                        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
                    }

                    let mut apply_shares = |left: f32, right: f32| {
                        if let Some(Tile::Container(Container::Linear(linear))) =
                            tree.tiles.get_mut(container_id)
                        {
                            linear.shares.set_share(left_id, left);
                            linear.shares.set_share(right_id, right);
                        }
                        share_updates.push((container_id, left_id, right_id, left, right));
                        ui.ctx().request_repaint();
                    };

                    let mut sidebar_masked = mask_state.masked(sidebar_kind);

                    let click_inside_gutter = response.clicked_by(egui::PointerButton::Primary)
                        && ui
                            .input(|i| i.pointer.interact_pos())
                            .map(|p| gutter_rect.shrink(1.0).contains(p))
                            .unwrap_or(false);

                    if click_inside_gutter {
                        if sidebar_masked {
                            let default_px = (parent_rect.width() * 0.15).max(min_sidebar_px);
                            let restore_share = if share_per_px > 0.0 {
                                default_px * share_per_px
                            } else {
                                pair_sum * 0.15
                            };
                            let max_sidebar_share = (pair_sum - min_other_share).max(0.01);
                            let target_sidebar_share = restore_share
                                .max(min_sidebar_share.max(0.01))
                                .min(max_sidebar_share);
                            let left_share = if sidebar_on_left {
                                target_sidebar_share
                            } else {
                                pair_sum - target_sidebar_share
                            };
                            let right_share = pair_sum - left_share;
                            apply_shares(left_share.max(0.01), right_share.max(0.01));
                            mask_state.set_masked(sidebar_kind, false);
                            sidebar_masked = false;
                        } else {
                            let current_sidebar_share = if sidebar_on_left {
                                share_left
                            } else {
                                share_right
                            };
                            mask_state.set_last_share(sidebar_kind, Some(current_sidebar_share));
                            let max_sidebar_share = (pair_sum - min_other_share).max(0.01);
                            let target_sidebar_share =
                                min_sidebar_share.max(0.01).min(max_sidebar_share);
                            let left_share = if sidebar_on_left {
                                target_sidebar_share
                            } else {
                                pair_sum - target_sidebar_share
                            };
                            let right_share = pair_sum - left_share;
                            apply_shares(left_share.max(0.01), right_share.max(0.01));
                            mask_state.set_masked(sidebar_kind, true);
                            sidebar_masked = true;
                        }
                    }

                    if sidebar_masked {
                        if drag_state.active {
                            ui.ctx().data_mut(|d| d.remove::<DragState>(id));
                        }
                        continue;
                    }

                    let pointer_pos = ui.input(|i| i.pointer.interact_pos());
                    let pointer_down = ui.input(|i| i.pointer.primary_down());
                    if !drag_state.active && pointer_down && response.hovered() {
                        let start_x = pointer_pos.map(|p| p.x).unwrap_or(gutter_rect.center().x);
                        drag_state = DragState {
                            left_width: left_rect.width(),
                            right_width: right_rect.width(),
                            start_x,
                            active: true,
                        };
                        ui.ctx().data_mut(|d| d.insert_temp(id, drag_state));
                    }

                    if drag_state.active && pointer_down {
                        let delta = pointer_pos.map(|p| p.x - drag_state.start_x).unwrap_or(0.0);
                        let min_left = if left_sidebar { 4.0 } else { 32.0 };
                        let min_right = if right_sidebar { 4.0 } else { 32.0 };
                        let new_left = (drag_state.left_width + delta).max(min_left);
                        let new_right = (drag_state.right_width - delta).max(min_right);
                        if let Some(Tile::Container(Container::Linear(linear))) =
                            tree.tiles.get_mut(container_id)
                        {
                            let share_left = linear.shares[left_id];
                            let share_right = linear.shares[right_id];
                            let share_sum = share_left + share_right;
                            let current_sidebar_share = if sidebar_on_left {
                                share_left
                            } else {
                                share_right
                            };

                            let total = new_left + new_right;
                            let new_left_share = share_sum * new_left / total.max(1.0);
                            let new_right_share = share_sum - new_left_share;
                            linear.shares.set_share(left_id, new_left_share.max(0.01));
                            linear.shares.set_share(right_id, new_right_share.max(0.01));
                            share_updates.push((
                                container_id,
                                left_id,
                                right_id,
                                new_left_share.max(0.01),
                                new_right_share.max(0.01),
                            ));
                            ui.ctx().request_repaint();

                            let new_sidebar_share = if sidebar_on_left {
                                new_left_share
                            } else {
                                new_right_share
                            };
                            let mask_threshold = min_sidebar_share * 1.05;
                            if new_sidebar_share <= mask_threshold
                                && !mask_state.masked(sidebar_kind)
                            {
                                let prev_last = mask_state.last_share(sidebar_kind);
                                let should_update_last =
                                    current_sidebar_share > min_sidebar_share * 1.5;
                                let store_share = if should_update_last {
                                    Some(current_sidebar_share)
                                } else {
                                    prev_last
                                };
                                mask_state.set_last_share(sidebar_kind, store_share);
                                mask_state.set_masked(sidebar_kind, true);
                            }
                        }
                    }

                    if drag_state.active && !pointer_down {
                        ui.ctx().data_mut(|d| d.remove::<DragState>(id));
                    }
                }
            }
            let TreeBehavior { tree_actions, .. } = behavior;
            (tree, tree_actions, share_updates, mask_state)
        };

        let mut state_mut = state.get_mut(world);
        let Some(mut ui_state) = state_mut.tile_param.target(window) else {
            return;
        };
        let _ = std::mem::replace(&mut ui_state.tree, tree);
        for (container_id, left_id, right_id, left_share, right_share) in share_updates {
            if let Some(Tile::Container(Container::Linear(linear))) =
                ui_state.tree.tiles.get_mut(container_id)
            {
                linear.shares.set_share(left_id, left_share);
                linear.shares.set_share(right_id, right_share);
            }
        }
        ui_state.hierarchy_masked = mask_state.hierarchy_masked;
        ui_state.inspector_masked = mask_state.inspector_masked;
        ui_state.last_hierarchy_share = mask_state.last_hierarchy_share;
        ui_state.last_inspector_share = mask_state.last_inspector_share;
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
            let tree_is_empty = !ui_state.has_non_sidebar_content();
            match diff {
                TreeAction::DeleteTab(tile_id) => {
                    if read_only {
                        continue;
                    }
                    let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
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

                    ui_state.tree.remove_recursively(tile_id);

                    if let Some(graph_id) = ui_state.graphs.get(&tile_id) {
                        state_mut.commands.entity(*graph_id).despawn();
                        ui_state.graphs.remove(&tile_id);
                    }
                    ui_state.recompute_has_non_sidebar();
                }
                TreeAction::AddViewport(parent_tile_id) => {
                    if read_only {
                        continue;
                    }
                    let parent_tile_id = if parent_tile_id.is_none() && tree_is_empty {
                        ui_state.apply_scaffold("Viewports")
                    } else {
                        parent_tile_id
                    };
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

                    if let Some(tile_id) = ui_state.insert_tile(
                        Tile::Pane(Pane::Viewport(viewport_pane)),
                        parent_tile_id,
                        true,
                    ) {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddGraph(parent_tile_id, graph_bundle) => {
                    if read_only {
                        continue;
                    }
                    let parent_tile_id = if parent_tile_id.is_none() && tree_is_empty {
                        ui_state.apply_scaffold("Graphs")
                    } else {
                        parent_tile_id
                    };
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
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        *state_mut.selected_object = SelectedObject::Graph { graph_id };
                        ui_state.tree.make_active(|id, _| id == tile_id);
                        ui_state.graphs.insert(tile_id, graph_id);
                    }
                }
                TreeAction::AddMonitor(parent_tile_id, eql) => {
                    if read_only {
                        continue;
                    }
                    let parent_tile_id = if parent_tile_id.is_none() && tree_is_empty {
                        ui_state.apply_scaffold("Monitors")
                    } else {
                        parent_tile_id
                    };
                    let monitor = MonitorPane::new("Monitor".to_string(), eql);

                    let pane = Pane::Monitor(monitor);
                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddVideoStream(parent_tile_id, msg_id, label) => {
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
                        label: label.clone(),
                    });
                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddDashboard(parent_tile_id, dashboard, label) => {
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
                    let pane = Pane::Dashboard(DashboardPane { entity, label });
                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }

                TreeAction::SelectTile(tile_id) => {
                    ui_state.tree.make_active(|id, _| id == tile_id);

                    if let Some(egui_tiles::Tile::Pane(pane)) = ui_state.tree.tiles.get(tile_id) {
                        match pane {
                            Pane::Graph(graph) => {
                                *state_mut.selected_object =
                                    SelectedObject::Graph { graph_id: graph.id };
                            }
                            Pane::QueryPlot(plot) => {
                                *state_mut.selected_object = SelectedObject::Graph {
                                    graph_id: plot.entity,
                                };
                            }
                            Pane::Viewport(viewport) => {
                                if let Some(camera) = viewport.camera {
                                    *state_mut.selected_object =
                                        SelectedObject::Viewport { camera };
                                }
                            }
                            Pane::Hierarchy => {
                                unmask_sidebar_by_kind(&mut ui_state, SidebarKind::Hierarchy);
                            }
                            Pane::Inspector => {
                                unmask_sidebar_by_kind(&mut ui_state, SidebarKind::Inspector);
                            }
                            _ => {}
                        }
                        // Always ensure inspector is visible after a selection.
                        unmask_sidebar_by_kind(&mut ui_state, SidebarKind::Inspector);
                    }
                }
                TreeAction::AddActionTile(parent_tile_id, button_name, lua_code) => {
                    let entity = state_mut
                        .commands
                        .spawn(super::actions::ActionTile {
                            button_name,
                            lua: lua_code,
                            status: Default::default(),
                        })
                        .id();
                    let pane = Pane::ActionTile(ActionTilePane {
                        entity,
                        label: "Action".to_string(),
                    });
                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddQueryTable(parent_tile_id) => {
                    let entity = state_mut.commands.spawn(QueryTableData::default()).id();
                    let pane = Pane::QueryTable(QueryTablePane { entity });
                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        ui_state.tree.make_active(|id, _| id == tile_id);
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
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        *state_mut.selected_object = SelectedObject::Graph { graph_id: entity };
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddSidebars => {
                    if read_only {
                        continue;
                    }
                    let sidebar_ids: std::collections::HashSet<TileId> = ui_state
                        .tree
                        .tiles
                        .iter()
                        .filter_map(|(id, tile)| {
                            matches!(tile, Tile::Pane(Pane::Hierarchy | Pane::Inspector))
                                .then_some(*id)
                        })
                        .collect();

                    if !sidebar_ids.is_empty() {
                        let container_ids: Vec<TileId> = ui_state
                            .tree
                            .tiles
                            .iter()
                            .filter_map(|(id, tile)| {
                                matches!(tile, Tile::Container(_)).then_some(*id)
                            })
                            .collect();

                        for cid in container_ids {
                            if let Some(Tile::Container(container)) =
                                ui_state.tree.tiles.get_mut(cid)
                            {
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
                                        linear
                                            .children
                                            .retain(|child| !sidebar_ids.contains(child));
                                    }
                                    Container::Grid(_) => {}
                                }
                            }
                        }
                    }

                    let mut main_content = ui_state.tree.root();
                    if let Some(root) = ui_state.tree.root() {
                        main_content = match ui_state.tree.tiles.get(root) {
                            Some(Tile::Container(Container::Tabs(tabs))) => {
                                if tabs.children.is_empty() {
                                    None
                                } else {
                                    Some(root)
                                }
                            }
                            Some(Tile::Container(Container::Linear(linear))) => {
                                match linear.children.len() {
                                    0 => None,
                                    1 => Some(linear.children[0]),
                                    _ => Some(root),
                                }
                            }
                            Some(Tile::Container(Container::Grid(grid))) => {
                                let children: Vec<_> = grid.children().copied().collect();
                                match children.len() {
                                    0 => None,
                                    1 => Some(children[0]),
                                    _ => Some(root),
                                }
                            }
                            Some(Tile::Pane(_)) => Some(root),
                            _ => None,
                        };
                    }

                    let hierarchy = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Hierarchy));
                    let inspector = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Inspector));

                    let mut main_content = main_content.unwrap_or_else(|| {
                        let tabs = egui_tiles::Tabs::new(Vec::new());
                        ui_state
                            .tree
                            .tiles
                            .insert_new(Tile::Container(Container::Tabs(tabs)))
                    });

                    let wrap_into_tabs = !matches!(
                        ui_state.tree.tiles.get(main_content),
                        Some(Tile::Container(Container::Tabs(_)))
                    );
                    if wrap_into_tabs {
                        let mut tabs = egui_tiles::Tabs::new(vec![main_content]);
                        tabs.set_active(main_content);
                        main_content = ui_state
                            .tree
                            .tiles
                            .insert_new(Tile::Container(Container::Tabs(tabs)));
                    }

                    let mut linear = egui_tiles::Linear::new(
                        egui_tiles::LinearDir::Horizontal,
                        vec![hierarchy, inspector],
                    );
                    linear.children.insert(1, main_content);
                    let hier_default = 0.2;
                    let insp_default = 0.2;
                    let hier_share = if ui_state.hierarchy_masked {
                        0.01
                    } else {
                        hier_default
                    };
                    let insp_share = if ui_state.inspector_masked {
                        0.01
                    } else {
                        insp_default
                    };
                    let mut center_share = 1.0 - (hier_share + insp_share);
                    if center_share <= 0.0 {
                        center_share = 0.1;
                    }
                    linear.shares.set_share(hierarchy, hier_share);
                    linear.shares.set_share(main_content, center_share);
                    linear.shares.set_share(inspector, insp_share);
                    ui_state.last_hierarchy_share.get_or_insert(hier_default);
                    ui_state.last_inspector_share.get_or_insert(insp_default);

                    let root = ui_state
                        .tree
                        .tiles
                        .insert_new(Tile::Container(Container::Linear(linear)));
                    ui_state.tree.root = Some(root);
                    ui_state.tree.make_active(|id, _| id == hierarchy);
                    ui_state.recompute_has_non_sidebar();
                }
                TreeAction::RenameContainer(tile_id, title) => {
                    if read_only {
                        continue;
                    }
                    ui_state.set_container_title(tile_id, title);
                }
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
