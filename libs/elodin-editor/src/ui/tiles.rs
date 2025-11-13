use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    ecs::system::{SystemParam, SystemState},
    input::keyboard::Key,
    log::info,
    prelude::*,
    window::{Monitor, Window, WindowPosition},
};
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion, OrbitConstraint};
use bevy_egui::{
    EguiContexts,
    egui::{
        self, Color32, CornerRadius, Frame, Id, RichText, Stroke, TopBottomPanel, Ui, Visuals, vec2,
    },
};
use bevy_render::{
    camera::{Exposure, PhysicalCameraParameters},
    view::RenderLayers,
};
use egui::UiBuilder;
use egui_tiles::{Container, Tile, TileId, Tiles};
use impeller2_wkt::{Dashboard, Graph, Viewport, WindowRect};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};
use std::{
    fmt::Write as _,
    path::PathBuf,
    time::{Duration, Instant},
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
    colors::{self, get_scheme, with_opacity},
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
use crate::ui::compute_secondary_window_title;
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
    tree_id: Id,
}

#[derive(Clone, Debug)]
pub struct SecondaryWindowDescriptor {
    pub path: PathBuf,
    pub title: Option<String>,
    pub screen: Option<usize>,
    pub screen_rect: Option<WindowRect>,
}

impl SecondaryWindowDescriptor {
    pub fn wants_explicit_layout(&self) -> bool {
        self.screen.is_some() || self.screen_rect.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrimaryWindowRelayoutPhase {
    #[default]
    Idle,
    NeedScreen,
    NeedRect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryWindowId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SecondaryWindowRelayoutPhase {
    #[default]
    Idle,
    NeedScreen,
    NeedRect,
}

#[derive(Clone)]
pub struct SecondaryWindowState {
    pub id: SecondaryWindowId,
    pub descriptor: SecondaryWindowDescriptor,
    pub tile_state: TileState,
    pub window_entity: Option<Entity>,
    pub graph_entities: Vec<Entity>,
    pub applied_screen: Option<usize>,
    pub applied_rect: Option<WindowRect>,
    pub relayout_phase: SecondaryWindowRelayoutPhase,
    pub pending_fullscreen_exit: bool,
    pub pending_exit_started_at: Option<Instant>,
    pub relayout_attempts: u8,
    pub relayout_started_at: Option<Instant>,
    pub awaiting_screen_confirmation: bool,
    pub skip_metadata_capture: bool,
    pub pending_exit_state: PendingFullscreenExit,
    pub metadata_capture_blocked_until: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PendingFullscreenExit {
    #[default]
    None,
    Requested,
}

#[derive(Clone, Default)]
pub struct PrimaryWindowLayout {
    pub screen: Option<usize>,
    pub screen_rect: Option<WindowRect>,
    pub relayout_phase: PrimaryWindowRelayoutPhase,
    pub applied_screen: Option<usize>,
    pub applied_rect: Option<WindowRect>,
    pub relayout_attempts: u8,
    pub relayout_started_at: Option<Instant>,
    pub pending_fullscreen_exit: bool,
    pub pending_fullscreen_exit_started_at: Option<Instant>,
    pub captured_screen: Option<usize>,
    pub captured_rect: Option<WindowRect>,
    pub requested_screen: Option<usize>,
    pub requested_rect: Option<WindowRect>,
    pub awaiting_screen_confirmation: bool,
}

impl PrimaryWindowLayout {
    pub fn set(&mut self, screen: Option<usize>, rect: Option<WindowRect>) {
        self.screen = screen;
        self.screen_rect = rect;
        self.applied_screen = None;
        self.applied_rect = None;
        self.relayout_attempts = 0;
        self.relayout_started_at = None;
        self.pending_fullscreen_exit = false;
        self.pending_fullscreen_exit_started_at = None;
        self.requested_screen = screen;
        self.requested_rect = rect;
        self.relayout_phase = if self.screen.is_some() {
            PrimaryWindowRelayoutPhase::NeedScreen
        } else if self.screen_rect.is_some() {
            PrimaryWindowRelayoutPhase::NeedRect
        } else {
            PrimaryWindowRelayoutPhase::Idle
        };
        self.awaiting_screen_confirmation = false;
    }
}

impl SecondaryWindowState {
    pub fn relayout_phase_from_descriptor(
        descriptor: &SecondaryWindowDescriptor,
    ) -> SecondaryWindowRelayoutPhase {
        if descriptor.wants_explicit_layout() {
            SecondaryWindowRelayoutPhase::NeedScreen
        } else {
            SecondaryWindowRelayoutPhase::Idle
        }
    }

    pub fn refresh_relayout_phase(&mut self) {
        self.relayout_phase = Self::relayout_phase_from_descriptor(&self.descriptor);
        self.relayout_attempts = 0;
        self.relayout_started_at = None;
        self.pending_fullscreen_exit = false;
        self.pending_exit_started_at = None;
        self.pending_exit_state = PendingFullscreenExit::None;
        self.awaiting_screen_confirmation = false;
    }

    pub fn extend_metadata_capture_block(&mut self, duration: Duration) {
        let candidate = Instant::now() + duration;
        if self
            .metadata_capture_blocked_until
            .map(|current| candidate > current)
            .unwrap_or(true)
        {
            self.metadata_capture_blocked_until = Some(candidate);
        }
    }

    pub fn clear_metadata_capture_block(&mut self) {
        self.metadata_capture_blocked_until = None;
    }

    pub fn is_metadata_capture_blocked(&mut self) -> bool {
        if let Some(until) = self.metadata_capture_blocked_until {
            if Instant::now() < until {
                return true;
            }
            self.metadata_capture_blocked_until = None;
        }
        false
    }

    pub fn update_descriptor_from_window(
        &mut self,
        window: &Window,
        screens: &Query<(Entity, &Monitor)>,
    ) {
        if self.is_metadata_capture_blocked() {
            return;
        }
        if self.skip_metadata_capture {
            self.skip_metadata_capture = false;
            return;
        }

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
            if let Some((monitor_pos, monitor_size)) = best_bounds
                && position_is_reliable_linux(position, (monitor_pos.x, monitor_pos.y))
                && let Some(rect) = rect_from_bounds(
                    (position.x, position.y),
                    (size.x, size.y),
                    (monitor_pos.x, monitor_pos.y),
                    (monitor_size.x, monitor_size.y),
                )
            {
                self.descriptor.screen_rect = Some(rect);
            }
        }
    }

    pub fn update_descriptor_from_winit_window(
        &mut self,
        window: &WinitWindow,
        monitors: &[MonitorHandle],
    ) -> bool {
        if self.is_metadata_capture_blocked() {
            return false;
        }
        if self.skip_metadata_capture {
            self.skip_metadata_capture = false;
            return false;
        }

        let current_monitor = window.current_monitor();
        let outer_position = window.outer_position().ok();
        let outer_size = window.outer_size();
        let mut updated = false;

        let monitor_index = current_monitor
            .as_ref()
            .and_then(|current| monitors.iter().position(|monitor| monitor == current))
            .or_else(|| {
                outer_position
                    .as_ref()
                    .and_then(|position| monitor_index_from_bounds(*position, outer_size, monitors))
            });

        if let Some(index) = monitor_index {
            self.descriptor.screen = Some(index);
            updated = true;
        }

        if let (Some(position), Some(monitor_handle)) = (
            outer_position,
            monitor_index
                .and_then(|idx| monitors.get(idx).cloned())
                .or_else(|| current_monitor.clone()),
        ) {
            let monitor_pos = monitor_handle.position();
            if position_is_reliable_linux(
                IVec2::new(position.x, position.y),
                (monitor_pos.x, monitor_pos.y),
            ) && let Some(rect) = rect_from_bounds(
                (position.x, position.y),
                (outer_size.width, outer_size.height),
                (monitor_pos.x, monitor_pos.y),
                (monitor_handle.size().width, monitor_handle.size().height),
            ) {
                self.descriptor.screen_rect = Some(rect);
                updated = true;
            }
        }

        updated
    }
}

pub(crate) fn monitor_index_from_bounds(
    position: PhysicalPosition<i32>,
    size: PhysicalSize<u32>,
    monitors: &[MonitorHandle],
) -> Option<usize> {
    let center_x = position.x + size.width as i32 / 2;
    let center_y = position.y + size.height as i32 / 2;

    monitors
        .iter()
        .enumerate()
        .filter_map(|(index, monitor)| {
            let monitor_pos = monitor.position();
            let monitor_size = monitor.size();
            let min_x = monitor_pos.x;
            let max_x = monitor_pos.x + monitor_size.width as i32;
            let min_y = monitor_pos.y;
            let max_y = monitor_pos.y + monitor_size.height as i32;

            if center_x >= min_x && center_x < max_x && center_y >= min_y && center_y < max_y {
                let distance = (center_x - monitor_pos.x).abs() + (center_y - monitor_pos.y).abs();
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
    monitor_position: (i32, i32),
    monitor_size: (u32, u32),
) -> Option<WindowRect> {
    if monitor_size.0 == 0 || monitor_size.1 == 0 {
        return None;
    }

    let width_pct = (size.0 as f32 / monitor_size.0 as f32) * 100.0;
    let height_pct = (size.1 as f32 / monitor_size.1 as f32) * 100.0;

    let offset_x = position.0 - monitor_position.0;
    let offset_y = position.1 - monitor_position.1;

    let x_pct = (offset_x as f32 / monitor_size.0 as f32) * 100.0;
    let y_pct = (offset_y as f32 / monitor_size.1 as f32) * 100.0;

    Some(WindowRect {
        x: clamp_percent(x_pct),
        y: clamp_percent(y_pct),
        width: clamp_percent(width_pct),
        height: clamp_percent(height_pct),
    })
}

fn position_is_reliable_linux(position: IVec2, monitor_position: (i32, i32)) -> bool {
    if !cfg!(target_os = "linux") {
        return true;
    }
    if position.x == 0 && position.y == 0 {
        monitor_position.0 == 0 && monitor_position.1 == 0
    } else {
        true
    }
}

pub(crate) fn clamp_percent(value: f32) -> u32 {
    value.round().clamp(0.0, 100.0) as u32
}

#[derive(Resource)]
pub struct WindowManager {
    main: TileState,
    primary: PrimaryWindowLayout,
    secondary: Vec<SecondaryWindowState>,
    next_id: u32,
}

impl Default for WindowManager {
    fn default() -> Self {
        Self {
            main: TileState::new(Id::new("main_tab_tree")),
            primary: PrimaryWindowLayout::default(),
            secondary: Vec::new(),
            next_id: 0,
        }
    }
}

impl WindowManager {
    pub fn main(&self) -> &TileState {
        &self.main
    }

    pub fn main_mut(&mut self) -> &mut TileState {
        &mut self.main
    }

    pub fn primary_layout(&self) -> &PrimaryWindowLayout {
        &self.primary
    }

    pub fn primary_layout_mut(&mut self) -> &mut PrimaryWindowLayout {
        &mut self.primary
    }

    pub fn clear_primary_layout(&mut self) {
        self.primary = PrimaryWindowLayout::default();
    }

    pub fn take_main(&mut self) -> TileState {
        std::mem::take(&mut self.main)
    }

    pub fn replace_main(&mut self, state: TileState) {
        self.main = state;
    }

    pub fn secondary(&self) -> &[SecondaryWindowState] {
        &self.secondary
    }

    pub fn secondary_mut(&mut self) -> &mut Vec<SecondaryWindowState> {
        &mut self.secondary
    }

    pub fn take_secondary(&mut self) -> Vec<SecondaryWindowState> {
        std::mem::take(&mut self.secondary)
    }

    pub fn replace_secondary(&mut self, states: Vec<SecondaryWindowState>) {
        self.secondary = states;
    }

    pub fn alloc_id(&mut self) -> SecondaryWindowId {
        let id = SecondaryWindowId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    pub fn get_secondary(&self, id: SecondaryWindowId) -> Option<&SecondaryWindowState> {
        self.secondary.iter().find(|s| s.id == id)
    }

    pub fn get_secondary_mut(
        &mut self,
        id: SecondaryWindowId,
    ) -> Option<&mut SecondaryWindowState> {
        self.secondary.iter_mut().find(|s| s.id == id)
    }

    pub fn find_secondary_by_entity(&self, entity: Entity) -> Option<SecondaryWindowId> {
        self.secondary
            .iter()
            .find(|state| state.window_entity == Some(entity))
            .map(|state| state.id)
    }

    pub fn create_secondary_window(&mut self, title: Option<String>) -> SecondaryWindowId {
        let id = self.alloc_id();
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
        let descriptor = SecondaryWindowDescriptor {
            path: PathBuf::from(path),
            title: cleaned_title.or_else(|| Some(format!("Window {}", id.0 + 1))),
            screen: None,
            screen_rect: None,
        };
        let relayout_phase = SecondaryWindowState::relayout_phase_from_descriptor(&descriptor);
        let tile_state = TileState::new(Id::new(("secondary_tab_tree", id.0)));
        info!(
            id = id.0,
            title = descriptor.title.as_deref().unwrap_or(""),
            path = %descriptor.path.display(),
            "Created secondary window"
        );
        self.secondary.push(SecondaryWindowState {
            id,
            descriptor,
            tile_state,
            window_entity: None,
            graph_entities: Vec::new(),
            applied_screen: None,
            applied_rect: None,
            relayout_phase,
            pending_fullscreen_exit: false,
            pending_exit_started_at: None,
            pending_exit_state: PendingFullscreenExit::None,
            relayout_attempts: 0,
            relayout_started_at: None,
            awaiting_screen_confirmation: false,
            skip_metadata_capture: false,
            metadata_capture_blocked_until: None,
        });
        id
    }
}

#[derive(Resource, Default)]
pub struct ViewportContainsPointer(pub bool);
#[derive(Clone)]
pub struct ActionTilePane {
    pub entity: Entity,
    pub label: String,
}

#[derive(Clone)]
pub struct TreePane {
    pub entity: Entity,
}

#[derive(Clone)]
pub struct DashboardPane {
    pub entity: Entity,
    pub label: String,
}

impl TileState {
    pub fn has_inspector(&self) -> bool {
        self.tree
            .tiles
            .iter()
            .any(|(_, tile)| matches!(tile, Tile::Pane(Pane::Inspector)))
    }

    pub fn inspector_pending(&self) -> bool {
        self.tree_actions
            .iter()
            .any(|action| matches!(action, TreeAction::AddInspector(_)))
    }

    pub fn insert_tile(
        &mut self,
        tile: Tile<Pane>,
        parent_id: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        let parent_id = if let Some(id) = parent_id {
            id
        } else {
            let root_id = self.tree.root().or_else(|| {
                self.reset_tree();
                self.tree.root()
            })?;

            if let Some(Tile::Container(Container::Linear(linear))) = self.tree.tiles.get(root_id) {
                if let Some(center) = linear.children.get(linear.children.len() / 2) {
                    *center
                } else {
                    root_id
                }
            } else {
                root_id
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

    pub fn create_hierarchy_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddHierarchy(tile_id));
    }

    pub fn create_inspector_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddInspector(tile_id));
    }

    pub fn create_tree_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions
            .push(TreeAction::AddSchematicTree(tile_id));
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

    pub fn clear(&mut self, commands: &mut Commands, _selected_object: &mut SelectedObject) {
        for (tile_id, tile) in self.tree.tiles.iter() {
            match tile {
                Tile::Pane(Pane::Viewport(viewport)) => {
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
}

impl Pane {
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
                pane.label.to_string()
            }
            Pane::Viewport(viewport) => viewport.label.to_string(),
            Pane::Monitor(monitor) => monitor.label.to_string(),
            Pane::QueryTable(..) => "Query".to_string(),
            Pane::QueryPlot(query_plot) => {
                if let Ok(graph_state) = graph_states.get(query_plot.entity) {
                    return graph_state.label.to_string();
                }
                "Query Plot".to_string()
            }
            Pane::ActionTile(action) => action.label.to_string(),
            Pane::VideoStream(video_stream) => video_stream.label.to_string(),
            Pane::Dashboard(dashboard) => {
                if let Ok(dash) = dashboards.get(dashboard.entity) {
                    return dash
                        .root
                        .label
                        .as_deref()
                        .unwrap_or("Dashboard")
                        .to_string();
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
                ui.add_widget_with::<super::query_plot::QueryPlotWidget>(
                    world,
                    "query_plot",
                    pane.clone(),
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

#[derive(Default, Clone)]
pub struct ViewportPane {
    pub camera: Option<Entity>,
    pub nav_gizmo: Option<Entity>,
    pub nav_gizmo_camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
    pub label: String,
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
        if let Some(grid_layer) = render_layer_alloc.alloc() {
            main_camera_layers = main_camera_layers.with(grid_layer);
            grid_layers = grid_layers.with(grid_layer);
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
                let compiled_expr = eql_ctx.parse_str(eql).ok().map(compile_eql_expr);
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
                let compiled_expr = eql_ctx.parse_str(eql).ok().map(compile_eql_expr);
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
        }
    }
}

#[derive(Clone)]
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
            tree_actions: SmallVec::new(),
            graphs: HashMap::new(),
            container_titles: HashMap::new(),
            tree_id,
        }
    }

    fn reset_tree(&mut self) {
        self.tree = egui_tiles::Tree::new_tabs(self.tree_id, vec![]);
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
    target_window: Option<SecondaryWindowId>,
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
    AddHierarchy(Option<TileId>),
    AddInspector(Option<TileId>),
    AddSchematicTree(Option<TileId>),
    AddSidebars,
    DeleteTab(TileId),
    SelectTile(TileId),
    RenameContainer(TileId, String),
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
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());

        if !self.read_only && is_container && state.active && response.clicked() && !is_editing {
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

            if !self.read_only && is_container && is_editing {
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
        _tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.read_only {
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
    windows: Res<'w, WindowManager>,
}

impl<'w, 's> TileSystem<'w, 's> {
    fn prepare_panel_data(
        world: &mut World,
        state: &mut SystemState<Self>,
        target: Option<SecondaryWindowId>,
    ) -> Option<(TileIcons, bool, bool)> {
        let read_only = false;
        let params = state.get_mut(world);
        let mut contexts = params.contexts;
        let images = params.images;
        let is_empty = match target {
            Some(id) => params
                .windows
                .get_secondary(id)
                .map(|s| s.tile_state.is_empty() && s.tile_state.tree_actions.is_empty()),
            None => Some(
                params.windows.main().is_empty() && params.windows.main().tree_actions.is_empty(),
            ),
        };

        let is_empty_tile_tree = is_empty?;

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
        target: Option<SecondaryWindowId>,
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
    type Args = Option<SecondaryWindowId>;
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
    type Args = Option<SecondaryWindowId>;
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

        let header = target.and_then(|id| {
            world
                .get_resource::<WindowManager>()
                .and_then(|windows| windows.get_secondary(id))
                .map(|state| (id, compute_secondary_window_title(state)))
        });

        if let Some((id, title)) = header.as_ref() {
            TopBottomPanel::top(format!("secondary_header_{:?}", id))
                .exact_height(32.0)
                .show(ctx, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        // leave room for native window buttons on macOS
                        ui.add_space(60.0);
                        ui.label(RichText::new(title).color(Color32::WHITE).strong());
                    });
                });
        }

        let central = egui::CentralPanel::default().frame(Frame {
            fill: fill_color,
            ..Default::default()
        });

        central.show(ctx, |ui| {
            if header.is_some() {
                ui.add_space(6.0);
            }
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
    pub window: Option<SecondaryWindowId>,
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
}

#[derive(Clone)]
pub struct TileLayoutArgs {
    pub icons: TileIcons,
    pub window: Option<SecondaryWindowId>,
    pub read_only: bool,
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

        world.resource_scope::<WindowManager, _>(|world, mut windows| {
            let Some(ui_state) = (match window {
                Some(id) => windows.get_secondary_mut(id).map(|s| &mut s.tile_state),
                None => Some(windows.main_mut()),
            }) else {
                return;
            };
            let icons = icons;

            let mut tree_actions = {
                let tab_diffs = std::mem::take(&mut ui_state.tree_actions);
                let mut behavior = TreeBehavior {
                    icons,
                    world,
                    tree_actions: tab_diffs,
                    container_titles: ui_state.container_titles.clone(),
                    read_only,
                    target_window: window,
                };
                ui_state.tree.ui(&mut behavior, ui);

                let TreeBehavior { tree_actions, .. } = behavior;
                tree_actions
            };
            let mut state_mut = state.get_mut(world);
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
                        let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
                            continue;
                        };

                        if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
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

                        if let Some(egui_tiles::Tile::Pane(pane)) = ui_state.tree.tiles.get(tile_id)
                        {
                            match pane {
                                Pane::Graph(graph) => {
                                    *state_mut.selected_object =
                                        SelectedObject::Graph { graph_id: graph.id };
                                    if !ui_state.has_inspector() && !ui_state.inspector_pending() {
                                        ui_state.tree_actions.push(TreeAction::AddInspector(None));
                                    }
                                }
                                Pane::QueryPlot(plot) => {
                                    *state_mut.selected_object = SelectedObject::Graph {
                                        graph_id: plot.entity,
                                    };
                                    if !ui_state.has_inspector() && !ui_state.inspector_pending() {
                                        ui_state.tree_actions.push(TreeAction::AddInspector(None));
                                    }
                                }
                                Pane::Viewport(viewport) => {
                                    if let Some(camera) = viewport.camera {
                                        *state_mut.selected_object =
                                            SelectedObject::Viewport { camera };
                                    }
                                }
                                _ => {}
                            }
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
                        });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            *state_mut.selected_object = SelectedObject::Graph { graph_id: entity };
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddHierarchy(parent_tile_id) => {
                        if read_only {
                            continue;
                        }
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(Pane::Hierarchy), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddInspector(parent_tile_id) => {
                        if read_only {
                            continue;
                        }
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(Pane::Inspector), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
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
                        let pane = Pane::SchematicTree(TreePane { entity });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddSidebars => {
                        if read_only {
                            continue;
                        }
                        let hierarchy = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Hierarchy));
                        let inspector = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Inspector));

                        let mut linear = egui_tiles::Linear::new(
                            egui_tiles::LinearDir::Horizontal,
                            vec![hierarchy, inspector],
                        );
                        if let Some(root) = ui_state.tree.root() {
                            linear.children.insert(1, root);
                            linear.shares.set_share(hierarchy, 0.2);
                            linear.shares.set_share(root, 0.6);
                            linear.shares.set_share(inspector, 0.2);
                        }
                        let root = ui_state
                            .tree
                            .tiles
                            .insert_new(Tile::Container(Container::Linear(linear)));
                        ui_state.tree.root = Some(root);
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
                }
            }
        })
    }
}

pub fn shortcuts(key_state: Res<LogicalKeyState>, mut windows: ResMut<WindowManager>) {
    let ui_state = windows.main_mut();
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
