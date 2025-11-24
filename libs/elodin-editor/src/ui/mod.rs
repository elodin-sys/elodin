use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use bevy::{
    app::AppExit,
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    input::keyboard::Key,
    log::{error, info, warn},
    prelude::*,
    render::camera::{RenderTarget, Viewport},
    window::{
        EnabledButtons, Monitor, NormalizedWindowRef, PresentMode, PrimaryWindow,
        WindowCloseRequested, WindowMoved, WindowRef, WindowResized, WindowResolution,
    },
};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Color32, Label, RichText},
};
use egui_tiles::{Container, Tile};
#[cfg(target_os = "macos")]
use winit::window::Fullscreen;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    monitor::MonitorHandle,
    window::Window as WinitWindow,
};

pub(crate) const DEFAULT_SECONDARY_RECT: WindowRect = WindowRect {
    x: 10,
    y: 10,
    width: 80,
    height: 80,
};
const SCREEN_RELAYOUT_MAX_ATTEMPTS: u8 = 5;
const SECONDARY_RECT_CAPTURE_LOAD_GUARD: Duration = Duration::from_millis(2500);
const SECONDARY_RECT_CAPTURE_STABILIZE_GUARD: Duration = Duration::from_millis(400);
// Order ranges:
// 0          -> UI/egui (Bevy default)
// 10..        primary viewports (3D, gizmo/axes…)
// 100..       primary graphs
// 1000..      secondary windows (stride by window id)
const PRIMARY_VIEWPORT_ORDER_BASE: isize = 10;
const PRIMARY_GRAPH_ORDER_BASE: isize = 100;
const SECONDARY_GRAPH_ORDER_BASE: isize = 1000;
const SECONDARY_GRAPH_ORDER_STRIDE: isize = 50;
const NAV_GIZMO_ORDER_OFFSET: isize = 1;
const DEFAULT_PRESENT_MODE: PresentMode = PresentMode::Fifo;

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    pub const LINUX_MULTI_WINDOW: bool = true;
    pub const SCREEN_RELAYOUT_TIMEOUT: Duration = Duration::from_millis(2000);
    pub const PRIMARY_ORDER_OFFSET: isize = 0;
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;
    pub const LINUX_MULTI_WINDOW: bool = false;
    pub const SCREEN_RELAYOUT_TIMEOUT: Duration = Duration::from_millis(3500);
    pub const PRIMARY_ORDER_OFFSET: isize = 1000;
}

use platform::{LINUX_MULTI_WINDOW, PRIMARY_ORDER_OFFSET, SCREEN_RELAYOUT_TIMEOUT};

use big_space::GridCell;
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::get_scheme;
use self::{command_palette::CommandPaletteState, plot::GraphState, timeline::timeline_slider};
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::ComponentValue;
use impeller2_wkt::{ComponentMetadata, WindowRect};

use crate::{
    GridHandle, MainCamera,
    plugins::{LogicalKeyState, navigation_gizmo::NavGizmoCamera},
};

use self::inspector::entity::ComponentFilter;

use self::command_palette::CommandPalette;
use self::widgets::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystemExt};

pub mod actions;
pub mod button;
pub mod colors;
pub mod command_palette;
pub mod dashboard;
pub mod hierarchy;
pub mod images;
pub mod inspector;
pub mod label;
pub mod modal;
pub mod monitor;
pub mod plot;
pub mod plot_3d;
pub mod query_plot;
pub mod query_table;
pub mod schematic;
mod theme;
pub mod tiles;
pub mod time_label;
pub mod timeline;
pub mod utils;
pub mod video_stream;
pub mod widgets;

#[cfg(not(target_family = "wasm"))]
pub mod status_bar;

#[cfg(not(target_family = "wasm"))]
pub mod startup_window;

#[derive(Resource, Default)]
pub struct HdrEnabled(pub bool);

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
struct SecondaryLogState(
    HashMap<tiles::SecondaryWindowId, (Option<usize>, bool, tiles::SecondaryWindowRelayoutPhase)>,
);

#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub enum SelectedObject {
    #[default]
    None,
    Entity(EntityPair),
    Viewport {
        camera: Entity,
    },
    Graph {
        graph_id: Entity,
    },
    Action {
        action_id: Entity,
    },
    Object3D {
        entity: Entity,
    },
    DashboardNode {
        entity: Entity,
    },
}

impl SelectedObject {
    pub fn is_entity_selected(&self, id: impeller2::types::ComponentId) -> bool {
        matches!(self, SelectedObject::Entity(pair) if pair.impeller == id)
    }

    pub fn entity(&self) -> Option<Entity> {
        match self {
            SelectedObject::None => None,
            SelectedObject::Entity(pair) => Some(pair.bevy),
            SelectedObject::Viewport { camera } => Some(*camera),
            SelectedObject::Graph { graph_id } => Some(*graph_id),
            SelectedObject::Action { action_id } => Some(*action_id),
            SelectedObject::Object3D { entity } => Some(*entity),
            SelectedObject::DashboardNode { entity } => Some(*entity),
        }
    }
}

#[derive(Resource, Default)]
pub struct HoveredEntity(pub Option<EntityPair>);

#[derive(Resource, Default)]
pub struct EntityFilter(pub String);

#[derive(Resource, Default)]
pub struct InspectorAnchor(pub Option<egui::Pos2>);

#[derive(Component, Clone)]
pub struct ViewportRect(pub Option<egui::Rect>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityPair {
    pub bevy: Entity,
    pub impeller: ComponentId,
}

#[derive(Component)]
struct SecondaryWindowMarker {
    id: tiles::SecondaryWindowId,
}

#[derive(Component)]
struct ActiveSecondaryWindow;

pub fn shortcuts(
    mut paused: ResMut<Paused>,
    command_palette_state: Res<CommandPaletteState>,
    key_state: Res<LogicalKeyState>,
    mut context: Query<&mut EguiContext>,
) {
    let input_has_focus = command_palette_state.show
        || context
            .iter_mut()
            .any(|mut c| c.get_mut().memory(|m| m.focused().is_some()));

    if !input_has_focus && key_state.just_pressed(&Key::Space) {
        paused.0 = !paused.0;
    }
}

pub type EntityData<'a> = (
    &'a ComponentId,
    Entity,
    &'a mut ComponentValue,
    &'a ComponentMetadata,
);

pub type EntityDataReadOnly<'a> = (
    &'a ComponentId,
    Entity,
    &'a ComponentValueMap,
    &'a ComponentMetadata,
);

#[derive(QueryData)]
#[query_data(mutable)]
pub struct CameraQuery {
    entity: Entity,
    camera: &'static mut Camera,
    projection: &'static mut Projection,
    transform: &'static mut Transform,
    global_transform: &'static mut GlobalTransform,
    grid_cell: &'static mut GridCell<i128>,
    parent: Option<&'static ChildOf>,
    grid_handle: Option<&'static GridHandle>,
    no_propagate_rot: Option<&'static big_space::propagation::NoPropagateRot>,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        // Probe ELODIN_KDL_DIR once to inform or warn about an invalid
        // directory surfaces immediately on startup.
        match impeller2_kdl::env::schematic_dir() {
            Ok(Some(path)) => info!("ELODIN_KDL_DIR set to {:?}", path.display()),
            Ok(None) => info!("ELODIN_KDL_DIR defaulted to current working directory"),
            Err(err) => error!("{err}, falling back to current working directory"),
        }

        app.init_resource::<Paused>()
            .init_resource::<SelectedObject>()
            .init_resource::<HoveredEntity>()
            .init_resource::<EntityFilter>()
            .init_resource::<ComponentFilter>()
            .init_resource::<InspectorAnchor>()
            .init_resource::<tiles::WindowManager>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<timeline_slider::UITick>()
            .init_resource::<SecondaryLogState>()
            .init_resource::<timeline::StreamTickOrigin>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_event::<DialogEvent>()
            .add_systems(Update, timeline_slider::sync_ui_tick.before(render_layout))
            .add_systems(Update, actions::spawn_lua_actor)
            .add_systems(Update, shortcuts)
            .add_systems(Update, handle_primary_close.before(render_layout))
            .add_systems(Update, render_layout)
            .add_systems(First, warn_camera_order_ambiguities)
            .add_systems(
                Update,
                apply_primary_window_layout
                    .after(render_layout)
                    .before(sync_secondary_windows),
            )
            .add_systems(
                Update,
                confirm_primary_screen_assignment
                    .after(apply_primary_window_layout)
                    .before(sync_secondary_windows),
            )
            .add_systems(Update, sync_secondary_windows.after(render_layout))
            .add_systems(
                Update,
                apply_secondary_window_screens.after(sync_secondary_windows),
            )
            .add_systems(
                Update,
                confirm_secondary_screen_assignment
                    .after(apply_secondary_window_screens)
                    .before(track_secondary_window_geometry),
            )
            .add_systems(
                Update,
                track_secondary_window_geometry.after(confirm_secondary_screen_assignment),
            )
            .add_systems(
                Update,
                capture_secondary_window_screens.after(track_secondary_window_geometry),
            )
            .add_systems(
                Update,
                capture_primary_window_layout.after(capture_secondary_window_screens),
            )
            .add_systems(Update, handle_secondary_close.after(sync_secondary_windows))
            .add_systems(
                Update,
                set_secondary_camera_viewport.after(sync_secondary_windows),
            )
            .add_systems(
                Update,
                set_nav_gizmo_camera_orders
                    .after(set_secondary_camera_viewport)
                    .after(set_camera_viewport),
            )
            .add_systems(
                Update,
                warn_camera_order_ambiguities
                    .after(set_secondary_camera_viewport)
                    .after(set_camera_viewport),
            )
            .add_systems(
                Update,
                render_secondary_windows.after(handle_secondary_close),
            )
            .add_systems(First, fix_visibility_hierarchy)
            .add_systems(Update, sync_hdr)
            .add_systems(Update, tiles::shortcuts)
            .add_systems(Update, set_camera_viewport.after(render_layout))
            .add_systems(Update, sync_camera_grid_cell.after(render_layout))
            .add_systems(Update, query_plot::auto_bounds)
            .add_systems(Update, dashboard::update_nodes)
            .add_plugins(SchematicPlugin)
            .add_plugins(LinePlot3dPlugin)
            .add_plugins(command_palette::palette_items::plugin);
    }
}

#[derive(Clone, Debug)]
pub enum SettingModal {
    Graph(Entity, Option<ComponentId>),
    GraphRename(Entity, String),
    Dialog(Dialog),
}

#[derive(Clone, Debug)]
pub struct Dialog {
    pub id: String,
    pub title: String,
    pub message: String,
    pub buttons: Vec<DialogButton>,
}

#[derive(Clone, Debug)]
pub struct DialogButton {
    pub text: String,
    pub action: DialogAction,
}

#[derive(Clone, Debug)]
pub enum DialogAction {
    Close,
    Custom(String), // Custom action identifier
}

#[derive(Clone, Debug, Event)]
pub struct DialogEvent {
    pub action: DialogAction,
    pub id: String,
}

#[derive(Resource, Default, Clone, Debug)]
pub struct SettingModalState(pub Option<SettingModal>);

impl SettingModalState {
    /// Close any open modal
    pub fn close(&mut self) {
        self.0 = None;
    }
}

#[derive(SystemParam)]
pub struct MainLayout<'w, 's> {
    _contexts: EguiContexts<'w, 's>,
    _images: Local<'s, images::Images>,
}

impl RootWidgetSystem for MainLayout<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let _state = state.get_mut(world);

        theme::set_theme(ctx);

        #[cfg(not(target_family = "wasm"))]
        world.add_root_widget::<status_bar::StatusBar>("status_bar");

        let frame = egui::Frame::new();

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            ui.add_widget::<timeline::TimelinePanel>(world, "timeline_panel");
            ui.add_widget_with::<tiles::TileSystem>(world, "tile_system", None);
        });
    }
}

#[derive(SystemParam)]
pub struct ViewportOverlay<'w, 's> {
    window: Query<'w, 's, &'static Window>,
    entities_meta: Query<'w, 's, EntityData<'static>>,
    hovered_entity: Res<'w, HoveredEntity>,
}

impl RootWidgetSystem for ViewportOverlay<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let window = state_mut.window;
        let entities_meta = state_mut.entities_meta;
        let hovered_entity = state_mut.hovered_entity;

        let Ok(window) = window.single() else {
            return;
        };

        let hovered_entity_meta = if let Some(hovered_entity_pair) = hovered_entity.0 {
            entities_meta
                .iter()
                .find(|(id, _, _, _)| hovered_entity_pair.impeller == **id)
                .map(|(_, _, _, metadata)| metadata.to_owned())
        } else {
            None
        };

        if let Some(hovered_entity_meta) = hovered_entity_meta {
            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);

            if let Some(cursor_pos) = window.cursor_position() {
                let offset = 16.0;
                let window_pos = egui::pos2(cursor_pos.x + offset, cursor_pos.y + offset);

                egui::Window::new("hovered_entity")
                    .title_bar(false)
                    .resizable(false)
                    .frame(egui::Frame {
                        fill: colors::with_opacity(get_scheme().bg_secondary, 0.5),
                        stroke: egui::Stroke::new(
                            1.0,
                            colors::with_opacity(get_scheme().text_primary, 0.5),
                        ),
                        inner_margin: egui::Margin::symmetric(16, 8),
                        ..Default::default()
                    })
                    .fixed_pos(window_pos)
                    .show(ctx, |ui| {
                        ui.add(Label::new(
                            RichText::new(hovered_entity_meta.name).color(Color32::WHITE),
                        ));
                    });
            }
        }
    }
}

pub fn render_layout(world: &mut World) {
    world.add_root_widget::<MainLayout>("main_layout");

    world.add_root_widget::<ViewportOverlay>("viewport_overlay");

    world.add_root_widget::<modal::ModalWithSettings>("modal_graph");

    world.add_root_widget::<CommandPalette>("command_palette");
}

fn sync_secondary_windows(
    mut commands: Commands,
    mut windows: ResMut<tiles::WindowManager>,
    existing: Query<(Entity, &SecondaryWindowMarker)>,
    mut cameras: Query<&mut Camera>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut log_state: ResMut<SecondaryLogState>,
) {
    let mut existing_map: HashMap<tiles::SecondaryWindowId, Entity> = HashMap::new();
    for (entity, marker) in existing.iter() {
        existing_map.insert(marker.id, entity);
    }
    let screens_any = collect_screens_from_any_window(&winit_windows);
    if screens_any.is_none() {
        warn!("No screen info available; secondary windows will use default sizing/position");
    }

    for (id, entity) in existing_map.clone() {
        if windows.get_secondary(id).is_none() {
            commands.entity(entity).despawn();
            existing_map.remove(&id);
        }
    }

    for state in windows.secondary_mut().iter_mut() {
        let current_key = (state.descriptor.screen, false, state.relayout_phase);
        let last = log_state.0.get(&state.id).copied();
        if last != Some(current_key) {
            info!(
                id = state.id.0,
                screen = state.descriptor.screen.map(|s| s as i32).unwrap_or(-1),
                relayout = ?state.relayout_phase,
                "secondary_window_state"
            );
            log_state.0.insert(state.id, current_key);
        }
        state.graph_entities = state.tile_state.collect_graph_entities();

        if let Some(entity) = state.window_entity
            && existing_map.get(&state.id).copied() != Some(entity)
        {
            state.window_entity = None;
        }

        if let Some(entity) = state.window_entity {
            existing_map.insert(state.id, entity);
            let window_ref = WindowRef::Entity(entity);
            for (index, &graph) in state.graph_entities.iter().enumerate() {
                if let Ok(mut camera) = cameras.get_mut(graph) {
                    camera.target = RenderTarget::Window(window_ref);
                    camera.is_active = true;
                    let base_order = secondary_graph_order_base(state.id);
                    camera.order = base_order + index as isize;
                }
            }
            continue;
        }

        // Window is missing: ensure associated cameras are deactivated before spawn.
        for &graph in &state.graph_entities {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                camera.is_active = false;
            }
        }

        let title = compute_secondary_window_title(state);

        // Try to pre-size and pre-position the window to its target rect to avoid an
        // extra resize pass (and the resulting swapchain churn).
        let (resolution, position, pre_applied_screen) = if let Some(rect) =
            state.descriptor.screen_rect
            && let Some(screen_idx) = state.descriptor.screen
            && let Some(screens) = screens_any.as_ref()
            && let Some(screen) = screens.get(screen_idx)
        {
            let screen_pos = screen.position();
            let screen_size = screen.size();
            let width_px = ((rect.width as f64 / 100.0) * screen_size.width as f64)
                .round()
                .max(1.0);
            let height_px = ((rect.height as f64 / 100.0) * screen_size.height as f64)
                .round()
                .max(1.0);
            let x =
                screen_pos.x + ((rect.x as f64 / 100.0) * screen_size.width as f64).round() as i32;
            let y =
                screen_pos.y + ((rect.y as f64 / 100.0) * screen_size.height as f64).round() as i32;
            (
                WindowResolution::new(width_px as f32, height_px as f32),
                Some(WindowPosition::At(IVec2::new(x, y))),
                Some(screen_idx),
            )
        } else {
            (
                WindowResolution::new(640.0, 480.0),
                None,
                state.descriptor.screen,
            )
        };

        let window_component = Window {
            title,
            resolution,
            position: position.unwrap_or(WindowPosition::Automatic),
            present_mode: DEFAULT_PRESENT_MODE,
            enabled_buttons: EnabledButtons {
                close: true,
                minimize: true,
                maximize: true,
            },
            ..Default::default()
        };

        let window_entity = commands
            .spawn((window_component, SecondaryWindowMarker { id: state.id }))
            .id();

        state.window_entity = Some(window_entity);
        state.applied_screen = pre_applied_screen;
        state.applied_rect = None;
        state.relayout_phase = if state.descriptor.screen.is_some() {
            tiles::SecondaryWindowRelayoutPhase::NeedScreen
        } else if state.descriptor.screen_rect.is_some() {
            tiles::SecondaryWindowRelayoutPhase::NeedRect
        } else {
            tiles::SecondaryWindowRelayoutPhase::Idle
        };
        state.skip_metadata_capture = true;
        existing_map.insert(state.id, window_entity);
        let window_ref = WindowRef::Entity(window_entity);
        for (index, &graph) in state.graph_entities.iter().enumerate() {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                camera.target = RenderTarget::Window(window_ref);
                camera.is_active = true;
                let base_order = secondary_graph_order_base(state.id);
                camera.order = base_order + index as isize;
            }
        }
    }
}

fn apply_secondary_window_screens(
    mut windows: ResMut<tiles::WindowManager>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    for state in windows.secondary_mut().iter_mut() {
        if matches!(
            state.relayout_phase,
            tiles::SecondaryWindowRelayoutPhase::Idle
        ) {
            continue;
        }
        // layout locking removed; just proceed if we have the window.
        let Some(entity) = state.window_entity else {
            continue;
        };
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };

        match state.relayout_phase {
            tiles::SecondaryWindowRelayoutPhase::NeedScreen => {
                info!(
                    path = %state.descriptor.path.display(),
                    target_screen = state.descriptor.screen.map(|s| s as i32).unwrap_or(-1),
                    relayout_phase = ?state.relayout_phase,
                    "Attempting secondary screen assignment"
                );
                if state.awaiting_screen_confirmation {
                    info!(
                        path = %state.descriptor.path.display(),
                        "Still awaiting secondary screen confirmation"
                    );
                    let still_waiting = state
                        .relayout_started_at
                        .map(|started| started.elapsed() <= SCREEN_RELAYOUT_TIMEOUT)
                        .unwrap_or(false);
                    if still_waiting {
                        continue;
                    }
                    state.awaiting_screen_confirmation = false;
                }

                let screens = collect_sorted_screens(window);

                if window_on_target_screen(state, window, &screens) {
                    if LINUX_MULTI_WINDOW {
                        exit_fullscreen(window);
                        force_windowed(window);
                    } else if window.fullscreen().is_some() {
                        exit_fullscreen(window);
                    }
                    complete_screen_assignment(state, window, "Confirmed screen assignment (sync)");
                    continue;
                }

                let Some(screen) = state.descriptor.screen else {
                    complete_screen_assignment(
                        state,
                        window,
                        "No screen provided; skipping screen alignment",
                    );
                    continue;
                };

                if let Some(target_monitor) = screens.get(screen).cloned() {
                    assign_window_to_screen(state, window, target_monitor.clone());
                    if detect_window_screen(window, &screens) != Some(screen) {
                        recenter_window_on_screen(window, &target_monitor);
                        #[cfg(target_os = "macos")]
                        {
                            window.set_fullscreen(Some(Fullscreen::Borderless(Some(
                                target_monitor.clone(),
                            ))));
                            window.set_fullscreen(None);
                            recenter_window_on_screen(window, &target_monitor);
                        }
                    }
                    state.relayout_attempts = state.relayout_attempts.saturating_add(1);
                    if state.relayout_started_at.is_none() {
                        state.relayout_started_at = Some(Instant::now());
                    }
                    state.awaiting_screen_confirmation = true;
                    if let Some(started) = state.relayout_started_at
                        && started.elapsed() > SCREEN_RELAYOUT_TIMEOUT
                        && state.relayout_attempts >= SCREEN_RELAYOUT_MAX_ATTEMPTS
                    {
                        warn!(
                            attempts = state.relayout_attempts,
                            elapsed_ms = started.elapsed().as_millis(),
                            path = %state.descriptor.path.display(),
                            "Timed out while assigning screen; continuing with current monitor"
                        );
                        complete_screen_assignment(state, window, "Screen assignment timed out");
                    }
                } else {
                    warn!(
                        screen,
                        path = %state.descriptor.path.display(),
                        "screen out of range; skipping screen assignment"
                    );
                    state.descriptor.screen = None;
                    complete_screen_assignment(state, window, "screen out of range");
                }
            }
            tiles::SecondaryWindowRelayoutPhase::NeedRect => {
                if apply_secondary_window_rect(state, window) {
                    state.relayout_phase = tiles::SecondaryWindowRelayoutPhase::Idle;
                }
            }
            tiles::SecondaryWindowRelayoutPhase::Idle => {}
        }
    }
}

fn apply_primary_window_layout(
    mut windows: ResMut<tiles::WindowManager>,
    primary_query: Query<Entity, With<PrimaryWindow>>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    #[allow(deprecated)]
    let Ok(primary_entity) = primary_query.get_single() else {
        return;
    };
    let Some(window) = winit_windows.get_window(primary_entity) else {
        return;
    };
    let layout = windows.primary_layout_mut();
    match layout.relayout_phase {
        tiles::PrimaryWindowRelayoutPhase::Idle => {}
        tiles::PrimaryWindowRelayoutPhase::NeedScreen => {
            info!(
                target_screen = layout.screen.map(|s| s as i32).unwrap_or(-1),
                relayout_phase = ?layout.relayout_phase,
                "Attempting primary screen assignment"
            );

            if layout.awaiting_screen_confirmation {
                let still_waiting = layout
                    .relayout_started_at
                    .map(|started| started.elapsed() <= SCREEN_RELAYOUT_TIMEOUT)
                    .unwrap_or(false);
                if still_waiting {
                    return;
                }
                layout.awaiting_screen_confirmation = false;
            }

            let screens = collect_sorted_screens(window);
            if window_on_screen(layout.screen, window, &screens) {
                // If rect already applied, proceed to Idle to avoid extra swapchain reconfig.
                if let Some(rect) = layout.screen_rect
                    && layout.applied_rect == Some(rect)
                {
                    layout.relayout_phase = tiles::PrimaryWindowRelayoutPhase::Idle;
                    layout.awaiting_screen_confirmation = false;
                    complete_primary_screen_assignment(
                        layout,
                        window,
                        "Primary window confirmed on target screen",
                    );
                    return;
                }
                if LINUX_MULTI_WINDOW {
                    if window.fullscreen().is_some() {
                        exit_fullscreen(window);
                        return;
                    }
                    force_windowed(window);
                } else if window.fullscreen().is_some() {
                    exit_fullscreen(window);
                } else {
                    window.set_maximized(false);
                    linux_clear_minimized(window);
                }
                complete_primary_screen_assignment(
                    layout,
                    window,
                    "Primary window confirmed on target screen",
                );
                return;
            }
            let Some(screen) = layout.screen else {
                layout.awaiting_screen_confirmation = false;
                layout.relayout_phase = if layout.screen_rect.is_some() {
                    tiles::PrimaryWindowRelayoutPhase::NeedRect
                } else {
                    tiles::PrimaryWindowRelayoutPhase::Idle
                };
                return;
            };
            if let Some(target_monitor) = screens.get(screen).cloned() {
                info!(
                    screen = screen as i32,
                    "Moving primary window to target screen"
                );
                assign_primary_window_to_screen(layout, window, target_monitor);
                layout.awaiting_screen_confirmation = true;
                layout.relayout_attempts = layout.relayout_attempts.saturating_add(1);
                if layout.relayout_started_at.is_none() {
                    layout.relayout_started_at = Some(Instant::now());
                }
                if let Some(started) = layout.relayout_started_at
                    && started.elapsed() > SCREEN_RELAYOUT_TIMEOUT
                    && layout.relayout_attempts >= SCREEN_RELAYOUT_MAX_ATTEMPTS
                {
                    warn!(
                        screen = screen as i32,
                        attempts = layout.relayout_attempts,
                        elapsed_ms = started.elapsed().as_millis(),
                        "Primary window screen assignment timed out"
                    );
                    layout.awaiting_screen_confirmation = false;
                    layout.relayout_phase = if layout.screen_rect.is_some() {
                        tiles::PrimaryWindowRelayoutPhase::NeedRect
                    } else {
                        tiles::PrimaryWindowRelayoutPhase::Idle
                    };
                }
            } else {
                warn!(
                    screen = screen as i32,
                    "Primary window screen index out of range; skipping"
                );
                layout.screen = None;
                layout.awaiting_screen_confirmation = false;
                layout.relayout_phase = if layout.screen_rect.is_some() {
                    tiles::PrimaryWindowRelayoutPhase::NeedRect
                } else {
                    tiles::PrimaryWindowRelayoutPhase::Idle
                };
            }
        }
        tiles::PrimaryWindowRelayoutPhase::NeedRect => {
            if apply_primary_window_rect(layout, window) {
                layout.relayout_phase = tiles::PrimaryWindowRelayoutPhase::Idle;
            }
        }
    }
}

fn apply_secondary_window_rect(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
) -> bool {
    info!(
        path = %state.descriptor.path.display(),
        applied_rect = ?state.applied_rect,
        target_rect = ?state.descriptor.screen_rect,
        size_w = window.outer_size().width,
        size_h = window.outer_size().height,
        pos_x = window.outer_position().map(|p| p.x).unwrap_or_default(),
        pos_y = window.outer_position().map(|p| p.y).unwrap_or_default(),
        "apply_secondary_window_rect start"
    );
    if state.descriptor.screen_rect.is_some() {
        state.extend_metadata_capture_block(SECONDARY_RECT_CAPTURE_LOAD_GUARD);
    }

    let Some(rect) = state.descriptor.screen_rect else {
        if state.applied_rect.is_some() {
            state.applied_rect = None;
            linux_clear_minimized(window);
        }
        return true;
    };

    if state.applied_rect == Some(rect) {
        info!(
            path = %state.descriptor.path.display(),
            "Rect already applied"
        );
        return true;
    }

    if rect.width == 0 && rect.height == 0 {
        linux_request_minimize(window);
        state.applied_rect = Some(rect);
        state.skip_metadata_capture = true;
        info!(
            path = %state.descriptor.path.display(),
            "Applied minimize rect"
        );
        return true;
    }

    let screen_handle = if let Some(idx) = state.descriptor.screen {
        let screens = collect_sorted_screens(window);
        screens
            .get(idx)
            .cloned()
            .or_else(|| window.current_monitor())
    } else {
        window.current_monitor()
    };
    let Some(screen_handle) = screen_handle else {
        return false;
    };

    let screen_pos = screen_handle.position();
    let screen_size = screen_handle.size();
    if screen_size.width == 0 || screen_size.height == 0 {
        return false;
    }

    let is_full_rect =
        LINUX_MULTI_WINDOW && rect.x == 0 && rect.y == 0 && rect.width == 100 && rect.height == 100;

    if window.fullscreen().is_some() {
        exit_fullscreen(window);
        if !is_full_rect {
            force_windowed(window);
        }
        info!(
            path = %state.descriptor.path.display(),
            "Exited fullscreen before applying rect"
        );
    } else if LINUX_MULTI_WINDOW {
        if !is_full_rect {
            force_windowed(window);
        }
    } else if !is_full_rect {
        window.set_maximized(false);
        linux_clear_minimized(window);
    }
    let screen_width = screen_size.width as i32;
    let screen_height = screen_size.height as i32;

    let requested_width_px = ((rect.width as f64 / 100.0) * screen_width as f64).round() as i32;
    let requested_height_px = ((rect.height as f64 / 100.0) * screen_height as f64).round() as i32;
    let width_px = requested_width_px.clamp(1, screen_width.max(1));
    let height_px = requested_height_px.clamp(1, screen_height.max(1));
    if width_px != requested_width_px || height_px != requested_height_px {
        warn!(
            path = %state.descriptor.path.display(),
            rect = ?rect,
            "Window rect exceeds screen bounds; clamping size"
        );
    }

    let requested_x = screen_pos.x + ((rect.x as f64 / 100.0) * screen_width as f64).round() as i32;
    let requested_y =
        screen_pos.y + ((rect.y as f64 / 100.0) * screen_height as f64).round() as i32;

    let max_x = screen_pos.x + screen_width - width_px;
    let max_y = screen_pos.y + screen_height - height_px;
    let x = requested_x.clamp(screen_pos.x, max_x);
    let y = requested_y.clamp(screen_pos.y, max_y);
    if x != requested_x || y != requested_y {
        warn!(
            path = %state.descriptor.path.display(),
            rect = ?rect,
            "Window rect origin exceeds screen bounds; clamping position"
        );
    }

    if let Ok(current_pos) = window.outer_position() {
        let current_size = window.outer_size();
        if current_pos.x == x
            && current_pos.y == y
            && current_size.width as i32 == width_px
            && current_size.height as i32 == height_px
        {
            state.applied_rect = Some(rect);
            state.skip_metadata_capture = true;
            state.clear_metadata_capture_block();
            state.extend_metadata_capture_block(SECONDARY_RECT_CAPTURE_STABILIZE_GUARD);
            info!(
                path = %state.descriptor.path.display(),
                rect = ?rect,
                size_w = width_px,
                size_h = height_px,
                pos_x = x,
                pos_y = y,
                "Rect already applied"
            );
            return true;
        }
    }

    let _ = window.request_inner_size(PhysicalSize::new(
        width_px.max(1) as u32,
        height_px.max(1) as u32,
    ));
    window.set_outer_position(PhysicalPosition::new(x, y));
    state.applied_rect = Some(rect);
    state.skip_metadata_capture = true;
    state.clear_metadata_capture_block();
    state.extend_metadata_capture_block(SECONDARY_RECT_CAPTURE_STABILIZE_GUARD);
    info!(
        path = %state.descriptor.path.display(),
        rect = ?rect,
        size_w = width_px,
        size_h = height_px,
        pos_x = x,
        pos_y = y,
        "Applied rect"
    );
    true
}

fn assign_window_to_screen(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
    target_monitor: MonitorHandle,
) {
    let screen_pos = target_monitor.position();
    let screen_size = target_monitor.size();
    let window_size = window.outer_size();
    let x = screen_pos.x + (screen_size.width as i32 - window_size.width as i32) / 2;
    let y = screen_pos.y + (screen_size.height as i32 - window_size.height as i32) / 2;

    info!(
        path = %state.descriptor.path.display(),
        screen = state.descriptor.screen.map(|idx| idx as i32).unwrap_or(-1),
        "assign_window_to_screen"
    );
    // Align with Linux path: avoid fullscreen hops on macOS and keep windowed positioning.
    exit_fullscreen(window);
    window.set_visible(true);
    force_windowed(window);
    window.set_outer_position(PhysicalPosition::new(x, y));
    state.skip_metadata_capture = true;
}

fn complete_screen_assignment(
    state: &mut tiles::SecondaryWindowState,
    _window: &WinitWindow,
    reason: &'static str,
) {
    state.awaiting_screen_confirmation = false;
    state.applied_screen = state.descriptor.screen;
    state.relayout_attempts = 0;
    state.relayout_started_at = None;
    // If a rect was already applied, don't re-enter NeedRect and reconfigure swapchain.
    state.relayout_phase = match state.descriptor.screen_rect {
        Some(rect) if state.applied_rect != Some(rect) => {
            state.extend_metadata_capture_block(SECONDARY_RECT_CAPTURE_LOAD_GUARD);
            tiles::SecondaryWindowRelayoutPhase::NeedRect
        }
        _ => {
            state.clear_metadata_capture_block();
            tiles::SecondaryWindowRelayoutPhase::Idle
        }
    };

    info!(
        screen = state
            .descriptor
            .screen
            .map(|idx| idx as i32)
            .unwrap_or(-1),
        path = %state.descriptor.path.display(),
        "{reason}"
    );
}

fn complete_primary_screen_assignment(
    layout: &mut tiles::PrimaryWindowLayout,
    _window: &WinitWindow,
    reason: &'static str,
) {
    layout.awaiting_screen_confirmation = false;
    layout.applied_screen = layout.screen;
    layout.relayout_phase = if layout.screen_rect.is_some() {
        tiles::PrimaryWindowRelayoutPhase::NeedRect
    } else {
        tiles::PrimaryWindowRelayoutPhase::Idle
    };
    layout.relayout_attempts = 0;
    layout.relayout_started_at = None;
    info!(
        screen = layout.screen.map(|idx| idx as i32).unwrap_or(-1),
        "{reason}"
    );
}

fn assign_primary_window_to_screen(
    _layout: &mut tiles::PrimaryWindowLayout,
    window: &WinitWindow,
    target_monitor: MonitorHandle,
) {
    let screen_pos = target_monitor.position();
    let screen_size = target_monitor.size();
    let window_size = window.outer_size();
    let x = screen_pos.x + (screen_size.width as i32 - window_size.width as i32) / 2;
    let y = screen_pos.y + (screen_size.height as i32 - window_size.height as i32) / 2;

    if LINUX_MULTI_WINDOW {
        window.set_fullscreen(None);
        window.set_visible(true);
        force_windowed(window);
    } else {
        // Align with Linux path: avoid fullscreen hops on macOS.
        exit_fullscreen(window);
        window.set_visible(true);
    }
    window.set_outer_position(PhysicalPosition::new(x, y));
}

fn recenter_window_on_screen(window: &WinitWindow, target_monitor: &MonitorHandle) {
    let screen_pos = target_monitor.position();
    let screen_size = target_monitor.size();
    let window_size = window.outer_size();
    // Recenter, puis nudge vers l'intérieur de l'écran pour aider winit à rafraîchir current_monitor().
    let center_x = screen_pos.x + (screen_size.width as i32 - window_size.width as i32) / 2;
    let center_y = screen_pos.y + (screen_size.height as i32 - window_size.height as i32) / 2;
    window.set_outer_position(PhysicalPosition::new(center_x, center_y));

    let nudge_x = screen_pos
        .x
        .saturating_add(10)
        .min(screen_pos.x + screen_size.width as i32 - 1);
    let nudge_y = screen_pos
        .y
        .saturating_add(10)
        .min(screen_pos.y + screen_size.height as i32 - 1);
    window.set_outer_position(PhysicalPosition::new(nudge_x, nudge_y));
    // Secoue légèrement la taille pour forcer un rafraîchissement du monitor courant.
    let size = window.outer_size();
    let _ = window.request_inner_size(size);
}

fn apply_primary_window_rect(
    layout: &mut tiles::PrimaryWindowLayout,
    window: &WinitWindow,
) -> bool {
    let Some(rect) = layout.screen_rect else {
        layout.applied_rect = None;
        linux_clear_minimized(window);
        return true;
    };

    if layout.applied_rect == Some(rect) {
        return true;
    }

    if rect.width == 0 && rect.height == 0 {
        linux_request_minimize(window);
        layout.applied_rect = Some(rect);
        return true;
    }

    let screen_handle = if let Some(idx) = layout.screen {
        let screens = collect_sorted_screens(window);
        screens
            .get(idx)
            .cloned()
            .or_else(|| window.current_monitor())
    } else {
        window.current_monitor()
    };
    let Some(screen_handle) = screen_handle else {
        return false;
    };

    let screen_pos = screen_handle.position();
    let screen_size = screen_handle.size();
    if screen_size.width == 0 || screen_size.height == 0 {
        return false;
    }

    if window.fullscreen().is_some() {
        exit_fullscreen(window);
        force_windowed(window);
    } else if LINUX_MULTI_WINDOW {
        force_windowed(window);
    } else {
        window.set_maximized(false);
        linux_clear_minimized(window);
    }

    let screen_width = screen_size.width as i32;
    let screen_height = screen_size.height as i32;

    let requested_width_px = ((rect.width as f64 / 100.0) * screen_width as f64).round() as i32;
    let requested_height_px = ((rect.height as f64 / 100.0) * screen_height as f64).round() as i32;
    let width_px = requested_width_px.clamp(1, screen_width.max(1));
    let height_px = requested_height_px.clamp(1, screen_height.max(1));
    if width_px != requested_width_px || height_px != requested_height_px {
        warn!("Primary window rect exceeds screen bounds; clamping size (rect={rect:?})");
    }

    let requested_x = screen_pos.x + ((rect.x as f64 / 100.0) * screen_width as f64).round() as i32;
    let requested_y =
        screen_pos.y + ((rect.y as f64 / 100.0) * screen_height as f64).round() as i32;

    let max_x = screen_pos.x + screen_width - width_px;
    let max_y = screen_pos.y + screen_height - height_px;
    let x = requested_x.clamp(screen_pos.x, max_x);
    let y = requested_y.clamp(screen_pos.y, max_y);
    if x != requested_x || y != requested_y {
        warn!(
            "Primary window rect origin exceeds screen bounds; clamping position (rect={rect:?})"
        );
    }

    info!(
        rect = ?rect,
        width_px,
        height_px,
        x,
        y,
        "Applying primary window rect"
    );

    if let Ok(current_pos) = window.outer_position() {
        let current_size = window.outer_size();
        if current_pos.x == x
            && current_pos.y == y
            && current_size.width as i32 == width_px
            && current_size.height as i32 == height_px
        {
            layout.applied_rect = Some(rect);
            return true;
        }
    }
    let _ = window.request_inner_size(PhysicalSize::new(
        width_px.max(1) as u32,
        height_px.max(1) as u32,
    ));
    window.set_outer_position(PhysicalPosition::new(x, y));
    layout.applied_rect = Some(rect);
    true
}

fn capture_secondary_window_screens(
    mut windows: ResMut<tiles::WindowManager>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    window_query: Query<(Entity, &Window)>,
    screens: Query<(Entity, &Monitor)>,
) {
    for state in windows.secondary_mut().iter_mut() {
        #[cfg(target_os = "macos")]
        if matches!(
            state.relayout_phase,
            tiles::SecondaryWindowRelayoutPhase::NeedScreen
        ) && state.awaiting_screen_confirmation
        {
            continue;
        }

        let Some(entity) = state.window_entity else {
            continue;
        };
        let winit_window = winit_windows.get_window(entity);
        let mut updated = false;
        if let Some(window) = winit_window {
            let screens_sorted = collect_sorted_screens(window);
            updated = state.update_descriptor_from_winit_window(window, &screens_sorted);
        }

        if !updated && let Ok((_, window_component)) = window_query.get(entity) {
            state.update_descriptor_from_window(window_component, &screens);
        }
    }
}

fn confirm_secondary_screen_assignment(
    mut moved_events: EventReader<WindowMoved>,
    mut resized_events: EventReader<WindowResized>,
    mut windows: ResMut<tiles::WindowManager>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    let mut touched: Vec<Entity> = Vec::new();
    touched.extend(moved_events.read().map(|evt| evt.window));
    touched.extend(resized_events.read().map(|evt| evt.window));
    touched.sort_unstable();
    touched.dedup();

    for entity in touched {
        let Some(id) = windows.find_secondary_by_entity(entity) else {
            continue;
        };
        let Some(state) = windows.get_secondary_mut(id) else {
            continue;
        };
        if !matches!(
            state.relayout_phase,
            tiles::SecondaryWindowRelayoutPhase::NeedScreen
        ) {
            continue;
        }
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };
        let screens_sorted = collect_sorted_screens(window);
        if window_on_target_screen(state, window, &screens_sorted) {
            complete_screen_assignment(
                state,
                window,
                "Confirmed screen assignment for secondary window",
            );
        }
    }
}

fn track_secondary_window_geometry(
    mut moved_events: EventReader<WindowMoved>,
    mut resized_events: EventReader<WindowResized>,
    mut windows: ResMut<tiles::WindowManager>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    use std::collections::HashMap;

    let mut touched: HashMap<Entity, Option<PhysicalPosition<i32>>> = HashMap::new();
    for evt in moved_events.read() {
        let position = PhysicalPosition::new(evt.position.x, evt.position.y);
        touched.insert(evt.window, Some(position));
    }
    for evt in resized_events.read() {
        touched.entry(evt.window).or_insert(None);
    }

    for (entity, forced_position) in touched {
        let Some(id) = windows.find_secondary_by_entity(entity) else {
            continue;
        };
        let Some(state) = windows.get_secondary_mut(id) else {
            continue;
        };
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };
        let screens_sorted = collect_sorted_screens(window);
        record_window_rect_from_window(state, window, &screens_sorted, forced_position);
    }
}

fn confirm_primary_screen_assignment(
    mut moved_events: EventReader<WindowMoved>,
    mut resized_events: EventReader<WindowResized>,
    mut windows: ResMut<tiles::WindowManager>,
    primary_query: Query<Entity, With<PrimaryWindow>>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
) {
    #[allow(deprecated)]
    let Ok(primary_entity) = primary_query.get_single() else {
        return;
    };

    let mut touched = false;
    for evt in moved_events.read() {
        if evt.window == primary_entity {
            touched = true;
        }
    }
    for evt in resized_events.read() {
        if evt.window == primary_entity {
            touched = true;
        }
    }

    if !touched {
        return;
    }

    let Some(window) = winit_windows.get_window(primary_entity) else {
        return;
    };
    let screens_sorted = collect_sorted_screens(window);
    let layout = windows.primary_layout_mut();
    if matches!(
        layout.relayout_phase,
        tiles::PrimaryWindowRelayoutPhase::NeedScreen
    ) {
        layout.awaiting_screen_confirmation = false;
    }
    let _ = screens_sorted;
}

fn collect_sorted_screens(window: &WinitWindow) -> Vec<MonitorHandle> {
    let mut screens: Vec<MonitorHandle> = window.available_monitors().collect();
    screens.sort_by(|a, b| {
        let result = a
            .position()
            .x
            .cmp(&b.position().x)
            .then(a.position().y.cmp(&b.position().y));
        if result == std::cmp::Ordering::Equal {
            let name_a = a.name();
            let name_b = b.name();
            name_a.cmp(&name_b)
        } else {
            result
        }
    });
    screens
}

fn screens_match(a: &MonitorHandle, b: &MonitorHandle) -> bool {
    if a.position() == b.position() && a.size() == b.size() {
        return true;
    }
    match (a.name(), b.name()) {
        (Some(an), Some(bn)) => an == bn && a.size() == b.size(),
        _ => false,
    }
}

fn collect_screens_from_any_window(
    windows: &bevy::winit::WinitWindows,
) -> Option<Vec<MonitorHandle>> {
    let handle = windows.windows.values().next()?;
    Some(collect_sorted_screens(handle))
}

fn fix_visibility_hierarchy(
    mut commands: Commands,
    inherited_with_parent: Query<&ChildOf, With<InheritedVisibility>>,
    global_with_parent: Query<&ChildOf, With<GlobalTransform>>,
    has_inherited: Query<(), With<InheritedVisibility>>,
    has_global: Query<(), With<GlobalTransform>>,
) {
    for parent in inherited_with_parent.iter() {
        let parent_entity = parent.parent();
        if has_inherited.get(parent_entity).is_err() {
            commands
                .entity(parent_entity)
                .insert(InheritedVisibility::default());
        }
    }

    for parent in global_with_parent.iter() {
        let parent_entity = parent.parent();
        if has_global.get(parent_entity).is_err() {
            commands
                .entity(parent_entity)
                .insert(GlobalTransform::default());
        }
    }
}

fn record_window_rect_from_window(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
    screens: &[MonitorHandle],
    forced_position: Option<PhysicalPosition<i32>>,
) {
    if state.is_metadata_capture_blocked() {
        return;
    }
    if state.skip_metadata_capture {
        state.skip_metadata_capture = false;
        return;
    }
    let size = window.outer_size();
    if size.width == 0 || size.height == 0 {
        return;
    }
    let position = forced_position.or_else(|| window.outer_position().ok());
    let Some(position) = position else {
        return;
    };

    let screen_index = state
        .descriptor
        .screen
        .or_else(|| tiles::screen_index_from_bounds(position, size, screens));
    let Some(idx) = screen_index else {
        return;
    };
    let Some(screen_handle) = screens.get(idx).cloned() else {
        return;
    };
    let screen_pos = screen_handle.position();
    if !event_position_is_reliable(position, screen_pos) {
        return;
    }

    if let Some(rect) = tiles::rect_from_bounds(
        (position.x, position.y),
        (size.width, size.height),
        (screen_pos.x, screen_pos.y),
        (screen_handle.size().width, screen_handle.size().height),
    ) {
        state.descriptor.screen = Some(idx);
        state.descriptor.screen_rect = Some(rect);
    }
}

fn event_position_is_reliable(
    position: PhysicalPosition<i32>,
    screen_position: PhysicalPosition<i32>,
) -> bool {
    if !LINUX_MULTI_WINDOW {
        return true;
    }
    if position.x == 0 && position.y == 0 {
        screen_position.x == 0 && screen_position.y == 0
    } else {
        true
    }
}

fn window_on_target_screen(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
    screens: &[MonitorHandle],
) -> bool {
    if window_on_screen(state.descriptor.screen, window, screens) {
        return true;
    }

    if state.descriptor.screen.is_none()
        && let Some(idx) = detect_window_screen(window, screens)
    {
        state.descriptor.screen = Some(idx);
        return true;
    }

    false
}

fn window_on_screen(
    screen: Option<usize>,
    window: &WinitWindow,
    screens: &[MonitorHandle],
) -> bool {
    let detected = detect_window_screen(window, screens);
    match (screen, detected) {
        (None, _) => true,
        (Some(target), Some(idx)) => idx == target,
        _ => false,
    }
}

fn exit_fullscreen(window: &WinitWindow) {
    window.set_fullscreen(None);
    window.set_maximized(false);
    linux_clear_minimized(window);
    window.set_decorations(true);
}

fn force_windowed(window: &WinitWindow) {
    if !LINUX_MULTI_WINDOW {
        return;
    }
    window.set_visible(true);
    window.set_decorations(true);
    window.set_maximized(false);
}

fn linux_clear_minimized(window: &WinitWindow) {
    if !LINUX_MULTI_WINDOW {
        window.set_minimized(false);
    }
}

fn linux_request_minimize(window: &WinitWindow) {
    window.set_minimized(true);
}

fn detect_window_screen(window: &WinitWindow, screens: &[MonitorHandle]) -> Option<usize> {
    window
        .current_monitor()
        .as_ref()
        .and_then(|current| {
            screens
                .iter()
                .position(|monitor| screens_match(current, monitor))
        })
        .or_else(|| {
            window
                .outer_position()
                .ok()
                .and_then(|pos| tiles::screen_index_from_bounds(pos, window.outer_size(), screens))
        })
}

fn secondary_graph_order_base(id: tiles::SecondaryWindowId) -> isize {
    SECONDARY_GRAPH_ORDER_BASE + SECONDARY_GRAPH_ORDER_STRIDE * id.0 as isize
}

fn capture_primary_window_layout(
    mut windows: ResMut<tiles::WindowManager>,
    primary_query: Query<(Entity, &Window), With<PrimaryWindow>>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    screens: Query<(Entity, &Monitor)>,
) {
    #[allow(deprecated)]
    let Ok((primary_entity, primary_window_component)) = primary_query.get_single() else {
        return;
    };
    let Some(window) = winit_windows.get_window(primary_entity) else {
        return;
    };
    let layout = windows.primary_layout_mut();
    if !matches!(
        layout.relayout_phase,
        tiles::PrimaryWindowRelayoutPhase::Idle
    ) {
        return;
    }

    let screens_sorted = collect_sorted_screens(window);
    let mut captured = false;
    if let Ok(outer_pos) = window.outer_position() {
        let outer_size = window.outer_size();
        let screen_index = window
            .current_monitor()
            .as_ref()
            .and_then(|current| {
                screens_sorted
                    .iter()
                    .position(|screen| screens_match(screen, current))
            })
            .or_else(|| tiles::screen_index_from_bounds(outer_pos, outer_size, &screens_sorted));
        if let Some(idx) = screen_index {
            layout.captured_screen = Some(idx);
            layout.requested_screen = Some(idx);
            if let Some(screen_handle) = screens_sorted.get(idx) {
                let screen_pos = screen_handle.position();
                let screen_size = screen_handle.size();
                if let Some(rect) = tiles::rect_from_bounds(
                    (outer_pos.x, outer_pos.y),
                    (outer_size.width, outer_size.height),
                    (screen_pos.x, screen_pos.y),
                    (screen_size.width, screen_size.height),
                ) {
                    layout.captured_rect = Some(rect);
                    layout.requested_rect = Some(rect);
                    captured = true;
                }
            }
        }
    }

    if !captured {
        let fallback_pos = match primary_window_component.position {
            WindowPosition::At(pos) => pos,
            WindowPosition::Centered(_) | WindowPosition::Automatic => IVec2::ZERO,
        };
        let fallback_size = primary_window_component.resolution.physical_size();

        let mut best: Option<(usize, i32)> = None;
        let mut best_bounds: Option<(IVec2, UVec2)> = None;
        for (index, (_, screen)) in screens.iter().enumerate() {
            let min = screen.physical_position;
            let size = screen.physical_size();
            let max = IVec2::new(min.x + size.x as i32, min.y + size.y as i32);
            if fallback_pos.x >= min.x
                && fallback_pos.x < max.x
                && fallback_pos.y >= min.y
                && fallback_pos.y < max.y
            {
                let distance = (fallback_pos.x - min.x).abs() + (fallback_pos.y - min.y).abs();
                if best.map(|(_, d)| distance < d).unwrap_or(true) {
                    best = Some((index, distance));
                    best_bounds = Some((min, size));
                }
            }
        }

        if let Some((index, _)) = best {
            layout.captured_screen = Some(index);
            layout.requested_screen = Some(index);
            if let Some((screen_pos, screen_size)) = best_bounds
                && let Some(rect) = tiles::rect_from_bounds(
                    (fallback_pos.x, fallback_pos.y),
                    (fallback_size.x, fallback_size.y),
                    (screen_pos.x, screen_pos.y),
                    (screen_size.x, screen_size.y),
                )
            {
                layout.captured_rect = Some(rect);
                layout.requested_rect = Some(rect);
            }
        }
    }
}

fn handle_secondary_close(
    mut events: EventReader<WindowCloseRequested>,
    mut windows: ResMut<tiles::WindowManager>,
    window_query: Query<(Entity, &Window)>,
    screens: Query<(Entity, &Monitor)>,
) {
    let mut to_remove = Vec::new();
    for evt in events.read() {
        if let Some(id) = windows.find_secondary_by_entity(evt.window) {
            to_remove.push(id);
        }
    }

    if !to_remove.is_empty() {
        windows.secondary_mut().retain_mut(|state| {
            let keep = !to_remove.contains(&state.id);
            if !keep
                && let Some((_, window)) = state
                    .window_entity
                    .and_then(|entity| window_query.get(entity).ok())
            {
                state.update_descriptor_from_window(window, &screens);
            }
            keep
        });
    }
}

fn handle_primary_close(
    mut events: EventReader<WindowCloseRequested>,
    primary: Query<Entity, With<PrimaryWindow>>,
    mut exit: EventWriter<AppExit>,
) {
    let Some(primary_entity) = primary.iter().next() else {
        return;
    };

    for evt in events.read() {
        let entity = evt.window;
        if entity == primary_entity {
            exit.write(AppExit::Success);
        }
    }
}

fn secondary_window_container_title(state: &tiles::SecondaryWindowState) -> Option<String> {
    let root = state.tile_state.tree.root()?;
    find_named_container_title(
        &state.tile_state.tree,
        &state.tile_state.container_titles,
        root,
    )
}

fn find_named_container_title(
    tree: &egui_tiles::Tree<tiles::Pane>,
    titles: &HashMap<egui_tiles::TileId, String>,
    tile_id: egui_tiles::TileId,
) -> Option<String> {
    if let Some(title) = titles
        .get(&tile_id)
        .and_then(|value| normalize_title(value))
    {
        return Some(title);
    }

    let tile = tree.tiles.get(tile_id)?;
    if let Tile::Container(container) = tile {
        match container {
            Container::Tabs(tabs) => {
                for child in &tabs.children {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
            Container::Linear(linear) => {
                for child in &linear.children {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
            Container::Grid(grid) => {
                for child in grid.children() {
                    if let Some(found) = find_named_container_title(tree, titles, *child) {
                        return Some(found);
                    }
                }
            }
        }
    }

    None
}

fn normalize_title(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn friendly_title_from_stem(stem: &str) -> Option<String> {
    let words: Vec<String> = stem
        .split(|c: char| !c.is_alphanumeric())
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            let mut chars = segment.chars();
            let mut word = String::new();
            if let Some(first) = chars.next() {
                word.extend(first.to_uppercase());
            }
            for ch in chars {
                word.extend(ch.to_lowercase());
            }
            word
        })
        .filter(|word| !word.is_empty())
        .collect();

    if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    }
}

fn render_secondary_windows(world: &mut World) {
    let window_entries: Vec<(tiles::SecondaryWindowId, Entity, String)> = {
        let windows = world.resource::<tiles::WindowManager>();
        windows
            .secondary()
            .iter()
            .filter_map(|state| {
                state
                    .window_entity
                    .map(|entity| (state.id, entity, compute_secondary_window_title(state)))
            })
            .collect()
    };

    if let Some(mut palette_state) = world.get_resource_mut::<CommandPaletteState>()
        && let Some(target) = palette_state.target_window
        && !window_entries.iter().any(|(id, _, _)| *id == target)
    {
        palette_state.target_window = None;
        if palette_state.auto_open_item.is_none() {
            palette_state.show = false;
            palette_state.filter.clear();
            palette_state.page_stack.clear();
        }
    }

    for (id, entity, desired_title) in window_entries {
        let Ok(mut entity_mut) = world.get_entity_mut(entity) else {
            continue;
        };

        if let Some(mut window) = entity_mut.get_mut::<Window>()
            && window.title != desired_title
        {
            window.title = desired_title;
        }

        entity_mut.insert(ActiveSecondaryWindow);

        let widget_id = format!("secondary_window_{}", id.0);
        world.add_root_widget_with::<tiles::TileSystem, With<ActiveSecondaryWindow>>(
            &widget_id,
            Some(id),
        );
        let palette_widget_id = format!("secondary_command_palette_{}", id.0);
        world.add_root_widget_with::<command_palette::PaletteWindow, With<ActiveSecondaryWindow>>(
            &palette_widget_id,
            Some(id),
        );

        if let Ok(mut entity_mut) = world.get_entity_mut(entity) {
            entity_mut.remove::<ActiveSecondaryWindow>();
        }
    }
}

pub(crate) fn compute_secondary_window_title(state: &tiles::SecondaryWindowState) -> String {
    state
        .descriptor
        .title
        .clone()
        .or_else(|| secondary_window_container_title(state))
        .or_else(|| {
            state
                .descriptor
                .path
                .file_stem()
                .and_then(|s| friendly_title_from_stem(&s.to_string_lossy()))
        })
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "Panel".to_string())
}

fn clamp_viewport_to_window(
    mut pos: Vec2,
    mut size: Vec2,
    window_size: Vec2,
) -> Option<(Vec2, Vec2)> {
    if size.x <= 0.0 || size.y <= 0.0 {
        return None;
    }
    if pos.x >= window_size.x || pos.y >= window_size.y {
        return None;
    }
    if pos.x < 0.0 {
        size.x += pos.x;
        pos.x = 0.0;
    }
    if pos.y < 0.0 {
        size.y += pos.y;
        pos.y = 0.0;
    }
    size.x = size.x.min(window_size.x - pos.x);
    size.y = size.y.min(window_size.y - pos.y);
    if size.x <= 0.0 || size.y <= 0.0 {
        return None;
    }
    Some((pos, size))
}

fn set_camera_viewport(
    window: Query<(&Window, &bevy_egui::EguiContextSettings), With<PrimaryWindow>>,
    mut main_camera_query: Query<
        (Entity, &ViewportRect, Option<&GraphState>, &mut Camera),
        With<MainCamera>,
    >,
) {
    let order_offset = PRIMARY_ORDER_OFFSET;
    let mut next_viewport_order = PRIMARY_VIEWPORT_ORDER_BASE;
    let mut next_graph_order = PRIMARY_GRAPH_ORDER_BASE;
    let mut entries: Vec<_> = main_camera_query
        .iter()
        .map(|(entity, _, graph_state, _)| (entity, graph_state.is_some()))
        .collect();
    // Stable ordering: non-graph cameras first, then graphs; break ties by entity id.
    entries.sort_by_key(|(entity, is_graph)| (*is_graph, entity.index()));

    for (entity, is_graph) in entries {
        let Ok((_, viewport_rect, _graph_state, mut camera)) = main_camera_query.get_mut(entity)
        else {
            continue;
        };
        let order = if is_graph {
            let order = next_graph_order;
            next_graph_order += 1;
            order
        } else {
            let order = next_viewport_order;
            next_viewport_order += 1;
            order
        };
        camera.order = order + order_offset;
        let Some(available_rect) = viewport_rect.0 else {
            camera.is_active = false;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(0, 0),
                physical_size: UVec2::new(1, 1),
                depth: 0.0..1.0,
            });

            continue;
        };
        camera.is_active = true;
        let Some((window, egui_settings)) = window.iter().next() else {
            continue;
        };
        let scale_factor = window.scale_factor() * egui_settings.scale_factor;
        let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = available_rect.size() * scale_factor;
        let viewport_pos = Vec2::new(viewport_pos.x, viewport_pos.y);
        let viewport_size = Vec2::new(viewport_size.x, viewport_size.y);
        let window_size = Vec2::new(
            window.physical_width() as f32,
            window.physical_height() as f32,
        );
        if let Some((clamped_pos, clamped_size)) =
            clamp_viewport_to_window(viewport_pos, viewport_size, window_size)
        {
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(clamped_pos.x as u32, clamped_pos.y as u32),
                physical_size: UVec2::new(clamped_size.x as u32, clamped_size.y as u32),
                depth: 0.0..1.0,
            });
        } else {
            camera.is_active = false;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(0, 0),
                physical_size: UVec2::new(1, 1),
                depth: 0.0..1.0,
            });
        }
    }
}

fn set_secondary_camera_viewport(
    windows: Res<tiles::WindowManager>,
    mut cameras: Query<(&mut Camera, &ViewportRect, Option<&NavGizmoCamera>)>,
    window_query: Query<(&Window, &bevy_egui::EguiContextSettings)>,
) {
    for state in windows.secondary() {
        let Some(window_entity) = state.window_entity else {
            continue;
        };

        let Ok((window, egui_settings)) = window_query.get(window_entity) else {
            continue;
        };
        let scale_factor = window.scale_factor() * egui_settings.scale_factor;

        let mut next_order = 0;

        for &graph in state.graph_entities.iter() {
            let Ok((mut camera, viewport_rect, is_nav_gizmo)) = cameras.get_mut(graph) else {
                continue;
            };
            if is_nav_gizmo.is_some() {
                // Nav gizmo cameras get their order/viewport in dedicated systems.
                continue;
            }

            let Some(available_rect) = viewport_rect.0 else {
                camera.is_active = false;
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(1, 1),
                    depth: 0.0..1.0,
                });
                continue;
            };

            let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
            let viewport_size = available_rect.size() * scale_factor;
            let viewport_pos = Vec2::new(viewport_pos.x, viewport_pos.y);
            let viewport_size = Vec2::new(viewport_size.x, viewport_size.y);

            if viewport_size.x < 1.0 || viewport_size.y < 1.0 {
                camera.is_active = false;
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(1, 1),
                    depth: 0.0..1.0,
                });
                continue;
            }

            let window_size = Vec2::new(
                window.physical_width() as f32,
                window.physical_height() as f32,
            );
            if let Some((clamped_pos, clamped_size)) =
                clamp_viewport_to_window(viewport_pos, viewport_size, window_size)
            {
                camera.is_active = true;
                // Offset secondary cameras to avoid colliding with primary orders.
                let base_order =
                    SECONDARY_GRAPH_ORDER_BASE + SECONDARY_GRAPH_ORDER_STRIDE * state.id.0 as isize;
                let offset = base_order + next_order;
                next_order += 1;
                camera.order = offset;
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(clamped_pos.x as u32, clamped_pos.y as u32),
                    physical_size: UVec2::new(clamped_size.x as u32, clamped_size.y as u32),
                    depth: 0.0..1.0,
                });
            } else {
                camera.is_active = false;
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(1, 1),
                    depth: 0.0..1.0,
                });
            }
        }
    }
}

fn set_nav_gizmo_camera_orders(
    windows: Res<tiles::WindowManager>,
    primary_query: Query<Entity, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera, With<NavGizmoCamera>>,
) {
    let mut base_by_window: HashMap<Entity, isize> = HashMap::new();

    if let Ok(primary) = primary_query.single() {
        base_by_window.insert(
            primary,
            PRIMARY_VIEWPORT_ORDER_BASE + PRIMARY_ORDER_OFFSET + NAV_GIZMO_ORDER_OFFSET,
        );
    }

    for state in windows.secondary() {
        if let Some(window_entity) = state.window_entity {
            let base = SECONDARY_GRAPH_ORDER_BASE
                + SECONDARY_GRAPH_ORDER_STRIDE * state.id.0 as isize
                + NAV_GIZMO_ORDER_OFFSET;
            base_by_window.insert(window_entity, base);
        }
    }

    for mut camera in cameras.iter_mut() {
        let RenderTarget::Window(window_ref) = &camera.target else {
            continue;
        };
        match window_ref {
            WindowRef::Primary => {
                if let Ok(primary) = primary_query.single()
                    && let Some(base) = base_by_window.get(&primary)
                {
                    camera.order = *base;
                }
            }
            WindowRef::Entity(entity) => {
                if let Some(base) = base_by_window.get(entity) {
                    camera.order = *base;
                }
            }
        }
    }
}

fn warn_camera_order_ambiguities(
    cameras: Query<(Entity, &Camera)>,
    primary_query: Query<Entity, With<PrimaryWindow>>,
) {
    let primary = primary_query.iter().next();
    let mut seen: HashMap<(NormalizedWindowRef, isize), Entity> = HashMap::new();
    let mut warned: HashSet<NormalizedWindowRef> = HashSet::new();

    for (entity, camera) in cameras.iter() {
        if !camera.is_active {
            continue;
        }
        if let RenderTarget::Window(window_ref) = &camera.target
            && let Some(norm) = window_ref.normalize(primary)
        {
            let key = (norm, camera.order);
            if let Some(prev) = seen.insert(key, entity)
                && warned.insert(norm)
            {
                warn!(
                    window = ?norm,
                    order = camera.order,
                    first = ?prev,
                    second = ?entity,
                    "Camera order collision on window"
                );
            }
        }
    }
}

fn sync_camera_grid_cell(
    mut query: Query<(Option<&ChildOf>, &mut GridCell<i128>), With<MainCamera>>,
    entity_transform_query: Query<&GridCell<i128>, Without<MainCamera>>,
) {
    for (parent, mut grid_cell) in query.iter_mut() {
        if let Some(parent) = parent
            && let Ok(entity_cell) = entity_transform_query.get(parent.parent())
        {
            *grid_cell = *entity_cell;
        }
    }
}
fn sync_hdr(hdr_enabled: ResMut<HdrEnabled>, mut query: Query<&mut Camera>) {
    for mut cam in query.iter_mut() {
        cam.hdr = hdr_enabled.0;
    }
}
