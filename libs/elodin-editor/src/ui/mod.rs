use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bevy::{
    app::AppExit,
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    // platform::collections::{HashMap, HashSet},
    input::keyboard::Key,
    log::{error, info, warn},
    prelude::*,
    render::camera::{RenderTarget, Viewport},
    window::{
        EnabledButtons, Monitor, NormalizedWindowRef, PresentMode, PrimaryWindow,
        WindowCloseRequested, WindowRef, WindowResolution,
    },
};
#[cfg(target_os = "macos")]
use bevy_defer::AccessResult;
use bevy_defer::{AccessError, AsyncAccess, AsyncCommandsExtension, AsyncPlugin, AsyncWorld};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Color32, Label, RichText},
};
use egui_tiles::{Container, Tile};
#[cfg(target_os = "macos")]
use winit::{
    dpi::{LogicalPosition, LogicalSize},
    monitor::MonitorHandle,
    window::Window as WinitWindow,
};
#[cfg(not(target_os = "macos"))]
use winit::{
    dpi::{LogicalPosition, PhysicalPosition, PhysicalSize},
    monitor::MonitorHandle,
    window::Window as WinitWindow,
};

pub(crate) const DEFAULT_SECONDARY_RECT: WindowRect = WindowRect {
    x: 10,
    y: 10,
    width: 80,
    height: 80,
};
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
    pub const LINUX_MULTI_WINDOW: bool = true;
    pub const PRIMARY_ORDER_OFFSET: isize = 0;
}

#[cfg(not(target_os = "linux"))]
mod platform {
    #[allow(dead_code)]
    pub const LINUX_MULTI_WINDOW: bool = false;
    pub const PRIMARY_ORDER_OFFSET: isize = 1000;
}

#[cfg(not(target_os = "macos"))]
use platform::LINUX_MULTI_WINDOW;
use platform::PRIMARY_ORDER_OFFSET;

use big_space::GridCell;
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::get_scheme;
use self::{command_palette::CommandPaletteState, plot::GraphState, timeline::timeline_slider};
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::{ComponentMetadata, ComponentValue, DbConfig, WindowRect};

use crate::{
    GridHandle, MainCamera,
    plugins::{
        LogicalKeyState,
        navigation_gizmo::{NavGizmoCamera, NavGizmoParent},
    },
    tiles::{WindowId, WindowRelayout},
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
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<timeline_slider::UITick>()
            .init_resource::<timeline::StreamTickOrigin>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_event::<DialogEvent>()
            .add_systems(Update, timeline_slider::sync_ui_tick.before(render_layout))
            .add_systems(Update, actions::spawn_lua_actor)
            .add_systems(Update, shortcuts)
            .add_systems(
                Update,
                (
                    handle_window_close,
                    render_layout,
                    sync_camera_grid_cell,
                    sync_windows,
                    handle_window_relayout_events,
                    set_secondary_camera_viewport,
                    set_camera_viewport,
                    set_nav_gizmo_camera_orders,
                    warn_camera_order_ambiguities,
                )
                    .chain(),
            )
            .add_systems(First, fix_visibility_hierarchy)
            .add_systems(Update, sync_hdr)
            .add_systems(Update, tiles::shortcuts)
            .add_systems(Update, query_plot::auto_bounds)
            .add_systems(Update, dashboard::update_nodes)
            .add_plugins(tiles::plugin)
            .add_plugins(SchematicPlugin)
            .add_plugins(LinePlot3dPlugin)
            .add_plugins(AsyncPlugin::default_settings())
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
    entities_meta: Query<'w, 's, EntityDataReadOnly<'static>>,
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
        // Arrow labels are now rendered using Bevy UI in gizmos.rs
    }
}

pub fn render_layout(
    world: &mut World,
    mut windows: Local<Vec<(Entity, WindowId)>>,
    mut widget_id: Local<String>,
) {
    windows.extend(
        world
            .query::<(Entity, &WindowId)>()
            .iter(world)
            .map(|(id, window_id)| (id, *window_id)),
    );
    for (id, window_id) in windows.drain(..) {
        if window_id.is_primary() {
            world.add_root_widget_to::<MainLayout>(id, "main_layout", ());

            world.add_root_widget_to::<ViewportOverlay>(id, "viewport_overlay", ());

            world.add_root_widget_to::<modal::ModalWithSettings>(id, "modal_graph", ());

            world.add_root_widget_to::<CommandPalette>(id, "command_palette", ());
        } else {
            widget_id.clear();
            let _ = write!(widget_id, "secondary_window_{}", window_id.0);
            world.add_root_widget_to::<tiles::TileSystem>(id, &widget_id, Some(id));
            widget_id.clear();
            let _ = write!(widget_id, "secondary_command_palette_{}", window_id.0);
            world.add_root_widget_to::<command_palette::PaletteWindow>(id, &widget_id, Some(id));
        }
    }
}

fn sync_windows(
    mut commands: Commands,
    mut windows_state: Query<(Entity, &WindowId, &mut tiles::WindowState, Option<&Window>)>,
    mut cameras: Query<&mut Camera>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut existing_map: Local<HashMap<tiles::WindowId, Entity>>,
) {
    let screens_any = winit_windows
        .windows
        .values()
        .next()
        .map(|w| collect_sorted_screens(w));
    if screens_any.is_none() {
        warn!("No screen info available; secondary windows will use default sizing/position");
    }

    for (entity, marker, mut state, window_maybe) in &mut windows_state {
        state.graph_entities = state.tile_state.collect_graph_entities();

        if window_maybe.is_some() {
            existing_map.insert(*marker, entity);
            let window_ref = WindowRef::Entity(entity);
            for (index, &graph) in state.graph_entities.iter().enumerate() {
                if let Ok(mut camera) = cameras.get_mut(graph) {
                    camera.target = RenderTarget::Window(window_ref);
                    camera.is_active = true;
                    let base_order = secondary_graph_order_base(*marker);
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

        let title = compute_secondary_window_title(&state);

        // Try to pre-size and pre-position the window to its target rect to avoid an
        // extra resize pass (and the resulting swapchain churn).
        let (resolution, position, _pre_applied_screen) = if let Some(rect) =
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
            .entity(entity)
            .insert((window_component, *marker))
            .id();

        if let Some(screen) = state.descriptor.screen.as_ref() {
            commands.send_event(WindowRelayout::Screen {
                window: window_entity,
                screen: *screen,
            });
        }

        if let Some(rect) = state.descriptor.screen_rect.as_ref() {
            commands.send_event(WindowRelayout::Rect {
                window: window_entity,
                rect: *rect,
            });
        }
        #[cfg(target_os = "macos")]
        {
            if let (Some(screen_idx), Some(rect)) =
                (state.descriptor.screen, state.descriptor.screen_rect)
            {
                info!(
                    window = %window_entity,
                    target_screen = screen_idx,
                    rect = ?rect,
                    "mac spawn: apply physical rect"
                );
                commands.spawn_task(move || async move {
                    apply_physical_screen_rect(window_entity, screen_idx, rect)
                        .await
                        .ok();
                    Ok(())
                });
            }
        }
        existing_map.insert(*marker, window_entity);
        info!(
            "Created window entity {window_entity} with window id {:?}",
            marker
        );
        let window_ref = WindowRef::Entity(window_entity);
        for (index, &graph) in state.graph_entities.iter().enumerate() {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                camera.target = RenderTarget::Window(window_ref);
                camera.is_active = true;
                let base_order = secondary_graph_order_base(*marker);
                camera.order = base_order + index as isize;
            }
        }
    }
}

/// Wait until a winit window is created or timeout. Return true when found or
/// false on timeout.
async fn wait_for_winit_window(window_id: Entity, timeout: Duration) -> Result<bool, AccessError> {
    let start = Instant::now();
    let winit_windows_async = AsyncWorld.non_send_resource::<bevy::winit::WinitWindows>();
    while start.elapsed() < timeout {
        if winit_windows_async.get(|winit_windows| winit_windows.get_window(window_id).is_some())? {
            return Ok(true);
        }
        AsyncWorld.yield_now().await;
    }
    Ok(false)
}

#[cfg(target_os = "macos")]
pub(crate) async fn apply_physical_screen_rect(
    window_entity: Entity,
    screen_index: usize,
    rect: WindowRect,
) -> AccessResult {
    if !wait_for_winit_window(window_entity, Duration::from_millis(2000)).await? {
        warn!(%window_entity, "apply_physical_screen_rect: winit window not ready");
        return Ok(());
    }

    let target = AsyncWorld.run(
        |world| -> Option<(LogicalPosition<f64>, LogicalSize<f64>)> {
            let winit_windows = world.get_non_send_resource::<bevy::winit::WinitWindows>()?;
            let any_window = winit_windows.windows.values().next()?;
            let screens = collect_sorted_screens(any_window);
            log_screens("macos.load.screens", &screens);
            let screen = screens.get(screen_index)?;
            let pos = screen_physical_position(screen);
            let size = screen_physical_size(screen);
            if size.width <= 0.0 || size.height <= 0.0 {
                return None;
            }
            let req_x = pos.x + (rect.x as f64 / 100.0) * size.width;
            let req_y = pos.y + (rect.y as f64 / 100.0) * size.height;
            let req_w = ((rect.width as f64 / 100.0) * size.width).round().max(1.0);
            let req_h = ((rect.height as f64 / 100.0) * size.height)
                .round()
                .max(1.0);

            let max_x = pos.x + size.width - req_w;
            let max_y = pos.y + size.height - req_h;
            let clamped_x = req_x.clamp(pos.x, max_x.max(pos.x));
            let clamped_y = req_y.clamp(pos.y, max_y.max(pos.y));
            info!(
                target_screen = screen_index,
                screen_pos = ?pos,
                screen_size = ?size,
                req_pos = ?LogicalPosition::new(req_x, req_y),
                req_size = ?LogicalSize::new(req_w, req_h),
                clamped_pos = ?LogicalPosition::new(clamped_x, clamped_y),
                "mac apply_physical_screen_rect request"
            );

            Some((
                LogicalPosition::new(clamped_x, clamped_y),
                LogicalSize::new(req_w, req_h),
            ))
        },
    );

    let Some((pos, size)) = target else {
        warn!(
            ?screen_index,
            "apply_physical_screen_rect: no screen found for index"
        );
        return Ok(());
    };

    AsyncWorld.run(move |world| {
        let Some(winit_windows) = world.get_non_send_resource::<bevy::winit::WinitWindows>() else {
            return;
        };
        if let Some(window) = winit_windows.get_window(window_entity) {
            window.set_visible(true);
            window.set_outer_position(pos);
            let _ = window.request_inner_size(size);
            info!(
                %window_entity,
                target_screen = screen_index,
                pos = ?pos,
                size = ?size,
                "mac apply_physical_screen_rect applied"
            );
        }
    });

    Ok(())
}

/// Wait for a window to change to a target screen or timeout.
#[cfg(not(target_os = "macos"))]
async fn wait_for_window_to_change_screens(
    window_id: Entity,
    target_screen: usize,
    timeout: Duration,
) -> Result<bool, AccessError> {
    let start = Instant::now();
    let winit_windows_async = AsyncWorld.non_send_resource::<bevy::winit::WinitWindows>();
    while start.elapsed() < timeout {
        match winit_windows_async.get(|winit_windows| {
            let Some(window) = winit_windows.get_window(window_id) else {
                error!(%window_id, "No winit window in change screen");
                return None;
            };

            let screens = collect_sorted_screens(window);
            if let Some(screen) = detect_window_screen(window, &screens)
                && screen == target_screen
            {
                return Some(true);
            }
            None
        }) {
            Ok(Some(result)) => {
                return Ok(result);
            }
            Err(e) => {
                return Err(e);
            }
            _ => {}
        }

        AsyncWorld.yield_now().await;
    }
    Ok(false)
}

#[cfg(not(target_os = "macos"))]
async fn apply_window_screen(entity: Entity, screen: usize) -> Result<(), bevy_defer::AccessError> {
    info!(
        window = %entity,
        %screen,
        "apply_window_screen start"
    );
    if !wait_for_winit_window(entity, Duration::from_millis(2000)).await? {
        error!(%entity, "Unable to apply window to screen: winit window not found.");
        return Ok(());
    }
    let window_states = AsyncWorld.query::<&tiles::WindowState>();
    let mut state = window_states
        .entity(entity)
        .get_mut(|state| state.clone())?;
    let winit_windows_async = AsyncWorld.non_send_resource::<bevy::winit::WinitWindows>();
    // This might need to loop.
    let target_monitor_maybe = winit_windows_async.get(|winit_windows| {
        let Some(window) = winit_windows.get_window(entity) else {
            error!(%entity, "No winit window in apply window screen");
            return None;
        };

        let screens = collect_sorted_screens(window);

        if window_on_target_screen(&mut state, window, &screens) {
            if LINUX_MULTI_WINDOW {
                exit_fullscreen(window);
                force_windowed(window);
            } else if window.fullscreen().is_some() {
                exit_fullscreen(window);
            }
            return None;
        }

        if let Some(target_monitor) = screens.get(screen).cloned() {
            assign_window_to_screen(window, target_monitor.clone());
            // We have to do some retries in an async context. Inside this
            // `.get()` we're not in an async context.
            Some(target_monitor)
        } else {
            warn!(
                screen,
                path = ?state.descriptor.path,
                "screen out of range; skipping screen assignment"
            );
            state.descriptor.screen = None;
            warn!(%entity, "screen out of range");
            None
        }
    })?;
    if let Some(target_monitor) = target_monitor_maybe {
        // We can't do this await with the `winit_windows_async.get(|| {...})` block.
        let success =
            wait_for_window_to_change_screens(entity, screen, Duration::from_millis(1000)).await?;
        winit_windows_async.get(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                error!(%entity, "No winit window in apply screen");
                return;
            };
            let screens = collect_sorted_screens(window);
            let detected_screen = detect_window_screen(window, &screens);
            let on_target = detected_screen == Some(screen);
            if success || on_target {
                if LINUX_MULTI_WINDOW {
                    exit_fullscreen(window);
                    force_windowed(window);
                } else if window.fullscreen().is_some() {
                    exit_fullscreen(window);
                }
            } else {
                recenter_window_on_screen(window, &target_monitor);
            }
        })?;
    }
    Ok(())
}
fn handle_window_relayout_events(
    mut relayout_events: EventReader<WindowRelayout>,
    mut commands: Commands,
    mut per_window: Local<HashMap<Entity, Vec<WindowRelayout>>>,
) {
    if relayout_events.is_empty() {
        return;
    }
    per_window.clear();
    for relayout_event in relayout_events.read() {
        match relayout_event {
            e @ WindowRelayout::Screen { window, screen: _ } => {
                per_window.entry(*window).or_default().push(e.clone());
            }
            e @ WindowRelayout::Rect { window, rect: _ } => {
                per_window.entry(*window).or_default().push(e.clone());
            }
            WindowRelayout::UpdateDescriptors => {}
        }
    }

    for (_id, relayout_events) in per_window.drain() {
        commands.spawn_task(move || async {
            for relayout_event in relayout_events {
                match relayout_event {
                    WindowRelayout::Screen { window, screen } => {
                        info!(
                            target_screen = screen,
                            "Attempting secondary screen assignment"
                        );
                        #[cfg(target_os = "macos")]
                        {
                            let window_states = AsyncWorld.query::<&tiles::WindowState>();
                            if let Ok(Some(rect)) = window_states
                                .entity(window)
                                .get(|state| state.descriptor.screen_rect)
                            {
                                apply_physical_screen_rect(window, screen, rect).await.ok();
                                continue;
                            }
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            apply_window_screen(window, screen).await?;
                        }
                    }
                    WindowRelayout::Rect { window, rect } => {
                        #[cfg(target_os = "macos")]
                        {
                            if let Some(screen) = AsyncWorld
                                .query::<&tiles::WindowState>()
                                .entity(window)
                                .get(|state| state.descriptor.screen)
                                .ok()
                                .flatten()
                            {
                                apply_physical_screen_rect(window, screen, rect).await.ok();
                                continue;
                            }
                            apply_physical_screen_rect(window, 0, rect).await.ok();
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            apply_window_rect(rect, window, Duration::from_millis(1000)).await?;
                        }
                    }
                    WindowRelayout::UpdateDescriptors => {
                        unreachable!();
                    }
                }
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "macos"))]
async fn apply_window_rect(
    rect: WindowRect,
    entity: Entity,
    timeout: Duration,
) -> Result<(), AccessError> {
    info!(
        window = %entity,
        ?rect,
        "apply_window_rect start"
    );

    let window_states = AsyncWorld.query::<&tiles::WindowState>();
    let state = window_states.entity(entity).get(|state| state.clone())?;

    let winit_windows_async = AsyncWorld.non_send_resource::<bevy::winit::WinitWindows>();

    let is_full_rect =
        LINUX_MULTI_WINDOW && rect.x == 0 && rect.y == 0 && rect.width == 100 && rect.height == 100;

    let start = Instant::now();
    let mut wait = true;
    while wait && start.elapsed() < timeout {
        AsyncWorld.yield_now().await;
        wait = winit_windows_async.get(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                error!(%entity, "No winit window in apply rect");
                return true;
            };

            if rect.width == 0 && rect.height == 0 {
                linux_request_minimize(window);
                info!(
                    // path = %state.descriptor.path.display(),
                    "Applied minimize rect"
                );
                return false;
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
                return true;
            };
            let monitor_name = screen_handle
                .name()
                .unwrap_or_else(|| "unknown".to_string());
            let screen_pos = screen_handle.position();
            let screen_size = screen_handle.size();
            if screen_size.width == 0 || screen_size.height == 0 {
                return true;
            }

            if window.fullscreen().is_some() {
                exit_fullscreen(window);
                if !is_full_rect {
                    force_windowed(window);
                }
                info!(
                    // path = %state.descriptor.path.display(),
                    "Exited fullscreen before applying rect"
                );
            } else if LINUX_MULTI_WINDOW {
                force_windowed(window);
            } else if !is_full_rect {
                window.set_maximized(false);
                linux_clear_minimized(window);
            }
            let screen_width = screen_size.width as i32;
            let screen_height = screen_size.height as i32;

            let requested_width_px =
                ((rect.width as f64 / 100.0) * screen_width as f64).round() as i32;
            let requested_height_px =
                ((rect.height as f64 / 100.0) * screen_height as f64).round() as i32;
            let width_px = requested_width_px.clamp(1, screen_width.max(1));
            let height_px = requested_height_px.clamp(1, screen_height.max(1));
            if width_px != requested_width_px || height_px != requested_height_px {
                warn!(
                    path = ?state.descriptor.path,
                    rect = ?rect,
                    "Window rect exceeds screen bounds; clamping size"
                );
            }

            let requested_x =
                screen_pos.x + ((rect.x as f64 / 100.0) * screen_width as f64).round() as i32;
            let requested_y =
                screen_pos.y + ((rect.y as f64 / 100.0) * screen_height as f64).round() as i32;

            let max_x = screen_pos.x + screen_width - width_px;
            let max_y = screen_pos.y + screen_height - height_px;
            let x = requested_x.clamp(screen_pos.x, max_x);
            let y = requested_y.clamp(screen_pos.y, max_y);
            if x != requested_x || y != requested_y {
                warn!(
                    path = ?state.descriptor.path,
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
                    info!(
                        path = ?state.descriptor.path,
                        rect = ?rect,
                        screen = state.descriptor.screen,
                        monitor = %monitor_name,
                        screen_pos = ?screen_pos,
                        screen_size = ?screen_size,
                        req_x = requested_x,
                        req_y = requested_y,
                        req_w = requested_width_px,
                        req_h = requested_height_px,
                        size_w = width_px,
                        size_h = height_px,
                        pos_x = x,
                        pos_y = y,
                        "Rect already applied"
                    );
                    return false;
                }
            }

            let _ = window.request_inner_size(PhysicalSize::new(
                width_px.max(1) as u32,
                height_px.max(1) as u32,
            ));
            window.set_outer_position(PhysicalPosition::new(x, y));
            info!(
                // path = %state.descriptor.path.display(),
                rect = ?rect,
                screen = state.descriptor.screen,
                monitor = %monitor_name,
                screen_pos = ?screen_pos,
                screen_size = ?screen_size,
                req_x = requested_x,
                req_y = requested_y,
                req_w = requested_width_px,
                req_h = requested_height_px,
                size_w = width_px,
                size_h = height_px,
                pos_x = x,
                pos_y = y,
                "Applied rect"
            );
            false
        })?;
    }

    #[cfg(target_os = "macos")]
    if is_full_rect {
        // After layout is applied, request a maximize to keep native decorations.
        // Small delay lets the WM settle before the maximize request.
        AsyncWorld.sleep(Duration::from_millis(100)).await;
        winit_windows_async.get(|winit_windows| {
            if let Some(window) = winit_windows.get_window(entity) {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                }
                window.set_visible(true);
                #[cfg(not(target_os = "macos"))]
                window.set_decorations(true);
                window.set_maximized(true);
            }
        })?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn assign_window_to_screen(
    // state: &mut tiles::WindowState,
    window: &WinitWindow,
    target_monitor: MonitorHandle,
) {
    let screen_pos = target_monitor.position();
    let screen_size = target_monitor.size();
    let window_size = window.outer_size();
    let x = screen_pos.x + (screen_size.width as i32 - window_size.width as i32) / 2;
    let y = screen_pos.y + (screen_size.height as i32 - window_size.height as i32) / 2;

    // Align with Linux path: avoid fullscreen hops on macOS and keep windowed positioning.
    exit_fullscreen(window);
    window.set_visible(true);
    force_windowed(window);
    window.set_outer_position(PhysicalPosition::new(x, y));
}

#[cfg(not(target_os = "macos"))]
fn recenter_window_on_screen(window: &WinitWindow, target_monitor: &MonitorHandle) {
    let screen_pos = target_monitor.position();
    let screen_size = target_monitor.size();
    let window_size = window.outer_size();
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
    let size = window.outer_size();
    let _ = window.request_inner_size(size);
}

/// Runs as a one-off system to capture the window descriptor. Does not run
/// every frameñ.
pub(crate) fn capture_window_screens_oneoff(
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut window_query: Query<(Entity, &Window, &mut tiles::WindowState)>,
    screens: Query<(Entity, &Monitor)>,
) {
    for (entity, window_component, mut state) in &mut window_query {
        let winit_window = winit_windows.get_window(entity);
        let mut updated = false;
        if let Some(window) = winit_window {
            let screens_sorted = collect_sorted_screens(window);
            updated = state.update_descriptor_from_winit_window(window, &screens_sorted);
            let screen_seen = state.descriptor.screen;
            // Fallback: best-effort detection if current_monitor is not reliable.
            if (!updated || screen_seen.is_none())
                && let Some(idx) = detect_window_screen(window, &screens_sorted)
            {
                state.descriptor.screen = Some(idx);
                updated = true;
            }
        }
        if !updated {
            state.update_descriptor_from_window(window_component, &screens);
        }
    }
}

#[cfg(target_os = "macos")]
static SCREEN_CACHE: OnceLock<Vec<MonitorHandle>> = OnceLock::new();

fn collect_sorted_screens(window: &WinitWindow) -> Vec<MonitorHandle> {
    let mut screens: Vec<MonitorHandle> = window.available_monitors().collect();

    screens.sort_by(|a, b| {
        let p_a = screen_physical_position(a);
        let p_b = screen_physical_position(b);
        p_a.x
            .partial_cmp(&p_b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                p_a.y
                    .partial_cmp(&p_b.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                let name_a = a.name();
                let name_b = b.name();
                name_a.cmp(&name_b)
            })
    });

    #[cfg(target_os = "macos")]
    {
        let _ = SCREEN_CACHE.set(screens.clone());
    }
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

#[cfg(target_os = "macos")]
fn log_screens(label: &str, screens: &[MonitorHandle]) {
    info!(log_label = label, count = screens.len(), "screen list");
    for (idx, screen) in screens.iter().enumerate() {
        let pos = screen_physical_position(screen);
        let size = screen_physical_size(screen);
        let scale = screen.scale_factor();
        let name = screen.name().unwrap_or_else(|| "unknown".to_string());
        info!(
            log_label = label,
            idx,
            name,
            pos = ?pos,
            size = ?size,
            scale,
            "screen"
        );
    }
}

#[cfg(target_os = "macos")]
fn screen_physical_position(screen: &MonitorHandle) -> LogicalPosition<f64> {
    let pos = screen.position();
    LogicalPosition::new(pos.x as f64, pos.y as f64)
}

#[cfg(target_os = "macos")]
fn screen_physical_size(screen: &MonitorHandle) -> LogicalSize<f64> {
    let size = screen.size();
    let scale = screen.scale_factor().max(0.0001);
    LogicalSize::new(size.width as f64 / scale, size.height as f64 / scale)
}

#[cfg(not(target_os = "macos"))]
fn screen_physical_position(screen: &MonitorHandle) -> LogicalPosition<f64> {
    let pos = screen.position();
    LogicalPosition::new(pos.x as f64, pos.y as f64)
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

pub(crate) fn update_primary_descriptor_path(
    db_config: Res<DbConfig>,
    mut q: Query<(&tiles::WindowId, &mut tiles::WindowState)>,
) {
    let Some(path) = db_config.schematic_path() else {
        return;
    };
    for (id, mut state) in q.iter_mut() {
        if id.is_primary() {
            state.descriptor.path = Some(PathBuf::from(path));
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn window_on_target_screen(
    state: &mut tiles::WindowState,
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

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
fn exit_fullscreen(window: &WinitWindow) {
    #[cfg(target_os = "macos")]
    {
        if window.fullscreen().is_some() {
            window.set_fullscreen(None);
        }
    }
    #[cfg(not(target_os = "macos"))]
    window.set_fullscreen(None);
    window.set_maximized(false);
    linux_clear_minimized(window);
    #[cfg(not(target_os = "macos"))]
    window.set_decorations(true);
}

#[cfg(not(target_os = "macos"))]
fn force_windowed(window: &WinitWindow) {
    if !LINUX_MULTI_WINDOW {
        return;
    }
    window.set_visible(true);
    #[cfg(not(target_os = "macos"))]
    window.set_decorations(true);
    window.set_maximized(false);
}

#[cfg(not(target_os = "macos"))]
fn linux_clear_minimized(window: &WinitWindow) {
    if !LINUX_MULTI_WINDOW {
        window.set_minimized(false);
    }
}

#[cfg(not(target_os = "macos"))]
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

fn secondary_graph_order_base(id: tiles::WindowId) -> isize {
    SECONDARY_GRAPH_ORDER_BASE + SECONDARY_GRAPH_ORDER_STRIDE * id.0 as isize
}

fn handle_window_close(
    mut events: EventReader<WindowCloseRequested>,
    primary: Query<&WindowId, With<PrimaryWindow>>,
    mut exit: EventWriter<AppExit>,
) {
    for evt in events.read() {
        let entity = evt.window;
        if primary
            .get(entity)
            .map(|window_id| window_id.is_primary())
            .unwrap_or(false)
        {
            exit.write(AppExit::Success);
        }
    }
}

fn secondary_window_container_title(state: &tiles::WindowState) -> Option<String> {
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

pub(crate) fn compute_secondary_window_title(state: &tiles::WindowState) -> String {
    state
        .descriptor
        .title
        .clone()
        .or_else(|| secondary_window_container_title(state))
        .or_else(|| {
            state
                .descriptor
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
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
    mut entries: Local<Vec<(Entity, bool)>>,
) {
    let order_offset = PRIMARY_ORDER_OFFSET;
    let mut next_viewport_order = PRIMARY_VIEWPORT_ORDER_BASE;
    let mut next_graph_order = PRIMARY_GRAPH_ORDER_BASE;
    entries.clear();
    entries.extend(
        main_camera_query
            .iter()
            .map(|(entity, _, graph_state, _)| (entity, graph_state.is_some())),
    );
    // Stable ordering: non-graph cameras first, then graphs; break ties by entity id.
    entries.sort_by_key(|(entity, is_graph)| (*is_graph, entity.index()));

    for (entity, is_graph) in &entries {
        let Ok((_, viewport_rect, _graph_state, mut camera)) = main_camera_query.get_mut(*entity)
        else {
            continue;
        };
        let order = if *is_graph {
            let order = next_graph_order;
            next_graph_order += 1;
            order
        } else {
            let order = next_viewport_order;
            // Increment by 2 to leave room for nav gizmo camera (order + 1)
            next_viewport_order += 2;
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
        let window_size: Vec2 = window.physical_size().as_vec2();
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
    mut cameras: Query<(&mut Camera, &ViewportRect, Option<&NavGizmoCamera>)>,
    window_query: Query<(
        Entity,
        &Window,
        &tiles::WindowId,
        &tiles::WindowState,
        &bevy_egui::EguiContextSettings,
    )>,
) {
    for (_window_entity, window, id, state, egui_settings) in &window_query {
        if id.is_primary() {
            continue;
        }
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
                    SECONDARY_GRAPH_ORDER_BASE + SECONDARY_GRAPH_ORDER_STRIDE * id.0 as isize;
                let offset = base_order + next_order;
                // Increment by 2 to leave room for nav gizmo camera (order + 1)
                next_order += 2;
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
    mut cameras: Query<(&mut Camera, &NavGizmoParent), With<NavGizmoCamera>>,
    main_cameras: Query<&Camera, Without<NavGizmoCamera>>,
) {
    for (mut camera, parent) in cameras.iter_mut() {
        if let Ok(main) = main_cameras.get(parent.main_camera) {
            camera.order = main.order + NAV_GIZMO_ORDER_OFFSET;
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
fn sync_hdr(hdr_enabled: Res<HdrEnabled>, mut query: Query<&mut Camera>) {
    for mut cam in query.iter_mut() {
        cam.hdr = hdr_enabled.0;
    }
}
