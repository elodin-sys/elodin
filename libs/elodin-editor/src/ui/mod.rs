use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use std::fmt::Write;

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
        WindowCloseRequested, WindowMoved, WindowRef, WindowResized, WindowResolution,
    },
};
use bevy_defer::{AccessError, AsyncAccess, AsyncCommandsExtension, AsyncPlugin, AsyncWorld};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Align2, Color32, Label, RichText},
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

use big_space::{GridCell, precision::GridPrecision};
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::{ColorExt, get_scheme};
use self::{command_palette::CommandPaletteState, plot::GraphState, timeline::timeline_slider};
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::{
    ComponentMetadata, ComponentValue, ComponentValue as WktComponentValue, VectorArrow3d,
    WindowRect,
};

use crate::{
    GridHandle, MainCamera,
    plugins::{
        LogicalKeyState,
        gizmos::{MIN_ARROW_LENGTH_SQUARED, evaluate_vector_arrow},
        navigation_gizmo::NavGizmoCamera,
    },
    tiles::{WindowId, WindowRelayout},
    vector_arrow::VectorArrowState,
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

#[derive(Clone, Copy)]
struct CachedLabel {
    screen: egui::Pos2,
    anchor_world: Vec3,
    cam_pos: Vec3,
    cam_rot: Quat,
}

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
            .add_systems(First, warn_camera_order_ambiguities)
            .add_systems(
                Update,
                (handle_window_close,
                 render_layout,
                 sync_camera_grid_cell,
                 sync_windows,
                 handle_window_relayout_events,
                 track_window_geometry,
                 set_secondary_camera_viewport,
                 set_camera_viewport,
                 set_nav_gizmo_camera_orders,
                 warn_camera_order_ambiguities,
                 ).chain()
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
    vector_arrows: Query<'w, 's, (Entity, &'static VectorArrow3d, &'static VectorArrowState)>,
    component_values: Query<'w, 's, &'static WktComponentValue>,
    entity_map: Res<'w, impeller2_bevy::EntityMap>,
    floating_origin: Res<'w, big_space::FloatingOriginSettings>,
    cameras: Query<
        'w,
        's,
        (
            &'static Camera,
            &'static GlobalTransform,
            &'static GridCell<i128>,
        ),
        With<MainCamera>,
    >,
    label_cache: Local<'s, HashMap<Entity, CachedLabel>>,
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
        let mut state_mut = state.get_mut(world);

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

        // Overlay labels for vector_arrows (simple, stable path)
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("gizmo_labels"),
        ));
        let font_id = egui::FontId::proportional(12.0);
        let shadow_color = Color32::from_rgba_unmultiplied(0, 0, 0, 180);
        let screen_offset = egui::vec2(0.0, -8.0);

        let Some((camera, camera_transform, cam_cell)) =
            state_mut.cameras.iter().find(|(cam, _, _)| cam.is_active)
        else {
            return;
        };

        let cam_pos = camera_transform.translation();
        let cam_rot = camera_transform.to_scale_rotation_translation().1;

        let mut seen = HashSet::new();

        for (entity, arrow, arrow_state) in state_mut.vector_arrows.iter() {
            if !arrow.display_name {
                state_mut.label_cache.remove(&entity);
                continue;
            }
            let Some(result) = evaluate_vector_arrow(
                arrow,
                arrow_state,
                &state_mut.entity_map,
                &state_mut.component_values,
            ) else {
                state_mut.label_cache.remove(&entity);
                continue;
            };
            let Some(name) = result.name.as_ref() else {
                state_mut.label_cache.remove(&entity);
                continue;
            };

            let (start_cell, start_local) = state_mut
                .floating_origin
                .translation_to_grid::<i128>(result.start);
            let (end_cell, end_local) = state_mut
                .floating_origin
                .translation_to_grid::<i128>(result.end);

            let edge = state_mut.floating_origin.grid_edge_length();
            let to_cam = |cell: GridCell<i128>, local: Vec3| {
                let dx = (cell.x.as_f64() - cam_cell.x.as_f64()) as f32 * edge;
                let dy = (cell.y.as_f64() - cam_cell.y.as_f64()) as f32 * edge;
                let dz = (cell.z.as_f64() - cam_cell.z.as_f64()) as f32 * edge;
                local + Vec3::new(dx, dy, dz)
            };

            let start = to_cam(start_cell, start_local);
            let end = to_cam(end_cell, end_local);
            let direction = end - start;
            if direction.length_squared() <= MIN_ARROW_LENGTH_SQUARED as f32 {
                state_mut.label_cache.remove(&entity);
                continue;
            }

            // Reuse cached screen position if arrow and camera didn't move.
            let pos_stable = |a: Vec3, b: Vec3| (a - b).length_squared() <= 0.7 * 0.7; // ~0.7 m
            let rot_stable = |a: Quat, b: Quat| {
                // Use dot to be insensitive to quaternion sign; small angular delta -> dot ~ 1
                (1.0 - a.dot(b).abs()) <= 1.0e-3
            };

            let label_position = result.label_position;
            let anchor_world = start + direction * label_position;

            if let Some(cached) = state_mut.label_cache.get(&entity)
                && pos_stable(cached.anchor_world, anchor_world)
                && pos_stable(cached.cam_pos, cam_pos)
                && rot_stable(cached.cam_rot, cam_rot)
            {
                seen.insert(entity);
                let screen_pos = cached.screen;
                painter.text(
                    screen_pos + egui::vec2(1.0, 1.0),
                    Align2::CENTER_CENTER,
                    name,
                    font_id.clone(),
                    shadow_color,
                );
                painter.text(
                    screen_pos,
                    Align2::CENTER_CENTER,
                    name,
                    font_id.clone(),
                    readable_label_color(result.color),
                );
                continue;
            }

            let dir_norm = direction.try_normalize().unwrap_or(Vec3::ZERO);
            let along_offset = if label_position > 0.9 { 0.06 } else { 0.04 };
            let label_world = anchor_world + dir_norm * along_offset;

            let Ok(anchor_screen) = camera.world_to_viewport(camera_transform, anchor_world) else {
                state_mut.label_cache.remove(&entity);
                continue;
            };

            let mut screen_pos = if let Ok(offset_screen) =
                camera.world_to_viewport(camera_transform, label_world)
            {
                // Offset in screen space following the arrow direction for a more “attached” feel
                let delta = offset_screen - anchor_screen;
                egui::pos2(anchor_screen.x + delta.x, anchor_screen.y + delta.y)
            } else {
                egui::pos2(anchor_screen.x, anchor_screen.y)
            };

            // Nudge label sideways in screen space to keep it from sitting on top of the shaft.
            if let Ok(offset_screen) = camera.world_to_viewport(camera_transform, label_world) {
                let delta = offset_screen - anchor_screen;
                let perp = egui::vec2(-delta.y, delta.x);
                let len = perp.length();
                if len > 0.001 {
                    let side_offset = 10.0;
                    let normalized = perp * (side_offset / len);
                    screen_pos += normalized;
                }
            }

            // If the new projection is within 3px of the cached one, reuse cached to avoid micro-jitter.
            if let Some(cached) = state_mut.label_cache.get(&entity) {
                let delta = screen_pos - cached.screen;
                if delta.length_sq() <= 9.0 {
                    screen_pos = cached.screen;
                }
            }

            screen_pos += screen_offset;
            screen_pos.x = screen_pos.x.round();
            screen_pos.y = screen_pos.y.round();

            state_mut.label_cache.insert(
                entity,
                CachedLabel {
                    screen: screen_pos,
                    anchor_world,
                    cam_pos,
                    cam_rot,
                },
            );
            seen.insert(entity);

            painter.text(
                screen_pos + egui::vec2(1.0, 1.0),
                Align2::CENTER_CENTER,
                name,
                font_id.clone(),
                shadow_color,
            );
            painter.text(
                screen_pos,
                Align2::CENTER_CENTER,
                name,
                font_id.clone(),
                readable_label_color(result.color),
            );
        }

        // Drop cache entries for arrows that disappeared.
        state_mut
            .label_cache
            .retain(|entity, _| seen.contains(entity));
    }
}

fn readable_label_color(color: Color) -> Color32 {
    let color32 = Color32::from_bevy(color);
    Color32::from_rgba_unmultiplied(color32.r(), color32.g(), color32.b(), 255)
}

pub fn render_layout(world: &mut World,
                     mut windows: Local<Vec<(Entity, WindowId)>>,
                     mut widget_id: Local<String>,
) {
    windows.extend(world.query::<(Entity, &WindowId)>().iter(world).map(|(id, window_id)| (id, window_id.clone())));
    let palette_window = world.get_resource_mut::<CommandPaletteState>()
        .and_then(|palette_state| palette_state.target_window.clone());
    for (id, window_id) in windows.drain(..) {
        if window_id.is_primary() {
            world.add_root_widget_to::<MainLayout>(id, "main_layout", ());

            world.add_root_widget_to::<ViewportOverlay>(id, "viewport_overlay", ());

            world.add_root_widget_to::<modal::ModalWithSettings>(id, "modal_graph", ());

            world.add_root_widget_to::<CommandPalette>(id, "command_palette", ());

            // BUG: This plus button is not working on the main window. I tried adding this but to no avail.
            // world.add_root_widget_to::<command_palette::PaletteWindow>(id, "palette_window", None);
        } else {
            widget_id.clear();
            let _ = write!(widget_id, "secondary_window_{}", window_id.0);
            world.add_root_widget_to::<tiles::TileSystem>(id, &widget_id, Some(id));
            // if palette_window.map(|window| window == id).unwrap_or(false) {
                widget_id.clear();
                let _ = write!(widget_id, "secondary_command_palette_{}", window_id.0);
                // This doesn't seem to show anything.
                // world.add_root_widget_to::<CommandPalette>(id, &widget_id, ());
                world.add_root_widget_to::<command_palette::PaletteWindow>(
                    id,
                    &widget_id,
                    Some(id),
                );
            // }
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
    // existing_map.clear();
    // for (entity, marker, state) in existing.iter() {
    //     // Add secondary windows.
    //     if !marker.is_primary() {
    //         existing_map.insert(*marker, entity);
    //     }
    // }
    let screens_any = collect_screens_from_any_window(&winit_windows);
    if screens_any.is_none() {
        warn!("No screen info available; secondary windows will use default sizing/position");
    }

    // for (id, entity) in existing_map.drain() {
    //     if windows.find_secondary(id).is_none() {
    //         // Despawn windows not represented as secondary windows.
    //         commands.entity(entity).despawn();
    //         existing_map.remove(&id);
    //     }
    // }

    for (entity, marker, mut state, window_maybe) in &mut windows_state {
        // let phase = crate::ui::tiles::WindowRelayoutPhase::Idle;
        // let current_key = (state.descriptor.screen, false, phase);
        // let last = log_state.0.get(&marker).copied();
        // if last != Some(current_key) {
        //     info!(
        //         id = ?marker,
        //         screen = state.descriptor.screen.map(|s| s as i32).unwrap_or(-1),
        //         relayout = ?phase,
        //         "window_state"
        //     );
        //     log_state.0.insert(*marker, current_key);
        // }
        state.graph_entities = state.tile_state.collect_graph_entities();

        // if let Some(entity) = state.window_entity
        //     && existing_map.get(&state.id).copied() != Some(entity)
        // {
        //     state.window_entity = None;
        // }

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
                screen: screen.clone(),
            });
        }

        if let Some(rect) = state.descriptor.screen_rect.as_ref() {
            commands.send_event(WindowRelayout::Rect {
                window: window_entity,
                rect: rect.clone(),
            });
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

/// Wait for a window to change to a target screen or timeout.
///
/// ```ignore
/// set_screen_for_window(id, 2);
/// let success = wait_for_window_to_change_screens(id, 2, Duration::millis(2000)).await?;
/// ```
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
            if let Some(screen) = detect_window_screen(window, &screens) {
                if screen == target_screen {
                    return Some(true);
                }
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
            info!(%entity, "Confirmed screen assignment");
            return None;
        }

        // let Some(screen) = state.descriptor.screen else {
        //     complete_screen_assignment(
        //         &mut state,
        //         "No screen provided; skipping screen alignment",
        //     );
        //     return None;
        // };

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
            wait_for_window_to_change_screens(entity, 2, Duration::from_millis(1000)).await?;
        let _ = winit_windows_async.get(|winit_windows| {
            let Some(window) = winit_windows.get_window(entity) else {
                error!(%entity, "No winit window in apply screen");
                return;
            };
            if success {
                if LINUX_MULTI_WINDOW {
                    exit_fullscreen(window);
                    force_windowed(window);
                } else if window.fullscreen().is_some() {
                    exit_fullscreen(window);
                }
            } else {
                recenter_window_on_screen(window, &target_monitor);
                #[cfg(target_os = "macos")]
                {
                    window
                        .set_fullscreen(Some(Fullscreen::Borderless(Some(target_monitor.clone()))));
                    window.set_fullscreen(None);
                    recenter_window_on_screen(window, &target_monitor);
                }
            }
        })?;
    }
    Ok(())
}
fn handle_window_relayout_events(
    mut relayout_events: EventReader<WindowRelayout>,
    mut commands: Commands,
) {
    for relayout_event in relayout_events.read().cloned() {
        match relayout_event {
            WindowRelayout::Screen { window, screen } => {
                info!(
                    // path = %state.descriptor.path.display(),
                    target_screen = screen,
                    // relayout_phase = ?state.relayout_phase,
                    "Attempting secondary screen assignment"
                );

                // Let's spawn an asynchronous task instead of carrying a bunch
                // state in our components.
                commands.spawn_task(move || apply_window_screen(window, screen));
                // state.relayout_phase = tiles::WindowRelayoutPhase::Idle;
            }
            WindowRelayout::Rect { window, rect } => {
                commands.spawn_task(move || {
                    apply_window_rect(rect, window, Duration::from_millis(1000))
                });
                // let Some(window) = winit_windows.get_window(entity) else {
                //     continue;
                // };

                // if ! apply_window_rect(rect, window) {
                // }
            }
            WindowRelayout::UpdateDescriptors => {
                // Instead of placing this system into the schedule and blocking
                // it from running, let's run it only when necessary.
                commands.run_system_cached(capture_window_screens_oneoff);
            }
        }
    }
}

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

            // XXX We may need this.
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
            let screen_pos = screen_handle.position();
            let screen_size = screen_handle.size();
            if screen_size.width == 0 || screen_size.height == 0 {
                return true;
            }

            let is_full_rect = LINUX_MULTI_WINDOW
                && rect.x == 0
                && rect.y == 0
                && rect.width == 100
                && rect.height == 100;

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
                if !is_full_rect {
                    force_windowed(window);
                }
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
                size_w = width_px,
                size_h = height_px,
                pos_x = x,
                pos_y = y,
                "Applied rect"
            );
            false
        })?;
    }
    Ok(())
}

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

    // info!(
    //     path = %state.descriptor.path.display(),
    //     screen = state.descriptor.screen.map(|idx| idx as i32).unwrap_or(-1),
    //     "assign_window_to_screen"
    // );
    // Align with Linux path: avoid fullscreen hops on macOS and keep windowed positioning.
    exit_fullscreen(window);
    window.set_visible(true);
    force_windowed(window);
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


/// Runs as a one-off system to capture the window descriptor. Does not run
/// every frameñ.
fn capture_window_screens_oneoff(
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut window_query: Query<(Entity, &Window, &mut tiles::WindowState)>,
    screens: Query<(Entity, &Monitor)>,
) {
    for (entity, window, mut state) in &mut window_query {
        let winit_window = winit_windows.get_window(entity);
        let mut updated = false;
        if let Some(window) = winit_window {
            let screens_sorted = collect_sorted_screens(window);
            updated = state.update_descriptor_from_winit_window(window, &screens_sorted);
        }
        if !updated {
            state.update_descriptor_from_window(window, &screens);
        }
    }
}

fn track_window_geometry(
    mut moved_events: EventReader<WindowMoved>,
    mut resized_events: EventReader<WindowResized>,
    winit_windows: NonSend<bevy::winit::WinitWindows>,
    mut windows_state: Query<(&Window, &tiles::WindowId, &mut tiles::WindowState)>,
    mut touched: Local<HashMap<Entity, Option<PhysicalPosition<i32>>>>,
) {
    for evt in moved_events.read() {
        let position = PhysicalPosition::new(evt.position.x, evt.position.y);
        touched.insert(evt.window, Some(position));
    }
    for evt in resized_events.read() {
        touched.entry(evt.window).or_insert(None);
    }

    for (entity, forced_position) in touched.drain() {
        let Ok((_window, id, mut state)) = windows_state.get_mut(entity) else {
            continue;
        };
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };
        let screens_sorted = collect_sorted_screens(window);
        record_window_rect_from_window(&mut state, window, &screens_sorted, forced_position);
    }
}

fn collect_sorted_screens(window: &WinitWindow) -> Vec<MonitorHandle> {
    let mut screens: Vec<MonitorHandle> = window.available_monitors().collect();
    screens.sort_by(|a, b| {
        let p_a = a.position();
        let p_b = b.position();
        p_a.x.cmp(&p_b.x)
             .then(p_a.y.cmp(&p_b.y))
             .then_with(|| {
                 
                 let name_a = a.name();
                 let name_b = b.name();
                 name_a.cmp(&name_b)
             })
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

// XXX: What does this do that's different from `collect_sorted_screens`?
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
    state: &mut tiles::WindowState,
    window: &WinitWindow,
    screens: &[MonitorHandle],
    forced_position: Option<PhysicalPosition<i32>>,
) {
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
    for (window_entity, window, id, state, egui_settings) in &window_query {
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
    states: Query<(Entity, &tiles::WindowState, &tiles::WindowId)>,
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

    for (window_entity, state, id) in &states {
        let base = SECONDARY_GRAPH_ORDER_BASE
            + SECONDARY_GRAPH_ORDER_STRIDE * id.0 as isize
            + NAV_GIZMO_ORDER_OFFSET;
        base_by_window.insert(window_entity, base);
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
fn sync_hdr(hdr_enabled: Res<HdrEnabled>, mut query: Query<&mut Camera>) {
    for mut cam in query.iter_mut() {
        cam.hdr = hdr_enabled.0;
    }
}
