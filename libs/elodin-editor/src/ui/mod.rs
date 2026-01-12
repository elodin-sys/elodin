use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::PathBuf;

use bevy::{
    camera::{RenderTarget, Viewport},
    ecs::{
        query::QueryData,
        system::{NonSendMarker, SystemParam, SystemState},
    },
    // platform::collections::{HashMap, HashSet},
    input::keyboard::Key,
    log::{error, info},
    prelude::*,
    render::view::Hdr,
    window::{Monitor, NormalizedWindowRef, PrimaryWindow, WindowFocused},
    winit::WINIT_WINDOWS,
};
use bevy_defer::AsyncPlugin;
use bevy_egui::{
    EguiContext, EguiContexts, EguiPreUpdateSet,
    egui::{self, Color32, Label, RichText},
};
pub(crate) const DEFAULT_SECONDARY_RECT: WindowRect = WindowRect {
    x: 10,
    y: 10,
    width: 80,
    height: 80,
};
// Order ranges:
// 10..        primary viewports (3D, gizmo/axes…)
// 100..       primary graphs
// 100,000..   -> Gizmo arrow labels
// 200,000..   -> UI/egui (on top of everything)
const PRIMARY_VIEWPORT_ORDER_BASE: isize = 10;
const PRIMARY_GRAPH_ORDER_BASE: isize = 100;
pub const UI_ORDER_BASE: isize = 200_000;
const NAV_GIZMO_ORDER_OFFSET: isize = 1;

pub type PaneName = String;

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

use platform::PRIMARY_ORDER_OFFSET;

use big_space::GridCell;
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::get_scheme;
use self::{command_palette::CommandPaletteState, plot::GraphState, timeline::timeline_slider};
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::{ComponentMetadata, ComponentValue, DbConfig, WindowRect};

use crate::ui::window::window_entity_from_target;
use crate::{
    GridHandle, MainCamera,
    plugins::{
        LogicalKeyState,
        navigation_gizmo::{NavGizmoCamera, NavGizmoParent},
    },
    tiles::WindowId,
};

use self::inspector::entity::ComponentFilter;

use self::command_palette::CommandPalette;
use self::widgets::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystemExt};

pub mod actions;
pub mod button;
pub mod colors;
pub mod command_palette;
pub mod dashboard;
pub mod data_overview;
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
pub mod window;

// Re-export window helpers for existing call sites.
pub use window::{
    base_window, collect_sorted_screens, default_composite_alpha_mode, default_present_mode,
    default_window_theme, detect_window_screen, handle_window_close, handle_window_destroyed,
    handle_window_relayout_events, sync_windows, wait_for_winit_window, window_graph_order_base,
    window_theme_for_mode,
};

#[cfg(not(target_family = "wasm"))]
pub mod status_bar;

#[cfg(not(target_family = "wasm"))]
pub mod startup_window;

#[derive(Resource, Default)]
pub struct HdrEnabled(pub bool);

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct FocusedWindow(pub Option<Entity>);

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
    QueryTable {
        table_id: Entity,
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
            SelectedObject::QueryTable { table_id } => Some(*table_id),
            SelectedObject::Action { action_id } => Some(*action_id),
            SelectedObject::Object3D { entity } => Some(*entity),
            SelectedObject::DashboardNode { entity } => Some(*entity),
        }
    }
}

#[derive(Resource, Default)]
pub struct HoveredEntity(pub Option<EntityPair>);

#[derive(Resource, Default, Clone, Debug)]
pub struct EntityFilter(pub String);

#[derive(Resource, Default, Clone, Debug)]
pub struct InspectorAnchor(pub Option<egui::Pos2>);

#[derive(Default, Clone, Debug)]
pub struct WindowUiState {
    pub selected_object: SelectedObject,
    pub entity_filter: EntityFilter,
    pub inspector_anchor: InspectorAnchor,
    pub left_sidebar_visible: bool,
    pub right_sidebar_visible: bool,
}

#[derive(Component, Clone)]
pub struct ViewportRect(pub Option<egui::Rect>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityPair {
    pub bevy: Entity,
    pub impeller: ComponentId,
}

pub fn create_egui_context() -> EguiContext {
    let mut bevy_egui_ctx = EguiContext::default();
    let egui_ctx = bevy_egui_ctx.get_mut();

    theme::set_theme(egui_ctx);

    bevy_egui_ctx
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

pub fn update_focused_window(
    mut focused_window: ResMut<FocusedWindow>,
    mut focus_events: MessageReader<WindowFocused>,
) {
    for event in focus_events.read() {
        if event.focused {
            focused_window.0 = Some(event.window);
        } else if focused_window.0 == Some(event.window) {
            focused_window.0 = None;
        }
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
            .init_resource::<HoveredEntity>()
            .init_resource::<ComponentFilter>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<FocusedWindow>()
            .init_resource::<timeline_slider::UITick>()
            .init_resource::<timeline::StreamTickOrigin>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_message::<DialogEvent>()
            .add_systems(Update, timeline_slider::sync_ui_tick.before(render_layout))
            .add_systems(Update, actions::spawn_lua_actor)
            .add_systems(Update, update_focused_window)
            .add_systems(Update, shortcuts)
            .add_systems(PreUpdate, sync_windows.before(EguiPreUpdateSet::BeginPass))
            .add_systems(
                Update,
                (
                    handle_window_close,
                    handle_window_destroyed,
                    render_layout,
                    sync_camera_grid_cell,
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
            .add_plugins(timeline::plugin)
            .add_plugins(tiles::plugin)
            .add_plugins(SchematicPlugin)
            .add_plugins(LinePlot3dPlugin)
            .add_plugins(AsyncPlugin::default_settings())
            .add_plugins(command_palette::palette_items::plugin);
    }
}

#[derive(Clone, Debug)]
pub enum SettingModal {
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

#[derive(Clone, Debug, Message)]
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

        // Update theme every frame to reflect color scheme changes
        theme::set_theme(ctx);

        #[cfg(not(target_family = "wasm"))]
        world.add_root_widget::<status_bar::StatusBar>("status_bar");

        #[cfg(target_os = "macos")]
        let frame = {
            // Leave room for the native titlebar controls on the primary window.
            let mut f = egui::Frame::new();
            f.inner_margin.top = 32;
            f
        };
        #[cfg(not(target_os = "macos"))]
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
        }
        widget_id.clear();
        if window_id.is_primary() {
            widget_id.push_str("command_palette_window");
        } else {
            let _ = write!(widget_id, "secondary_command_palette_{}", window_id.0);
        }
        world.add_root_widget_to::<command_palette::PaletteWindow>(id, &widget_id, Some(id));
    }
}

/// Runs as a one-off system to capture the window descriptor. Does not run
/// every frameñ.
pub(crate) fn capture_window_screens_oneoff(
    mut window_query: Query<(Entity, &Window, &mut tiles::WindowState)>,
    screens: Query<(Entity, &Monitor)>,
    _non_send_marker: NonSendMarker,
) {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
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
    });
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
    window: Query<(Entity, &Window, &bevy_egui::EguiContextSettings), With<PrimaryWindow>>,
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

    let Some((primary_entity, window, egui_settings)) = window.iter().next() else {
        return;
    };
    let scale_factor = window.scale_factor() * egui_settings.scale_factor;
    let window_size: Vec2 = window.physical_size().as_vec2();

    for (entity, is_graph) in &entries {
        let Ok((_, viewport_rect, _graph_state, mut camera)) = main_camera_query.get_mut(*entity)
        else {
            continue;
        };
        let Some(camera_window) = window_entity_from_target(&camera.target, primary_entity) else {
            continue;
        };
        if camera_window != primary_entity {
            continue;
        }
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
        let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = available_rect.size() * scale_factor;
        let viewport_pos = Vec2::new(viewport_pos.x, viewport_pos.y);
        let viewport_size = Vec2::new(viewport_size.x, viewport_size.y);
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
                // Offset cameras to avoid colliding with primary orders.
                let base_order = window_graph_order_base(*id);
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
                warn_once!(
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

fn sync_hdr(
    hdr_enabled: Res<HdrEnabled>,
    mut commands: Commands,
    cameras: Query<(Entity, Has<Hdr>), With<Camera>>,
) {
    for (entity, has_hdr) in cameras.iter() {
        if hdr_enabled.0 && !has_hdr {
            commands.entity(entity).insert(Hdr);
        } else if !hdr_enabled.0 && has_hdr {
            commands.entity(entity).remove::<Hdr>();
        }
    }
}
