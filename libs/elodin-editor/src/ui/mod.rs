use std::collections::HashMap;

use bevy::{
    app::AppExit,
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    input::keyboard::Key,
    prelude::*,
    camera::{RenderTarget,Viewport},
    window::{
        EnabledButtons, PresentMode, PrimaryWindow, WindowCloseRequested, WindowRef,
        WindowResolution,
    },
};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Color32, Label, Margin, RichText},
};
use egui_tiles::{Container, Tile};

use big_space::GridCell;
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::get_scheme;
use self::{command_palette::CommandPaletteState, timeline::timeline_slider};
use egui::CornerRadius;
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::ComponentMetadata;
use impeller2_wkt::ComponentValue;

use crate::{GridHandle, MainCamera, plugins::LogicalKeyState};

use self::inspector::entity::ComponentFilter;

use self::command_palette::CommandPalette;
use self::widgets::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystemExt};
use self::{button::EImageButton, utils::MarginSides};

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
            .init_resource::<FullscreenState>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<timeline_slider::UITick>()
            .init_resource::<timeline::StreamTickOrigin>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_event::<DialogEvent>()
            .add_systems(Update, timeline_slider::sync_ui_tick.before(render_layout))
            .add_systems(Update, actions::spawn_lua_actor)
            .add_systems(Update, shortcuts)
            .add_systems(Update, handle_primary_close.before(render_layout))
            .add_systems(Update, render_layout)
            .add_systems(Update, sync_secondary_windows.after(render_layout))
            .add_systems(Update, handle_secondary_close.after(sync_secondary_windows))
            .add_systems(
                Update,
                set_secondary_camera_viewport.after(sync_secondary_windows),
            )
            .add_systems(
                Update,
                render_secondary_windows.after(handle_secondary_close),
            )
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

#[derive(Resource, Default)]
pub struct FullscreenState(pub bool);

pub struct TitlebarIcons {
    pub icon_close: egui::TextureId,
    pub icon_fullscreen: egui::TextureId,
    pub icon_exit_fullscreen: egui::TextureId,
}

#[derive(SystemParam)]
pub struct Titlebar<'w, 's> {
    fullscreen_state: ResMut<'w, FullscreenState>,
    app_exit: MessageWriter<'w, AppExit>,
    windows: Query<
        'w,
        's,
        (
            Entity,
            &'static Window,
            &'static bevy::window::PrimaryWindow,
        ),
    >,
    winit_windows: NonSend<'w, bevy::winit::WinitWindows>,
}

impl RootWidgetSystem for Titlebar<'_, '_> {
    type Args = TitlebarIcons;
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        args: Self::Args,
    ) {
        let mut state_mut = state.get_mut(world);

        let mut fullscreen_state = state_mut.fullscreen_state;

        let TitlebarIcons {
            icon_fullscreen,
            icon_exit_fullscreen,
            icon_close,
        } = args;

        let titlebar_height = if cfg!(target_os = "macos")
            || cfg!(target_os = "windows")
            || cfg!(target_os = "linux")
        {
            45.0
        } else {
            34.0
        };
        let traffic_light_offset = if cfg!(target_os = "macos") { 72. } else { 0. };
        let titlebar_scale = if cfg!(target_os = "macos") { 1.4 } else { 1.3 };
        let titlebar_margin = if cfg!(target_os = "macos") {
            8
        } else if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
            0
        } else {
            4
        };
        let titlebar_right_margin = if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
            0.
        } else {
            10.
        };

        theme::set_theme(ctx);
        egui::TopBottomPanel::top("title_bar")
            .frame(
                egui::Frame {
                    fill: Color32::TRANSPARENT,
                    stroke: egui::Stroke::new(0.0, get_scheme().border_primary),
                    ..Default::default()
                }
                .inner_margin(
                    Margin::same(titlebar_margin)
                        .left(16.0)
                        .right(titlebar_right_margin),
                ),
            )
            .exact_height(titlebar_height)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(traffic_light_offset);
                    if cfg!(target_family = "wasm") {
                        if ui
                            .add(
                                EImageButton::new(
                                    if fullscreen_state.bypass_change_detection().0 {
                                        icon_exit_fullscreen
                                    } else {
                                        icon_fullscreen
                                    },
                                )
                                .scale(titlebar_scale, titlebar_scale)
                                .bg_color(Color32::TRANSPARENT),
                            )
                            .clicked()
                        {
                            fullscreen_state.0 = !fullscreen_state.0;
                        }
                        ui.add_space(8.0);
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
                            let Ok((window_id, _, _)) = state_mut.windows.single() else {
                                return;
                            };
                            let winit_window =
                                state_mut.winit_windows.get_window(window_id).unwrap();
                            ui.horizontal_centered(|ui| {
                                ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                                    egui::Color32::from_hex("#E81123").expect("invalid red color");

                                ui.style_mut().visuals.widgets.hovered.fg_stroke =
                                    egui::Stroke::new(1.0, Color32::WHITE);
                                ui.style_mut().visuals.widgets.hovered.corner_radius =
                                    CornerRadius {
                                        nw: 0,
                                        ne: if cfg!(target_os = "windows") { 4 } else { 0 },
                                        sw: 0,
                                        se: 0,
                                    };
                                let btn = if cfg!(target_os = "windows") {
                                    egui::Button::new(
                                        RichText::new("\u{e8bb}")
                                            .font(egui::FontId {
                                                size: 10.0,
                                                family: egui::FontFamily::Proportional,
                                            })
                                            .line_height(Some(11.0)),
                                    )
                                } else {
                                    egui::Button::image(egui::load::SizedTexture::new(
                                        icon_close,
                                        egui::vec2(11., 11.),
                                    ))
                                    .image_tint_follows_text_color(true)
                                };
                                if ui
                                    .add_sized(
                                        egui::vec2(45.0, 40.0),
                                        btn.stroke(egui::Stroke::NONE),
                                    )
                                    .clicked()
                                {
                                    state_mut.app_exit.write(AppExit::Success);
                                }
                                ui.add_space(2.0);
                            });

                            let maximized = winit_window.is_maximized();
                            ui.scope(|ui| {
                                ui.style_mut().visuals.widgets.hovered.fg_stroke =
                                    egui::Stroke::new(1.0, Color32::WHITE);
                                ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                                    egui::Color32::from_hex("#4E4D53").expect("invalid red color");
                                ui.style_mut().visuals.widgets.hovered.corner_radius =
                                    CornerRadius::ZERO;
                                let btn = if cfg!(target_os = "windows") {
                                    egui::Button::new(
                                        RichText::new(if maximized {
                                            "\u{e923}"
                                        } else {
                                            "\u{e922}"
                                        })
                                        .font(egui::FontId {
                                            size: 10.0,
                                            family: egui::FontFamily::Proportional,
                                        })
                                        .line_height(Some(11.0)),
                                    )
                                } else {
                                    egui::Button::image(egui::load::SizedTexture::new(
                                        if maximized {
                                            icon_exit_fullscreen
                                        } else {
                                            icon_fullscreen
                                        },
                                        egui::vec2(11., 11.),
                                    ))
                                    .image_tint_follows_text_color(true)
                                };
                                if ui
                                    .add_sized(
                                        egui::vec2(45.0, 40.0),
                                        btn.stroke(egui::Stroke::NONE),
                                    )
                                    .clicked()
                                {
                                    winit_window.set_maximized(!maximized);
                                }

                                ui.add_space(2.0);
                                if ui
                                    .add_sized(
                                        egui::vec2(45.0, 40.0),
                                        egui::Button::new(
                                            RichText::new(if cfg!(target_os = "windows") {
                                                "\u{e921}"
                                            } else {
                                                "—"
                                            })
                                            .font(egui::FontId {
                                                size: 10.0,
                                                family: egui::FontFamily::Proportional,
                                            })
                                            .line_height(Some(11.0)),
                                        )
                                        .stroke(egui::Stroke::NONE),
                                    )
                                    .clicked()
                                {
                                    winit_window.set_minimized(true);
                                }
                            });
                            ui.add_space(8.0);
                        }
                    });
                });
            });
    }
}

#[derive(SystemParam)]
pub struct MainLayout<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
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
        let state_mut = state.get_mut(world);

        let mut contexts = state_mut.contexts;
        let images = state_mut.images;

        theme::set_theme(ctx);

        let titlebar_icons = TitlebarIcons {
            icon_fullscreen: contexts.add_image(images.icon_fullscreen.clone_weak()),
            icon_exit_fullscreen: contexts.add_image(images.icon_exit_fullscreen.clone_weak()),
            icon_close: contexts.add_image(images.icon_close.clone_weak()),
        };

        world.add_root_widget_with::<Titlebar, With<PrimaryWindow>>("titlebar", titlebar_icons);

        #[cfg(not(target_family = "wasm"))]
        world.add_root_widget::<status_bar::StatusBar>("status_bar");

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
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
) {
    let mut existing_map: HashMap<tiles::SecondaryWindowId, Entity> = HashMap::new();
    for (entity, marker) in existing.iter() {
        existing_map.insert(marker.id, entity);
    }

    for (id, entity) in existing_map.clone() {
        if windows.get_secondary(id).is_none() {
            commands.entity(entity).despawn();
            existing_map.remove(&id);
        }
    }

    for state in windows.secondary_mut().iter_mut() {
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
                    camera.order = 10 + index as isize;
                }
            }
            continue;
        } else {
            for &graph in &state.graph_entities {
                if let Ok(mut camera) = cameras.get_mut(graph) {
                    camera.is_active = false;
                }
            }
        }

        let title = compute_secondary_window_title(state);

        let window_entity = commands
            .spawn((
                Window {
                    title,
                    resolution: WindowResolution::new(640.0, 480.0),
                    present_mode: PresentMode::AutoVsync,
                    enabled_buttons: EnabledButtons {
                        close: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                SecondaryWindowMarker { id: state.id },
            ))
            .id();

        state.window_entity = Some(window_entity);
        existing_map.insert(state.id, window_entity);
        let window_ref = WindowRef::Entity(window_entity);
        for (index, &graph) in state.graph_entities.iter().enumerate() {
            if let Ok(mut camera) = cameras.get_mut(graph) {
                camera.target = RenderTarget::Window(window_ref);
                camera.is_active = true;
                camera.order = 10 + index as isize;
            }
        }
    }
}

fn handle_secondary_close(
    mut events: MessageReader<WindowCloseRequested>,
    mut windows: ResMut<tiles::WindowManager>,
) {
    let mut to_remove = Vec::new();
    for evt in events.read() {
        let entity = evt.window;
        if let Some(id) = windows.find_secondary_by_entity(entity) {
            to_remove.push(id);
        }
    }

    if !to_remove.is_empty() {
        windows
            .secondary_mut()
            .retain(|state| !to_remove.contains(&state.id));
    }
}

fn handle_primary_close(
    mut events: MessageReader<WindowCloseRequested>,
    primary: Query<Entity, With<PrimaryWindow>>,
    mut exit: MessageWriter<AppExit>,
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

#[derive(QueryData)]
#[query_data(mutable)]
struct CameraViewportQuery {
    camera: &'static mut Camera,
    viewport_rect: &'static ViewportRect,
}

fn set_camera_viewport(
    window: Query<(&Window, &bevy_egui::EguiContextSettings), With<PrimaryWindow>>,
    mut main_camera_query: Query<CameraViewportQuery, With<MainCamera>>,
) {
    for CameraViewportQueryItem {
        mut camera,
        viewport_rect,
    } in main_camera_query.iter_mut()
    {
        let Some(available_rect) = viewport_rect.0 else {
            camera.order = 0;
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
        if available_rect.size().x > window.width() || available_rect.size().y > window.height() {
            return;
        }
        if viewport_size.x < 10.0 || viewport_size.y < 10.0 {
            camera.is_active = false;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(0, 0),
                physical_size: UVec2::new(1, 1),
                depth: 0.0..1.0,
            });

            continue;
        }
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
            physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
            depth: 0.0..1.0,
        });
    }
}

fn set_secondary_camera_viewport(
    windows: Res<tiles::WindowManager>,
    mut cameras: Query<(&mut Camera, &ViewportRect)>,
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

        for (index, &graph) in state.graph_entities.iter().enumerate() {
            let Ok((mut camera, viewport_rect)) = cameras.get_mut(graph) else {
                continue;
            };

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

            if viewport_size.x < 1.0 || viewport_size.y < 1.0 {
                camera.is_active = false;
                camera.viewport = Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(1, 1),
                    depth: 0.0..1.0,
                });
                continue;
            }

            camera.is_active = true;
            camera.order = 10 + index as isize;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
                physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
                depth: 0.0..1.0,
            });
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
