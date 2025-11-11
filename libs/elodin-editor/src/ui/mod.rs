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
        EnabledButtons, Monitor, PresentMode, PrimaryWindow, WindowCloseRequested, WindowMoved,
        WindowRef, WindowResized, WindowResolution,
    },
};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Align2, Color32, Label, Margin, RichText},
};
use egui_tiles::{Container, Tile};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    monitor::MonitorHandle,
    window::{Fullscreen, Window as WinitWindow},
};

const SCREEN_RELAYOUT_MAX_ATTEMPTS: u8 = 5;
const SCREEN_RELAYOUT_TIMEOUT: Duration = Duration::from_millis(750);
const FULLSCREEN_EXIT_CONFIRMATION_TIMEOUT: Duration = Duration::from_millis(500);
const LINUX_MULTI_WINDOW: bool = cfg!(target_os = "linux");
const PRIMARY_VIEWPORT_ORDER_BASE: isize = 0;
const PRIMARY_GRAPH_ORDER_BASE: isize = 100;
const SECONDARY_GRAPH_ORDER_BASE: isize = 10;
const SECONDARY_GRAPH_ORDER_STRIDE: isize = 50;

use big_space::GridCell;
use plot_3d::LinePlot3dPlugin;
use schematic::SchematicPlugin;

use self::colors::{ColorExt, get_scheme};
use self::{command_palette::CommandPaletteState, plot::GraphState, timeline::timeline_slider};
use egui::CornerRadius;
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValueMap;
use impeller2_wkt::{
    ComponentMetadata, ComponentValue, ComponentValue as WktComponentValue, VectorArrow3d,
};

use crate::{
    GridHandle, MainCamera,
    plugins::{
        LogicalKeyState,
        gizmos::{MIN_ARROW_LENGTH_SQUARED, evaluate_vector_arrow},
    },
    vector_arrow::VectorArrowState,
};

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
            .add_systems(
                Update,
                apply_primary_window_layout
                    .after(render_layout)
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
                    .before(capture_secondary_window_screens),
            )
            .add_systems(
                Update,
                capture_secondary_window_screens.after(apply_secondary_window_screens),
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
    app_exit: EventWriter<'w, AppExit>,
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
                    let base_order = secondary_graph_order_base(state.id);
                    camera.order = base_order + index as isize;
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

        let window_component = Window {
            title,
            resolution: WindowResolution::new(640.0, 480.0),
            present_mode: PresentMode::AutoVsync,
            enabled_buttons: EnabledButtons {
                close: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let window_entity = commands
            .spawn((window_component, SecondaryWindowMarker { id: state.id }))
            .id();

        state.window_entity = Some(window_entity);
        state.applied_screen = None;
        state.applied_rect = None;
        state.refresh_relayout_phase();
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
        let Some(entity) = state.window_entity else {
            continue;
        };
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };

        let monitors = collect_sorted_monitors(window);

        match state.relayout_phase {
            tiles::SecondaryWindowRelayoutPhase::NeedScreen => {
                if state.pending_exit_state == tiles::PendingFullscreenExit::Requested {
                    if LINUX_MULTI_WINDOW {
                        if window.fullscreen().is_some() {
                            exit_fullscreen(window);
                            if !fullscreen_exit_timed_out(state.pending_exit_started_at) {
                                info!(
                                    path = %state.descriptor.path.display(),
                                    "Waiting for Linux fullscreen exit before reassigning screen"
                                );
                                continue;
                            }
                            info!(
                                path = %state.descriptor.path.display(),
                                "Timed out while waiting for fullscreen exit; forcing screen assignment"
                            );
                        }
                        linux_force_windowed(window);
                        state.pending_fullscreen_exit = false;
                        state.pending_exit_started_at = None;
                    } else {
                        if window.fullscreen().is_some() {
                            continue;
                        }
                        window.set_maximized(false);
                        window.set_minimized(false);
                    }
                    state.pending_exit_state = tiles::PendingFullscreenExit::None;
                }

                if window_on_target_screen(state, window, &monitors) {
                    if LINUX_MULTI_WINDOW {
                        linux_force_windowed(window);
                    } else if window.fullscreen().is_some() {
                        exit_fullscreen(window);
                        state.pending_fullscreen_exit = false;
                        state.pending_exit_state = tiles::PendingFullscreenExit::None;
                        state.pending_exit_started_at = None;
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

                if let Some(target_monitor) = monitors.get(screen) {
                    assign_window_to_screen(state, window, target_monitor.clone());
                    state.pending_exit_state = tiles::PendingFullscreenExit::Requested;
                    state.relayout_attempts = state.relayout_attempts.saturating_add(1);
                    if state.relayout_started_at.is_none() {
                        state.relayout_started_at = Some(Instant::now());
                    }
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
                if apply_secondary_window_rect(state, window, &monitors) {
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
    let monitors = collect_sorted_monitors(window);
    let layout = windows.primary_layout_mut();
    let phase = layout.relayout_phase;
    match phase {
        tiles::PrimaryWindowRelayoutPhase::Idle => {}
        tiles::PrimaryWindowRelayoutPhase::NeedScreen => {
            if window_on_screen(layout.screen, window, &monitors) {
                if layout.pending_fullscreen_exit {
                    if LINUX_MULTI_WINDOW {
                        if window.fullscreen().is_some() {
                            exit_fullscreen(window);
                            if fullscreen_exit_timed_out(layout.pending_fullscreen_exit_started_at)
                            {
                                info!(
                                    screen = layout.screen.map(|idx| idx as i32).unwrap_or(-1),
                                    "Timed out while waiting for primary window fullscreen exit; forcing apply"
                                );
                            } else {
                                return;
                            }
                        }
                        linux_force_windowed(window);
                        layout.pending_fullscreen_exit_started_at = None;
                    } else {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        }
                        window.set_maximized(false);
                        window.set_minimized(false);
                    }
                    layout.pending_fullscreen_exit = false;
                } else if LINUX_MULTI_WINDOW {
                    if window.fullscreen().is_some() {
                        exit_fullscreen(window);
                        return;
                    }
                    linux_force_windowed(window);
                } else {
                    if window.fullscreen().is_some() {
                        window.set_fullscreen(None);
                    }
                    window.set_maximized(false);
                    window.set_minimized(false);
                }
                layout.applied_screen = layout.screen;
                layout.relayout_phase = if layout.screen_rect.is_some() {
                    tiles::PrimaryWindowRelayoutPhase::NeedRect
                } else {
                    tiles::PrimaryWindowRelayoutPhase::Idle
                };
                layout.relayout_attempts = 0;
                layout.relayout_started_at = None;
                layout.pending_fullscreen_exit = false;
                if LINUX_MULTI_WINDOW {
                    layout.pending_fullscreen_exit_started_at = None;
                }
                info!(
                    screen = layout.screen.map(|idx| idx as i32).unwrap_or(-1),
                    "Primary window confirmed on target screen"
                );
                return;
            }
            let Some(screen) = layout.screen else {
                layout.relayout_phase = if layout.screen_rect.is_some() {
                    tiles::PrimaryWindowRelayoutPhase::NeedRect
                } else {
                    tiles::PrimaryWindowRelayoutPhase::Idle
                };
                return;
            };
            if let Some(target_monitor) = monitors.get(screen).cloned() {
                info!(
                    screen = screen as i32,
                    "Moving primary window to target screen"
                );
                assign_primary_window_to_screen(layout, window, target_monitor);
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
                    layout.relayout_phase = if layout.screen_rect.is_some() {
                        tiles::PrimaryWindowRelayoutPhase::NeedRect
                    } else {
                        tiles::PrimaryWindowRelayoutPhase::Idle
                    };
                    layout.pending_fullscreen_exit = false;
                    if LINUX_MULTI_WINDOW {
                        layout.pending_fullscreen_exit_started_at = None;
                    }
                }
            } else {
                warn!(
                    screen = screen as i32,
                    "Primary window screen index out of range; skipping"
                );
                layout.screen = None;
                layout.relayout_phase = if layout.screen_rect.is_some() {
                    tiles::PrimaryWindowRelayoutPhase::NeedRect
                } else {
                    tiles::PrimaryWindowRelayoutPhase::Idle
                };
            }
        }
        tiles::PrimaryWindowRelayoutPhase::NeedRect => {
            if apply_primary_window_rect(layout, window, &monitors) {
                layout.relayout_phase = tiles::PrimaryWindowRelayoutPhase::Idle;
            }
        }
    }
}

fn apply_secondary_window_rect(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
    monitors: &[MonitorHandle],
) -> bool {
    if state.pending_fullscreen_exit {
        if LINUX_MULTI_WINDOW {
            if window.fullscreen().is_some() {
                exit_fullscreen(window);
                if fullscreen_exit_timed_out(state.pending_exit_started_at) {
                    info!(
                        path = %state.descriptor.path.display(),
                        "Timed out while exiting fullscreen; applying rect anyway"
                    );
                } else {
                    return false;
                }
            }
            linux_force_windowed(window);
            state.pending_fullscreen_exit = false;
            state.pending_exit_started_at = None;
        } else {
            if window.fullscreen().is_some() {
                window.set_fullscreen(None);
                window.set_maximized(false);
                window.set_minimized(false);
                return false;
            }
            state.pending_fullscreen_exit = false;
        }
    }

    let Some(rect) = state.descriptor.screen_rect else {
        if state.applied_rect.is_some() {
            state.applied_rect = None;
            window.set_minimized(false);
        }
        return true;
    };

    if state.applied_rect == Some(rect) {
        return true;
    }

    if rect.width == 0 && rect.height == 0 {
        window.set_minimized(true);
        state.applied_rect = Some(rect);
        state.skip_metadata_capture = true;
        return true;
    }

    let monitor_handle = state
        .descriptor
        .screen
        .and_then(|idx| monitors.get(idx).cloned())
        .or_else(|| window.current_monitor());
    let Some(monitor_handle) = monitor_handle else {
        return false;
    };

    let monitor_pos = monitor_handle.position();
    let monitor_size = monitor_handle.size();
    if monitor_size.width == 0 || monitor_size.height == 0 {
        return false;
    }

    if window.fullscreen().is_some() {
        if LINUX_MULTI_WINDOW {
            exit_fullscreen(window);
            linux_force_windowed(window);
        } else {
            window.set_fullscreen(None);
            window.set_maximized(false);
            window.set_minimized(false);
        }
    } else {
        if LINUX_MULTI_WINDOW {
            linux_force_windowed(window);
        }
        window.set_maximized(false);
        window.set_minimized(false);
    }

    let monitor_width = monitor_size.width as i32;
    let monitor_height = monitor_size.height as i32;

    let requested_width_px = ((rect.width as f64 / 100.0) * monitor_width as f64).round() as i32;
    let requested_height_px = ((rect.height as f64 / 100.0) * monitor_height as f64).round() as i32;
    let width_px = requested_width_px.clamp(1, monitor_width.max(1));
    let height_px = requested_height_px.clamp(1, monitor_height.max(1));
    if width_px != requested_width_px || height_px != requested_height_px {
        warn!(
            path = %state.descriptor.path.display(),
            rect = ?rect,
            "Window rect exceeds screen bounds; clamping size"
        );
    }

    let requested_x =
        monitor_pos.x + ((rect.x as f64 / 100.0) * monitor_width as f64).round() as i32;
    let requested_y =
        monitor_pos.y + ((rect.y as f64 / 100.0) * monitor_height as f64).round() as i32;

    let max_x = monitor_pos.x + monitor_width - width_px;
    let max_y = monitor_pos.y + monitor_height - height_px;
    let x = requested_x.clamp(monitor_pos.x, max_x);
    let y = requested_y.clamp(monitor_pos.y, max_y);
    if x != requested_x || y != requested_y {
        warn!(
            path = %state.descriptor.path.display(),
            rect = ?rect,
            "Window rect origin exceeds screen bounds; clamping position"
        );
    }

    let _ = window.request_inner_size(PhysicalSize::new(
        width_px.max(1) as u32,
        height_px.max(1) as u32,
    ));
    window.set_outer_position(PhysicalPosition::new(x, y));
    state.applied_rect = Some(rect);
    state.skip_metadata_capture = true;
    true
}

fn assign_window_to_screen(
    state: &mut tiles::SecondaryWindowState,
    window: &WinitWindow,
    target_monitor: MonitorHandle,
) {
    let monitor_pos = target_monitor.position();
    let monitor_size = target_monitor.size();
    let window_size = window.outer_size();
    let x = monitor_pos.x + (monitor_size.width as i32 - window_size.width as i32) / 2;
    let y = monitor_pos.y + (monitor_size.height as i32 - window_size.height as i32) / 2;

    window.set_fullscreen(Some(Fullscreen::Borderless(Some(target_monitor))));
    if LINUX_MULTI_WINDOW {
        window.set_visible(true);
    }
    state.pending_fullscreen_exit = true;
    if LINUX_MULTI_WINDOW {
        state.pending_exit_started_at = Some(Instant::now());
    } else {
        state.pending_exit_started_at = None;
    }
    state.pending_exit_state = tiles::PendingFullscreenExit::Requested;
    window.set_outer_position(PhysicalPosition::new(x, y));
    state.skip_metadata_capture = true;
}

fn complete_screen_assignment(
    state: &mut tiles::SecondaryWindowState,
    _window: &WinitWindow,
    reason: &'static str,
) {
    state.applied_screen = state.descriptor.screen;
    state.relayout_attempts = 0;
    state.relayout_started_at = None;
    state.relayout_phase = if state.descriptor.screen_rect.is_some() {
        tiles::SecondaryWindowRelayoutPhase::NeedRect
    } else {
        tiles::SecondaryWindowRelayoutPhase::Idle
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

fn assign_primary_window_to_screen(
    layout: &mut tiles::PrimaryWindowLayout,
    window: &WinitWindow,
    target_monitor: MonitorHandle,
) {
    let monitor_pos = target_monitor.position();
    let monitor_size = target_monitor.size();
    let window_size = window.outer_size();
    let x = monitor_pos.x + (monitor_size.width as i32 - window_size.width as i32) / 2;
    let y = monitor_pos.y + (monitor_size.height as i32 - window_size.height as i32) / 2;

    window.set_fullscreen(Some(Fullscreen::Borderless(Some(target_monitor))));
    if LINUX_MULTI_WINDOW {
        window.set_visible(true);
    }
    layout.pending_fullscreen_exit = true;
    if LINUX_MULTI_WINDOW {
        layout.pending_fullscreen_exit_started_at = Some(Instant::now());
    } else {
        layout.pending_fullscreen_exit_started_at = None;
    }
    window.set_outer_position(PhysicalPosition::new(x, y));
}

fn apply_primary_window_rect(
    layout: &mut tiles::PrimaryWindowLayout,
    window: &WinitWindow,
    monitors: &[MonitorHandle],
) -> bool {
    if layout.pending_fullscreen_exit {
        if LINUX_MULTI_WINDOW {
            if window.fullscreen().is_some() {
                exit_fullscreen(window);
                if fullscreen_exit_timed_out(layout.pending_fullscreen_exit_started_at) {
                    info!(
                        screen = layout.screen.map(|idx| idx as i32).unwrap_or(-1),
                        "Timed out while exiting fullscreen for primary window; applying rect anyway"
                    );
                } else {
                    return false;
                }
            }
            linux_force_windowed(window);
            layout.pending_fullscreen_exit = false;
            layout.pending_fullscreen_exit_started_at = None;
        } else {
            if window.fullscreen().is_some() {
                window.set_fullscreen(None);
                window.set_maximized(false);
                window.set_minimized(false);
                return false;
            }
            layout.pending_fullscreen_exit = false;
        }
    }

    let Some(rect) = layout.screen_rect else {
        layout.applied_rect = None;
        window.set_minimized(false);
        return true;
    };

    if layout.applied_rect == Some(rect) {
        return true;
    }

    if rect.width == 0 && rect.height == 0 {
        window.set_minimized(true);
        layout.applied_rect = Some(rect);
        return true;
    }

    let monitor_handle = layout
        .screen
        .and_then(|idx| monitors.get(idx).cloned())
        .or_else(|| window.current_monitor());
    let Some(monitor_handle) = monitor_handle else {
        return false;
    };

    let monitor_pos = monitor_handle.position();
    let monitor_size = monitor_handle.size();
    if monitor_size.width == 0 || monitor_size.height == 0 {
        return false;
    }

    if window.fullscreen().is_some() {
        if LINUX_MULTI_WINDOW {
            exit_fullscreen(window);
            linux_force_windowed(window);
        } else {
            window.set_fullscreen(None);
            window.set_maximized(false);
            window.set_minimized(false);
        }
    } else {
        if LINUX_MULTI_WINDOW {
            linux_force_windowed(window);
        }
        window.set_maximized(false);
        window.set_minimized(false);
    }

    let monitor_width = monitor_size.width as i32;
    let monitor_height = monitor_size.height as i32;

    let requested_width_px = ((rect.width as f64 / 100.0) * monitor_width as f64).round() as i32;
    let requested_height_px = ((rect.height as f64 / 100.0) * monitor_height as f64).round() as i32;
    let width_px = requested_width_px.clamp(1, monitor_width.max(1));
    let height_px = requested_height_px.clamp(1, monitor_height.max(1));
    if width_px != requested_width_px || height_px != requested_height_px {
        warn!("Primary window rect exceeds screen bounds; clamping size (rect={rect:?})");
    }

    let requested_x =
        monitor_pos.x + ((rect.x as f64 / 100.0) * monitor_width as f64).round() as i32;
    let requested_y =
        monitor_pos.y + ((rect.y as f64 / 100.0) * monitor_height as f64).round() as i32;

    let max_x = monitor_pos.x + monitor_width - width_px;
    let max_y = monitor_pos.y + monitor_height - height_px;
    let x = requested_x.clamp(monitor_pos.x, max_x);
    let y = requested_y.clamp(monitor_pos.y, max_y);
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
) {
    for state in windows.secondary_mut().iter_mut() {
        if !matches!(
            state.relayout_phase,
            tiles::SecondaryWindowRelayoutPhase::Idle
        ) {
            continue;
        }

        let Some(entity) = state.window_entity else {
            continue;
        };
        let Some(window) = winit_windows.get_window(entity) else {
            continue;
        };

        let monitors = collect_sorted_monitors(window);
        state.update_descriptor_from_winit_window(window, &monitors);
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
        let monitors = collect_sorted_monitors(window);
        if window_on_target_screen(state, window, &monitors) {
            complete_screen_assignment(
                state,
                window,
                "Confirmed screen assignment for secondary window",
            );
        }
    }
}

fn collect_sorted_monitors(window: &WinitWindow) -> Vec<MonitorHandle> {
    let mut monitors: Vec<MonitorHandle> = window.available_monitors().collect();
    monitors.sort_by(|a, b| {
        a.position()
            .x
            .cmp(&b.position().x)
            .then(a.position().y.cmp(&b.position().y))
    });
    monitors
}

fn monitors_match(a: &MonitorHandle, b: &MonitorHandle) -> bool {
    a.position() == b.position() && a.size() == b.size()
}

fn window_on_target_screen(
    state: &tiles::SecondaryWindowState,
    window: &WinitWindow,
    monitors: &[MonitorHandle],
) -> bool {
    window_on_screen(state.descriptor.screen, window, monitors)
}

fn window_on_screen(
    screen: Option<usize>,
    window: &WinitWindow,
    monitors: &[MonitorHandle],
) -> bool {
    let Some(screen) = screen else {
        return true;
    };
    let Some(target_monitor) = monitors.get(screen) else {
        return false;
    };

    window
        .current_monitor()
        .is_some_and(|current| monitors_match(&current, target_monitor))
}

fn exit_fullscreen(window: &WinitWindow) {
    window.set_fullscreen(None);
    window.set_maximized(false);
    window.set_minimized(false);
    window.set_decorations(true);
}

fn fullscreen_exit_timed_out(started_at: Option<Instant>) -> bool {
    started_at
        .map(|instant| instant.elapsed() > FULLSCREEN_EXIT_CONFIRMATION_TIMEOUT)
        .unwrap_or(false)
}

fn linux_force_windowed(window: &WinitWindow) {
    if !LINUX_MULTI_WINDOW {
        return;
    }
    window.set_visible(true);
    window.set_decorations(true);
    window.set_maximized(false);
    window.set_minimized(false);
}

fn secondary_graph_order_base(id: tiles::SecondaryWindowId) -> isize {
    SECONDARY_GRAPH_ORDER_BASE + SECONDARY_GRAPH_ORDER_STRIDE * id.0 as isize
}

fn capture_primary_window_layout(
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
    if !matches!(
        layout.relayout_phase,
        tiles::PrimaryWindowRelayoutPhase::Idle
    ) {
        return;
    }

    let monitors = collect_sorted_monitors(window);
    let outer_pos = match window.outer_position() {
        Ok(pos) => pos,
        Err(_) => return,
    };
    let outer_size = window.outer_size();
    let monitor_index = window
        .current_monitor()
        .as_ref()
        .and_then(|current| {
            monitors
                .iter()
                .position(|monitor| monitors_match(monitor, current))
        })
        .or_else(|| tiles::monitor_index_from_bounds(outer_pos, outer_size, &monitors));
    layout.captured_screen = monitor_index;

    if let Some(idx) = monitor_index
        && let Some(monitor) = monitors.get(idx)
    {
        let monitor_pos = monitor.position();
        let monitor_size = monitor.size();
        if let Some(rect) = tiles::rect_from_bounds(
            (outer_pos.x, outer_pos.y),
            (outer_size.width, outer_size.height),
            (monitor_pos.x, monitor_pos.y),
            (monitor_size.width, monitor_size.height),
        ) {
            layout.captured_rect = Some(rect);
            layout.requested_rect = Some(rect);
        }
    }
    if let Some(screen) = layout.captured_screen {
        layout.requested_screen = Some(screen);
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
        let entity = evt.window;
        if let Some(id) = windows.find_secondary_by_entity(entity) {
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

#[derive(QueryData)]
#[query_data(mutable)]
struct CameraViewportQuery {
    camera: &'static mut Camera,
    viewport_rect: &'static ViewportRect,
    graph_state: Option<&'static GraphState>,
}

fn set_camera_viewport(
    window: Query<(&Window, &bevy_egui::EguiContextSettings), With<PrimaryWindow>>,
    mut main_camera_query: Query<CameraViewportQuery, With<MainCamera>>,
) {
    let mut next_viewport_order = PRIMARY_VIEWPORT_ORDER_BASE;
    let mut next_graph_order = PRIMARY_GRAPH_ORDER_BASE;
    for CameraViewportQueryItem {
        mut camera,
        viewport_rect,
        graph_state,
    } in main_camera_query.iter_mut()
    {
        let order = if graph_state.is_some() {
            let order = next_graph_order;
            next_graph_order += 1;
            order
        } else {
            let order = next_viewport_order;
            next_viewport_order += 1;
            order
        };
        camera.order = order;
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
            let base_order = secondary_graph_order_base(state.id);
            camera.order = base_order + index as isize;
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
