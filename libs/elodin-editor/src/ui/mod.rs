use bevy::{
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    input::keyboard::Key,
    prelude::*,
    render::camera::Viewport,
    window::PrimaryWindow,
};
use bevy_egui::{
    EguiContext, EguiContexts,
    egui::{self, Color32, Label, Margin, RichText},
};

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
        app.init_resource::<Paused>()
            .init_resource::<SelectedObject>()
            .init_resource::<HoveredEntity>()
            .init_resource::<EntityFilter>()
            .init_resource::<ComponentFilter>()
            .init_resource::<InspectorAnchor>()
            .init_resource::<tiles::TileState>()
            .init_resource::<FullscreenState>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<timeline_slider::UITick>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_systems(Update, timeline_slider::sync_ui_tick.before(render_layout))
            .add_systems(Update, actions::spawn_lua_actor)
            .add_systems(Update, shortcuts)
            .add_systems(Update, render_layout)
            .add_systems(Update, sync_hdr)
            .add_systems(Update, tiles::shortcuts)
            .add_systems(Update, set_camera_viewport.after(render_layout))
            .add_systems(Update, sync_camera_grid_cell.after(render_layout))
            .add_systems(Update, query_plot::auto_bounds)
            .add_systems(Update, dashboard::update_nodes)
            .add_plugins(SchematicPlugin)
            .add_plugins(LinePlot3dPlugin);
    }
}

#[derive(Clone, Debug)]
pub enum SettingModal {
    Graph(Entity, Option<ComponentId>),
    GraphRename(Entity, String),
}

#[derive(Resource, Default, Clone, Debug)]
pub struct SettingModalState(pub Option<SettingModal>);

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
                ui.add_widget::<tiles::TileSystem>(world, "tile_system");
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

    //world.add_root_widget::<ModalWithSettings>("modal_graph");

    world.add_root_widget::<CommandPalette>("command_palette");
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

fn sync_camera_grid_cell(
    mut query: Query<(Option<&ChildOf>, &mut GridCell<i128>), With<MainCamera>>,
    entity_transform_query: Query<&GridCell<i128>, Without<MainCamera>>,
) {
    for (parent, mut grid_cell) in query.iter_mut() {
        if let Some(parent) = parent {
            if let Ok(entity_cell) = entity_transform_query.get(parent.parent()) {
                *grid_cell = *entity_cell;
            }
        }
    }
}
fn sync_hdr(hdr_enabled: ResMut<HdrEnabled>, mut query: Query<&mut Camera>) {
    for mut cam in query.iter_mut() {
        cam.hdr = hdr_enabled.0;
    }
}
