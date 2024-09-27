use bevy::{
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    input::keyboard::Key,
    prelude::*,
    render::camera::Viewport,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText},
    EguiContexts,
};

use big_space::GridCell;

use egui::Rounding;
use egui_tiles::TileId;
use impeller::{
    bevy::{ComponentValueMap, Received},
    well_known::EntityMetadata,
    ComponentId, EntityId,
};
use widgets::{command_palette::CommandPaletteState, timeline};
use widgets::{status_bar::StatusBar, timeline::timeline_ranges};

use crate::{plugins::LogicalKeyState, GridHandle, MainCamera};

use self::widgets::inspector::{entity::ComponentFilter, Inspector};
use self::widgets::modal::ModalWithSettings;
use self::widgets::timeline::timeline_ranges::TimelineRangeId;

use self::widgets::{
    command_palette::{self, CommandPalette},
    hierarchy::Hierarchy,
};
use self::widgets::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystemExt};
use self::{
    utils::MarginSides,
    widgets::{button::EImageButton, inspector},
};

pub mod colors;
pub mod images;
mod theme;
pub mod tiles;
pub mod utils;
pub mod widgets;

#[derive(Resource, Default)]
pub struct HdrEnabled(pub bool);

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ViewportRange(pub Option<TimelineRangeId>);

#[derive(Resource, Default, Debug, Clone)]
pub enum SelectedObject {
    #[default]
    None,
    Entity(EntityPair),
    Viewport {
        camera: Entity,
        tile_id: TileId,
    },
    Graph {
        tile_id: TileId,
        label: String,
        graph_id: Entity,
    },
}

impl SelectedObject {
    pub fn is_entity_selected(&self, id: impeller::EntityId) -> bool {
        matches!(self, SelectedObject::Entity(pair) if pair.impeller == id)
    }

    pub fn is_tile_selected(&self, tile_id: TileId) -> bool {
        self.tile_id() == Some(tile_id)
    }

    pub fn tile_id(&self) -> Option<TileId> {
        match self {
            SelectedObject::None => None,
            SelectedObject::Entity(_) => None,
            SelectedObject::Viewport { tile_id, .. } => Some(*tile_id),
            SelectedObject::Graph { tile_id, .. } => Some(*tile_id),
        }
    }
}

#[derive(Resource, Default)]
pub struct HoveredEntity(pub Option<EntityPair>);

#[derive(Resource, Default)]
pub struct EntityFilter(pub String);

#[derive(Resource, Default)]
pub struct InspectorAnchor(pub Option<egui::Pos2>);

#[derive(Component)]
pub struct ViewportRect(pub Option<egui::Rect>);

#[derive(Clone, Copy, Debug)]
pub struct EntityPair {
    pub bevy: Entity,
    pub impeller: EntityId,
}

pub fn shortcuts(
    mut paused: ResMut<Paused>,
    timeline_ranges_focused: Res<timeline_ranges::TimelineRangesFocused>,
    command_palette_state: Res<CommandPaletteState>,
    key_state: Res<LogicalKeyState>,
) {
    let input_has_focus = timeline_ranges_focused.0 || command_palette_state.show;

    if !input_has_focus && key_state.just_pressed(&Key::Space) {
        paused.0 = !paused.0;
    }
}

pub type EntityData<'a> = (
    &'a EntityId,
    Entity,
    &'a mut ComponentValueMap,
    &'a EntityMetadata,
);

pub type EntityDataReadOnly<'a> = (
    &'a EntityId,
    Entity,
    &'a ComponentValueMap,
    &'a EntityMetadata,
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
    parent: Option<&'static Parent>,
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
            .init_resource::<tiles::NewTileState>()
            .init_resource::<SidebarState>()
            .init_resource::<FullscreenState>()
            .init_resource::<timeline_ranges::TimelineRanges>()
            .init_resource::<timeline_ranges::TimelineRangesFocused>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<ViewportRange>()
            .init_resource::<command_palette::CommandPaletteState>()
            .add_systems(Update, shortcuts)
            .add_systems(Update, render_layout)
            .add_systems(Update, sync_hdr)
            .add_systems(Update, tiles::sync_viewports.after(render_layout))
            .add_systems(Update, tiles::shortcuts)
            .add_systems(Update, set_camera_viewport.after(render_layout))
            .add_systems(Update, sync_camera_grid_cell.after(render_layout));
    }
}

#[derive(Clone, Debug)]
pub enum SettingModal {
    Graph(Entity, Option<EntityId>, Option<ComponentId>),
    GraphRename(Entity, String),
}

#[derive(Resource, Default, Clone, Debug)]
pub struct SettingModalState(pub Option<SettingModal>);

#[derive(Resource)]
pub struct SidebarState {
    pub left_open: bool,
    pub right_open: bool,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            left_open: true,
            right_open: true,
        }
    }
}

#[derive(Resource, Default)]
pub struct FullscreenState(pub bool);

pub struct TitlebarIcons {
    pub icon_side_bar_right: egui::TextureId,
    pub icon_side_bar_left: egui::TextureId,
    pub icon_fullscreen: egui::TextureId,
    pub icon_exit_fullscreen: egui::TextureId,
}

#[derive(SystemParam)]
pub struct Titlebar<'w, 's> {
    sidebar_state: ResMut<'w, SidebarState>,
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

        let mut sidebar_state = state_mut.sidebar_state;
        let mut fullscreen_state = state_mut.fullscreen_state;

        let TitlebarIcons {
            icon_side_bar_right,
            icon_side_bar_left,
            icon_fullscreen,
            icon_exit_fullscreen,
        } = args;

        let titlebar_height = if cfg!(target_os = "macos") {
            52.0
        } else if cfg!(target_os = "windows") {
            45.0
        } else {
            34.0
        };
        let traffic_light_offset = if cfg!(target_os = "macos") { 72.0 } else { 0.0 };
        let titlebar_scale = if cfg!(target_os = "macos") { 1.4 } else { 1.3 };
        let titlebar_margin = if cfg!(target_os = "macos") {
            8.0
        } else if cfg!(target_os = "windows") {
            0.0
        } else {
            4.0
        };
        let titlebar_right_margin = if cfg!(target_os = "windows") {
            0.0
        } else {
            16.0
        };

        theme::set_theme(ctx);
        egui::TopBottomPanel::top("title_bar")
            .frame(
                egui::Frame {
                    fill: Color32::TRANSPARENT,
                    stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
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
                        if cfg!(target_os = "windows") {
                            let (window_id, _, _) = state_mut.windows.single();
                            let winit_window =
                                state_mut.winit_windows.get_window(window_id).unwrap();
                            ui.horizontal_centered(|ui| {
                                ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                                    egui::Color32::from_hex("#E81123").expect("invalid red color");

                                ui.style_mut().visuals.widgets.hovered.rounding = Rounding {
                                    nw: 0.0,
                                    ne: 4.0,
                                    sw: 0.0,
                                    se: 0.0,
                                };

                                if ui
                                    .add_sized(
                                        egui::vec2(45.0, 40.0),
                                        egui::Button::new(
                                            RichText::new("\u{e8bb}")
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
                                    state_mut.app_exit.send(AppExit::Success);
                                }
                                ui.add_space(2.0);
                            });

                            let maximized = winit_window.is_maximized();
                            ui.scope(|ui| {
                                ui.style_mut().visuals.widgets.hovered.weak_bg_fill =
                                    egui::Color32::from_hex("#4E4D53").expect("invalid red color");
                                ui.style_mut().visuals.widgets.hovered.rounding = Rounding::ZERO;
                                if ui
                                    .add_sized(
                                        egui::vec2(45.0, 40.0),
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
                                        .stroke(egui::Stroke::NONE),
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
                                            RichText::new("\u{e921}")
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
                        if ui
                            .add(
                                EImageButton::new(icon_side_bar_right)
                                    .scale(titlebar_scale, titlebar_scale)
                                    .bg_color(Color32::TRANSPARENT),
                            )
                            .clicked()
                        {
                            sidebar_state.right_open = !sidebar_state.right_open;
                        };
                        ui.add_space(4.0);
                        if ui
                            .add(
                                EImageButton::new(icon_side_bar_left)
                                    .scale(titlebar_scale, titlebar_scale)
                                    .bg_color(Color32::TRANSPARENT),
                            )
                            .clicked()
                        {
                            sidebar_state.left_open = !sidebar_state.left_open;
                        };
                    });
                });
            });
    }
}

#[derive(SystemParam)]
pub struct MainLayout<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    window: Query<'w, 's, &'static Window>,
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
        let window = state_mut.window;
        let images = state_mut.images;

        let Ok(window) = window.get_single() else {
            return;
        };
        let width = window.resolution.width();
        let height = window.resolution.height();

        theme::set_theme(ctx);

        let icon_search = contexts.add_image(images.icon_search.clone_weak());

        let inspector_icons = inspector::InspectorIcons {
            chart: contexts.add_image(images.icon_chart.clone_weak()),
            add: contexts.add_image(images.icon_add.clone_weak()),
            subtract: contexts.add_image(images.icon_subtract.clone_weak()),
            setting: contexts.add_image(images.icon_setting.clone_weak()),
            search: contexts.add_image(images.icon_search.clone_weak()),
        };

        let titlebar_icons = TitlebarIcons {
            icon_side_bar_right: contexts.add_image(images.icon_side_bar_right.clone_weak()),
            icon_side_bar_left: contexts.add_image(images.icon_side_bar_left.clone_weak()),
            icon_fullscreen: contexts.add_image(images.icon_fullscreen.clone_weak()),
            icon_exit_fullscreen: contexts.add_image(images.icon_exit_fullscreen.clone_weak()),
        };

        world.add_root_widget_with::<Titlebar>("titlebar", titlebar_icons);

        let landscape_layout = width * 0.75 > height;

        world.add_root_widget::<StatusBar>("status_bar");

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                if landscape_layout {
                    ui.add_widget::<timeline::TimelinePanel>(world, "timeline_panel");
                }

                if landscape_layout {
                    ui.add_widget_with::<Hierarchy>(
                        world,
                        "hierarchy",
                        (false, icon_search, width),
                    );

                    ui.add_widget_with::<Inspector>(
                        world,
                        "inspector",
                        (false, inspector_icons, width),
                    );
                } else {
                    egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "section_bottom")
                        .resizable(true)
                        .frame(egui::Frame::default())
                        .default_height(200.0)
                        .max_height(width * 0.5)
                        .show_inside(ui, |ui| {
                            let hierarchy_width = ui.add_widget_with::<Hierarchy>(
                                world,
                                "hierarchy",
                                (true, icon_search, width),
                            );

                            let inspector_width = width - hierarchy_width;

                            ui.add_widget_with::<Inspector>(
                                world,
                                "inspector",
                                (true, inspector_icons, inspector_width),
                            );
                        });

                    ui.add_widget::<timeline::TimelinePanel>(world, "timeline_panel");
                }

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

        let Ok(window) = window.get_single() else {
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
                        fill: colors::with_opacity(colors::PRIMARY_SMOKE, 0.5),
                        stroke: egui::Stroke::new(1.0, colors::with_opacity(colors::WHITE, 0.5)),
                        inner_margin: egui::Margin::symmetric(16.0, 8.0),
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

    world.add_root_widget::<ModalWithSettings>("modal_graph");

    world.add_root_widget::<CommandPalette>("command_palette");
}

#[derive(QueryData)]
#[query_data(mutable)]
struct CameraViewportQuery {
    camera: &'static mut Camera,
    viewport_rect: &'static ViewportRect,
}

fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
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
        let Some(window) = window.iter().next() else {
            continue;
        };
        let scale_factor = window.scale_factor() * egui_settings.scale_factor;
        let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = available_rect.size() * scale_factor;
        if available_rect.size().x > window.width() || available_rect.size().y > window.height() {
            return;
        }
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
            physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
            depth: 0.0..1.0,
        });
    }
}

fn sync_camera_grid_cell(
    mut query: Query<(Option<&Parent>, &mut GridCell<i128>), With<MainCamera>>,
    entity_transform_query: Query<&GridCell<i128>, (With<Received>, Without<MainCamera>)>,
) {
    for (parent, mut grid_cell) in query.iter_mut() {
        if let Some(parent) = parent {
            if let Ok(entity_cell) = entity_transform_query.get(parent.get()) {
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
