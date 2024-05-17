use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::{
        query::QueryData,
        system::{SystemParam, SystemState},
    },
    prelude::*,
    render::camera::Viewport,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText},
    EguiContexts,
};

use big_space::GridCell;
use conduit::bevy::MaxTick;

use conduit::{
    bevy::{ComponentValueMap, Received, TimeStep},
    well_known::EntityMetadata,
    ComponentId, EntityId,
};
use egui_tiles::TileId;

use crate::{GridHandle, MainCamera};

use self::widgets::hierarchy::Hierarchy;
use self::widgets::inspector::Inspector;
use self::widgets::modal::ModalWithSettings;
use self::widgets::timeline::tagged_range::{TaggedRangeId, TaggedRangesPanel};
use self::widgets::timeline::TimelineArgs;
use self::widgets::timeline::{tagged_range::TaggedRanges, timeline_widget};
use self::widgets::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystemExt};
use self::{
    utils::MarginSides,
    widgets::{button::EImageButton, inspector},
};

pub mod colors;
pub mod images;
mod theme;
mod tiles;
pub mod utils;
pub mod widgets;

#[derive(Resource, Default)]
pub struct HdrEnabled(pub bool);

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

#[derive(Resource, Default)]
pub struct ViewportRange(pub Option<TaggedRangeId>);

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
    pub fn is_entity_selected(&self, id: conduit::EntityId) -> bool {
        matches!(self, SelectedObject::Entity(pair) if pair.conduit == id)
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
    pub conduit: EntityId,
}

pub fn shortcuts(
    mut show_stats: ResMut<ShowStats>,
    mut paused: ResMut<Paused>,
    kbd: Res<ButtonInput<KeyCode>>,
) {
    if kbd.just_pressed(KeyCode::F12) {
        show_stats.0 = !show_stats.0;
    }

    if kbd.just_pressed(KeyCode::Space) {
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
            .init_resource::<ShowStats>()
            .init_resource::<SelectedObject>()
            .init_resource::<HoveredEntity>()
            .init_resource::<EntityFilter>()
            .init_resource::<InspectorAnchor>()
            .init_resource::<tiles::TileState>()
            .init_resource::<SidebarState>()
            .init_resource::<FullscreenState>()
            .init_resource::<TaggedRanges>()
            .init_resource::<SettingModalState>()
            .init_resource::<HdrEnabled>()
            .init_resource::<ViewportRange>()
            .add_systems(Update, shortcuts)
            .add_systems(Update, render_layout)
            .add_systems(Update, sync_hdr)
            .add_systems(Update, tiles::sync_viewports.after(render_layout))
            .add_systems(Update, tiles::setup_default_tiles.after(render_layout))
            .add_systems(Update, tiles::shortcuts)
            .add_systems(Update, set_camera_viewport.after(render_layout))
            .add_systems(Update, sync_camera_grid_cell.after(render_layout));
    }
}

#[derive(Clone, Debug)]
pub enum SettingModal {
    Graph(Entity, Option<EntityId>, Option<ComponentId>),
    GraphRename(Entity, String),
    RangeEdit(TaggedRangeId, String, egui::Color32),
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
pub struct Titlebar<'w> {
    sidebar_state: ResMut<'w, SidebarState>,
    fullscreen_state: ResMut<'w, FullscreenState>,
}

impl RootWidgetSystem for Titlebar<'_> {
    type Args = TitlebarIcons;
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

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
        } else {
            16.0
        };
        let traffic_light_offset = if cfg!(target_os = "macos") { 72.0 } else { 0.0 };
        let titlebar_scale = if cfg!(target_os = "macos") { 1.4 } else { 1.3 };
        let titlebar_margin = if cfg!(target_os = "macos") { 8.0 } else { 4.0 };

        egui::TopBottomPanel::top("titlebar")
            .frame(
                egui::Frame {
                    fill: colors::PRIMARY_ONYX,
                    stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
                    ..Default::default()
                }
                .inner_margin(Margin::same(titlebar_margin).left(16.0).right(16.0)),
            )
            .resizable(false)
            .show(ctx, |ui| {
                ui.set_height(titlebar_height - titlebar_margin * 2.0);
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

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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

        theme::set_theme(contexts.ctx_mut());

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

        if width * 0.75 > height {
            world.add_root_widget_with::<Hierarchy>("hierarchy", (icon_search, width));

            world.add_root_widget_with::<Inspector>("inspector", (inspector_icons, width));
        } else {
            egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "section_bottom")
                .resizable(true)
                .frame(egui::Frame::default())
                .default_height(200.0)
                .max_height(width * 0.5)
                .show(ctx, |ui| {
                    let hierarchy_width =
                        ui.add_widget_with::<Hierarchy>(world, "hierarchy", (icon_search, width));

                    let inspector_width = width - hierarchy_width;

                    ui.add_widget_with::<Inspector>(
                        world,
                        "inspector",
                        (inspector_icons, inspector_width),
                    );
                });
        }
    }
}

#[derive(SystemParam)]
pub struct TimelinePanel<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    max_tick: Res<'w, MaxTick>,
    tick_time: Res<'w, TimeStep>,
    tagged_ranges: Res<'w, TaggedRanges>,
}

impl RootWidgetSystem for TimelinePanel<'_, '_> {
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
        let max_tick = state_mut.max_tick;
        let tick_time = state_mut.tick_time;

        let ranges_not_empty = state_mut.tagged_ranges.is_not_empty();

        let active_range = 0..=max_tick.0;
        let frames_per_second = 1.0 / tick_time.0.as_secs_f64();

        theme::set_theme(ctx);

        let timeline_icons = timeline_widget::TimelineIcons {
            jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
            jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
            frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
            frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
            play: contexts.add_image(images.icon_play.clone_weak()),
            pause: contexts.add_image(images.icon_pause.clone_weak()),
            handle: contexts.add_image(images.icon_scrub.clone_weak()),
        };

        egui::TopBottomPanel::bottom("timeline_panel")
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                ..Default::default()
            })
            .resizable(false)
            .show(ctx, |ui| {
                let available_width = ui.available_width();
                let timeline_args = TimelineArgs {
                    available_width,
                    segment_count: (available_width / 100.0) as u8,
                    frames_per_second,
                    active_range,
                };

                ui.add_widget_with::<timeline_widget::TimelineWithControls>(
                    world,
                    "timeline_with_controls",
                    (timeline_icons, timeline_args.clone()),
                );

                if ranges_not_empty {
                    ui.add_widget_with::<TaggedRangesPanel>(
                        world,
                        "tagged_ranges_panel",
                        timeline_args,
                    );
                }
            });
    }
}

#[derive(SystemParam)]
pub struct ViewportOverlay<'w, 's> {
    window: Query<'w, 's, &'static Window>,
    entities_meta: Query<'w, 's, EntityData<'static>>,
    show_stats: Res<'w, ShowStats>,
    tick_time: Res<'w, TimeStep>,
    diagnostics: Res<'w, DiagnosticsStore>,
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
        let show_stats = state_mut.show_stats;
        let tick_time = state_mut.tick_time;
        let diagnostics = state_mut.diagnostics;
        let hovered_entity = state_mut.hovered_entity;

        let Ok(window) = window.get_single() else {
            return;
        };

        let hovered_entity_meta = if let Some(hovered_entity_pair) = hovered_entity.0 {
            entities_meta
                .iter()
                .find(|(id, _, _, _)| hovered_entity_pair.conduit == **id)
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

        if show_stats.0 {
            let sim_fps = 1.0 / tick_time.0.as_secs_f64();

            let viewport_left_top = ctx.available_rect().left_top();
            let viewport_margins = egui::vec2(16.0, 40.0);

            egui::Window::new("stats")
                .title_bar(false)
                .resizable(false)
                .frame(egui::Frame::default())
                .fixed_pos(viewport_left_top + viewport_margins)
                .show(ctx, |ui| {
                    let render_fps_str = diagnostics
                        .get(&FrameTimeDiagnosticsPlugin::FPS)
                        .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                        .map_or(" N/A".to_string(), |value| format!("{value:>6.2}"));

                    ui.add(Label::new(
                        RichText::new(format!("FPS [VIEW]: {render_fps_str}"))
                            .color(Color32::WHITE),
                    ));

                    ui.add(Label::new(
                        RichText::new(format!("FPS [SIM]:  {sim_fps:>6.2}")).color(Color32::WHITE),
                    ));
                });
        }
    }
}

pub fn render_layout(world: &mut World) {
    world.add_root_widget::<MainLayout>("main_layout");

    world.add_root_widget::<TimelinePanel>("timeline_panel");

    world.add_root_widget::<tiles::TileLayout>("tile_layout");

    world.add_root_widget::<ViewportOverlay>("viewport_overlay");

    world.add_root_widget::<ModalWithSettings>("modal_graph");
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
