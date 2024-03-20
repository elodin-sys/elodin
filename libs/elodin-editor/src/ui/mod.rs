use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::camera::Viewport,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText},
    EguiContexts,
};
use big_space::GridCell;
use conduit::{
    bevy::{ComponentValueMap, MaxTick, Received, Tick, TimeStep},
    query::MetadataStore,
    well_known::EntityMetadata,
    ControlMsg, EntityId,
};

use crate::MainCamera;

use self::widgets::{hierarchy, inspector, timeline};

pub mod colors;
pub mod images;
mod theme;
mod utils;
mod widgets;

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

#[derive(Resource, Default)]
pub struct SelectedEntity(pub Option<EntityPair>);

#[derive(Resource, Default)]
pub struct HoveredEntity(pub Option<EntityPair>);

#[derive(Resource, Default)]
pub struct EntityFilter(pub String);

#[derive(Clone, Copy)]
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

pub type EntityData<'a> = (&'a EntityId, Entity, &'a ComponentValueMap);
pub type EntityMeta<'a> = (&'a EntityId, Entity, &'a EntityMetadata);

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Paused>()
            .init_resource::<ShowStats>()
            .init_resource::<SelectedEntity>()
            .init_resource::<HoveredEntity>()
            .init_resource::<EntityFilter>()
            .add_systems(Update, shortcuts)
            .add_systems(Update, render)
            .add_systems(Update, render_viewport.after(render))
            .add_systems(Update, set_camera_viewport.after(render_viewport));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    mut event: EventWriter<ControlMsg>,
    mut contexts: EguiContexts,
    mut paused: ResMut<Paused>,
    mut tick: ResMut<Tick>,
    entity_filter: ResMut<EntityFilter>,
    selected_entity: ResMut<SelectedEntity>,
    max_tick: Res<MaxTick>,
    tick_time: Res<TimeStep>,
    entities: Query<EntityData>,
    entities_meta: Query<EntityMeta>,
    window: Query<&Window>,
    images: Local<images::Images>,
    metadata_store: Res<MetadataStore>,
) {
    let window = window.single();
    let width = window.resolution.width();
    let height = window.resolution.height();

    theme::set_theme(contexts.ctx_mut());

    let is_loading = entities_meta.is_empty();

    if is_loading {
        let logo_size = egui::vec2(48.0, 60.0);
        let logo_image_id = contexts.add_image(images.logo.clone_weak());

        egui::CentralPanel::default()
            .frame(egui::Frame {
                fill: colors::STONE_950,
                ..Default::default()
            })
            .show(contexts.ctx_mut(), |ui| {
                ui.painter().image(
                    logo_image_id,
                    egui::Rect::from_center_size(egui::pos2(width / 2.0, height / 2.0), logo_size),
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    colors::WHITE,
                );
            });

        return;
    }

    let selected_entity_pair = selected_entity.0;

    let selected_entity_full = selected_entity_pair.and_then(|eid| entities.get(eid.bevy).ok());
    let selected_entity_meta = entities_meta
        .iter()
        .find(|(id, _, _)| selected_entity_pair.is_some_and(|eid| eid.conduit == **id))
        .map(|(_, _, metadata)| metadata.to_owned());

    let timeline_icons = timeline::TimelineIcons {
        jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
        jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
        frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
        frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
        play: contexts.add_image(images.icon_play.clone_weak()),
        pause: contexts.add_image(images.icon_pause.clone_weak()),
        handle: contexts.add_image(images.icon_scrub.clone_weak()),
    };
    let search_icon = contexts.add_image(images.icon_search.clone_weak());

    #[cfg(target_os = "macos")]
    egui::TopBottomPanel::top("titlebar")
        .frame(egui::Frame {
            fill: colors::INTERFACE_BACKGROUND_BLACK,
            stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
            ..Default::default()
        })
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| ui.set_height(48.0));

    if width * 0.75 > height {
        egui::SidePanel::new(egui::panel::Side::Left, "outline_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::STONE_950,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                inner_margin: Margin::same(4.0),
                ..Default::default()
            })
            .min_width(width * 0.15)
            .default_width(width * 0.20)
            .max_width(width * 0.35)
            .show(contexts.ctx_mut(), |ui| {
                let search_text = entity_filter.0.clone();

                hierarchy::header(ui, entity_filter, search_icon, false);

                hierarchy::entity_list(ui, entities_meta, selected_entity, &search_text);
            });

        egui::SidePanel::new(egui::panel::Side::Right, "inspector_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::STONE_950,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                ..Default::default()
            })
            .min_width(width * 0.15)
            .default_width(width * 0.25)
            .max_width(width * 0.35)
            .show(contexts.ctx_mut(), |ui| {
                inspector::inspector(
                    ui,
                    selected_entity_meta,
                    selected_entity_full,
                    &metadata_store,
                );
            });
    } else {
        egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "section_bottom")
            .resizable(true)
            .frame(egui::Frame::default())
            .default_height(200.0)
            .max_height(width * 0.5)
            .show(contexts.ctx_mut(), |ui| {
                let outline = egui::SidePanel::new(egui::panel::Side::Left, "outline_bottom")
                    .resizable(true)
                    .frame(egui::Frame {
                        fill: colors::STONE_950,
                        stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                        inner_margin: Margin::same(4.0),
                        ..Default::default()
                    })
                    .min_width(width * 0.25)
                    .default_width(width * 0.4)
                    .max_width(width * 0.75)
                    .show_inside(ui, |ui| {
                        let search_text = entity_filter.0.clone();

                        hierarchy::header(ui, entity_filter, search_icon, true);

                        hierarchy::entity_list(ui, entities_meta, selected_entity, &search_text);

                        ui.allocate_space(ui.available_size());
                    });

                egui::SidePanel::new(egui::panel::Side::Right, "inspector_bottom")
                    .resizable(false)
                    .frame(egui::Frame {
                        fill: colors::STONE_950,
                        ..Default::default()
                    })
                    .exact_width(width - outline.response.rect.width())
                    .show_inside(ui, |ui| {
                        inspector::inspector(
                            ui,
                            selected_entity_meta,
                            selected_entity_full,
                            &metadata_store,
                        );
                    });
            });
    }

    let sim_fps = 1.0 / tick_time.0.as_secs_f64();

    egui::TopBottomPanel::bottom("timeline")
        .frame(egui::Frame {
            fill: colors::STONE_950,
            stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
            ..Default::default()
        })
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            timeline::timeline_area(
                ui,
                &mut paused,
                &max_tick,
                &mut tick,
                sim_fps,
                &mut event,
                timeline_icons,
            );
        });
}

pub fn render_viewport(
    mut contexts: EguiContexts,
    window: Query<&Window>,
    entities_meta: Query<EntityMeta>,
    show_stats: Res<ShowStats>,
    tick_time: Res<TimeStep>,
    diagnostics: Res<DiagnosticsStore>,
    hovered_entity: Res<HoveredEntity>,
) {
    let window = window.single();

    let hovered_entity_meta = if let Some(hovered_entity_pair) = hovered_entity.0 {
        entities_meta
            .iter()
            .find(|(id, _, _)| hovered_entity_pair.conduit == **id)
            .map(|(_, _, metadata)| metadata.to_owned())
    } else {
        None
    };

    if let Some(hovered_entity_meta) = hovered_entity_meta {
        contexts
            .ctx_mut()
            .set_cursor_icon(egui::CursorIcon::PointingHand);

        if let Some(cursor_pos) = window.cursor_position() {
            let offset = 16.0;
            let window_pos = egui::pos2(cursor_pos.x + offset, cursor_pos.y + offset);

            egui::Window::new("hovered_entity")
                .title_bar(false)
                .resizable(false)
                .frame(egui::Frame {
                    fill: colors::with_opacity(colors::STONE_950, 0.5),
                    stroke: egui::Stroke::new(1.0, colors::with_opacity(colors::WHITE, 0.5)),
                    inner_margin: egui::Margin::symmetric(16.0, 8.0),
                    ..Default::default()
                })
                .fixed_pos(window_pos)
                .show(contexts.ctx_mut(), |ui| {
                    ui.add(Label::new(
                        RichText::new(hovered_entity_meta.name).color(Color32::WHITE),
                    ));
                });
        }
    }

    if show_stats.0 {
        let sim_fps = 1.0 / tick_time.0.as_secs_f64();

        let viewport_left_top = contexts.ctx_mut().available_rect().left_top();
        let viewport_margins = egui::vec2(16.0, 16.0);

        egui::Window::new("stats")
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::default())
            .fixed_pos(viewport_left_top + viewport_margins)
            .show(contexts.ctx_mut(), |ui| {
                let render_fps_str = diagnostics
                    .get(&FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                    .map_or(" N/A".to_string(), |value| format!("{value:>4.0}"));

                ui.add(Label::new(
                    RichText::new(format!("FPS [VIEW]: {render_fps_str}")).color(Color32::WHITE),
                ));

                ui.add(Label::new(
                    RichText::new(format!("FPS [SIM]: {sim_fps:>4.0}")).color(Color32::WHITE),
                ));
            });
    }
}

pub fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
    mut contexts: EguiContexts,
    mut main_camera_query: Query<
        (Entity, &mut Camera, &mut Transform, &mut GridCell<i128>),
        With<MainCamera>,
    >,
    selected_entity: ResMut<SelectedEntity>,
    entity_transform_query: Query<&GridCell<i128>, (With<Received>, Without<MainCamera>)>,
    mut commands: Commands,
) {
    let available_rect = contexts.ctx_mut().available_rect();

    let window = window.single();
    let scale_factor = window.scale_factor() * egui_settings.scale_factor;
    let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
    let viewport_size = available_rect.size() * scale_factor;
    if available_rect.size().x > window.width() || available_rect.size().y > window.height() {
        return;
    }

    let (entity, mut camera, mut cam_transform, mut grid_cell) = main_camera_query.single_mut();
    camera.viewport = Some(Viewport {
        physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
        physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
        depth: 0.0..1.0,
    });

    let mut entity = commands.entity(entity);
    if selected_entity.is_changed() {
        if let Some(entity_pair) = selected_entity.0 {
            if let Ok(entity_cell) = entity_transform_query.get(entity_pair.bevy) {
                entity.set_parent(entity_pair.bevy);
                let rot_matrix = Mat3::from_quat(cam_transform.rotation);
                cam_transform.translation = rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, 10.0));
                *grid_cell = *entity_cell;
            }
        } else {
            entity.remove_parent();
        }
    }
}
