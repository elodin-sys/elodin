use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::camera::Viewport,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText},
    EguiContexts,
};
use conduit::{
    bevy::{ComponentValueMap, MaxTick, Tick},
    well_known::EntityMetadata,
    ControlMsg, EntityId,
};

use crate::MainCamera;

use self::widgets::{list, timeline};

mod colors;
pub mod images;
mod theme;
mod widgets;

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

#[derive(Resource, Default)]
pub struct SelectedEntity(pub Option<EntityPair>);

#[derive(Clone, Copy)]
pub struct EntityPair {
    pub bevy: Entity,
    pub conduit: EntityId,
}

pub fn shortcuts(
    mut show_stats: ResMut<ShowStats>,
    mut paused: ResMut<Paused>,
    kbd: Res<Input<KeyCode>>,
) {
    if kbd.just_pressed(KeyCode::F12) {
        show_stats.0 = !show_stats.0;
    }

    if kbd.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    mut event: EventWriter<ControlMsg>,
    mut contexts: EguiContexts,
    mut paused: ResMut<Paused>,
    mut tick: ResMut<Tick>,
    selected_entity: ResMut<SelectedEntity>,
    max_tick: Res<MaxTick>,
    show_stats: Res<ShowStats>,
    diagnostics: Res<DiagnosticsStore>,
    meta_entities: Query<(Entity, &EntityId, &EntityMetadata)>,
    entity_components: Query<(Entity, &EntityId, &ComponentValueMap)>,
    window: Query<&Window>,
    images: Local<images::Images>,
) {
    let window = window.single();
    let width = window.resolution.width();
    let height = window.resolution.height();
    let selected = selected_entity.0;

    theme::set_theme(contexts.ctx_mut());

    let timeline_icons = timeline::TimelineIcons {
        jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
        jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
        frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
        frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
        play: contexts.add_image(images.icon_play.clone_weak()),
        pause: contexts.add_image(images.icon_pause.clone_weak()),
        handle: contexts.add_image(images.icon_scrub.clone_weak()),
    };

    let selected_metadata = selected_entity
        .0
        .and_then(|e| meta_entities.get(e.bevy).ok())
        .map(|(b, _, meta)| (b, meta.clone()));

    egui::TopBottomPanel::top("titlebar")
        .frame(egui::Frame {
            fill: colors::INTERFACE_BACKGROUND_BLACK,
            stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
            ..Default::default()
        })
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| ui.set_height(48.0));

    // NOTE(temp fix): Hide panels until simulation is loaded
    if !meta_entities.is_empty() {
        // let entities =
        // entity_components
        //     .iter()
        //     .collect::<Vec<(Entity, &EntityId, &WorldPos, &ComponentValueMap)>>();
        // NOTE(sphw): temporarily not needed @andrei will use this

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
                .max_width(width * 0.25)
                .show(contexts.ctx_mut(), |ui| {
                    list::entity_list(ui, meta_entities, selected_entity);
                });

            if let Some((entity, metadata)) = selected_metadata.as_ref() {
                egui::SidePanel::new(egui::panel::Side::Right, "inspector_side")
                    .resizable(true)
                    .frame(egui::Frame {
                        fill: colors::STONE_950,
                        stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                        inner_margin: Margin::same(4.0),
                        ..Default::default()
                    })
                    .min_width(width * 0.15)
                    .default_width(width * 0.20)
                    .max_width(width * 0.25)
                    .show(contexts.ctx_mut(), |ui| {
                        ui.label(metadata.name.clone());
                        if let Ok((_, _, map)) = entity_components.get(*entity) {
                            draw_component_values(ui, map)
                        }

                        ui.allocate_space(ui.available_size());
                    });
            }
        } else {
            egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "section_bottom")
                .resizable(true)
                .frame(egui::Frame::default())
                .default_height(200.0)
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
                        .default_width(width * 0.5)
                        .max_width(width * 0.75)
                        .show_inside(ui, |ui| {
                            list::entity_list(ui, meta_entities, selected_entity);
                            ui.allocate_space(ui.available_size());
                        });

                    if let Some((entity, metadata)) = selected_metadata.as_ref() {
                        egui::SidePanel::new(egui::panel::Side::Right, "inspector_bottom")
                            .resizable(false)
                            .frame(egui::Frame {
                                fill: colors::STONE_950,
                                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                                inner_margin: Margin::same(4.0),
                                ..Default::default()
                            })
                            .exact_width(width - outline.response.rect.width())
                            .show_inside(ui, |ui| {
                                ui.label(metadata.name.clone());
                                if let Ok((_, _, map)) = entity_components.get(*entity) {
                                    draw_component_values(ui, map)
                                }
                                ui.allocate_space(ui.available_size());
                            });
                    }
                });
        }
    }

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
                &mut event,
                timeline_icons,
            );
        });

    let viewport_left_top = contexts.ctx_mut().available_rect().left_top();
    let viewport_margins = egui::vec2(16.0, 16.0);

    if show_stats.0 {
        egui::Window::new("stats")
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::default())
            .fixed_pos(viewport_left_top + viewport_margins)
            .show(contexts.ctx_mut(), |ui| {
                let fps_str = diagnostics
                    .get(FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                    .map_or(" N/A".to_string(), |value| format!("{value:>4.0}"));

                ui.add(Label::new(
                    RichText::new(format!("FPS: {fps_str}")).color(Color32::WHITE),
                ));
                if let Some((_, entity_id, map)) =
                    selected.and_then(|s| entity_components.get(s.bevy).ok())
                {
                    ui.add(Label::new(
                        RichText::new(format!("SELECTED: ID[{}] ", entity_id.0,))
                            .color(Color32::WHITE),
                    ));
                    draw_component_values(ui, map)
                }
            });
    }
}

fn draw_component_values(ui: &mut egui::Ui, map: &ComponentValueMap) {
    for (id, value) in map.0.iter() {
        ui.add(Label::new(
            RichText::new(format!("COMP ID[{}] VAL = {:?}", id.0, value)).color(Color32::WHITE),
        ));
    }
}

pub fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
    mut contexts: EguiContexts,
    mut main_camera_query: Query<&mut Camera, With<MainCamera>>,
) {
    let available_rect = contexts.ctx_mut().available_rect();

    let window = window.single();
    let scale_factor = (window.scale_factor() * egui_settings.scale_factor) as f32;

    let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
    let viewport_size = available_rect.size() * scale_factor;

    let mut camera = main_camera_query.single_mut();
    camera.viewport = Some(Viewport {
        physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
        physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
        depth: 0.0..1.0,
    });
}
