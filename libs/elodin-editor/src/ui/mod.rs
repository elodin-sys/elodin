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
    bevy::{EntityMap, MaxTick, Tick},
    ControlMsg,
};

use crate::MainCamera;

use self::widgets::timeline::{timeline_area, TimelineIcons};

mod colors;
pub mod images;
mod theme;
mod widgets;

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

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
    max_tick: Res<MaxTick>,
    show_stats: Res<ShowStats>,
    diagnostics: Res<DiagnosticsStore>,
    entities: Res<EntityMap>,
    window: Query<&Window>,
    images: Local<images::Images>,
) {
    let window = window.single();
    let width = window.resolution.width();
    let height = window.resolution.height();

    theme::set_theme(contexts.ctx_mut());

    let timeline_icons = TimelineIcons {
        jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
        jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
        frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
        frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
        play: contexts.add_image(images.icon_play.clone_weak()),
        pause: contexts.add_image(images.icon_pause.clone_weak()),
        handle: contexts.add_image(images.icon_scrub.clone_weak()),
    };

    if show_stats.0 {
        egui::Window::new("stats")
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::default())
            .fixed_pos(egui::pos2(32.0, 32.0))
            .show(contexts.ctx_mut(), |ui| {
                let fps_str = diagnostics
                    .get(FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                    .map_or(" N/A".to_string(), |value| format!("{value:>4.0}"));

                ui.add(Label::new(
                    RichText::new(format!("FPS: {fps_str}")).color(Color32::WHITE),
                ));
            });
    }

    egui::TopBottomPanel::top("titlebar")
        .frame(egui::Frame {
            fill: colors::INTERFACE_BACKGROUND_BLACK,
            stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
            ..Default::default()
        })
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| ui.set_height(48.0));

    // NOTE(temp fix): Hide panels until simulation is loaded
    if entities.len() > 0 {
        if width > height {
            egui::SidePanel::new(egui::panel::Side::Left, "outline_side")
                .resizable(true)
                .frame(egui::Frame {
                    fill: colors::STONE_950,
                    stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                    inner_margin: Margin::same(4.0),
                    ..Default::default()
                })
                .default_width(200.0)
                .show(contexts.ctx_mut(), |ui| {
                    entity_list(ui, entities);
                    // ui.allocate_space(ui.available_size());
                });
        } else {
            egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "outline_bottom")
                .resizable(true)
                .frame(egui::Frame {
                    fill: colors::STONE_950,
                    stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                    inner_margin: Margin::same(4.0),
                    ..Default::default()
                })
                .default_height(200.0)
                .show(contexts.ctx_mut(), |ui| {
                    entity_list(ui, entities);
                    ui.allocate_space(ui.available_size());
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
            timeline_area(
                ui,
                &mut paused,
                &max_tick,
                &mut tick,
                &mut event,
                timeline_icons,
            );
        });
}

fn entity_list(ui: &mut egui::Ui, entities: Res<EntityMap>) -> egui::Response {
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                egui::Frame::none()
                    .inner_margin(Margin::symmetric(16.0, 16.0))
                    .show(ui, |ui| {
                        ui.add(
                            Label::new(RichText::new("ENTITIES").color(Color32::WHITE)).wrap(false),
                        );
                    });

                ui.separator();

                for entity_id in entities.0.values() {
                    // TODO: Replace with custom `toggle` widget
                    egui::Frame::none()
                        .inner_margin(Margin::symmetric(32.0, 8.0))
                        .outer_margin(2.0)
                        .fill(colors::STONE_950)
                        .show(ui, |ui| {
                            ui.add(
                                Label::new(
                                    RichText::new(format!("{entity_id:?}")).color(Color32::WHITE),
                                )
                                .wrap(false),
                            );
                        });
                }

                ui.allocate_space(ui.available_size());
            })
        })
        .inner
        .response
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
