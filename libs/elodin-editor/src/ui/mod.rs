use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText, Rounding, Separator},
    EguiContexts,
};
use elodin_conduit::{
    bevy::{MaxTick, Tick},
    ControlMsg,
};

use self::widgets::{button::ImageButton, timeline::Timeline};

mod colors;
pub mod images;
mod theme;
mod widgets;

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

pub fn shortcuts(mut show_stats: ResMut<ShowStats>, kbd: Res<Input<KeyCode>>) {
    if kbd.just_pressed(KeyCode::F12) {
        show_stats.0 = !show_stats.0;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    mut contexts: EguiContexts,
    mut paused: ResMut<Paused>,
    max_tick: Res<MaxTick>,
    mut tick: ResMut<Tick>,
    show_stats: Res<ShowStats>,
    // picked: Query<(EntityQuery, &Picked, Entity)>,
    diagnostics: Res<DiagnosticsStore>,
    window: Query<&Window>,
    images: Local<images::Images>,
    mut event: EventWriter<ControlMsg>,
) {
    let window = window.single();
    let width = window.resolution.width();
    let height = window.resolution.height();

    theme::set_theme(contexts.ctx_mut());

    let icon_play_id = contexts.add_image(images.icon_play.clone_weak());
    let icon_pause_id = contexts.add_image(images.icon_pause.clone_weak());
    let icon_scrub_id = contexts.add_image(images.icon_scrub.clone_weak());
    let icon_skip_next_id = contexts.add_image(images.icon_skip_next.clone_weak());
    let icon_skip_prev_id = contexts.add_image(images.icon_skip_prev.clone_weak());

    // egui::Window::new("picked components")
    //     .title_bar(false)
    //     .resizable(false)
    //     // .fixed_pos(egui::pos2(0.0, 0.0))
    //     .show(contexts.ctx_mut(), |ui| {
    //         picked
    //             .iter()
    //             .filter(|(_, picked, _)| picked.0)
    //             .for_each(|(entity, _, e)| {
    //                 ui.collapsing(format!("entity {:?}", e), |ui| {
    //                     vec3_component(ui, "pos (m/s)", &entity.world_pos.0.pos);
    //                     vec3_component(ui, "vel (m/s)", &entity.world_vel.0.vel);
    //                     let euler_angles = vec_from_tuple(entity.world_pos.0.att.euler_angles());
    //                     vec3_component(ui, "euler angle (rad)", &euler_angles);
    //                     vec3_component(ui, "ang vel (m/s)", &entity.world_vel.0.vel);
    //                 });
    //             })
    //     });

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

    egui::Window::new("timeline")
        .title_bar(false)
        .resizable(false)
        .frame(egui::Frame {
            rounding: Rounding::same(8.0),
            fill: colors::SURFACE_SECONDARY,
            ..Default::default()
        })
        .fixed_size(egui::vec2(500.0, 50.0))
        .fixed_pos(egui::pos2(width / 2.0 - 250.0, height - 160.0))
        .show(contexts.ctx_mut(), |ui| {
            let mut tick_changed = false;
            egui::Frame::none()
                .inner_margin(Margin::symmetric(16.0, 12.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Control Buttons Scope
                        ui.scope(|ui| {
                            ui.style_mut().spacing.item_spacing.x = 8.;

                            let skip_prev_btn =
                                ui.add(ImageButton::new(icon_skip_prev_id).scale(1.4, 1.4));

                            if skip_prev_btn.clicked() && tick.0 > 0 {
                                tick.0 = 0;
                                tick_changed = true;
                            }

                            if paused.0 {
                                let play_btn =
                                    ui.add(ImageButton::new(icon_play_id).scale(1.4, 1.4));

                                if play_btn.clicked() {
                                    paused.0 = false;
                                }
                            } else {
                                let pause_btn =
                                    ui.add(ImageButton::new(icon_pause_id).scale(1.4, 1.4));

                                if pause_btn.clicked() {
                                    paused.0 = true;
                                }
                            }

                            let skip_next_btn =
                                ui.add(ImageButton::new(icon_skip_next_id).scale(1.4, 1.4));

                            if skip_next_btn.clicked() && tick.0 < max_tick.0 {
                                tick.0 = max_tick.0 - 1;
                                tick_changed = true;
                            }
                        });

                        ui.add_space(ui.available_width());

                        ui.add(Label::new(
                            RichText::new(format!("{:0>5}", tick.0)).color(Color32::WHITE),
                        ));
                    });
                });

            ui.add(Separator::default().spacing(0.0));

            let max_count = max_tick.0;
            let frames_per_second = 60.0;
            ui.horizontal(|ui| {
                let response = ui.add(
                    Timeline::new(&mut tick.bypass_change_detection().0, 0..=max_count)
                        .width(ui.available_width())
                        .height(32.0)
                        .handle_image_id(&icon_scrub_id)
                        .handle_aspect_ratio(12.0 / 30.0)
                        .segments(8)
                        .time(max_count as f64 / frames_per_second),
                );
                tick_changed |= response.changed();
            });
            if tick_changed {
                event.send(ControlMsg::Rewind(tick.0));
            }
        });
}

// fn vec_from_tuple(tuple: (f64, f64, f64)) -> Vector3<f64> {
//     Vector3::new(tuple.0, tuple.1, tuple.2)
// }

// fn vec3_component(ui: &mut Ui, label: &str, vec3: &Vector3<f64>) {
//     ui.horizontal(|ui| {
//         ui.label(label);
//         let x = format!("{:+.5}", vec3.x);
//         let y = format!("{:+.5}", vec3.y);
//         let z = format!("{:+.5}", vec3.z);
//         ui.add_sized(
//             egui::vec2(70., 16.),
//             egui::TextEdit::singleline(&mut x.as_str()),
//         );
//         ui.add_sized(
//             egui::vec2(70., 16.),
//             egui::TextEdit::singleline(&mut y.as_str()),
//         );
//         ui.add_sized(
//             egui::vec2(70., 16.),
//             egui::TextEdit::singleline(&mut z.as_str()),
//         );
//     });
// }
