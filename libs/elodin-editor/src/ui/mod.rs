use bevy::prelude::*;
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText, Rounding, Separator},
    EguiContexts,
};
use elodin_conduit::well_known::SimState;

use self::widgets::{button::ImageButton, timeline::Timeline};

mod colors;
pub mod images;
mod theme;
mod widgets;

#[derive(Resource)]
pub struct UiState {
    history_index: usize,
    history_count: usize,
}

// NOTE: Temporary local state to test the UI
impl Default for UiState {
    fn default() -> UiState {
        UiState {
            history_index: 1025,
            history_count: 140 * 30, // 2min20sec at 30fps
        }
    }
}

pub fn render(
    mut contexts: EguiContexts,
    mut sim_state: ResMut<SimState>,
    mut ui_state: ResMut<UiState>,
    // picked: Query<(EntityQuery, &Picked, Entity)>,
    window: Query<&Window>,
    images: Local<images::Images>,
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
            egui::Frame::none()
                .inner_margin(Margin::symmetric(16.0, 12.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Control Buttons Scope
                        ui.scope(|ui| {
                            ui.style_mut().spacing.item_spacing.x = 8.;

                            let skip_prev_btn =
                                ui.add(ImageButton::new(icon_skip_prev_id).scale(1.4, 1.4));

                            if skip_prev_btn.clicked() && ui_state.history_index > 0 {
                                ui_state.history_index -= 1;
                            }

                            if sim_state.paused {
                                let play_btn =
                                    ui.add(ImageButton::new(icon_play_id).scale(1.4, 1.4));

                                if play_btn.clicked() {
                                    sim_state.paused = false;
                                }
                            } else {
                                let pause_btn =
                                    ui.add(ImageButton::new(icon_pause_id).scale(1.4, 1.4));

                                if pause_btn.clicked() {
                                    sim_state.paused = true;
                                }
                            }

                            let skip_next_btn =
                                ui.add(ImageButton::new(icon_skip_next_id).scale(1.4, 1.4));

                            if skip_next_btn.clicked()
                                && ui_state.history_index < ui_state.history_count
                            {
                                ui_state.history_index += 1;
                            }
                        });

                        ui.add_space(ui.available_width());

                        ui.add(Label::new(
                            RichText::new(format!("{:0>5}", ui_state.history_index))
                                .color(Color32::WHITE),
                        ));
                    });
                });

            ui.add(Separator::default().spacing(0.0));

            let max_count = ui_state.history_count;
            let frames_per_second = 60.0;
            ui.horizontal(|ui| {
                ui.add(
                    Timeline::new(&mut ui_state.history_index, 0..=max_count)
                        .width(ui.available_width())
                        .height(32.0)
                        .handle_image_id(&icon_scrub_id)
                        .handle_aspect_ratio(12.0 / 30.0)
                        .segments(8)
                        .time(max_count as f64 / frames_per_second),
                );
            });
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
