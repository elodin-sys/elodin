use bevy::ecs::{
    system::{Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use impeller2::types::Timestamp;
use impeller2_bevy::{CurrentStreamId, PacketTx};
use impeller2_wkt::{
    CurrentTimestamp, EarliestTimestamp, LastUpdated, SetStreamState, SimulationTimeStep,
};

use crate::ui::{
    colors::{self, with_opacity},
    widgets::{button::EImageButton, time_label::time_label, WidgetSystem},
    Paused,
};

use super::TimelineIcons;

#[derive(SystemParam)]
pub struct TimelineControls<'w> {
    event: Res<'w, PacketTx>,
    paused: ResMut<'w, Paused>,
    tick: ResMut<'w, CurrentTimestamp>,
    max_tick: Res<'w, LastUpdated>,
    tick_time: Res<'w, SimulationTimeStep>,
    stream_id: Res<'w, CurrentStreamId>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
}

impl WidgetSystem for TimelineControls<'_> {
    type Args = TimelineIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let icons = args;
        let TimelineControls {
            event,
            mut paused,
            mut tick,
            max_tick,
            tick_time,
            stream_id,
            earliest_timestamp,
        } = state.get_mut(world);

        let mut tick_changed = false;

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.horizontal_centered(|col_ui| {
                        let btn_scale = 1.4;
                        col_ui.spacing_mut().item_spacing.x = 8.0;

                        let jump_to_start_btn = col_ui.add(
                            EImageButton::new(icons.jump_to_start).scale(btn_scale, btn_scale),
                        );

                        if jump_to_start_btn.clicked() {
                            tick.0 = earliest_timestamp.0;
                            tick_changed = true;
                        }

                        let frame_back_btn = col_ui
                            .add(EImageButton::new(icons.frame_back).scale(btn_scale, btn_scale));

                        if frame_back_btn.clicked() && tick.0 > earliest_timestamp.0 {
                            tick.0 .0 -= (hifitime::Duration::from_seconds(tick_time.0)
                                .total_nanoseconds()
                                / 1000) as i64;
                            tick_changed = true;
                        }

                        if paused.0 {
                            let play_btn = col_ui
                                .add(EImageButton::new(icons.play).scale(btn_scale, btn_scale));

                            if play_btn.clicked() {
                                paused.0 = false;
                            }
                        } else {
                            let pause_btn = col_ui
                                .add(EImageButton::new(icons.pause).scale(btn_scale, btn_scale));

                            if pause_btn.clicked() {
                                paused.0 = true;
                            }
                        }

                        let frame_forward_btn = col_ui.add(
                            EImageButton::new(icons.frame_forward).scale(btn_scale, btn_scale),
                        );

                        if frame_forward_btn.clicked() && tick.0 < max_tick.0 {
                            tick.0 .0 += (hifitime::Duration::from_seconds(tick_time.0)
                                .total_nanoseconds()
                                / 1000) as i64;

                            tick_changed = true;
                        }

                        let jump_to_end_btn = col_ui
                            .add(EImageButton::new(icons.jump_to_end).scale(btn_scale, btn_scale));

                        if jump_to_end_btn.clicked() {
                            tick.0 = Timestamp(max_tick.0 .0 - 1);
                            tick_changed = true;
                        }
                    });

                    ui.allocate_ui_with_layout(
                        ui.available_size(),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |col_ui| {
                            egui::Frame::NONE
                                .inner_margin(egui::Margin::symmetric(8, 0))
                                .show(col_ui, |ui| {
                                    ui.spacing_mut().item_spacing.x = 12.0;

                                    // TIME

                                    let time: hifitime::Epoch = tick.0.into();
                                    ui.add(time_label(time));
                                    // let time_value = egui::RichText::new(time.to_string())
                                    //     .color(colors::PRIMARY_CREAME);

                                    // ui.add(egui::Label::new(time_value).selectable(false));

                                    let time_label = egui::RichText::new("TIME")
                                        .color(with_opacity(colors::PRIMARY_CREAME, 0.4));

                                    ui.add(egui::Label::new(time_label).selectable(false));

                                    ui.add_space(24.0);
                                });
                        },
                    );
                });
            });

        if tick_changed {
            event.send_msg(SetStreamState::rewind(**stream_id, tick.0));
        }
    }
}
