use bevy::ecs::{
    system::{Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use egui::{Ui, load::SizedTexture};
use impeller2::types::Timestamp;
use impeller2_bevy::{CurrentStreamId, PacketTx};
use impeller2_wkt::{
    CurrentTimestamp, EarliestTimestamp, LastUpdated, SetStreamState, SimulationTimeStep,
};

use crate::{
    TimeRangeBehavior,
    ui::{
        Paused,
        colors::{self, ColorExt, with_opacity},
        widgets::{WidgetSystem, button::EImageButton, time_label::time_label},
    },
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
    behavior: ResMut<'w, TimeRangeBehavior>,
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
            mut behavior,
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
                            tick.0.0 -= (hifitime::Duration::from_seconds(tick_time.0)
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
                            tick.0.0 += (hifitime::Duration::from_seconds(tick_time.0)
                                .total_nanoseconds()
                                / 1000) as i64;

                            tick_changed = true;
                        }

                        let jump_to_end_btn = col_ui
                            .add(EImageButton::new(icons.jump_to_end).scale(btn_scale, btn_scale));

                        if jump_to_end_btn.clicked() {
                            tick.0 = Timestamp(max_tick.0.0 - 1);
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
                                    let popup_id = ui.make_persistent_id("time_selector");
                                    let res = ui.add(time_range_selector_button(
                                        icons.vertical_chevrons,
                                        &mut behavior,
                                    ));

                                    if res.clicked() {
                                        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
                                    }
                                    ui.style_mut().visuals.window_fill = colors::SURFACE_SECONDARY;
                                    ui.style_mut().visuals.window_stroke =
                                        egui::Stroke::new(1., colors::PRIMARY_CREAME_5);
                                    ui.style_mut().spacing.menu_margin =
                                        egui::Margin::symmetric(0, 0);
                                    egui::popup::popup_above_or_below_widget(
                                        ui,
                                        popup_id,
                                        &res,
                                        egui::containers::AboveOrBelow::Above,
                                        egui::popup::PopupCloseBehavior::CloseOnClickOutside,
                                        time_range_window(
                                            &mut behavior,
                                            earliest_timestamp.0,
                                            max_tick.0,
                                        ),
                                    );

                                    // TIME

                                    let time: hifitime::Epoch = tick.0.into();
                                    ui.add(time_label(time));

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

fn time_range_selector_button(
    icon: egui::TextureId,
    behavior: &mut TimeRangeBehavior,
) -> impl FnOnce(&mut Ui) -> egui::Response + '_ {
    move |ui| {
        let behavior_string = behavior.to_string();
        let width = behavior_string.len() as f32 * 7.5;
        ui.allocate_ui_with_layout(
            egui::vec2(width + 50.0, 34.0),
            egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
            |ui| {
                let font_id = egui::TextStyle::Button.resolve(ui.style());
                let response =
                    ui.allocate_rect(ui.max_rect(), egui::Sense::CLICK | egui::Sense::HOVER);
                ui.painter().rect_filled(
                    ui.max_rect(),
                    egui::CornerRadius::ZERO,
                    if response.is_pointer_button_down_on() {
                        colors::SURFACE_SECONDARY.opacity(0.5)
                    } else if response.hovered() {
                        colors::SURFACE_SECONDARY.opacity(0.75)
                    } else {
                        colors::SURFACE_SECONDARY
                    },
                );

                ui.painter().text(
                    ui.max_rect().left_center() + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    behavior_string,
                    font_id,
                    colors::PRIMARY_CREAME,
                );
                egui::Image::new(SizedTexture::new(icon, egui::vec2(18., 18.))).paint_at(
                    ui,
                    egui::Rect::from_center_size(
                        egui::Pos2::new(ui.max_rect().max.x - 17., ui.max_rect().center().y),
                        egui::vec2(18., 18.),
                    ),
                );
                response
            },
        )
        .inner
    }
}

fn time_range_window(
    behavior: &mut TimeRangeBehavior,
    earliest: Timestamp,
    latest: Timestamp,
) -> impl FnOnce(&mut egui::Ui) + '_ {
    const VISIBLE_RANGES: &[TimeRangeBehavior] = &[
        TimeRangeBehavior::FULL,
        TimeRangeBehavior::LAST_30S,
        TimeRangeBehavior::LAST_1M,
        TimeRangeBehavior::LAST_5M,
        TimeRangeBehavior::LAST_15M,
        TimeRangeBehavior::LAST_30M,
        TimeRangeBehavior::LAST_1H,
        TimeRangeBehavior::LAST_6H,
        TimeRangeBehavior::LAST_12H,
        TimeRangeBehavior::LAST_24H,
    ];
    move |ui| {
        let size = egui::vec2(225., 370.);
        ui.allocate_ui_with_layout(
            size,
            egui::Layout::default().with_cross_justify(true),
            |ui| {
                let font_id = egui::TextStyle::Button.resolve(ui.style());
                ui.add_space(16.0);
                for range in VISIBLE_RANGES
                    .iter()
                    .filter(|b| b.is_subset(earliest, latest))
                {
                    let (response, painter) = ui.allocate_painter(
                        egui::vec2(215., 34.),
                        egui::Sense::HOVER | egui::Sense::CLICK,
                    );

                    painter.rect_filled(
                        response.rect,
                        egui::CornerRadius::ZERO,
                        if response.hovered() {
                            colors::PRIMARY_SMOKE.opacity(0.75)
                        } else {
                            colors::TRANSPARENT
                        },
                    );
                    painter.text(
                        egui::Pos2::new(response.rect.min.x + 24.0, response.rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        range.to_string(),
                        font_id.clone(),
                        colors::BONE_DEFAULT,
                    );

                    if response.clicked() {
                        *behavior = *range;
                    }
                }
                ui.add_space(16.0);
            },
        );
    }
}
