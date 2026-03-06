use bevy::ecs::{
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::*;
use bevy_egui::egui;
use egui::{Ui, load::SizedTexture};
use impeller2::types::Timestamp;
use impeller2_bevy::CurrentStreamId;
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp, LastUpdated, SimulationTimeStep};
use std::convert::TryFrom;
use std::time::Duration;
use std::time::Instant;

use crate::{
    TimeRangeBehavior,
    ui::{
        FocusedWindow, Paused, SelectedObject,
        button::EImageButton,
        colors::{ColorExt, EColor, get_scheme},
        theme::configure_combo_box,
        tiles::WindowState,
        time_label::time_label,
        widgets::WidgetSystem,
    },
};

use super::{
    AutoFollowLatestState, LatestFollow, PlaybackSpeed, StreamTickOrigin, TimelineIcons,
    TimelineSettings,
};

pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<TimelineStepButtons>();
}

#[derive(SystemParam)]
pub struct TimelineControls<'w, 's> {
    paused: ResMut<'w, Paused>,
    tick: ResMut<'w, CurrentTimestamp>,
    max_tick: Res<'w, LastUpdated>,
    tick_time: Res<'w, SimulationTimeStep>,
    playback_speed: Res<'w, PlaybackSpeed>,
    stream_id: Res<'w, CurrentStreamId>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    behavior: ResMut<'w, TimeRangeBehavior>,
    tick_origin: ResMut<'w, StreamTickOrigin>,
    step_buttons: ResMut<'w, TimelineStepButtons>,
    latest_follow: ResMut<'w, LatestFollow>,
    auto_follow_latest_state: ResMut<'w, AutoFollowLatestState>,
    timeline_settings: Res<'w, TimelineSettings>,
    focused_window: Res<'w, FocusedWindow>,
    primary_windows: Query<'w, 's, Entity, With<bevy::window::PrimaryWindow>>,
    window_states: Query<'w, 's, &'static mut WindowState>,
    replay_mode: Option<Res<'w, crate::ReplayMode>>,
}

#[derive(Default, Debug, Resource)]
struct TimelineStepButtons {
    back: Option<Instant>,
    forward: Option<Instant>,
}

impl WidgetSystem for TimelineControls<'_, '_> {
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
            mut paused,
            mut tick,
            max_tick,
            tick_time,
            playback_speed,
            stream_id,
            earliest_timestamp,
            mut behavior,
            mut tick_origin,
            mut step_buttons,
            mut latest_follow,
            mut auto_follow_latest_state,
            timeline_settings,
            focused_window,
            primary_windows,
            mut window_states,
            replay_mode,
        } = state.get_mut(world);

        tick_origin.observe_stream(**stream_id);
        tick_origin.observe_tick(tick.0, earliest_timestamp.0);

        let tick_step_duration = hifitime::Duration::from_seconds(tick_time.0);
        let tick_step_micros_i128 = tick_step_duration.total_nanoseconds() / 1000;
        let tick_step_micros = i64::try_from(tick_step_micros_i128).unwrap_or(0);
        let played_color = timeline_settings.played_color.into_color32();
        ui.set_height(50.0);
        let typical_mouse_click = Duration::from_millis(85);
        let wait_before_advancing = typical_mouse_click * 2;

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(ui.available_width(), 37.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let btn_scale = 1.4;
                            ui.spacing_mut().item_spacing.x = 8.0;

                            let jump_to_start_btn = ui.add(
                                EImageButton::new(icons.jump_to_start).scale(btn_scale, btn_scale),
                            );

                            if jump_to_start_btn.clicked() {
                                auto_follow_latest_state.cancel();
                                latest_follow.0 = false;
                                tick.0 = earliest_timestamp.0;
                                tick_origin.request_rebase();
                            }

                            let frame_back_btn = ui.add(
                                EImageButton::new(icons.frame_back).scale(btn_scale, btn_scale),
                            );

                            if frame_back_btn.is_pointer_button_down_on()
                                && tick.0 > earliest_timestamp.0
                                && tick_step_micros > 0
                            {
                                auto_follow_latest_state.cancel();
                                latest_follow.0 = false;
                                let mut first = false;
                                let down = step_buttons.back.get_or_insert_with(|| {
                                    first = true;
                                    Instant::now()
                                });

                                if first || down.elapsed() > wait_before_advancing {
                                    tick.0.0 -= tick_step_micros;
                                    if tick.0 <= earliest_timestamp.0 {
                                        tick_origin.request_rebase();
                                    }
                                }
                            } else {
                                let _ = step_buttons.back.take();
                            }

                            if paused.0 {
                                let play_btn = ui
                                    .add(EImageButton::new(icons.play).scale(btn_scale, btn_scale));

                                if play_btn.clicked() {
                                    auto_follow_latest_state.cancel();
                                    paused.0 = false;
                                }
                            } else {
                                let pause_btn = ui.add(
                                    EImageButton::new(icons.pause).scale(btn_scale, btn_scale),
                                );

                                if pause_btn.clicked() {
                                    auto_follow_latest_state.cancel();
                                    paused.0 = true;
                                    latest_follow.0 = false;
                                }
                            }

                            let frame_forward_btn = ui.add(
                                EImageButton::new(icons.frame_forward).scale(btn_scale, btn_scale),
                            );

                            if frame_forward_btn.is_pointer_button_down_on()
                                && tick.0 < max_tick.0
                                && tick_step_micros > 0
                            {
                                auto_follow_latest_state.cancel();
                                latest_follow.0 = false;
                                let mut first = false;
                                let down = step_buttons.forward.get_or_insert_with(|| {
                                    first = true;
                                    Instant::now()
                                });

                                if first || down.elapsed() > wait_before_advancing {
                                    tick.0.0 += tick_step_micros;
                                }
                            } else {
                                let _ = step_buttons.forward.take();
                            }

                            let jump_to_end_btn = ui.add(
                                EImageButton::new(icons.jump_to_end).scale(btn_scale, btn_scale),
                            );

                            if jump_to_end_btn.clicked() {
                                auto_follow_latest_state.cancel();
                                tick.0 = max_tick.0;
                                paused.0 = false;
                                latest_follow.0 = replay_mode.is_none();
                            }
                        },
                    );

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

                                    configure_combo_box(ui.style_mut());

                                    let ui_func = time_range_window(
                                        &mut behavior,
                                        earliest_timestamp.0,
                                        max_tick.0,
                                    );

                                    egui::Popup::from_toggle_button_response(&res)
                                        .layout(egui::Layout::top_down_justified(egui::Align::LEFT))
                                        .close_behavior(
                                            egui::PopupCloseBehavior::CloseOnClickOutside,
                                        )
                                        .id(popup_id)
                                        .align(egui::RectAlign::TOP_START)
                                        .width(res.rect.width())
                                        .show(ui_func);

                                    let settings_response = ui
                                        .add(EImageButton::new(icons.setting).scale(1.2, 1.2))
                                        .on_hover_text("Timeline settings");
                                    if settings_response.clicked() {
                                        let target_window = focused_window
                                            .0
                                            .or_else(|| primary_windows.iter().next());
                                        if let Some(target_window) = target_window
                                            && let Ok(mut window_state) =
                                                window_states.get_mut(target_window)
                                        {
                                            window_state.ui_state.selected_object =
                                                SelectedObject::Timeline;
                                            window_state.ui_state.right_sidebar_visible = true;
                                        }
                                    }

                                    // TIME

                                    let time: hifitime::Epoch = tick.0.into();
                                    ui.add(time_label(time));

                                    let time_label = egui::RichText::new("TIME")
                                        .color(get_scheme().text_secondary);
                                    ui.add_space(8.0);

                                    ui.add(egui::Label::new(time_label).selectable(false));

                                    ui.add_space(24.0);

                                    let origin_timestamp = tick_origin.origin(earliest_timestamp.0);
                                    let tick_text = if tick_step_micros_i128 <= 0 {
                                        "-".to_owned()
                                    } else {
                                        let delta =
                                            i128::from(tick.0.0) - i128::from(origin_timestamp.0);
                                        let clamped_delta = delta.max(0);
                                        (clamped_delta / tick_step_micros_i128).to_string()
                                    };

                                    let tick_value = egui::RichText::new(tick_text)
                                        .color(get_scheme().text_primary);
                                    ui.add(
                                        egui::Label::new(tick_value)
                                            .selectable(false)
                                            .halign(egui::Align::BOTTOM),
                                    );

                                    let tick_label = egui::RichText::new("TICK")
                                        .color(get_scheme().text_secondary);
                                    ui.add_space(8.0);

                                    ui.add(egui::Label::new(tick_label).selectable(false));

                                    ui.add_space(24.0);

                                    let speed_text = egui::RichText::new(format_playback_speed(
                                        playback_speed.0,
                                    ))
                                    .color(get_scheme().text_primary);
                                    ui.add(
                                        egui::Label::new(speed_text)
                                            .selectable(false)
                                            .halign(egui::Align::BOTTOM),
                                    );

                                    let speed_label = egui::RichText::new("SPEED")
                                        .color(get_scheme().text_secondary);
                                    ui.add_space(8.0);

                                    ui.add(egui::Label::new(speed_label).selectable(false));

                                    ui.add_space(16.0);

                                    let latest_enabled = replay_mode.is_none();
                                    let lag_micros = max_tick.0.0.saturating_sub(tick.0.0);
                                    let latest_response = live_follow_button(
                                        ui,
                                        latest_enabled,
                                        latest_follow.0,
                                        lag_micros,
                                        played_color,
                                    );
                                    if latest_enabled && latest_response.clicked() {
                                        auto_follow_latest_state.cancel();
                                        latest_follow.0 = !latest_follow.0;
                                    }

                                    if latest_follow.0 {
                                        paused.0 = false;
                                        tick.0 = max_tick.0;
                                    }
                                });
                        },
                    );
                });
            });
    }
}

fn format_playback_speed(speed: f64) -> String {
    if !speed.is_finite() || speed < 0.0 {
        return "-".to_string();
    }

    let mut value = format!("{speed:.3}");
    while value.ends_with('0') {
        value.pop();
    }
    if value.ends_with('.') {
        value.pop();
    }
    format!("{value}x")
}

fn format_lag_counter(micros: i64) -> String {
    if micros <= 0 {
        return "0ms".to_owned();
    }

    if micros >= 1_000_000 {
        return format!("+{:.1}s", micros as f64 / 1_000_000.0);
    }

    if micros >= 1_000 {
        return format!("+{}ms", micros / 1_000);
    }

    format!("+{micros}us")
}

fn live_follow_button(
    ui: &mut egui::Ui,
    enabled: bool,
    following: bool,
    lag_micros: i64,
    played_color: egui::Color32,
) -> egui::Response {
    let is_delayed = lag_micros > 0;
    let live_label = "LIVE";
    let counter_label = format_lag_counter(lag_micros);

    let scheme = get_scheme();
    let (live_text_color, counter_text_color, fill_color, stroke_color, dot_color) = if !enabled {
        (
            scheme.text_tertiary,
            scheme.text_tertiary.opacity(0.7),
            scheme.bg_primary,
            scheme.border_primary.opacity(0.25),
            scheme.text_tertiary.opacity(0.4),
        )
    } else if following {
        (
            played_color,
            played_color.opacity(0.9),
            scheme.bg_secondary.opacity(0.7),
            played_color.opacity(0.45),
            played_color,
        )
    } else if is_delayed {
        (
            scheme.text_primary,
            scheme.text_primary,
            scheme.bg_secondary.opacity(0.7),
            scheme.border_primary.opacity(0.75),
            scheme.text_secondary,
        )
    } else {
        (
            scheme.text_secondary,
            scheme.text_secondary,
            scheme.bg_secondary.opacity(0.6),
            scheme.border_primary.opacity(0.55),
            scheme.text_tertiary.opacity(0.8),
        )
    };

    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let live_galley =
        ui.painter()
            .layout_no_wrap(live_label.to_owned(), font_id.clone(), live_text_color);
    let counter_galley =
        ui.painter()
            .layout_no_wrap(counter_label.clone(), font_id.clone(), counter_text_color);
    let dot_radius = 3.0;
    let height = (live_galley.size().y.max(counter_galley.size().y) + 8.0).max(22.0);
    let live_fixed_width = ui
        .painter()
        .layout_no_wrap("LIVE".to_owned(), font_id.clone(), live_text_color)
        .size()
        .x;
    let counter_fixed_width = ["0ms", "+9999ms", "+9999us", "+999.9s"]
        .into_iter()
        .map(|sample| {
            ui.painter()
                .layout_no_wrap(sample.to_owned(), font_id.clone(), counter_text_color)
                .size()
                .x
        })
        .fold(counter_galley.size().x, f32::max);
    let width = 32.0 + live_fixed_width + 10.0 + counter_fixed_width;

    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), sense);

    if ui.is_rect_visible(rect) {
        let hover = enabled && response.hovered();
        let pressed = enabled && response.is_pointer_button_down_on();
        let fill = if pressed {
            fill_color.opacity(0.85)
        } else if hover {
            fill_color.opacity(0.92)
        } else {
            fill_color
        };

        ui.painter().rect(
            rect,
            egui::CornerRadius::same(10),
            fill,
            egui::Stroke::new(1.0, stroke_color),
            egui::StrokeKind::Middle,
        );

        let dot_center = egui::pos2(rect.left() + 10.0, rect.center().y);
        ui.painter()
            .circle_filled(dot_center, dot_radius, dot_color);
        let live_pos = egui::pos2(rect.left() + 18.0, rect.center().y);
        ui.painter().text(
            live_pos,
            egui::Align2::LEFT_CENTER,
            live_label,
            font_id.clone(),
            live_text_color,
        );

        let separator_x = live_pos.x + live_fixed_width + 5.0;
        ui.painter().line_segment(
            [
                egui::pos2(separator_x, rect.top() + 4.0),
                egui::pos2(separator_x, rect.bottom() - 4.0),
            ],
            egui::Stroke::new(1.0, stroke_color.opacity(0.65)),
        );
        ui.painter().text(
            egui::pos2(separator_x + 6.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            counter_label,
            font_id,
            counter_text_color,
        );
    }

    let response = if enabled {
        response.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        response
    };

    let hover_text = if !enabled {
        "Disabled in replay mode"
    } else if following {
        "Following latest data"
    } else {
        "Click to jump to latest and keep following"
    };

    response.on_hover_text(hover_text)
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
                        get_scheme().bg_secondary.opacity(0.5)
                    } else if response.hovered() {
                        get_scheme().bg_secondary.opacity(0.75)
                    } else {
                        get_scheme().bg_secondary
                    },
                );

                ui.painter().text(
                    ui.max_rect().left_center() + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    behavior_string,
                    font_id,
                    get_scheme().text_primary,
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
        TimeRangeBehavior::LAST_5S,
        TimeRangeBehavior::LAST_15S,
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
                            get_scheme().bg_primary.opacity(0.75)
                        } else {
                            egui::Color32::TRANSPARENT
                        },
                    );
                    painter.text(
                        egui::Pos2::new(response.rect.min.x + 8.0, response.rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        range.to_string(),
                        font_id.clone(),
                        get_scheme().text_primary,
                    );

                    if response.clicked() {
                        *behavior = *range;
                    }
                }
            },
        );
    }
}
