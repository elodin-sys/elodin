use bevy::ecs::{
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::*;
use bevy_egui::egui;
use egui::{Ui, load::SizedTexture};
use impeller2::types::Timestamp;
use impeller2_bevy::{
    CurrentStreamId, SeriesFetchPriority, TelemetryCache, next_subscribed_sample,
    prev_subscribed_sample,
};
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
    playback::{self, PlaybackLoop, PlaybackRegion, set_playback_speed},
};
use crate::ui::widgets::SystemStateExt;

pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<TimelineStepButtons>();
}

#[derive(SystemParam)]
pub struct TimelineControls<'w, 's> {
    paused: ResMut<'w, Paused>,
    tick: ResMut<'w, CurrentTimestamp>,
    max_tick: Res<'w, LastUpdated>,
    tick_time: Res<'w, SimulationTimeStep>,
    series: Res<'w, TelemetryCache>,
    series_priority: Res<'w, SeriesFetchPriority>,
    playback_speed: ResMut<'w, PlaybackSpeed>,
    playback_loop: ResMut<'w, PlaybackLoop>,
    playback_region: Res<'w, PlaybackRegion>,
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
            series,
            series_priority,
            mut playback_speed,
            mut playback_loop,
            playback_region,
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
        } = state.params_mut(world);

        tick_origin.observe_stream(**stream_id);
        tick_origin.observe_tick(tick.0, earliest_timestamp.0);

        let tick_step_duration = hifitime::Duration::from_seconds(tick_time.0);
        let tick_step_micros_i128 = tick_step_duration.total_nanoseconds() / 1000;
        let tick_step_micros = i64::try_from(tick_step_micros_i128).unwrap_or(0);
        // Prefer the real sample boundaries of the displayed series: a nominal
        // step lands between samples on irregular data, and on a recording that
        // declares no rate it is the only thing that can step at all.
        let nominal_step = |delta: i64| (tick_step_micros > 0).then(|| Timestamp(tick.0.0 + delta));
        let step_forward_to = next_subscribed_sample(&series, &series_priority, tick.0)
            .or_else(|| nominal_step(tick_step_micros));
        let step_back_to = prev_subscribed_sample(&series, &series_priority, tick.0)
            .or_else(|| nominal_step(-tick_step_micros));
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

                            let jump_to_start_btn = ui
                                .add(
                                    EImageButton::new(icons.jump_to_start)
                                        .scale(btn_scale, btn_scale),
                                )
                                .on_hover_text("Jump to start");

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
                                && let Some(target) = step_back_to
                            {
                                auto_follow_latest_state.cancel();
                                latest_follow.0 = false;
                                let mut first = false;
                                let down = step_buttons.back.get_or_insert_with(|| {
                                    first = true;
                                    Instant::now()
                                });

                                if first || down.elapsed() > wait_before_advancing {
                                    tick.0 = target;
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
                                && let Some(target) = step_forward_to
                            {
                                auto_follow_latest_state.cancel();
                                latest_follow.0 = false;
                                let mut first = false;
                                let down = step_buttons.forward.get_or_insert_with(|| {
                                    first = true;
                                    Instant::now()
                                });

                                if first || down.elapsed() > wait_before_advancing {
                                    tick.0 = target;
                                }
                            } else {
                                let _ = step_buttons.forward.take();
                            }

                            let jump_to_end_btn = ui
                                .add(
                                    EImageButton::new(icons.jump_to_end)
                                        .scale(btn_scale, btn_scale),
                                )
                                .on_hover_text("Jump to end");

                            if jump_to_end_btn.clicked() {
                                auto_follow_latest_state.cancel();
                                tick.0 = max_tick.0;
                                paused.0 = false;
                                latest_follow.0 = replay_mode.is_none();
                                if latest_follow.0 {
                                    playback_loop.0 = false;
                                }
                            }

                            let loop_btn = ui
                                .add(
                                    EImageButton::new(icons.range_loop)
                                        .scale(btn_scale, btn_scale)
                                        .image_tint(if playback_loop.0 {
                                            played_color
                                        } else {
                                            get_scheme().icon_primary
                                        }),
                                )
                                .on_hover_text(if playback_region.0.is_some() {
                                    "Loop selected range"
                                } else {
                                    "Loop recording"
                                });
                            if loop_btn.clicked() {
                                playback_loop.0 = !playback_loop.0;
                                if playback_loop.0 {
                                    auto_follow_latest_state.cancel();
                                    latest_follow.0 = false;
                                }
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

                                    let playback_leads = !latest_follow.0 && !paused.0;
                                    speed_control(
                                        ui,
                                        &mut playback_speed,
                                        &mut latest_follow,
                                        &mut auto_follow_latest_state,
                                        playback_leads,
                                        played_color,
                                    );

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
                                        if latest_follow.0 {
                                            playback_loop.0 = false;
                                        }
                                    }
                                });
                        },
                    );
                });
            });
    }
}

/// Speed field drawn as the same pill as [`live_follow_button`]. Whichever of
/// the two drives the playhead is lit in `played_color`: LIVE while it
/// follows, the speed pill while playback runs on its own clock.
fn speed_control(
    ui: &mut egui::Ui,
    playback_speed: &mut PlaybackSpeed,
    latest_follow: &mut LatestFollow,
    auto_follow: &mut AutoFollowLatestState,
    playback_leads: bool,
    played_color: egui::Color32,
) {
    let scheme = get_scheme();
    let edit_id = ui.make_persistent_id("playback_speed_edit");
    let buffer_id = edit_id.with("buffer");
    let rect_id = edit_id.with("rect");
    let editing = ui.memory(|memory| memory.has_focus(edit_id));
    // The pill is drawn before its response exists, so hover reads last frame's rect.
    let hovered = ui
        .data(|data| data.get_temp::<egui::Rect>(rect_id))
        .is_some_and(|rect| ui.rect_contains_pointer(rect));

    let (text_color, fill_color, stroke_color) = if playback_leads {
        (
            played_color,
            scheme.bg_secondary.opacity(0.7),
            played_color.opacity(if hovered { 0.75 } else { 0.45 }),
        )
    } else {
        (
            scheme.text_primary,
            scheme.bg_secondary.opacity(0.6),
            scheme
                .border_primary
                .opacity(if hovered { 0.9 } else { 0.55 }),
        )
    };
    let stroke_color = if editing {
        played_color.opacity(0.9)
    } else {
        stroke_color
    };

    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let text_height = ui
        .painter()
        .layout_no_wrap("0".to_owned(), font_id.clone(), text_color)
        .size()
        .y;
    // Hug the value like LIVE hugs its label; the caret needs a sliver more.
    let painter = ui.painter().clone();
    let field_width = |text: &str| {
        let shown = if text.is_empty() {
            format_speed_value(playback_speed.0)
        } else {
            text.to_owned()
        };
        painter
            .layout_no_wrap(shown, font_id.clone(), text_color)
            .size()
            .x
            .max(8.0)
            + 2.0
    };
    let height = (text_height + 8.0).max(22.0);
    const MARGIN_Y: i8 = 3;

    let frame = egui::Frame::NONE
        .fill(fill_color)
        .stroke(egui::Stroke::new(1.0_f32, stroke_color))
        .corner_radius(egui::CornerRadius::same(10))
        .inner_margin(egui::Margin::symmetric(10, MARGIN_Y))
        .show(ui, |ui| {
            ui.set_min_height(height - 2.0 * f32::from(MARGIN_Y));
            ui.spacing_mut().item_spacing.x = 1.0;
            // The parent row is right-to-left, so the suffix goes in first.
            ui.add(
                egui::Label::new(
                    egui::RichText::new("x")
                        .font(font_id.clone())
                        .color(text_color.opacity(0.6)),
                )
                .selectable(false),
            );
            let current = format_speed_value(playback_speed.0);
            let mut text = if editing {
                let text = ui
                    .data(|data| data.get_temp::<String>(buffer_id))
                    .unwrap_or_else(|| current.clone());
                filter_speed_events(ui, edit_id, &text);
                text
            } else {
                current.clone()
            };
            let width = field_width(&text);
            let response = ui.add(
                egui::TextEdit::singleline(&mut text)
                    .id(edit_id)
                    .frame(egui::Frame::NONE)
                    .font(font_id.clone())
                    .text_color(text_color)
                    .hint_text(egui::RichText::new(current).color(text_color.opacity(0.45)))
                    .char_limit(playback::SPEED_INPUT_MAX_CHARS)
                    .desired_width(width)
                    .horizontal_align(egui::Align::RIGHT)
                    .margin(egui::Margin::ZERO),
            );
            (response, text)
        });
    let (field, mut text) = frame.inner;

    if field.changed() {
        text = playback::sanitize_speed_input(&text);
    }
    if field.gained_focus() {
        select_all(ui.ctx(), edit_id, text.chars().count());
    }

    if field.has_focus() {
        let step = ui.input_mut(|input| {
            if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                1
            } else if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                -1
            } else {
                0
            }
        });
        if step != 0 {
            let from = playback::parse_playback_speed(&text).unwrap_or(playback_speed.0);
            let speed = playback::adjacent_playback_speed(from, step);
            set_playback_speed(speed, playback_speed, latest_follow, auto_follow);
            text = format_speed_value(speed);
            select_all(ui.ctx(), edit_id, text.chars().count());
        }
        ui.data_mut(|data| data.insert_temp(buffer_id, text));
    } else if field.lost_focus() {
        let (cancelled, submitted) = ui.input(|input| {
            (
                input.key_pressed(egui::Key::Escape),
                input.key_pressed(egui::Key::Enter),
            )
        });
        // In LIVE the stored speed is stale, so Enter on the same value must
        // still hand the lead back to playback.
        if !cancelled
            && let Some(speed) = playback::parse_playback_speed(&text)
            && (!playback::same_playback_speed(speed, playback_speed.0)
                || (submitted && latest_follow.0))
        {
            set_playback_speed(speed, playback_speed, latest_follow, auto_follow);
        }
        ui.data_mut(|data| data.remove::<String>(buffer_id));
    }

    ui.data_mut(|data| data.insert_temp(rect_id, frame.response.rect));
    // At rest the whole pill is the click target, not just the digits. While
    // editing the overlay is dropped so clicks place the caret in the field.
    let response = if editing {
        frame.response
    } else {
        let pill = ui.interact(
            frame.response.rect,
            edit_id.with("pill"),
            egui::Sense::click(),
        );
        if pill.clicked() {
            ui.memory_mut(|memory| memory.request_focus(edit_id));
        }
        pill.on_hover_cursor(egui::CursorIcon::Text).on_hover_text(
            "Type a speed like 0.5 or 2.4, Enter to apply, Esc to cancel. ↑/↓ or scroll steps presets",
        )
    };

    if response.hovered() && !editing {
        // egui keeps `smooth_scroll_delta` nonzero for several frames after one
        // wheel notch, so stepping a preset per nonzero frame jumps several
        // presets per gesture. Accumulate the delta and step once the summed
        // scroll crosses a notch's worth of points.
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        if scroll != 0.0 {
            const POINTS_PER_STEP: f32 = 50.0;
            let acc_id = ui.make_persistent_id("playback_speed_scroll_acc");
            let mut acc = ui.data(|data| data.get_temp::<f32>(acc_id).unwrap_or(0.0)) + scroll;
            let mut speed = playback_speed.0;
            let mut changed = false;
            while acc.abs() >= POINTS_PER_STEP {
                let direction = acc.signum();
                speed = playback::adjacent_playback_speed(speed, direction as i32);
                acc -= direction * POINTS_PER_STEP;
                changed = true;
            }
            ui.data_mut(|data| data.insert_temp(acc_id, acc));
            if changed {
                set_playback_speed(speed, playback_speed, latest_follow, auto_follow);
            }
        }
    }

    let speed_label = egui::RichText::new("SPEED").color(scheme.text_secondary);
    ui.add_space(8.0);
    ui.add(egui::Label::new(speed_label).selectable(false));
}

/// Drop typed or pasted characters the speed field rejects before the
/// `TextEdit` sees them. A separator is only let through when the field has
/// none, or the selection about to be replaced holds it.
fn filter_speed_events(ui: &mut egui::Ui, edit_id: egui::Id, text: &str) {
    let is_separator = |c: char| c == '.' || c == ',';
    let selection = egui::TextEdit::load_state(ui.ctx(), edit_id)
        .and_then(|state| state.cursor.char_range())
        .map(|range| range.as_sorted_char_range())
        .unwrap_or(0..0);
    let selected_separator = text
        .chars()
        .skip(selection.start)
        .take(selection.len())
        .any(is_separator);
    let mut separator_allowed = !text.contains(is_separator) || selected_separator;
    ui.input_mut(|input| {
        input.events.retain_mut(|event| match event {
            egui::Event::Text(typed) | egui::Event::Paste(typed) => {
                *typed = playback::filter_speed_keystrokes(typed, &mut separator_allowed);
                !typed.is_empty()
            }
            _ => true,
        });
    });
}

/// Select the whole field so the next keystroke replaces the value.
fn select_all(ctx: &egui::Context, id: egui::Id, len: usize) {
    if let Some(mut state) = egui::TextEdit::load_state(ctx, id) {
        state
            .cursor
            .set_char_range(Some(egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(len),
            )));
        state.store(ctx, id);
    }
}

fn format_speed_value(speed: f64) -> String {
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
    value
}

fn format_lag_counter(micros: i64) -> String {
    let micros = micros.max(0);
    if micros == 0 {
        return "0ms".to_owned();
    }

    if micros >= 3_600_000_000 {
        return format!("+{:.1}h", micros as f64 / 3_600_000_000.0);
    }
    if micros >= 60_000_000 {
        return format!("+{:.1}m", micros as f64 / 60_000_000.0);
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
    let counter_fixed_width = ["0ms", "+9999ms", "+9999us", "+999.9s", "+99.9m", "+99.9h"]
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
            egui::Stroke::new(1.0_f32, stroke_color),
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
            egui::Stroke::new(1.0_f32, stroke_color.opacity(0.65)),
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
