use bevy::ecs::{
    system::{Local, Res, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiTextureHandle, egui};
use impeller2_bevy::CurrentStreamId;
use impeller2_wkt::{
    CurrentTimestamp, EarliestTimestamp, LastUpdated, SimulationTimeStep, StreamId,
};
use timeline_controls::TimelineControls;

use std::ops::RangeInclusive;
use timeline_slider::TimelineSlider;

use crate::{
    FullTimeRange, SelectedTimeRange, TimeRangeBehavior,
    ui::{colors::get_scheme, images},
};

use super::widgets::{WidgetSystem, WidgetSystemExt};

pub mod timeline_controls;
pub mod timeline_slider;

pub(crate) fn plugin(app: &mut App) {
    app.add_plugins(timeline_controls::plugin)
        .init_resource::<PlaybackSpeed>()
        .init_resource::<TimelineSettings>()
        .init_resource::<LatestFollow>()
        .init_resource::<AutoFollowLatestState>()
        .add_systems(
            Update,
            (
                reset_playback_speed_on_stream_change,
                reset_latest_follow_on_stream_change,
                reset_auto_follow_latest_state,
                auto_start_follow_latest,
            ),
        );
}

#[derive(bevy::prelude::Resource, Clone, Copy, Debug)]
pub struct PlaybackSpeed(pub f64);

impl Default for PlaybackSpeed {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(bevy::prelude::Resource, Default, Clone, Copy, Debug)]
pub struct LatestFollow(pub bool);

#[derive(bevy::prelude::Resource, Clone, Copy, Debug, PartialEq)]
pub struct TimelineSettings {
    pub played_color: impeller2_wkt::Color,
    pub future_color: impeller2_wkt::Color,
    pub follow_latest: bool,
}

impl Default for TimelineSettings {
    fn default() -> Self {
        Self::from(impeller2_wkt::TimelineConfig::default())
    }
}

impl From<impeller2_wkt::TimelineConfig> for TimelineSettings {
    fn from(value: impeller2_wkt::TimelineConfig) -> Self {
        Self {
            played_color: value.played_color,
            future_color: value.future_color,
            follow_latest: value.follow_latest,
        }
    }
}

impl From<TimelineSettings> for impeller2_wkt::TimelineConfig {
    fn from(value: TimelineSettings) -> Self {
        Self {
            played_color: value.played_color,
            future_color: value.future_color,
            follow_latest: value.follow_latest,
        }
    }
}

#[derive(bevy::prelude::Resource, Default, Clone, Copy, Debug)]
pub(crate) struct AutoFollowLatestState {
    stream_id: Option<StreamId>,
    baseline_latest: Option<impeller2::types::Timestamp>,
    armed: bool,
}

impl AutoFollowLatestState {
    pub fn cancel(&mut self) {
        self.armed = false;
    }
}

#[derive(SystemParam)]
struct AutoFollowLatestParams<'w> {
    current_stream_id: Res<'w, CurrentStreamId>,
    timeline_settings: Res<'w, TimelineSettings>,
    earliest: Res<'w, EarliestTimestamp>,
    latest: Res<'w, LastUpdated>,
    replay_mode: Option<Res<'w, crate::ReplayMode>>,
    current_timestamp: ResMut<'w, CurrentTimestamp>,
    paused: ResMut<'w, crate::ui::Paused>,
    latest_follow: ResMut<'w, LatestFollow>,
    state: ResMut<'w, AutoFollowLatestState>,
}

fn reset_playback_speed_on_stream_change(
    current_stream_id: Res<CurrentStreamId>,
    mut playback_speed: ResMut<PlaybackSpeed>,
) {
    if current_stream_id.is_changed() {
        *playback_speed = PlaybackSpeed::default();
    }
}

fn reset_latest_follow_on_stream_change(
    current_stream_id: Res<CurrentStreamId>,
    mut latest_follow: ResMut<LatestFollow>,
) {
    if current_stream_id.is_changed() {
        latest_follow.0 = false;
    }
}

fn reset_auto_follow_latest_state(
    current_stream_id: Res<CurrentStreamId>,
    timeline_settings: Res<TimelineSettings>,
    replay_mode: Option<Res<crate::ReplayMode>>,
    mut state: ResMut<AutoFollowLatestState>,
) {
    if replay_mode.is_some() || !timeline_settings.follow_latest {
        state.armed = false;
        state.baseline_latest = None;
        return;
    }

    if current_stream_id.is_changed() || timeline_settings.is_changed() {
        state.stream_id = Some(**current_stream_id);
        state.baseline_latest = None;
        state.armed = true;
    }
}

fn auto_start_follow_latest(params: AutoFollowLatestParams) {
    let AutoFollowLatestParams {
        current_stream_id,
        timeline_settings,
        earliest,
        latest,
        replay_mode,
        mut current_timestamp,
        mut paused,
        mut latest_follow,
        mut state,
    } = params;

    if replay_mode.is_some() || !timeline_settings.follow_latest || latest_follow.0 {
        return;
    }

    if state.stream_id != Some(**current_stream_id) {
        state.stream_id = Some(**current_stream_id);
        state.baseline_latest = None;
        state.armed = true;
    }
    if !state.armed || earliest.0.0 == i64::MAX || latest.0.0 == i64::MIN {
        return;
    }

    match state.baseline_latest {
        None => {
            state.baseline_latest = Some(latest.0);
        }
        Some(baseline_latest) if latest.0 > baseline_latest => {
            latest_follow.0 = true;
            paused.0 = false;
            current_timestamp.0 = latest.0;
            state.armed = false;
        }
        Some(_) => {}
    }
}

#[derive(bevy::prelude::Resource, Default, Clone, Copy, Debug)]
pub struct StreamTickOrigin {
    stream_id: Option<StreamId>,
    timestamp: Option<impeller2::types::Timestamp>,
    pending_rebase: bool,
}

impl StreamTickOrigin {
    pub fn observe_stream(&mut self, stream_id: StreamId) {
        if self.stream_id != Some(stream_id) {
            self.stream_id = Some(stream_id);
            self.timestamp = None;
            self.pending_rebase = false;
        }
    }

    pub fn request_rebase(&mut self) {
        self.pending_rebase = true;
    }

    pub fn observe_tick(
        &mut self,
        tick: impeller2::types::Timestamp,
        earliest: impeller2::types::Timestamp,
    ) {
        if tick < earliest {
            return;
        }

        if self.pending_rebase {
            self.timestamp = Some(match self.timestamp {
                Some(origin) => origin.min(tick),
                None => tick,
            });
            self.pending_rebase = false;
        } else if let Some(origin) = self.timestamp
            && tick < origin
        {
            self.timestamp = Some(tick);
        }
    }

    pub fn origin(&self, fallback: impeller2::types::Timestamp) -> impeller2::types::Timestamp {
        self.timestamp.unwrap_or(fallback)
    }
}

#[derive(Clone)]
pub struct TimelineArgs {
    pub available_width: f32,
    pub line_height: f32,
    pub segment_count: u8,
    pub frames_per_second: f64,
    pub active_range: RangeInclusive<i64>,
    pub focus_range: Option<RangeInclusive<i64>>,
}

/// Returns a `value` based on the `position` in the timeline
///
/// # Arguments
///
/// * `position` - A mouse position
/// * `value_range` - A range of the timeline values
/// * `position_range` - A range of the timeline on the screen
pub fn value_from_position(
    position: f32,
    value_range: RangeInclusive<f64>,
    position_range: egui::Rangef,
) -> f64 {
    let normalized = egui::emath::remap_clamp(position, position_range, 0.0..=1.0) as f64;
    egui::emath::lerp(value_range, normalized.clamp(0.0, 1.0))
}

/// Returns a `position` in the timeline based on the `value`
///
/// # Arguments
///
/// * `value` - A value from the underling value range
/// * `value_range` - A range of the timeline values
/// * `position_range` - A range of the timeline on the screen
pub fn position_from_value(
    value: f64,
    value_range: RangeInclusive<f64>,
    position_range: egui::Rangef,
) -> f32 {
    let normalized = egui::emath::remap(value, value_range, 0.0..=1.0);
    egui::emath::lerp(position_range, normalized as f32)
}

pub fn get_position_range(
    x_range: egui::Rangef,
    active_duration: f64,
    full_duration: f64,
) -> egui::Rangef {
    let active_range = (x_range.span() as f64 / full_duration) * active_duration;
    egui::Rangef {
        min: x_range.min,
        max: x_range.min + active_range as f32,
    }
}

pub trait DurationExt {
    fn ceil_approx(&self) -> hifitime::Duration;
    fn segment_round(&self) -> hifitime::Duration;
}

impl DurationExt for hifitime::Duration {
    fn ceil_approx(&self) -> hifitime::Duration {
        use hifitime::Unit;
        let (_, days, hours, minutes, seconds, milli, us, _) = self.decompose();

        let round_to = if days > 0 {
            1 * Unit::Day
        } else if hours > 0 {
            1 * Unit::Hour
        } else if minutes > 0 {
            1 * Unit::Minute
        } else if seconds > 0 {
            1 * Unit::Second
        } else if milli > 0 {
            1 * Unit::Millisecond
        } else if us > 0 {
            1 * Unit::Microsecond
        } else {
            1 * Unit::Nanosecond
        };

        self.ceil(round_to)
    }
    #[allow(clippy::match_overlapping_arm)]
    fn segment_round(&self) -> hifitime::Duration {
        use hifitime::{TimeUnits, Unit};
        let (_, days, hours, minutes, seconds, milli, us, _) = self.decompose();
        let round_to = if days > 0 {
            match days {
                ..=2 => 1,
                ..=5 => 5,
                ..=15 => 15,
                ..=30 => 30,
                _ => 50,
            }
            .days()
        } else if hours > 0 {
            match hours {
                ..=2 => 1,
                ..=6 => 6,
                ..=12 => 12,
                _ => 24,
            }
            .hours()
        } else if minutes > 0 {
            match minutes {
                ..=2 => 1,
                ..=5 => 5,
                ..=15 => 15,
                ..=30 => 30,
                _ => 60,
            }
            .minutes()
        } else if seconds > 0 {
            match seconds {
                ..=2 => 1,
                ..=5 => 5,
                ..=15 => 15,
                ..=30 => 30,
                _ => 60,
            }
            .seconds()
        } else if milli > 0 {
            match milli {
                ..=2 => 1,
                ..=5 => 5,
                ..=10 => 10,
                ..=25 => 25,
                ..=50 => 50,
                ..=100 => 100,
                ..=250 => 250,
                ..=500 => 500,
                _ => 1000,
            }
            .milliseconds()
        } else if us > 0 {
            1 * Unit::Microsecond
        } else {
            1 * Unit::Nanosecond
        };

        self.ceil(round_to)
    }
}

#[derive(Clone, Copy)]
pub struct TimelineIcons {
    pub jump_to_start: egui::TextureId,
    pub jump_to_end: egui::TextureId,
    pub frame_forward: egui::TextureId,
    pub frame_back: egui::TextureId,
    pub play: egui::TextureId,
    pub pause: egui::TextureId,
    pub handle: egui::TextureId,
    pub add: egui::TextureId,
    pub remove: egui::TextureId,
    pub range_loop: egui::TextureId,
    pub setting: egui::TextureId,
    pub vertical_chevrons: egui::TextureId,
}

#[derive(SystemParam)]
pub struct TimelinePanel<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    tick_time: Res<'w, SimulationTimeStep>,
    selected_time_range: Res<'w, SelectedTimeRange>,
    full_time_range: Res<'w, FullTimeRange>,
    time_range_behavior: Res<'w, TimeRangeBehavior>,
}

impl WidgetSystem for TimelinePanel<'_, '_> {
    type Args = ();
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let mut contexts = state_mut.contexts;
        let images = state_mut.images;
        let tick_time = state_mut.tick_time;
        let active_range = state_mut.full_time_range.0.start.0..=state_mut.full_time_range.0.end.0;
        let is_full = *state_mut.time_range_behavior == TimeRangeBehavior::default();
        let focus_range = if is_full {
            None
        } else {
            Some(state_mut.selected_time_range.0.start.0..=state_mut.selected_time_range.0.end.0)
        };

        let frames_per_second = 1.0 / tick_time.0;

        let timeline_icons = TimelineIcons {
            jump_to_start: contexts
                .add_image(EguiTextureHandle::Weak(images.icon_jump_to_start.id())),
            jump_to_end: contexts.add_image(EguiTextureHandle::Weak(images.icon_jump_to_end.id())),
            frame_forward: contexts
                .add_image(EguiTextureHandle::Weak(images.icon_frame_forward.id())),
            frame_back: contexts.add_image(EguiTextureHandle::Weak(images.icon_frame_back.id())),
            play: contexts.add_image(EguiTextureHandle::Weak(images.icon_play.id())),
            pause: contexts.add_image(EguiTextureHandle::Weak(images.icon_pause.id())),
            handle: contexts.add_image(EguiTextureHandle::Weak(images.icon_scrub.id())),
            add: contexts.add_image(EguiTextureHandle::Weak(images.icon_add.id())),
            remove: contexts.add_image(EguiTextureHandle::Weak(images.icon_subtract.id())),
            range_loop: contexts.add_image(EguiTextureHandle::Weak(images.icon_loop.id())),
            setting: contexts.add_image(EguiTextureHandle::Weak(images.icon_setting.id())),
            vertical_chevrons: contexts
                .add_image(EguiTextureHandle::Weak(images.icon_vertical_chevrons.id())),
        };

        egui::TopBottomPanel::bottom("timeline_panel")
            .frame(egui::Frame {
                fill: get_scheme().bg_primary,
                //stroke: egui::Stroke::new(1.0, get_scheme().border_primary),
                ..Default::default()
            })
            .resizable(false)
            .show_inside(ui, |ui| {
                let available_width = ui.available_width();

                ui.add_widget_with::<TimelineControls>(world, "timeline_controls", timeline_icons);

                ui.add(egui::Separator::default().spacing(0.0));

                let timeline_args = TimelineArgs {
                    available_width,
                    line_height: 40.0,
                    segment_count: (available_width / 90.0) as u8,
                    frames_per_second,
                    active_range,
                    focus_range,
                };

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.add_widget_with::<TimelineSlider>(
                            world,
                            "timeline_slider",
                            (timeline_icons, timeline_args.clone()),
                        );
                    });
                });
            });
    }
}
