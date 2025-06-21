use bevy::ecs::{
    system::{Local, Res, SystemParam, SystemState},
    world::World,
};
use bevy_egui::{EguiContexts, egui};
use impeller2_wkt::SimulationTimeStep;
use timeline_controls::TimelineControls;

use std::ops::RangeInclusive;
use timeline_slider::TimelineSlider;

use crate::{
    SelectedTimeRange,
    ui::{colors::get_scheme, images},
};

use super::widgets::{WidgetSystem, WidgetSystemExt};

pub mod timeline_controls;
pub mod timeline_slider;

#[derive(Clone)]
pub struct TimelineArgs {
    pub available_width: f32,
    pub line_height: f32,
    pub segment_count: u8,
    pub frames_per_second: f64,
    pub active_range: RangeInclusive<i64>,
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
    pub vertical_chevrons: egui::TextureId,
}

#[derive(SystemParam)]
pub struct TimelinePanel<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    tick_time: Res<'w, SimulationTimeStep>,
    selected_time_range: Res<'w, SelectedTimeRange>,
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
        let active_range =
            state_mut.selected_time_range.0.start.0..=state_mut.selected_time_range.0.end.0;

        let frames_per_second = 1.0 / tick_time.0;

        let timeline_icons = TimelineIcons {
            jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
            jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
            frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
            frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
            play: contexts.add_image(images.icon_play.clone_weak()),
            pause: contexts.add_image(images.icon_pause.clone_weak()),
            handle: contexts.add_image(images.icon_scrub.clone_weak()),
            add: contexts.add_image(images.icon_add.clone_weak()),
            remove: contexts.add_image(images.icon_subtract.clone_weak()),
            range_loop: contexts.add_image(images.icon_loop.clone_weak()),
            vertical_chevrons: contexts.add_image(images.icon_vertical_chevrons.clone_weak()),
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
