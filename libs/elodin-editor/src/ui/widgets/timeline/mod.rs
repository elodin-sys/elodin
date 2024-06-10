use bevy::ecs::{
    system::{Local, Res, SystemParam, SystemState},
    world::World,
};
use bevy_egui::{egui, EguiContexts};
use conduit::bevy::{MaxTick, Tick, TimeStep};
use egui::emath::Numeric;
use timeline_controls::TimelineControls;
use timeline_ranges::{TimelineRanges, TimelineRangesPanel};

use std::ops::RangeInclusive;
use timeline_slider::TimelineSlider;

use crate::ui::{colors, images};

use super::{WidgetSystem, WidgetSystemExt};

pub mod timeline_controls;
mod timeline_range_list;
pub mod timeline_ranges;
pub mod timeline_slider;

#[derive(Clone)]
pub struct TimelineArgs {
    pub available_width: f32,
    pub line_height: f32,
    pub segment_count: u8,
    pub frames_per_second: f64,
    pub active_range: RangeInclusive<u64>,
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
    let normalized = egui::emath::remap_clamp(value, value_range, 0.0..=1.0);
    egui::emath::lerp(position_range, normalized as f32)
}

pub fn get_segment_size(fps: f64, segments: f64, active_range_end: f64) -> usize {
    let min_offset = 0.1;
    let min_end = fps * segments;

    let visual_end = if active_range_end > min_end {
        active_range_end + (active_range_end * min_offset)
    } else {
        min_end + (min_end * min_offset)
    };
    let total_time_sec = (visual_end / fps).ceil();

    (total_time_sec / segments).ceil() as usize
}

pub fn get_position_range(
    x_range: egui::Rangef,
    fps: f64,
    segments: f64,
    segment_size: f64,
    active_range_start: f64,
    active_range_end: f64,
) -> egui::Rangef {
    let total_timeline_frames = segment_size * segments * fps;
    let not_sim_offset =
        (total_timeline_frames - active_range_end) / (total_timeline_frames - active_range_start);

    // NOTE: Active zone starts in a center of the first segment and ends in a the center of the last one
    let offset = (x_range.span() / (segments as f32 + 1.0)) / 2.0;
    let full_range = x_range.shrink(offset);

    egui::Rangef {
        min: full_range.min,
        max: full_range.max - ((full_range.max - full_range.min) * (not_sim_offset as f32)),
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
}

#[derive(SystemParam)]
pub struct TimelinePanel<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    tick: Res<'w, Tick>,
    max_tick: Res<'w, MaxTick>,
    tick_time: Res<'w, TimeStep>,
    timeline_ranges: Res<'w, TimelineRanges>,
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
        let tick = state_mut.tick;
        let max_tick = state_mut.max_tick;
        let tick_time = state_mut.tick_time;

        let ranges_not_empty = state_mut.timeline_ranges.is_not_empty();

        let active_range = 0..=max_tick.0;
        let frames_per_second = 1.0 / tick_time.0.as_secs_f64();

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
        };

        let tick_value = tick.0 as f64;

        egui::TopBottomPanel::bottom("timeline_panel")
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                ..Default::default()
            })
            .resizable(false)
            .show_inside(ui, |ui| {
                let available_width = ui.available_width();

                ui.add_widget_with::<TimelineControls>(world, "timeline_controls", timeline_icons);

                ui.add(egui::Separator::default().spacing(0.0));

                let range_list_width = 200.0;
                let timeline_args = TimelineArgs {
                    available_width: available_width - range_list_width,
                    line_height: 40.0,
                    segment_count: (available_width / 100.0) as u8,
                    frames_per_second,
                    active_range,
                };

                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        let line_size = egui::vec2(range_list_width, timeline_args.line_height);

                        ui.add_widget_with::<timeline_range_list::TimelineRangeListHeader>(
                            world,
                            "range_list_header",
                            (timeline_icons.range_loop, timeline_icons.add, line_size),
                        );

                        ui.add_widget_with::<timeline_range_list::TimelineRangeList>(
                            world,
                            "range_list_items",
                            (timeline_icons.range_loop, timeline_icons.remove, line_size),
                        );
                    });

                    ui.vertical(|ui| {
                        let segments = timeline_args.segment_count as f64;
                        let active_range_start = timeline_args.active_range.start().to_f64();
                        let active_range_end = timeline_args.active_range.end().to_f64();
                        let full_range = active_range_start..=active_range_end;
                        // NOTE: Replicate same calculation done in timeline widget to be in sync
                        let position_range = get_position_range(
                            ui.available_rect_before_wrap().x_range(),
                            frames_per_second,
                            segments,
                            get_segment_size(frames_per_second, segments, active_range_end) as f64,
                            active_range_start,
                            active_range_end,
                        );

                        ui.add_widget_with::<TimelineSlider>(
                            world,
                            "timeline_slider",
                            (timeline_icons, timeline_args.clone()),
                        );

                        if ranges_not_empty {
                            let timeline_ranges_rect = ui.add_widget_with::<TimelineRangesPanel>(
                                world,
                                "timeline_ranges_panel",
                                (
                                    timeline_args.line_height,
                                    full_range.clone(),
                                    position_range,
                                ),
                            );

                            let y_range = timeline_ranges_rect.y_range();

                            let handle_size = egui::vec2(12.0, 30.0);
                            let stroke_width =
                                ((timeline_args.line_height / handle_size.y) * handle_size.x) / 6.0;

                            let x_pos = position_from_value(
                                active_range_end,
                                full_range.clone(),
                                position_range,
                            );
                            let line_stroke = egui::Stroke::new(stroke_width, colors::WHITE);
                            ui.painter().vline(x_pos, y_range, line_stroke);

                            let x_pos = position_from_value(tick_value, full_range, position_range);
                            let line_stroke = egui::Stroke::new(stroke_width, colors::MINT_DEFAULT);
                            ui.painter().vline(x_pos, y_range, line_stroke);
                        }
                    });
                });
            });
    }
}
