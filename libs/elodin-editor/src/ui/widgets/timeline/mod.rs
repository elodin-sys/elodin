use bevy_egui::egui;
use std::ops::RangeInclusive;

pub mod tagged_range;
pub mod timeline_widget;

#[derive(Clone)]
pub struct TimelineArgs {
    pub available_width: f32,
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
