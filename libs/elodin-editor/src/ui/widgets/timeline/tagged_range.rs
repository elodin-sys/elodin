use std::{collections::BTreeMap, ops::RangeInclusive};

use bevy::ecs::{
    system::{Res, ResMut, Resource, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui::{self, emath::Numeric, Response};
use conduit::bevy::Tick;
use nalgebra::clamp;

use crate::ui::colors;
use crate::ui::widgets::WidgetSystem;

use super::{
    get_position_range, get_segment_size, position_from_value, value_from_position, TimelineArgs,
};

#[derive(Debug, Clone)]
pub struct TaggedRange {
    pub values: (u64, u64),
    pub label: String,
    pub color: egui::Color32,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct TaggedRanges(pub BTreeMap<TaggedRangeId, TaggedRange>);

impl TaggedRanges {
    pub fn create_range(&mut self, max: u64) -> TaggedRangeId {
        let new_range_id = self
            .0
            .keys()
            .max()
            .map_or(TaggedRangeId(0), |lk| TaggedRangeId(lk.0 + 1));

        let new_range = TaggedRange {
            values: (0, max),
            label: format!("Range #{:?}", &new_range_id.0),
            color: colors::PEACH_DEFAULT,
        };

        self.0.insert(new_range_id.clone(), new_range);

        new_range_id
    }

    pub fn remove_range(&mut self, range_id: &TaggedRangeId) {
        self.0.remove(range_id);
    }

    pub fn is_not_empty(&self) -> bool {
        !self.0.is_empty()
    }
}

#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TaggedRangeId(pub u64);

impl From<u64> for TaggedRangeId {
    fn from(val: u64) -> Self {
        TaggedRangeId(val)
    }
}

pub fn tagged_range(
    ui: &mut egui::Ui,
    full_range: &RangeInclusive<f64>,
    range: &mut TaggedRange,
    position_range: egui::Rangef,
) -> Response {
    let line_width = 4.0;
    let circle_radius = 6.0;
    let circle_border_width = 2.0;

    let (start, end) = range.values;
    let range_color = range.color;
    let start_f64 = start.to_f64();
    let end_f64 = end.to_f64();
    let full_range_min = full_range.start();
    let full_range_max = full_range.end();

    let desired_size = egui::vec2(ui.available_width(), ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    let bg_stroke = egui::Stroke::new(line_width / 2.0, egui::Color32::DARK_GRAY);

    let min_pos_x = position_from_value(start_f64, full_range.clone(), position_range);
    let max_pos_x = position_from_value(end_f64, full_range.clone(), position_range);

    let bg_line = [
        egui::pos2(position_range.min, rect.center().y),
        egui::pos2(position_range.max, rect.center().y),
    ];
    ui.painter().line_segment(bg_line, bg_stroke);

    // HANDLES

    let circle_diameter = circle_radius * 2.0;
    let min_pos = egui::pos2(min_pos_x, rect.center().y);
    let max_pos = egui::pos2(max_pos_x, rect.center().y);

    let handle_size = egui::vec2(circle_diameter, circle_diameter);

    // LINE

    let fg_line = [min_pos, max_pos];
    let mut fg_stroke = egui::Stroke::new(line_width, range_color);

    let line_rect = egui::Rect::from_min_max(min_pos, max_pos)
        .expand2(egui::vec2(-circle_radius, circle_radius));
    let line_response = ui.allocate_rect(line_rect, egui::Sense::click_and_drag());

    if let Some(interact_pos) = line_response.interact_pointer_pos() {
        let new_center = value_from_position(interact_pos.x, full_range.clone(), position_range);
        let center = start_f64 + ((end_f64 - start_f64) / 2.0);
        let center_delta = new_center - center;

        range.values = (
            clamp(start_f64 + center_delta, *full_range_min, *full_range_max).floor() as u64,
            clamp(end_f64 + center_delta, *full_range_min, *full_range_max).floor() as u64,
        );

        response.changed();

        line_response.on_hover_and_drag_cursor(egui::CursorIcon::Grabbing);
        fg_stroke = egui::Stroke::new(line_width, egui::Color32::WHITE);
    } else if line_response.hovered() {
        line_response.on_hover_cursor(egui::CursorIcon::PointingHand);
        fg_stroke = egui::Stroke::new(line_width, egui::Color32::WHITE);
    }

    ui.painter().line_segment(fg_line, fg_stroke);

    // START

    let min_pos_rect = egui::Rect::from_center_size(min_pos, handle_size);
    let min_pos_response = ui.allocate_rect(min_pos_rect, egui::Sense::click_and_drag());
    let mut min_stroke = egui::Stroke::NONE;

    if let Some(interact_pos) = min_pos_response.interact_pointer_pos() {
        let new_value = value_from_position(interact_pos.x, full_range.clone(), position_range);
        range.values = (new_value.floor() as u64, end);

        if min_pos_response.drag_released() {
            let (final_start, final_end) = range.values;

            if final_start > final_end {
                range.values = (final_end, final_start);
            }
        }

        response.changed();

        min_pos_response.on_hover_and_drag_cursor(egui::CursorIcon::Grabbing);
        min_stroke = egui::Stroke::new(circle_border_width, egui::Color32::WHITE);
    } else if min_pos_response.hovered() {
        min_pos_response.on_hover_cursor(egui::CursorIcon::PointingHand);
        min_stroke = egui::Stroke::new(circle_border_width, egui::Color32::WHITE);
    }

    ui.painter()
        .circle(min_pos, circle_radius, range_color, min_stroke);

    // END

    let max_pos_rect = egui::Rect::from_center_size(max_pos, handle_size);
    let max_pos_response = ui.allocate_rect(max_pos_rect, egui::Sense::click_and_drag());

    let mut max_stroke = egui::Stroke::NONE;

    if let Some(interact_pos) = max_pos_response.interact_pointer_pos() {
        let new_value = value_from_position(interact_pos.x, full_range.clone(), position_range);
        range.values = (start, new_value.floor() as u64);

        if max_pos_response.drag_released() {
            let (final_start, final_end) = range.values;

            if final_start > final_end {
                range.values = (final_end, final_start);
            }
        }

        response.changed();

        max_pos_response.on_hover_and_drag_cursor(egui::CursorIcon::Grabbing);
        max_stroke = egui::Stroke::new(circle_border_width, egui::Color32::WHITE);
    } else if max_pos_response.hovered() {
        max_pos_response.on_hover_cursor(egui::CursorIcon::PointingHand);
        max_stroke = egui::Stroke::new(circle_border_width, egui::Color32::WHITE);
    }

    ui.painter()
        .circle(max_pos, circle_radius, range_color, max_stroke);

    response
}

#[derive(SystemParam)]
pub struct TaggedRangesPanel<'w> {
    tagged_ranges: ResMut<'w, TaggedRanges>,
    tick: Res<'w, Tick>,
}

impl WidgetSystem for TaggedRangesPanel<'_> {
    type Args = TimelineArgs;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let timeline_args = args;

        let state_mut = state.get_mut(world);
        let mut tagged_ranges = state_mut.tagged_ranges;
        let tick = state_mut.tick;

        let y_range = ui.available_rect_before_wrap().y_range();

        egui::Frame::none()
            .stroke(egui::Stroke::new(1.0, colors::BLACK_BLACK_600))
            .inner_margin(egui::Margin::symmetric(0.0, 8.0))
            .show(ui, |ui| {
                let fps = timeline_args.frames_per_second;
                let segments = timeline_args.segment_count as f64;
                let active_range_start = timeline_args.active_range.start().to_f64();
                let active_range_end = timeline_args.active_range.end().to_f64();
                let full_range = active_range_start..=active_range_end;
                // NOTE: Replicate same calculation done in timeline widget to be in sync
                let position_range = get_position_range(
                    ui.available_rect_before_wrap().x_range(),
                    fps,
                    segments,
                    get_segment_size(fps, segments, active_range_end) as f64,
                    active_range_start,
                    active_range_end,
                );

                for (_, range) in tagged_ranges.0.iter_mut() {
                    let range_label = range.label.clone();
                    egui::Frame::none()
                        .inner_margin(egui::Margin::symmetric(0.0, 8.0))
                        .show(ui, |ui| {
                            tagged_range(ui, &full_range, range, position_range);
                        })
                        .response
                        .on_hover_text_at_pointer(range_label);
                }

                let x_pos =
                    position_from_value(active_range_end, full_range.clone(), position_range);
                let line_stroke = egui::Stroke::new(2.0, colors::WHITE);
                ui.painter().vline(x_pos, y_range, line_stroke);

                let x_pos = position_from_value(tick.0 as f64, full_range.clone(), position_range);
                let line_stroke = egui::Stroke::new(2.0, colors::MINT_DEFAULT);
                ui.painter().vline(x_pos, y_range, line_stroke);
            });
    }
}
