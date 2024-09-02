use std::{collections::BTreeMap, ops::RangeInclusive};

use bevy::ecs::{
    event::EventWriter,
    system::{Query, Res, ResMut, Resource, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui::{self, emath::Numeric, Response};
use impeller::{bevy::Tick, ControlMsg};

use crate::ui::{colors, ViewportRange};
use crate::ui::{
    widgets::{plot::GraphState, WidgetSystem},
    SelectedObject,
};

use super::{position_from_value, value_from_position};

#[derive(Debug, Clone)]
pub struct TimelineRange {
    pub values: (u64, u64),
    pub label: String,
    pub color: egui::Color32,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct TimelineRanges(pub BTreeMap<TimelineRangeId, TimelineRange>);

#[derive(Resource, Debug, Default, Clone)]
pub struct TimelineRangesFocused(pub bool);

impl TimelineRanges {
    pub fn create_range(&mut self, max: u64) -> TimelineRangeId {
        let new_range_id = self
            .0
            .keys()
            .max()
            .map_or(TimelineRangeId(0), |lk| TimelineRangeId(lk.0 + 1));

        let new_range = TimelineRange {
            values: (0, max),
            label: format!("Range #{:?}", &new_range_id.0),
            color: colors::PRIMARY_CREAME,
        };

        self.0.insert(new_range_id, new_range);

        new_range_id
    }

    pub fn remove_range(&mut self, range_id: &TimelineRangeId) {
        self.0.remove(range_id);
    }

    pub fn is_not_empty(&self) -> bool {
        !self.0.is_empty()
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TimelineRangeId(pub u64);

impl From<u64> for TimelineRangeId {
    fn from(val: u64) -> Self {
        TimelineRangeId(val)
    }
}

pub fn timeline_range(
    ui: &mut egui::Ui,
    full_range: &RangeInclusive<f64>,
    range: &mut TimelineRange,
    range_color: egui::Color32,
    position_range: egui::Rangef,
) -> Response {
    let line_width = 4.0;
    let circle_radius = 6.0;
    let circle_border_width = 2.0;

    let (start, end) = range.values;
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
            (start_f64 + center_delta)
                .clamp(*full_range_min, *full_range_max)
                .floor() as u64,
            (end_f64 + center_delta)
                .clamp(*full_range_min, *full_range_max)
                .floor() as u64,
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

        if min_pos_response.drag_stopped() {
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

        if max_pos_response.drag_stopped() {
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
pub struct TimelineRangesPanel<'w, 's> {
    timeline_ranges: ResMut<'w, TimelineRanges>,
    tick: ResMut<'w, Tick>,
    event: EventWriter<'w, ControlMsg>,
    viewport_range: Res<'w, ViewportRange>,
    selected_object: Res<'w, SelectedObject>,
    graph_states: Query<'w, 's, &'static GraphState>,
}

impl WidgetSystem for TimelineRangesPanel<'_, '_> {
    type Args = (f32, RangeInclusive<f64>, egui::Rangef);
    type Output = egui::Rect;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let (line_height, full_range, position_range) = args;

        let state_mut = state.get_mut(world);
        let mut timeline_ranges = state_mut.timeline_ranges;
        let mut tick = state_mut.tick;
        let mut event = state_mut.event;
        let viewport_range = state_mut.viewport_range;
        let graph_states = state_mut.graph_states;

        let selected_graph_range_id = match state_mut.selected_object.to_owned() {
            SelectedObject::Graph { graph_id, .. } => {
                if let Ok(graph_state) = graph_states.get(graph_id) {
                    if let Some(range_id) = graph_state.range_id {
                        if timeline_ranges.0.contains_key(&range_id) {
                            Some(range_id)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(viewport_range_id) = &viewport_range.0 {
            if let Some(viewport_range) = timeline_ranges.0.get(viewport_range_id) {
                let (a, b) = viewport_range.values;
                let fixed_range = if a > b { b..a } else { a..b };

                if !fixed_range.contains(&tick.0) {
                    tick.0 = fixed_range.start;
                    event.send(ControlMsg::Rewind(tick.0));
                }
            }
        }

        let frame = egui::Frame::none()
            .stroke(egui::Stroke::new(1.0, colors::BLACK_BLACK_600))
            .show(ui, |ui| {
                let line_padding = (line_height - ui.spacing().interact_size.y) / 2.0;

                for (range_id, range) in timeline_ranges.0.iter_mut() {
                    let range_label = range.label.clone();
                    let range_color = if let Some(selected_graph_range_id) = selected_graph_range_id
                    {
                        if selected_graph_range_id == *range_id {
                            colors::PRIMARY_CREAME
                        } else {
                            colors::PRIMARY_CREAME_6
                        }
                    } else {
                        colors::PRIMARY_CREAME
                    };

                    egui::Frame::none()
                        .inner_margin(egui::Margin::symmetric(0.0, line_padding))
                        .show(ui, |ui| {
                            timeline_range(ui, &full_range, range, range_color, position_range);
                        })
                        .response
                        .on_hover_text_at_pointer(range_label);
                }
            });

        frame.response.rect
    }
}
