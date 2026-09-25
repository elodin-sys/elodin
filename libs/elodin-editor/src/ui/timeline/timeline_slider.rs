use bevy::{
    ecs::{
        change_detection::DetectChangesMut,
        system::{Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::{Deref, DerefMut, Resource},
};
use bevy_egui::egui;
use egui::{CornerRadius, Margin};
use impeller2::types::Timestamp;
use impeller2_bevy::CurrentStreamId;
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp};
use std::ops::RangeInclusive;

use crate::ui::{
    colors::{ColorExt, EColor, get_scheme},
    time_label::PrettyDuration,
    utils::{MarginSides, Shrink4},
    widgets::WidgetSystem,
};

use super::{
    AutoFollowLatestState, DurationExt, LatestFollow, StreamTickOrigin, TimelineArgs,
    TimelineIcons, TimelineSettings, get_position_range, playback::PlaybackDiscontinuities,
    playback::PlaybackRegion, position_from_value, value_from_position,
};
use crate::ui::widgets::SystemStateExt;

// ----------------------------------------------------------------------------

/// Combined into one function (rather than two) to make it easier for the borrow checker.
type GetSetValue<'a> = Box<dyn 'a + FnMut(Option<f64>) -> f64>;

fn get(get_set_value: &mut GetSetValue<'_>) -> f64 {
    (get_set_value)(None)
}

fn set(get_set_value: &mut GetSetValue<'_>, value: f64) {
    (get_set_value)(Some(value));
}

// ----------------------------------------------------------------------------

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct Timeline<'a> {
    get_set_value: GetSetValue<'a>,
    active_range: RangeInclusive<f64>,
    full_range: RangeInclusive<f64>,
    focus_range: Option<RangeInclusive<f64>>,
    selection: Option<&'a mut Option<(i64, i64)>>,
    gaps: &'a [(i64, i64)],
    handle_image_id: Option<egui::TextureId>,
    handle_image_tint: egui::Color32,
    max_handle_image_tint: egui::Color32,
    handle_aspect_ratio: f32,
    segments: u8,
    label_font_size: f32,
    height: f32,
    width: f32,
}

impl<'a> Timeline<'a> {
    /// Creates a new timeline
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut current_frame = 17;
    /// let frame_count = 3600;
    /// ui.add(
    ///     Timeline::new(&mut current_frame, 0..=frame_count)
    ///         .width(400.0)
    ///         .height(40.0)
    ///         .handle_aspect_ratio(12.0 / 30.0)
    ///         .segments(8)
    ///         .end(frame_count as f64),
    /// );
    /// ```
    pub fn new<Num: egui::emath::Numeric>(
        value: &'a mut Num,
        active_range: RangeInclusive<i64>,
    ) -> Self {
        let range_f64 = (*active_range.start() as f64)..=(*active_range.end() as f64);
        Self::from_get_set(range_f64, move |v: Option<f64>| {
            if let Some(v) = v {
                *value = Num::from_f64(v);
            }
            value.to_f64()
        })
    }

    pub fn from_get_set(
        active_range: RangeInclusive<f64>,
        get_set_value: impl 'a + FnMut(Option<f64>) -> f64,
    ) -> Self {
        Self {
            get_set_value: Box::new(get_set_value),
            full_range: active_range.clone(),
            active_range,
            focus_range: None,
            selection: None,
            gaps: &[],
            handle_image_id: None,
            handle_image_tint: get_scheme().success,
            max_handle_image_tint: get_scheme().success,
            handle_aspect_ratio: 0.5,
            segments: 12,
            label_font_size: 10.0,
            height: 40.0,
            width: 400.0,
        }
    }

    pub fn handle_image_id(mut self, image_id: egui::TextureId) -> Self {
        self.handle_image_id = Some(image_id);
        self
    }

    pub fn handle_aspect_ratio(mut self, handle_aspect_ratio: f32) -> Self {
        self.handle_aspect_ratio = handle_aspect_ratio;
        self
    }

    pub fn handle_image_tint(mut self, handle_image_tint: egui::Color32) -> Self {
        self.handle_image_tint = handle_image_tint;
        self
    }

    pub fn max_handle_image_tint(mut self, max_handle_image_tint: egui::Color32) -> Self {
        self.max_handle_image_tint = max_handle_image_tint;
        self
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub fn segments(mut self, segments: u8) -> Self {
        self.segments = segments;
        self
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    pub fn focus_range(mut self, range: Option<RangeInclusive<i64>>) -> Self {
        self.focus_range = range.map(|r| (*r.start() as f64)..=(*r.end() as f64));
        self
    }

    /// Playback region from shift-drag. Not the graph window: that one is
    /// [`Self::focus_range`].
    pub fn selection(mut self, selection: &'a mut Option<(i64, i64)>) -> Self {
        self.selection = Some(selection);
        self
    }

    pub fn gaps(mut self, gaps: &'a [(i64, i64)]) -> Self {
        self.gaps = gaps;
        self
    }

    fn get_value(&mut self) -> f64 {
        get(&mut self.get_set_value)
    }

    fn set_value(&mut self, value: f64) {
        set(&mut self.get_set_value, value);
    }

    fn range(&self) -> RangeInclusive<f64> {
        self.active_range.clone()
    }
}

/// The playback region after a plain seek to `pos`. Seeking outside the band
/// clears it (so it can be dismissed and the loop falls back to the whole
/// recording); seeking inside keeps it, so the playhead can be repositioned
/// within an active loop.
fn region_after_seek(region: Option<(i64, i64)>, pos: i64) -> Option<(i64, i64)> {
    match region {
        Some((start, end)) if pos < start || pos > end => None,
        other => other,
    }
}

impl Timeline<'_> {
    fn allocate_slider_space(&self, ui: &mut egui::Ui) -> egui::Response {
        ui.allocate_response(
            egui::emath::vec2(self.width, self.height),
            egui::Sense::drag(),
        )
    }

    fn render(&mut self, ui: &mut egui::Ui, response: &egui::Response) {
        let rect = response.rect.shrink2(egui::vec2(25.0, 0.0));

        let active_range_start = self.active_range.start();
        let active_range_end = self.active_range.end();
        let active_duration = hifitime::Duration::from_microseconds(
            self.active_range.end() - self.active_range.start(),
        );
        let full_duration = active_duration.segment_round();
        let segment_size = (full_duration / (self.segments) as f64).segment_round();
        self.segments =
            (full_duration.total_nanoseconds() / segment_size.total_nanoseconds().max(1)) as u8;
        let visual_segments = self.segments + 1;
        let segment_size = (segment_size.total_nanoseconds() / 1000) as f64;

        let full_duration_float = (full_duration.total_nanoseconds() / 1000) as f64;
        let position_range = get_position_range(
            rect.x_range(),
            active_range_end - active_range_start,
            full_duration_float,
        );

        let value = self.get_value();

        if let Some(pointer_position_2d) = response.interact_pointer_pos() {
            let position = pointer_position_2d.x;
            let aim_radius = ui.input(|i| i.aim_radius());
            let new_value = egui::emath::smart_aim::best_in_range_f64(
                value_from_position(position - aim_radius, self.range(), position_range),
                value_from_position(position + aim_radius, self.range(), position_range),
            );
            let shift = ui.input(|input| input.modifiers.shift);
            if shift {
                let id = response.id;
                let aimed = new_value.round() as i64;
                let start = if response.drag_started() {
                    ui.ctx().data_mut(|data| data.insert_temp(id, aimed));
                    aimed
                } else {
                    ui.ctx()
                        .data(|data| data.get_temp::<i64>(id))
                        .unwrap_or(aimed)
                };
                if let Some(selection) = self.selection.as_deref_mut() {
                    let (start, end) = if start <= aimed {
                        (start, aimed)
                    } else {
                        (aimed, start)
                    };
                    if end > start {
                        *selection = Some((start, end));
                    }
                }
            } else {
                self.set_value(new_value);
                // A plain seek away from the band dismisses it. Without a path
                // to clear the region, loop_bounds would keep preferring that
                // stale range over the full recording forever, even though the
                // loop button still advertises "Loop recording".
                if let Some(selection) = self.selection.as_deref_mut() {
                    *selection = region_after_seek(*selection, new_value.round() as i64);
                }
            }
        }
        self.full_range =
            *self.active_range.start()..=self.active_range.start() + full_duration_float;

        // Paint the UI
        if ui.is_rect_visible(response.rect) {
            // Default Styles

            let style = (*ui.style()).clone();
            let visuals = style.interact(response);

            // Trailing fill

            let max_value = *self.active_range.end();

            let max_position_1d =
                position_from_value(max_value, self.active_range.clone(), position_range);
            let max_center = Timeline::pointer_center(max_position_1d, &rect);

            ui.painter().rect_filled(
                rect.with_max_x(max_center.x).shrink4(Margin::ZERO.top(2.0)),
                CornerRadius::ZERO,
                get_scheme().bg_secondary,
            );

            // Rail

            ui.put(
                rect,
                self.rail_ui(
                    visual_segments.into(),
                    segment_size,
                    self.label_font_size,
                    position_range,
                ),
            );

            // Focus overlay
            if let Some(ref focus) = self.focus_range {
                let focus_start_x =
                    position_from_value(*focus.start(), self.active_range.clone(), position_range);
                let focus_end_x =
                    position_from_value(*focus.end(), self.active_range.clone(), position_range);
                let overlay_rect =
                    egui::Rect::from_x_y_ranges(focus_start_x..=focus_end_x, rect.y_range());
                ui.painter().rect_filled(
                    overlay_rect,
                    CornerRadius::ZERO,
                    get_scheme().success.opacity(0.12),
                );
                let edge_stroke = egui::Stroke::new(1.0_f32, get_scheme().success.opacity(0.5));
                ui.painter().line_segment(
                    [overlay_rect.left_top(), overlay_rect.left_bottom()],
                    edge_stroke,
                );
                ui.painter().line_segment(
                    [overlay_rect.right_top(), overlay_rect.right_bottom()],
                    edge_stroke,
                );
            }

            // Playback region. Blue, so it stays distinct from the green graph
            // window above.
            let drawn = self.selection.as_ref().and_then(|slot| **slot);
            if let Some((start, end)) = drawn {
                let start_x =
                    position_from_value(start as f64, self.active_range.clone(), position_range);
                let end_x =
                    position_from_value(end as f64, self.active_range.clone(), position_range);
                let overlay_rect = egui::Rect::from_x_y_ranges(start_x..=end_x, rect.y_range());
                ui.painter().rect_filled(
                    overlay_rect,
                    CornerRadius::ZERO,
                    get_scheme().blue.opacity(0.18),
                );
                let edge = egui::Stroke::new(1.0_f32, get_scheme().blue);
                ui.painter()
                    .line_segment([overlay_rect.left_top(), overlay_rect.left_bottom()], edge);
                ui.painter().line_segment(
                    [overlay_rect.right_top(), overlay_rect.right_bottom()],
                    edge,
                );
            }

            for &(start, end) in self.gaps {
                let start_x =
                    position_from_value(start as f64, self.active_range.clone(), position_range);
                let end_x =
                    position_from_value(end as f64, self.active_range.clone(), position_range);
                paint_discontinuity(ui, start_x, end_x, rect);
            }

            // Fixed Max Handle

            let handle_size = Timeline::get_handle_size(&rect, self.handle_aspect_ratio);
            let max_handle_rect = egui::Rect::from_center_size(max_center, handle_size);

            if let Some(image_id) = self.handle_image_id {
                ui.painter().image(
                    image_id,
                    max_handle_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    self.max_handle_image_tint,
                );
            }

            // Handle

            let position_1d = position_from_value(value, self.range(), position_range);
            let center = Timeline::pointer_center(position_1d, &rect);

            let handle_rect = egui::Rect::from_center_size(center, handle_size);

            if let Some(image_id) = self.handle_image_id {
                ui.painter().image(
                    image_id,
                    handle_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    self.handle_image_tint,
                );
            } else {
                ui.painter().rect(
                    handle_rect,
                    visuals.corner_radius,
                    visuals.bg_fill,
                    visuals.fg_stroke,
                    egui::StrokeKind::Inside,
                );
            }
        }
    }

    fn pointer_center(position_1d: f32, rail_rect: &egui::Rect) -> egui::Pos2 {
        egui::emath::pos2(position_1d, rail_rect.center().y)
    }

    fn rail_ui(
        &self,
        segments: usize,
        segment_size: f64,
        font_size: f32,
        position_range: egui::Rangef,
    ) -> impl egui::Widget + '_ {
        move |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.add_space(-15.0);
                let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                font_id.size = font_size;
                for i in 0..segments {
                    let offset_f64 = segment_size * i as f64;
                    let offset = PrettyDuration(hifitime::Duration::from_microseconds(offset_f64));
                    let position_1d = position_from_value(
                        offset_f64 + self.active_range.start(),
                        self.active_range.clone(),
                        position_range,
                    );
                    let segment_label = format!("{offset}");
                    let col_center_btm = egui::pos2(position_1d, ui.max_rect().bottom());
                    let col_center_top = egui::pos2(position_1d, ui.max_rect().top());
                    ui.painter().text(
                        egui::pos2(position_1d, ui.max_rect().center().y),
                        egui::Align2::CENTER_CENTER,
                        segment_label,
                        font_id.clone(),
                        get_scheme().text_secondary,
                    );

                    let top_point = egui::emath::pos2(
                        col_center_btm.x,
                        col_center_btm.y - ((col_center_btm.y - col_center_top.y) / 5.0),
                    );

                    ui.painter().line_segment(
                        [col_center_btm, top_point],
                        egui::Stroke::new(1.0_f32, get_scheme().border_primary),
                    );
                }
            })
            .response
        }
    }
}

fn paint_discontinuity(ui: &egui::Ui, start_x: f32, end_x: f32, rect: egui::Rect) {
    if end_x - start_x < 2.0 {
        return;
    }
    let mid_y = rect.center().y;
    let amplitude = (rect.height() * 0.22).max(2.0);
    let step = ((end_x - start_x) / 24.0).max(6.0);
    let mut points = Vec::new();
    let mut up = true;
    let mut x = start_x;
    while x < end_x {
        let y = if up {
            mid_y - amplitude
        } else {
            mid_y + amplitude
        };
        points.push(egui::pos2(x, y));
        up = !up;
        x += step;
    }
    points.push(egui::pos2(end_x, mid_y));
    ui.painter().add(egui::Shape::line(
        points,
        egui::Stroke::new(1.0_f32, get_scheme().text_secondary),
    ));
}

impl Timeline<'_> {
    fn get_handle_size(rect: &egui::Rect, aspect_ratio: f32) -> egui::Vec2 {
        let rect_height = rect.height();
        egui::vec2(rect_height * aspect_ratio, rect_height)
    }

    // Widget Wrapper

    fn add_contents(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let old_value = self.get_value();

        let mut response = self.allocate_slider_space(ui);

        self.render(ui, &response);

        let value = self.get_value();
        if value != old_value {
            response.mark_changed();
        }

        response
    }
}

impl egui::Widget for Timeline<'_> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let inner_response = ui.horizontal(|ui| self.add_contents(ui));
        inner_response.inner | inner_response.response
    }
}

#[derive(Resource, Deref, DerefMut, Clone, Debug, Default)]
pub struct UITick(pub i64);

#[derive(SystemParam)]
pub struct TimelineSlider<'w> {
    tick: ResMut<'w, UITick>,
    current_timestamp: ResMut<'w, CurrentTimestamp>,
    current_stream_id: Res<'w, CurrentStreamId>,
    tick_origin: ResMut<'w, StreamTickOrigin>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    latest_follow: ResMut<'w, LatestFollow>,
    auto_follow_latest_state: ResMut<'w, AutoFollowLatestState>,
    timeline_settings: Res<'w, TimelineSettings>,
    playback_region: ResMut<'w, PlaybackRegion>,
    discontinuities: Res<'w, PlaybackDiscontinuities>,
}

impl WidgetSystem for TimelineSlider<'_> {
    type Args = (TimelineIcons, TimelineArgs);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let TimelineSlider {
            mut tick,
            mut current_timestamp,
            current_stream_id,
            mut tick_origin,
            earliest_timestamp,
            mut latest_follow,
            mut auto_follow_latest_state,
            timeline_settings,
            mut playback_region,
            discontinuities,
        } = state.params_mut(world);

        tick_origin.observe_stream(**current_stream_id);

        let (icons, timeline_args) = args;
        let handle_icon = icons.handle;
        let playhead_color = timeline_settings.played_color.into_color32();
        let latest_color = timeline_settings.future_color.into_color32();

        let mut selection = playback_region.0.map(|(start, end)| (start.0, end.0));
        let response = ui
            .add(
                Timeline::new(
                    &mut tick.bypass_change_detection().0,
                    timeline_args.active_range,
                )
                .width(timeline_args.available_width)
                .height(timeline_args.line_height)
                .handle_image_id(handle_icon)
                .handle_image_tint(playhead_color)
                .max_handle_image_tint(latest_color)
                .handle_aspect_ratio(12.0 / 30.0)
                .segments(timeline_args.segment_count)
                .focus_range(timeline_args.focus_range)
                .selection(&mut selection)
                .gaps(&discontinuities.gaps),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand);

        let region_now = selection.map(|(start, end)| (Timestamp(start), Timestamp(end)));
        if region_now.map(|(start, end)| (start.0, end.0))
            != playback_region.0.map(|(s, e)| (s.0, e.0))
        {
            playback_region.0 = region_now;
            auto_follow_latest_state.cancel();
            latest_follow.0 = false;
        }

        if response.changed() {
            let target_timestamp = Timestamp(tick.0);
            auto_follow_latest_state.cancel();
            latest_follow.0 = false;
            current_timestamp.0 = target_timestamp;
            if target_timestamp <= earliest_timestamp.0 {
                tick_origin.request_rebase();
            }
        }
    }
}

pub fn sync_ui_tick(tick: Res<CurrentTimestamp>, mut ui_tick: ResMut<UITick>) {
    ui_tick.0 = tick.0.0;
}

#[cfg(test)]
mod tests {
    use super::region_after_seek;

    #[test]
    fn seeking_outside_the_band_clears_it_and_inside_keeps_it() {
        let region = Some((100, 200));
        assert_eq!(region_after_seek(region, 150), region, "inside: keep");
        assert_eq!(region_after_seek(region, 100), region, "on the edge: keep");
        assert_eq!(region_after_seek(region, 200), region, "on the edge: keep");
        assert_eq!(region_after_seek(region, 50), None, "before: dismiss");
        assert_eq!(region_after_seek(region, 250), None, "after: dismiss");
        assert_eq!(region_after_seek(None, 150), None, "nothing to clear");
    }
}
