use std::ops::RangeInclusive;

use bevy_egui::egui::{
    self,
    emath::{self, Numeric},
    epaint::PathShape,
    load::SizedTexture,
    Image, Label, Pos2, Rangef, Rect, Response, RichText, Sense, TextureId, Vec2, Widget,
};

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
    range: RangeInclusive<f64>,

    clamp_to_range: bool,
    step: Option<f64>,

    handle_image_id: Option<TextureId>,
    handle_aspect_ratio: f32,
    segments: u8,
    label_font_size: f32,
    time: f64,
    height: f32,
    width: f32,
    empty_bg: bool,
}

impl<'a> Timeline<'a> {
    /// Creates a new timeline
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut current_frame = 17;
    /// let frame_count = 3600;
    /// let frames_per_second = 30.0;
    /// ui.add(
    ///     Timeline::new(&mut current_frame, 0..=frame_count)
    ///         .width(400.0)
    ///         .height(40.0)
    ///         .handle_aspect_ratio(12.0 / 30.0)
    ///         .segments(8)
    ///         .time(frame_count as f64 / frames_per_second),
    /// );
    /// ```
    pub fn new<Num: emath::Numeric>(value: &'a mut Num, range: RangeInclusive<Num>) -> Self {
        let range_f64 = range.start().to_f64()..=range.end().to_f64();
        let timeline = Self::from_get_set(range_f64, move |v: Option<f64>| {
            if let Some(v) = v {
                *value = Num::from_f64(v);
            }
            value.to_f64()
        });

        timeline
    }

    pub fn from_get_set(
        range: RangeInclusive<f64>,
        get_set_value: impl 'a + FnMut(Option<f64>) -> f64,
    ) -> Self {
        let default_time = range.end() / 30.0;
        Self {
            get_set_value: Box::new(get_set_value),
            range,
            clamp_to_range: true,
            step: Some(1.0),
            handle_image_id: None,
            handle_aspect_ratio: 0.5,
            segments: 8,
            label_font_size: 10.0,
            time: default_time,
            height: 40.0,
            width: 400.0,
            empty_bg: false,
        }
    }

    pub fn handle_image_id(mut self, image_id: &TextureId) -> Self {
        self.handle_image_id = Some(*image_id);
        self
    }

    pub fn handle_aspect_ratio(mut self, handle_aspect_ratio: f32) -> Self {
        self.handle_aspect_ratio = handle_aspect_ratio;
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

    pub fn time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    fn get_value(&mut self) -> f64 {
        let value = get(&mut self.get_set_value);
        if self.clamp_to_range {
            let start = *self.range.start();
            let end = *self.range.end();
            value.clamp(start.min(end), start.max(end))
        } else {
            value
        }
    }

    fn set_value(&mut self, mut value: f64) {
        if self.clamp_to_range {
            let start = *self.range.start();
            let end = *self.range.end();
            value = value.clamp(start.min(end), start.max(end));
        }
        if let Some(step) = self.step {
            value = (value / step).round() * step;
        }
        set(&mut self.get_set_value, value);
    }

    fn range(&self) -> RangeInclusive<f64> {
        self.range.clone()
    }

    /// Returns a `value` based on the `position` in the timeline
    ///
    /// # Arguments
    ///
    /// * `position` - A mouse position
    /// * `position_range` - A range of the timeline on the screen
    fn value_from_position(&self, position: f32, position_range: Rangef) -> f64 {
        let normalized = emath::remap_clamp(position, position_range, 0.0..=1.0) as f64;
        emath::lerp(self.range(), normalized.clamp(0.0, 1.0))
    }

    fn position_from_value(&self, value: f64, position_range: Rangef) -> f32 {
        let normalized = emath::remap_clamp(value, self.range(), 0.0..=1.0);
        emath::lerp(position_range, normalized as f32)
    }
}

impl<'a> Timeline<'a> {
    fn allocate_slider_space(&self, ui: &mut egui::Ui) -> Response {
        ui.allocate_response(emath::vec2(self.width, self.height), Sense::drag())
    }

    fn render(&mut self, ui: &mut egui::Ui, response: &Response) {
        let rect = response.rect;

        // NOTE: Active zone starts in a center of the first segment and ends in a the center of the last one
        let offset = rect.width() / (self.segments as f32 * 2.0);
        let position_range = Timeline::position_range(&rect, offset);

        if let Some(pointer_position_2d) = response.interact_pointer_pos() {
            let position = pointer_position_2d.x;
            let aim_radius = ui.input(|i| i.aim_radius());
            let new_value = emath::smart_aim::best_in_range_f64(
                self.value_from_position(position - aim_radius, position_range),
                self.value_from_position(position + aim_radius, position_range),
            );

            self.set_value(new_value);
        }

        // Paint the UI
        if ui.is_rect_visible(response.rect) {
            let value = self.get_value();

            // Default Styles

            let style = (*ui.style()).clone();
            let visuals = style.interact(response);
            let widget_visuals = &style.visuals.widgets;

            // Rail

            if self.empty_bg {
                ui.painter().rect_filled(
                    rect,
                    widget_visuals.inactive.rounding,
                    widget_visuals.inactive.bg_fill,
                );
            } else {
                ui.put(rect, self.rail_ui());
            }

            let position_1d = self.position_from_value(value, position_range);
            let center = Timeline::pointer_center(position_1d, &rect);

            // Handle

            let handle_size = Timeline::get_handle_size(&rect, self.handle_aspect_ratio);
            let handle_rect = Rect::from_center_size(center, handle_size);

            if let Some(image_id) = self.handle_image_id {
                ui.put(
                    handle_rect,
                    Image::new(SizedTexture::new(image_id, handle_size)),
                );
            } else {
                ui.painter().rect(
                    handle_rect,
                    visuals.rounding,
                    visuals.bg_fill,
                    visuals.fg_stroke,
                );
            }
        }
    }

    fn pointer_center(position_1d: f32, rail_rect: &Rect) -> Pos2 {
        emath::pos2(position_1d, rail_rect.center().y)
    }

    fn position_range(rect: &Rect, offset: f32) -> Rangef {
        rect.x_range().shrink(offset)
    }

    fn rail_ui(&self) -> impl Widget {
        let segments = self.segments as usize;
        let segment_size = (self.time / (segments.to_f64() - 1.0)).ceil() as usize;
        let label_font_size = self.label_font_size;

        fn time_label(time_in_seconds: usize) -> String {
            let seconds = time_in_seconds % 60;
            let minutes = (time_in_seconds / 60) % 60;
            format!("{minutes:0>2}:{seconds:0>2}")
        }

        move |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.columns(segments, |columns| {
                    for (i, column) in columns.iter_mut().enumerate().take(segments) {
                        let column_rect = column.max_rect();

                        // label

                        let segment_label = time_label(segment_size * i);
                        let label_text = RichText::new(segment_label).size(label_font_size);

                        column.put(column_rect, Label::new(label_text));

                        // center line

                        let col_center_btm = column_rect.center_bottom();
                        let col_center_top = column_rect.center_top();

                        let top_point = emath::pos2(
                            col_center_btm.x,
                            col_center_btm.y - ((col_center_btm.y - col_center_top.y) / 5.0),
                        );

                        column.painter().add(PathShape::line(
                            vec![col_center_btm, top_point],
                            column.style().visuals.widgets.noninteractive.bg_stroke,
                        ));
                    }
                });
            })
            .response
        }
    }

    fn get_handle_size(rect: &Rect, aspect_ratio: f32) -> Vec2 {
        let rect_height = rect.height();
        egui::vec2(rect_height * aspect_ratio, rect_height)
    }

    // Widget Wrapper

    fn add_contents(&mut self, ui: &mut egui::Ui) -> Response {
        let old_value = self.get_value();

        let mut response = self.allocate_slider_space(ui);

        self.render(ui, &response);

        let value = self.get_value();
        response.changed = value != old_value;

        response
    }
}

impl<'a> Widget for Timeline<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> Response {
        let inner_response = ui.horizontal(|ui| self.add_contents(ui));
        inner_response.inner | inner_response.response
    }
}
