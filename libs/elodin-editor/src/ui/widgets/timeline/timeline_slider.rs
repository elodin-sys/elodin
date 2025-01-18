use std::ops::RangeInclusive;

use bevy::{
    ecs::{
        change_detection::DetectChangesMut,
        system::{ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::{Deref, DerefMut, Res, Resource},
};
use bevy_egui::egui;
use impeller2_bevy::{CurrentStreamId, PacketTx};
use impeller2_wkt::{SetStreamState, Tick};

use crate::ui::{colors, utils, widgets::WidgetSystem};

use super::{
    get_position_range, get_segment_size, position_from_value, value_from_position, TimelineArgs,
    TimelineIcons,
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
    active_range: RangeInclusive<f64>,
    clamp_to_range: bool,
    step: Option<f64>,
    fps: f64,
    handle_image_id: Option<egui::TextureId>,
    handle_image_tint: egui::Color32,
    handle_aspect_ratio: f32,
    segments: u8,
    label_font_size: f32,
    height: f32,
    width: f32,
    empty_bg: bool,
    trailing_fill: bool,
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
    ///         .end(frame_count as f64)
    ///         .fps(frames_per_second),
    /// );
    /// ```
    pub fn new<Num: egui::emath::Numeric>(
        value: &'a mut Num,
        active_range: RangeInclusive<Num>,
    ) -> Self {
        let range_f64 = active_range.start().to_f64()..=active_range.end().to_f64();
        let timeline = Self::from_get_set(range_f64, move |v: Option<f64>| {
            if let Some(v) = v {
                *value = Num::from_f64(v);
            }
            value.to_f64()
        });

        timeline
    }

    pub fn from_get_set(
        active_range: RangeInclusive<f64>,
        get_set_value: impl 'a + FnMut(Option<f64>) -> f64,
    ) -> Self {
        Self {
            get_set_value: Box::new(get_set_value),
            active_range,
            clamp_to_range: true,
            step: Some(1.0),
            handle_image_id: None,
            handle_image_tint: colors::MINT_DEFAULT,
            handle_aspect_ratio: 0.5,
            segments: 8,
            label_font_size: 10.0,
            fps: 60.0,
            height: 40.0,
            width: 400.0,
            empty_bg: false,
            trailing_fill: true,
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

    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    pub fn segments(mut self, segments: u8) -> Self {
        self.segments = segments;
        self
    }

    pub fn fps(mut self, fps: f64) -> Self {
        self.fps = fps;
        self
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    fn get_value(&mut self) -> f64 {
        let value = get(&mut self.get_set_value);
        if self.clamp_to_range {
            let start = *self.active_range.start();
            let end = *self.active_range.end();
            value.clamp(start.min(end), start.max(end))
        } else {
            value
        }
    }

    fn set_value(&mut self, mut value: f64) {
        if self.clamp_to_range {
            let start = *self.active_range.start();
            let end = *self.active_range.end();
            value = value.clamp(start.min(end), start.max(end));
        }
        if let Some(step) = self.step {
            value = (value / step).round() * step;
        }
        set(&mut self.get_set_value, value);
    }

    fn range(&self) -> RangeInclusive<f64> {
        self.active_range.clone()
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
        let rect = response.rect;

        let active_range_start = self.active_range.start();
        let active_range_end = self.active_range.end();
        let visual_segments = self.segments + 1;
        let segment_size = get_segment_size(self.fps, self.segments as f64, *active_range_end);
        let position_range = get_position_range(
            rect.x_range(),
            self.fps,
            self.segments as f64,
            segment_size as f64,
            *active_range_start,
            *active_range_end,
        );

        let value = self.get_value();

        if let Some(pointer_position_2d) = response.interact_pointer_pos() {
            let position = pointer_position_2d.x;
            let aim_radius = ui.input(|i| i.aim_radius());
            let new_value = egui::emath::smart_aim::best_in_range_f64(
                value_from_position(position - aim_radius, self.range(), position_range),
                value_from_position(position + aim_radius, self.range(), position_range),
            );

            self.set_value(new_value);
            response.changed();
        }

        // Paint the UI
        if ui.is_rect_visible(response.rect) {
            // Default Styles

            let style = (*ui.style()).clone();
            let visuals = style.interact(response);
            let widget_visuals = &style.visuals.widgets;

            // Trailing fill

            let max_value = self.active_range.end();
            let max_position_1d = position_from_value(*max_value, self.range(), position_range);
            let max_center = Timeline::pointer_center(max_position_1d, &rect);

            if self.trailing_fill {
                ui.painter().rect_filled(
                    rect.with_max_x(max_center.x),
                    widget_visuals.inactive.rounding,
                    colors::PRIMARY_ONYX,
                );
            }

            // Rail

            if self.empty_bg {
                ui.painter().rect_filled(
                    rect,
                    widget_visuals.inactive.rounding,
                    widget_visuals.inactive.bg_fill,
                );
            } else {
                ui.put(
                    rect,
                    self.rail_ui(visual_segments.into(), segment_size, self.label_font_size),
                );
            }

            // Fixed Max Handle

            let handle_size = Timeline::get_handle_size(&rect, self.handle_aspect_ratio);
            let max_handle_rect = egui::Rect::from_center_size(max_center, handle_size);

            if let Some(image_id) = self.handle_image_id {
                ui.painter().image(
                    image_id,
                    max_handle_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    colors::WHITE,
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
                    visuals.rounding,
                    visuals.bg_fill,
                    visuals.fg_stroke,
                );
            }
        }
    }

    fn pointer_center(position_1d: f32, rail_rect: &egui::Rect) -> egui::Pos2 {
        egui::emath::pos2(position_1d, rail_rect.center().y)
    }

    fn rail_ui(&self, segments: usize, segment_size: usize, font_size: f32) -> impl egui::Widget {
        move |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.columns(segments, |columns| {
                    for (i, column) in columns.iter_mut().enumerate().take(segments) {
                        let column_rect = column.max_rect();

                        // label

                        let segment_label = utils::time_label(segment_size * i, false);
                        let label_text = egui::RichText::new(segment_label)
                            .color(colors::PRIMARY_CREAME_6)
                            .size(font_size);

                        column.put(column_rect, egui::Label::new(label_text).selectable(false));

                        // center line

                        let col_center_btm = column_rect.center_bottom();
                        let col_center_top = column_rect.center_top();

                        let top_point = egui::emath::pos2(
                            col_center_btm.x,
                            col_center_btm.y - ((col_center_btm.y - col_center_top.y) / 5.0),
                        );

                        column.painter().line_segment(
                            [col_center_btm, top_point],
                            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_6),
                        );
                    }
                });
            })
            .response
        }
    }

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
        response.changed = value != old_value;

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
pub struct UITick(pub u64);

#[derive(SystemParam)]
pub struct TimelineSlider<'w> {
    event: Res<'w, PacketTx>,
    tick: ResMut<'w, UITick>,
    current_stream_id: Res<'w, CurrentStreamId>,
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
            event,
            mut tick,
            current_stream_id,
        } = state.get_mut(world);

        let (icons, timeline_args) = args;
        let handle_icon = icons.handle;

        ui.horizontal(|ui| {
            let response = ui
                .add(
                    Timeline::new(
                        &mut tick.bypass_change_detection().0,
                        timeline_args.active_range,
                    )
                    .width(timeline_args.available_width)
                    .height(timeline_args.line_height)
                    .handle_image_id(handle_icon)
                    .handle_aspect_ratio(12.0 / 30.0)
                    .segments(timeline_args.segment_count)
                    .fps(timeline_args.frames_per_second),
                )
                .on_hover_cursor(egui::CursorIcon::PointingHand);

            if response.changed() {
                event.send_msg(SetStreamState::rewind(**current_stream_id, tick.0))
            }
        });
    }
}

pub fn sync_ui_tick(tick: Res<Tick>, mut ui_tick: ResMut<UITick>) {
    ui_tick.0 = tick.0;
}
