use bevy_egui::egui::{self, epaint::util::FloatOrd};
use conduit::{ComponentId, ComponentValue};
use itertools::{Itertools, MinMaxResult};
use std::{
    collections::BTreeMap,
    ops::{Range, RangeInclusive},
};

use crate::{
    ui::{colors, utils, GraphState},
    CollectedGraphData,
};

#[derive(Debug, Clone)]
pub struct EPlotDataLine {
    pub label: String,
    pub values: Vec<f64>,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug)]
pub struct EPlotDataComponent {
    pub label: String,
    pub lines: BTreeMap<usize, EPlotDataLine>,
}

impl EPlotDataComponent {
    pub fn new(component_label: impl ToString) -> Self {
        Self {
            label: component_label.to_string(),
            lines: BTreeMap::new(),
        }
    }

    pub fn add_values(&mut self, component_value: &ComponentValue) {
        for (i, new_value) in component_value.iter().enumerate() {
            let new_value = new_value.as_f64();
            let line = self.lines.entry(i).or_insert_with(|| EPlotDataLine {
                label: format!("[{i}]"),
                values: Vec::new(),
                min: new_value,
                max: new_value,
            });
            line.values.push(new_value);
            if line.min > new_value {
                line.min = new_value;
            }
            if line.max < new_value {
                line.max = new_value;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EPlotDataEntity {
    pub label: String,
    pub components: BTreeMap<ComponentId, EPlotDataComponent>,
}

type EPlotComponentGroup = Vec<(String, Vec<EPlotLine>)>;
type EPlotEntityGroup = Vec<(String, EPlotComponentGroup)>;

#[derive(Debug)]
pub struct EPlot {
    tick_range: Range<u64>,
    lines: EPlotEntityGroup,
    bounds: EPlotBounds,
    rect: egui::Rect,
    inner_rect: egui::Rect,

    invert_x: bool,
    invert_y: bool,
    steps_x: usize,
    steps_y: usize,

    padding: egui::Margin,
    margin: egui::Margin,
    notch_length: f32,
    axis_label_margin: f32,

    fill_color: egui::Color32,
    text_color: egui::Color32,
    border_stroke: egui::Stroke,

    show_modal: bool,
    show_legend: bool,
}

fn range_x_from_rect(rect: &egui::Rect, invert: bool) -> RangeInclusive<f32> {
    if invert {
        rect.max.x..=rect.min.x
    } else {
        rect.min.x..=rect.max.x
    }
}

fn range_y_from_rect(rect: &egui::Rect, invert: bool) -> RangeInclusive<f32> {
    if invert {
        rect.max.y..=rect.min.y
    } else {
        rect.min.y..=rect.max.y
    }
}

impl Default for EPlot {
    fn default() -> Self {
        Self::new()
    }
}

impl EPlot {
    pub fn new() -> Self {
        Self {
            tick_range: Range::default(),
            lines: Vec::new(),
            bounds: EPlotBounds::default(),
            rect: egui::Rect::ZERO,
            inner_rect: egui::Rect::ZERO,

            invert_x: false,
            invert_y: true,
            steps_x: 6,
            steps_y: 6,

            padding: egui::Margin::same(20.0),
            margin: egui::Margin::same(60.0),
            notch_length: 20.0,
            axis_label_margin: 10.0,

            fill_color: colors::TRANSPARENT,
            text_color: colors::PRIMARY_CREAME,
            border_stroke: egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),

            show_modal: true,
            show_legend: false,
        }
    }

    /// Calculate bounds and point positions based on the current UI allocation
    /// Should be run last, right before render
    pub fn calculate_lines(
        mut self,
        ui: &egui::Ui,
        collected_graph_data: &CollectedGraphData,
        graph_state: &GraphState,
    ) -> Self {
        self.rect = ui.max_rect();
        self.inner_rect = self.get_inner_rect(self.rect);

        // calc max_length

        let width_in_px = ui.available_width() * ui.ctx().pixels_per_point();
        let max_length = width_in_px.ceil() as usize;
        let ticks_len = collected_graph_data
            .tick_range
            .start
            .saturating_sub(collected_graph_data.tick_range.end) as usize;
        let chunk_size = if ticks_len > max_length {
            ticks_len / max_length
        } else {
            1
        };

        self.tick_range = collected_graph_data.tick_range.clone();

        // calc bounds

        let mut minmax_lines = vec![];

        for (entity_id, components) in graph_state {
            if let Some(entity) = collected_graph_data.entities.get(entity_id) {
                for (component_id, component_values) in components {
                    if let Some(component) = entity.components.get(component_id) {
                        for (value_index, (enabled, _)) in component_values.iter().enumerate() {
                            if *enabled {
                                if let Some(line) = component.lines.get(&value_index) {
                                    minmax_lines.push(vec![line.min, line.max]);
                                }
                            }
                        }
                    }
                }
            }
        }

        self.bounds = EPlotBounds::from_lines(&self.tick_range, &minmax_lines);

        // calc lines

        for (entity_id, components) in graph_state {
            if let Some(entity) = collected_graph_data.entities.get(entity_id) {
                let mut component_group = Vec::new();

                for (component_id, component_values) in components {
                    if let Some(component) = entity.components.get(component_id) {
                        let mut component_group_lines = Vec::new();

                        for (value_index, (enabled, color)) in component_values.iter().enumerate() {
                            if *enabled {
                                if let Some(line) = component.lines.get(&value_index) {
                                    let value_chunks = line.values.chunks_exact(chunk_size);
                                    let values = value_chunks.into_iter().map(|values| {
                                        values.iter().sum::<f64>() / values.len() as f64
                                    });

                                    let label = format!("[{value_index}]");
                                    component_group_lines.push(EPlotLine::from_values(
                                        &self, chunk_size, values, label, *color,
                                    ));
                                }
                            }
                        }

                        component_group.push((component.label.to_owned(), component_group_lines));
                    }
                }

                self.lines.push((entity.label.to_owned(), component_group));
            }
        }

        self
    }

    pub fn invert(mut self, x: bool, y: bool) -> Self {
        self.invert_x = x;
        self.invert_y = y;
        self
    }

    pub fn padding(mut self, padding: egui::Margin) -> Self {
        self.padding = padding;
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    pub fn steps(mut self, x: usize, y: usize) -> Self {
        self.steps_x = x;
        self.steps_y = y;
        self
    }

    pub fn fill_color(mut self, color: egui::Color32) -> Self {
        self.fill_color = color;
        self
    }

    pub fn text_color(mut self, color: egui::Color32) -> Self {
        self.text_color = color;
        self
    }

    pub fn border_stroke(mut self, stroke: egui::Stroke) -> Self {
        self.border_stroke = stroke;
        self
    }

    fn draw_x_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let step_size = self.bounds.width() / self.steps_x as f64;
        let steps_x = (0..=self.steps_x)
            .map(|i| self.bounds.min_x + (i as f64) * step_size)
            .collect::<Vec<f64>>();

        for x_step in steps_x {
            let x_position = EPlotPoint::from_plot_point(self, x_step, self.bounds.min_y).pos2;

            ui.painter().line_segment(
                [
                    egui::pos2(x_position.x, x_position.y + self.padding.bottom),
                    egui::pos2(
                        x_position.x,
                        x_position.y + (self.padding.bottom + self.notch_length),
                    ),
                ],
                self.border_stroke,
            );

            ui.painter().text(
                egui::pos2(
                    x_position.x,
                    x_position.y
                        + (self.padding.bottom + self.notch_length + self.axis_label_margin),
                ),
                egui::Align2::CENTER_TOP,
                utils::time_label_ms(x_step),
                font_id.clone(),
                self.text_color,
            );
        }
    }

    fn draw_y_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let step_size = self.bounds.height() / self.steps_y as f64;
        let steps_y = (0..=self.steps_y)
            .map(|i| self.bounds.min_y + (i as f64) * step_size)
            .collect::<Vec<f64>>();

        for y_step in steps_y {
            let y_position = EPlotPoint::from_plot_point(self, self.bounds.min_x, y_step).pos2;

            ui.painter().line_segment(
                [
                    egui::pos2(y_position.x - self.padding.left, y_position.y),
                    egui::pos2(
                        y_position.x - (self.padding.left + self.notch_length),
                        y_position.y,
                    ),
                ],
                self.border_stroke,
            );

            ui.painter().text(
                egui::pos2(
                    y_position.x - (self.padding.left + self.notch_length + self.axis_label_margin),
                    y_position.y,
                ),
                egui::Align2::RIGHT_CENTER,
                format!("{:.2}", y_step),
                font_id.clone(),
                self.text_color,
            );
        }
    }

    fn draw_y_axis_flag(
        &self,
        ui: &mut egui::Ui,
        pointer_plot_point: EPlotPoint,
        border_rect: egui::Rect,
        font_id: egui::FontId,
    ) {
        let label_size = egui::vec2(50.0, 20.0);
        let label_margin = 4.0;

        let pointer_rect = egui::Rect::from_center_size(
            egui::pos2(
                border_rect.min.x - ((label_size.x / 2.0) + label_margin),
                pointer_plot_point.pos2.y,
            ),
            label_size,
        );

        ui.painter().rect(
            pointer_rect,
            egui::Rounding::same(2.0),
            colors::MINT_DEFAULT,
            egui::Stroke::NONE,
        );

        ui.painter().text(
            pointer_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.2}", pointer_plot_point.y),
            font_id,
            colors::PRIMARY_ONYX,
        );
    }

    fn draw_cursor(
        &self,
        ui: &mut egui::Ui,
        pointer_pos: egui::Pos2,
        closest_point: &EPlotPoint,
        border_rect: egui::Rect,
    ) {
        ui.painter().vline(
            closest_point.pos2.x,
            border_rect.min.y..=border_rect.max.y,
            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
        );

        // NOTE: HLine is not attached to points
        ui.painter().hline(
            border_rect.min.x..=border_rect.max.x,
            pointer_pos.y,
            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
        );
    }

    fn draw_modal(&self, ui: &mut egui::Ui, closest_point: &EPlotPoint, pointer_pos: egui::Pos2) {
        let inner_rect_size = self.inner_rect.size();
        let modal_min_size = egui::vec2(200.0, inner_rect_size.y);
        let modal_rect = egui::Rect::from_min_size(pointer_pos, modal_min_size);

        ui.allocate_ui_at_rect(modal_rect, |ui| {
            egui::Frame::none()
                .fill(colors::PRIMARY_SMOKE)
                .inner_margin(egui::Margin::same(10.0))
                .outer_margin(egui::Margin::same(20.0))
                .stroke(egui::Stroke::new(0.4, colors::PRIMARY_CREAME))
                .show(ui, |ui| {
                    let point_on_baseline = self
                        .tick_range
                        .clone()
                        .find_position(|x| *x as f64 == closest_point.x);

                    if let Some((index, value)) = point_on_baseline {
                        let time_text = utils::time_label_ms(value as f64);

                        ui.label(time_text);

                        for (entity_label, component_group) in &self.lines {
                            ui.separator();
                            ui.label(entity_label);

                            for (component_label, component_lines) in component_group {
                                ui.separator();
                                ui.label(component_label);
                                ui.separator();

                                for line in component_lines {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::Label::new(
                                            egui::RichText::new(line.label.to_owned())
                                                .color(line.color),
                                        ));
                                        ui.label(format!("   {:.2}", line.values[index].y));
                                    });
                                }
                            }
                        }
                    }
                });
        });
    }

    fn draw_legend(&self, ui: &mut egui::Ui) {
        let legend_size = egui::vec2(200.0, 50.0);
        let legend_margin = 10.0;

        let modal_rect = egui::Rect::from_min_max(
            egui::pos2(
                self.inner_rect.max.x - (legend_size.x + legend_margin),
                self.inner_rect.max.y - (legend_size.y + legend_margin),
            ),
            egui::pos2(
                self.inner_rect.max.x - legend_margin,
                self.inner_rect.max.y - legend_margin,
            ),
        );

        ui.painter().rect(
            modal_rect,
            egui::Rounding::same(2.0),
            colors::with_opacity(colors::PRIMARY_SMOKE, 0.9),
            egui::Stroke::new(0.4, colors::PRIMARY_CREAME),
        );

        // ui.allocate_ui_at_rect(modal_rect, |ui| {
        //     ui.horizontal(|ui| {
        //         for line in &self.lines {
        //             ui.add(egui::Label::new(
        //                 egui::RichText::new(line.label.to_owned()).color(line.color),
        //             ));
        //         }
        //     });
        // });
    }

    fn get_inner_rect(&self, rect: egui::Rect) -> egui::Rect {
        egui::Rect {
            min: egui::pos2(
                rect.min.x + (self.margin.left + self.padding.left),
                rect.min.y + (self.margin.top + self.padding.top),
            ),
            max: egui::pos2(
                rect.max.x - (self.margin.right + self.padding.right),
                rect.max.y - (self.margin.bottom + self.padding.bottom),
            ),
        }
    }

    fn get_border_rect(&self, rect: egui::Rect) -> egui::Rect {
        egui::Rect {
            min: egui::pos2(rect.min.x + self.margin.left, rect.min.y + self.margin.top),
            max: egui::pos2(
                rect.max.x - self.margin.right,
                rect.max.y - self.margin.bottom,
            ),
        }
    }

    pub fn render(&self, ui: &mut egui::Ui) {
        let _response = ui.allocate_rect(self.rect, egui::Sense::click_and_drag());
        let pointer_pos = ui.input(|i| i.pointer.latest_pos());

        let border_rect = self.get_border_rect(self.rect);

        // Style

        let style = ui.style();
        let font_id = egui::TextStyle::Button.resolve(style);

        // Draw inner container

        if self.lines.is_empty() {
            ui.painter().text(
                self.rect.center(),
                egui::Align2::CENTER_CENTER,
                "NO DATA POINTS SELECTED",
                font_id.clone(),
                colors::WHITE,
            );

            return;
        }

        ui.painter()
            .rect_filled(self.inner_rect, egui::Rounding::ZERO, self.fill_color);

        // Draw borders

        let left_border = [border_rect.left_top(), border_rect.left_bottom()];
        ui.painter().line_segment(left_border, self.border_stroke);

        let bottom_border = [border_rect.left_bottom(), border_rect.right_bottom()];
        ui.painter().line_segment(bottom_border, self.border_stroke);

        // Draw axis

        self.draw_x_axis(ui, &font_id);
        self.draw_y_axis(ui, &font_id);

        // Draw lines

        for (_entity_label, component_group) in &self.lines {
            for (_component_label, component_group_lines) in component_group {
                for line in component_group_lines {
                    line.draw(ui);

                    if let Some(pointer_pos) = pointer_pos {
                        if border_rect.contains(pointer_pos) {
                            line.draw_highlight(ui, &pointer_pos);
                        }
                    }
                }
            }
        }

        // Draw cursor

        if let Some(pointer_pos) = pointer_pos {
            if self.inner_rect.contains(pointer_pos) {
                let pointer_plot_point = EPlotPoint::from_plot_pos2(self, pointer_pos);

                self.draw_y_axis_flag(ui, pointer_plot_point, border_rect, font_id);
            }

            if border_rect.contains(pointer_pos) {
                if let Some(closest_point) = self
                    .lines
                    .first()
                    .and_then(|l| l.1.first())
                    .and_then(|l| l.1.first())
                    .and_then(|l| l.closest_by_x(&pointer_pos))
                {
                    self.draw_cursor(ui, pointer_pos, &closest_point, border_rect);

                    if self.show_modal {
                        self.draw_modal(ui, &closest_point, pointer_pos);
                    }
                }
            }
        }

        // Draw legend

        if self.show_legend {
            self.draw_legend(ui);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct EPlotBounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl EPlotBounds {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn from_lines(baseline: &Range<u64>, lines: &[Vec<f64>]) -> Self {
        let (min_x, max_x) = (baseline.end as f64, baseline.start as f64);

        let minmax_y = lines.iter().flatten().minmax();

        let (min_y, max_y) = match minmax_y {
            MinMaxResult::MinMax(min_y, max_y) => (*min_y, *max_y),
            MinMaxResult::OneElement(y) => (*y, *y),
            _ => (0.0, 0.0),
        };

        Self::new(min_x, min_y, max_x, max_y)
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn range_x_f32(&self) -> RangeInclusive<f32> {
        (self.min_x as f32)..=(self.max_x as f32)
    }

    pub fn range_y_f32(&self) -> RangeInclusive<f32> {
        (self.min_y as f32)..=(self.max_y as f32)
    }
}

#[derive(Debug, Clone)]
pub struct EPlotLine {
    values: Vec<EPlotPoint>,
    label: String,
    color: egui::Color32,
}

impl EPlotLine {
    pub fn new(values: Vec<EPlotPoint>, label: String, color: egui::Color32) -> Self {
        Self {
            values,
            label,
            color,
        }
    }

    pub fn from_values(
        plot: &EPlot,
        chunk_size: usize,
        values: impl Iterator<Item = f64>,
        label: String,
        color: egui::Color32,
    ) -> Self {
        Self::new(
            plot.tick_range
                .clone()
                .rev()
                .step_by(chunk_size)
                .zip(values)
                .map(|(x, y)| EPlotPoint::from_plot_point(plot, x as f64, y))
                .collect::<Vec<EPlotPoint>>(),
            label,
            color,
        )
    }

    pub fn pos2(&self) -> Vec<egui::Pos2> {
        self.values
            .iter()
            .map(|point| point.pos2)
            .collect::<Vec<egui::Pos2>>()
    }

    pub fn closest_by_x(&self, pos: &egui::Pos2) -> Option<EPlotPoint> {
        self.values
            .iter()
            .map(|point| (point.clone(), (point.pos2.x - pos.x).abs()))
            .min_by_key(|(_, distance)| distance.ord())
            .map(|(point, _)| point)
    }

    pub fn draw(&self, ui: &mut egui::Ui) -> egui::layers::ShapeIdx {
        ui.painter().add(egui::Shape::line(
            self.pos2(),
            egui::Stroke::new(2.0, self.color),
        ))
    }

    pub fn draw_highlight(&self, ui: &mut egui::Ui, pointer_pos: &egui::Pos2) {
        if let Some(closest_point) = self.closest_by_x(pointer_pos) {
            let closest_pos = closest_point.pos2;

            ui.painter().circle(
                closest_pos,
                6.0,
                colors::PRIMARY_SMOKE,
                egui::Stroke::new(2.0, self.color),
            );

            // ui.painter().text(
            //     egui::pos2(closest_pos.x + 10.0, closest_pos.y - 10.0),
            //     egui::Align2::LEFT_BOTTOM,
            //     format!("point: {:.2} x {:.2}", closest_point.x, closest_point.y),
            //     font_id,
            //     COLOR_ORANGE_50,
            // );
        }
    }

    // TODO: Simplify Line
}

#[derive(Debug, Clone)]
pub struct EPlotPoint {
    x: f64,
    y: f64,
    pos2: egui::Pos2,
}

impl EPlotPoint {
    pub fn new(x: f64, y: f64, pos2: egui::Pos2) -> Self {
        Self { x, y, pos2 }
    }

    pub fn from_plot_pos2(plot: &EPlot, pos: egui::Pos2) -> Self {
        Self::new(
            egui::remap(
                pos.x,
                range_x_from_rect(&plot.inner_rect, plot.invert_x),
                plot.bounds.range_x_f32(),
            ) as f64,
            egui::remap(
                pos.y,
                range_y_from_rect(&plot.inner_rect, plot.invert_y),
                plot.bounds.range_y_f32(),
            ) as f64,
            pos,
        )
    }

    pub fn from_plot_point(plot: &EPlot, x: f64, y: f64) -> Self {
        Self::new(x, y, Self::pos2(plot, x, y))
    }

    fn pos2(plot: &EPlot, x: f64, y: f64) -> egui::Pos2 {
        egui::pos2(
            egui::remap(
                x as f32,
                plot.bounds.range_x_f32(),
                range_x_from_rect(&plot.inner_rect, plot.invert_x),
            ),
            egui::remap(
                y as f32,
                plot.bounds.range_y_f32(),
                range_y_from_rect(&plot.inner_rect, plot.invert_y),
            ),
        )
    }
}
