use crate::ui::{
    colors,
    utils::{self, Shrink4},
};
use bevy_egui::egui;

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct ELabel {
    label: String,
    padding: egui::Margin,
    margin: egui::Margin,
    text_color: egui::Color32,
    height: Option<f32>,
    bottom_stroke: Option<egui::Stroke>,
}

impl ELabel {
    pub const DEFAULT_STROKE: egui::Stroke = egui::Stroke {
        width: 1.0,
        color: colors::PRIMARY_ONYX_9,
    };

    pub fn new(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            padding: egui::Margin::same(8.0),
            margin: egui::Margin::same(0.0),
            text_color: colors::PRIMARY_CREAME,
            bottom_stroke: None,
            height: None,
        }
    }

    pub fn text_color(mut self, color: egui::Color32) -> Self {
        self.text_color = color;
        self
    }

    pub fn bottom_stroke(mut self, stroke: egui::Stroke) -> Self {
        self.bottom_stroke = Some(stroke);
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

    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height);
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let style = ui.style_mut();
        let font_id = egui::TextStyle::Button.resolve(style);

        let default_height = font_id.size + self.margin.sum().y + self.padding.sum().y;

        // Set widget size and allocate space
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), self.height.unwrap_or(default_height)),
            egui::Sense::click(),
        );

        let border_rect = rect.shrink4(self.margin);
        let text_rect = border_rect.shrink4(self.padding);

        // Paint the UI
        if ui.is_rect_visible(rect) {
            // Label

            let layout_job = utils::get_galley_layout_job(
                self.label.to_owned(),
                text_rect.width(),
                font_id,
                self.text_color,
            );
            let galley = ui.fonts(|f| f.layout_job(layout_job));
            let text_galley_rect = egui::Align2::LEFT_CENTER.anchor_rect(
                egui::Rect::from_min_size(text_rect.left_center(), galley.size()),
            );
            ui.painter()
                .galley(text_galley_rect.min, galley, self.text_color);

            // Bottom Stroke

            if let Some(border) = self.bottom_stroke {
                ui.painter().hline(
                    border_rect.min.x..=border_rect.max.x,
                    border_rect.max.y,
                    border,
                );
            }
        }

        response
    }
}

impl egui::Widget for ELabel {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
    }
}
