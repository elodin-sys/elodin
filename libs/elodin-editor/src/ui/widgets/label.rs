use crate::ui::{
    colors::{self, get_scheme},
    utils::{self, Shrink4},
};
use bevy_egui::egui;
use egui::UiBuilder;

use super::button::EImageButton;

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
    pub fn new(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            padding: egui::Margin::same(8),
            margin: egui::Margin::same(0),
            text_color: get_scheme().text_primary,
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

pub fn editable_label_with_buttons<const N: usize>(
    ui: &mut egui::Ui,
    btn_icons: [egui::TextureId; N],
    label: &mut String,
    color: egui::Color32,
    margin: egui::Margin,
) -> [bool; N] {
    let mut clicked = [false; N];

    egui::Frame::NONE.inner_margin(margin).show(ui, |ui| {
        ui.horizontal(|ui| {
            let (label_rect, btn_rect) = utils::get_rects_from_relative_width(
                ui.max_rect(),
                0.8,
                ui.spacing().interact_size.y,
            );

            ui.allocate_new_ui(UiBuilder::new().max_rect(label_rect), |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                    font_id.size = 12.0;
                    ui.add(egui::TextEdit::singleline(label).font(font_id).margin(8.0))
                });
            });

            ui.allocate_new_ui(UiBuilder::new().max_rect(btn_rect), |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for (i, btn_icon) in btn_icons.iter().enumerate() {
                        let btn = ui.add(
                            EImageButton::new(*btn_icon)
                                .scale(1.2, 1.2)
                                .image_tint(color)
                                .bg_color(colors::TRANSPARENT),
                        );

                        if btn.clicked() {
                            clicked[i] = true;
                        }
                    }
                });
            });
        });
    });

    clicked
}

pub fn label_with_buttons<const N: usize>(
    ui: &mut egui::Ui,
    btn_icons: [egui::TextureId; N],
    label: impl ToString,
    color: egui::Color32,
    margin: egui::Margin,
) -> [bool; N] {
    let mut clicked = [false; N];

    egui::Frame::NONE.inner_margin(margin).show(ui, |ui| {
        ui.horizontal(|ui| {
            let (label_rect, btn_rect) = utils::get_rects_from_relative_width(
                ui.max_rect(),
                0.8,
                ui.spacing().interact_size.y,
            );

            ui.allocate_new_ui(UiBuilder::new().max_rect(label_rect), |ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let text = egui::RichText::new(label.to_string()).color(color);
                    ui.add(egui::Label::new(text));
                });
            });

            ui.allocate_new_ui(UiBuilder::new().max_rect(btn_rect), |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for (i, btn_icon) in btn_icons.iter().enumerate() {
                        let btn = ui.add(
                            EImageButton::new(*btn_icon)
                                .scale(1.2, 1.2)
                                .image_tint(color)
                                .bg_color(colors::TRANSPARENT),
                        );

                        if btn.clicked() {
                            clicked[i] = true;
                        }
                    }
                });
            });
        });
    });

    clicked
}

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct EImageLabel {
    image_id: egui::TextureId,
    image_tint: egui::Color32,
    bg_color: egui::Color32,
    frame_size: egui::Vec2,
    margin: egui::Margin,
    corner_radius: egui::CornerRadius,
}

impl EImageLabel {
    pub fn new(image_id: egui::TextureId) -> Self {
        Self {
            image_id,
            image_tint: colors::WHITE,
            bg_color: get_scheme().bg_secondary,
            frame_size: egui::vec2(40.0, 40.0),
            margin: egui::Margin::same(2),
            corner_radius: egui::CornerRadius::same(4),
        }
    }

    pub fn frame_size(mut self, width: f32, height: f32) -> Self {
        self.frame_size = egui::vec2(width, height);
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    pub fn bg_color(mut self, bg_color: egui::Color32) -> Self {
        self.bg_color = bg_color;
        self
    }

    pub fn image_tint(mut self, image_tint: egui::Color32) -> Self {
        self.image_tint = image_tint;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // Set widget size and allocate space
        let (rect, response) = ui.allocate_exact_size(self.frame_size, egui::Sense::hover());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            // Background
            ui.painter().rect(
                rect,
                self.corner_radius,
                self.bg_color,
                egui::Stroke::NONE,
                egui::StrokeKind::Middle,
            );

            // Icon
            let default_uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let image_rect = rect.shrink4(self.margin);

            ui.painter()
                .image(self.image_id, image_rect, default_uv, self.image_tint);
        }

        response
    }
}

impl egui::Widget for EImageLabel {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
    }
}
