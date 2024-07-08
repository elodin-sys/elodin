use bevy_egui::egui::{self, Stroke};

use crate::ui::{
    colors::{self, with_opacity},
    utils::Shrink4,
};

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct EImageButton {
    image_id: egui::TextureId,
    image_tint: egui::Color32,
    image_tint_click: egui::Color32,
    bg_color: egui::Color32,
    hovered_bg_color: egui::Color32,
    /// Multiplier for `ui.spacing().interact_size.y`
    width: f32,
    /// Multiplier for `ui.spacing().interact_size.y`
    height: f32,
}

impl EImageButton {
    pub fn new(image_id: egui::TextureId) -> Self {
        Self {
            image_id,
            image_tint: colors::WHITE,
            image_tint_click: colors::PRIMARY_ONYX_5,
            bg_color: colors::PRIMARY_SMOKE,
            width: 1.0,
            height: 1.0,
            hovered_bg_color: colors::PRIMARY_ONYX_9,
        }
    }

    pub fn scale(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn bg_color(mut self, bg_color: egui::Color32) -> Self {
        self.bg_color = bg_color;
        self
    }

    pub fn hovered_bg_color(mut self, bg_color: egui::Color32) -> Self {
        self.hovered_bg_color = bg_color;
        self
    }

    pub fn image_tint(mut self, image_tint: egui::Color32) -> Self {
        self.image_tint = image_tint;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // Set widget size and allocate space
        let min_interact_size = ui.spacing().interact_size.y;
        let (rect, response) = ui.allocate_exact_size(
            min_interact_size * egui::vec2(self.width, self.height),
            egui::Sense::click(),
        );

        // Paint the UI
        if ui.is_rect_visible(rect) {
            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_fill = self.image_tint;
            style.visuals.widgets.hovered.bg_fill = self.image_tint;
            style.visuals.widgets.active.bg_fill = self.image_tint_click;

            let visuals = ui.style().interact(&response);

            let bg_color = if response.hovered() || response.clicked() {
                self.hovered_bg_color
            } else {
                self.bg_color
            };

            // Background
            ui.painter()
                .rect(rect, egui::Rounding::same(3.0), bg_color, Stroke::NONE);

            // Icon
            let default_uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let image_rect = rect.shrink(3.0);

            ui.painter()
                .image(self.image_id, image_rect, default_uv, visuals.bg_fill);
        }

        response
    }
}

impl egui::Widget for EImageButton {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct ECheckboxButton {
    on_color: egui::Color32,
    off_color: egui::Color32,
    text_color: egui::Color32,
    margin: egui::Margin,
    is_on: bool,
    label: String,
    rounding: egui::Rounding,
}

impl ECheckboxButton {
    pub fn new(label: String, is_on: bool) -> Self {
        Self {
            on_color: colors::PRIMARY_CREAME,
            off_color: colors::PRIMARY_SMOKE,
            text_color: colors::PRIMARY_CREAME,
            margin: egui::Margin::same(8.0),
            is_on,
            label,
            rounding: egui::Rounding::same(2.0),
        }
    }

    pub fn on_color(mut self, color: egui::Color32) -> Self {
        self.on_color = color;
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let font_id = egui::TextStyle::Monospace.resolve(ui.style());

        // Set widget size and allocate space
        let galley =
            ui.painter()
                .layout_no_wrap(self.label.to_string(), font_id.clone(), self.text_color);
        let checkbox_side = galley.size().y;
        let spacing = checkbox_side * 0.5;
        let desired_size = egui::vec2(
            checkbox_side + spacing + galley.size().x + self.margin.sum().x,
            galley.size().y + self.margin.sum().y,
        );

        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, self.on_color);
            let visuals = ui.style().interact(&response);

            let inner_rect = rect.shrink4(self.margin);

            // Checkbox

            let checkbox_rect =
                egui::Rect::from_min_size(inner_rect.min, egui::Vec2::splat(checkbox_side));
            let fill_color = if self.is_on {
                self.on_color
            } else {
                self.off_color
            };

            ui.painter()
                .rect(checkbox_rect, self.rounding, fill_color, visuals.bg_stroke);

            // Label

            let label_rect_pos = egui::pos2(
                inner_rect.min.x + checkbox_side + spacing,
                inner_rect.center().y,
            );

            ui.painter().text(
                label_rect_pos,
                egui::Align2::LEFT_CENTER,
                self.label.to_string(),
                font_id,
                self.text_color,
            );
        }

        response
    }
}

impl egui::Widget for ECheckboxButton {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct EButton {
    label: String,
    disabled: bool,
    color: egui::Color32,
    bg_color: egui::Color32,
    margin: egui::Margin,
    rounding: egui::Rounding,
    stroke: egui::Stroke,
    width: Option<f32>,
}

impl EButton {
    pub fn new(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            disabled: false,
            color: colors::WHITE,
            bg_color: colors::PRIMARY_SMOKE,
            stroke: egui::Stroke::new(1.0, colors::WHITE),
            rounding: egui::Rounding::same(2.0),
            margin: egui::Margin::same(8.0),
            width: None,
        }
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    pub fn color(mut self, color: egui::Color32) -> Self {
        self.color = color;
        self
    }

    pub fn bg_color(mut self, color: egui::Color32) -> Self {
        self.bg_color = color;
        self
    }

    pub fn stroke(mut self, stroke: egui::Stroke) -> Self {
        self.stroke = stroke;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let font_id = egui::TextStyle::Button.resolve(ui.style());

        // Set widget size and allocate space
        let galley =
            ui.painter()
                .layout_no_wrap(self.label.to_string(), font_id.clone(), self.color);

        let desired_width = self.width.unwrap_or(ui.available_width());
        let desired_size = egui::vec2(desired_width, galley.size().y + self.margin.sum().y);
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            if self.disabled {
                ui.disable();
            }

            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_fill = self.bg_color;
            style.visuals.widgets.active.bg_fill = if self.disabled {
                self.bg_color
            } else {
                with_opacity(self.bg_color, 0.6)
            };
            style.visuals.widgets.hovered.bg_fill = if self.disabled {
                self.bg_color
            } else {
                with_opacity(self.bg_color, 0.9)
            };
            let visuals = ui.style().interact(&response);

            // Background

            ui.painter()
                .rect(rect, self.rounding, visuals.bg_fill, self.stroke);

            // Label

            let inner_rect = rect.shrink4(self.margin);
            ui.painter().text(
                inner_rect.center(),
                egui::Align2::CENTER_CENTER,
                self.label.to_string(),
                font_id,
                self.color,
            );
        }

        response
    }
}

impl egui::Widget for EButton {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui).on_hover_cursor(if self.disabled {
            egui::CursorIcon::Default
        } else {
            egui::CursorIcon::PointingHand
        })
    }
}

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct ETileButton {
    label: String,
    description: Option<String>,
    image_id: egui::TextureId,
    width: Option<f32>,
    height: Option<f32>,
}

impl ETileButton {
    pub fn new(label: impl ToString, image_id: egui::TextureId) -> Self {
        Self {
            label: label.to_string(),
            description: None,
            image_id,
            width: None,
            height: None,
        }
    }

    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height);
        self
    }

    pub fn description(mut self, description: impl ToString) -> Self {
        self.description = Some(description.to_string());
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // Set widget size and allocate space
        let desired_size = egui::vec2(
            self.width.unwrap_or(ui.available_width()),
            self.height.unwrap_or(ui.available_height()),
        );
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            let font_id = egui::TextStyle::Button.resolve(ui.style());

            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_fill = with_opacity(colors::PRIMARY_ONYX, 0.35);
            style.visuals.widgets.active.bg_fill = with_opacity(colors::PRIMARY_ONYX, 0.6);
            style.visuals.widgets.hovered.bg_fill = with_opacity(colors::PRIMARY_ONYX, 0.9);
            let visuals = ui.style().interact(&response);

            // Background

            ui.painter().rect(
                rect,
                egui::Rounding::same(1.0),
                visuals.bg_fill,
                egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
            );

            // Label

            let label_rect = ui.painter().text(
                egui::pos2(
                    rect.center().x,
                    rect.center().y + (ui.spacing().interact_size.y / 2.0),
                ),
                egui::Align2::CENTER_CENTER,
                self.label.to_uppercase(),
                font_id.clone(),
                colors::PRIMARY_CREAME,
            );

            let label_rect = label_rect.expand(8.0);

            // Image

            let default_uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let image_side = ui.spacing().interact_size.y;
            let image_rect = egui::Rect::from_center_size(
                egui::pos2(
                    label_rect.center_top().x,
                    label_rect.center_top().y - image_side,
                ),
                egui::vec2(image_side, image_side),
            );

            ui.painter().image(
                self.image_id,
                image_rect,
                default_uv,
                colors::PRIMARY_CREAME,
            );

            // Description

            if let Some(description) = &self.description {
                ui.painter().text(
                    label_rect.center_bottom(),
                    egui::Align2::CENTER_TOP,
                    description.to_string(),
                    font_id,
                    colors::PRIMARY_ONYX_6,
                );
            }
        }

        response
    }
}

impl egui::Widget for ETileButton {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}
