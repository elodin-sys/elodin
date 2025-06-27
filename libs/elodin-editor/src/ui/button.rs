use bevy_egui::egui::{self, Stroke};
use egui::Color32;

use crate::ui::{
    colors::{ColorExt, get_scheme, with_opacity},
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
            image_tint: get_scheme().icon_primary,
            image_tint_click: get_scheme().icon_secondary,
            bg_color: get_scheme().bg_primary,
            width: 1.0,
            height: 1.0,
            hovered_bg_color: get_scheme().bg_secondary,
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
            ui.painter().rect(
                rect,
                egui::CornerRadius::same(3),
                bg_color,
                Stroke::NONE,
                egui::StrokeKind::Middle,
            );

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
    corner_radius: egui::CornerRadius,
    left_label: bool,
}

impl ECheckboxButton {
    pub fn new(label: impl ToString, is_on: bool) -> Self {
        Self {
            on_color: get_scheme().text_primary,
            off_color: get_scheme().bg_secondary,
            text_color: get_scheme().text_primary,
            margin: egui::Margin::same(8),
            is_on,
            label: label.to_string(),
            corner_radius: egui::CornerRadius::same(2),
            left_label: false,
        }
    }

    pub fn left_label(mut self, left_label: bool) -> Self {
        self.left_label = left_label;
        self
    }

    pub fn on_color(mut self, color: egui::Color32) -> Self {
        self.on_color = color;
        self
    }

    pub fn text_color(mut self, color: egui::Color32) -> Self {
        self.text_color = color;
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let font_id = egui::TextStyle::Button.resolve(ui.style());

        // Set widget size and allocate space
        let galley =
            ui.painter()
                .layout_no_wrap(self.label.to_string(), font_id.clone(), self.text_color);
        let checkbox_side = galley.size().y;
        let spacing = checkbox_side * 0.5;
        let desired_size = egui::vec2(
            if self.left_label {
                ui.available_width()
            } else {
                checkbox_side + spacing + galley.size().x + self.margin.sum().x
            },
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

            let (checkbox_rect, label_rect_pos) = if self.left_label {
                let label_rect_pos = egui::pos2(inner_rect.min.x, inner_rect.center().y);
                let checkbox_rect = egui::Rect::from_center_size(
                    inner_rect.right_center() - egui::vec2(checkbox_side + spacing, 0.0),
                    egui::Vec2::splat(checkbox_side),
                );
                (checkbox_rect, label_rect_pos)
            } else {
                let checkbox_rect =
                    egui::Rect::from_min_size(inner_rect.min, egui::Vec2::splat(checkbox_side));

                let label_rect_pos = egui::pos2(
                    inner_rect.min.x + checkbox_side + spacing,
                    inner_rect.center().y,
                );
                (checkbox_rect, label_rect_pos)
            };

            let fill_color = if self.is_on {
                self.on_color
            } else {
                self.off_color
            };

            ui.painter().rect(
                checkbox_rect,
                self.corner_radius,
                fill_color,
                visuals.bg_stroke,
                egui::StrokeKind::Middle,
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
pub struct EColorButton {
    color: egui::Color32,
    corner_radius: egui::CornerRadius,
}

impl EColorButton {
    pub fn new(color: egui::Color32) -> Self {
        Self {
            color,
            corner_radius: egui::CornerRadius::same(2),
        }
    }

    pub fn color(mut self, color: egui::Color32) -> Self {
        self.color = color;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let desired_size = egui::vec2(16.0, 16.0);

        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, self.color);
            let visuals = ui.style().interact(&response);

            let checkbox_rect = egui::Rect::from_min_size(rect.min, egui::Vec2::splat(16.0));

            ui.painter().rect(
                checkbox_rect,
                self.corner_radius,
                self.color,
                visuals.bg_stroke,
                egui::StrokeKind::Middle,
            );
        }

        response
    }
}

impl egui::Widget for EColorButton {
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
    corner_radius: egui::CornerRadius,
    stroke: egui::Stroke,
    width: Option<f32>,
    loading: bool,
}

impl EButton {
    pub fn new(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            disabled: false,
            color: get_scheme().text_primary,
            bg_color: get_scheme().bg_secondary,
            stroke: egui::Stroke::new(1.0, get_scheme().border_primary),
            corner_radius: egui::CornerRadius::same(2),
            margin: egui::Margin::same(8),
            width: None,
            loading: false,
        }
    }

    pub fn green(label: impl ToString) -> Self {
        EButton::new(label)
            .color(get_scheme().success)
            .bg_color(get_scheme().success.opacity(0.04))
            .stroke(Stroke::new(1.0, get_scheme().success.opacity(0.4)))
    }

    pub fn highlight(label: impl ToString) -> Self {
        EButton::new(label)
            .color(get_scheme().highlight)
            .bg_color(get_scheme().highlight.opacity(0.04))
            .stroke(Stroke::new(1.0, get_scheme().highlight.opacity(0.4)))
    }

    pub fn red(label: impl ToString) -> Self {
        EButton::new(label)
            .color(get_scheme().error)
            .bg_color(get_scheme().error.opacity(0.04))
            .stroke(Stroke::new(1.0, get_scheme().error))
    }

    pub fn gray(label: impl ToString) -> Self {
        EButton::new(label)
            .color(get_scheme().text_primary)
            .bg_color(Color32::TRANSPARENT)
            .stroke(Stroke::new(1.0, get_scheme().border_primary))
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

    pub fn loading(mut self, loading: bool) -> Self {
        self.loading = loading;
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

            ui.painter().rect(
                rect,
                self.corner_radius,
                visuals.bg_fill,
                self.stroke,
                egui::StrokeKind::Middle,
            );

            // Label
            //
            if self.loading {
                egui::Spinner::new().paint_at(
                    ui,
                    egui::Rect::from_center_size(rect.center(), egui::vec2(10., 10.)),
                );
                ui.disable();
            }

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
            style.visuals.widgets.inactive.bg_fill = with_opacity(get_scheme().bg_secondary, 0.35);
            style.visuals.widgets.active.bg_fill = with_opacity(get_scheme().bg_secondary, 0.6);
            style.visuals.widgets.hovered.bg_fill = with_opacity(get_scheme().bg_secondary, 0.9);
            let visuals = ui.style().interact(&response);

            // Background

            ui.painter().rect(
                rect,
                egui::CornerRadius::same(1),
                visuals.bg_fill,
                egui::Stroke::new(1.0, get_scheme().border_primary),
                egui::StrokeKind::Middle,
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
                get_scheme().text_primary,
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
                get_scheme().icon_primary,
            );

            // Description

            if let Some(description) = &self.description {
                ui.painter().text(
                    label_rect.center_bottom(),
                    egui::Align2::CENTER_TOP,
                    description.to_string(),
                    font_id,
                    get_scheme().text_secondary,
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
