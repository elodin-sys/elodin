use bevy_egui::egui::{self, Color32, Rect, Response, TextureId, Widget};

use crate::ui::colors;

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct ImageButton {
    image_id: TextureId,
    image_tint: Color32,
    image_tint_click: Color32,
    background_color: Color32,
    /// Multiplier for `ui.spacing().interact_size.y`
    width: f32,
    /// Multiplier for `ui.spacing().interact_size.y`
    height: f32,
}

impl ImageButton {
    pub fn new(image_id: TextureId) -> Self {
        Self {
            image_id,
            image_tint: colors::WHITE,
            image_tint_click: colors::GREY_OPACITY_500,
            background_color: colors::STONE_950,
            width: 1.0,
            height: 1.0,
        }
    }

    pub fn scale(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    fn render(&mut self, ui: &mut egui::Ui) -> Response {
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

            // Background
            ui.painter().rect(
                rect,
                visuals.rounding,
                self.background_color,
                visuals.bg_stroke,
            );

            // Icon
            let default_uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let image_rect = rect.shrink(3.0);

            ui.painter()
                .image(self.image_id, image_rect, default_uv, visuals.bg_fill);
        }

        response
    }
}

impl Widget for ImageButton {
    fn ui(mut self, ui: &mut egui::Ui) -> Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}
