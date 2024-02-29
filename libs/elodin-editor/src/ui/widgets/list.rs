use bevy::ecs::{entity::Entity, system::ResMut};
use bevy_egui::egui;
use conduit::{bevy::ComponentValueMap, well_known::WorldPos, EntityId};

use crate::ui::{colors, SelectedEntity};

pub fn entity_list(
    ui: &mut egui::Ui,
    entities: Vec<(Entity, &EntityId, &WorldPos, &ComponentValueMap)>,
    mut selected_entity: ResMut<SelectedEntity>,
) -> egui::Response {
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                egui::Frame::none()
                    .inner_margin(egui::Margin::symmetric(16.0, 16.0))
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new("ENTITIES").color(colors::WHITE))
                                .wrap(false),
                        );
                    });

                ui.separator();

                for (_, entity_id, _, _) in entities {
                    let selected = selected_entity.0.is_some_and(|id| id == *entity_id);
                    let list_item = ui.add(list_item(
                        selected,
                        format!("Untitled Entity - {}", entity_id.0),
                    ));

                    if list_item.clicked() {
                        selected_entity.0 = Some(*entity_id);
                    }
                }

                ui.allocate_space(ui.available_size());
            })
        })
        .inner
        .response
}

fn list_item_ui(ui: &mut egui::Ui, on: bool, label: String) -> egui::Response {
    let image_tint = colors::WHITE;
    let image_tint_click = colors::GREY_OPACITY_500;
    let background_color = if on { colors::WHITE } else { colors::STONE_950 };
    let text_color = if on { colors::STONE_950 } else { colors::WHITE };

    // Set widget size and allocate space
    let height_scale = 2.0;
    let desired_size = egui::vec2(
        ui.available_width(),
        ui.spacing().interact_size.y * height_scale,
    );
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    // Paint the UI
    if ui.is_rect_visible(rect) {
        let outer_margin = 1.0;
        let rect = rect.shrink(outer_margin);

        let style = ui.style_mut();
        let font_id = egui::TextStyle::Button.resolve(style);
        style.visuals.widgets.inactive.bg_fill = image_tint;
        style.visuals.widgets.hovered.bg_fill = image_tint;
        style.visuals.widgets.active.bg_fill = image_tint_click;

        let visuals = ui.style().interact(&response);

        // Background
        ui.painter()
            .rect(rect, visuals.rounding, background_color, visuals.bg_stroke);

        // Icon
        let left_center_pos = rect.left_center();
        let horizontal_margin = 20.0;
        let icon_side = 8.0;
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(left_center_pos.x + horizontal_margin, left_center_pos.y),
            egui::vec2(icon_side, icon_side),
        );
        ui.painter().rect(
            icon_rect,
            egui::Rounding::same(2.0),
            colors::GREEN_300,
            egui::Stroke::NONE,
        );

        // Label
        let left_text_margin = horizontal_margin + 12.0 + icon_side;
        let text = egui::RichText::new(label).color(text_color);

        let wrap_width = ui.available_width() - left_text_margin - horizontal_margin;
        let mut layout_job = egui::text::LayoutJob::single_section(
            text.text().to_string(),
            egui::TextFormat::simple(font_id, text_color),
        );
        layout_job.wrap.max_width = wrap_width;
        layout_job.wrap.max_rows = 1;
        layout_job.wrap.break_anywhere = true;

        let text_job = egui::widget_text::WidgetTextJob {
            job: layout_job,
            job_has_color: true,
        };
        let galley = ui.fonts(|f| f.layout_job(text_job.job));

        let text_rect = egui::Align2::LEFT_CENTER.anchor_rect(egui::Rect::from_min_size(
            egui::pos2(left_center_pos.x + left_text_margin, left_center_pos.y),
            galley.size(),
        ));
        ui.painter().galley(text_rect.min, galley);
    }

    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

pub fn list_item(on: bool, label: String) -> impl egui::Widget {
    move |ui: &mut egui::Ui| list_item_ui(ui, on, label)
}
