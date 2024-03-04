use bevy::{ecs::system::Res, log::warn};
use bevy_egui::egui;
use conduit::{query::MetadataStore, well_known::EntityMetadata, ComponentValue, TagValue};

use crate::ui::{colors, utils, EntityData};

pub fn inspector(
    ui: &mut egui::Ui,
    entity_metadata: Option<EntityMetadata>,
    entity_data: Option<EntityData>,
    metadata_store: &Res<MetadataStore>,
) -> egui::Response {
    egui::ScrollArea::vertical()
        .show(ui, |ui| {
            egui::Frame::none()
                .fill(colors::STONE_950)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        let separator_spacing = 32.0;
                        let label_spacing = 8.0;

                        let Some(metadata) = entity_metadata else {
                            ui.add(empty_inspector());
                            return;
                        };
                        let Some((entity_id, _, map)) = entity_data else {
                            ui.add(empty_inspector());
                            return;
                        };

                        let title_text =
                            egui::RichText::new(metadata.name).color(colors::ORANGE_50);

                        egui::Frame::none()
                            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(title_text).wrap(false));
                            });

                        ui.add(egui::Separator::default().spacing(separator_spacing));

                        let line_size =
                            egui::vec2(ui.available_size().x, ui.spacing().interact_size.y * 1.4);
                        ui.add(inspector_item_value("ID", entity_id.0, line_size));

                        for (id, value) in map.0.iter() {
                            match value {
                                ComponentValue::F64(arr_f64) => {
                                    let name = if let Some(name) = metadata_store
                                        .get_metadata(id)
                                        .and_then(|m| m.tags.get("name"))
                                        .and_then(TagValue::as_str)
                                    {
                                        name.to_string().to_uppercase()
                                    } else {
                                        format!("ID[{}]", id.0)
                                    };

                                    let values = arr_f64.iter().collect::<Vec<&f64>>();

                                    ui.add(egui::Separator::default().spacing(separator_spacing));

                                    inspector_item_multi(ui, name, values, label_spacing);
                                }
                                _ => {
                                    warn!("Unimplemented ComponentValue");
                                }
                            }
                        }
                    })
                })
        })
        .inner
        .response
}

fn empty_inspector_ui(ui: &mut egui::Ui) -> egui::Response {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            let text = egui::RichText::new("SELECT AN ENTITY")
                .color(colors::with_opacity(colors::WHITE, 0.1));
            ui.add(egui::Label::new(text));
        },
    )
    .response
}

pub fn empty_inspector() -> impl egui::Widget {
    move |ui: &mut egui::Ui| empty_inspector_ui(ui)
}

fn inspector_item_value_ui(
    ui: &mut egui::Ui,
    label: impl ToString,
    value: impl ToString,
    size: egui::Vec2,
) -> egui::Response {
    let size_label = egui::vec2(size.x * 0.3, size.y);
    let size_value = egui::vec2(size.x * 0.7, size.y);

    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());

    // Paint the UI
    if ui.is_rect_visible(rect) {
        let style = ui.style();
        let font_id = egui::TextStyle::Button.resolve(style);
        let label_color = colors::with_opacity(colors::ORANGE_50, 0.4);
        let value_color = colors::ORANGE_50;

        // Background
        ui.painter().rect(
            rect,
            egui::Rounding::ZERO,
            egui::Color32::TRANSPARENT,
            egui::Stroke::NONE,
        );

        // Label

        let layout_job =
            utils::get_galley_layout_job(label, size_label.x, font_id.clone(), label_color);
        let galley = ui.fonts(|f| f.layout_job(layout_job));
        let text_rect = egui::Align2::LEFT_CENTER
            .anchor_rect(egui::Rect::from_min_size(rect.left_center(), galley.size()));
        ui.painter().galley(text_rect.min, galley, label_color);

        // Value

        let layout_job = utils::get_galley_layout_job(value, size_value.x, font_id, value_color);
        let galley = ui.fonts(|f| f.layout_job(layout_job));
        let text_rect = egui::Align2::RIGHT_CENTER.anchor_rect(egui::Rect::from_min_size(
            rect.right_center(),
            galley.size(),
        ));
        ui.painter().galley(text_rect.min, galley, value_color);
    }

    response
}

pub fn inspector_item_value(
    label: impl ToString,
    value: impl ToString,
    size: egui::Vec2,
) -> impl egui::Widget {
    move |ui: &mut egui::Ui| inspector_item_value_ui(ui, label, value, size)
}

fn inspector_item_label_ui(ui: &mut egui::Ui, label: impl ToString) -> egui::Response {
    egui::Frame::none()
        .outer_margin(egui::Margin::same(6.0))
        .show(ui, |ui| {
            let desired_size = egui::vec2(ui.available_size().x, ui.spacing().interact_size.y);

            ui.allocate_ui_with_layout(
                desired_size,
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    let text = egui::RichText::new(label.to_string()).color(colors::ORANGE_50);
                    ui.add(egui::Label::new(text));
                },
            )
        })
        .response
}

pub fn inspector_item_label(label: impl ToString) -> impl egui::Widget {
    move |ui: &mut egui::Ui| inspector_item_label_ui(ui, label)
}

fn inspector_item_multi(
    ui: &mut egui::Ui,
    label: impl ToString,
    values: Vec<&f64>,
    label_spacing: f32,
) -> egui::Response {
    ui.vertical(|ui| {
        ui.add(inspector_item_label(label));

        ui.add_space(label_spacing);

        let item_spacing = egui::vec2(16.0, 0.0);

        let line_width = ui.available_size().x;
        let line_height = ui.spacing().interact_size.y * 1.4;

        let item_width_min = ui.spacing().interact_size.x * 2.2;
        let items_per_line = (line_width / item_width_min).floor();

        let necessary_spacing = (items_per_line - 1.0) * item_spacing.x;
        let item_width = (line_width - necessary_spacing) / items_per_line;

        let desired_size = egui::vec2(item_width - 1.0, line_height);

        ui.horizontal_wrapped(|ui| {
            ui.style_mut().spacing.item_spacing = item_spacing;

            for (i, value) in values.iter().enumerate() {
                let value_text = format!("{:.3}", value);
                ui.add(inspector_item_value(
                    format!("[{i}]"),
                    value_text,
                    desired_size,
                ));
            }
        });
    })
    .response
}
