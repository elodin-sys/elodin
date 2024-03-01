use bevy::{ecs::system::Res, log::warn};
use bevy_egui::egui;
use conduit::{query::MetadataStore, well_known::EntityMetadata, ComponentValue, TagValue};

use crate::ui::{colors, EntityData};

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
                .inner_margin(egui::Margin::same(4.0))
                .show(ui, |ui| {
                    ui.vertical(|ui| {
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
                            .inner_margin(egui::Margin::symmetric(16.0, 16.0))
                            .show(ui, |ui| {
                                ui.add(egui::Label::new(title_text).wrap(false));
                            });

                        ui.separator();

                        ui.add(inspector_item("ID", entity_id.0));

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

                                    inspector_item_multi(ui, name, values);
                                }
                                _ => {
                                    warn!("Unimplemented ComponentValue");
                                }
                            }
                        }

                        ui.allocate_space(ui.available_size());
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

fn inspector_item_value_ui(ui: &mut egui::Ui, value: impl ToString) -> egui::Response {
    egui::Frame::none()
        .stroke(egui::Stroke::new(
            0.2,
            colors::with_opacity(colors::WHITE, 0.1),
        ))
        .inner_margin(egui::Margin::same(4.0))
        .outer_margin(egui::Margin::same(2.0))
        .show(ui, |ui| {
            let desired_size = egui::vec2(ui.available_size().x, ui.spacing().interact_size.y);

            ui.allocate_ui_with_layout(
                desired_size,
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    let text = egui::RichText::new(value.to_string()).color(colors::ORANGE_50);
                    ui.add(egui::Label::new(text));
                },
            )
        })
        .response
}

pub fn inspector_item_value(value: impl ToString) -> impl egui::Widget {
    move |ui: &mut egui::Ui| inspector_item_value_ui(ui, value)
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
                    let text = egui::RichText::new(label.to_string())
                        .color(colors::with_opacity(colors::ORANGE_50, 0.4));
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
) -> egui::Response {
    ui.vertical(|ui| {
        ui.add(inspector_item_label(label));

        let chunk_size = if (values.len() % 3) == 0 { 3 } else { 4 };
        let chunks = values.chunks(chunk_size);

        for chunk in chunks {
            let field_count = chunk.len();

            ui.columns(field_count, |columns| {
                for (i, column) in columns.iter_mut().enumerate().take(field_count) {
                    let value_text = format!("{:.3}", chunk[i]);
                    column.add(inspector_item_value(value_text));
                }
            })
        }
    })
    .response
}

fn inspector_item_ui(
    ui: &mut egui::Ui,
    label: impl ToString,
    value: impl ToString,
) -> egui::Response {
    ui.horizontal(|ui| {
        ui.columns(2, |columns| {
            columns[0].add(inspector_item_label(label));
            columns[1].add(inspector_item_value(value));
        })
    })
    .response
}

pub fn inspector_item(label: impl ToString, value: impl ToString) -> impl egui::Widget {
    move |ui: &mut egui::Ui| inspector_item_ui(ui, label, value)
}
