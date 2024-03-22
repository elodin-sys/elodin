use bevy::{
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res},
    },
    hierarchy::BuildChildren,
    log::warn,
    math::{Mat3, Vec3},
    render::camera::Projection,
};
use bevy_egui::egui::{self, Align};
use big_space::GridCell;
use conduit::{
    bevy::ComponentValueMap, query::MetadataStore, well_known::EntityMetadata, ComponentId,
    ComponentValue, EntityId, TagValue,
};

use crate::{
    ui::{
        colors::{self, with_opacity},
        theme, utils, CameraQuery, CameraQueryItem, EntityData, SelectedObject,
    },
    MainCamera,
};

const SEPARATOR_SPACING: f32 = 32.0;
const LABEL_SPACING: f32 = 8.0;

#[allow(clippy::too_many_arguments)]
pub fn inspector(
    ui: &mut egui::Ui,
    selected_object: &SelectedObject,
    entities: &Query<EntityData>,
    metadata_store: &Res<MetadataStore>,
    camera_query: &mut Query<CameraQuery, With<MainCamera>>,
    commands: &mut Commands,
    entity_transform_query: &Query<&GridCell<i128>, Without<MainCamera>>,
) -> egui::Response {
    egui::ScrollArea::vertical()
        .show(ui, |ui| {
            egui::Frame::none()
                .fill(colors::STONE_950)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| match selected_object {
                        SelectedObject::None => {
                            ui.add(empty_inspector());
                        }
                        SelectedObject::Entity(pair) => {
                            let Ok((entity_id, _, map, metadata)) = entities.get(pair.bevy) else {
                                ui.add(empty_inspector());
                                return;
                            };
                            entity_inspector(ui, metadata, *entity_id, map, metadata_store);
                        }
                        SelectedObject::Viewport { camera, .. } => {
                            let Ok(cam) = camera_query.get_mut(*camera) else {
                                ui.add(empty_inspector());
                                return;
                            };
                            viewport_inspector(ui, entities, cam, commands, entity_transform_query);
                        }
                        SelectedObject::Graph { .. } => {}
                    })
                })
        })
        .inner
        .response
}

pub fn viewport_inspector(
    ui: &mut egui::Ui,
    entities_meta: &Query<EntityData>,
    mut cam: CameraQueryItem<'_>,
    commands: &mut Commands,
    entity_transform_query: &Query<&GridCell<i128>, Without<MainCamera>>,
) {
    title_ui(ui, "VIEWPORT");
    let before_parent = cam.parent.map(|p| p.get());
    let mut selected_parent: Option<Entity> = cam.parent.map(|p| p.get());
    let selected_name = selected_parent
        .and_then(|id| entities_meta.get(id).ok())
        .map(|(_, _, _, meta)| meta.name.clone())
        .unwrap_or_else(|| "NONE".to_string());

    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(8.0, 8.0))
        .show(ui, |ui| {
            ui.style_mut().spacing.combo_width = ui.available_size().x;
            ui.label(egui::RichText::new("TRACK ENTITY").color(with_opacity(colors::CREMA, 0.6)));
            ui.add_space(8.0);
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_source("TRACK ENTITY")
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());
                    ui.selectable_value(&mut selected_parent, None, "NONE");
                    let mut entities = entities_meta
                        .iter()
                        .filter(|(_, _, values, _)| {
                            values
                                .0
                                .contains_key(&ComponentId::new("world_pos:f64:[7]"))
                        })
                        .collect::<Vec<_>>();
                    entities.sort_by(|a, b| a.0.cmp(b.0));
                    for (_, id, values, meta) in entities {
                        if values
                            .0
                            .contains_key(&ComponentId::new("world_pos:f64:[7]"))
                        {
                            ui.selectable_value(&mut selected_parent, Some(id), meta.name.clone());
                        }
                    }
                });
        });
    let mut cam_entity = commands.entity(cam.entity);
    if before_parent != selected_parent {
        if let Some(entity) = selected_parent {
            if let Ok(entity_cell) = entity_transform_query.get(entity) {
                cam_entity.set_parent(entity);
                let rot_matrix = Mat3::from_quat(cam.transform.rotation);
                cam.transform.translation = rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, 10.0));
                *cam.grid_cell = *entity_cell;
            }
        } else {
            cam_entity.remove_parent();
            cam_entity.insert(cam.global_transform.compute_transform());
        }
    }
    if let Projection::Perspective(persp) = cam.projection.as_mut() {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                let mut fov = persp.fov.to_degrees();
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("FOV").color(with_opacity(colors::CREMA, 0.6)));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        if ui.add(egui::DragValue::new(&mut fov).speed(0.1)).changed() {
                            persp.fov = fov.to_radians();
                        }
                    });
                });
                ui.add_space(8.0);
                ui.style_mut().spacing.slider_width = ui.available_size().x;
                ui.style_mut().visuals.widgets.inactive.bg_fill = colors::ONYX;
                if ui
                    .add(egui::Slider::new(&mut fov, 5.0..=120.0).show_value(false))
                    .changed()
                {
                    persp.fov = fov.to_radians();
                }
            });
    }
}

fn title_ui(ui: &mut egui::Ui, title: impl ToString) {
    let title_text = egui::RichText::new(title.to_string()).color(colors::ORANGE_50);

    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(8.0, 8.0))
        .show(ui, |ui| {
            ui.add(egui::Label::new(title_text).wrap(false));
        });
    ui.add(egui::Separator::default().spacing(SEPARATOR_SPACING));
}

pub fn entity_inspector(
    ui: &mut egui::Ui,
    metadata: &EntityMetadata,
    entity_id: EntityId,
    map: &ComponentValueMap,
    metadata_store: &Res<MetadataStore>,
) {
    title_ui(ui, metadata.name.to_uppercase());

    let line_size = egui::vec2(ui.available_size().x, ui.spacing().interact_size.y * 1.4);
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

                ui.add(egui::Separator::default().spacing(SEPARATOR_SPACING));

                inspector_item_multi(ui, name, values, LABEL_SPACING);
            }
            _ => {
                warn!("Unimplemented ComponentValue");
            }
        }
    }
}

fn empty_inspector_ui(ui: &mut egui::Ui) -> egui::Response {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            let text = egui::RichText::new("SELECT AN ENTITY OR TABLE TO INSPECT")
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
