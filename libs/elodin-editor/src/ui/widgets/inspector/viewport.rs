use bevy::{
    ecs::{
        entity::Entity,
        query::Without,
        system::{Commands, Query},
    },
    hierarchy::BuildChildren,
    math::{Mat3, Vec3},
    render::camera::Projection,
};
use bevy_egui::egui::{self, Align};
use big_space::propagation::NoPropagateRot;
use big_space::GridCell;
use conduit::{well_known::WorldPos, Component};

use crate::{
    ui::{
        colors::{self, with_opacity},
        theme,
        utils::MarginSides,
        widgets::label::ELabel,
        CameraQueryItem, EntityData,
    },
    MainCamera,
};

pub fn inspector(
    ui: &mut egui::Ui,
    entities_meta: &Query<EntityData>,
    mut cam: CameraQueryItem<'_>,
    commands: &mut Commands,
    entity_transform_query: &Query<&GridCell<i128>, Without<MainCamera>>,
) {
    ui.add(
        ELabel::new("Viewport")
            .padding(egui::Margin::same(8.0).bottom(24.0))
            .bottom_stroke(ELabel::DEFAULT_STROKE)
            .margin(egui::Margin::same(0.0).bottom(16.0)),
    );

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
            ui.label(
                egui::RichText::new("TRACK ENTITY")
                    .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
            );
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
                            values.0.contains_key(&WorldPos::component_id())
                        })
                        .collect::<Vec<_>>();
                    entities.sort_by(|a, b| a.0.cmp(b.0));
                    for (_, id, values, meta) in entities {
                        if values.0.contains_key(&WorldPos::component_id()) {
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

    if selected_parent.is_some() {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("TRACK ROTATION")
                            .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        let mut track_rotation = cam.no_propagate_rot.is_none();
                        ui.checkbox(&mut track_rotation, "");

                        if track_rotation != cam.no_propagate_rot.is_none() {
                            if track_rotation {
                                cam_entity.remove::<NoPropagateRot>();
                            } else {
                                cam_entity.insert(NoPropagateRot);
                            }
                        }
                    });
                });
            });
    }

    if let Projection::Perspective(persp) = cam.projection.as_mut() {
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                let mut fov = persp.fov.to_degrees();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("FOV").color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        if ui.add(egui::DragValue::new(&mut fov).speed(0.1)).changed() {
                            persp.fov = fov.to_radians();
                        }
                    });
                });
                ui.add_space(8.0);
                ui.style_mut().spacing.slider_width = ui.available_size().x;
                ui.style_mut().visuals.widgets.inactive.bg_fill = colors::PRIMARY_ONYX_8;
                if ui
                    .add(egui::Slider::new(&mut fov, 5.0..=120.0).show_value(false))
                    .changed()
                {
                    persp.fov = fov.to_radians();
                }
            });
    }
}
