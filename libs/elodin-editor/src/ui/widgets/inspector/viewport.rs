use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy::{
    ecs::{
        entity::Entity,
        query::Without,
        system::{Commands, Query},
    },
    hierarchy::BuildChildren,
    math::Vec3,
    render::{camera::Projection, view::Visibility},
};
use bevy_egui::egui::{self, Align};
use bevy_infinite_grid::InfiniteGrid;
use big_space::GridCell;
use big_space::propagation::NoPropagateRot;
use impeller2::component::Component;
use impeller2_wkt::WorldPos;

use crate::ui::CameraQuery;
use crate::ui::widgets::WidgetSystem;
use crate::ui::widgets::label::label_with_buttons;
use crate::{
    GridHandle, MainCamera,
    ui::{
        EntityData,
        colors::{self, with_opacity},
        theme,
        utils::MarginSides,
        widgets::label::ELabel,
    },
};

use super::{InspectorIcons, empty_inspector};

#[derive(SystemParam)]
pub struct InspectorViewport<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    commands: Commands<'w, 's>,
    entity_transform_query: Query<'w, 's, &'static GridCell<i128>, Without<MainCamera>>,
    grid_visibility: Query<'w, 's, &'static mut Visibility, With<InfiniteGrid>>,
}

impl WidgetSystem for InspectorViewport<'_, '_> {
    type Args = (InspectorIcons, Entity);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, camera) = args;

        let entities_meta = state_mut.entities;
        let mut camera_query = state_mut.camera_query;
        let mut commands = state_mut.commands;
        let entity_transform_query = state_mut.entity_transform_query;
        let mut grid_visibility = state_mut.grid_visibility;

        let Ok(mut cam) = camera_query.get_mut(camera) else {
            ui.add(empty_inspector());
            return;
        };

        ui.add(
            ELabel::new("Viewport")
                .padding(egui::Margin::same(8).bottom(24.0))
                .bottom_stroke(ELabel::DEFAULT_STROKE)
                .margin(egui::Margin::same(0).bottom(16.0)),
        );

        let before_parent = cam.parent.map(|p| p.get());
        let mut selected_parent: Option<Entity> = cam.parent.map(|p| p.get());
        let selected_name = selected_parent
            .and_then(|id| entities_meta.get(id).ok())
            .map(|(_, _, _, meta)| meta.name.clone())
            .unwrap_or_else(|| "NONE".to_string());

        let mut reset_focus = false;

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                ui.style_mut().spacing.combo_width = ui.available_size().x;

                reset_focus = label_with_buttons(
                    ui,
                    [icons.search],
                    "TRACK ENTITY",
                    colors::PRIMARY_CREAME,
                    egui::Margin::same(0).bottom(8.0),
                )[0];

                theme::configure_combo_box(ui.style_mut());
                egui::ComboBox::from_id_salt("TRACK ENTITY")
                    .selected_text(selected_name)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());
                        ui.selectable_value(&mut selected_parent, None, "NONE");
                        let mut entities = entities_meta
                            .iter()
                            .filter(|(_, _, values, _)| {
                                values.0.contains_key(&WorldPos::COMPONENT_ID)
                            })
                            .collect::<Vec<_>>();
                        entities.sort_by(|a, b| a.0.cmp(b.0));
                        for (_, id, values, meta) in entities {
                            if values.0.contains_key(&WorldPos::COMPONENT_ID) {
                                ui.selectable_value(
                                    &mut selected_parent,
                                    Some(id),
                                    meta.name.clone(),
                                );
                            }
                        }
                    });
            });
        let mut cam_entity = commands.entity(cam.entity);
        if before_parent != selected_parent || reset_focus {
            if let Some(entity) = selected_parent {
                if let Ok(entity_cell) = entity_transform_query.get(entity) {
                    cam_entity.set_parent(entity);
                    cam.transform.translation = Vec3::new(10.0, 0.0, 0.0);
                    cam.transform.look_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
                    *cam.grid_cell = *entity_cell;
                }
            } else {
                cam_entity.remove_parent();
                cam_entity.insert(cam.global_transform.compute_transform());
            }
        }

        if selected_parent.is_some() {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("TRACK ROTATION")
                                .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                        );
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut track_rotation = cam.no_propagate_rot.is_none();
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut track_rotation, "");

                            if track_rotation != cam.no_propagate_rot.is_none() {
                                if track_rotation {
                                    cam_entity.remove::<NoPropagateRot>();
                                } else {
                                    cam_entity.try_insert(NoPropagateRot);
                                }
                            }
                        });
                    });
                });
        }

        if let Projection::Perspective(persp) = cam.projection.as_mut() {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    let mut fov = persp.fov.to_degrees();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("FOV")
                                .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
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

        if let Some(&GridHandle { grid }) = cam.grid_handle {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("SHOW GRID")
                                .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                        );
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut visibility = grid_visibility.get_mut(grid).unwrap();
                            let mut visible = *visibility == Visibility::Visible;
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut visible, "");
                            if visible {
                                *visibility = Visibility::Visible;
                            } else {
                                *visibility = Visibility::Hidden;
                            }
                        });
                    });
                });
        }
    }
}
