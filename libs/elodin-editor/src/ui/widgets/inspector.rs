use bevy::{
    ecs::{
        entity::Entity,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    hierarchy::BuildChildren,
    math::{Mat3, Vec3},
    render::camera::Projection,
};
use bevy_egui::egui::{self, Align};
use big_space::propagation::NoPropagateRot;
use big_space::GridCell;
use conduit::{
    bevy::ComponentValueMap,
    query::MetadataStore,
    well_known::{EntityMetadata, WorldPos},
    Component, EntityId, GraphId,
};

use crate::{
    ui::{
        colors::{self, with_opacity},
        theme, tiles,
        utils::{self, MarginSides},
        widgets::button::ImageButton,
        CameraQuery, CameraQueryItem, EntityData, GraphsState, SelectedObject,
    },
    MainCamera,
};

use super::label::ELabel;

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
    graphs_state: &mut ResMut<GraphsState>,
    tile_state: &mut ResMut<tiles::TileState>,
    icon_chart: egui::TextureId,
) -> egui::Response {
    egui::ScrollArea::vertical()
        .show(ui, |ui| {
            egui::Frame::none()
                .fill(colors::PRIMARY_SMOKE)
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
                            entity_inspector(
                                ui,
                                metadata,
                                *entity_id,
                                map,
                                metadata_store,
                                graphs_state,
                                tile_state,
                                icon_chart,
                            );
                        }
                        SelectedObject::Viewport { camera, .. } => {
                            let Ok(cam) = camera_query.get_mut(*camera) else {
                                ui.add(empty_inspector());
                                return;
                            };
                            viewport_inspector(ui, entities, cam, commands, entity_transform_query);
                        }
                        SelectedObject::Graph {
                            label, graph_id, ..
                        } => {
                            graph_inspector(
                                ui,
                                graph_id,
                                label,
                                entities,
                                graphs_state,
                                metadata_store,
                            );
                        }
                    })
                })
        })
        .inner
        .response
}

pub fn graph_inspector(
    ui: &mut egui::Ui,
    graph_id: &GraphId,
    label: &str,
    entities: &Query<EntityData>,
    graphs_state: &mut ResMut<GraphsState>,
    metadata_store: &Res<MetadataStore>,
) {
    ui.add(
        ELabel::new(label)
            .padding(egui::Margin::same(8.0).bottom(24.0))
            .bottom_stroke(ELabel::DEFAULT_STROKE)
            .margin(egui::Margin::same(0.0).bottom(16.0)),
    );

    if ui.button("Add Component").clicked() {
        graphs_state.modal_graph = Some(*graph_id);
    }

    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(8.0, 8.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.style_mut().spacing.item_spacing = egui::vec2(8.0, 8.0);

                let ro_graphs_state = graphs_state.clone();

                let Some(graph_state) = ro_graphs_state.graphs.get(graph_id) else {
                    return;
                };

                for (entity_id, components) in graph_state {
                    let entity = entities.iter().find(|(eid, _, _, _)| *eid == entity_id);

                    if let Some((_, _, _, entity_metadata)) = entity {
                        ui.label(entity_metadata.name.to_string());

                        for (component_id, component_values) in components {
                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "  {}",
                                    utils::get_component_label(metadata_store, component_id)
                                ));
                                ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                                    if ui.button("-").clicked() {
                                        println!("remove {graph_id:?} / {entity_id:?} / {component_id:?}");
                                        graphs_state.remove_component(
                                            graph_id,
                                            entity_id,
                                            component_id,
                                        );
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                let mut new_component_values = Vec::new();
                                let mut clicked = false;

                                for (index, (enabled, color)) in component_values.iter().enumerate()
                                {
                                    let display_color = if *enabled { *color } else { colors::BLACK_BLACK_600 };
                                    let label = egui::RichText::new(format!("[{index}]"))   
                                        .color(display_color);

                                    if ui.button(label).clicked() {
                                        new_component_values.push((!*enabled, *color));

                                        clicked = true;
                                    }
                                    else {
                                        new_component_values.push((*enabled, *color));
                                    }
                                }

                                if clicked {
                                    graphs_state.insert_component(
                                        graph_id,
                                        entity_id,
                                        component_id,
                                        new_component_values,
                                    );
                                }
                            });
                        }
                    }
                }
            });
        });
}

pub fn viewport_inspector(
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
                        // if ui.add(egui::DragValue::new(&mut fov).speed(0.1)).changed() {
                        //     persp.fov = fov.to_radians();
                        // }
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

#[derive(Default)]
pub struct ItemActions {
    create_graph: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn entity_inspector(
    ui: &mut egui::Ui,
    metadata: &EntityMetadata,
    entity_id: EntityId,
    map: &ComponentValueMap,
    metadata_store: &Res<MetadataStore>,
    graphs_state: &mut ResMut<GraphsState>,
    tile_state: &mut ResMut<tiles::TileState>,
    icon_chart: egui::TextureId,
) {
    ui.add(
        ELabel::new(&metadata.name)
            .padding(egui::Margin::same(8.0).bottom(24.0))
            .bottom_stroke(ELabel::DEFAULT_STROKE)
            .margin(egui::Margin::same(0.0).bottom(16.0)),
    );

    let line_size = egui::vec2(ui.available_size().x, ui.spacing().interact_size.y * 1.4);
    ui.add(inspector_item_value("ID", entity_id.0, line_size));

    for (component_id, component_value) in map.0.iter() {
        let values = utils::component_value_to_vec(component_value);
        let label = utils::get_component_label(metadata_store, component_id);

        ui.add(egui::Separator::default().spacing(SEPARATOR_SPACING));

        let mut item_actions = ItemActions::default();

        inspector_item_multi(
            ui,
            label,
            &values,
            LABEL_SPACING,
            icon_chart,
            &mut item_actions,
        );

        if item_actions.create_graph {
            let (graph_id, _) = graphs_state.get_or_create_graph(&None);
            let component_values = values
                .iter()
                .enumerate()
                .map(|_| (true, colors::get_random_color()))
                .collect::<Vec<(bool, egui::Color32)>>();

            graphs_state.insert_component(&graph_id, &entity_id, component_id, component_values);

            tile_state.create_graph_tile(graph_id);
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
        let label_color = colors::with_opacity(colors::PRIMARY_CREAME, 0.4);
        let value_color = colors::PRIMARY_CREAME;

        // Background
        ui.painter().rect(
            rect,
            egui::Rounding::ZERO,
            colors::TRANSPARENT,
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

fn inspector_item_label_ui(
    ui: &mut egui::Ui,
    label: impl ToString,
    icon_chart: egui::TextureId,
    item_actions: &mut ItemActions,
) -> egui::Response {
    egui::Frame::none()
        .outer_margin(egui::Margin::same(6.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (label_rect, btn_rect) = utils::get_rects_from_relative_width(
                    ui.max_rect(),
                    0.8,
                    ui.spacing().interact_size.y,
                );

                ui.allocate_ui_at_rect(label_rect, |ui| {
                    let text = egui::RichText::new(label.to_string()).color(colors::PRIMARY_CREAME);
                    ui.add(egui::Label::new(text));
                });

                ui.allocate_ui_at_rect(btn_rect, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let create_graph_btn = ui.add(
                            ImageButton::new(icon_chart)
                                .scale(1.1, 1.1)
                                .image_tint(colors::PRIMARY_CREAME)
                                .bg_color(colors::TRANSPARENT),
                        );

                        item_actions.create_graph = create_graph_btn.clicked();
                    });
                });
            })
        })
        .response
}

pub fn inspector_item_label<'a>(
    label: impl ToString + 'a,
    icon_chart: egui::TextureId,
    item_actions: &'a mut ItemActions,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| inspector_item_label_ui(ui, label, icon_chart, item_actions)
}

fn inspector_item_multi(
    ui: &mut egui::Ui,
    label: impl ToString,
    values: &[f64],
    label_spacing: f32,
    icon_chart: egui::TextureId,
    item_actions: &mut ItemActions,
) -> egui::Response {
    ui.vertical(|ui| {
        ui.add(inspector_item_label(label, icon_chart, item_actions));

        ui.add_space(label_spacing);

        let item_spacing = egui::vec2(16.0, 0.0);

        let line_width = ui.available_size().x;
        let line_height = ui.spacing().interact_size.y * 1.4;

        let mut has_long_value = false;
        let mut label_value_list = Vec::new();
        for (i, value) in values.iter().enumerate() {
            let label_text = format!("[{i}]");
            let value_text = format!("{:.3}", value);

            if value_text.len() > 6 {
                has_long_value = true;
            }

            label_value_list.push((label_text, value_text));
        }

        let items_per_line = if has_long_value {
            1.0
        } else {
            let item_width_min = ui.spacing().interact_size.x * 2.2;
            (line_width / item_width_min).floor()
        };

        let necessary_spacing = (items_per_line - 1.0) * item_spacing.x;
        let item_width = (line_width - necessary_spacing) / items_per_line;

        let desired_size = egui::vec2(item_width - 1.0, line_height);

        ui.horizontal_wrapped(|ui| {
            ui.style_mut().spacing.item_spacing = item_spacing;

            for (label, value) in label_value_list {
                ui.add(inspector_item_value(label, value, desired_size));
            }
        });
    })
    .response
}
