use bevy::prelude::Entity;
use bevy::{
    ecs::{
        system::{Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::*,
    scene::SceneRoot,
};
use bevy_egui::egui::{self, Align, RichText};
use impeller2_wkt::{ComponentMetadata, Material, Mesh, Object3DMesh};
use smallvec::SmallVec;

use crate::object_3d::spawn_mesh;
use crate::ui::inspector::{eql_textfield, node_color_picker};
use crate::{
    object_3d::Object3DState,
    ui::{
        colors::get_scheme,
        label,
        theme::{self, configure_combo_box, configure_input_with_border},
        tiles::TreeAction,
        utils::MarginSides,
        widgets::WidgetSystem,
    },
};

use super::{InspectorIcons, empty_inspector, inspector_text_field};

#[derive(SystemParam)]
pub struct InspectorObject3D<'w, 's> {
    object_3d: Query<'w, 's, &'static mut Object3DState>,
    metadata_query: Query<'w, 's, &'static ComponentMetadata>,
    eql_context: ResMut<'w, crate::EqlContext>,
    commands: Commands<'w, 's>,
    material_assets: ResMut<'w, Assets<StandardMaterial>>,
    mesh_assets: ResMut<'w, Assets<bevy::prelude::Mesh>>,
    assets: Res<'w, AssetServer>,
}

impl WidgetSystem for InspectorObject3D<'_, '_> {
    type Args = (InspectorIcons, Entity);
    type Output = SmallVec<[TreeAction; 4]>;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let tree_actions = SmallVec::new();
        let InspectorObject3D {
            mut object_3d,
            metadata_query,
            eql_context,
            mut commands,
            assets,
            mut material_assets,
            mut mesh_assets,
        } = state.get_mut(world);

        let (_icons, entity) = args;

        let Ok(mut object_3d_state) = object_3d.get_mut(entity) else {
            ui.add(empty_inspector());
            return tree_actions;
        };

        let metadata = metadata_query.get(entity).ok();
        let object_name = metadata.map(|m| m.name.as_str()).unwrap_or("Object3D");

        let mono_font = egui::TextStyle::Monospace.resolve(ui.style_mut());

        ui.spacing_mut().item_spacing.y = 8.0;

        ui.horizontal(|ui| {
            ui.add(
                label::ELabel::new(object_name)
                    .padding(egui::Margin::same(0).bottom(24.))
                    .bottom_stroke(egui::Stroke {
                        width: 1.0,
                        color: get_scheme().border_primary,
                    })
                    .margin(egui::Margin::same(0).bottom(8.)),
            );

            ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                ui.label(
                    RichText::new(format!("{:?}", entity))
                        .color(get_scheme().text_primary)
                        .font(mono_font.clone()),
                );
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Entity")
                        .color(get_scheme().text_secondary)
                        .font(mono_font.clone()),
                );
            });
        });

        egui::Frame::NONE.show(ui, |ui| {
            ui.label(egui::RichText::new("EQL Expression").color(get_scheme().text_secondary));
            configure_input_with_border(ui.style_mut());

            let query_res = eql_textfield(ui, true, &eql_context.0, &mut object_3d_state.data.eql);

            if query_res.changed() {
                match eql_context.0.parse_str(&object_3d_state.data.eql) {
                    Ok(expr) => {
                        object_3d_state.compiled_expr =
                            Some(crate::object_3d::compile_eql_expr(expr));
                    }
                    Err(err) => {
                        ui.colored_label(get_scheme().error, err.to_string());
                    }
                }
            }

            ui.separator();

            // Mesh Type Selection
            ui.label(egui::RichText::new("Mesh Type").color(get_scheme().text_secondary));

            let mut changed = false;
            let current_mesh_type = match &object_3d_state.data.mesh {
                Object3DMesh::Glb(_) => "GLB",
                Object3DMesh::Mesh { mesh, .. } => match mesh {
                    Mesh::Sphere { .. } => "Sphere",
                    Mesh::Box { .. } => "Box",
                    Mesh::Cylinder { .. } => "Cylinder",
                },
            };

            let mut selected_mesh_type = current_mesh_type;
            ui.scope(|ui| {
                ui.style_mut().spacing.combo_width = ui.available_size().x;
                configure_combo_box(ui.style_mut());
                egui::ComboBox::from_label("")
                    .selected_text(current_mesh_type)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());
                        ui.selectable_value(&mut selected_mesh_type, "GLB", "GLB");
                        ui.selectable_value(&mut selected_mesh_type, "Sphere", "Sphere");
                        ui.selectable_value(&mut selected_mesh_type, "Box", "Box");
                        ui.selectable_value(&mut selected_mesh_type, "Cylinder", "Cylinder");
                    });
            });

            // Handle mesh type changes
            if selected_mesh_type != current_mesh_type {
                changed = true;
                match selected_mesh_type {
                    "GLB" => {
                        object_3d_state.data.mesh = Object3DMesh::Glb(String::new());
                    }
                    "Sphere" => {
                        object_3d_state.data.mesh = Object3DMesh::Mesh {
                            mesh: Mesh::Sphere { radius: 1.0 },
                            material: Material {
                                base_color: impeller2_wkt::Color::HYPERBLUE,
                            },
                        };
                    }
                    "Box" => {
                        object_3d_state.data.mesh = Object3DMesh::Mesh {
                            mesh: Mesh::Box {
                                x: 1.0,
                                y: 1.0,
                                z: 1.0,
                            },
                            material: Material {
                                base_color: impeller2_wkt::Color::HYPERBLUE,
                            },
                        };
                    }
                    "Cylinder" => {
                        object_3d_state.data.mesh = Object3DMesh::Mesh {
                            mesh: Mesh::Cylinder {
                                radius: 0.5,
                                height: 2.0,
                            },
                            material: Material {
                                base_color: impeller2_wkt::Color::HYPERBLUE,
                            },
                        };
                    }
                    _ => {}
                }
            }

            ui.separator();

            match &mut object_3d_state.data.mesh {
                Object3DMesh::Glb(path) => {
                    ui.label(egui::RichText::new("GLB Path").color(get_scheme().text_secondary));
                    ui.add_space(8.0);
                    if ui
                        .add(inspector_text_field(path, "Enter a path to a glb"))
                        .changed()
                    {
                        changed = true;
                    }
                }
                Object3DMesh::Mesh { mesh, material } => {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        match mesh {
                            Mesh::Sphere { radius } => {
                                ui.label(
                                    egui::RichText::new("Radius")
                                        .color(get_scheme().text_secondary),
                                );
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(radius)
                                            .speed(0.01)
                                            .range(0.01..=100.0),
                                    )
                                    .changed();
                            }
                            Mesh::Box { x, y, z } => {
                                ui.label(
                                    egui::RichText::new("Dimensions")
                                        .color(get_scheme().text_secondary),
                                );
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(x).speed(0.01).range(0.01..=100.0),
                                        )
                                        .changed();
                                    ui.label("Y:");
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(y).speed(0.01).range(0.01..=100.0),
                                        )
                                        .changed();
                                    ui.label("Z:");
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(z).speed(0.01).range(0.01..=100.0),
                                        )
                                        .changed();
                                });
                            }
                            Mesh::Cylinder { radius, height } => {
                                ui.label(
                                    egui::RichText::new("Radius")
                                        .color(get_scheme().text_secondary),
                                );
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(radius)
                                            .speed(0.01)
                                            .range(0.01..=100.0),
                                    )
                                    .changed();
                                ui.label(
                                    egui::RichText::new("Height")
                                        .color(get_scheme().text_secondary),
                                );
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(height)
                                            .speed(0.01)
                                            .range(0.01..=100.0),
                                    )
                                    .changed();
                            }
                        }
                    });

                    ui.separator();

                    node_color_picker(ui, "Material Color", &mut material.base_color);
                }
            }

            if changed {
                let mut entity = commands.entity(entity);
                entity
                    .remove::<SceneRoot>()
                    .remove::<Mesh3d>()
                    .remove::<MeshMaterial3d<StandardMaterial>>();
                spawn_mesh(
                    &mut entity,
                    &object_3d_state.data.mesh,
                    &mut material_assets,
                    &mut mesh_assets,
                    &assets,
                );
            }
        });

        tree_actions
    }
}
