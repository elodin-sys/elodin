use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy::{
    camera::Projection,
    camera::visibility::Visibility,
    ecs::{entity::Entity, system::Query},
};
use bevy_egui::egui::{self, Align};
use bevy_infinite_grid::InfiniteGrid;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, QueryType, WorldPos};
use nox::{ArrayBuf, ArrayRepr, Vector3};

use crate::EqlContext;
use crate::object_3d::{ComponentArrayExt, EditableEQL, compile_eql_expr};
use crate::ui::button::EButton;
use crate::ui::colors::{EColor, get_scheme};
use crate::ui::theme::configure_input_with_border;
use crate::ui::widgets::WidgetSystem;
use crate::ui::{CameraQuery, ViewportRect};
use crate::{
    GridHandle, MainCamera,
    ui::tiles::ViewportConfig,
    ui::{label::ELabel, theme, utils::MarginSides},
};

use super::{color_popup, empty_inspector, eql_autocomplete, query};

/// Extract a 3-vector from a ComponentValue (e.g. F64 array of length >= 3). Returns None if not a numeric array or length < 3.
fn extract_vec3(val: &ComponentValue) -> Option<Vector3<f64, ArrayRepr>> {
    let ComponentValue::F64(array) = val else {
        return None;
    };
    let data = array.buf.as_buf();
    if data.len() < 3 {
        return None;
    }
    Some(Vector3::new(data[0], data[1], data[2]))
}

#[derive(Component)]
pub struct Viewport {
    parent_entity: Entity,
    pub pos: EditableEQL,
    pub look_at: EditableEQL,
    /// Optional camera up vector in world frame. EQL that evaluates to a 3-vector (e.g. "(0,0,1)" or "pose.direction(0,1,1)").
    pub up: EditableEQL,
}

impl Viewport {
    pub fn new(
        parent_entity: Entity,
        pos: EditableEQL,
        look_at: EditableEQL,
        up: EditableEQL,
    ) -> Self {
        Self {
            parent_entity,
            pos,
            look_at,
            up,
        }
    }
}

#[derive(SystemParam)]
pub struct InspectorViewport<'w, 's> {
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    viewports: Query<'w, 's, &'static mut Viewport>,
    viewport_configs: Query<'w, 's, &'static mut ViewportConfig>,
    viewport_rects: Query<'w, 's, &'static ViewportRect, With<MainCamera>>,
    grid_visibility: Query<'w, 's, &'static mut Visibility, With<InfiniteGrid>>,
    eql_ctx: ResMut<'w, EqlContext>,
}

impl WidgetSystem for InspectorViewport<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let scheme = get_scheme();
        let state_mut = state.get_mut(world);

        let camera = args;

        let InspectorViewport {
            mut camera_query,
            mut viewports,
            mut viewport_configs,
            viewport_rects,
            mut grid_visibility,
            eql_ctx,
        } = state_mut;

        let Ok(mut cam) = camera_query.get_mut(camera) else {
            ui.add(empty_inspector());
            return;
        };

        let Ok(mut viewport) = viewports.get_mut(camera) else {
            return;
        };
        let Ok(mut viewport_config) = viewport_configs.get_mut(camera) else {
            return;
        };

        ui.spacing_mut().item_spacing.y = 8.0;
        ui.add(
            ELabel::new("Viewport")
                .padding(egui::Margin::same(8).bottom(24.0))
                .bottom_stroke(egui::Stroke {
                    color: get_scheme().border_primary,
                    width: 1.0,
                })
                .margin(egui::Margin::same(0).bottom(16.0)),
        );

        ui.label(egui::RichText::new("POSITION").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.pos, &eql_ctx.0);
        ui.separator();
        ui.label(egui::RichText::new("LOOK AT").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.look_at, &eql_ctx.0);
        ui.separator();
        ui.label(egui::RichText::new("UP").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.up, &eql_ctx.0);
        ui.separator();

        if ui.add(EButton::highlight("Reset Pos")).clicked() {
            *cam.transform = <bevy::prelude::Transform as std::default::Default>::default();
        }

        if let Projection::Perspective(persp) = cam.projection.as_mut() {
            ui.separator();
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    let mut fov = persp.fov.to_degrees();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FOV").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            if ui.add(egui::DragValue::new(&mut fov).speed(0.1)).changed() {
                                persp.fov = fov.to_radians();
                            }
                        });
                    });
                    ui.add_space(8.0);
                    ui.style_mut().spacing.slider_width = ui.available_size().x;
                    ui.style_mut().visuals.widgets.inactive.bg_fill = scheme.bg_secondary;
                    if ui
                        .add(egui::Slider::new(&mut fov, 5.0..=120.0).show_value(false))
                        .changed()
                    {
                        persp.fov = fov.to_radians();
                    }

                    ui.add_space(8.0);
                    let mut near = persp.near;
                    let mut far = persp.far;

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("NEAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            ui.add(egui::DragValue::new(&mut near).speed(0.001));
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            ui.add(egui::DragValue::new(&mut far).speed(0.01));
                        });
                    });

                    near = near.max(0.0001);
                    if far <= near + 0.0001 {
                        far = near + 0.0001;
                    }
                    persp.near = near;
                    persp.far = far;

                    ui.add_space(8.0);
                    let derived_aspect = viewport_rects
                        .get(camera)
                        .ok()
                        .and_then(|rect| rect.0)
                        .and_then(|rect| {
                            let size = rect.size();
                            if size.x > 0.0 && size.y > 0.0 {
                                Some((size.x / size.y, size.x, size.y))
                            } else {
                                None
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("REAL ASPECT").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            if let Some((aspect, width, height)) = derived_aspect {
                                ui.label(format!("{aspect:.3} ({width:.0}x{height:.0})"));
                            } else {
                                ui.label("N/A");
                            }
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("ASPECT MODE").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let fixed_selected = viewport_config.aspect.is_some();
                            if ui.selectable_label(fixed_selected, "FIXED").clicked()
                                && viewport_config.aspect.is_none()
                            {
                                viewport_config.aspect = Some(
                                    derived_aspect
                                        .map(|(aspect, _, _)| aspect)
                                        .unwrap_or(persp.aspect_ratio.max(0.0001)),
                                );
                            }
                            ui.add_space(8.0);
                            if ui.selectable_label(!fixed_selected, "AUTO").clicked() {
                                viewport_config.aspect = None;
                            }
                        });
                    });

                    if let Some(aspect) = viewport_config.aspect.as_mut() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("ASPECT").color(scheme.text_secondary));
                            ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                                if ui.add(egui::DragValue::new(aspect).speed(0.01)).changed() {
                                    *aspect = (*aspect).max(0.0001);
                                }
                            });
                        });
                    }
                });
        }

        if let Some(&GridHandle { grid }) = cam.grid_handle {
            ui.separator();
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("SHOW GRID").color(scheme.text_secondary));
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

        ui.separator();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                let create_button_width = 88.0;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("FRUSTRUM").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        let response = ui.add(
                            EButton::highlight("CREATE")
                                .width(create_button_width)
                                .disabled(viewport_config.create_frustum),
                        );
                        if response.clicked() && !viewport_config.create_frustum {
                            viewport_config.create_frustum = true;
                        }
                    });
                });
                if viewport_config.create_frustum {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("");
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            ui.add_sized(
                                [create_button_width, 18.0],
                                egui::Label::new(
                                    egui::RichText::new("CREATED")
                                        .color(scheme.highlight)
                                        .strong(),
                                ),
                            );
                        });
                    });
                }

                if viewport_config.create_frustum {
                    ui.add_space(8.0);

                    let mut frustums_color = viewport_config.frustums_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FRUSTUM COLOR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(frustums_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("frustums_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut frustums_color, color_id, &swatch);
                        });
                    });
                    viewport_config.frustums_color =
                        impeller2_wkt::Color::from_color32(frustums_color);

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("THICKNESS").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut thickness = viewport_config.frustums_thickness;
                            if ui
                                .add(egui::DragValue::new(&mut thickness).speed(0.001))
                                .changed()
                            {
                                viewport_config.frustums_thickness = thickness.max(0.0001);
                            }
                        });
                    });
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SHOW FRUSTUMS").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        theme::configure_input_with_border(ui.style_mut());
                        ui.checkbox(&mut viewport_config.show_frustums, "");
                    });
                });
            });
    }
}

fn eql_input(ui: &mut egui::Ui, editable_expr: &mut EditableEQL, ctx: &eql::Context) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        configure_input_with_border(ui.style_mut());
        let query_res = ui.add(query(&mut editable_expr.eql, QueryType::EQL));
        eql_autocomplete(ui, ctx, &query_res, &mut editable_expr.eql);
        if query_res.changed() {
            if editable_expr.eql.is_empty() {
                editable_expr.compiled_expr = None;
                return;
            }
            match ctx.parse_str(&editable_expr.eql) {
                Ok(expr) => {
                    editable_expr.compiled_expr = Some(compile_eql_expr(expr));
                }
                Err(err) => {
                    ui.colored_label(get_scheme().error, err.to_string());
                }
            }
        }
    });
}

pub fn set_viewport_pos(
    viewports: Query<&Viewport>,
    mut pos: Query<&mut WorldPos>,
    entity_map: Res<EntityMap>,
    values: Query<&'static ComponentValue>,
) {
    for viewport in viewports.iter() {
        let Ok(mut pos) = pos.get_mut(viewport.parent_entity) else {
            continue;
        };
        if let Some(compiled_expr) = &viewport.pos.compiled_expr {
            match compiled_expr.execute(&entity_map, &values) {
                Ok(val) => {
                    if let Some(world_pos) = val.as_world_pos() {
                        *pos = world_pos;
                    } else {
                        bevy::log::warn!("viewport pos expression didn't produce a WorldPos");
                    }
                }
                Err(e) => {
                    bevy::log::error!("viewport pos formula execution error: {}", e);
                }
            }
            if let Some(compiled_expr) = &viewport.look_at.compiled_expr
                && let Ok(val) = compiled_expr.execute(&entity_map, &values)
                && let Some(look_at) = val.as_world_pos()
            {
                let dir = (look_at.pos - pos.pos).normalize();
                let up_vec = viewport
                    .up
                    .compiled_expr
                    .as_ref()
                    .and_then(|up_expr| up_expr.execute(&entity_map, &values).ok())
                    .and_then(|v| extract_vec3(&v))
                    .and_then(|v| {
                        let n_sq = v.norm_squared();
                        if n_sq.into_buf() > 1e-20 {
                            Some(v.normalize())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(nox::Vec3::z_axis);
                pos.att = nox::Quaternion::look_at_rh(dir, up_vec);
            }
        }
    }
}
