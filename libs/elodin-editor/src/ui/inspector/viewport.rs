use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy::{
    camera::Projection,
    camera::visibility::Visibility,
    ecs::{entity::Entity, system::Query},
};
use bevy_editor_cam::prelude::EditorCam;
use bevy_egui::egui::{self, Align};
use bevy_geo_frames::GeoFrame;
use bevy_infinite_grid::InfiniteGrid;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, QueryType, WorldPos};
use nox::{ArrayBuf, ArrayRepr, Vector3};

use crate::EqlContext;
use crate::object_3d::{ComponentArrayExt, EditableEQL, Object3DState, compile_eql_expr};
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
    /// Optional geo frame for interpreting position and rotation.
    pub frame: Option<GeoFrame>,
}

impl Viewport {
    pub fn new(
        parent_entity: Entity,
        pos: EditableEQL,
        look_at: EditableEQL,
        up: EditableEQL,
        frame: Option<GeoFrame>,
    ) -> Self {
        Self {
            parent_entity,
            pos,
            look_at,
            up,
            frame,
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
    editor_cams: Query<'w, 's, &'static mut EditorCam>,
    object_3d_states: Query<'w, 's, &'static Object3DState>,
    eql_ctx: ResMut<'w, EqlContext>,
}

impl WidgetSystem for InspectorViewport<'_, '_> {
    type Args = (Entity, String);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let scheme = get_scheme();
        let state_mut = state.get_mut(world);

        let (camera, title) = args;

        let InspectorViewport {
            mut camera_query,
            mut viewports,
            mut viewport_configs,
            viewport_rects,
            mut grid_visibility,
            mut editor_cams,
            object_3d_states,
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
        let has_detected_ellipsoid = object_3d_states.iter().any(|object_state| {
            matches!(
                &object_state.data.mesh,
                impeller2_wkt::Object3DMesh::Ellipsoid { .. }
            )
        });

        ui.spacing_mut().item_spacing.y = 8.0;
        let title = title.trim();
        let title = if title.is_empty() { "Viewport" } else { title };
        ui.add(
            ELabel::new(title)
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
            let mut configured_clip_planes = None;
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
                    let mut near_changed = false;
                    let mut far_changed = false;

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("NEAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            near_changed |= ui
                                .add(egui::DragValue::new(&mut near).speed(0.001))
                                .changed();
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FAR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            far_changed |=
                                ui.add(egui::DragValue::new(&mut far).speed(0.01)).changed();
                        });
                    });

                    if near_changed || far_changed {
                        near = near.max(0.0001);
                        if far <= near + 0.0001 {
                            far = near + 0.0001;
                        }
                        persp.near = near;
                        persp.far = far;
                        configured_clip_planes = Some((near, far));

                        if near_changed && let Ok(mut editor_cam) = editor_cams.get_mut(camera) {
                            editor_cam.perspective.near_clip_limits = near..near;
                        }
                    }

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
            if let Some((near, far)) = configured_clip_planes {
                viewport_config.configured_near = Some(near);
                viewport_config.configured_far = Some(far);
            }
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
                    ui.label(egui::RichText::new("FRUSTUM").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        let frustum_created = viewport_config.create_frustum;
                        let button = if frustum_created {
                            EButton::red("DELETE")
                        } else {
                            EButton::highlight("CREATE")
                        };
                        if ui.add(button.width(create_button_width)).clicked() {
                            viewport_config.create_frustum = !frustum_created;
                        }
                    });
                });

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
                    let mut projection_color = viewport_config.projection_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("PROJ. 2D COLOR").color(scheme.text_secondary),
                        );
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(projection_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("projection_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut projection_color, color_id, &swatch);
                        });
                    });
                    viewport_config.projection_color =
                        impeller2_wkt::Color::from_color32(projection_color);

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
                let show_intersection_options =
                    viewport_config.show_frustums && has_detected_ellipsoid;
                if !show_intersection_options {
                    viewport_config.show_coverage_in_viewport = false;
                    viewport_config.show_projection_2d = false;
                }

                if show_intersection_options {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("COVERAGE").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut viewport_config.show_coverage_in_viewport, "");
                        });
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("PROJ. 2D").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            theme::configure_input_with_border(ui.style_mut());
                            ui.checkbox(&mut viewport_config.show_projection_2d, "");
                        });
                    });
                }
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
                    editable_expr.compiled_expr = compile_eql_expr(expr).ok();
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
                let dir = match viewport.frame {
                    Some(GeoFrame::NED) => nox::Vec3::new(dir.y(), dir.x(), -dir.z()),
                    _ => dir,
                };
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
                    .unwrap_or(nox::Vec3::z_axis());
                pos.att = nox::Quaternion::look_at_rh(dir, up_vec);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::{Mat3, Quat, Vec3};
    use crate::WorldPosExt;

    /// Constructs a look_at rotation matrix using the same algorithm as nox::Matrix3::look_at_rh
    fn glam_look_at_rh(dir: Vec3, up: Vec3) -> Mat3 {
        // let f = dir.normalize();
        // let up = if up.dot(dir).abs() >= 1.0 - 1e-6 {
        //     Vec3::Y
        // } else {
        //     up
        // };
        // let s = f.cross(up).normalize();
        // let u = s.cross(f);
        // // nox uses from_rows then transpose, which equals from_cols
        // Mat3::from_cols(s, f, u)
        Mat3::look_to_rh(dir, up)
    }

    fn bevy_R_enu(R: Mat3) -> Mat3 {
        // R_enu = C.transpose() * R_bevy * C
        //
        // C columns:
        //   ENU east  -> Bevy +X
        //   ENU north -> Bevy -Z
        //   ENU up    -> Bevy +Y

        Mat3::from_cols(
            R.x_axis,
            R.z_axis,
            -R.y_axis,
        ).transpose()
    }

    fn enu_R_bevy(R: Mat3) -> Mat3 {
        bevy_R_enu(R)
    }

    #[test]
    fn test_look_at_rh_nox_vs_glam() {
        let test_cases = [
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z), // This is the identity
 // transform for Elodin's look_to. No surprise. It's ENU with north as the
 // facing direction.
            (Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
            (Vec3::new(1.0, 2.0, 3.0).normalize(), Vec3::Y),
            (Vec3::new(-1.0, 0.5, 0.3).normalize(), Vec3::Y),
            (Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
        ];

        for (i, (dir, up)) in test_cases.into_iter().enumerate() {
            let nox_dir = nox::Vec3::new(dir.x as f64, dir.y as f64, dir.z as f64);
            let nox_up = nox::Vec3::new(up.x as f64, up.y as f64, up.z as f64);

            let nox_mat = nox::Matrix3::look_at_rh(nox_dir, nox_up);
            let nox_mat = bevy_R_enu(nox_mat);
            let glam_mat = glam_look_at_rh(dir, up);
            // let glam_mat = bevy_R_enu(glam_mat);

            // Compare the matrices
            let nox_buf = nox_mat.into_buf();
            let glam_cols = [
                glam_mat.x_axis,
                glam_mat.y_axis,
                glam_mat.z_axis,
            ];

            println!("Testing case {i} dir={:?}, up={:?}", dir, up);
            println!("nox matrix (column-major):");
            println!("  col0: [{:.6}, {:.6}, {:.6}]", nox_buf[0][0], nox_buf[1][0], nox_buf[2][0]);
            println!("  col1: [{:.6}, {:.6}, {:.6}]", nox_buf[0][1], nox_buf[1][1], nox_buf[2][1]);
            println!("  col2: [{:.6}, {:.6}, {:.6}]", nox_buf[0][2], nox_buf[1][2], nox_buf[2][2]);
            println!("glam matrix:");
            println!("  col0: {:?}", glam_cols[0]);
            println!("  col1: {:?}", glam_cols[1]);
            println!("  col2: {:?}", glam_cols[2]);

            // Check if matrices are approximately equal
            let eps = 1e-5;
            for col in 0..3 {
                for row in 0..3 {
                    let nox_val = nox_buf[row][col];
                    let glam_val = glam_cols[col][row];
                    // assert!(
                    //     (nox_val - glam_val as f64).abs() < eps,
                    //     "Mismatch at [{},{}]: nox={}, glam={}, dir={:?}, up={:?}",
                    //     row, col, nox_val, glam_val, dir, up
                    // );
                }
            }

            // Also compare resulting quaternions
            let nox_quat = nox::Quaternion::look_at_rh(nox_dir, nox_up);
            let world_pos = super::WorldPos {
                att: nox_quat,
                pos: nox::Vec3::new(0.0, 0.0, 0.0),
            };
            // let nox_quat = world_pos.att();
            let nox_quat = world_pos.bevy_att();
            let glam_quat = Quat::from_mat3(&glam_mat);

            let nox_q = nox_quat.to_array();
            println!("nox quat: [{:.6}, {:.6}, {:.6}, {:.6}]", nox_q[0], nox_q[1], nox_q[2], nox_q[3]);
            println!("glam quat: {:?}", glam_quat);

            // Quaternions can differ by sign (q and -q represent the same rotation)
            let same_sign = (nox_q[3] - glam_quat.w as f64).abs() < eps;
            let sign = if same_sign { 1.0 } else { -1.0 };
            assert!(
                (nox_q[0] - sign * glam_quat.x as f64).abs() < eps &&
                (nox_q[1] - sign * glam_quat.y as f64).abs() < eps &&
                (nox_q[2] - sign * glam_quat.z as f64).abs() < eps &&
                (nox_q[3] - sign * glam_quat.w as f64).abs() < eps,
                "Quaternion mismatch: nox={:?}, glam={:?}, dir={:?}, up={:?}",
                nox_q, glam_quat, dir, up
            );
            println!("PASSED\n");
        }
    }
}
