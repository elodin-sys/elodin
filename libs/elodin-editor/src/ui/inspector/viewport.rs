#![allow(non_snake_case)]
use std::collections::HashSet;

use bevy::ecs::system::{SystemParam, SystemState};
use bevy::math::DVec3;
use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use bevy::{
    camera::Projection,
    camera::visibility::Visibility,
    ecs::{entity::Entity, system::Query},
};
use bevy_editor_cam::prelude::EditorCam;
use bevy_egui::egui::{self, Align};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation, OrDefault};
use bevy_infinite_grid::InfiniteGrid;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, QueryType, WorldPos};
use nox::ArrayBuf;

use crate::EqlContext;
use crate::WorldPosExt;
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

const DEFAULT_EDITOR_CAM_ANCHOR_DEPTH: f64 = -2.0;
const ANCHOR_DEPTH_EPSILON: f64 = 1.0e-9;

#[derive(Component)]
pub struct ViewportFocusPickTarget;

/// Extract a 3-vector from a ComponentValue (e.g. F64 array of length >= 3). Returns None if not a numeric array or length < 3.
fn extract_vec3(val: &ComponentValue) -> Option<DVec3> {
    let ComponentValue::F64(array) = val else {
        return None;
    };
    let data = array.buf.as_buf();
    if data.len() < 3 {
        return None;
    }
    Some(DVec3::new(data[0], data[1], data[2]))
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

                        if let Ok(mut editor_cam) = editor_cams.get_mut(camera) {
                            if near_changed {
                                editor_cam.perspective.near_clip_limits = near..near;
                            }
                            if far_changed {
                                let (min_size_per_pixel, max_size_per_pixel) =
                                    crate::ui::tiles::zoom_limits_for_far(far);
                                editor_cam.zoom_limits.min_size_per_pixel = min_size_per_pixel;
                                editor_cam.zoom_limits.max_size_per_pixel = max_size_per_pixel;
                            }
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
    mut viewports: Query<(&Viewport, &mut EditorCam)>,
    mut pos: Query<&mut WorldPos>,
    entity_map: Res<EntityMap>,
    values: Query<&'static ComponentValue>,
    geo_context: Res<GeoContext>,
) {
    for (viewport, mut editor_cam) in viewports.iter_mut() {
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
                    bevy::log::error_once!("viewport pos formula execution error: {}", e);
                }
            }
            if let Some(compiled_expr) = &viewport.look_at.compiled_expr
                && let Ok(val) = compiled_expr.execute(&entity_map, &values)
                && let Some(look_at) = val.as_world_pos()
            {
                let frame = viewport.frame.or_default().unwrap_or(GeoFrame::ENU);
                // Everything stays in the viewport's frame: direction and up
                // are frame coordinates, and `GeoRotation::look_at` yields the
                // attitude expressed in that frame. `sync_pos` carries it into
                // the entity's `GeoRotation` unchanged.
                let dir = look_at.pos() - pos.pos();
                let target_distance = dir.length();

                if !is_valid_viewport_target_distance(target_distance) {
                    continue;
                }
                refresh_default_anchor_depth(&mut editor_cam, target_distance);
                let up = viewport
                    .up
                    .compiled_expr
                    .as_ref()
                    .and_then(|up_expr| up_expr.execute(&entity_map, &values).ok())
                    .and_then(|v| extract_vec3(&v))
                    .filter(|v| v.length_squared() > 1e-20);
                pos.att = GeoRotation::look_at(frame, dir, up, &geo_context).1.into();
            }
        }
    }
}

fn refresh_default_anchor_depth(editor_cam: &mut EditorCam, target_distance: f64) {
    if !is_valid_viewport_target_distance(target_distance) {
        return;
    }
    if (editor_cam.last_anchor_depth - DEFAULT_EDITOR_CAM_ANCHOR_DEPTH).abs() > ANCHOR_DEPTH_EPSILON
    {
        return;
    }
    editor_cam.last_anchor_depth = -target_distance;
}

fn is_valid_viewport_target_distance(target_distance: f64) -> bool {
    target_distance.is_finite() && target_distance > f32::EPSILON as f64
}

pub fn sync_viewport_focus_pick_targets(
    mut commands: Commands,
    viewports: Query<&Viewport>,
    objects: Query<(Entity, &Object3DState)>,
    children: Query<&Children>,
    mesh_entities: Query<(), With<Mesh3d>>,
    current_targets: Query<Entity, With<ViewportFocusPickTarget>>,
) {
    let focus_eqls = viewport_focus_eqls(&viewports);
    let mut desired_targets = HashSet::new();
    let current_targets = current_targets.iter().collect::<HashSet<_>>();

    for (entity, object) in &objects {
        if is_focus_object_eql(&focus_eqls, &object.data.eql) {
            collect_mesh_descendants(entity, &children, &mesh_entities, &mut desired_targets);
        }
    }

    // `try_*` variants silence the command if the entity was despawned between
    // building the target set and applying these deferred commands (e.g. a
    // schematic reload triggered by skybox generation despawns Object3D meshes).
    for entity in current_targets.difference(&desired_targets) {
        commands
            .entity(*entity)
            .try_remove::<(Pickable, ViewportFocusPickTarget)>();
    }

    for entity in desired_targets.difference(&current_targets) {
        commands
            .entity(*entity)
            .try_insert((Pickable::default(), ViewportFocusPickTarget));
    }
}

fn viewport_focus_eqls(viewports: &Query<&Viewport>) -> HashSet<String> {
    viewports
        .iter()
        .filter_map(|viewport| normalized_focus_eql(&viewport.look_at.eql))
        .map(ToOwned::to_owned)
        .collect()
}

fn normalized_focus_eql(eql: &str) -> Option<&str> {
    let eql = eql.trim();
    (!eql.is_empty()).then_some(eql)
}

fn is_focus_object_eql(focus_eqls: &HashSet<String>, object_eql: &str) -> bool {
    normalized_focus_eql(object_eql).is_some_and(|eql| focus_eqls.contains(eql))
}

fn collect_mesh_descendants(
    entity: Entity,
    children: &Query<&Children>,
    mesh_entities: &Query<(), With<Mesh3d>>,
    output: &mut HashSet<Entity>,
) {
    if mesh_entities.contains(entity) {
        output.insert(entity);
    }
    if let Ok(child_list) = children.get(entity) {
        for child in child_list.iter() {
            collect_mesh_descendants(child, children, mesh_entities, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorldPosExt;
    use bevy::math::{Mat3, Quat, Vec3};

    #[test]
    fn refresh_default_anchor_depth_uses_viewport_look_at_distance() {
        let mut editor_cam = EditorCam::default();

        refresh_default_anchor_depth(&mut editor_cam, 14.696_938);

        assert!((editor_cam.last_anchor_depth + 14.696_938).abs() < 1.0e-9);
    }

    #[test]
    fn refresh_default_anchor_depth_keeps_user_adjusted_depth() {
        let mut editor_cam = EditorCam {
            last_anchor_depth: -8.0,
            ..Default::default()
        };

        refresh_default_anchor_depth(&mut editor_cam, 14.696_938);

        assert_eq!(editor_cam.last_anchor_depth, -8.0);
    }

    #[test]
    fn refresh_default_anchor_depth_rejects_invalid_distance() {
        for target_distance in [0.0, f64::NAN, f64::INFINITY] {
            let mut editor_cam = EditorCam::default();

            refresh_default_anchor_depth(&mut editor_cam, target_distance);

            assert_eq!(
                editor_cam.last_anchor_depth,
                DEFAULT_EDITOR_CAM_ANCHOR_DEPTH
            );
        }
    }

    /// Full pipeline test with real EQL: `set_viewport_pos` -> `sync_pos` ->
    /// `apply_geo_rotation`. A NED viewport with an explicit `up="(0,0,-1)"`
    /// (up, away from the ground in NED) must produce a right-side-up rig
    /// looking at the target; `up="(0,0,1)"` (down) must produce an inverted
    /// one.
    #[test]
    fn ned_viewport_explicit_up_through_pipeline() {
        use crate::object_3d::{EditableEQL, compile_eql_expr};
        use bevy::math::{DQuat, DVec3};
        use bevy::prelude::{IntoScheduleConfigs, Transform};
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

        let eql_ctx = eql::Context::default();
        let editable = |s: &str| EditableEQL {
            eql: s.to_string(),
            compiled_expr: Some(
                compile_eql_expr(
                    eql_ctx
                        .parse_str(s)
                        .unwrap_or_else(|e| panic!("parse {s:?}: {e}")),
                )
                .unwrap_or_else(|e| panic!("compile {s:?}: {e}")),
            ),
        };

        for (up_eql, expect_up_y) in [("(0,0,-1)", 1.0f32), ("(0,0,1)", -1.0f32)] {
            let mut app = bevy::app::App::new();
            app.insert_resource(GeoContext::default());
            app.init_resource::<super::EntityMap>();
            crate::register_world_pos_components(&mut app);
            app.add_systems(
                bevy::app::Update,
                (
                    super::set_viewport_pos,
                    crate::sync_pos,
                    bevy_geo_frames::apply_geo_rotation,
                )
                    .chain(),
            );

            let frame = GeoFrame::NED;
            let parent = app
                .world_mut()
                .spawn((
                    super::WorldPos::default(),
                    GeoPosition(frame, DVec3::ZERO),
                    GeoRotation::new(frame, DQuat::IDENTITY),
                ))
                .id();
            // Values from the failing ball.kdl viewport.
            let viewport_entity = app
                .world_mut()
                .spawn((
                    super::Viewport::new(
                        parent,
                        editable("(0,0,0,0, 0,0,0)"),
                        editable("(0,0,0,0, 0,-3,0)"),
                        editable(up_eql),
                        Some(frame),
                    ),
                    EditorCam::default(),
                ))
                .id();
            app.update();

            let editor_cam = app.world().get::<EditorCam>(viewport_entity).unwrap();
            assert!(
                (editor_cam.last_anchor_depth + 3.0).abs() < 1e-9,
                "up={up_eql}: viewport positioning system did not run"
            );
            let transform = *app.world().get::<Transform>(parent).unwrap();
            let up = transform.rotation * Vec3::Y;
            assert!(
                up.y * expect_up_y > 0.5,
                "up={up_eql}: rig up {up:?}, expected y sign {expect_up_y}"
            );
            // NED (0,-3,0) is 3 m west of the origin => bevy -X.
            let fwd = transform.rotation * Vec3::NEG_Z;
            assert!(
                (fwd.x - -1.0).abs() < 1e-5 && fwd.y.abs() < 1e-5,
                "up={up_eql}: camera fwd = {fwd:?}, expected -X"
            );
        }
    }

    macro_rules! assert_eq_mat {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_mat!($a, $b, "");
        }};
        ($a:expr, $b:expr, $($arg:tt)+) => {{
            let a = $a;
            let b = $b;

            for i in 0..3 {
                let aa = a.col(i);
                let bb = b.col(i);
                for j in 0..3 {
                    let delta = aa[j] - bb[j];
                    if delta.abs() > 1e-5 {
                        panic!("First mismatch on column {}:\nleft:  {}\nright: {}: {}",
                                i + 1, a, b, format_args!($($arg)+));
                    }
                }
            }
        }};
    }
    macro_rules! assert_eq_vec {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_vec!($a, $b, "");
        }};
        ($a:expr, $b:expr, $($arg:tt)+) => {{
            let a = $a;
            let b = $b;

            for i in 0..3 {
                let delta = a[i] - b[i];
                if delta.abs() > 1e-5 {
                    panic!("First mismatch on index {}:\nleft:  {}\nright: {}: {}",
                           i, a, b, format_args!($($arg)+));
                }
            }
        }};
    }
    macro_rules! assert_eq_quat {
        ($a:expr, $b:expr $(,)?) => {{
            assert_eq_quat!($a, $b, "");
        }};

        ($a:expr, $b:expr, $($arg:tt)+) => {{

            let a = $a;
            let b = $b;


            let dot = a.dot(b).abs();

            assert!(
                (1.0 - dot) <= 1e-5,
                "Quat mismatch:\nleft:  {:?}\nright: {:?}: {}",
                a,
                b,
                format_args!($($arg)+)
            );
        }};
    }

    #[inline]
    fn are_collinear(a: Vec3, b: Vec3) -> bool {
        a.cross(b).length_squared() < 1e-6
    }

    /// Constructs a look_at rotation matrix that matches
    /// [nox::Matrix3::look_at_rh].
    fn glam_look_at_rh(dir: Vec3, up: Vec3) -> (Mat3, Vec3) {
        let up_candidates = [up, Vec3::Y, Vec3::X, Vec3::Z];
        let up = up_candidates
            .into_iter()
            .find(|up| !are_collinear(*up, dir))
            .expect("it can't be collinear with everyone");
        // Constructs a look_at rotation matrix using the same algorithm as
        // nox::Matrix3::look_at_rh.
        //
        // let f = dir.normalize();
        // let s = f.cross(up).normalize();
        // let u = s.cross(f);
        // // nox uses from_rows then transpose, which equals from_cols
        // Mat3::from_cols(s, f, u)
        (Mat3::look_to_rh(dir, up), up)
    }

    /// This function converts an Elodin rotation matrix to an EUS/Bevy rotation
    /// matrix and vice versa. It behaves as though one right-multiplied M by
    /// bevy_R_enu and transposed, i.e., (M * bevy_R_elodin)^T but no actual matrix
    /// multiplication happens because column re-ordering is faster.
    ///
    ///
    /// ```ignore
    ///   elodin_R_bevy =  [ 1  0  0 ]
    ///                    [ 0  0 -1 ]
    ///                    [ 0  1  0 ]
    /// ```
    ///
    /// Note: It's orthonormal, so its transpose is its inverse.
    #[inline]
    fn elodin_R_bevy(M: Mat3) -> Mat3 {
        // Bevy +X -> ENU East
        // Bevy +Y -> ENU Up
        // Bevy +Z -> -ENU North
        Mat3::from_cols(M.x_axis, M.z_axis, -M.y_axis).transpose()
    }

    // #[inline]
    fn bevy_R_elodin(M: Mat3) -> Mat3 {
        // ENU East  -> Bevy +X
        // ENU North -> Bevy -Z
        // ENU Up    -> Bevy +Y
        let M = M.transpose();
        Mat3::from_cols(M.x_axis, -M.z_axis, M.y_axis)
    }

    #[test]
    fn test_inverses() {
        let A = Mat3::from_cols(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let B = elodin_R_bevy(A);
        let C = bevy_R_elodin(B);
        assert_eq_mat!(A, C, "b to e to b");
        let B = bevy_R_elodin(A);
        let C = elodin_R_bevy(B);
        assert_eq_mat!(A, C, "e to b to e");
    }

    /// Compare against elodin's ENU.
    #[test]
    fn test_look_at_rh_nox_vs_glam_elodin() {
        test_look_at_rh_nox_vs_glam(
            |glam_mat, nox_mat| (elodin_R_bevy(glam_mat), nox_mat),
            |M| M,
        );
    }

    /// Compare against Bevy's EUS.
    #[test]
    fn test_look_at_rh_nox_vs_glam_bevy() {
        test_look_at_rh_nox_vs_glam(
            |glam_mat, nox_mat| (glam_mat, bevy_R_elodin(nox_mat)),
            elodin_R_bevy,
        );
    }

    /// `WorldPosExt::bevy_att` must match `GeoRotation::to_bevy` in plane mode.
    #[test]
    fn test_bevy_att_vs_geo_frames_plane() {
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoRotation, Present};

        let ctx = GeoContext::default().with_present(Present::Plane);

        for (i, (dir, up)) in look_at_test_cases().into_iter().enumerate() {
            let nox_dir = nox::Vec3::from(dir.as_dvec3());
            let nox_up = nox::Vec3::from(up.as_dvec3());
            let (nox_mat, _) = nox::Matrix3::look_at_rh_up(nox_dir, nox_up);
            let nox_quat = nox::Quaternion::from_rot_mat(nox_mat);
            let world_pos = super::WorldPos {
                att: nox_quat,
                pos: nox::Vec3::new(0.0, 0.0, 0.0),
            };

            let elodin_bevy = world_pos.bevy_att();
            let geo_frames_bevy = GeoRotation::new(GeoFrame::ENU, world_pos.att()).to_bevy(&ctx);
            assert_eq_quat!(
                elodin_bevy.as_quat(),
                geo_frames_bevy.as_quat(),
                "case {i} dir {dir} up {up}"
            );
        }
    }

    #[test]
    fn focus_object_eql_matches_trimmed_viewport_look_at() {
        let focus_eqls = HashSet::from(["lander.world_pos".to_string()]);

        assert!(is_focus_object_eql(&focus_eqls, " lander.world_pos "));
        assert!(!is_focus_object_eql(&focus_eqls, "lander_truth.world_pos"));
        assert!(!is_focus_object_eql(&focus_eqls, ""));
    }
    #[test]
    fn test_from_mat3() {
        let q = Quat::from_mat3(&Mat3::IDENTITY);
        assert_eq_quat!(q, Quat::IDENTITY);

        let dir = Vec3::Y;
        let up = Vec3::Z;
        let (M, _) = glam_look_at_rh(dir, up);
        assert_eq_mat!(M, Mat3::from_cols(Vec3::X, Vec3::NEG_Z, Vec3::Y));
        let q = Quat::from_mat3(&Mat3::IDENTITY);
        assert_eq_quat!(q, Quat::IDENTITY);
        let S = elodin_R_bevy(M).transpose();
        assert_eq_mat!(S, Mat3::IDENTITY);

        assert_eq_mat!(
            Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Y, Vec3::Z),
            glam_look_at_rh(Vec3::NEG_Z, Vec3::NEG_Y).0,
            "trial 0"
        );
        // assert_eq_mat!(Mat3::from_cols(Vec3::NEG_X, Vec3::NEG_Y, Vec3::Z), glam_look_at_rh(Vec3::Z, Vec3::Y).0, "current");
    }

    fn look_at_test_cases() -> [(Vec3, Vec3); 10] {
        [
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z), // 0: identity for Elodin ENU
            (Vec3::NEG_Z, Vec3::Y),              // 1: identity for Bevy
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
            (Vec3::new(1.0, 2.0, 3.0).normalize(), Vec3::Y),
            (Vec3::new(-1.0, 0.5, 0.3).normalize(), Vec3::Y),
            (Vec3::new(0.0, 0.0, -1.0), Vec3::Y),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
            (Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
            (Vec3::new(0.0, 0.0, 1.0), Vec3::Z),
        ]
    }

    fn test_look_at_rh_nox_vs_glam(f: fn(Mat3, Mat3) -> (Mat3, Mat3), g: fn(Mat3) -> Mat3) {
        for (i, (dir, up)) in look_at_test_cases().into_iter().enumerate() {
            let nox_dir = nox::Vec3::from(dir.as_dvec3());
            let nox_up = nox::Vec3::from(up.as_dvec3());

            let (nox_mat, nox_up_actual) = nox::Matrix3::look_at_rh_up(nox_dir, nox_up);
            let (glam_mat, up_actual) = glam_look_at_rh(dir, up);
            assert_eq_vec!(
                up_actual,
                bevy::math::DVec3::from(nox_up_actual).as_vec3(),
                "case {i} nox and bevy up vector don't match"
            );
            // let glam_mat = elodin_R_bevy(glam_mat);
            let nox_mat_bevy: bevy::math::Mat3 = bevy::math::DMat3::from(nox_mat).as_mat3();
            // let nox_mat_bevy = bevy_R_elodin(nox_mat_bevy);
            let (glam_mat, _nox_mat_bevy) = f(glam_mat, nox_mat_bevy);

            // Weird thing. The matrices are not always the same but the
            // quaternions are.
            // assert_eq_mat!(nox_mat_bevy, glam_mat, "\ncase {i} dir {dir} up {up}");

            // Also compare resulting quaternions
            let nox_quat_look_at = nox::Quaternion::look_at_rh(nox_dir, nox_up);
            let nox_quat = nox::Quaternion::from_rot_mat(nox_mat);
            assert_eq_quat!(
                bevy::math::DQuat::from(nox_quat),
                bevy::math::DQuat::from(nox_quat_look_at),
                "case {i} look_at_rh vs mat"
            );
            let world_pos = super::WorldPos {
                att: nox_quat,
                pos: nox::Vec3::new(0.0, 0.0, 0.0),
            };
            let nox_quat_bevy = world_pos.bevy_att();
            // let glam_quat = Quat::from_mat3(&elodin_R_bevy(glam_mat).transpose());
            let glam_quat = Quat::from_mat3(&g(glam_mat));
            assert_eq_quat!(nox_quat_bevy.as_quat(), glam_quat, "case {i} second");
        }
    }

    fn focus_object_state(eql: &str) -> Object3DState {
        use impeller2_wkt::{Object3D, Object3DMesh};
        Object3DState {
            compiled_expr: None,
            scale_expr: None,
            scale_error: None,
            error_covariance_cholesky_expr: None,
            joint_animations: Vec::new(),
            data: Object3D {
                eql: eql.to_string(),
                mesh: Object3DMesh::glb("model.glb"),
                frame: None,
                icon: None,
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            },
        }
    }

    fn focus_viewport(eql: &str, parent_entity: Entity) -> Viewport {
        Viewport {
            parent_entity,
            pos: EditableEQL {
                eql: String::new(),
                compiled_expr: None,
            },
            look_at: EditableEQL {
                eql: eql.to_string(),
                compiled_expr: None,
            },
            up: EditableEQL {
                eql: String::new(),
                compiled_expr: None,
            },
            frame: None,
        }
    }

    /// Regression for the panic where a focus mesh entity is despawned between
    /// `sync_viewport_focus_pick_targets` queueing its `insert` and the deferred
    /// commands applying (e.g. skybox generation reloads the schematic). The
    /// queued command must be silenced rather than panic on the dead entity.
    #[test]
    fn sync_viewport_focus_pick_targets_survives_despawned_target() {
        let mut world = World::new();
        let parent = world.spawn(focus_object_state("e.world_pos")).id();
        let mesh = world.spawn(Mesh3d(Handle::default())).id();
        world.entity_mut(parent).add_child(mesh);
        let viewport_parent = world.spawn_empty().id();
        world.spawn(focus_viewport("e.world_pos", viewport_parent));

        let mut state: SystemState<(
            Commands,
            Query<&Viewport>,
            Query<(Entity, &Object3DState)>,
            Query<&Children>,
            Query<(), With<Mesh3d>>,
            Query<Entity, With<ViewportFocusPickTarget>>,
        )> = SystemState::new(&mut world);

        {
            let (commands, viewports, objects, children, mesh_entities, current_targets) =
                state.get_mut(&mut world);
            sync_viewport_focus_pick_targets(
                commands,
                viewports,
                objects,
                children,
                mesh_entities,
                current_targets,
            );
        }

        // Despawn the target after the insert is queued but before it applies.
        world.entity_mut(mesh).despawn();
        state.apply(&mut world);

        assert!(world.get_entity(mesh).is_err());
    }
}
