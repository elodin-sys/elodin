use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy::{
    ecs::{entity::Entity, system::Query},
    render::{camera::Projection, view::Visibility},
};
use bevy_egui::egui::{self, Align};
use bevy_infinite_grid::InfiniteGrid;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{ComponentValue, QueryType, WorldPos};

use crate::EqlContext;
use crate::object_3d::{ComponentArrayExt, EditableEQL, compile_eql_expr};
use crate::ui::CameraQuery;
use crate::ui::button::EButton;
use crate::ui::colors::get_scheme;
use crate::ui::theme::configure_input_with_border;
use crate::ui::widgets::WidgetSystem;
use crate::{
    GridHandle, MainCamera,
    ui::{label::ELabel, theme, utils::MarginSides},
};

use super::{empty_inspector, eql_autocomplete, query};

#[derive(Component)]
pub struct Viewport {
    parent_entity: Entity,
    pub pos: EditableEQL,
    pub look_at: EditableEQL,
}

impl Viewport {
    pub fn new(parent_entity: Entity, pos: EditableEQL, look_at: EditableEQL) -> Self {
        Self {
            parent_entity,
            pos,
            look_at,
        }
    }
}

#[derive(SystemParam)]
pub struct InspectorViewport<'w, 's> {
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    viewports: Query<'w, 's, &'static mut Viewport>,
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

        ui.label(egui::RichText::new("Position").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.pos, &eql_ctx.0);
        ui.separator();
        ui.label(egui::RichText::new("Look At").color(get_scheme().text_secondary));
        eql_input(ui, &mut viewport.look_at, &eql_ctx.0);
        ui.separator();

        if ui.add(EButton::highlight("Reset Pos")).clicked() {
            *cam.transform = Transform::default();
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
            if let Ok(val) = compiled_expr.execute(&entity_map, &values) {
                if let Some(world_pos) = val.as_world_pos() {
                    *pos = world_pos;
                }
            }
            if let Some(compiled_expr) = &viewport.look_at.compiled_expr {
                if let Ok(val) = compiled_expr.execute(&entity_map, &values) {
                    if let Some(look_at) = val.as_world_pos() {
                        let dir = (look_at.pos - pos.pos).normalize();
                        pos.att = nox::Quaternion::look_at_rh(dir, nox::Vec3::z_axis());
                    }
                }
            }
        }
    }
}
