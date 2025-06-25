use std::str::FromStr;

use bevy::{ecs::system::SystemParam, prelude::*};
use impeller2_wkt::{Dashboard, DashboardNode};
use smallvec::SmallVec;

use crate::{
    EqlContext,
    ui::{
        SelectedObject,
        colors::{self, get_scheme},
        dashboard::{DashboardNodePath, spawn_node},
        inspector::graph::{eql_autocomplete, query},
        theme::{configure_combo_box, configure_combo_item, configure_input_with_border},
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorDashboardNode<'w, 's> {
    pub paths: Query<'w, 's, (&'static ChildOf, &'static DashboardNodePath)>,
    pub dashboards: Query<'w, 's, &'static mut Dashboard<Entity>>,
    pub eql_ctx: Res<'w, EqlContext>,
    pub commands: Commands<'w, 's>,
    pub selected_object: ResMut<'w, SelectedObject>,
}

impl WidgetSystem for InspectorDashboardNode<'_, '_> {
    type Args = Entity;

    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorDashboardNode {
            paths,
            mut dashboards,
            eql_ctx,
            mut commands,
            mut selected_object,
        } = state.get_mut(world);
        let Ok((parent, path)) = paths.get(entity) else {
            ui.colored_label(get_scheme().error, "Node found");
            return;
        };
        let Ok(mut dashboard) = dashboards.get_mut(path.root) else {
            ui.colored_label(get_scheme().error, "Dashboard not found");
            return;
        };

        let dashboard_entity = dashboard.aux;
        let Some(node) = dashboard.root.get_node_mut(&path.path) else {
            ui.colored_label(
                get_scheme().error,
                format!("Node not found {:?}", path.path),
            );
            return;
        };

        ui.spacing_mut().item_spacing.y = 8.0;
        ui.label("Dashboard");
        ui.separator();

        let mut changed = false;
        changed |= val_editor(ui, "Width", &eql_ctx.0, &mut node.width);
        changed |= val_editor(ui, "Height", &eql_ctx.0, &mut node.height);
        changed |= enum_select(ui, "Flex Direction", &mut node.flex_direction);
        changed |= enum_select(ui, "Justify Content", &mut node.justify_content);
        if changed {
            commands.entity(node.aux).despawn();
            if let Ok(new) = spawn_node(
                Some(parent.0),
                node,
                &eql_ctx.0,
                &mut commands,
                dashboard_entity,
                path.path.clone(),
            ) {
                *selected_object = SelectedObject::DashboardNode { entity: new.aux };
                *node = new;
            }
        }
    }
}

fn val_editor(
    ui: &mut egui::Ui,
    label: &str,
    eql_ctx: &eql::Context,
    val: &mut impeller2_wkt::Val,
) -> bool {
    use impeller2_wkt::Val;
    ui.label(egui::RichText::new(label).color(get_scheme().text_secondary));
    configure_input_with_border(ui.style_mut());

    let current_val_type = match val {
        Val::Auto => "Auto",
        Val::Px(_) => "px",
        Val::Percent(_) => "%",
        Val::Vw(_) => "vw",
        Val::Vh(_) => "vh",
        Val::VMin(_) => "vmin",
        Val::VMax(_) => "vmax",
    };

    let changed = ui
        .with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            let mut changed = false;
            if let Some(new_val) = val_ty_select(ui, current_val_type, val) {
                *val = new_val;
                changed = true;
            }
            ui.add_space(8.0);
            changed |= match val {
                Val::Auto => false,
                Val::Px(eql)
                | Val::Percent(eql)
                | Val::Vw(eql)
                | Val::Vh(eql)
                | Val::VMin(eql)
                | Val::VMax(eql) => {
                    let eql_res = ui
                        .vertical(|ui| {
                            let eql_res = query(ui, eql, impeller2_wkt::QueryType::EQL);
                            eql_autocomplete(ui, &eql_ctx, &eql_res, eql);
                            eql_res
                        })
                        .inner;
                    eql_res.changed()
                }
            };

            changed
        })
        .inner;

    ui.separator();
    changed
}

fn val_ty_select(
    ui: &mut egui::Ui,
    current_val_type: &'static str,
    val: &mut impeller2_wkt::Val,
) -> Option<impeller2_wkt::Val> {
    use impeller2_wkt::Val;
    ui.scope(|ui| {
        configure_combo_box(ui.style_mut());
        let mut selected_val_type = current_val_type;
        egui::ComboBox::from_id_salt(ui.next_auto_id())
            .selected_text(current_val_type)
            .show_ui(ui, |ui| {
                configure_combo_item(ui.style_mut());
                ui.selectable_value(&mut selected_val_type, "Auto", "Auto");
                ui.selectable_value(&mut selected_val_type, "px", "px");
                ui.selectable_value(&mut selected_val_type, "%", "%");
                ui.selectable_value(&mut selected_val_type, "vw", "vw");
                ui.selectable_value(&mut selected_val_type, "vh", "vh");
                ui.selectable_value(&mut selected_val_type, "vmin", "vmin");
                ui.selectable_value(&mut selected_val_type, "vmax", "vmax");
            });
        if selected_val_type != current_val_type {
            let eql = val.eql().to_string();
            Some(match selected_val_type {
                "Auto" => Val::Auto,
                "px" => Val::Px(eql),
                "%" => Val::Percent(eql),
                "vw" => Val::Vw(eql),
                "vh" => Val::Vh(eql),
                "vmin" => Val::VMin(eql),
                "vmax" => Val::VMax(eql),
                _ => {
                    unimplemented!()
                }
            })
        } else {
            None
        }
    })
    .inner
}

fn enum_select<T: strum::VariantNames + FromStr + Copy>(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut T,
) -> bool
where
    &'static str: From<T>,
{
    ui.label(label);
    let current_ty = <&str>::from(*value);
    ui.scope(|ui| {
        configure_combo_box(ui.style_mut());
        let mut selected_ty = current_ty;
        egui::ComboBox::from_id_salt(ui.next_auto_id())
            .selected_text(current_ty)
            .show_ui(ui, |ui| {
                configure_combo_item(ui.style_mut());
                for name in T::VARIANTS {
                    ui.selectable_value(&mut selected_ty, name, *name);
                }
            });
        ui.separator();
        if selected_ty != current_ty {
            *value = selected_ty.parse().map_err(|_| ()).unwrap();
            true
        } else {
            false
        }
    })
    .inner
}

pub trait DashboardExt<T> {
    fn get_node_mut(&mut self, path: &[usize]) -> Option<&mut DashboardNode<T>>;
}

impl<T> DashboardExt<T> for DashboardNode<T> {
    fn get_node_mut(&mut self, path: &[usize]) -> Option<&mut DashboardNode<T>> {
        let Some(index) = path.first() else {
            return Some(self);
        };
        self.children.get_mut(*index)?.get_node_mut(&path[1..])
    }
}
