use std::str::FromStr;

use bevy::{ecs::system::SystemParam, prelude::*};
use impeller2_wkt::{Dashboard, DashboardNode};

use crate::ui::dashboard::DashboardNodePath;
use crate::{
    EqlContext,
    ui::{
        SelectedObject,
        colors::get_scheme,
        dashboard::{NodeUpdaterParams, spawn_node},
        inspector::{eql_textfield, node_color_picker},
        label,
        theme::{configure_combo_box, configure_combo_item, configure_input_with_border},
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorDashboardNode<'w, 's> {
    pub paths: Query<'w, 's, (&'static DashboardNodePath, Option<&'static Children>)>,
    pub dashboards: Query<'w, 's, &'static mut Dashboard<Entity>>,
    pub eql_ctx: Res<'w, EqlContext>,
    pub commands: Commands<'w, 's>,
    pub selected_object: ResMut<'w, SelectedObject>,
    pub node_updater_params: NodeUpdaterParams<'w, 's>,
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
            node_updater_params,
        } = state.get_mut(world);
        let Ok((path, children)) = paths.get(entity) else {
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

        label::editable_label_with_buttons(
            ui,
            [],
            node.label.get_or_insert_with(|| {
                if path.path.is_empty() {
                    "Dashboard".to_string()
                } else {
                    "node".to_string()
                }
            }),
            get_scheme().text_primary,
            egui::Margin::ZERO,
        );
        ui.separator();

        let mut changed = false;
        changed |= val_editor(ui, "Width", &eql_ctx.0, &mut node.width);
        changed |= val_editor(ui, "Height", &eql_ctx.0, &mut node.height);
        changed |= node_color_picker(ui, "Background", &mut node.color);
        changed |= eql_editor(ui, "Text", &eql_ctx.0, node.text.get_or_insert_default());
        changed |= font_size_editor(ui, "Font Size", &mut node.font_size);
        changed |= node_color_picker(ui, "Text Color", &mut node.text_color);
        changed |= enum_select(ui, "Flex Direction", &mut node.flex_direction);
        changed |= enum_select(ui, "Flex Wrap", &mut node.flex_wrap);
        changed |= enum_select(ui, "Justify Content", &mut node.justify_content);
        changed |= enum_select(ui, "Align Items", &mut node.align_items);
        changed |= enum_select(ui, "Align Content", &mut node.align_content);
        changed |= enum_select(ui, "Align Self", &mut node.align_self);
        changed |= enum_select(ui, "Justify Items", &mut node.justify_items);
        changed |= val_editor(ui, "Min Width", &eql_ctx.0, &mut node.min_width);
        changed |= val_editor(ui, "Min Height", &eql_ctx.0, &mut node.min_height);
        changed |= val_editor(ui, "Max Width", &eql_ctx.0, &mut node.max_width);
        changed |= val_editor(ui, "Max Height", &eql_ctx.0, &mut node.max_height);
        changed |= val_editor(ui, "left", &eql_ctx.0, &mut node.left);
        changed |= val_editor(ui, "right", &eql_ctx.0, &mut node.right);
        changed |= val_editor(ui, "top", &eql_ctx.0, &mut node.top);
        changed |= val_editor(ui, "bottom", &eql_ctx.0, &mut node.bottom);

        if changed {
            if let Some(children) = children {
                for child in children.iter() {
                    commands.entity(child).despawn();
                }
            }
            let mut entity = commands.entity(node.aux);
            if let Ok(new) = spawn_node(
                node,
                &eql_ctx.0,
                &mut entity,
                dashboard_entity,
                path.path.clone(),
                &node_updater_params,
            ) {
                *selected_object = SelectedObject::DashboardNode { entity: new.aux };
                *node = new;
            }
        }
    }
}

fn eql_editor(ui: &mut egui::Ui, label: &str, eql_ctx: &eql::Context, eql: &mut String) -> bool {
    ui.label(egui::RichText::new(label).color(get_scheme().text_secondary));
    configure_input_with_border(ui.style_mut());
    let changed = eql_textfield(ui, true, eql_ctx, eql).changed();
    ui.separator();
    changed
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
            let mut empty = String::new();
            let enabled = !matches!(val, Val::Auto);
            let eql = match val {
                Val::Auto => &mut empty,
                Val::Px(eql)
                | Val::Percent(eql)
                | Val::Vw(eql)
                | Val::Vh(eql)
                | Val::VMin(eql)
                | Val::VMax(eql) => eql,
            };

            let eql_res = eql_textfield(ui, enabled, eql_ctx, eql);
            changed | eql_res.changed()
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
        ui.style_mut().spacing.combo_width = 60.0;
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
    ui.label(egui::RichText::new(label).color(get_scheme().text_secondary));
    let current_ty = <&str>::from(*value);
    ui.scope(|ui| {
        ui.style_mut().spacing.combo_width = ui.available_size().x;
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
        ui.add_space(8.0);
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

fn font_size_editor(ui: &mut egui::Ui, label: &str, font_size: &mut f32) -> bool {
    ui.label(egui::RichText::new(label).color(get_scheme().text_secondary));
    configure_input_with_border(ui.style_mut());

    let mut changed = false;
    ui.horizontal(|ui| {
        let response = ui.add(
            egui::DragValue::new(font_size)
                .range(1.0..=200.0)
                .speed(0.5)
                .suffix(" px"),
        );
        changed = response.changed();
    });

    ui.separator();
    changed
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
