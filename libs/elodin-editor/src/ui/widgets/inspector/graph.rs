use bevy::{
    ecs::{
        entity::Entity,
        system::{Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    utils::smallvec::SmallVec,
};
use bevy_egui::egui;

use conduit::{query::MetadataStore, ComponentId, EntityId};
use egui::Align;

use crate::ui::{
    colors::{self, with_opacity},
    utils::MarginSides,
    widgets::{button::ECheckboxButton, label::label_with_buttons, plot::GraphState, WidgetSystem},
    EntityData, SettingModal, SettingModalState,
};

use super::InspectorIcons;

#[derive(SystemParam)]
pub struct InspectorGraph<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, MetadataStore>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
}

impl WidgetSystem for InspectorGraph<'_, '_> {
    type Args = (InspectorIcons, Entity, String);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, graph_id, label) = args;

        let InspectorGraph {
            entities,
            mut setting_modal_state,
            metadata_store,
            mut graph_states,
        } = state_mut;

        let graph_label_margin = egui::Margin::same(0.0).top(10.0).bottom(14.0);
        let [add_clicked] = label_with_buttons(
            ui,
            [icons.add],
            label,
            colors::PRIMARY_CREAME,
            graph_label_margin,
        );
        if add_clicked {
            setting_modal_state.0 = Some(SettingModal::Graph(graph_id, None, None));
        }

        ui.separator();

        let Ok(mut graph_state) = graph_states.get_mut(graph_id) else {
            return;
        };

        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(8.0, 8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("LINE WIDTH")
                            .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.add(egui::DragValue::new(&mut graph_state.line_width).speed(0.2))
                    });
                });

                ui.add_space(8.0);
                ui.style_mut().spacing.slider_width = ui.available_size().x;
                ui.style_mut().visuals.widgets.inactive.bg_fill = colors::PRIMARY_ONYX_8;
                ui.add(egui::Slider::new(&mut graph_state.line_width, 1.0..=15.0).show_value(false))
            });

        let mut remove_list: SmallVec<[(EntityId, ComponentId); 1]> = SmallVec::new();
        for (entity_id, components) in &mut graph_state.entities {
            let entity = entities.iter().find(|(eid, _, _, _)| *eid == entity_id);

            if let Some((_, _, _, entity_metadata)) = entity {
                let entity_label_margin = egui::Margin::same(0.0).top(18.0).bottom(4.0);
                let [add_clicked] = label_with_buttons(
                    ui,
                    [icons.add],
                    &entity_metadata.name,
                    colors::PRIMARY_CREAME,
                    entity_label_margin,
                );
                if add_clicked {
                    setting_modal_state.0 =
                        Some(SettingModal::Graph(graph_id, Some(*entity_id), None));
                }

                for (component_id, component_values) in components.iter_mut() {
                    let Some(metadata) = metadata_store.get_metadata(component_id) else {
                        continue;
                    };
                    let component_label = metadata.component_name();
                    let element_names = metadata.element_names();

                    let component_label_margin = egui::Margin::symmetric(0.0, 18.0);
                    let [subtract_clicked] = label_with_buttons(
                        ui,
                        [icons.subtract],
                        component_label,
                        with_opacity(colors::PRIMARY_CREAME, 0.3),
                        component_label_margin,
                    );
                    if subtract_clicked {
                        remove_list.push((*entity_id, *component_id));
                    }

                    component_value(ui, component_values, element_names);
                }
            }
        }
        for (entity_id, component_id) in remove_list.into_iter() {
            graph_state.remove_component(&entity_id, &component_id);
        }
    }
}

fn component_value(
    ui: &mut egui::Ui,
    component_values: &mut [(bool, egui::Color32)],
    element_names: &str,
) {
    let element_names = element_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(Option::Some)
        .chain(std::iter::repeat(None));
    ui.horizontal_wrapped(|ui| {
        for (index, ((enabled, color), element_name)) in
            component_values.iter_mut().zip(element_names).enumerate()
        {
            ui.style_mut().override_font_id =
                Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
            let label = element_name
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("[{index}]"));
            let value_toggle =
                ui.add(ECheckboxButton::new(label.to_string(), *enabled).on_color(*color));

            if value_toggle.clicked() {
                *enabled = !*enabled;
            }
        }
    });
}
