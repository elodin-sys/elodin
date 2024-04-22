use bevy::ecs::system::{Query, Res, ResMut};
use bevy_egui::egui;

use conduit::{query::MetadataStore, GraphId};

use crate::ui::{
    colors::{self, with_opacity},
    utils::MarginSides,
    widgets::{button::ECheckboxButton, label::label_with_button},
    EntityData, GraphsState,
};

use super::InspectorIcons;

pub fn inspector(
    ui: &mut egui::Ui,
    graph_id: &GraphId,
    label: &str,
    entities: &Query<EntityData>,
    graphs_state: &mut ResMut<GraphsState>,
    metadata_store: &Res<MetadataStore>,
    icons: InspectorIcons,
) {
    let graph_label_margin = egui::Margin::same(0.0).top(10.0).bottom(14.0);
    if label_with_button(
        ui,
        icons.add,
        label,
        colors::PRIMARY_CREAME,
        graph_label_margin,
    ) {
        graphs_state.modal_graph = Some(*graph_id);
    }

    ui.separator();

    let ro_graphs_state = graphs_state.clone();
    let Some(graph_state) = ro_graphs_state.graphs.get(graph_id) else {
        return;
    };

    for (entity_id, components) in graph_state {
        let entity = entities.iter().find(|(eid, _, _, _)| *eid == entity_id);

        if let Some((_, _, _, entity_metadata)) = entity {
            let entity_label_margin = egui::Margin::same(0.0).top(18.0).bottom(4.0);
            if label_with_button(
                ui,
                icons.add,
                &entity_metadata.name,
                colors::PRIMARY_CREAME,
                entity_label_margin,
            ) {
                graphs_state.modal_graph = Some(*graph_id);
                graphs_state.modal_entity = Some(*entity_id);
                graphs_state.modal_component = None;
            }

            for (component_id, component_values) in components {
                let Some(metadata) = metadata_store.get_metadata(component_id) else {
                    continue;
                };
                let component_label = metadata.component_name();
                let element_names = metadata.element_names();

                let component_label_margin = egui::Margin::symmetric(0.0, 18.0);
                if label_with_button(
                    ui,
                    icons.subtract,
                    &component_label,
                    with_opacity(colors::PRIMARY_CREAME, 0.3),
                    component_label_margin,
                ) {
                    graphs_state.remove_component(graph_id, entity_id, component_id);
                }

                let mut new_component_values = Vec::new();
                let mut replace_component = false;

                component_value(
                    ui,
                    &mut new_component_values,
                    &mut replace_component,
                    component_values,
                    element_names,
                );

                if replace_component {
                    graphs_state.insert_component(
                        graph_id,
                        entity_id,
                        component_id,
                        new_component_values,
                    );
                }
            }
        }
    }
}

fn component_value(
    ui: &mut egui::Ui,
    new_component_values: &mut Vec<(bool, egui::Color32)>,
    replace_component: &mut bool,
    component_values: &[(bool, egui::Color32)],
    element_names: &str,
) {
    let element_names = element_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(Option::Some)
        .chain(std::iter::repeat(None));
    ui.horizontal_wrapped(|ui| {
        for (index, ((enabled, color), element_name)) in
            component_values.iter().zip(element_names).enumerate()
        {
            ui.style_mut().override_font_id =
                Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
            let label = element_name
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("[{index}]"));
            let value_toggle =
                ui.add(ECheckboxButton::new(label.to_string(), *enabled).on_color(*color));

            if value_toggle.clicked() {
                new_component_values.push((!*enabled, *color));
                *replace_component = true;
            } else {
                new_component_values.push((*enabled, *color));
            }
        }
    });
}
