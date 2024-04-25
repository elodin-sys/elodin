use bevy::ecs::system::{Query, Res, ResMut};
use bevy_egui::egui;

use conduit::{bevy::MaxTick, query::MetadataStore, GraphId};

use crate::ui::{
    colors::{self, with_opacity},
    theme,
    utils::MarginSides,
    widgets::{
        button::ECheckboxButton,
        label::{label_with_button, label_with_n_buttons},
        timeline::tagged_range::TaggedRanges,
    },
    EntityData, GraphsState, SettingModal, SettingModalState,
};

use super::InspectorIcons;

#[allow(clippy::too_many_arguments)]
pub fn inspector(
    ui: &mut egui::Ui,
    graph_id: &GraphId,
    label: &str,
    entities: &Query<EntityData>,
    graphs_state: &mut ResMut<GraphsState>,
    setting_modal_state: &mut ResMut<SettingModalState>,
    tagged_ranges: &mut ResMut<TaggedRanges>,
    max_tick: Res<MaxTick>,
    metadata_store: &Res<MetadataStore>,
    icons: InspectorIcons,
) {
    let graph_label_margin = egui::Margin::same(0.0).top(10.0).bottom(14.0);
    let btn_clicked = label_with_n_buttons(
        ui,
        &[
            icons.add,
            // icons.setting,
        ],
        label,
        colors::PRIMARY_CREAME,
        graph_label_margin,
    );

    match btn_clicked {
        0 => {
            setting_modal_state.0 = Some(SettingModal::Graph(*graph_id, None, None));
        }
        1 => {
            println!("edit graph");
        }
        _ => {}
    }

    ui.separator();

    let ro_graphs_state = graphs_state.clone();
    let Some(graph_state) = ro_graphs_state.0.get(graph_id) else {
        return;
    };

    if let Some(graph_state) = graphs_state.0.get_mut(graph_id) {
        let selected_range_label = graph_state
            .range_id
            .as_ref()
            .and_then(|rid| tagged_ranges.0.get(rid))
            .map_or("Default", |r| &r.label)
            .to_owned();

        let ro_tagged_ranges = tagged_ranges.0.clone();

        let btn_clicked = label_with_n_buttons(
            ui,
            &[icons.add, icons.setting, icons.subtract],
            "RANGE",
            colors::PRIMARY_CREAME,
            egui::Margin::same(0.0).top(18.0).bottom(4.0),
        );

        match btn_clicked {
            0 => {
                graph_state.range_id = Some(tagged_ranges.create_range(max_tick.0));
            }
            1 => {
                if let Some(current_range_id) = &graph_state.range_id {
                    if let Some(current_range) = ro_tagged_ranges.get(current_range_id) {
                        setting_modal_state.0 = Some(SettingModal::RangeEdit(
                            current_range_id.clone(),
                            current_range.label.to_owned(),
                            current_range.color.to_owned(),
                        ));
                    }
                }
            }
            2 => {
                if let Some(current_range_id) = &graph_state.range_id {
                    tagged_ranges.remove_range(current_range_id);
                    graph_state.range_id = None;
                }
            }
            _ => {}
        }

        ui.scope(|ui| {
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_source("RANGE")
                .width(ui.available_width())
                .selected_text(selected_range_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(&mut graph_state.range_id, None, "Default");

                    for (range_id, range) in ro_tagged_ranges {
                        ui.selectable_value(&mut graph_state.range_id, Some(range_id), range.label);
                    }
                });
        });
    }

    for (entity_id, components) in &graph_state.entities {
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
                setting_modal_state.0 =
                    Some(SettingModal::Graph(*graph_id, Some(*entity_id), None));
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
                    component_label,
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
