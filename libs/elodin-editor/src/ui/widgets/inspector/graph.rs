use bevy::ecs::system::{Query, Res, ResMut};
use bevy_egui::egui::{self, Align};

use conduit::{query::MetadataStore, GraphId};

use crate::ui::{
    colors::{self},
    utils::{self, MarginSides},
    widgets::label::ELabel,
    EntityData, GraphsState,
};

pub fn inspector(
    ui: &mut egui::Ui,
    graph_id: &GraphId,
    label: &str,
    entities: &Query<EntityData>,
    graphs_state: &mut ResMut<GraphsState>,
    metadata_store: &Res<MetadataStore>,
) {
    ui.add(
        ELabel::new(label)
            .padding(egui::Margin::same(8.0).bottom(24.0))
            .bottom_stroke(ELabel::DEFAULT_STROKE)
            .margin(egui::Margin::same(0.0).bottom(16.0)),
    );

    if ui.button("Add Component").clicked() {
        graphs_state.modal_graph = Some(*graph_id);
    }

    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(8.0, 8.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.style_mut().spacing.item_spacing = egui::vec2(8.0, 8.0);

                let ro_graphs_state = graphs_state.clone();

                let Some(graph_state) = ro_graphs_state.graphs.get(graph_id) else {
                    return;
                };

                for (entity_id, components) in graph_state {
                    let entity = entities.iter().find(|(eid, _, _, _)| *eid == entity_id);

                    if let Some((_, _, _, entity_metadata)) = entity {
                        ui.label(entity_metadata.name.to_string());

                        for (component_id, component_values) in components {
                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "  {}",
                                    utils::get_component_label(metadata_store, component_id)
                                ));
                                ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                                    if ui.button("-").clicked() {
                                        println!("remove {graph_id:?} / {entity_id:?} / {component_id:?}");
                                        graphs_state.remove_component(
                                            graph_id,
                                            entity_id,
                                            component_id,
                                        );
                                    }
                                });
                            });

                            ui.horizontal(|ui| {
                                let mut new_component_values = Vec::new();
                                let mut clicked = false;

                                for (index, (enabled, color)) in component_values.iter().enumerate()
                                {
                                    let display_color = if *enabled { *color } else { colors::BLACK_BLACK_600 };
                                    let label = egui::RichText::new(format!("[{index}]"))   
                                        .color(display_color);

                                    if ui.button(label).clicked() {
                                        new_component_values.push((!*enabled, *color));

                                        clicked = true;
                                    }
                                    else {
                                        new_component_values.push((*enabled, *color));
                                    }
                                }

                                if clicked {
                                    graphs_state.insert_component(
                                        graph_id,
                                        entity_id,
                                        component_id,
                                        new_component_values,
                                    );
                                }
                            });
                        }
                    }
                }
            });
        });
}
