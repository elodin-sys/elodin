use bevy::ecs::system::{Query, Res, ResMut};
use bevy_egui::egui;
use conduit::{query::MetadataStore, GraphId};

use crate::ui::{
    colors::{self, with_opacity},
    theme,
    utils::{self, MarginSides},
    EntityData, GraphsState,
};

use super::{
    button::EButton,
    label::{self, ELabel},
};

pub fn modal_graph(
    ctx: &egui::Context,
    rect: egui::Rect,
    close_icon: egui::TextureId,
    entities_meta: Query<EntityData>,
    mut graph_states: ResMut<GraphsState>,
    metadata_store: Res<MetadataStore>,
    graph_id: GraphId,
) {
    egui::Window::new("UPDATE_GRAPH")
        .title_bar(false)
        .resizable(false)
        .frame(egui::Frame {
            fill: colors::PRIMARY_SMOKE,
            stroke: egui::Stroke::NONE,
            inner_margin: egui::Margin::same(16.0),
            outer_margin: egui::Margin::symmetric(4.0, 0.0),
            ..Default::default()
        })
        .fixed_rect(rect)
        .show(ctx, |ui| {
            let title_margin = egui::Margin::same(8.0).bottom(16.0);
            if label::label_with_button(
                ui,
                close_icon,
                "Add Component",
                colors::PRIMARY_CREAME,
                title_margin,
            ) {
                graph_states.modal_graph = None;
                graph_states.modal_entity = None;
                graph_states.modal_component = None;
            }

            ui.add(egui::Separator::default().spacing(0.0));

            ui.add(
                ELabel::new("ENTITY")
                    .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                    .padding(egui::Margin::same(0.0).top(16.0).bottom(8.0)),
            );

            let selected_entity = entities_meta.iter().find(|(entity_id, _, _, _)| {
                graph_states
                    .modal_entity
                    .is_some_and(|eid| eid == **entity_id)
            });

            let selected_entity_label =
                selected_entity.map_or("NONE", |(_, _, _, metadata)| &metadata.name);

            let width = ui.available_width();

            ui.scope(|ui| {
                theme::configure_combo_box(ui.style_mut());
                egui::ComboBox::from_id_source("ENTITY")
                    .width(width)
                    .selected_text(selected_entity_label)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());

                        ui.selectable_value(&mut graph_states.modal_entity, None, "NONE");

                        for (entity_id, _, _, metadata) in entities_meta.iter() {
                            ui.selectable_value(
                                &mut graph_states.modal_entity,
                                Some(*entity_id),
                                metadata.name.to_string(),
                            );
                        }
                    });
            });

            if let Some((entity_id, _, components, _)) = selected_entity {
                ui.add(
                    ELabel::new("COMPONENT")
                        .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                        .padding(egui::Margin::same(0.0).top(16.0).bottom(8.0)),
                );

                let selected_component = components.0.iter().find(|(component_id, _)| {
                    graph_states
                        .modal_component
                        .is_some_and(|cid| cid == **component_id)
                });

                let selected_component_label = selected_component
                    .map_or("NONE".to_string(), |(component_id, _)| {
                        utils::get_component_label(&metadata_store, component_id)
                    });

                ui.scope(|ui| {
                    theme::configure_combo_box(ui.style_mut());
                    egui::ComboBox::from_id_source("COMPONENT")
                        .width(width)
                        .selected_text(selected_component_label)
                        .show_ui(ui, |ui| {
                            theme::configure_combo_item(ui.style_mut());

                            ui.selectable_value(&mut graph_states.modal_component, None, "NONE");

                            for (component_id, _) in components.0.iter() {
                                ui.selectable_value(
                                    &mut graph_states.modal_component,
                                    Some(*component_id),
                                    utils::get_component_label(&metadata_store, component_id),
                                );
                            }
                        });
                });

                if let Some((component_id, component)) = selected_component {
                    ui.add_space(16.0);

                    let add_component_btn = ui.add(
                        EButton::new("ADD COMPONENT")
                            .color(colors::MINT_DEFAULT)
                            .bg_color(with_opacity(colors::MINT_DEFAULT, 0.05))
                            .stroke(egui::Stroke::new(
                                1.0,
                                with_opacity(colors::MINT_DEFAULT, 0.4),
                            )),
                    );

                    if add_component_btn.clicked() {
                        let values = utils::component_value_to_vec(component)
                            .iter()
                            .enumerate()
                            .map(|_| (true, colors::get_random_color()))
                            .collect::<Vec<(bool, egui::Color32)>>();

                        graph_states.insert_component(&graph_id, entity_id, component_id, values);

                        graph_states.modal_graph = None;
                        graph_states.modal_entity = None;
                        graph_states.modal_component = None;
                    }
                }
            }
        });
}
