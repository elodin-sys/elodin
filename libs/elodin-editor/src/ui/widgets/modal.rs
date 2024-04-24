use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    window::Window,
};
use bevy_egui::{egui, EguiContexts};
use conduit::{query::MetadataStore, GraphId};

use crate::ui::{
    colors::{self, with_opacity},
    images, theme,
    utils::MarginSides,
    EntityData, GraphsState, InspectorAnchor,
};

use super::{
    button::EButton,
    label::{self, ELabel},
    RootWidgetSystem, WidgetSystem, WidgetSystemExt,
};

#[derive(SystemParam)]
pub struct ModalUpdateGraph<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    window: Query<'w, 's, &'static Window>,
    inspector_anchor: Res<'w, InspectorAnchor>,
    graph_states: ResMut<'w, GraphsState>,
}

impl RootWidgetSystem for ModalUpdateGraph<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let mut contexts = state_mut.contexts;
        let images = state_mut.images;
        let window = state_mut.window;
        let inspector_anchor = state_mut.inspector_anchor;
        let graph_states = state_mut.graph_states;

        let modal_size = egui::vec2(280.0, 480.0);

        let modal_rect = if let Some(inspector_anchor) = inspector_anchor.0 {
            egui::Rect::from_min_size(
                egui::pos2(inspector_anchor.x - modal_size.x, inspector_anchor.y),
                modal_size,
            )
        } else {
            let window = window.single();
            egui::Rect::from_center_size(
                egui::pos2(
                    window.resolution.width() / 2.0,
                    window.resolution.height() / 2.0,
                ),
                modal_size,
            )
        };

        if let Some(graph_id) = graph_states.modal_graph {
            let close_icon = contexts.add_image(images.icon_close.clone_weak());

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
                .fixed_rect(modal_rect)
                .show(ctx, |ui| {
                    ui.add_widget_with::<ModalUpdateGraphContent>(
                        world,
                        "modal_update_graph_content",
                        (graph_id, close_icon),
                    );
                });
        }
    }
}

#[derive(SystemParam)]
pub struct ModalUpdateGraphContent<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    graph_states: ResMut<'w, GraphsState>,
    metadata_store: Res<'w, MetadataStore>,
}

impl WidgetSystem for ModalUpdateGraphContent<'_, '_> {
    type Args = (GraphId, egui::TextureId);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (graph_id, close_icon) = args;

        let mut graph_states = state_mut.graph_states;
        let entities_meta = state_mut.entities_meta;
        let metadata_store = state_mut.metadata_store;

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
                .and_then(|(component_id, _)| metadata_store.get_metadata(component_id))
                .map(|m| m.component_name())
                .unwrap_or_else(|| "NONE");

            ui.scope(|ui| {
                theme::configure_combo_box(ui.style_mut());
                egui::ComboBox::from_id_source("COMPONENT")
                    .width(width)
                    .selected_text(selected_component_label)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());

                        ui.selectable_value(&mut graph_states.modal_component, None, "NONE");

                        for (component_id, _) in components.0.iter() {
                            let Some(metadata) = metadata_store.get_metadata(component_id) else {
                                continue;
                            };
                            ui.selectable_value(
                                &mut graph_states.modal_component,
                                Some(*component_id),
                                metadata.component_name(),
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
                    let values =
                        GraphsState::default_component_values(entity_id, component_id, component);
                    graph_states.insert_component(&graph_id, entity_id, component_id, values);

                    graph_states.modal_graph = None;
                    graph_states.modal_entity = None;
                    graph_states.modal_component = None;
                }
            }
        }
    }
}
