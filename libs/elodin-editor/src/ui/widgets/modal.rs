use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    window::Window,
};
use bevy_egui::{EguiContexts, egui};
use impeller2_bevy::{ComponentMetadataRegistry, ComponentPath, ComponentPathRegistry};

use crate::ui::{
    EntityData, InspectorAnchor, SettingModal, SettingModalState, colors::get_scheme, images,
    theme, utils::MarginSides,
};

use super::{
    RootWidgetSystem, WidgetSystem, WidgetSystemExt,
    button::EButton,
    label::{self, ELabel},
    plot::{GraphState, default_component_values},
};

#[derive(SystemParam)]
pub struct ModalWithSettings<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    window: Query<'w, 's, &'static Window>,
    inspector_anchor: Res<'w, InspectorAnchor>,
    setting_modal_state: Res<'w, SettingModalState>,
}

impl RootWidgetSystem for ModalWithSettings<'_, '_> {
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
        let setting_modal_state = state_mut.setting_modal_state;

        let modal_size = egui::vec2(280.0, 480.0);

        let modal_rect = if let Some(inspector_anchor) = inspector_anchor.0 {
            egui::Rect::from_min_size(
                egui::pos2(inspector_anchor.x - modal_size.x, inspector_anchor.y),
                modal_size,
            )
        } else {
            let window = window.iter().next().unwrap();
            egui::Rect::from_center_size(
                egui::pos2(
                    window.resolution.width() / 2.0,
                    window.resolution.height() / 2.0,
                ),
                modal_size,
            )
        };

        if let Some(setting_modal_state) = setting_modal_state.0.clone() {
            let close_icon = contexts.add_image(images.icon_close.clone_weak());

            egui::Window::new("SETTING_MODAL")
                .title_bar(false)
                .resizable(false)
                .frame(egui::Frame {
                    fill: get_scheme().bg_secondary,
                    stroke: egui::Stroke::NONE,
                    inner_margin: egui::Margin::same(16),
                    outer_margin: egui::Margin::symmetric(4, 0),
                    ..Default::default()
                })
                .fixed_rect(modal_rect)
                .show(ctx, |ui| {
                    match setting_modal_state {
                        SettingModal::Graph(_, _) => {
                            ui.add_widget_with::<ModalUpdateGraph>(
                                world,
                                "modal_update_graph",
                                close_icon,
                            );
                        }
                        SettingModal::GraphRename(_, _) => {
                            // TODO: Rename graph
                        }
                    }
                });
        }
    }
}

#[derive(SystemParam)]
pub struct ModalUpdateGraph<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    path_reg: Res<'w, ComponentPathRegistry>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
}

impl WidgetSystem for ModalUpdateGraph<'_, '_> {
    type Args = egui::TextureId;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let close_icon = args;

        let ModalUpdateGraph {
            entities_meta,
            mut setting_modal_state,
            metadata_store,
            path_reg,
            mut graph_states,
        } = state_mut;

        let Some(setting_modal) = setting_modal_state.0.as_mut() else {
            return;
        };
        let SettingModal::Graph(m_graph_id, m_component_id) = setting_modal else {
            return;
        };

        // Reset modal if Graph was removed
        let Ok(mut graph_state) = graph_states.get_mut(*m_graph_id) else {
            setting_modal_state.0 = None;
            return;
        };

        let title_margin = egui::Margin::same(8).bottom(16.0);
        let [close_clicked] = label::label_with_buttons(
            ui,
            [close_icon],
            "Add Component",
            get_scheme().text_primary,
            title_margin,
        );
        if close_clicked {
            setting_modal_state.0 = None;
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        ui.add(
            ELabel::new("ENTITY")
                .text_color(get_scheme().text_secondary)
                .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
        );

        let selected_entity = entities_meta
            .iter()
            .find(|(entity_id, _, _, _)| m_entity_id.is_some_and(|eid| eid == **entity_id));

        let selected_entity_label =
            selected_entity.map_or("NONE", |(_, _, _, metadata)| &metadata.name);

        let width = ui.available_width();

        ui.scope(|ui| {
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("ENTITY")
                .width(width)
                .selected_text(selected_entity_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_entity_id, None, "NONE");

                    for (entity_id, _, _, metadata) in entities_meta.iter() {
                        ui.selectable_value(
                            m_entity_id,
                            Some(*entity_id),
                            metadata.name.to_string(),
                        );
                    }
                });
        });

        if let Some((entity_id, _, components, _)) = selected_entity {
            ui.add(
                ELabel::new("COMPONENT")
                    .text_color(get_scheme().text_secondary)
                    .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
            );

            let selected_component = components
                .0
                .iter()
                .find(|(component_id, _)| m_component_id.is_some_and(|cid| cid == **component_id));

            let selected_component_label = selected_component
                .and_then(|(component_id, _)| metadata_store.get_metadata(component_id))
                .map(|m| m.name.as_ref())
                .unwrap_or_else(|| "NONE");

            ui.scope(|ui| {
                theme::configure_combo_box(ui.style_mut());
                egui::ComboBox::from_id_salt("COMPONENT")
                    .width(width)
                    .selected_text(selected_component_label)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());

                        ui.selectable_value(m_component_id, None, "NONE");

                        for (component_id, _) in components.0.iter() {
                            let Some(metadata) = metadata_store.get_metadata(component_id) else {
                                continue;
                            };
                            ui.selectable_value(
                                m_component_id,
                                Some(*component_id),
                                metadata.name.clone(),
                            );
                        }
                    });
            });

            if let Some((component_id, component)) = selected_component {
                ui.add_space(16.0);

                let add_component_btn = ui.add(EButton::green("ADD COMPONENT"));

                if add_component_btn.clicked() {
                    // let values = default_component_values(entity_id, component_id, component);
                    // let component_path = path_reg
                    //     .get(component_id)
                    //     .cloned()
                    //     .unwrap_or_else(|| ComponentPath::from_name(&metadata.name));
                    // graph_state.insert_component(component_path, values);

                    // setting_modal_state.0 = None;
                }
            }
        }
    }
}
