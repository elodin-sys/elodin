use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::Children,
    window::Window,
};
use bevy_egui::{EguiContexts, egui};
use impeller2::types::ComponentId;
use impeller2_bevy::{ComponentMetadataRegistry, ComponentPathRegistry};

// Modal system for displaying dialogs and error messages.
// 
// # Examples
// 
// ## Simple Error Dialog
// ```rust
// // In any system with access to SettingModalState
// setting_modal_state.show_error("Error", "Something went wrong!");
// ```
// 
// ## Custom Error Dialog with Multiple Buttons
// ```rust
// use crate::ui::{ErrorDialog, ErrorDialogButton, ErrorDialogAction};
// 
// let dialog = ErrorDialog {
//     title: "Confirm Action".to_string(),
//     message: "Are you sure you want to delete this item?".to_string(),
//     buttons: vec![
//         ErrorDialogButton {
//             text: "Cancel".to_string(),
//             action: ErrorDialogAction::Close,
//         },
//         ErrorDialogButton {
//             text: "Delete".to_string(),
//             action: ErrorDialogAction::Custom("delete".to_string()),
//         },
//     ],
// };
// setting_modal_state.show_error_dialog(dialog);
// ```
// 
// ## Closing a Modal
// ```rust
// setting_modal_state.close();
// ```

use crate::ui::{
    EntityData, InspectorAnchor, SettingModal, SettingModalState, ErrorDialog, ErrorDialogAction,
    colors::get_scheme, images, theme, utils::MarginSides,
};

use super::{
    RootWidgetSystem, WidgetSystemExt,
    button::EButton,
    label::{self, ELabel},
    plot::GraphState,
    widgets::WidgetSystem,
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
                        SettingModal::ErrorDialog(dialog) => {
                            ui.add_widget_with::<ModalErrorDialog>(
                                world,
                                "modal_error_dialog",
                                (close_icon, dialog.clone()),
                            );
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
    component_ids: Query<'w, 's, &'static ComponentId>,
    children: Query<'w, 's, &'static Children>,
}

impl WidgetSystem for ModalUpdateGraph<'_, '_> {
    type Args = egui::TextureId;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: <Self as WidgetSystem>::Args,
    ) {
        let state_mut = state.get_mut(world);
        let close_icon = args;

        let ModalUpdateGraph {
            entities_meta,
            mut setting_modal_state,
            metadata_store,
            path_reg: _path_reg,
            mut graph_states,
            component_ids,
            children,
        } = state_mut;

        let Some(setting_modal) = setting_modal_state.0.as_mut() else {
            return;
        };
        let SettingModal::Graph(m_graph_id, m_component_id) = setting_modal else {
            return;
        };
        

        // Reset modal if Graph was removed
        let Ok(_graph_state) = graph_states.get_mut(*m_graph_id) else {
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
            .find(|(entity_id, _, _, _)| m_component_id.is_some_and(|eid| eid == **entity_id));

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

                    ui.selectable_value(m_component_id, None, "NONE");

                    for (entity_id, _, _, metadata) in entities_meta.iter() {
                        ui.selectable_value(
                            m_component_id,
                            Some(*entity_id),
                            metadata.name.to_string(),
                        );
                    }
                });
        });

        if let Some((_entity_id, bevy_entity, _, _)) = selected_entity {
            ui.add(
                ELabel::new("COMPONENT")
                    .text_color(get_scheme().text_secondary)
                    .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
            );

            // Get available components for the selected entity
            let available_components: Vec<_> = children
                .iter_descendants(bevy_entity)
                .chain(std::iter::once(bevy_entity))
                .filter_map(|child| {
                    let component_id = component_ids.get(child).ok()?;
                    let metadata = metadata_store.get_metadata(component_id)?;
                    Some((component_id, child, metadata))
                })
                .collect();

            let selected_component = available_components
                .iter()
                .find(|(component_id, _, _)| m_component_id.is_some_and(|cid| cid == **component_id));

            let selected_component_label = selected_component
                .map(|(_, _, metadata)| metadata.name.as_ref())
                .unwrap_or_else(|| "NONE");

            ui.scope(|ui| {
                theme::configure_combo_box(ui.style_mut());
                egui::ComboBox::from_id_salt("COMPONENT")
                    .width(width)
                    .selected_text(selected_component_label)
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());

                        ui.selectable_value(m_component_id, None, "NONE");

                        for (component_id, _, metadata) in &available_components {
                            ui.selectable_value(
                                m_component_id,
                                Some(**component_id),
                                metadata.name.clone(),
                            );
                        }
                    });
            });

            if let Some((_component_id, _, _metadata)) = selected_component {
                ui.add_space(16.0);

                let add_component_btn = ui.add(EButton::green("ADD COMPONENT"));

                if add_component_btn.clicked() {
                    // TODO: Implement adding component to graph
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

#[derive(SystemParam)]
pub struct ModalErrorDialog<'w, 's> {
    setting_modal_state: ResMut<'w, SettingModalState>,
    _phantom: std::marker::PhantomData<&'s ()>,
}

impl WidgetSystem for ModalErrorDialog<'_, '_> {
    type Args = (egui::TextureId, ErrorDialog);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: <Self as WidgetSystem>::Args,
    ) {
        let mut state_mut = state.get_mut(world);
        let (close_icon, dialog) = args;

        let title_margin = egui::Margin::same(8).bottom(16.0);
        let [close_clicked] = label::label_with_buttons(
            ui,
            [close_icon],
            &dialog.title,
            get_scheme().text_primary,
            title_margin,
        );
        
        if close_clicked {
            state_mut.setting_modal_state.close();
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        // Display the error message
        ui.add_space(16.0);
        ui.add(
            ELabel::new(&dialog.message)
                .text_color(get_scheme().text_secondary)
                .padding(egui::Margin::same(0).bottom(16.0)),
        );

        // Add buttons
        ui.add_space(16.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for (i, button) in dialog.buttons.iter().enumerate() {
                    let button_ui = ui.add(EButton::new(&button.text));
                    
                    if button_ui.clicked() {
                        match &button.action {
                            ErrorDialogAction::Close => {
                                state_mut.setting_modal_state.close();
                            }
                            ErrorDialogAction::Custom(_action_id) => {
                                // TODO: Handle custom actions
                                // For now, just close the dialog
                                state_mut.setting_modal_state.close();
                            }
                        }
                    }
                    
                    // Add space between buttons (except after the last button)
                    if i < dialog.buttons.len() - 1 {
                        ui.add_space(8.0);
                    }
                }
            });
        });
    }
}
