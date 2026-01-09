use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    prelude::In,
    window::{PrimaryWindow, Window},
};
use bevy_egui::{EguiContexts, egui, EguiTextureHandle};

// Modal system for displaying dialogs and messages.
//
// # Examples
//
// ## Simple Message Dialog
// ```rust
// // In any system with access to ModalDialog
// fn my_system(mut modal_dialog: ModalDialog) {
//     modal_dialog.show_message("Info", "Operation completed successfully!");
// }
// ```
//
// ## Custom Dialog with Multiple Buttons
// ```rust
// use crate::ui::{Dialog, DialogButton, DialogAction};
//
// fn my_system(mut modal_dialog: ModalDialog) {
//     let dialog = Dialog {
//         id: "confirm_delete".to_string(),
//         title: "Confirm Action".to_string(),
//         message: "Are you sure you want to delete this item?".to_string(),
//         buttons: vec![
//             DialogButton {
//                 text: "Cancel".to_string(),
//                 action: DialogAction::Close,
//             },
//             DialogButton {
//                 text: "Delete".to_string(),
//                 action: DialogAction::Custom("delete".to_string()),
//             },
//         ],
//     };
//     modal_dialog.show_dialog(dialog);
// }
// ```
//
// ## Listening for Dialog Events
// ```rust
// use bevy::prelude::*;
// use crate::ui::{DialogEvent, DialogAction};
//
// fn handle_dialog_events(mut dialog_events: MessageReader<DialogEvent>) {
//     for event in dialog_events.read() {
//         match &event.action {
//             DialogAction::Close => {
//                 println!("Dialog '{}' was closed", event.id);
//             }
//             DialogAction::Custom(action_id) => {
//                 println!("Custom action '{}' triggered for dialog '{}'", action_id, event.id);
//                 match (event.id.as_str(), action_id.as_str()) {
//                     ("confirm_delete", "delete") => {
//                         // Handle delete confirmation
//                     }
//                     ("save_changes", "save") => {
//                         // Handle save confirmation
//                     }
//                     _ => {
//                         // Handle other combinations
//                     }
//                 }
//             }
//         }
//     }
// }
// ```
//
// ## Closing a Modal
// ```rust
// setting_modal_state.close();
// ```

use crate::ui::{
    Dialog, DialogAction, DialogButton, DialogEvent, FocusedWindow, SettingModal,
    SettingModalState, colors::get_scheme, images, tiles::WindowState, utils::MarginSides,
};
use bevy::prelude::*;

use super::{RootWidgetSystem, WidgetSystemExt, button::EButton, label, widgets::WidgetSystem};

#[derive(SystemParam)]
pub struct ModalWithSettings<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    window: Query<'w, 's, &'static Window>,
    window_states: Query<'w, 's, &'static WindowState>,
    focused_window: Res<'w, FocusedWindow>,
    primary_window: Query<'w, 's, Entity, With<PrimaryWindow>>,
    setting_modal_state: Res<'w, SettingModalState>,
}

pub mod action {
    use super::*;

    /// Any system producing a `Result` may pipe to this so that any errors produced
    /// will show a dialog.
    pub fn dialog_err<E: std::error::Error>(
        In(result): In<Result<(), E>>,
        mut modal_dialog: ModalDialog,
    ) {
        if let Err(e) = result {
            bevy::log::warn!("Show error dialog: {}", e);
            modal_dialog.show_message("Error", format!("{}", e));
        }
    }
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
        let window_states = state_mut.window_states;
        let focused_window = state_mut.focused_window;
        let primary_window = state_mut.primary_window;
        let setting_modal_state = state_mut.setting_modal_state;

        let modal_size = egui::vec2(400.0, 480.0);

        let target_window = focused_window.0.or_else(|| primary_window.iter().next());
        let inspector_anchor = target_window
            .and_then(|entity| window_states.get(entity).ok())
            .and_then(|state| state.ui_state.inspector_anchor.0);

        let modal_rect = if let Some(inspector_anchor) = inspector_anchor {
            egui::Rect::from_min_size(
                egui::pos2(inspector_anchor.x - modal_size.x, inspector_anchor.y),
                modal_size,
            )
        } else {
            let window = target_window
                .and_then(|entity| window.get(entity).ok())
                .or_else(|| window.iter().next())
                .expect("no window available");
            egui::Rect::from_center_size(
                egui::pos2(
                    window.resolution.width() / 2.0,
                    window.resolution.height() / 2.0,
                ),
                modal_size,
            )
        };

        if let Some(setting_modal_state) = setting_modal_state.0.clone() {
            let close_icon = contexts.add_image(EguiTextureHandle::Weak(images.icon_close.id()));

            egui::Window::new("SETTING_MODAL")
                .title_bar(false)
                .resizable(false)
                .frame(egui::Frame {
                    fill: get_scheme().bg_secondary,
                    stroke: egui::Stroke {
                        width: 1.0,
                        color: get_scheme().text_secondary,
                    },
                    inner_margin: egui::Margin::same(16),
                    outer_margin: egui::Margin::symmetric(4, 0),
                    ..Default::default()
                })
                .fixed_rect(modal_rect)
                .show(ctx, |ui| {
                    let SettingModal::Dialog(dialog) = setting_modal_state;
                    ui.add_widget_with::<ModalDialog>(
                        world,
                        "modal_dialog",
                        (close_icon, dialog.clone()),
                    );
                });
        }
    }
}

#[derive(SystemParam)]
pub struct ModalDialog<'w, 's> {
    setting_modal_state: ResMut<'w, SettingModalState>,
    dialog_events: MessageWriter<'w, DialogEvent>,
    _phantom: std::marker::PhantomData<&'s ()>,
}

impl WidgetSystem for ModalDialog<'_, '_> {
    type Args = (egui::TextureId, Dialog);
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
            state_mut.dialog_events.write(DialogEvent {
                action: DialogAction::Close,
                id: dialog.id.clone(),
            });
            state_mut.setting_modal_state.close();
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        // Display the error message
        ui.add_space(16.0);
        ui.monospace(&dialog.message);

        // Add buttons
        ui.add_space(16.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for (i, button) in dialog.buttons.iter().enumerate() {
                    let button_ui = ui.add(EButton::new(&button.text));

                    if button_ui.clicked() {
                        match &button.action {
                            DialogAction::Close => {
                                state_mut.dialog_events.write(DialogEvent {
                                    action: DialogAction::Close,
                                    id: dialog.id.clone(),
                                });
                                state_mut.setting_modal_state.close();
                            }
                            DialogAction::Custom(action_id) => {
                                state_mut.dialog_events.write(DialogEvent {
                                    action: DialogAction::Custom(action_id.clone()),
                                    id: dialog.id.clone(),
                                });
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

impl ModalDialog<'_, '_> {
    /// Show a simple message dialog with just a close button
    pub fn show_message(&mut self, title: impl Into<String>, message: impl Into<String>) {
        let id = format!(
            "message_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );
        self.setting_modal_state.0 = Some(SettingModal::Dialog(Dialog {
            id,
            title: title.into(),
            message: message.into(),
            buttons: vec![DialogButton {
                text: "OK".to_string(),
                action: DialogAction::Close,
            }],
        }));
    }

    /// Show a custom dialog with multiple buttons
    pub fn show(&mut self, dialog: Dialog) {
        self.setting_modal_state.0 = Some(SettingModal::Dialog(dialog));
    }

    /// Any system producing a `Result` may pipe to this so that any errors produced
    /// will show a dialog.
    pub fn dialog_err<T, E: std::error::Error>(
        &mut self,
        title: impl Into<String>,
        result: Result<T, E>,
    ) -> Result<T, E> {
        if let Err(e) = &result {
            self.show_message(title, format!("{}", e));
        }
        result
    }

    /// Any system producing a `Result` may pipe to this so that any errors produced
    /// will show a dialog.
    pub fn dialog_error<E: std::fmt::Display>(&mut self, title: impl Into<String>, error: &E) {
        self.show_message(title, format!("{}", error));
    }
}
