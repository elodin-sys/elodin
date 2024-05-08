use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    window::Window,
};
use bevy_egui::{egui, EguiContexts};
use conduit::query::MetadataStore;

use crate::ui::{
    colors::{self, with_opacity},
    images, theme,
    utils::MarginSides,
    widgets::plot::GraphsState,
    EntityData, InspectorAnchor, SettingModal, SettingModalState,
};

use super::{
    button::{EButton, EImageButton},
    label::{self, ELabel},
    timeline::tagged_range::TaggedRanges,
    RootWidgetSystem, WidgetSystem, WidgetSystemExt,
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
            let window = window.single();
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
            let color_icon = contexts.add_image(images.icon_lightning.clone_weak());

            egui::Window::new("SETTING_MODAL")
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
                    match setting_modal_state {
                        SettingModal::Graph(_, _, _) => {
                            ui.add_widget_with::<ModalUpdateGraph>(
                                world,
                                "modal_update_graph",
                                close_icon,
                            );
                        }
                        SettingModal::GraphRename(_, _) => {
                            // TODO: Rename graph
                        }
                        SettingModal::RangeEdit(_, _, _) => {
                            ui.add_widget_with::<ModalUpdateRangeName>(
                                world,
                                "modal_update_range_name",
                                (color_icon, close_icon),
                            );
                        }
                    }
                });
        }
    }
}

#[derive(SystemParam)]
pub struct ModalUpdateRangeName<'w> {
    setting_modal_state: ResMut<'w, SettingModalState>,
    tagged_ranges: ResMut<'w, TaggedRanges>,
}

impl WidgetSystem for ModalUpdateRangeName<'_> {
    type Args = (egui::TextureId, egui::TextureId);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let (color_icon, close_icon) = args;

        let mut setting_modal_state = state_mut.setting_modal_state;
        let mut tagged_ranges = state_mut.tagged_ranges;

        let Some(setting_modal) = setting_modal_state.0.as_mut() else {
            return;
        };
        let SettingModal::RangeEdit(m_range_id, m_range_label, m_range_color) = setting_modal
        else {
            return;
        };

        let Some(current_range) = tagged_ranges.0.get_mut(m_range_id) else {
            // Reset modal if Range was removed
            setting_modal_state.0 = None;
            return;
        };

        if label::label_with_button(
            ui,
            close_icon,
            "Range Settings",
            colors::PRIMARY_CREAME,
            egui::Margin::same(8.0).bottom(16.0),
        ) {
            setting_modal_state.0 = None;
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        ui.add(
            ELabel::new("RANGE")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0.0).top(16.0).bottom(8.0)),
        );

        ui.horizontal(|ui| {
            egui::Frame::none()
                .outer_margin(egui::Margin::symmetric(8.0, 16.0))
                .show(ui, |ui| {
                    let color_btn = EImageButton::new(color_icon)
                        .scale(1.4, 1.4)
                        .image_tint(*m_range_color);

                    if ui.add(color_btn).clicked() {
                        *m_range_color =
                            colors::get_color_by_index_solid(fastrand::u32(1..) as usize);
                    }
                });

            ui.scope(|ui| {
                theme::configure_input_with_border(ui.style_mut());
                ui.add(egui::TextEdit::singleline(m_range_label).margin(egui::vec2(16.0, 16.0)));
            })
        });

        ui.add_space(16.0);

        let rename_btn = ui.add(
            EButton::new("UPDATE")
                .color(colors::MINT_DEFAULT)
                .bg_color(with_opacity(colors::MINT_DEFAULT, 0.05))
                .stroke(egui::Stroke::new(
                    1.0,
                    with_opacity(colors::MINT_DEFAULT, 0.4),
                )),
        );

        if rename_btn.clicked() {
            current_range.label = m_range_label.to_string();
            current_range.color = *m_range_color;
            setting_modal_state.0 = None;
        }
    }
}

#[derive(SystemParam)]
pub struct ModalUpdateGraph<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    graph_states: ResMut<'w, GraphsState>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, MetadataStore>,
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

        let mut graph_states = state_mut.graph_states;
        let mut setting_modal_state = state_mut.setting_modal_state;
        let entities_meta = state_mut.entities_meta;
        let metadata_store = state_mut.metadata_store;

        let Some(setting_modal) = setting_modal_state.0.as_mut() else {
            return;
        };
        let SettingModal::Graph(m_graph_id, m_entity_id, m_component_id) = setting_modal else {
            return;
        };

        // Reset modal if Graph was removed
        if !graph_states.contains_graph(m_graph_id) {
            setting_modal_state.0 = None;
            return;
        }

        let title_margin = egui::Margin::same(8.0).bottom(16.0);
        if label::label_with_button(
            ui,
            close_icon,
            "Add Component",
            colors::PRIMARY_CREAME,
            title_margin,
        ) {
            setting_modal_state.0 = None;
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        ui.add(
            ELabel::new("ENTITY")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0.0).top(16.0).bottom(8.0)),
        );

        let selected_entity = entities_meta
            .iter()
            .find(|(entity_id, _, _, _)| m_entity_id.is_some_and(|eid| eid == **entity_id));

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
                    .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                    .padding(egui::Margin::same(0.0).top(16.0).bottom(8.0)),
            );

            let selected_component = components
                .0
                .iter()
                .find(|(component_id, _)| m_component_id.is_some_and(|cid| cid == **component_id));

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

                        ui.selectable_value(m_component_id, None, "NONE");

                        for (component_id, _) in components.0.iter() {
                            let Some(metadata) = metadata_store.get_metadata(component_id) else {
                                continue;
                            };
                            ui.selectable_value(
                                m_component_id,
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
                    graph_states.insert_component(m_graph_id, entity_id, component_id, values);

                    setting_modal_state.0 = None;
                }
            }
        }
    }
}
