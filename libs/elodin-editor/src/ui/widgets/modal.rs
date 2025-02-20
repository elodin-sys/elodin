use std::collections::BTreeMap;

use bevy::{
    ecs::{
        system::{Local, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    window::Window,
};
use bevy_egui::{egui, EguiContexts};
use impeller2::component::Component;
use impeller2_bevy::ComponentMetadataRegistry;
use impeller2_wkt::WorldPos;

use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        colors::{self, with_opacity},
        images, theme, tiles,
        utils::MarginSides,
        EntityData, InspectorAnchor, SettingModal, SettingModalState,
    },
};

use super::{
    button::EButton,
    label::{self, EImageLabel, ELabel},
    plot::{default_component_values, GraphBundle, GraphState},
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
                    fill: colors::PRIMARY_SMOKE,
                    stroke: egui::Stroke::NONE,
                    inner_margin: egui::Margin::same(16),
                    outer_margin: egui::Margin::symmetric(4, 0),
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
            mut graph_states,
        } = state_mut;

        let Some(setting_modal) = setting_modal_state.0.as_mut() else {
            return;
        };
        let SettingModal::Graph(m_graph_id, m_entity_id, m_component_id) = setting_modal else {
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
            colors::PRIMARY_CREAME,
            title_margin,
        );
        if close_clicked {
            setting_modal_state.0 = None;
            return;
        }

        ui.add(egui::Separator::default().spacing(0.0));

        ui.add(
            ELabel::new("ENTITY")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
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
                    .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
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
                    let values = default_component_values(entity_id, component_id, component);
                    graph_state.insert_component(entity_id, component_id, values);

                    setting_modal_state.0 = None;
                }
            }
        }
    }
}

#[derive(SystemParam)]
pub struct ModalNewTile<'w> {
    new_tile_state: Res<'w, tiles::NewTileState>,
}

impl WidgetSystem for ModalNewTile<'_> {
    type Args = (egui::Pos2, tiles::TileIcons);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let (center_pos, icons) = args;

        let new_tile_state = state_mut.new_tile_state.clone();

        let rect = egui::Rect::from_center_size(center_pos, egui::vec2(320.0, 320.0));

        if matches!(new_tile_state, tiles::NewTileState::None) {
            return;
        }

        egui::Window::new("NEW_TILE_MODAL")
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame {
                fill: colors::PRIMARY_ONYX,
                stroke: egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
                inner_margin: egui::Margin::same(16),
                corner_radius: egui::CornerRadius::same(4),
                shadow: egui::epaint::Shadow {
                    color: colors::PRIMARY_SMOKE,
                    blur: 16,
                    offset: [3, 3],
                    spread: 3,
                },
                ..Default::default()
            })
            .fixed_rect(rect)
            .show(ui.ctx(), |ui| {
                ui.vertical(|ui| match new_tile_state {
                    tiles::NewTileState::Viewport(_) => {
                        ui.add_widget_with::<ModalNewViewportTile>(
                            world,
                            "modal_new_viewport_tile",
                            icons,
                        );
                    }
                    tiles::NewTileState::Graph { .. } => {
                        ui.add_widget_with::<ModalNewGraphTile>(
                            world,
                            "modal_new_graph_tile",
                            icons,
                        );
                    }
                    tiles::NewTileState::ComponentMonitor { .. } => {
                        ui.add_widget_with::<ModalNewComponentMonitorTile>(
                            world,
                            "modal_new_component_monitor_tile",
                            icons,
                        );
                    }
                    tiles::NewTileState::None => {
                        panic!("NewTileState is None after validating otherwise")
                    }
                });
            });
    }
}

#[derive(SystemParam)]
pub struct ModalNewGraphTile<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    new_tile_state: ResMut<'w, tiles::NewTileState>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    tile_state: ResMut<'w, tiles::TileState>,
}

impl WidgetSystem for ModalNewGraphTile<'_, '_> {
    type Args = tiles::TileIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let icons = args;

        let mut new_tile_state = state_mut.new_tile_state;
        let entities_meta = state_mut.entities_meta;
        let metadata_store = state_mut.metadata_store;
        let mut render_layer_alloc = state_mut.render_layer_alloc;
        let mut tile_state = state_mut.tile_state;

        let tiles::NewTileState::Graph {
            entity_id: m_entity_id,
            component_id: m_component_id,
            parent_id,
        } = new_tile_state.as_mut()
        else {
            *new_tile_state = tiles::NewTileState::None;
            return;
        };

        let mut close_modal = false;
        let can_create = m_entity_id.is_some() && m_component_id.is_some();

        // HEADER

        ui.add(
            EImageLabel::new(icons.tile_graph)
                .image_tint(colors::MINT_DEFAULT)
                .bg_color(with_opacity(colors::MINT_DEFAULT, 0.01))
                .margin(egui::Margin::same(1)),
        );

        ui.add(
            ELabel::new("Create new point graph")
                .text_color(colors::PRIMARY_CREAME)
                .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
        );

        // ENTITY

        ui.add(
            ELabel::new("ENTITY")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
        );

        let selected_entity = entities_meta
            .iter()
            .find(|(entity_id, _, _, _)| m_entity_id.is_some_and(|eid| eid == **entity_id));

        let selected_entity_label =
            selected_entity.map_or("None", |(_, _, _, metadata)| &metadata.name);

        let width = ui.available_width();

        ui.scope(|ui| {
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("ENTITY")
                .width(width)
                .selected_text(selected_entity_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_entity_id, None, "None");
                    let mut entities = entities_meta
                        .iter()
                        .filter(|(_, _, values, _)| values.0.contains_key(&WorldPos::COMPONENT_ID))
                        .collect::<Vec<_>>();
                    entities.sort_by(|a, b| a.0.cmp(b.0));

                    for (entity_id, _, values, metadata) in entities {
                        if values.0.contains_key(&WorldPos::COMPONENT_ID) {
                            ui.selectable_value(
                                m_entity_id,
                                Some(*entity_id),
                                metadata.name.to_string(),
                            );
                        }
                    }
                });
        });

        // COMPONENT

        ui.add(
            ELabel::new("COMPONENT")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
        );

        let components = if let Some((_, _, components, _)) = selected_entity {
            components.0.clone()
        } else {
            BTreeMap::new()
        };

        let selected_component = components
            .iter()
            .find(|(component_id, _)| m_component_id.is_some_and(|cid| cid == **component_id));

        let selected_component_label = selected_component
            .and_then(|(component_id, _)| metadata_store.get_metadata(component_id))
            .map(|m| m.name.as_ref())
            .unwrap_or_else(|| "None");

        ui.scope(|ui| {
            if components.is_empty() {
                ui.disable();
            }

            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("COMPONENT")
                .width(width)
                .selected_text(selected_component_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_component_id, None, "None");

                    for (component_id, _) in components.iter() {
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

        // BUTTONS

        ui.add_space(16.0);

        ui.horizontal(|ui| {
            let item_spacing = 10.0;
            ui.spacing_mut().item_spacing.x = item_spacing;
            let button_width = (ui.available_width() - item_spacing) / 2.0;

            let cancel_btn = ui.add(
                EButton::new("CANCEL")
                    .width(button_width)
                    .color(colors::PRIMARY_CREAME)
                    .bg_color(colors::PRIMARY_ONYX)
                    .stroke(egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9)),
            );
            close_modal = cancel_btn.clicked();

            let create_btn = ui.add(
                EButton::new("CREATE")
                    .width(button_width)
                    .color(colors::MINT_DEFAULT)
                    .disabled(!can_create)
                    .bg_color(with_opacity(colors::MINT_DEFAULT, 0.05))
                    .stroke(egui::Stroke::new(
                        1.0,
                        with_opacity(colors::MINT_DEFAULT, 0.4),
                    )),
            );

            if create_btn.clicked() {
                let Some(component_id) = m_component_id.to_owned() else {
                    return;
                };
                let Some((entity_id, _, component_value_map, _)) = selected_entity else {
                    return;
                };

                let component_value = component_value_map.0.get(&component_id).unwrap();
                let values = default_component_values(entity_id, &component_id, component_value);
                let entities = BTreeMap::from_iter(std::iter::once((
                    entity_id.to_owned(),
                    BTreeMap::from_iter(std::iter::once((component_id, values.clone()))),
                )));
                let bundle = GraphBundle::new(&mut render_layer_alloc, entities);
                tile_state.create_graph_tile(*parent_id, bundle);

                close_modal = true;
            }
        });

        if close_modal {
            *new_tile_state = tiles::NewTileState::None;
        }
    }
}

#[derive(SystemParam)]
pub struct ModalNewViewportTile<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    new_tile_state: ResMut<'w, tiles::NewTileState>,
    tile_state: ResMut<'w, tiles::TileState>,
}

impl WidgetSystem for ModalNewViewportTile<'_, '_> {
    type Args = tiles::TileIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let icons = args;

        let mut new_tile_state = state_mut.new_tile_state;
        let entities_meta = state_mut.entities_meta;
        let mut tile_state = state_mut.tile_state;

        let tiles::NewTileState::Viewport(m_entity_id) = new_tile_state.as_mut() else {
            *new_tile_state = tiles::NewTileState::None;
            return;
        };

        let mut close_modal = false;

        // HEADER

        ui.add(
            EImageLabel::new(icons.tile_3d_viewer)
                .image_tint(colors::PUMPKIN_DEFAULT)
                .bg_color(with_opacity(colors::PUMPKIN_DEFAULT, 0.01))
                .margin(egui::Margin::same(1)),
        );

        ui.add(
            ELabel::new("Create new 3D viewer")
                .text_color(colors::PRIMARY_CREAME)
                .padding(egui::Margin::same(0).top(16.).bottom(8.)),
        );

        // ENTITY

        ui.add(
            ELabel::new("ENTITY")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0).top(16.).bottom(8.)),
        );

        let selected_entity = entities_meta
            .iter()
            .find(|(entity_id, _, _, _)| m_entity_id.is_some_and(|eid| eid == **entity_id));

        let selected_entity_label =
            selected_entity.map_or("None", |(_, _, _, metadata)| &metadata.name);

        let width = ui.available_width();

        ui.scope(|ui| {
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("ENTITY")
                .width(width)
                .selected_text(selected_entity_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_entity_id, None, "None");
                    let mut entities = entities_meta
                        .iter()
                        .filter(|(_, _, values, _)| values.0.contains_key(&WorldPos::COMPONENT_ID))
                        .collect::<Vec<_>>();
                    entities.sort_by(|a, b| a.0.cmp(b.0));

                    for (entity_id, _, values, metadata) in entities {
                        if values.0.contains_key(&WorldPos::COMPONENT_ID) {
                            ui.selectable_value(
                                m_entity_id,
                                Some(*entity_id),
                                metadata.name.to_string(),
                            );
                        }
                    }
                });
        });

        // BUTTONS

        ui.add_space(16.0);

        ui.horizontal(|ui| {
            let item_spacing = 10.0;
            ui.spacing_mut().item_spacing.x = item_spacing;
            let button_width = (ui.available_width() - item_spacing) / 2.0;

            let cancel_btn = ui.add(
                EButton::new("CANCEL")
                    .width(button_width)
                    .color(colors::PRIMARY_CREAME)
                    .bg_color(colors::PRIMARY_ONYX)
                    .stroke(egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9)),
            );
            close_modal = cancel_btn.clicked();

            let create_btn = ui.add(
                EButton::new("CREATE")
                    .width(button_width)
                    .color(colors::MINT_DEFAULT)
                    .bg_color(with_opacity(colors::MINT_DEFAULT, 0.05))
                    .stroke(egui::Stroke::new(
                        1.0,
                        with_opacity(colors::MINT_DEFAULT, 0.4),
                    )),
            );

            if create_btn.clicked() {
                let focused_entity = m_entity_id.to_owned();

                tile_state.create_viewport_tile(focused_entity);

                close_modal = true;
            }
        });

        if close_modal {
            *new_tile_state = tiles::NewTileState::None;
        }
    }
}

#[derive(SystemParam)]
pub struct ModalNewComponentMonitorTile<'w, 's> {
    entities_meta: Query<'w, 's, EntityData<'static>>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    new_tile_state: ResMut<'w, tiles::NewTileState>,
    tile_state: ResMut<'w, tiles::TileState>,
}

impl WidgetSystem for ModalNewComponentMonitorTile<'_, '_> {
    type Args = tiles::TileIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);
        let icons = args;

        let mut new_tile_state = state_mut.new_tile_state;
        let entities_meta = state_mut.entities_meta;
        let metadata_store = state_mut.metadata_store;
        let mut tile_state = state_mut.tile_state;

        let tiles::NewTileState::ComponentMonitor {
            entity_id: m_entity_id,
            component_id: m_component_id,
            parent_id: _m_parent_id,
        } = new_tile_state.as_mut()
        else {
            *new_tile_state = tiles::NewTileState::None;
            return;
        };

        let mut close_modal = false;
        let can_create = m_entity_id.is_some() && m_component_id.is_some();

        // Header
        ui.add(
            EImageLabel::new(icons.tile_graph) // Todo: create and add tile icon for data monitor
                .image_tint(colors::PEACH_DEFAULT)
                .bg_color(with_opacity(colors::PEACH_DEFAULT, 0.01))
                .margin(egui::Margin::same(1)),
        );

        ui.add(
            ELabel::new("Create new component monitor")
                .text_color(colors::PRIMARY_CREAME)
                .padding(egui::Margin::same(0).top(16.).bottom(8.)),
        );

        // Entity selection
        ui.add(
            ELabel::new("ENTITY")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0).top(16.).bottom(8.)),
        );

        let selected_entity = entities_meta
            .iter()
            .find(|(entity_id, _, _, _)| m_entity_id.is_some_and(|eid| eid == **entity_id));

        let selected_entity_label =
            selected_entity.map_or("None", |(_, _, _, metadata)| &metadata.name);

        let width = ui.available_width();

        ui.scope(|ui| {
            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("ENTITY")
                .width(width)
                .selected_text(selected_entity_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_entity_id, None, "None");
                    let mut entities = entities_meta
                        .iter()
                        .filter(|(_, _, values, _)| values.0.contains_key(&WorldPos::COMPONENT_ID))
                        .collect::<Vec<_>>();
                    entities.sort_by(|a, b| a.0.cmp(b.0));

                    for (entity_id, _, values, metadata) in entities {
                        if values.0.contains_key(&WorldPos::COMPONENT_ID) {
                            ui.selectable_value(
                                m_entity_id,
                                Some(*entity_id),
                                metadata.name.to_string(),
                            );
                        }
                    }
                });
        });

        // Component Selection
        ui.add(
            ELabel::new("COMPONENT")
                .text_color(colors::with_opacity(colors::PRIMARY_CREAME, 0.6))
                .padding(egui::Margin::same(0).top(16.0).bottom(8.0)),
        );

        let components = if let Some((_, _, components, _)) = selected_entity {
            components.0.clone()
        } else {
            BTreeMap::new()
        };

        let selected_component = components
            .iter()
            .find(|(component_id, _)| m_component_id.is_some_and(|cid| cid == **component_id));

        let selected_component_label = selected_component
            .and_then(|(component_id, _)| metadata_store.get_metadata(component_id))
            .map(|m| m.name.as_ref())
            .unwrap_or_else(|| "None");

        ui.scope(|ui| {
            if components.is_empty() {
                ui.disable();
            }

            theme::configure_combo_box(ui.style_mut());
            egui::ComboBox::from_id_salt("COMPONENT")
                .width(width)
                .selected_text(selected_component_label)
                .show_ui(ui, |ui| {
                    theme::configure_combo_item(ui.style_mut());

                    ui.selectable_value(m_component_id, None, "None");

                    for (component_id, _) in components.iter() {
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

        // Buttons

        ui.add_space(16.0);

        ui.horizontal(|ui| {
            let item_spacing = 10.0;
            ui.spacing_mut().item_spacing.x = item_spacing;
            let button_width = (ui.available_width() - item_spacing) / 2.0;

            let cancel_btn = ui.add(
                EButton::new("CANCEL")
                    .width(button_width)
                    .color(colors::PRIMARY_CREAME)
                    .bg_color(colors::PRIMARY_ONYX)
                    .stroke(egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9)),
            );
            close_modal = cancel_btn.clicked();

            let create_btn = ui.add(
                EButton::new("CREATE")
                    .width(button_width)
                    .color(colors::MINT_DEFAULT)
                    .disabled(!can_create)
                    .bg_color(with_opacity(colors::MINT_DEFAULT, 0.05))
                    .stroke(egui::Stroke::new(
                        1.0,
                        with_opacity(colors::MINT_DEFAULT, 0.4),
                    )),
            );

            if create_btn.clicked() {
                let Some(component_id) = m_component_id.to_owned() else {
                    return;
                };
                let Some((entity_id, _, _, _)) = selected_entity else {
                    return;
                };
                tile_state.create_component_monitor_tile(*entity_id, component_id);
                close_modal = true;
            }
        });
        if close_modal {
            *new_tile_state = tiles::NewTileState::None;
        }
    }
}
