use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui::{self, Frame, RichText, Stroke};
use impeller2::types::{ComponentId, EntityId};
use impeller2_bevy::ComponentValueExt;
use impeller2_bevy::{ComponentMetadataRegistry, ComponentValueMap, EntityMap};
use impeller2_wkt::EntityMetadata;

use super::{colors::get_scheme, widgets::WidgetSystem};

#[derive(Clone)]
pub struct MonitorPane {
    pub label: String,
    pub entity_id: EntityId,
    pub component_id: ComponentId,
}

impl MonitorPane {
    pub fn new(label: String, entity_id: EntityId, component_id: ComponentId) -> Self {
        Self {
            label,
            entity_id,
            component_id,
        }
    }
}

#[derive(SystemParam)]
pub struct MonitorWidget<'w, 's> {
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    component_value_query: Query<'w, 's, &'static mut ComponentValueMap>,
    entity_metadata: Query<'w, 's, &'static EntityMetadata>,
    entity_map: Res<'w, EntityMap>,
}

impl WidgetSystem for MonitorWidget<'_, '_> {
    type Args = MonitorPane;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        pane: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Some(entity) = state.entity_map.get(&pane.entity_id) else {
            return;
        };
        let Ok(mut component_value_map) = state.component_value_query.get_mut(*entity) else {
            return;
        };
        let Some(metadata) = state.metadata_store.get_metadata(&pane.component_id) else {
            return;
        };

        let Some(value) = component_value_map.get_mut(&pane.component_id) else {
            return;
        };
        let Ok(entity_metadata) = state.entity_metadata.get(*entity) else {
            return;
        };
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                let label = RichText::new(format!("{}.{}", entity_metadata.name, metadata.name))
                    .monospace()
                    .size(25.);
                ui.label(label);
                ui.add_space(20.0);
                let width = ui.max_rect().width();
                ui.horizontal_wrapped(|ui| {
                    ui.set_width(width);
                    let element_names = metadata
                        .element_names()
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(Option::Some)
                        .chain(std::iter::repeat(None));
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.spacing_mut().item_spacing.y = 0.0;

                    for ((dim_i, value), element_name) in
                        value.indexed_iter_mut().zip(element_names)
                    {
                        let layout = egui::Layout::centered_and_justified(ui.layout().main_dir());

                        ui.allocate_ui_with_layout([220., 170.].into(), layout, |ui| {
                            Frame::NONE
                                .stroke(Stroke::new(1.0, get_scheme().border_primary))
                                .outer_margin(egui::Margin::symmetric(8, 8))
                                .inner_margin(egui::Margin::symmetric(8, 0))
                                .show(ui, |ui| {
                                    ui.set_width(210. - 8.);
                                    ui.set_height(150.);
                                    ui.with_layout(
                                        egui::Layout::bottom_up(egui::Align::LEFT),
                                        |ui| {
                                            let label = element_name
                                                .map(|name| name.to_string())
                                                .unwrap_or_else(|| format!("{dim_i:?}"));

                                            let value = match value {
                                                impeller2_bevy::ElementValueMut::U8(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::U16(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::U32(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::U64(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::I8(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::I16(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::I32(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::I64(v) => {
                                                    v.to_string()
                                                }
                                                impeller2_bevy::ElementValueMut::F64(v) => {
                                                    format!("{v:.8}")
                                                }
                                                impeller2_bevy::ElementValueMut::F32(v) => {
                                                    format!("{v:.8}")
                                                }
                                                impeller2_bevy::ElementValueMut::Bool(v) => {
                                                    v.to_string()
                                                }
                                            };
                                            ui.add_space(8.0);
                                            let value = RichText::new(value).monospace().size(18.);
                                            ui.label(value);
                                            let label = RichText::new(label)
                                                .size(13.0)
                                                .monospace()
                                                .color(get_scheme().text_secondary);
                                            ui.add_space(8.0);
                                            ui.label(label);
                                        },
                                    );
                                })
                                .response
                        });
                    }
                });
            });
    }
}
