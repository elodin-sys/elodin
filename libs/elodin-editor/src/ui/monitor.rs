use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui::{self, Frame, RichText, Stroke};
use impeller2::types::ComponentId;
use impeller2_bevy::ComponentValue;
use impeller2_bevy::ComponentValueExt;
use impeller2_bevy::{ComponentMetadataRegistry, EntityMap};

use super::{PaneName, colors::get_scheme, widgets::WidgetSystem};

#[derive(Clone)]
pub struct MonitorPane {
    pub entity: Entity,
    pub name: PaneName,
}

impl MonitorPane {
    pub fn new(entity: Entity, name: PaneName) -> Self {
        Self { entity, name }
    }
}

#[derive(Component, Clone)]
pub struct MonitorData {
    pub component_name: String,
}

#[derive(SystemParam)]
pub struct MonitorWidget<'w, 's> {
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    monitors: Query<'w, 's, &'static MonitorData>,
    component_value_query: Query<'w, 's, &'static mut ComponentValue>,
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
        let MonitorWidget {
            metadata_store,
            monitors,
            mut component_value_query,
            entity_map,
        } = state.get_mut(world);
        let Ok(monitor) = monitors.get(pane.entity) else {
            return;
        };
        let component_id = ComponentId::new(&monitor.component_name);
        let Some(entity) = entity_map.get(&component_id) else {
            return;
        };
        let Some(metadata) = metadata_store.get_metadata(&component_id) else {
            return;
        };

        let Ok(mut value) = component_value_query.get_mut(*entity) else {
            return;
        };

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
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

                        ui.allocate_ui_with_layout([130., 60.].into(), layout, |ui| {
                            Frame::NONE
                                .stroke(Stroke::new(1.0, get_scheme().border_primary))
                                .outer_margin(egui::Margin::symmetric(8, 8))
                                .inner_margin(egui::Margin::symmetric(8, 0))
                                .show(ui, |ui| {
                                    ui.set_width(120. - 8.);
                                    ui.set_height(50.);
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
                                                    let mut str = format!("{v:.8}");
                                                    str.truncate(10);
                                                    str
                                                }
                                                impeller2_bevy::ElementValueMut::F32(v) => {
                                                    let mut str = format!("{v:.8}");
                                                    str.truncate(10);
                                                    str
                                                }
                                                impeller2_bevy::ElementValueMut::Bool(v) => {
                                                    v.to_string()
                                                }
                                            };
                                            ui.add_space(8.0);
                                            let scheme = get_scheme();
                                            let value = RichText::new(value)
                                                .monospace()
                                                .size(18.)
                                                .color(scheme.text_primary);
                                            ui.label(value);
                                            let label = RichText::new(label)
                                                .size(13.0)
                                                .monospace()
                                                .color(scheme.text_secondary);
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
