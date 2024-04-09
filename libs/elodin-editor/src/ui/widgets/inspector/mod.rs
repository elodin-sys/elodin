use bevy::{
    ecs::{
        event::EventWriter,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut},
    },
    render::view::Visibility,
};
use bevy_egui::egui;
use bevy_infinite_grid::InfiniteGrid;
use big_space::GridCell;
use conduit::{bevy::ColumnPayloadMsg, query::MetadataStore};

use crate::{
    ui::{colors, tiles, CameraQuery, EntityData, GraphsState, SelectedObject},
    MainCamera,
};

pub mod entity;
pub mod graph;
pub mod viewport;

pub struct InspectorIcons {
    pub chart: egui::TextureId,
    pub add: egui::TextureId,
    pub subtract: egui::TextureId,
}

#[allow(clippy::too_many_arguments)]
pub fn inspector(
    ui: &mut egui::Ui,
    selected_object: &SelectedObject,
    entities: &mut Query<EntityData>,
    metadata_store: &Res<MetadataStore>,
    camera_query: &mut Query<CameraQuery, With<MainCamera>>,
    commands: &mut Commands,
    entity_transform_query: &Query<&GridCell<i128>, Without<MainCamera>>,
    graphs_state: &mut ResMut<GraphsState>,
    tile_state: &mut ResMut<tiles::TileState>,
    icons: InspectorIcons,
    column_payload_writer: &mut EventWriter<ColumnPayloadMsg>,
    grid_visibility: &mut Query<&mut Visibility, With<InfiniteGrid>>,
) -> egui::Response {
    egui::ScrollArea::vertical()
        .show(ui, |ui| {
            egui::Frame::none()
                .fill(colors::PRIMARY_SMOKE)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| match selected_object {
                        SelectedObject::None => {
                            ui.add(empty_inspector());
                        }
                        SelectedObject::Entity(pair) => {
                            let Ok((entity_id, _, mut map, metadata)) = entities.get_mut(pair.bevy)
                            else {
                                ui.add(empty_inspector());
                                return;
                            };
                            entity::inspector(
                                ui,
                                metadata,
                                *entity_id,
                                map.as_mut(),
                                metadata_store,
                                graphs_state,
                                tile_state,
                                icons.chart,
                                column_payload_writer,
                            );
                        }
                        SelectedObject::Viewport { camera, .. } => {
                            let Ok(cam) = camera_query.get_mut(*camera) else {
                                ui.add(empty_inspector());
                                return;
                            };
                            viewport::inspector(
                                ui,
                                entities,
                                cam,
                                commands,
                                entity_transform_query,
                                grid_visibility,
                            );
                        }
                        SelectedObject::Graph {
                            label, graph_id, ..
                        } => {
                            graph::inspector(
                                ui,
                                graph_id,
                                label,
                                entities,
                                graphs_state,
                                metadata_store,
                                icons,
                            );
                        }
                    })
                })
        })
        .inner
        .response
}

fn empty_inspector_ui(ui: &mut egui::Ui) -> egui::Response {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            let text = egui::RichText::new("SELECT AN ENTITY OR TABLE TO INSPECT")
                .color(colors::with_opacity(colors::WHITE, 0.1));
            ui.add(egui::Label::new(text));
        },
    )
    .response
}

pub fn empty_inspector() -> impl egui::Widget {
    move |ui: &mut egui::Ui| empty_inspector_ui(ui)
}
