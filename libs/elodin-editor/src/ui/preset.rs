use std::collections::HashMap;

use super::CameraQuery;
use super::actions::ActionTile;
use super::query_table::QueryTable;
use super::tiles::{Pane, TileState};
use super::widgets::plot::GraphState;
use super::widgets::query_plot::QueryPlot;
use bevy::prelude::*;
use egui_tiles::{Tile, TileId};
use impeller2::types::EntityId;
use impeller2_wkt::{
    ActionPane, Color, ComponentMonitor, Panel, SQLPlot, SQLTable, Split, Viewport,
};
use nox::{Quaternion, Vector3};

#[allow(clippy::too_many_arguments)]
pub fn tile_to_panel(
    query_tables: &Query<&QueryTable>,
    cameras: &Query<CameraQuery>,
    entity_id: &Query<&EntityId>,
    action_tiles: &Query<&ActionTile>,
    graph_states: &Query<&GraphState>,
    query_plots: &Query<&QueryPlot>,
    tile_id: TileId,
    ui_state: &TileState,
) -> Option<Panel> {
    let tiles = &ui_state.tree.tiles;
    let tile = tiles.get(tile_id)?;

    match tile {
        Tile::Pane(pane) => match pane {
            Pane::Viewport(viewport) => {
                let camera = viewport.camera?;
                let cam = cameras.get(camera).ok()?;

                let pos = Vector3::new(
                    cam.transform.translation.x,
                    -cam.transform.translation.z,
                    cam.transform.translation.y,
                );

                let quat = cam.transform.rotation;

                let rotation = Quaternion::new(quat.w, quat.x, quat.y, quat.z);

                let track_entity = cam
                    .parent
                    .and_then(|e| entity_id.get(e.parent()).ok())
                    .copied();

                let track_rotation = cam.no_propagate_rot.is_none();

                Some(Panel::Viewport(Viewport {
                    track_entity,
                    track_rotation,
                    fov: 45.0,
                    active: false,
                    pos,
                    rotation,
                    show_grid: false,
                    hdr: false,
                    name: Some(viewport.label.clone()),
                }))
            }
            Pane::Graph(graph) => {
                let graph_state = graph_states.get(graph.id).ok()?;
                let mut entities: HashMap<EntityId, impeller2_wkt::GraphEntity> = HashMap::new();
                for ((entity_id, component_id, index), (_, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let entity_id = *entity_id;
                    let entity =
                        entities
                            .entry(entity_id)
                            .or_insert_with(|| impeller2_wkt::GraphEntity {
                                entity_id,
                                components: vec![],
                            });
                    let [r, g, b, _] = color.to_normalized_gamma_f32();
                    entity.components.push(impeller2_wkt::GraphComponent {
                        component_id: *component_id,
                        indexes: vec![*index],
                        color: vec![Color::rgb(r, g, b)],
                    });
                }
                let entities = entities.into_values().collect();
                Some(Panel::Graph(impeller2_wkt::Graph {
                    entities,
                    name: Some(graph_state.label.clone()),
                    graph_type: graph_state.graph_type,
                    auto_y_range: graph_state.auto_y_range,
                    y_range: graph_state.y_range.clone(),
                }))
            }
            Pane::Monitor(monitor) => Some(Panel::ComponentMonitor(ComponentMonitor {
                component_id: monitor.component_id,
                entity_id: monitor.entity_id,
            })),
            Pane::QueryTable(query_table) => {
                let query_table = query_tables.get(query_table.entity).ok()?;

                Some(Panel::SQLTable(SQLTable {
                    query: query_table.current_query.clone(),
                }))
            }
            Pane::QueryPlot(plot) => {
                let query_plot = query_plots.get(plot.entity).ok()?;
                Some(Panel::SQLPlot(SQLPlot {
                    query: query_plot.current_query.clone(),
                    auto_refresh: query_plot.auto_refresh,
                    refresh_interval: query_plot.refresh_interval,
                }))
            }
            Pane::ActionTile(action) => {
                let action_tile = action_tiles.get(action.entity).ok()?;

                Some(Panel::ActionPane(ActionPane {
                    label: action_tile.button_name.clone(),
                    lua: action_tile.lua.clone(),
                }))
            }
            Pane::VideoStream(_) => None,
            Pane::Hierarchy => Some(Panel::Hierarchy),
            Pane::Inspector => Some(Panel::Inspector),
        },
        Tile::Container(container) => match container {
            egui_tiles::Container::Tabs(t) => {
                let mut tabs = vec![];
                for tile_id in &t.children {
                    if let Some(tab) = tile_to_panel(
                        query_tables,
                        cameras,
                        entity_id,
                        action_tiles,
                        graph_states,
                        query_plots,
                        *tile_id,
                        ui_state,
                    ) {
                        tabs.push(tab)
                    }
                }
                Some(Panel::Tabs(tabs))
            }
            egui_tiles::Container::Linear(linear) => {
                let mut panels = Vec::new();
                let mut shares = HashMap::new();
                for child_id in &linear.children {
                    if let Some(panel) = tile_to_panel(
                        query_tables,
                        cameras,
                        entity_id,
                        action_tiles,
                        graph_states,
                        query_plots,
                        *child_id,
                        ui_state,
                    ) {
                        if let Some((_, share)) =
                            linear.shares.iter().find(|(id, _)| *id == child_id)
                        {
                            shares.insert(panels.len(), *share);
                        }
                        panels.push(panel);
                    }
                }

                if panels.is_empty() {
                    return None;
                }

                let split = Split {
                    panels,
                    shares,
                    active: false,
                };

                match linear.dir {
                    egui_tiles::LinearDir::Horizontal => Some(Panel::HSplit(split)),
                    egui_tiles::LinearDir::Vertical => Some(Panel::VSplit(split)),
                }
            }
            _ => None,
        },
    }
}
