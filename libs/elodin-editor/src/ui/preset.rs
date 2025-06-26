use std::collections::HashMap;

use super::actions::ActionTile;
use super::query_table::QueryTable;
use super::tiles::{Pane, TileState};
use super::widgets::plot::GraphState;
use super::widgets::query_plot::QueryPlot;
use bevy::prelude::*;
use egui_tiles::{Tile, TileId};
use impeller2_wkt::{
    ActionPane, ComponentMonitor, ComponentPath, Panel, SQLPlot, SQLTable, Split, Viewport,
};

#[allow(clippy::too_many_arguments)]
pub fn tile_to_panel(
    query_tables: &Query<&QueryTable>,
    action_tiles: &Query<&ActionTile>,
    graph_states: &Query<&GraphState>,
    query_plots: &Query<&QueryPlot>,
    viewports: &Query<&crate::ui::widgets::inspector::viewport::Viewport>,
    tile_id: TileId,
    ui_state: &TileState,
) -> Option<Panel> {
    let tiles = &ui_state.tree.tiles;
    let tile = tiles.get(tile_id)?;

    match tile {
        Tile::Pane(pane) => match pane {
            Pane::Viewport(viewport) => {
                let cam_entity = viewport.camera?;
                let viewport_data = viewports.get(cam_entity).ok()?;

                Some(Panel::Viewport(Viewport {
                    fov: 45.0,
                    active: false,
                    show_grid: false,
                    hdr: false,
                    name: Some(viewport.label.clone()),
                    pos: Some(viewport_data.pos.eql.clone()),
                    look_at: Some(viewport_data.look_at.eql.clone()),
                }))
            }
            Pane::Graph(graph) => {
                let graph_state = graph_states.get(graph.id).ok()?;
                let mut eql = vec![];
                for ((path, index), (_, _color)) in graph_state.enabled_lines.iter() {
                    // TODO(sphw): add back color support
                    eql.push(format!("{}[{}]", path, index));
                }
                Some(Panel::Graph(impeller2_wkt::Graph {
                    eql: eql.join(","),
                    name: Some(graph_state.label.clone()),
                    graph_type: graph_state.graph_type,
                    auto_y_range: graph_state.auto_y_range,
                    y_range: graph_state.y_range.clone(),
                }))
            }
            Pane::Monitor(monitor) => Some(Panel::ComponentMonitor(ComponentMonitor {
                component_id: monitor.component_id,
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
                        action_tiles,
                        graph_states,
                        query_plots,
                        viewports,
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
                        action_tiles,
                        graph_states,
                        query_plots,
                        viewports,
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

pub trait EqlExt {
    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)>;
}

impl EqlExt for eql::Expr {
    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)> {
        match self {
            eql::Expr::ComponentPart(component_part) => {
                let Some(component) = &component_part.component else {
                    return vec![];
                };
                (0..component.element_names.len())
                    .map(|i| (ComponentPath::from_name(&component_part.name), i))
                    .collect()
            }
            eql::Expr::ArrayAccess(expr, i) => {
                let eql::Expr::ComponentPart(component_part) = &**expr else {
                    return vec![];
                };
                vec![(ComponentPath::from_name(&component_part.name), *i)]
            }
            eql::Expr::Tuple(exprs) => exprs
                .iter()
                .flat_map(|expr| expr.to_graph_components().into_iter())
                .collect(),
            _ => vec![],
        }
    }
}
