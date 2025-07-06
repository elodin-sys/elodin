use std::collections::HashMap;

use crate::{
    object_3d::Object3DState,
    ui::{
        actions, inspector, plot, query_plot, query_table,
        tiles::{self, Pane},
    },
};
use bevy::{ecs::system::SystemParam, prelude::*};
use egui_tiles::{Tile, TileId};
use impeller2_wkt::{
    ActionPane, ComponentMonitor, ComponentPath, Dashboard, Line3d, Panel, Schematic,
    SchematicElem, Split, Viewport,
};

pub mod tree;
pub use tree::*;
mod load;
pub use load::*;

#[derive(Resource, Debug, Clone, Deref, DerefMut)]
pub struct CurrentSchematic(pub Schematic<Entity>);

#[derive(SystemParam)]
pub struct SchematicParam<'w, 's> {
    pub query_tables: Query<'w, 's, &'static query_table::QueryTableData>,
    pub action_tiles: Query<'w, 's, &'static actions::ActionTile>,
    pub graph_states: Query<'w, 's, &'static plot::GraphState>,
    pub query_plots: Query<'w, 's, &'static query_plot::QueryPlotData>,
    pub viewports: Query<'w, 's, &'static inspector::viewport::Viewport>,
    pub objects_3d: Query<'w, 's, (Entity, &'static Object3DState)>,
    pub lines_3d: Query<'w, 's, (Entity, &'static Line3d)>,
    pub ui_state: Res<'w, tiles::TileState>,
    pub dashboards: Query<'w, 's, &'static Dashboard<Entity>>,
}

impl SchematicParam<'_, '_> {
    pub fn get_panel(&self, tile_id: TileId) -> Option<Panel<Entity>> {
        let tiles = &self.ui_state.tree.tiles;
        let tile = tiles.get(tile_id)?;

        match tile {
            Tile::Pane(pane) => match pane {
                Pane::Viewport(viewport) => {
                    let cam_entity = viewport.camera?;
                    let viewport_data = self.viewports.get(cam_entity).ok()?;

                    Some(Panel::Viewport(Viewport {
                        fov: 45.0,
                        active: false,
                        show_grid: false,
                        hdr: false,
                        name: Some(viewport.label.clone()),
                        pos: Some(viewport_data.pos.eql.clone()),
                        look_at: Some(viewport_data.look_at.eql.clone()),
                        aux: cam_entity,
                    }))
                }
                Pane::Graph(graph) => {
                    let graph_state = self.graph_states.get(graph.id).ok()?;
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
                        aux: graph.id,
                    }))
                }
                Pane::Monitor(monitor) => Some(Panel::ComponentMonitor(ComponentMonitor {
                    component_id: monitor.component_id,
                })),
                Pane::QueryTable(query_table) => {
                    let query_table = self.query_tables.get(query_table.entity).ok()?;

                    Some(Panel::QueryTable(query_table.data.clone()))
                }
                Pane::QueryPlot(plot) => {
                    let query_plot = self.query_plots.get(plot.entity).ok()?;
                    Some(Panel::QueryPlot(query_plot.data.map_aux(|_| plot.entity)))
                }
                Pane::ActionTile(action) => {
                    let action_tile = self.action_tiles.get(action.entity).ok()?;

                    Some(Panel::ActionPane(ActionPane {
                        label: action_tile.button_name.clone(),
                        lua: action_tile.lua.clone(),
                    }))
                }
                Pane::VideoStream(_) => None,
                Pane::Hierarchy => Some(Panel::Hierarchy),
                Pane::Inspector => Some(Panel::Inspector),
                Pane::SchematicTree(_) => Some(Panel::SchematicTree),
                Pane::Dashboard(dash) => {
                    let dashboard = self.dashboards.get(dash.entity).ok()?;
                    Some(Panel::Dashboard(Box::new(dashboard.clone())))
                }
            },
            Tile::Container(container) => match container {
                egui_tiles::Container::Tabs(t) => {
                    let mut tabs = vec![];
                    for tile_id in &t.children {
                        if let Some(tab) = self.get_panel(*tile_id) {
                            tabs.push(tab)
                        }
                    }
                    Some(Panel::Tabs(tabs))
                }
                egui_tiles::Container::Linear(linear) => {
                    let mut panels = Vec::new();
                    let mut shares = HashMap::new();
                    for child_id in &linear.children {
                        if let Some(panel) = self.get_panel(*child_id) {
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
}

pub fn tiles_to_schematic(param: SchematicParam, mut schematic: ResMut<CurrentSchematic>) {
    schematic.elems.clear();
    if let Some(tile_id) = param.ui_state.tree.root() {
        schematic
            .elems
            .extend(param.get_panel(tile_id).map(SchematicElem::Panel))
    }
    schematic.elems.extend(
        param
            .objects_3d
            .iter()
            .map(|(entity, o)| o.data.map_aux(|_| entity))
            .map(SchematicElem::Object3d),
    );
    schematic.elems.extend(
        param
            .lines_3d
            .iter()
            .map(|(entity, line)| SchematicElem::Line3d(line.map_aux(|_| entity))),
    );
}

pub struct SchematicPlugin;

impl Plugin for SchematicPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CurrentSchematic(Default::default()))
            .add_systems(PostUpdate, tiles_to_schematic)
            .add_systems(PostUpdate, sync_schematic.before(tiles_to_schematic))
            .init_resource::<SchematicLiveReloadRx>()
            .add_systems(PreUpdate, load::schematic_live_reload);
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
