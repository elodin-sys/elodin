use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui::Color32;
use egui_tiles::{Container, Tile, TileId};
use impeller2_bevy::{ComponentPath, ComponentSchemaRegistry};
use impeller2_kdl::FromKdl;
use impeller2_kdl::KdlSchematicError;
use impeller2_wkt::{DbConfig, Graph, Line3d, Object3D, Panel, Schematic, Viewport};
use std::time::Duration;
use std::{collections::BTreeMap, path::Path};

use crate::{
    EqlContext,
    object_3d::Object3DState,
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        HdrEnabled, SelectedObject,
        colors::{self},
        dashboard::{NodeUpdaterParams, spawn_dashboard},
        monitor::MonitorPane,
        plot::GraphBundle,
        query_plot::QueryPlotData,
        schematic::EqlExt,
        tiles::{DashboardPane, GraphPane, Pane, TileState, TreePane, ViewportPane},
    },
};

#[derive(Component)]
pub struct SyncedViewport;

#[derive(SystemParam)]
pub struct LoadSchematicParams<'w, 's> {
    pub commands: Commands<'w, 's>,
    pub tile_state: ResMut<'w, TileState>,
    pub asset_server: Res<'w, AssetServer>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
    pub render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    pub hdr_enabled: ResMut<'w, HdrEnabled>,
    pub schema_reg: Res<'w, ComponentSchemaRegistry>,
    pub eql: Res<'w, EqlContext>,
    pub selected_object: ResMut<'w, SelectedObject>,
    pub node_updater_params: NodeUpdaterParams<'w, 's>,
    objects_3d: Query<'w, 's, Entity, With<Object3DState>>,
}

pub fn sync_schematic(
    config: Res<DbConfig>,
    mut params: LoadSchematicParams,
    live_reload_rx: ResMut<SchematicLiveReloadRx>,
) {
    if !config.is_changed() {
        return;
    }
    if let Some(path) = config.schematic_path() {
        let path = Path::new(path);
        if path.exists() {
            if load_schematic_file(&path, params, live_reload_rx).is_ok() {
                return;
            }
        }
        return;
    }
    if let Some(content) = config.schematic_content() {
        let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(&content) else {
            return;
        };
        params.load_schematic(&schematic);
    }
}

pub fn load_schematic_file(
    path: &Path,
    mut params: LoadSchematicParams,
    mut live_reload_rx: ResMut<SchematicLiveReloadRx>,
) -> Result<(), KdlSchematicError> {
    let (tx, rx) = flume::bounded(1);
    live_reload_rx.0 = Some(rx);
    let watch_path = path.to_path_buf();
    std::thread::spawn(move || {
        let cb_path = watch_path.clone();
        let mut debouncer = notify_debouncer_mini::new_debouncer(
            Duration::from_millis(100),
            move |res: notify_debouncer_mini::DebounceEventResult| {
                if res.is_err() {
                    return;
                }

                info!(path = ?cb_path, "refreshing schematic");
                if let Ok(kdl) = std::fs::read_to_string(&cb_path) {
                    let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(&kdl) else {
                        return;
                    };
                    let _ = tx.send(schematic);
                }
            },
        )
        .unwrap();
        debouncer
            .watcher()
            .watch(
                &watch_path,
                notify_debouncer_mini::notify::RecursiveMode::NonRecursive,
            )
            .unwrap();
        loop {
            std::thread::park();
        }
    });
    if let Ok(kdl) = std::fs::read_to_string(path) {
        let schematic = impeller2_wkt::Schematic::from_kdl(&kdl)?;
        params.load_schematic(&schematic);
    }
    Ok(())
}

impl LoadSchematicParams<'_, '_> {
    pub fn load_schematic(&mut self, schematic: &Schematic) {
        self.render_layer_alloc.free_all();
        self.tile_state
            .clear(&mut self.commands, &mut self.selected_object);
        for entity in self.objects_3d.iter() {
            self.commands.entity(entity).despawn();
        }
        for elem in &schematic.elems {
            match elem {
                impeller2_wkt::SchematicElem::Panel(p) => {
                    self.spawn_panel(p, None);
                }
                impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                    self.spawn_object_3d(object_3d.clone());
                }
                impeller2_wkt::SchematicElem::Line3d(line_3d) => {
                    println!("spawn line 3d {:?}", line_3d);
                    self.spawn_line_3d(line_3d.clone());
                }
            }
        }
    }

    pub fn spawn_object_3d(&mut self, object_3d: Object3D) {
        let Ok(expr) = self.eql.0.parse_str(&object_3d.eql) else {
            return;
        };
        crate::object_3d::create_object_3d_entity(
            &mut self.commands,
            object_3d.clone(),
            expr,
            &mut self.materials,
            &mut self.meshes,
            &self.asset_server,
        );
    }

    pub fn spawn_line_3d(&mut self, line_3d: Line3d) {
        self.commands.spawn(line_3d);
    }

    pub fn spawn_panel(&mut self, panel: &Panel, parent_id: Option<TileId>) -> Option<TileId> {
        match panel {
            Panel::Viewport(viewport) => {
                let label = viewport_label(viewport);
                let pane = ViewportPane::spawn(
                    &mut self.commands,
                    &self.asset_server,
                    &mut self.meshes,
                    &mut self.materials,
                    &mut self.render_layer_alloc,
                    &self.eql.0,
                    viewport,
                    label,
                );
                self.hdr_enabled.0 |= viewport.hdr;
                self.tile_state.insert_tile(
                    Tile::Pane(Pane::Viewport(pane)),
                    parent_id,
                    viewport.active,
                )
            }
            Panel::HSplit(split) | Panel::VSplit(split) => {
                let linear = egui_tiles::Linear::new(
                    match panel {
                        Panel::HSplit(_) => egui_tiles::LinearDir::Horizontal,
                        Panel::VSplit(_) => egui_tiles::LinearDir::Vertical,
                        _ => unreachable!(),
                    },
                    vec![],
                );
                let tile_id = self.tile_state.insert_tile(
                    Tile::Container(Container::Linear(linear)),
                    parent_id,
                    false,
                );
                for (i, panel) in split.panels.iter().enumerate() {
                    let child_id = self.spawn_panel(panel, tile_id);
                    let Some(tile_id) = tile_id else {
                        continue;
                    };

                    let Some(child_id) = child_id else {
                        continue;
                    };
                    let Some(share) = split.shares.get(&i) else {
                        continue;
                    };
                    let Some(Tile::Container(Container::Linear(linear))) =
                        self.tile_state.tree.tiles.get_mut(tile_id)
                    else {
                        continue;
                    };
                    linear.shares.set_share(child_id, *share);
                }
                tile_id
            }
            Panel::Tabs(tabs) => {
                let tile_id = self.tile_state.insert_tile(
                    Tile::Container(Container::new_tabs(vec![])),
                    parent_id,
                    false,
                );

                tabs.iter().for_each(|panel| {
                    self.spawn_panel(panel, tile_id);
                });
                tile_id
            }
            Panel::Graph(graph) => {
                let eql = self
                    .eql
                    .0
                    .parse_str(&graph.eql)
                    .inspect_err(|err| {
                        warn!(?err, "error parsing graph eql");
                    })
                    .ok()?;
                let mut component_vec = eql.to_graph_components();
                component_vec.sort();
                let mut components_tree: BTreeMap<ComponentPath, Vec<(bool, Color32)>> =
                    BTreeMap::new();
                for (j, (component, i)) in component_vec.iter().enumerate() {
                    if let Some(elements) = components_tree.get_mut(component) {
                        elements[*i] = (true, colors::get_color_by_index_all(j));
                    } else {
                        let Some(schema) = self.schema_reg.0.get(&component.id) else {
                            continue;
                        };
                        let len: usize = schema.shape().iter().copied().product();
                        let mut elements: Vec<(bool, Color32)> = (0..len)
                            .map(|_| (false, colors::get_color_by_index_all(j)))
                            .collect();
                        elements[*i] = (true, colors::get_color_by_index_all(j));
                        components_tree.insert(component.clone(), elements);
                    }
                }

                let graph_label = graph_label(graph);
                let mut bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    components_tree,
                    graph_label.clone(),
                );
                bundle.graph_state.auto_y_range = graph.auto_y_range;
                bundle.graph_state.y_range = graph.y_range.clone();
                bundle.graph_state.graph_type = graph.graph_type;
                let graph_id = self.commands.spawn(bundle).id();
                let graph = GraphPane::new(graph_id, graph_label);
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::Graph(graph)), parent_id, false)
            }
            Panel::ComponentMonitor(monitor) => {
                // Create a MonitorPane and add it to the UI
                let pane = MonitorPane::new("Monitor".to_string(), monitor.component_id);
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::Monitor(pane)), parent_id, false)
            }
            Panel::QueryTable(data) => {
                // Create a new SQL table entity
                let entity = self
                    .commands
                    .spawn(super::query_table::QueryTableData {
                        data: data.clone(),
                        ..Default::default()
                    })
                    .id();
                let pane = super::query_table::QueryTablePane { entity };
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::QueryTable(pane)), parent_id, false)
            }
            Panel::ActionPane(action) => {
                // Create a new action tile entity
                let entity = self
                    .commands
                    .spawn(super::actions::ActionTile {
                        button_name: action.label.clone(),
                        lua: action.lua.clone(),
                        status: Default::default(),
                    })
                    .id();
                let pane = super::tiles::ActionTilePane {
                    entity,
                    label: "Action".to_string(),
                };
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::ActionTile(pane)), parent_id, false)
            }
            Panel::Inspector => {
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::Inspector), parent_id, false)
            }
            Panel::Hierarchy => {
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::Hierarchy), parent_id, false)
            }
            Panel::SchematicTree => {
                let entity = self.commands.spawn(super::TreeWidgetState::default()).id();
                let pane = TreePane { entity };
                self.tile_state
                    .insert_tile(Tile::Pane(Pane::SchematicTree(pane)), parent_id, false)
            }
            Panel::QueryPlot(plot) => {
                let graph_bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    BTreeMap::default(),
                    "Query Plot".to_string(),
                );
                let entity = self
                    .commands
                    .spawn(QueryPlotData {
                        data: plot.clone(),
                        ..Default::default()
                    })
                    .insert(graph_bundle)
                    .id();
                let pane = Pane::QueryPlot(super::query_plot::QueryPlotPane { entity, rect: None });
                self.tile_state
                    .insert_tile(Tile::Pane(pane), parent_id, false)
            }
            Panel::Dashboard(dashboard) => {
                let Ok(dashboard) = spawn_dashboard(
                    dashboard,
                    &self.eql.0,
                    &mut self.commands,
                    &self.node_updater_params,
                )
                .inspect_err(|err| {
                    warn!(?err, "Failed to spawn dashboard");
                }) else {
                    return None;
                };
                self.tile_state.insert_tile(
                    Tile::Pane(Pane::Dashboard(DashboardPane {
                        entity: dashboard,
                        label: "dashboard".to_string(),
                    })),
                    parent_id,
                    false,
                )
            }
        }
    }
}

pub fn viewport_label(viewport: &Viewport) -> String {
    viewport
        .name
        .clone()
        .unwrap_or_else(|| "Viewport".to_string())
}

pub fn graph_label(graph: &Graph) -> String {
    // TODO: Update graph labeling once Graph structure is migrated to use ComponentPath
    graph.name.clone().unwrap_or_else(|| "Graph".to_string())
}

#[derive(Default, Deref, DerefMut, Resource)]
pub struct SchematicLiveReloadRx(pub Option<flume::Receiver<Schematic>>);

pub fn schematic_live_reload(
    mut rx: ResMut<SchematicLiveReloadRx>,
    mut params: LoadSchematicParams,
) {
    let Some(rx) = &mut rx.0 else { return };
    let Ok(schematic) = rx.try_recv() else { return };
    params.load_schematic(&schematic);
}
