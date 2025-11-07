use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui::{Color32, Id};
use bevy_infinite_grid::InfiniteGrid;
use egui_tiles::{Container, Tile, TileId};
use impeller2_bevy::{ComponentPath, ComponentSchemaRegistry};
use impeller2_kdl::FromKdl;
use impeller2_kdl::KdlSchematicError;
use impeller2_wkt::{
    DbConfig, Graph, Line3d, Object3D, Panel, Schematic, VectorArrow3d, Viewport, WindowSchematic,
};
use miette::{Diagnostic, miette};
use std::time::Duration;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::{
    EqlContext, MainCamera,
    object_3d::Object3DState,
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        HdrEnabled, SelectedObject,
        colors::{self, EColor},
        dashboard::{NodeUpdaterParams, spawn_dashboard},
        modal::ModalDialog,
        monitor::MonitorPane,
        plot::GraphBundle,
        query_plot::QueryPlotData,
        schematic::EqlExt,
        tiles::{
            DashboardPane, GraphPane, Pane, SecondaryWindowDescriptor, SecondaryWindowId,
            SecondaryWindowState, TileState, TreePane, ViewportPane, WindowManager,
        },
    },
    vector_arrow::VectorArrowState,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PanelContext {
    Main,
    Secondary(SecondaryWindowId),
}

#[derive(Component)]
pub struct SyncedViewport;

#[derive(SystemParam)]
pub struct LoadSchematicParams<'w, 's> {
    pub commands: Commands<'w, 's>,
    pub windows: ResMut<'w, WindowManager>,
    pub asset_server: Res<'w, AssetServer>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
    pub render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    pub hdr_enabled: ResMut<'w, HdrEnabled>,
    pub schema_reg: Res<'w, ComponentSchemaRegistry>,
    pub eql: Res<'w, EqlContext>,
    pub selected_object: ResMut<'w, SelectedObject>,
    pub node_updater_params: NodeUpdaterParams<'w, 's>,
    cameras: Query<'w, 's, &'static mut Camera>,
    objects_3d: Query<'w, 's, Entity, With<Object3DState>>,
    vector_arrows: Query<'w, 's, Entity, With<VectorArrowState>>,
    grid_lines: Query<'w, 's, Entity, With<InfiniteGrid>>,
}

pub fn sync_schematic(
    config: Res<DbConfig>,
    mut params: LoadSchematicParams,
    live_reload_rx: ResMut<SchematicLiveReloadRx>,
    mut modal: ModalDialog,
) {
    if !config.is_changed() {
        return;
    }
    if let Some(path) = config.schematic_path() {
        let path = Path::new(path);
        if path.try_exists().unwrap_or(false) {
            if let Err(e) = load_schematic_file(path, &mut params, live_reload_rx) {
                modal.dialog_error(
                    format!("Invalid Schematic in {:?}", path.display()),
                    &render_diag(&e),
                );
                let report = miette!(e.clone());
                bevy::log::error!(?report, "Invalid schematic for {path:?}");
            } else {
                return;
            }
        }
    }
    if let Some(content) = config.schematic_content() {
        let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(content).inspect_err(|e| {
            modal.dialog_error("Invalid Schematic", &render_diag(e));
            let report = miette!(e.clone());
            bevy::log::error!(?report, "Invalid schematic content")
        }) else {
            return;
        };
        params.load_schematic(&schematic, None);
    }
}

fn resolve_window_descriptor(
    window: &WindowSchematic,
    base_dir: Option<&Path>,
) -> Option<SecondaryWindowDescriptor> {
    let mut resolved = PathBuf::from(&window.path);

    if resolved.as_os_str().is_empty() {
        return None;
    }

    if resolved.is_relative() {
        if let Some(base) = base_dir {
            resolved = base.join(resolved);
        } else if let Ok(cwd) = std::env::current_dir() {
            resolved = cwd.join(resolved);
        }
    }

    Some(SecondaryWindowDescriptor {
        path: resolved,
        title: window.title.clone(),
        screen_index: window.screen_index.map(|idx| idx as usize),
    })
}

pub fn render_diag(diagnostic: &dyn Diagnostic) -> String {
    let mut buf = String::new();
    miette::GraphicalReportHandler::new_themed(miette::GraphicalTheme::unicode_nocolor())
        .render_report(&mut buf, diagnostic)
        .expect("Failed to render diagnostic");
    buf
}

pub fn load_schematic_file(
    path: &Path,
    params: &mut LoadSchematicParams,
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
        params.load_schematic(&schematic, path.parent());
    }
    Ok(())
}

impl LoadSchematicParams<'_, '_> {
    pub fn load_schematic(&mut self, schematic: &Schematic, base_dir: Option<&Path>) {
        self.render_layer_alloc.free_all();
        for secondary in self.windows.take_secondary() {
            for graph in secondary.graph_entities {
                self.commands.entity(graph).despawn();
            }
            if let Some(entity) = secondary.window_entity {
                self.commands.entity(entity).despawn();
            }
        }
        let mut main_state = self.windows.take_main();
        main_state.clear(&mut self.commands, &mut self.selected_object);
        self.hdr_enabled.0 = false;
        for entity in self.objects_3d.iter() {
            self.commands.entity(entity).despawn();
        }
        for entity in self.vector_arrows.iter() {
            self.commands.entity(entity).despawn();
        }
        // Remove all GridLines before loading new schematic.
        for entity in self.grid_lines.iter() {
            self.commands.entity(entity).despawn();
        }
        let mut secondary_descriptors: Vec<SecondaryWindowDescriptor> = Vec::new();

        for elem in &schematic.elems {
            match elem {
                impeller2_wkt::SchematicElem::Panel(p) => {
                    self.spawn_panel(&mut main_state, p, None, PanelContext::Main);
                }
                impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                    self.spawn_object_3d(object_3d.clone());
                }
                impeller2_wkt::SchematicElem::Line3d(line_3d) => {
                    self.spawn_line_3d(line_3d.clone());
                }
                impeller2_wkt::SchematicElem::VectorArrow(vector_arrow) => {
                    self.spawn_vector_arrow(vector_arrow.clone());
                }
                impeller2_wkt::SchematicElem::Window(window) => {
                    if let Some(descriptor) = resolve_window_descriptor(window, base_dir) {
                        secondary_descriptors.push(descriptor);
                    }
                }
            }
        }

        self.windows.replace_main(main_state);

        let mut secondary_states = Vec::new();

        for descriptor in secondary_descriptors {
            match std::fs::read_to_string(&descriptor.path) {
                Ok(kdl) => match impeller2_wkt::Schematic::from_kdl(&kdl) {
                    Ok(sec_schematic) => {
                        let id = self.windows.alloc_id();
                        let mut tile_state = TileState::new(Id::new(("secondary_tab_tree", id.0)));
                        for elem in &sec_schematic.elems {
                            if let impeller2_wkt::SchematicElem::Panel(panel) = elem {
                                self.spawn_panel(
                                    &mut tile_state,
                                    panel,
                                    None,
                                    PanelContext::Secondary(id),
                                );
                            }
                        }

                        let graph_entities = tile_state.collect_graph_entities();
                        info!(
                            path = %descriptor.path.display(),
                            "Loaded secondary schematic"
                        );

                        for &graph in &graph_entities {
                            if let Ok(mut camera) = self.cameras.get_mut(graph) {
                                camera.is_active = false;
                            }
                        }

                        secondary_states.push(SecondaryWindowState {
                            id,
                            descriptor,
                            tile_state,
                            window_entity: None,
                            graph_entities,
                        });
                    }
                    Err(err) => {
                        let diag = render_diag(&err);
                        let report = miette!(err.clone());
                        warn!(
                            ?report,
                            path = %descriptor.path.display(),
                            "Failed to parse secondary schematic: \n{diag}"
                        );
                    }
                },
                Err(err) => {
                    warn!(
                        ?err,
                        path = %descriptor.path.display(),
                        "Failed to read secondary schematic"
                    );
                }
            }
        }

        self.windows.replace_secondary(secondary_states);
    }

    pub fn spawn_object_3d(&mut self, object_3d: Object3D) {
        let Ok(expr) = self.eql.0.parse_str(&object_3d.eql) else {
            return;
        };
        crate::object_3d::create_object_3d_entity(
            &mut self.commands,
            object_3d.clone(),
            expr,
            &self.eql.0,
            &mut self.materials,
            &mut self.meshes,
            &self.asset_server,
        );
    }

    pub fn spawn_line_3d(&mut self, line_3d: Line3d) {
        self.commands.spawn(line_3d);
    }

    pub fn spawn_vector_arrow(&mut self, vector_arrow: VectorArrow3d) {
        use crate::object_3d::compile_eql_expr;

        let vector_expr = self
            .eql
            .0
            .parse_str(&vector_arrow.vector)
            .map(compile_eql_expr)
            .ok();

        let origin_expr = vector_arrow
            .origin
            .as_ref()
            .and_then(|origin| self.eql.0.parse_str(origin).ok())
            .map(compile_eql_expr);

        self.commands.spawn((
            vector_arrow,
            VectorArrowState {
                vector_expr,
                origin_expr,
            },
        ));
    }

    fn spawn_panel(
        &mut self,
        tile_state: &mut TileState,
        panel: &Panel,
        parent_id: Option<TileId>,
        context: PanelContext,
    ) -> Option<TileId> {
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
                tile_state.insert_tile(Tile::Pane(Pane::Viewport(pane)), parent_id, viewport.active)
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
                let tile_id = tile_state.insert_tile(
                    Tile::Container(Container::Linear(linear)),
                    parent_id,
                    false,
                );
                if let (Some(tile_id), Some(name)) = (tile_id, split.name.clone()) {
                    tile_state.container_titles.insert(tile_id, name);
                }
                for (i, panel) in split.panels.iter().enumerate() {
                    let child_id = self.spawn_panel(tile_state, panel, tile_id, context);
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
                        tile_state.tree.tiles.get_mut(tile_id)
                    else {
                        continue;
                    };
                    linear.shares.set_share(child_id, *share);
                }
                tile_id
            }
            Panel::Tabs(tabs) => {
                let tile_id = tile_state.insert_tile(
                    Tile::Container(Container::new_tabs(vec![])),
                    parent_id,
                    false,
                );

                tabs.iter().for_each(|panel| {
                    self.spawn_panel(tile_state, panel, tile_id, context);
                });
                tile_id
            }
            Panel::Graph(graph) => {
                let eql = self
                    .eql
                    .0
                    .parse_str(&graph.eql)
                    .inspect_err(|err| {
                        let (ctx, path) = match context {
                            PanelContext::Main => ("main".to_string(), None),
                            PanelContext::Secondary(id) => {
                                let path = self
                                    .windows
                                    .get_secondary(id)
                                    .map(|s| s.descriptor.path.display().to_string());
                                (format!("secondary({})", id.0), path)
                            }
                        };
                        if let Some(p) = path {
                            warn!(
                                ?err,
                                eql = %graph.eql,
                                name = ?graph.name,
                                context = %ctx,
                                path = %p,
                                "error parsing graph eql"
                            );
                        } else {
                            warn!(
                                ?err,
                                eql = %graph.eql,
                                name = ?graph.name,
                                context = %ctx,
                                "error parsing graph eql"
                            );
                        }
                    })
                    .ok()?;
                let mut component_vec = eql.to_graph_components();
                component_vec.sort();
                let mut components_tree: BTreeMap<ComponentPath, Vec<(bool, Color32)>> =
                    BTreeMap::new();
                for (j, (component, i)) in component_vec.iter().enumerate() {
                    let line_color = graph
                        .colors
                        .get(j)
                        .copied()
                        .map(EColor::into_color32)
                        .unwrap_or_else(|| colors::get_color_by_index_all(j));
                    if let Some(elements) = components_tree.get_mut(component) {
                        elements[*i] = (true, line_color);
                    } else {
                        let Some(schema) = self.schema_reg.0.get(&component.id) else {
                            continue;
                        };
                        let len: usize = schema.shape().iter().copied().product();
                        let mut elements: Vec<(bool, Color32)> =
                            (0..len).map(|_| (false, line_color)).collect();
                        elements[*i] = (true, line_color);
                        components_tree.insert(component.clone(), elements);
                    }
                }

                let graph_label = graph_label(graph);

                let mut bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    components_tree,
                    graph_label.clone(),
                );
                if matches!(context, PanelContext::Secondary(_)) {
                    bundle.camera.is_active = false;
                }
                bundle.graph_state.auto_y_range = graph.auto_y_range;
                bundle.graph_state.y_range = graph.y_range.clone();
                bundle.graph_state.graph_type = graph.graph_type;
                let graph_id = self.commands.spawn(bundle).id();
                if matches!(context, PanelContext::Secondary(_)) {
                    self.commands.entity(graph_id).remove::<MainCamera>();
                }
                let graph = GraphPane::new(graph_id, graph_label);
                tile_state.insert_tile(Tile::Pane(Pane::Graph(graph)), parent_id, false)
            }
            Panel::ComponentMonitor(monitor) => {
                let pane = MonitorPane::new("Monitor".to_string(), monitor.component_name.clone());
                tile_state.insert_tile(Tile::Pane(Pane::Monitor(pane)), parent_id, false)
            }
            Panel::QueryTable(data) => {
                let entity = self
                    .commands
                    .spawn(super::query_table::QueryTableData {
                        data: data.clone(),
                        ..Default::default()
                    })
                    .id();
                let pane = super::query_table::QueryTablePane { entity };
                tile_state.insert_tile(Tile::Pane(Pane::QueryTable(pane)), parent_id, false)
            }
            Panel::ActionPane(action) => {
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
                tile_state.insert_tile(Tile::Pane(Pane::ActionTile(pane)), parent_id, false)
            }
            Panel::Inspector => {
                tile_state.insert_tile(Tile::Pane(Pane::Inspector), parent_id, false)
            }
            Panel::Hierarchy => {
                tile_state.insert_tile(Tile::Pane(Pane::Hierarchy), parent_id, false)
            }
            Panel::SchematicTree => {
                let entity = self.commands.spawn(super::TreeWidgetState::default()).id();
                let pane = TreePane { entity };
                tile_state.insert_tile(Tile::Pane(Pane::SchematicTree(pane)), parent_id, false)
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
                tile_state.insert_tile(Tile::Pane(pane), parent_id, false)
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
                tile_state.insert_tile(
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

/// Prefer the explicit `name` when set (and not the generic "Graph").
/// Otherwise, derive a readable label from the first EQL term.
pub fn graph_label(graph: &Graph) -> String {
    if let Some(name) = graph.name.as_ref() {
        let trimmed = name.trim();
        if !trimmed.is_empty() && trimmed != "Graph" {
            return trimmed.to_string();
        }
    }
    graph
        .eql
        .split(',')
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Graph".to_string())
}

#[derive(Default, Deref, DerefMut, Resource)]
pub struct SchematicLiveReloadRx(pub Option<flume::Receiver<Schematic>>);

pub fn schematic_live_reload(
    mut rx: ResMut<SchematicLiveReloadRx>,
    mut params: LoadSchematicParams,
) {
    let Some(rx) = &mut rx.0 else { return };
    let Ok(schematic) = rx.try_recv() else { return };
    params.load_schematic(&schematic, None);
}
