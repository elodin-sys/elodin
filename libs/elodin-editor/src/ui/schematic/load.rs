use bevy::{ecs::system::SystemParam, prelude::*, window::PrimaryWindow};
#[cfg(target_os = "macos")]
use bevy_defer::AsyncCommandsExtension;
use bevy_egui::egui::{Color32, Id};
use bevy_infinite_grid::InfiniteGrid;
use egui_tiles::{Container, Tile, TileId};
use impeller2_bevy::{ComponentPath, ComponentSchemaRegistry, DbMessage};
use impeller2_kdl::FromKdl;
use impeller2_kdl::KdlSchematicError;
use impeller2_wkt::{
    DbConfig, Graph, Line3d, Object3D, Panel, Schematic, VectorArrow3d, Viewport, WindowSchematic,
};

/// When set (e.g. by CLI `--kdl`), the path is applied to `DbConfig` once so that the schematic
/// is loaded after connecting to the database.
#[derive(Resource, Default)]
pub struct InitialKdlPath(pub Option<PathBuf>);

pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<InitialKdlPath>();
}

/// Applies `InitialKdlPath` to `DbConfig` so that `sync_schematic` will load that file.
/// Runs before `sync_schematic`. Re-applies when the path is missing or different (e.g.
/// after the connection overwrote DbConfig with metadata) so the schematic loads.
pub fn apply_initial_kdl_path(
    mut reader: MessageReader<DbMessage>,
    initial: Res<InitialKdlPath>,
) -> Option<PathBuf> {
    if !reader.read().any(|m| matches!(m, DbMessage::UpdateConfig)) {
        None
    } else {
        initial.0.clone()
    }
}
use miette::{Diagnostic, miette};
#[cfg(target_os = "linux")]
use std::time::{Instant, SystemTime};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(target_os = "linux")]
const NOTIFY_SILENCE: Duration = Duration::from_millis(3000);

#[cfg(not(target_os = "macos"))]
use crate::tiles::WindowRelayout;
#[cfg(target_os = "macos")]
use crate::ui::window::placement::apply_physical_screen_rect;
use crate::{
    EqlContext, MainCamera,
    object_3d::Object3DState,
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        DEFAULT_SECONDARY_RECT, HdrEnabled,
        colors::{self, EColor},
        dashboard::{NodeUpdaterParams, spawn_dashboard},
        data_overview::DataOverviewPane,
        modal::ModalDialog,
        monitor::MonitorPane,
        plot::GraphBundle,
        query_plot::QueryPlotData,
        schematic::EqlExt,
        tiles::{
            DashboardPane, GraphPane, Pane, TileState, TreePane, ViewportPane, WindowDescriptor,
            WindowId, WindowState,
        },
    },
    vector_arrow::{VectorArrowState, ViewportArrow},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PanelContext {
    Main,
    Secondary(WindowId),
}

fn tabs_parent_for_panels(tile_state: &TileState, panel_count: usize) -> Option<TileId> {
    if panel_count > 0 {
        tile_state.tree.root()
    } else {
        None
    }
}

#[derive(Component)]
pub struct SyncedViewport;

#[derive(SystemParam)]
pub struct LoadSchematicParams<'w, 's> {
    pub commands: Commands<'w, 's>,
    primary_window: Single<'w, 's, Entity, With<PrimaryWindow>>,
    pub asset_server: Res<'w, AssetServer>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
    pub render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    pub hdr_enabled: ResMut<'w, HdrEnabled>,
    pub schema_reg: Res<'w, ComponentSchemaRegistry>,
    pub eql: Res<'w, EqlContext>,
    pub node_updater_params: NodeUpdaterParams<'w, 's>,
    cameras: Query<'w, 's, &'static mut Camera>,
    objects_3d: Query<'w, 's, Entity, With<Object3DState>>,
    vector_arrows: Query<'w, 's, Entity, With<VectorArrowState>>,
    grid_lines: Query<'w, 's, Entity, With<InfiniteGrid>>,
    window_states: Query<'w, 's, (Entity, &'static WindowId, &'static mut WindowState)>,
}

pub fn sync_schematic(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
    mut params: LoadSchematicParams,
    live_reload_rx: ResMut<SchematicLiveReloadRx>,
    mut modal: ModalDialog,
) {
    if given_path.is_none() && !config.is_changed() {
        return;
    }
    if let Some(path) = given_path.or(config.schematic_path().map(PathBuf::from)) {
        // NOTE: This path is not resolved yet. We can't test if it exists here.
        // load_schematic_file resolves it and should do that test there.
        if let Err(e) = load_schematic_file(&path, &mut params, live_reload_rx) {
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
    if let Some(content) = config.schematic_content() {
        let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(content).inspect_err(|e| {
            modal.dialog_error("Invalid Schematic", &render_diag(e));
            let report = miette!(e.clone());
            bevy::log::error!(?report, "Invalid schematic content")
        }) else {
            return;
        };
        match config.schematic_path() {
            Some(p) => {
                let base_file = impeller2_kdl::env::schematic_file(Path::new(p));
                params.load_schematic(&schematic, base_file.parent());
            }
            None => {
                params.load_schematic(&schematic, None);
            }
        }
        return;
    }

    // No schematic found - create a default Data Overview panel
    params.load_default_data_overview();
}

fn apply_theme(theme: Option<&impeller2_wkt::ThemeConfig>) -> colors::SchemeSelection {
    let current = colors::current_selection();
    let scheme = theme
        .and_then(|t| t.scheme.as_deref())
        .unwrap_or(&current.scheme);
    let mode = theme
        .and_then(|t| t.mode.as_deref())
        .unwrap_or(&current.mode);
    colors::set_active_scheme(scheme, mode)
}

fn resolve_window_descriptor(
    window: &WindowSchematic,
    base_dir: Option<&Path>,
    theme_mode: Option<&str>,
) -> Option<WindowDescriptor> {
    let path_str = window.path.as_ref()?;
    let mut resolved = PathBuf::from(path_str);

    if resolved.is_relative() {
        if let Some(base) = base_dir {
            resolved = base.join(resolved);
        } else if let Ok(cwd) = std::env::current_dir() {
            resolved = cwd.join(resolved);
        }
    }

    Some(WindowDescriptor {
        path: Some(resolved),
        title: window.title.clone(),
        screen: window.screen.map(|idx| idx as usize),
        mode: theme_mode.map(|m| m.to_string()),
        screen_rect: window.screen_rect.or(Some(DEFAULT_SECONDARY_RECT)),
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
    let resolved_path = impeller2_kdl::env::schematic_file(path);
    if !resolved_path.try_exists().unwrap_or(false) {
        return Err(KdlSchematicError::NoSuchFile {
            path: resolved_path,
        });
    }
    let watch_path = resolved_path.clone();
    let (tx, rx) = flume::bounded(1);
    live_reload_rx.set_receiver(rx);
    #[cfg(target_os = "linux")]
    live_reload_rx.guard_for(Duration::from_millis(2000));
    std::thread::spawn(move || {
        let cb_path = watch_path.clone();
        let mut debouncer = notify_debouncer_mini::new_debouncer(
            Duration::from_millis(100),
            move |res: notify_debouncer_mini::DebounceEventResult| {
                if res.is_err() {
                    return;
                }

                if let Ok(kdl) = std::fs::read_to_string(&cb_path) {
                    let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(&kdl) else {
                        return;
                    };
                    let _ = tx.send(SchematicReloadEvent {
                        path: cb_path.clone(),
                        schematic,
                    });
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
    if let Ok(kdl) = std::fs::read_to_string(&resolved_path) {
        let schematic = impeller2_wkt::Schematic::from_kdl(&kdl)?;
        let base_dir_maybe = resolved_path.parent();
        params.load_schematic(&schematic, base_dir_maybe);
        live_reload_rx.record_loaded(&resolved_path);
    }
    Ok(())
}

impl LoadSchematicParams<'_, '_> {
    pub fn load_schematic(&mut self, schematic: &Schematic, base_dir: Option<&Path>) {
        self.render_layer_alloc.free_all();
        for (id, window_id, window_state) in &self.window_states {
            if window_id.is_primary() {
                // We do not despawn the primary window ever.
                continue;
            }
            // for secondary in self.windows.take_secondary() {
            for graph in window_state.graph_entities.iter() {
                self.commands.entity(*graph).despawn();
            }
            self.commands.entity(id).despawn();
        }

        let primary_window = *self.primary_window;
        let mut main_state = {
            let mut window_state = self
                .window_states
                .get_mut(primary_window)
                .expect("no primary window")
                .2;
            std::mem::take(&mut window_state.tile_state)
        };
        main_state.clear(&mut self.commands, &mut self.render_layer_alloc);
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
        let mut secondary_descriptors: Vec<WindowDescriptor> = Vec::new();
        let theme_selection = apply_theme(schematic.theme.as_ref());
        let theme_mode = Some(theme_selection.mode.clone());
        let theme_mode_str = theme_mode.as_deref();
        let mut main_window_descriptor = None;

        let panel_count = schematic
            .elems
            .iter()
            .filter(|elem| matches!(elem, impeller2_wkt::SchematicElem::Panel(_)))
            .count();
        let tabs_parent = tabs_parent_for_panels(&main_state, panel_count);
        let mut main_ui_state = crate::ui::WindowUiState::default();

        for elem in &schematic.elems {
            match elem {
                impeller2_wkt::SchematicElem::Panel(p) => {
                    self.spawn_panel(
                        &mut main_state,
                        &mut main_ui_state,
                        p,
                        tabs_parent,
                        PanelContext::Main,
                    );
                }
                impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                    self.spawn_object_3d(object_3d.clone());
                }
                impeller2_wkt::SchematicElem::Line3d(line_3d) => {
                    self.spawn_line_3d(line_3d.clone());
                }
                impeller2_wkt::SchematicElem::VectorArrow(vector_arrow) => {
                    self.spawn_vector_arrow(vector_arrow.clone(), None);
                }
                impeller2_wkt::SchematicElem::Window(window) => {
                    if let Some(descriptor) =
                        resolve_window_descriptor(window, base_dir, theme_mode_str)
                    {
                        secondary_descriptors.push(descriptor);
                    } else {
                        main_window_descriptor = Some(WindowDescriptor {
                            screen: window.screen.map(|idx| idx as usize),
                            screen_rect: window.screen_rect,
                            ..default()
                        });
                    }
                }
                impeller2_wkt::SchematicElem::Theme(_) => {}
            }
        }

        {
            let mut window_state = self
                .window_states
                .get_mut(*self.primary_window)
                .expect("no primary window")
                .2;
            match main_window_descriptor {
                Some(d) => {
                    window_state.descriptor.screen = d.screen;
                    window_state.descriptor.screen_rect = d.screen_rect;
                }
                None => {
                    // No primary window entry in KDL: clear any stale placement to avoid
                    // reapplying an old rect.
                    window_state.descriptor.screen = None;
                    window_state.descriptor.screen_rect = None;
                }
            }
            let _ = std::mem::replace(&mut window_state.tile_state, main_state);
            // Apply sidebar visibility from loaded schematic.
            window_state.ui_state.left_sidebar_visible = main_ui_state.left_sidebar_visible;
            window_state.ui_state.right_sidebar_visible = main_ui_state.right_sidebar_visible;
            // Resize if necessary.
            //
            #[cfg(not(target_os = "macos"))]
            {
                if let Some(screen) = window_state.descriptor.screen.as_ref() {
                    self.commands.write_message(WindowRelayout::Screen {
                        window: primary_window,
                        screen: *screen,
                    });
                }

                if let Some(rect) = window_state.descriptor.screen_rect.as_ref() {
                    self.commands.write_message(WindowRelayout::Rect {
                        window: primary_window,
                        rect: *rect,
                    });
                }
            }
            #[cfg(target_os = "macos")]
            {
                if let (Some(screen_idx), Some(rect)) = (
                    window_state.descriptor.screen,
                    window_state.descriptor.screen_rect,
                ) {
                    self.commands.spawn_task(move || async move {
                        apply_physical_screen_rect(primary_window, screen_idx, rect)
                            .await
                            .ok();
                        Ok(())
                    });
                } else {
                    info!("mac primary: no screen/rect in KDL, skipping placement");
                }
            }
        }

        if let Some(mode) = theme_mode.as_ref() {
            for (_entity, _window_id, mut window_state) in self.window_states.iter_mut() {
                window_state.descriptor.mode = Some(mode.clone());
            }
        }

        for descriptor in secondary_descriptors {
            if let Some(path) = descriptor.path.as_ref() {
                match std::fs::read_to_string(path) {
                    Ok(kdl) => match impeller2_wkt::Schematic::from_kdl(&kdl) {
                        Ok(sec_schematic) => {
                            let theme_mode = sec_schematic
                                .theme
                                .as_ref()
                                .and_then(|t| t.mode.as_deref())
                                .or(theme_mode.as_deref());
                            let resolved_mode = theme_mode.map(|mode| {
                                if colors::scheme_supports_mode(&theme_selection.scheme, mode) {
                                    mode.to_string()
                                } else {
                                    "dark".to_string()
                                }
                            });
                            let id = WindowId::default();
                            let mut tile_state =
                                TileState::new(Id::new(("secondary_tab_tree", id.0)));
                            let panel_count = sec_schematic
                                .elems
                                .iter()
                                .filter(|elem| {
                                    matches!(elem, impeller2_wkt::SchematicElem::Panel(_))
                                })
                                .count();
                            let tabs_parent = tabs_parent_for_panels(&tile_state, panel_count);
                            let mut ui_state = crate::ui::WindowUiState::default();

                            for elem in &sec_schematic.elems {
                                if let impeller2_wkt::SchematicElem::Panel(panel) = elem {
                                    self.spawn_panel(
                                        &mut tile_state,
                                        &mut ui_state,
                                        panel,
                                        tabs_parent,
                                        PanelContext::Secondary(id),
                                    );
                                }
                            }

                            let graph_entities = tile_state.collect_graph_entities();
                            let descriptor = WindowDescriptor {
                                mode: resolved_mode,
                                ..descriptor
                            };
                            info!(
                                path = ?descriptor.path,
                                "Loaded secondary schematic"
                            );

                            for &graph in &graph_entities {
                                if let Ok(mut camera) = self.cameras.get_mut(graph) {
                                    // Why do we turn their cameras off?
                                    // Oh, so we can set their `WindowRef` first.
                                    camera.is_active = false;
                                }
                            }

                            let state = WindowState {
                                descriptor,
                                tile_state,
                                graph_entities,
                                ui_state,
                            };
                            self.commands.spawn((id, state));
                        }
                        Err(err) => {
                            let diag = render_diag(&err);
                            let report = miette!(err.clone());
                            warn!(
                                ?report,
                                path = ?descriptor.path,
                                "Failed to parse secondary schematic: \n{diag}"
                            );
                        }
                    },
                    Err(err) => {
                        warn!(
                            ?err,
                            path = ?descriptor.path,
                            "Failed to read secondary schematic"
                        );
                    }
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
            &self.eql.0,
            &mut self.materials,
            &mut self.meshes,
            &self.asset_server,
        );
    }

    pub fn spawn_line_3d(&mut self, line_3d: Line3d) {
        self.commands.spawn(line_3d);
    }

    pub fn spawn_vector_arrow(
        &mut self,
        vector_arrow: VectorArrow3d,
        viewport_camera: Option<Entity>,
    ) {
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

        let mut spawn = self.commands.spawn((
            vector_arrow,
            VectorArrowState {
                vector_expr,
                origin_expr,
                visuals: HashMap::new(),
                label: None,
                ..default()
            },
        ));

        if let Some(camera) = viewport_camera {
            spawn.insert(ViewportArrow { camera });
        }
    }

    fn spawn_panel(
        &mut self,
        tile_state: &mut TileState,
        ui_state: &mut crate::ui::WindowUiState,
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
                if let Some(camera) = pane.camera {
                    for arrow in viewport.local_arrows.clone() {
                        self.spawn_vector_arrow(arrow, Some(camera));
                    }
                }
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
                    let child_id = self.spawn_panel(tile_state, ui_state, panel, tile_id, context);
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
                let mut tile_id = None;
                let mut parent_for_children = parent_id;
                let mut reuse_parent = false;

                if matches!(context, PanelContext::Secondary(_))
                    && let Some(parent_id) = parent_id
                    && tile_state.tree.root() == Some(parent_id)
                    && let Some(Tile::Container(Container::Tabs(root_tabs))) =
                        tile_state.tree.tiles.get(parent_id)
                {
                    // Reuse parent if root tabs are empty
                    if root_tabs.children.is_empty() {
                        reuse_parent = true;
                        tile_id = Some(parent_id);
                        parent_for_children = Some(parent_id);
                    }
                }

                if !reuse_parent {
                    tile_id = tile_state.insert_tile(
                        Tile::Container(Container::new_tabs(vec![])),
                        parent_id,
                        false,
                    );
                    parent_for_children = tile_id;
                }

                for panel in tabs.iter() {
                    self.spawn_panel(tile_state, ui_state, panel, parent_for_children, context);
                }
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
                            PanelContext::Secondary(target_id) => {
                                let path = self
                                    .window_states
                                    .iter()
                                    .find(|(_entity, id, _state)| **id == target_id)
                                    .and_then(|(_, _, s)| {
                                        s.descriptor.path.as_ref().map(|p| p.display().to_string())
                                    });
                                (format!("secondary({})", target_id.0), path)
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
                bundle.graph_state.locked = graph.locked;
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
                let label = monitor
                    .name
                    .clone()
                    .unwrap_or_else(|| monitor.component_name.clone());
                let entity = self
                    .commands
                    .spawn(super::monitor::MonitorData {
                        component_name: monitor.component_name.clone(),
                    })
                    .id();
                let pane = MonitorPane::new(entity, label);
                tile_state.insert_tile(Tile::Pane(Pane::Monitor(pane)), parent_id, false)
            }
            Panel::QueryTable(data) => {
                let has_query = !data.query.trim().is_empty();
                let entity = self
                    .commands
                    .spawn(super::query_table::QueryTableData {
                        data: data.clone(),
                        pending_execution: has_query,
                        ..Default::default()
                    })
                    .id();
                let label = data
                    .name
                    .clone()
                    .filter(|name| !name.trim().is_empty())
                    .unwrap_or_else(|| "Query Table".to_string());
                let pane = super::query_table::QueryTablePane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::QueryTable(pane)), parent_id, false)
            }
            Panel::ActionPane(action) => {
                let entity = self
                    .commands
                    .spawn(super::actions::ActionTile {
                        button_name: action.name.clone(),
                        lua: action.lua.clone(),
                        status: Default::default(),
                    })
                    .id();
                let pane = super::tiles::ActionTilePane {
                    entity,
                    name: action.name.clone(),
                };
                tile_state.insert_tile(Tile::Pane(Pane::ActionTile(pane)), parent_id, false)
            }
            Panel::VideoStream(video_stream) => {
                let msg_id = impeller2::types::msg_id(&video_stream.msg_name);
                let label = video_stream
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Video Stream {}", video_stream.msg_name));

                let entity = self
                    .commands
                    .spawn((
                        crate::ui::video_stream::VideoStream {
                            msg_id,
                            msg_name: video_stream.msg_name.clone(),
                            ..Default::default()
                        },
                        bevy::ui::Node {
                            position_type: bevy::ui::PositionType::Absolute,
                            ..Default::default()
                        },
                        bevy::prelude::ImageNode {
                            image_mode: bevy::ui::widget::NodeImageMode::Stretch,
                            ..Default::default()
                        },
                        crate::ui::video_stream::VideoDecoderHandle::default(),
                    ))
                    .id();

                let pane = crate::ui::video_stream::VideoStreamPane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::VideoStream(pane)), parent_id, false)
            }
            // Inspector and Hierarchy are now fixed sidebars, not tile panels.
            // Set the corresponding sidebar visibility flags so they appear.
            Panel::Inspector => {
                ui_state.right_sidebar_visible = true;
                None
            }
            Panel::Hierarchy => {
                ui_state.left_sidebar_visible = true;
                None
            }
            Panel::SchematicTree(name) => {
                let entity = self.commands.spawn(super::TreeWidgetState::default()).id();
                let label = name.clone().unwrap_or_else(|| "Tree".to_string());
                let pane = TreePane {
                    entity,
                    name: label,
                };
                tile_state.insert_tile(Tile::Pane(Pane::SchematicTree(pane)), parent_id, false)
            }
            Panel::DataOverview(name) => {
                let mut pane = DataOverviewPane::default();
                if let Some(name) = name.clone() {
                    pane.name = name;
                }
                let pane = Pane::DataOverview(pane);
                tile_state.insert_tile(Tile::Pane(pane), parent_id, false)
            }
            Panel::QueryPlot(plot) => {
                let graph_bundle = GraphBundle::new(
                    &mut self.render_layer_alloc,
                    BTreeMap::default(),
                    plot.name.clone(),
                );
                let auto_color = plot.color.into_color32() == colors::get_scheme().highlight;
                let entity = self
                    .commands
                    .spawn(QueryPlotData {
                        data: plot.clone(),
                        auto_color,
                        last_refresh: None,
                        ..Default::default()
                    })
                    .insert(graph_bundle)
                    .id();
                let pane = Pane::QueryPlot(super::query_plot::QueryPlotPane {
                    entity,
                    rect: None,
                    scrub_icon: None,
                });
                tile_state.insert_tile(Tile::Pane(pane), parent_id, false)
            }
            Panel::Dashboard(dashboard) => {
                let label = dashboard
                    .root
                    .name
                    .clone()
                    .unwrap_or_else(|| "Dashboard".to_string());
                let Ok(dashboard_entity) = spawn_dashboard(
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
                        entity: dashboard_entity,
                        name: label,
                    })),
                    parent_id,
                    false,
                )
            }
        }
    }

    /// Create a default Data Overview panel when no schematic is found.
    /// This provides immediate visibility into the database contents.
    pub fn load_default_data_overview(&mut self) {
        if self.eql.0.component_parts.is_empty() {
            return;
        }

        let primary_window = *self.primary_window;
        let mut window_state = self
            .window_states
            .get_mut(primary_window)
            .expect("no primary window")
            .2;

        // Only add if the tile tree is empty (no non-sidebar content)
        if !window_state.tile_state.is_empty() {
            return;
        }

        let target_id = {
            let tree = &window_state.tile_state.tree;
            tree.root()
                .and_then(|root_id| match tree.tiles.get(root_id) {
                    Some(Tile::Container(Container::Linear(linear))) => {
                        let center_idx = linear.children.len() / 2;
                        linear.children.get(center_idx).copied()
                    }
                    _ => Some(root_id),
                })
        };

        let mut central_tabs_id = None;
        if let Some(target_id) = target_id {
            match window_state.tile_state.tree.tiles.get(target_id) {
                Some(Tile::Container(Container::Tabs(_))) => central_tabs_id = Some(target_id),
                Some(Tile::Container(_)) => {
                    let tabs_container = Tile::Container(Container::new_tabs(vec![]));
                    central_tabs_id =
                        window_state
                            .tile_state
                            .insert_tile(tabs_container, Some(target_id), false);
                }
                _ => {}
            }
        }

        if central_tabs_id.is_none() {
            let tabs_container = Tile::Container(Container::new_tabs(vec![]));
            central_tabs_id = window_state
                .tile_state
                .insert_tile(tabs_container, None, false);
        }

        // The central tabs container is effectively invisible; nest a new Tabs container
        // so the tab bar (+) is visible, then add Data Overview inside it.
        if let Some(central_tabs_id) = central_tabs_id {
            let tabs_container = Tile::Container(Container::new_tabs(vec![]));
            if let Some(tabs_id) =
                window_state
                    .tile_state
                    .insert_tile(tabs_container, Some(central_tabs_id), false)
            {
                let pane = Pane::DataOverview(DataOverviewPane::default());
                if let Some(tile_id) =
                    window_state
                        .tile_state
                        .insert_tile(Tile::Pane(pane), Some(tabs_id), true)
                {
                    window_state
                        .tile_state
                        .tree
                        .make_active(|id, _| id == tile_id);
                }
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

#[derive(Debug)]
pub struct SchematicReloadEvent {
    pub path: PathBuf,
    pub schematic: Schematic,
}

#[derive(Default, Resource)]
pub struct SchematicLiveReloadRx {
    receiver: Option<flume::Receiver<SchematicReloadEvent>>,
    ignored_paths: HashMap<PathBuf, usize>,
    #[cfg(target_os = "linux")]
    load_guard_until: Option<Instant>,
    #[cfg(target_os = "linux")]
    last_loaded: HashMap<PathBuf, SystemTime>,
}

impl SchematicLiveReloadRx {
    pub fn set_receiver(&mut self, receiver: flume::Receiver<SchematicReloadEvent>) {
        self.receiver = Some(receiver);
    }

    pub fn clear(&mut self) {
        self.receiver = None;
        self.ignored_paths.clear();
    }

    pub fn ignore_path(&mut self, path: PathBuf) {
        let canonical = canonicalize_for_ignore(path);
        *self.ignored_paths.entry(canonical).or_insert(0) += 1;
    }

    #[cfg(not(target_os = "linux"))]
    pub fn guard_for(&mut self, _duration: Duration) {}

    #[cfg(target_os = "linux")]
    pub fn guard_for(&mut self, duration: Duration) {
        let candidate = Instant::now() + duration;
        self.load_guard_until = Some(
            self.load_guard_until
                .map(|current| current.max(candidate))
                .unwrap_or(candidate),
        );
    }

    fn should_ignore(&mut self, path: &Path) -> bool {
        let canonical = canonicalize_for_ignore(path.to_path_buf());
        if let Some(count) = self.ignored_paths.get_mut(&canonical) {
            if *count > 1 {
                *count -= 1;
            } else {
                self.ignored_paths.remove(&canonical);
            }
            return true;
        }
        false
    }

    #[cfg(target_os = "linux")]
    fn guard_active(&mut self) -> bool {
        let now = Instant::now();
        let mut saw_event = false;
        if let Some(receiver) = &mut self.receiver {
            while receiver.try_recv().is_ok() {
                saw_event = true;
            }
        }
        if saw_event {
            let deadline = now + NOTIFY_SILENCE;
            self.load_guard_until = Some(
                self.load_guard_until
                    .map(|current| current.max(deadline))
                    .unwrap_or(deadline),
            );
        }
        if let Some(until) = self.load_guard_until
            && now < until
        {
            return true;
        }
        self.load_guard_until = None;
        false
    }

    #[cfg(target_os = "linux")]
    fn has_changed(&mut self, path: &Path) -> bool {
        let canonical = canonicalize_for_ignore(path.to_path_buf());
        let metadata = match std::fs::metadata(&canonical) {
            Ok(meta) => meta,
            Err(_) => {
                self.last_loaded.remove(&canonical);
                return true;
            }
        };
        let modified = match metadata.modified() {
            Ok(ts) => ts,
            Err(_) => {
                self.last_loaded.remove(&canonical);
                return true;
            }
        };
        let last = self.last_loaded.get(&canonical).copied();
        let changed = last.map(|ts| modified > ts).unwrap_or(true);
        if changed {
            self.last_loaded.insert(canonical, modified);
        }
        changed
    }

    #[cfg(target_os = "linux")]
    pub fn record_loaded(&mut self, path: &Path) {
        let canonical = canonicalize_for_ignore(path.to_path_buf());
        if let Ok(meta) = std::fs::metadata(&canonical)
            && let Ok(modified) = meta.modified()
        {
            self.last_loaded.insert(canonical, modified);
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn has_changed(&mut self, _path: &Path) -> bool {
        true
    }

    #[cfg(not(target_os = "linux"))]
    pub fn record_loaded(&mut self, _path: &Path) {}
}

fn canonicalize_for_ignore(path: PathBuf) -> PathBuf {
    std::fs::canonicalize(&path).unwrap_or(path)
}

pub fn schematic_live_reload(
    mut rx: ResMut<SchematicLiveReloadRx>,
    mut params: LoadSchematicParams,
) {
    #[cfg(target_os = "linux")]
    if rx.guard_active() {
        return;
    }
    let Some(receiver) = &mut rx.receiver else {
        return;
    };
    let Ok(event) = receiver.try_recv() else {
        return;
    };
    if rx.should_ignore(&event.path) {
        return;
    }
    if !rx.has_changed(&event.path) {
        return;
    }
    info!(path = ?event.path, "refreshing schematic");
    let base_dir = event.path.parent().map(|p| p.to_path_buf());
    params.load_schematic(&event.schematic, base_dir.as_deref());
    rx.record_loaded(&event.path);
}
