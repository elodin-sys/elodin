use super::*;
use crate::ui::{query_plot, video_stream};

#[derive(Clone)]
pub enum TreeAction {
    AddViewport(Option<TileId>, bool),
    AddGraph(Option<TileId>, Box<Option<GraphBundle>>, bool),
    AddMonitor(Option<TileId>, String, bool),
    AddQueryTable(Option<TileId>, bool),
    AddQueryPlot(Option<TileId>, bool),
    AddActionTile(Option<TileId>, String, String, bool),
    AddVideoStream(Option<TileId>, [u8; 2], String, bool),
    AddDashboard(Option<TileId>, Box<Dashboard>, String, bool),
    AddHierarchy(Option<TileId>, bool),
    AddSchematicTree(Option<TileId>, bool),
    AddSidebars,
    DeleteTab(TileId),
    SelectTile(TileId),
    RenameContainer(TileId, String),
}

// Despawn entities and free layers for a tile that is about to be removed.
pub(super) fn cleanup_tile(
    ui_state: &mut TileState,
    commands: &mut Commands,
    render_layer_alloc: &mut RenderLayerAlloc,
    tile_id: TileId,
) {
    let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
        return;
    };
    match tile {
        egui_tiles::Tile::Pane(Pane::Viewport(viewport)) => {
            if let Some(layer) = viewport.viewport_layer {
                render_layer_alloc.free(layer);
            }
            if let Some(layer) = viewport.grid_layer {
                render_layer_alloc.free(layer);
            }
            if let Some(camera) = viewport.camera {
                commands.entity(camera).despawn();
            }
            if let Some(nav_gizmo_camera) = viewport.nav_gizmo_camera {
                commands.entity(nav_gizmo_camera).despawn();
            }
            if let Some(nav_gizmo) = viewport.nav_gizmo {
                commands.entity(nav_gizmo).despawn();
            }
        }
        egui_tiles::Tile::Pane(Pane::Graph(graph)) => {
            commands.entity(graph.id).despawn();
            ui_state.graphs.remove(&tile_id);
        }
        egui_tiles::Tile::Pane(Pane::ActionTile(action)) => {
            commands.entity(action.entity).despawn();
        }
        egui_tiles::Tile::Pane(Pane::VideoStream(pane)) => {
            commands.entity(pane.entity).despawn();
        }
        egui_tiles::Tile::Pane(Pane::QueryPlot(pane)) => {
            commands.entity(pane.entity).despawn();
        }
        egui_tiles::Tile::Pane(Pane::QueryTable(pane)) => {
            commands.entity(pane.entity).despawn();
        }
        egui_tiles::Tile::Pane(Pane::SchematicTree(pane)) => {
            commands.entity(pane.entity).despawn();
        }
        egui_tiles::Tile::Pane(
            Pane::Monitor(_) | Pane::Inspector | Pane::Hierarchy | Pane::Dashboard(_),
        ) => {}
        egui_tiles::Tile::Container(container) => {
            let children: Vec<TileId> = container.children().copied().collect();
            for child in children {
                cleanup_tile(ui_state, commands, render_layer_alloc, child);
            }
        }
    }
}

pub(super) struct ActionContext<'a, 'w, 's> {
    pub ui_state: &'a mut TileState,
    pub commands: &'a mut Commands<'w, 's>,
    pub selected_object: &'a mut ResMut<'w, SelectedObject>,
    pub asset_server: &'a Res<'w, AssetServer>,
    pub meshes: &'a mut ResMut<'w, Assets<Mesh>>,
    pub materials: &'a mut ResMut<'w, Assets<StandardMaterial>>,
    pub render_layer_alloc: &'a mut ResMut<'w, RenderLayerAlloc>,
    pub eql_ctx: &'a Res<'w, EqlContext>,
    pub node_updater_params: &'a NodeUpdaterParams<'w, 's>,
    pub graphs: &'a mut Query<'w, 's, &'static mut GraphState>,
    pub query_plots: &'a mut Query<'w, 's, &'static mut QueryPlotData>,
    pub action_tiles: &'a mut Query<'w, 's, &'static mut ActionTile>,
    pub window: Option<Entity>,
    pub read_only: bool,
}

impl<'a, 'w, 's> ActionContext<'a, 'w, 's> {
    // Dispatch a single tree action with read-only guard.
    pub fn handle(&mut self, diff: TreeAction) {
        if self.read_only && !matches!(diff, TreeAction::SelectTile(_)) {
            return;
        }
        match diff {
            TreeAction::DeleteTab(tile_id) => self.delete_tab(tile_id),
            TreeAction::AddViewport(parent_tile_id, new_tab) => {
                self.add_viewport(parent_tile_id, new_tab)
            }
            TreeAction::AddGraph(parent_tile_id, bundle, new_tab) => {
                self.add_graph(parent_tile_id, bundle, new_tab)
            }
            TreeAction::AddMonitor(parent_tile_id, eql, new_tab) => {
                self.add_monitor(parent_tile_id, eql, new_tab)
            }
            TreeAction::AddVideoStream(parent_tile_id, msg_id, label, new_tab) => {
                self.add_video_stream(parent_tile_id, msg_id, label, new_tab)
            }
            TreeAction::AddDashboard(parent_tile_id, dashboard, label, new_tab) => {
                self.add_dashboard(parent_tile_id, dashboard, label, new_tab)
            }
            TreeAction::AddHierarchy(parent_tile_id, new_tab) => {
                self.add_hierarchy(parent_tile_id, new_tab)
            }
            TreeAction::AddSchematicTree(parent_tile_id, new_tab) => {
                self.add_schematic_tree(parent_tile_id, new_tab)
            }
            TreeAction::SelectTile(tile_id) => self.select_tile(tile_id),
            TreeAction::AddActionTile(parent_tile_id, button_name, lua_code, new_tab) => {
                self.add_action_tile(parent_tile_id, button_name, lua_code, new_tab)
            }
            TreeAction::AddQueryTable(parent_tile_id, new_tab) => {
                self.add_query_table(parent_tile_id, new_tab)
            }
            TreeAction::AddQueryPlot(parent_tile_id, new_tab) => {
                self.add_query_plot(parent_tile_id, new_tab)
            }
            TreeAction::AddSidebars => self.add_sidebars(),
            TreeAction::RenameContainer(tile_id, title) => self.rename_container(tile_id, title),
        }
    }

    fn delete_tab(&mut self, tile_id: TileId) {
        cleanup_tile(
            self.ui_state,
            self.commands,
            self.render_layer_alloc,
            tile_id,
        );
        self.ui_state.tree.remove_recursively(tile_id);
        self.ui_state.recompute_has_non_sidebar();
    }

    // Spawn a default viewport and insert it.
    fn add_viewport(&mut self, parent_tile_id: Option<TileId>, new_tab: bool) {
        let viewport = Viewport::default();
        let label = viewport_label(&viewport);
        let viewport_pane = ViewportPane::spawn(
            self.commands,
            self.asset_server,
            self.meshes,
            self.materials,
            self.render_layer_alloc,
            &self.eql_ctx.0,
            &viewport,
            label,
        );

        if let Some(tile_id) = self.ui_state.insert_pane_in_tabs(
            Pane::Viewport(viewport_pane),
            parent_tile_id,
            true,
            self.window,
            new_tab,
        ) {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Spawn a graph (or reuse a provided bundle) and insert it.
    fn add_graph(
        &mut self,
        parent_tile_id: Option<TileId>,
        graph_bundle: Box<Option<GraphBundle>>,
        new_tab: bool,
    ) {
        let graph_label = graph_label(&Graph::default());
        let graph_bundle = if let Some(graph_bundle) = *graph_bundle {
            graph_bundle
        } else {
            GraphBundle::new(
                self.render_layer_alloc,
                BTreeMap::default(),
                graph_label.clone(),
            )
        };
        let graph_id = self.commands.spawn(graph_bundle).id();
        let graph = GraphPane::new(graph_id, graph_label.clone());
        let pane = Pane::Graph(graph);

        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            **self.selected_object = SelectedObject::Graph { graph_id };
            self.ui_state.tree.make_active(|id, _| id == tile_id);
            self.ui_state.graphs.insert(tile_id, graph_id);
        }
    }

    // Insert a monitor pane with provided expression.
    fn add_monitor(&mut self, parent_tile_id: Option<TileId>, eql: String, new_tab: bool) {
        let monitor = MonitorPane::new("Monitor".to_string(), eql);
        let pane = Pane::Monitor(monitor);
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Spawn a video stream pane bound to a new entity.
    fn add_video_stream(
        &mut self,
        parent_tile_id: Option<TileId>,
        msg_id: [u8; 2],
        label: String,
        new_tab: bool,
    ) {
        let entity = self
            .commands
            .spawn((
                video_stream::VideoStream {
                    msg_id,
                    ..Default::default()
                },
                bevy::ui::Node {
                    position_type: PositionType::Absolute,
                    ..Default::default()
                },
                bevy::prelude::ImageNode {
                    image_mode: NodeImageMode::Stretch,
                    ..Default::default()
                },
                VideoDecoderHandle::default(),
            ))
            .id();
        let pane = Pane::VideoStream(video_stream::VideoStreamPane {
            entity,
            label: label.clone(),
        });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Spawn a dashboard entity and insert it.
    fn add_dashboard(
        &mut self,
        parent_tile_id: Option<TileId>,
        dashboard: Box<impeller2_wkt::Dashboard>,
        label: String,
        new_tab: bool,
    ) {
        let entity = match spawn_dashboard(
            &dashboard,
            &self.eql_ctx.0,
            self.commands,
            self.node_updater_params,
        ) {
            Ok(entity) => entity,
            Err(_) => self.commands.spawn(bevy::ui::Node::default()).id(),
        };
        let pane = Pane::Dashboard(DashboardPane { entity, label });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Insert a hierarchy pane into tabs.
    fn add_hierarchy(&mut self, parent_tile_id: Option<TileId>, new_tab: bool) {
        if let Some(tile_id) = self.ui_state.insert_pane_in_tabs(
            Pane::Hierarchy,
            parent_tile_id,
            true,
            self.window,
            new_tab,
        ) {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Spawn a schematic tree widget and insert it.
    fn add_schematic_tree(&mut self, parent_tile_id: Option<TileId>, new_tab: bool) {
        let entity = self
            .commands
            .spawn(super::schematic::TreeWidgetState::default())
            .id();
        let pane = Pane::SchematicTree(TreePane { entity });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Focus selection and unmask relevant sidebars.
    fn select_tile(&mut self, tile_id: TileId) {
        self.ui_state.tree.make_active(|id, _| id == tile_id);

        if let Some(egui_tiles::Tile::Pane(pane)) = self.ui_state.tree.tiles.get(tile_id) {
            match pane {
                Pane::Graph(graph) => {
                    **self.selected_object = SelectedObject::Graph { graph_id: graph.id };
                }
                Pane::QueryPlot(plot) => {
                    **self.selected_object = SelectedObject::Graph {
                        graph_id: plot.entity,
                    };
                }
                Pane::Viewport(viewport) => {
                    if let Some(camera) = viewport.camera {
                        **self.selected_object = SelectedObject::Viewport { camera };
                    }
                }
                Pane::Hierarchy => {
                    unmask_sidebar_by_kind(self.ui_state, SidebarKind::Hierarchy);
                }
                Pane::Inspector => {
                    unmask_sidebar_by_kind(self.ui_state, SidebarKind::Inspector);
                }
                _ => {}
            }
            // Always ensure inspector is visible after a selection.
            unmask_sidebar_by_kind(self.ui_state, SidebarKind::Inspector);
        }
    }

    // Spawn an action tile entity and insert it.
    fn add_action_tile(
        &mut self,
        parent_tile_id: Option<TileId>,
        button_name: String,
        lua_code: String,
        new_tab: bool,
    ) {
        let entity = self
            .commands
            .spawn(ActionTile {
                button_name,
                lua: lua_code,
                status: Default::default(),
            })
            .id();
        let pane = Pane::ActionTile(ActionTilePane {
            entity,
            label: "Action".to_string(),
        });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Create a query table pane with a new entity.
    fn add_query_table(&mut self, parent_tile_id: Option<TileId>, new_tab: bool) {
        let entity = self.commands.spawn(QueryTableData::default()).id();
        let pane = Pane::QueryTable(QueryTablePane { entity });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Create a query plot graph bundle and insert it.
    fn add_query_plot(&mut self, parent_tile_id: Option<TileId>, new_tab: bool) {
        let graph_bundle = GraphBundle::new(
            self.render_layer_alloc,
            BTreeMap::default(),
            "Query Plot".to_string(),
        );
        let entity = self
            .commands
            .spawn(QueryPlotData::default())
            .insert(graph_bundle)
            .id();
        let pane = Pane::QueryPlot(query_plot::QueryPlotPane {
            entity,
            rect: None,
            scrub_icon: None,
        });
        if let Some(tile_id) =
            self.ui_state
                .insert_pane_in_tabs(pane, parent_tile_id, true, self.window, new_tab)
        {
            **self.selected_object = SelectedObject::Graph { graph_id: entity };
            self.ui_state.tree.make_active(|id, _| id == tile_id);
        }
    }

    // Ensure the layout always contains hierarchy/inspector sidebars.
    fn add_sidebars(&mut self) {
        let sidebar_ids: std::collections::HashSet<TileId> = self
            .ui_state
            .tree
            .tiles
            .iter()
            .filter_map(|(id, tile)| {
                matches!(tile, Tile::Pane(Pane::Hierarchy | Pane::Inspector)).then_some(*id)
            })
            .collect();

        if !sidebar_ids.is_empty() {
            let container_ids: Vec<TileId> = self
                .ui_state
                .tree
                .tiles
                .iter()
                .filter_map(|(id, tile)| matches!(tile, Tile::Container(_)).then_some(*id))
                .collect();

            for cid in container_ids {
                if let Some(Tile::Container(container)) = self.ui_state.tree.tiles.get_mut(cid) {
                    match container {
                        Container::Tabs(tabs) => {
                            tabs.children.retain(|child| !sidebar_ids.contains(child));
                            if let Some(active) = tabs.active {
                                if !tabs.children.contains(&active) {
                                    tabs.active = tabs.children.first().copied();
                                }
                            }
                        }
                        Container::Linear(linear) => {
                            linear.children.retain(|child| !sidebar_ids.contains(child));
                        }
                        Container::Grid(_) => {}
                    }
                }
            }
        }

        let mut main_content = self.ui_state.tree.root();
        if let Some(root) = self.ui_state.tree.root() {
            main_content = match self.ui_state.tree.tiles.get(root) {
                Some(Tile::Container(Container::Tabs(tabs))) => {
                    if tabs.children.is_empty() {
                        None
                    } else {
                        Some(root)
                    }
                }
                Some(Tile::Container(Container::Linear(linear))) => match linear.children.len() {
                    0 => None,
                    1 => Some(linear.children[0]),
                    _ => Some(root),
                },
                Some(Tile::Container(Container::Grid(grid))) => {
                    let children: Vec<_> = grid.children().copied().collect();
                    match children.len() {
                        0 => None,
                        1 => Some(children[0]),
                        _ => Some(root),
                    }
                }
                Some(Tile::Pane(_)) => Some(root),
                _ => None,
            };
        }

        let hierarchy = self
            .ui_state
            .tree
            .tiles
            .insert_new(Tile::Pane(Pane::Hierarchy));
        let inspector = self
            .ui_state
            .tree
            .tiles
            .insert_new(Tile::Pane(Pane::Inspector));

        let mut main_content = main_content.unwrap_or_else(|| {
            let tabs = egui_tiles::Tabs::new(Vec::new());
            self.ui_state
                .tree
                .tiles
                .insert_new(Tile::Container(Container::Tabs(tabs)))
        });

        let wrap_into_tabs = !matches!(
            self.ui_state.tree.tiles.get(main_content),
            Some(Tile::Container(Container::Tabs(_)))
        );
        if wrap_into_tabs {
            let mut tabs = egui_tiles::Tabs::new(vec![main_content]);
            tabs.set_active(main_content);
            main_content = self
                .ui_state
                .tree
                .tiles
                .insert_new(Tile::Container(Container::Tabs(tabs)));
            if let Some(wrapped_id) = self.ui_state.tree.root() {
                if let Some(Tile::Container(Container::Linear(linear))) =
                    self.ui_state.tree.tiles.get(wrapped_id)
                {
                    if linear.children.len() == 3 {
                        self.ui_state.clear_container_title(main_content);
                    }
                }
            }
        }

        let mut linear = egui_tiles::Linear::new(
            egui_tiles::LinearDir::Horizontal,
            vec![hierarchy, inspector],
        );
        linear.children.insert(1, main_content);
        let hier_default = 0.2;
        let insp_default = 0.2;
        let hier_share = if self.ui_state.hierarchy_masked {
            0.01
        } else {
            hier_default
        };
        let insp_share = if self.ui_state.inspector_masked {
            0.01
        } else {
            insp_default
        };
        let mut center_share = 1.0 - (hier_share + insp_share);
        if center_share <= 0.0 {
            center_share = 0.1;
        }
        linear.shares.set_share(hierarchy, hier_share);
        linear.shares.set_share(main_content, center_share);
        linear.shares.set_share(inspector, insp_share);
        self.ui_state
            .last_hierarchy_share
            .get_or_insert(hier_default);
        self.ui_state
            .last_inspector_share
            .get_or_insert(insp_default);

        let root = self
            .ui_state
            .tree
            .tiles
            .insert_new(Tile::Container(Container::Linear(linear)));
        self.ui_state.tree.root = Some(root);
        self.ui_state.tree.make_active(|id, _| id == hierarchy);
        self.ui_state.recompute_has_non_sidebar();
    }

    // Update titles/labels for panes and containers.
    fn rename_container(&mut self, tile_id: TileId, title: String) {
        match self.ui_state.tree.tiles.get_mut(tile_id) {
            Some(Tile::Pane(pane)) => match pane {
                Pane::Viewport(view) => {
                    view.label = title.clone();
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::Graph(graph) => {
                    graph.label = title.clone();
                    if let Some(graph_entity) = self.ui_state.graphs.get(&tile_id) {
                        if let Ok(mut state) = self.graphs.get_mut(*graph_entity) {
                            state.label = title.clone();
                        }
                    }
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::Monitor(monitor) => {
                    monitor.label = title.clone();
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::QueryPlot(plot) => {
                    if let Ok(mut data) = self.query_plots.get_mut(plot.entity) {
                        data.data.label = title.clone();
                    }
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::ActionTile(action) => {
                    action.label = title.clone();
                    if let Ok(mut action_tile) = self.action_tiles.get_mut(action.entity) {
                        action_tile.button_name = title.clone();
                    }
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::VideoStream(stream) => {
                    stream.label = title.clone();
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::Dashboard(dashboard) => {
                    dashboard.label = title.clone();
                    self.ui_state.clear_container_title(tile_id);
                }
                Pane::QueryTable(_)
                | Pane::Inspector
                | Pane::Hierarchy
                | Pane::SchematicTree(_) => {
                    self.ui_state.set_container_title(tile_id, title);
                }
            },
            Some(Tile::Container(_)) => {
                self.ui_state.set_container_title(tile_id, title);
            }
            None => {}
        }
    }
}

pub(super) fn unmask_sidebar_on_select(
    ui_state: &mut TileState,
    sidebar_tile: TileId,
    kind: SidebarKind,
) {
    fn build_parent_map(tiles: &Tiles<Pane>) -> HashMap<TileId, TileId> {
        let mut parents = HashMap::new();
        for (id, tile) in tiles.iter() {
            if let Tile::Container(container) = tile {
                for child in container.children() {
                    parents.insert(*child, *id);
                }
            }
        }
        parents
    }

    fn rebalance_linear_shares(
        linear: &mut egui_tiles::Linear,
        target_child: TileId,
        target_frac: f32,
        min_share: f32,
    ) {
        let share_sum: f32 = linear
            .shares
            .iter()
            .map(|(_, s)| s)
            .sum::<f32>()
            .max(min_share);
        let old = linear.shares[target_child];
        let others_sum = (share_sum - old).max(0.0);
        let mut target = (share_sum * target_frac).max(min_share);
        let min_others = min_share * (linear.children.len().saturating_sub(1) as f32);
        if target > share_sum - min_others {
            target = (share_sum - min_others).max(min_share);
        }
        let factor = if others_sum > 0.0 {
            (share_sum - target) / others_sum
        } else {
            0.0
        };
        let children = linear.children.clone();
        for child in children {
            if child == target_child {
                linear.shares.set_share(child, target.max(min_share));
            } else {
                let new = (linear.shares[child] * factor).max(min_share);
                linear.shares.set_share(child, new);
            }
        }
    }

    let target_frac = 0.2;
    let min_share = 0.01;
    let parents = build_parent_map(&ui_state.tree.tiles);
    let mut current = Some(sidebar_tile);
    while let Some(child) = current {
        let Some(parent) = parents.get(&child).copied() else {
            break;
        };
        if let Some(Tile::Container(Container::Linear(linear))) =
            ui_state.tree.tiles.get_mut(parent)
        {
            rebalance_linear_shares(linear, child, target_frac, min_share);
        }
        current = Some(parent);
    }

    match kind {
        SidebarKind::Hierarchy => {
            ui_state.hierarchy_masked = false;
            ui_state.last_hierarchy_share = Some(target_frac);
        }
        SidebarKind::Inspector => {
            ui_state.inspector_masked = false;
            ui_state.last_inspector_share = Some(target_frac);
        }
    }
}

pub(super) fn unmask_sidebar_by_kind(ui_state: &mut TileState, kind: SidebarKind) {
    let target = ui_state
        .tree
        .tiles
        .iter()
        .find_map(|(id, tile)| match (kind, tile) {
            (SidebarKind::Hierarchy, Tile::Pane(Pane::Hierarchy)) => Some(*id),
            (SidebarKind::Inspector, Tile::Pane(Pane::Inspector)) => Some(*id),
            _ => None,
        });

    if let Some(tile_id) = target {
        if ui_state.tree.tiles.is_visible(tile_id) {
            unmask_sidebar_on_select(ui_state, tile_id, kind);
        }
    }
}
