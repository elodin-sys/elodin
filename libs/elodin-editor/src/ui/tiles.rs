use std::collections::{BTreeMap, HashMap};

use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_egui::{
    egui::{self, vec2, Color32, Frame, Margin, RichText, Rounding, Stroke, Ui, Visuals},
    EguiContexts,
};
use big_space::propagation::NoPropagateRot;
use big_space::GridCell;
use conduit::{
    bevy::{EntityMap, Tick, TimeStep},
    query::MetadataStore,
    well_known::{Panel, Viewport},
    ComponentId, EntityId, GraphId,
};
use egui_tiles::{Container, Tile, TileId, Tiles};

use super::{
    colors, images,
    utils::MarginSides,
    widgets::{
        button::EImageButton,
        eplot::{self, EPlot},
        eplot_gpu::Line,
        timeline::tagged_range::TaggedRanges,
        RootWidgetSystem,
    },
    GraphState, GraphsState, HdrEnabled, SelectedObject, ViewportRect,
};
use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc, spawn_main_camera, ui::GraphStateEntity,
    CollectedGraphData,
};

struct TileIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
    pub scrub: egui::TextureId,
}

#[derive(Resource)]
pub struct TileState {
    tree: egui_tiles::Tree<Pane>,
    tab_diffs: Vec<TabDiff>,
    graphs: HashMap<TileId, GraphId>,
}

impl TileState {
    fn insert_pane(&mut self, pane: Pane) -> Option<TileId> {
        Some(self.tree.tiles.insert_pane(pane))
    }
    fn insert_pane_with_parent(
        &mut self,
        pane: Pane,
        parent: TileId,
        active: bool,
    ) -> Option<TileId> {
        let tile_id = self.insert_pane(pane)?;
        self.set_parent(tile_id, Some(parent), active)
    }

    pub fn set_parent(
        &mut self,
        child: TileId,
        parent: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        let parent = parent.or_else(|| self.tree.root())?;
        let parent = self.tree.tiles.get_mut(parent)?;
        let Tile::Container(container) = parent else {
            return None;
        };
        container.add_child(child);

        if active {
            if let Container::Tabs(tabs) = container {
                tabs.set_active(child);
            }
        }
        Some(child)
    }

    pub fn create_graph_tile(&mut self, graph_state: GraphState) {
        if let Some(parent) = self.tree.root {
            self.tab_diffs
                .push(TabDiff::AddGraph(parent, Some(graph_state)));
        }
    }
}

enum Pane {
    Viewport(ViewportPane),
    Graph(GraphPane),
    Welcome,
}

impl Pane {
    fn title(&self) -> &str {
        match self {
            Pane::Graph(pane) => &pane.label,
            Pane::Viewport(_) => "VIEWPORT",
            Pane::Welcome => "WELCOME",
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        ui: &mut Ui,
        time_step: std::time::Duration,
        current_tick: u64,
        collected_graph_data: &CollectedGraphData,
        graphs_state: &mut GraphsState,
        tagged_ranges: &TaggedRanges,
        lines: &mut Assets<Line>,
        lines_query: &Query<&Handle<Line>>,
        commands: &mut Commands,
        icons: &TileIcons,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                pane.rect = Some(eplot::get_inner_rect(content_rect));
                // ui.painter()
                //     .rect_filled(content_rect, 0.0, colors::PRIMARY_SMOKE);

                let Some(graph_state) = graphs_state.get_mut(&pane.id) else {
                    return egui_tiles::UiResponse::None;
                };

                let tagged_range = graph_state
                    .range_id
                    .as_ref()
                    .and_then(|rid| tagged_ranges.0.get(rid));

                EPlot::new()
                    .padding(egui::Margin::same(0.0).left(20.0).bottom(20.0))
                    .margin(egui::Margin::same(60.0).left(85.0).top(40.0))
                    .steps(7, 4)
                    .time_step(time_step)
                    .current_tick(current_tick)
                    .calculate_lines(
                        ui,
                        collected_graph_data,
                        graph_state,
                        tagged_range,
                        lines,
                        commands,
                    )
                    .render(
                        ui,
                        lines,
                        lines_query,
                        collected_graph_data,
                        graph_state,
                        &icons.scrub,
                    );

                egui_tiles::UiResponse::None
            }
            Pane::Viewport(pane) => {
                pane.rect = Some(content_rect.shrink(1.0));
                egui_tiles::UiResponse::None
            }
            Pane::Welcome => {
                ui.painter()
                    .rect_filled(content_rect, 0.0, colors::PRIMARY_SMOKE);
                Frame::none()
                    .inner_margin(Margin::same(8.0).top(content_rect.height() * 0.4))
                    .show(ui, |ui| {
                        ui.vertical_centered_justified(|ui| {
                            ui.heading("Welcome to the Elodin Editor!");
                            ui.add_space(ui.spacing().interact_size.y);
                            ui.label("Get started by connecting to a simulator, and then adding a viewport or graph");
                        });
                    });

                egui_tiles::UiResponse::None
            }
        }
    }
}

#[derive(Default)]
struct ViewportPane {
    pub camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
}

impl ViewportPane {
    fn spawn(
        commands: &mut Commands,
        asset_server: &Res<AssetServer>,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
        viewport: &Viewport,
    ) -> Self {
        let camera = Some(spawn_main_camera(
            commands,
            asset_server,
            meshes,
            materials,
            render_layer_alloc,
            viewport,
        ));
        Self { camera, rect: None }
    }
}

struct GraphPane {
    pub id: GraphId,
    pub label: String,
    pub rect: Option<egui::Rect>,
}

impl GraphPane {
    fn spawn(graph_id: GraphId) -> Self {
        Self {
            id: graph_id,
            label: format!("Graph {}", graph_id.0),
            rect: None,
        }
    }
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            tree: egui_tiles::Tree::new_tabs("tab_tree", vec![]),
            tab_diffs: vec![],
            graphs: HashMap::new(),
        }
    }
}

struct TreeBehavior<'a, 'w, 's> {
    icons: TileIcons,
    tab_diffs: Vec<TabDiff>,
    selected_object: &'a mut SelectedObject,
    graphs_state: &'a mut GraphsState,
    tagged_ranges: &'a TaggedRanges,
    collected_graph_data: &'a CollectedGraphData,
    lines: &'a mut Assets<Line>,
    line_query: &'a Query<'w, 's, &'static Handle<Line>>,
    time_step: std::time::Duration,
    current_tick: u64,
    commands: &'a mut Commands<'w, 's>,
}

#[derive(Clone)]
pub enum TabDiff {
    AddViewport(TileId),
    AddGraph(TileId, Option<GraphState>),
    Delete(TileId),
}

enum TabState {
    Active,
    Selected,
    Inactive,
}

impl<'a, 'w, 's> egui_tiles::Behavior<Pane> for TreeBehavior<'a, 'w, 's> {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        pane.ui(
            ui,
            self.time_step,
            self.current_tick,
            self.collected_graph_data,
            self.graphs_state,
            self.tagged_ranges,
            self.lines,
            self.line_query,
            self.commands,
            &self.icons,
        )
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn tab_ui(
        &mut self,
        tiles: &Tiles<Pane>,
        ui: &mut Ui,
        id: egui::Id,
        tile_id: egui_tiles::TileId,
        active: bool,
        is_being_dragged: bool,
    ) -> egui::Response {
        let is_selected = self.selected_object.is_tile_selected(tile_id);
        let tab_state = if is_selected {
            TabState::Selected
        } else if active {
            TabState::Active
        } else {
            TabState::Inactive
        };
        let text = self.tab_title_for_tile(tiles, tile_id);
        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 11.0;
        let galley = text.into_galley(ui, Some(false), f32::INFINITY, font_id);
        let x_margin = self.tab_title_spacing(ui.visuals());
        let (_, rect) = ui.allocate_space(vec2(
            galley.size().x + x_margin * 4.0,
            ui.available_height(),
        ));
        let text_rect = rect
            .shrink2(vec2(x_margin * 4.0, 0.0))
            .translate(vec2(-3.0 * x_margin, 0.0));
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());

        if ui.is_rect_visible(rect) && !is_being_dragged {
            let bg_color = match tab_state {
                TabState::Active => colors::BLACK_BLACK_600,
                TabState::Selected => colors::PRIMARY_CREAME,
                TabState::Inactive => colors::PRIMARY_SMOKE,
            };

            let text_color = match tab_state {
                TabState::Active => colors::PRIMARY_CREAME,
                TabState::Selected => colors::PRIMARY_SMOKE,
                TabState::Inactive => colors::with_opacity(colors::PRIMARY_CREAME, 0.6),
            };

            ui.painter().rect_filled(rect, 0.0, bg_color);
            ui.painter().galley(
                egui::Align2::LEFT_CENTER
                    .align_size_within_rect(galley.size(), text_rect)
                    .min,
                galley,
                text_color,
            );
            ui.add_space(-3.0 * x_margin);
            let close_response = ui.add(
                EImageButton::new(self.icons.close)
                    .scale(1.3, 1.3)
                    .image_tint(match tab_state {
                        TabState::Active | TabState::Inactive => colors::PRIMARY_CREAME,
                        TabState::Selected => colors::BLACK_BLACK_600,
                    })
                    .bg_color(colors::TRANSPARENT)
                    .hovered_bg_color(colors::TRANSPARENT),
            );
            if close_response.clicked() {
                self.tab_diffs.push(TabDiff::Delete(tile_id));
            }

            ui.painter().hline(
                rect.x_range(),
                rect.bottom(),
                egui::Stroke::new(1.0, colors::BLACK_BLACK_600),
            );

            ui.painter().vline(
                rect.left(),
                rect.y_range(),
                egui::Stroke::new(1.0, colors::BLACK_BLACK_600),
            );

            ui.painter().vline(
                rect.right(),
                rect.y_range(),
                egui::Stroke::new(1.0, colors::BLACK_BLACK_600),
            );
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn on_tab_button(
        &mut self,
        tiles: &Tiles<Pane>,
        tile_id: TileId,
        button_response: egui::Response,
    ) -> egui::Response {
        if button_response.middle_clicked() {
            self.tab_diffs.push(TabDiff::Delete(tile_id));
        } else if button_response.clicked() {
            let Some(tile) = tiles.get(tile_id) else {
                return button_response;
            };
            match tile {
                Tile::Pane(Pane::Graph(graph)) => {
                    *self.selected_object = SelectedObject::Graph {
                        tile_id,
                        label: graph.label.to_owned(),
                        graph_id: graph.id,
                    };
                }
                Tile::Pane(Pane::Welcome) => {
                    *self.selected_object = SelectedObject::None;
                }
                Tile::Pane(Pane::Viewport(viewport)) => {
                    let Some(camera) = viewport.camera else {
                        return button_response;
                    };
                    *self.selected_object = SelectedObject::Viewport { tile_id, camera };
                }
                _ => {}
            }
        }
        button_response
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        32.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        colors::PRIMARY_SMOKE
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, colors::PRIMARY_CREAME)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::with_opacity(colors::PRIMARY_CREAME, 0.6)
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = colors::PRIMARY_CREAME;
        frame.rounding = Rounding::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = self.tab_title_for_tile(tiles, tile_id);
            let text = text.text();
            ui.label(RichText::new(text).color(colors::PRIMARY_SMOKE).size(11.0));
        });
    }

    fn top_bar_right_ui(
        &mut self,
        _tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        let top_bar_rect = ui.available_rect_before_wrap();
        ui.painter().hline(
            top_bar_rect.x_range(),
            top_bar_rect.bottom(),
            egui::Stroke::new(1.0, colors::BLACK_BLACK_600),
        );

        ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        ui.style_mut().visuals.widgets.active.bg_stroke = Stroke::NONE;
        ui.add_space(5.0);
        let mut resp = ui.add(EImageButton::new(self.icons.add).scale(1.4, 1.4));
        if resp.clicked() {
            resp.long_touched = true;
            //resp.clicked = [false, true, false, false, false];
        }
        resp.context_menu(|ui| {
            ui.style_mut().spacing.item_spacing = vec2(16.0, 8.0);
            if ui.button("VIEWPORT").clicked() {
                self.tab_diffs.push(TabDiff::AddViewport(tile_id));
                ui.close_menu();
            }
            ui.separator();
            if ui.button("GRAPH").clicked() {
                self.tab_diffs.push(TabDiff::AddGraph(tile_id, None));
                ui.close_menu();
            }
        });
    }
}
pub fn setup_default_tiles(mut tile_state: ResMut<TileState>) {
    if tile_state.tree.active_tiles().is_empty() {
        tile_state.tree = egui_tiles::Tree::new_tabs("tab_tree", vec![Pane::Welcome]);
    }
}

#[derive(SystemParam)]
pub struct TileLayout<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    commands: Commands<'w, 's>,
    graphs_state: ResMut<'w, GraphsState>,
    tagged_ranges: Res<'w, TaggedRanges>,
    selected_object: ResMut<'w, SelectedObject>,
    ui_state: ResMut<'w, TileState>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    collected_graph_data: Res<'w, CollectedGraphData>,
    time_step: Res<'w, TimeStep>,
    tick: Res<'w, Tick>,
    lines: ResMut<'w, Assets<Line>>,
    line_query: Query<'w, 's, &'static Handle<Line>>,
}

impl RootWidgetSystem for TileLayout<'_, '_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let TileLayout {
            mut contexts,
            images,
            mut commands,
            mut graphs_state,
            tagged_ranges,
            mut selected_object,
            mut ui_state,
            asset_server,
            mut meshes,
            mut materials,
            mut render_layer_alloc,
            collected_graph_data,
            time_step,
            tick,
            mut lines,
            line_query,
        } = state_mut;

        let icons = TileIcons {
            add: contexts.add_image(images.icon_add.clone_weak()),
            close: contexts.add_image(images.icon_close.clone_weak()),
            scrub: contexts.add_image(images.icon_scrub.clone_weak()),
        };

        egui::CentralPanel::default()
            .frame(Frame {
                fill: colors::TRANSPARENT,
                ..Default::default()
            })
            .show(ctx, |ui| {
                let mut behavior = TreeBehavior {
                    icons,
                    tab_diffs: ui_state.tab_diffs.clone(),
                    selected_object: selected_object.as_mut(),
                    graphs_state: graphs_state.as_mut(),
                    tagged_ranges: tagged_ranges.as_ref(),
                    collected_graph_data: collected_graph_data.as_ref(),
                    time_step: time_step.0,
                    current_tick: tick.0,
                    lines: lines.as_mut(),
                    commands: &mut commands,
                    line_query: &line_query,
                };
                ui_state.tab_diffs = vec![];
                ui_state.tree.ui(&mut behavior, ui);
                for diff in behavior.tab_diffs.drain(..) {
                    match diff {
                        TabDiff::Delete(tile_id) => {
                            let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
                                continue;
                            };

                            if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
                                if let Some(camera) = viewport.camera {
                                    commands.entity(camera).despawn(); // TODO(sphw): garbage collect old nav-gizmos
                                }
                            };

                            if let egui_tiles::Tile::Pane(Pane::Graph(graph)) = tile {
                                if let Some(state) = graphs_state.get(&graph.id) {
                                    commands.entity(state.camera).despawn();
                                    graphs_state.remove_graph(&graph.id);
                                }
                            };

                            if ui_state.tree.tiles.len() > 1 {
                                ui_state.tree.tiles.remove(tile_id);

                                if let Some(graph_id) = ui_state.graphs.get(&tile_id) {
                                    graphs_state.remove_graph(graph_id);
                                    ui_state.graphs.remove(&tile_id);
                                }
                            }
                        }
                        TabDiff::AddViewport(parent) => {
                            let pane = Pane::Viewport(ViewportPane::spawn(
                                &mut commands,
                                &asset_server,
                                &mut meshes,
                                &mut materials,
                                &mut render_layer_alloc,
                                &Viewport::default(),
                            ));
                            ui_state.insert_pane_with_parent(pane, parent, true);
                        }
                        TabDiff::AddGraph(parent, graph_state) => {
                            let graph_state = if let Some(graph_state) = graph_state {
                                graph_state
                            } else {
                                GraphState::spawn(
                                    &mut commands,
                                    &mut render_layer_alloc,
                                    BTreeMap::default(),
                                )
                            };
                            let graph_id = graphs_state.push_graph_state(graph_state);

                            let graph = GraphPane::spawn(graph_id);
                            let graph_id = graph.id;
                            let graph_label = graph.label.clone();
                            let pane = Pane::Graph(graph);

                            if let Some(tile_id) =
                                ui_state.insert_pane_with_parent(pane, parent, true)
                            {
                                *selected_object = SelectedObject::Graph {
                                    tile_id,
                                    label: graph_label,
                                    graph_id,
                                };
                                ui_state.graphs.insert(tile_id, graph_id);
                            }
                        }
                    }
                }
                let tiles = ui_state.tree.tiles.iter();
                let active_tiles = ui_state.tree.active_tiles();
                for (tile_id, tile) in tiles {
                    let egui_tiles::Tile::Pane(pane) = tile else {
                        continue;
                    };
                    match pane {
                        Pane::Viewport(viewport) => {
                            let Some(cam) = viewport.camera else { continue };
                            if active_tiles.contains(tile_id) {
                                if let Some(mut cam) = commands.get_entity(cam) {
                                    cam.insert(ViewportRect(viewport.rect));
                                }
                            } else if let Some(mut cam) = commands.get_entity(cam) {
                                cam.insert(ViewportRect(None));
                            }
                        }
                        Pane::Graph(graph) => {
                            let graph_state = graphs_state.get(&graph.id).expect("graph missing");
                            if active_tiles.contains(tile_id) {
                                if let Some(mut cam) = commands.get_entity(graph_state.camera) {
                                    cam.insert(ViewportRect(graph.rect));
                                }
                            } else if let Some(mut cam) = commands.get_entity(graph_state.camera) {
                                cam.insert(ViewportRect(None));
                            }
                        }
                        Pane::Welcome => {}
                    }
                }
            });
    }
}

#[derive(Component)]
pub struct SyncedViewport;

#[derive(SystemParam)]
pub struct SyncViewportParams<'w, 's> {
    panels: Query<'w, 's, (Entity, &'static conduit::well_known::Panel), Without<SyncedViewport>>,
    commands: Commands<'w, 's>,
    tile_state: ResMut<'w, TileState>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    entity_map: Res<'w, EntityMap>,
    grid_cell: Query<'w, 's, &'static GridCell<i128>>,
    graph_state: ResMut<'w, GraphsState>,
    metadata_store: Res<'w, MetadataStore>,
    hdr_enabled: ResMut<'w, HdrEnabled>,
}

pub fn sync_viewports(params: SyncViewportParams) {
    let SyncViewportParams {
        panels,
        mut commands,
        mut tile_state,
        asset_server,
        mut meshes,
        mut materials,
        mut render_layer_alloc,
        entity_map,
        grid_cell,
        mut graph_state,
        metadata_store,
        mut hdr_enabled,
    } = params;
    for (entity, panel) in panels.iter() {
        spawn_panel(
            panel,
            None,
            &asset_server,
            &mut tile_state,
            &mut meshes,
            &mut materials,
            &mut render_layer_alloc,
            &mut commands,
            &entity_map,
            &grid_cell,
            &mut graph_state,
            &metadata_store,
            &mut hdr_enabled,
        );

        commands.entity(entity).insert(SyncedViewport);
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_panel(
    panel: &Panel,
    parent_id: Option<TileId>,
    asset_server: &Res<AssetServer>,
    ui_state: &mut ResMut<TileState>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
    commands: &mut Commands,
    entity_map: &Res<EntityMap>,
    grid_cell: &Query<&GridCell<i128>>,
    graphs_state: &mut GraphsState,
    metadata_store: &Res<MetadataStore>,
    hdr_enabled: &mut ResMut<HdrEnabled>,
) -> Option<TileId> {
    match panel {
        conduit::well_known::Panel::Viewport(viewport) => {
            let pane = ViewportPane::spawn(
                commands,
                asset_server,
                meshes,
                materials,
                render_layer_alloc,
                viewport,
            );
            let camera = pane.camera.expect("no camera spawned for viewport");
            let mut camera = commands.entity(camera);
            if let Some(parent) = viewport.track_entity {
                if let Some(parent) = entity_map.0.get(&parent) {
                    if let Ok(grid_cell) = grid_cell.get(*parent) {
                        camera.insert(*grid_cell);
                    }
                    camera.set_parent(*parent);
                }
            };
            camera.insert(Transform {
                translation: Vec3::new(viewport.pos.x, viewport.pos.y, viewport.pos.z),
                rotation: Quat::from_xyzw(
                    viewport.rotation.i,
                    viewport.rotation.j,
                    viewport.rotation.k,
                    viewport.rotation.w,
                ),
                ..Default::default()
            });
            if !viewport.track_rotation {
                camera.insert(NoPropagateRot);
            } else {
                camera.remove::<NoPropagateRot>();
            }
            hdr_enabled.0 |= viewport.hdr;
            let pane = Pane::Viewport(pane);
            let tile_id = ui_state.insert_pane(pane)?;
            ui_state.set_parent(tile_id, parent_id, viewport.active)
        }
        conduit::well_known::Panel::VSplit(split) => {
            let tile_id = ui_state.tree.tiles.insert_vertical_tile(vec![]);
            split.panels.iter().for_each(|panel| {
                spawn_panel(
                    panel,
                    Some(tile_id),
                    asset_server,
                    ui_state,
                    meshes,
                    materials,
                    render_layer_alloc,
                    commands,
                    entity_map,
                    grid_cell,
                    graphs_state,
                    metadata_store,
                    hdr_enabled,
                );
            });
            ui_state.set_parent(tile_id, parent_id, split.active)
        }
        conduit::well_known::Panel::HSplit(split) => {
            let tile_id = ui_state.tree.tiles.insert_horizontal_tile(vec![]);
            split.panels.iter().for_each(|panel| {
                spawn_panel(
                    panel,
                    Some(tile_id),
                    asset_server,
                    ui_state,
                    meshes,
                    materials,
                    render_layer_alloc,
                    commands,
                    entity_map,
                    grid_cell,
                    graphs_state,
                    metadata_store,
                    hdr_enabled,
                );
            });
            ui_state.set_parent(tile_id, parent_id, split.active)
        }
        conduit::well_known::Panel::Graph(graph) => {
            let mut entities = BTreeMap::<EntityId, GraphStateEntity>::default();
            for entity in graph.entities.iter() {
                let mut components: BTreeMap<ComponentId, Vec<(bool, Color32)>> = BTreeMap::new();
                for component in entity.components.iter() {
                    if let Some(metadata) = metadata_store.get_metadata(&component.component_id) {
                        let len = metadata.component_type.shape.iter().product::<i64>() as usize;
                        let values =
                            components.entry(component.component_id).or_insert_with(|| {
                                (0..len)
                                    .map(|i| {
                                        (entity.entity_id.0 + component.component_id.0) as usize + i
                                    })
                                    .map(|i| (false, colors::get_color_by_index_all(i)))
                                    .collect()
                            });
                        for index in component.indexes.iter() {
                            if let Some((enabled, _)) = values.get_mut(*index) {
                                *enabled = true;
                            }
                        }
                    }
                }
                entities.insert(entity.entity_id, components);
            }
            let state = GraphState::spawn(commands, render_layer_alloc, entities);
            let graph_id = graphs_state.push_graph_state(state);
            let graph = GraphPane::spawn(graph_id);
            let pane = Pane::Graph(graph);
            let tile_id = ui_state.insert_pane(pane)?;
            ui_state.set_parent(tile_id, parent_id, false)
        }
    }
}

pub fn shortcuts(kbd: Res<ButtonInput<KeyCode>>, mut ui_state: ResMut<TileState>) {
    // tab switching
    if kbd.pressed(KeyCode::ControlLeft) && kbd.just_pressed(KeyCode::Tab) {
        // TODO(sphw): we should have a more inteligant focus system
        let Some(tile_id) = ui_state.tree.root() else {
            return;
        };
        let Some(tile) = ui_state.tree.tiles.get_mut(tile_id) else {
            return;
        };
        let Tile::Container(container) = tile else {
            return;
        };

        let Container::Tabs(tabs) = container else {
            return;
        };
        let Some(active_id) = tabs.active else {
            return;
        };
        let Some(index) = tabs.children.iter().position(|x| *x == active_id) else {
            return;
        };
        let offset = if kbd.pressed(KeyCode::ShiftLeft) {
            -1
        } else {
            1
        };
        let new_index = if let Some(index) = index.checked_add_signed(offset) {
            index % tabs.children.len()
        } else {
            tabs.children.len() - 1
        };
        let Some(new_active_id) = tabs.children.get(new_index) else {
            return;
        };
        tabs.set_active(*new_active_id);
    }
}
