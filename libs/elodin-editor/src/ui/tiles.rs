use std::collections::HashMap;

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
    bevy::{EntityMap, TimeStep},
    query::MetadataStore,
    well_known::{Panel, Viewport},
    ComponentId, GraphId,
};
use egui_tiles::{Container, Tile, TileId, Tiles};

use super::{
    colors, images,
    utils::MarginSides,
    widgets::{
        button::EImageButton, eplot::EPlot, timeline::tagged_range::TaggedRanges, RootWidgetSystem,
    },
    GraphsState, SelectedObject, ViewportRect,
};
use crate::{plugins::navigation_gizmo::RenderLayerAlloc, spawn_main_camera, CollectedGraphData};

struct TabIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
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

    pub fn create_graph_tile(&mut self, graph_id: GraphId) {
        if let Some(parent) = self.tree.root {
            self.tab_diffs
                .push(TabDiff::AddGraph(parent, Some(graph_id)));
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

    fn ui(
        &mut self,
        ui: &mut Ui,
        time_step: std::time::Duration,
        collected_graph_data: &CollectedGraphData,
        graphs_state: &mut GraphsState,
        tagged_ranges: &TaggedRanges,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                ui.painter()
                    .rect_filled(content_rect, 0.0, colors::PRIMARY_SMOKE);

                let (_, graph_state) = graphs_state.get_or_create_graph(&Some(pane.id));

                let tagged_range = graph_state
                    .range_id
                    .as_ref()
                    .and_then(|rid| tagged_ranges.0.get(rid));

                EPlot::new()
                    .padding(egui::Margin::same(0.0).left(20.0).bottom(20.0))
                    .margin(egui::Margin::same(60.0).left(85.0).top(40.0))
                    .steps(7, 4)
                    .time_step(time_step)
                    .calculate_lines(ui, collected_graph_data, graph_state, tagged_range)
                    .render(ui);

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
}

impl GraphPane {
    fn spawn(graph_id: GraphId) -> Self {
        Self {
            id: graph_id,
            label: format!("Graph {}", graph_id.0),
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

struct TreeBehavior<'a> {
    icons: TabIcons,
    tab_diffs: Vec<TabDiff>,
    selected_object: &'a mut SelectedObject,
    graphs_state: &'a mut GraphsState,
    tagged_ranges: &'a TaggedRanges,
    collected_graph_data: &'a CollectedGraphData,
    time_step: std::time::Duration,
}

#[derive(Clone)]
pub enum TabDiff {
    AddViewport(TileId),
    AddGraph(TileId, Option<GraphId>),
    Delete(TileId),
}

enum TabState {
    Active,
    Selected,
    Inactive,
}

impl<'a> egui_tiles::Behavior<Pane> for TreeBehavior<'a> {
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
            self.collected_graph_data,
            self.graphs_state,
            self.tagged_ranges,
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
            resp.clicked = [false, true, false, false, false];
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

        let mut contexts = state_mut.contexts;
        let mut ui_state = state_mut.ui_state;
        let mut commands = state_mut.commands;
        let asset_server = state_mut.asset_server;
        let images = state_mut.images;
        let mut meshes = state_mut.meshes;
        let mut materials = state_mut.materials;
        let mut render_layer_alloc = state_mut.render_layer_alloc;
        let mut selected_object = state_mut.selected_object;
        let collected_graph_data = state_mut.collected_graph_data;
        let mut graphs_state = state_mut.graphs_state;
        let tagged_ranges = state_mut.tagged_ranges;

        let icons = TabIcons {
            add: contexts.add_image(images.icon_add.clone_weak()),
            close: contexts.add_image(images.icon_close.clone_weak()),
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
                    time_step: state_mut.time_step.0,
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
                        TabDiff::AddGraph(parent, graph_id) => {
                            let (graph_id, _) = graphs_state.get_or_create_graph(&graph_id);

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
                    let Pane::Viewport(viewport) = pane else {
                        continue;
                    };
                    let Some(cam) = viewport.camera else { continue };
                    if active_tiles.contains(tile_id) {
                        if let Some(mut cam) = commands.get_entity(cam) {
                            cam.insert(ViewportRect(viewport.rect));
                        }
                    } else if let Some(mut cam) = commands.get_entity(cam) {
                        cam.insert(ViewportRect(None));
                    }
                }
            });
    }
}

#[derive(Component)]
pub struct SyncedViewport;

#[allow(clippy::too_many_arguments)]
pub fn sync_viewports(
    panels: Query<(Entity, &conduit::well_known::Panel), Without<SyncedViewport>>,
    asset_server: Res<AssetServer>,
    mut ui_state: ResMut<TileState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_layer_alloc: ResMut<RenderLayerAlloc>,
    mut commands: Commands,
    entity_map: Res<EntityMap>,
    grid_cell: Query<&GridCell<i128>>,
    mut graph_state: ResMut<GraphsState>,
    metadata_store: Res<MetadataStore>,
) {
    for (entity, panel) in panels.iter() {
        spawn_panel(
            panel,
            None,
            &asset_server,
            &mut ui_state,
            &mut meshes,
            &mut materials,
            &mut render_layer_alloc,
            &mut commands,
            &entity_map,
            &grid_cell,
            &mut graph_state,
            &metadata_store,
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
                );
            });
            ui_state.set_parent(tile_id, parent_id, split.active)
        }
        conduit::well_known::Panel::Graph(graph) => {
            let (graph_id, _) = graphs_state.get_or_create_graph(&None);
            for entity in graph.entities.iter() {
                let mut components: HashMap<ComponentId, Vec<(bool, Color32)>> = HashMap::new();
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
                for (id, values) in components.into_iter() {
                    graphs_state.insert_component(&graph_id, &entity.entity_id, &id, values);
                }
            }
            let graph = GraphPane::spawn(graph_id);
            // let graph_id = graph.id;
            // let graph_label = graph.label.clone();
            let pane = Pane::Graph(graph);
            let tile_id = ui_state.insert_pane(pane)?;
            ui_state.set_parent(tile_id, parent_id, false)
        }
    }
}
