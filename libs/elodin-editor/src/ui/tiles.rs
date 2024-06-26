use std::collections::{BTreeMap, HashMap};

use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion};
use bevy_egui::{
    egui::{self, vec2, Color32, Frame, RichText, Rounding, Stroke, Ui, Visuals},
    EguiContexts,
};
use big_space::propagation::NoPropagateRot;
use big_space::GridCell;
use conduit::{
    bevy::{EntityMap, Tick, TimeStep},
    query::MetadataStore,
    well_known::{EntityMetadata, Panel, Viewport},
    ComponentId, ControlMsg, EntityId,
};
use egui_tiles::{Container, Tile, TileId, Tiles};

use super::{
    colors, images,
    widgets::{
        button::{EImageButton, ETileButton},
        modal::ModalNewTile,
        plot::{self, GraphBundle, GraphState, Line, Plot},
        timeline::timeline_ranges::{TimelineRangeId, TimelineRanges},
        WidgetSystem, WidgetSystemExt,
    },
    HdrEnabled, SelectedObject, ViewportRect,
};
use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    spawn_main_camera,
    ui::widgets::plot::{CollectedGraphData, GraphStateEntity},
    MainCamera,
};

#[derive(Clone)]
pub struct TileIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
    pub scrub: egui::TextureId,
    pub tile_3d_viewer: egui::TextureId,
    pub tile_graph: egui::TextureId,
}

#[derive(Resource)]
pub struct TileState {
    tree: egui_tiles::Tree<Pane>,
    tree_actions: Vec<TreeAction>,
    graphs: HashMap<TileId, Entity>,
}

#[derive(Resource, Default)]
pub struct ViewportContainsPointer(pub bool);

impl TileState {
    fn insert_tile(
        &mut self,
        tile: Tile<Pane>,
        parent_id: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        let parent_id = parent_id.or_else(|| self.tree.root()).or_else(|| {
            self.tree = egui_tiles::Tree::new_tabs("tab_tree", vec![]);
            self.tree.root()
        })?;

        let tile_id = self.tree.tiles.insert_new(tile);
        let parent_tile = self.tree.tiles.get_mut(parent_id)?;

        let Tile::Container(container) = parent_tile else {
            return None;
        };

        container.add_child(tile_id);

        if active {
            if let Container::Tabs(tabs) = container {
                tabs.set_active(tile_id);
            }
        }

        Some(tile_id)
    }

    pub fn create_graph_tile(&mut self, parent_id: Option<TileId>, graph_state: GraphBundle) {
        self.tree_actions
            .push(TreeAction::AddGraph(parent_id, Some(graph_state)));
    }

    pub fn create_graph_tile_empty(&mut self) {
        self.tree_actions.push(TreeAction::AddGraph(None, None));
    }

    pub fn create_viewport_tile(&mut self, focus_entity: Option<EntityId>) {
        self.tree_actions
            .push(TreeAction::AddViewport(None, focus_entity));
    }

    pub fn create_viewport_tile_empty(&mut self) {
        self.tree_actions.push(TreeAction::AddViewport(None, None));
    }

    pub fn is_empty(&self) -> bool {
        self.tree.active_tiles().is_empty()
    }
}

enum Pane {
    Viewport(ViewportPane),
    Graph(GraphPane),
}

impl Pane {
    fn title(&self) -> &str {
        match self {
            Pane::Graph(pane) => &pane.label,
            Pane::Viewport(_) => "Viewport",
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        ui: &mut Ui,
        time_step: std::time::Duration,
        current_tick: u64,
        collected_graph_data: &mut CollectedGraphData,
        graphs_state: &mut Query<&mut GraphState>,
        timeline_ranges: &TimelineRanges,
        lines: &mut Assets<Line>,
        lines_query: &Query<&Handle<Line>>,
        commands: &mut Commands,
        icons: &TileIcons,
        entity_map: &EntityMap,
        entity_metadata: &Query<&EntityMetadata>,
        metadata_store: &MetadataStore,
        control_msg: &mut EventWriter<ControlMsg>,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                let mut rect = plot::get_inner_rect(content_rect);
                rect.min.y = content_rect.min.y - 5.0;
                rect.max.y = content_rect.max.y - 5.0;
                pane.rect = Some(rect);
                // ui.painter()
                //     .rect_filled(content_rect, 0.0, colors::PRIMARY_SMOKE);

                let Ok(mut graph_state) = graphs_state.get_mut(pane.id) else {
                    return egui_tiles::UiResponse::None;
                };

                let timeline_range = graph_state
                    .range_id
                    .as_ref()
                    .and_then(|rid| timeline_ranges.0.get(rid));

                Plot::new()
                    .time_step(time_step)
                    .current_tick(current_tick)
                    .calculate_lines(
                        ui,
                        collected_graph_data,
                        graph_state.as_mut(),
                        timeline_range,
                        lines,
                        commands,
                        pane.id,
                        entity_map,
                        entity_metadata,
                        metadata_store,
                        control_msg,
                    )
                    .render(
                        ui,
                        lines,
                        lines_query,
                        collected_graph_data,
                        graph_state.as_mut(),
                        &icons.scrub,
                    );

                egui_tiles::UiResponse::None
            }
            Pane::Viewport(pane) => {
                pane.rect = Some(content_rect.shrink(1.0));
                egui_tiles::UiResponse::None
            }
        }
    }
}

#[derive(Default)]
struct ViewportPane {
    pub camera: Option<Entity>,
    pub nav_gizmo: Option<Entity>,
    pub nav_gizmo_camera: Option<Entity>,
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
        let (camera, nav_gizmo, nav_gizmo_camera) = spawn_main_camera(
            commands,
            asset_server,
            meshes,
            materials,
            render_layer_alloc,
            viewport,
        );
        Self {
            camera: Some(camera),
            nav_gizmo,
            nav_gizmo_camera,
            rect: None,
        }
    }
}

struct GraphPane {
    pub id: Entity,
    pub label: String,
    pub rect: Option<egui::Rect>,
}

impl GraphPane {
    fn new(graph_id: Entity, index: usize) -> Self {
        Self {
            id: graph_id,
            label: format!("Graph {:?}", index),
            rect: None,
        }
    }
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            tree: egui_tiles::Tree::new_tabs("tab_tree", vec![]),
            tree_actions: vec![],
            graphs: HashMap::new(),
        }
    }
}

struct TreeBehavior<'a, 'w, 's> {
    icons: TileIcons,
    tree_actions: Vec<TreeAction>,
    selected_object: &'a mut SelectedObject,
    timeline_ranges: &'a TimelineRanges,
    collected_graph_data: &'a mut CollectedGraphData,
    lines: &'a mut Assets<Line>,
    line_query: &'a Query<'w, 's, &'static Handle<Line>>,
    graph_state_query: &'a mut Query<'w, 's, &'static mut GraphState>,
    time_step: std::time::Duration,
    current_tick: u64,
    commands: &'a mut Commands<'w, 's>,
    entity_map: &'a EntityMap,
    entity_metadata: &'a Query<'w, 's, &'static EntityMetadata>,
    metadata_store: &'a MetadataStore,
    control_msg: &'a mut EventWriter<'w, ControlMsg>,
    new_tile_state: ResMut<'w, NewTileState>,
}

pub enum TreeAction {
    AddViewport(Option<TileId>, Option<EntityId>),
    AddGraph(Option<TileId>, Option<GraphBundle>),
    DeleteTab(TileId),
    SelectTile(TileId),
}

enum TabState {
    Active,
    Selected,
    Inactive,
}

impl<'a, 'w, 's> egui_tiles::Behavior<Pane> for TreeBehavior<'a, 'w, 's> {
    fn on_edit(&mut self, edit_action: egui_tiles::EditAction) {
        // NOTE: Override accidental selection onDrag
        if edit_action == egui_tiles::EditAction::TabSelected {
            if let Some(tile_id) = self.selected_object.tile_id() {
                self.tree_actions.push(TreeAction::SelectTile(tile_id));
            }
        }
    }

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
            self.graph_state_query,
            self.timeline_ranges,
            self.lines,
            self.line_query,
            self.commands,
            &self.icons,
            self.entity_map,
            self.entity_metadata,
            self.metadata_store,
            self.control_msg,
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
                self.tree_actions.push(TreeAction::DeleteTab(tile_id));
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
            self.tree_actions.push(TreeAction::DeleteTab(tile_id));
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
                Tile::Pane(Pane::Viewport(viewport)) => {
                    let Some(camera) = viewport.camera else {
                        return button_response;
                    };
                    *self.selected_object = SelectedObject::Viewport { tile_id, camera };
                }
                _ => {
                    *self.selected_object = SelectedObject::None;
                }
            }
            self.tree_actions.push(TreeAction::SelectTile(tile_id));
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
                self.tree_actions
                    .push(TreeAction::AddViewport(Some(tile_id), None));
                ui.close_menu();
            }
            ui.separator();
            if ui.button("GRAPH").clicked() {
                *self.new_tile_state = NewTileState::Graph {
                    entity_id: None,
                    component_id: None,
                    range_id: None,
                    parent_id: Some(tile_id),
                };
                ui.close_menu();
            }
        });
    }
}

#[derive(SystemParam)]
pub struct TileSystem<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    ui_state: Res<'w, TileState>,
    new_tile_state: Res<'w, NewTileState>,
}

impl WidgetSystem for TileSystem<'_, '_> {
    type Args = ();
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let mut contexts = state_mut.contexts;
        let images = state_mut.images;
        let ui_state = state_mut.ui_state;
        let new_tile_state = state_mut.new_tile_state;

        let icons = TileIcons {
            add: contexts.add_image(images.icon_add.clone_weak()),
            close: contexts.add_image(images.icon_close.clone_weak()),
            scrub: contexts.add_image(images.icon_scrub.clone_weak()),
            tile_3d_viewer: contexts.add_image(images.icon_tile_3d_viewer.clone_weak()),
            tile_graph: contexts.add_image(images.icon_tile_graph.clone_weak()),
        };

        let is_empty_tile_tree = ui_state.is_empty() && ui_state.tree_actions.is_empty();
        let is_tile_modal_closed = matches!(new_tile_state.as_ref(), NewTileState::None);

        let center_panel = egui::CentralPanel::default()
            .frame(Frame {
                fill: if is_empty_tile_tree {
                    colors::PRIMARY_SMOKE
                } else {
                    colors::TRANSPARENT
                },
                ..Default::default()
            })
            .show_inside(ui, |ui| {
                if is_empty_tile_tree {
                    if is_tile_modal_closed {
                        ui.add_widget_with::<TileLayoutEmpty>(
                            world,
                            "tile_layout_empty",
                            icons.clone(),
                        );
                    }
                } else {
                    ui.add_widget_with::<TileLayout>(world, "tile_layout", icons.clone());
                }
            });

        let center_pos = center_panel.response.rect.center();
        ui.add_widget_with::<ModalNewTile>(world, "modal_new_tile", (center_pos, icons));
    }
}

#[derive(Resource, Default, Clone)]
pub enum NewTileState {
    #[default]
    None,
    Viewport(Option<EntityId>, Option<TimelineRangeId>),
    Graph {
        entity_id: Option<EntityId>,
        component_id: Option<ComponentId>,
        range_id: Option<TimelineRangeId>,
        parent_id: Option<TileId>,
    },
}

#[derive(SystemParam)]
pub struct TileLayoutEmpty<'w> {
    new_tile_state: ResMut<'w, NewTileState>,
}

impl WidgetSystem for TileLayoutEmpty<'_> {
    type Args = TileIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let mut new_tile_state = state_mut.new_tile_state;

        let icons = args;

        let button_height = 160.0;
        let button_width = 240.0;
        let button_spacing = 20.0;

        let desired_size = egui::vec2(button_width * 2.0 + button_spacing, button_height);

        ui.allocate_ui_at_rect(
            egui::Rect::from_center_size(ui.max_rect().center(), desired_size),
            |ui| {
                ui.horizontal(|ui| {
                    ui.style_mut().spacing.item_spacing = egui::vec2(button_spacing, 0.0);

                    let create_viewport_btn = ui.add(
                        ETileButton::new("Viewport", icons.add)
                            .description("3D Output")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_viewport_btn.clicked() {
                        *new_tile_state = NewTileState::Viewport(None, None);
                    }

                    let create_graph_btn = ui.add(
                        ETileButton::new("Graph", icons.add)
                            .description("Point Graph")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_graph_btn.clicked() {
                        *new_tile_state = NewTileState::Graph {
                            entity_id: None,
                            component_id: None,
                            range_id: None,
                            parent_id: None,
                        };
                    }
                });
            },
        );
    }
}

#[derive(SystemParam)]
pub struct TileLayout<'w, 's> {
    commands: Commands<'w, 's>,
    timeline_ranges: Res<'w, TimelineRanges>,
    selected_object: ResMut<'w, SelectedObject>,
    ui_state: ResMut<'w, TileState>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    collected_graph_data: ResMut<'w, CollectedGraphData>,
    time_step: Res<'w, TimeStep>,
    tick: Res<'w, Tick>,
    lines: ResMut<'w, Assets<Line>>,
    line_query: Query<'w, 's, &'static Handle<Line>>,
    graph_state_query: Query<'w, 's, &'static mut GraphState>,
    entity_map: Res<'w, EntityMap>,
    entity_metadata: Query<'w, 's, &'static EntityMetadata>,
    metadata_store: Res<'w, MetadataStore>,
    control_msg: EventWriter<'w, ControlMsg>,
    viewport_contains_pointer: ResMut<'w, ViewportContainsPointer>,
    editor_cam_query: Query<'w, 's, &'static mut EditorCam, With<MainCamera>>,
    grid_cell: Query<'w, 's, &'static GridCell<i128>, Without<MainCamera>>,
    new_tile_state: ResMut<'w, NewTileState>,
}

impl WidgetSystem for TileLayout<'_, '_> {
    type Args = TileIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let TileLayout {
            mut commands,
            mut graph_state_query,
            timeline_ranges,
            mut selected_object,
            mut ui_state,
            asset_server,
            mut meshes,
            mut materials,
            mut render_layer_alloc,
            mut collected_graph_data,
            time_step,
            tick,
            mut lines,
            line_query,
            entity_map,
            entity_metadata,
            metadata_store,
            mut control_msg,
            mut viewport_contains_pointer,
            mut editor_cam_query,
            grid_cell,
            new_tile_state,
        } = state_mut;

        let icons = args;

        viewport_contains_pointer.0 = ui.ui_contains_pointer();

        for mut editor_cam in editor_cam_query.iter_mut() {
            editor_cam.enabled_motion = EnabledMotion {
                pan: viewport_contains_pointer.0,
                orbit: viewport_contains_pointer.0,
                zoom: viewport_contains_pointer.0,
            }
        }

        let tab_diffs = std::mem::take(&mut ui_state.tree_actions);
        let mut behavior = TreeBehavior {
            icons,
            tree_actions: tab_diffs,
            selected_object: selected_object.as_mut(),
            timeline_ranges: timeline_ranges.as_ref(),
            collected_graph_data: collected_graph_data.as_mut(),
            time_step: time_step.0,
            current_tick: tick.0,
            lines: lines.as_mut(),
            commands: &mut commands,
            line_query: &line_query,
            graph_state_query: &mut graph_state_query,
            entity_map: &entity_map,
            entity_metadata: &entity_metadata,
            metadata_store: &metadata_store,
            control_msg: &mut control_msg,
            new_tile_state,
        };
        ui_state.tree.ui(&mut behavior, ui);
        for diff in behavior.tree_actions.drain(..) {
            match diff {
                TreeAction::DeleteTab(tile_id) => {
                    let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
                        continue;
                    };

                    if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
                        if let Some(camera) = viewport.camera {
                            commands.entity(camera).despawn();
                        }
                        if let Some(nav_gizmo_camera) = viewport.nav_gizmo_camera {
                            commands.entity(nav_gizmo_camera).despawn();
                        }
                        if let Some(nav_gizmo) = viewport.nav_gizmo {
                            commands.entity(nav_gizmo).despawn();
                        }
                    };

                    if let egui_tiles::Tile::Pane(Pane::Graph(graph)) = tile {
                        commands.entity(graph.id).despawn();
                    };
                    ui_state.tree.remove_recursively(tile_id);

                    if let Some(graph_id) = ui_state.graphs.get(&tile_id) {
                        commands.entity(*graph_id).despawn();
                        ui_state.graphs.remove(&tile_id);
                        if selected_object.is_tile_selected(tile_id) {
                            *selected_object = SelectedObject::None;
                        }
                    }
                }
                TreeAction::AddViewport(parent_tile_id, track_entity) => {
                    let viewport = Viewport {
                        track_entity,
                        ..Viewport::default()
                    };
                    let viewport_pane = ViewportPane::spawn(
                        &mut commands,
                        &asset_server,
                        &mut meshes,
                        &mut materials,
                        &mut render_layer_alloc,
                        &viewport,
                    );
                    if let Some(camera) = viewport_pane.camera {
                        let mut camera = commands.entity(camera);
                        if let Some(parent) = viewport.track_entity {
                            if let Some(parent) = entity_map.0.get(&parent) {
                                if let Ok(grid_cell) = grid_cell.get(*parent) {
                                    camera.insert(*grid_cell);
                                }
                                camera.set_parent(*parent);
                            }
                        }
                    }

                    if let Some(tile_id) = ui_state.insert_tile(
                        Tile::Pane(Pane::Viewport(viewport_pane)),
                        parent_tile_id,
                        true,
                    ) {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                }
                TreeAction::AddGraph(parent_tile_id, graph_bundle) => {
                    let graph_bundle = if let Some(graph_bundle) = graph_bundle {
                        graph_bundle
                    } else {
                        GraphBundle::new(&mut render_layer_alloc, BTreeMap::default(), None)
                    };
                    let graph_id = commands.spawn(graph_bundle).id();

                    let graph = GraphPane::new(graph_id, graph_state_query.iter().len());
                    let graph_id = graph.id;
                    let graph_label = graph.label.clone();
                    let pane = Pane::Graph(graph);

                    if let Some(tile_id) =
                        ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                    {
                        *selected_object = SelectedObject::Graph {
                            tile_id,
                            label: graph_label,
                            graph_id,
                        };
                        ui_state.tree.make_active(|id, _| id == tile_id);
                        ui_state.graphs.insert(tile_id, graph_id);
                    }
                }
                TreeAction::SelectTile(tile_id) => {
                    ui_state.tree.make_active(|id, _| id == tile_id);
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
                    if active_tiles.contains(tile_id) {
                        if let Some(mut cam) = commands.get_entity(graph.id) {
                            cam.insert(ViewportRect(graph.rect));
                        }
                    } else if let Some(mut cam) = commands.get_entity(graph.id) {
                        cam.insert(ViewportRect(None));
                    }
                }
            }
        }
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
    metadata_store: Res<'w, MetadataStore>,
    hdr_enabled: ResMut<'w, HdrEnabled>,
    graph_states: Query<'w, 's, &'static GraphState>,
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
        metadata_store,
        mut hdr_enabled,
        graph_states,
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
            &metadata_store,
            &mut hdr_enabled,
            &graph_states,
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
    metadata_store: &Res<MetadataStore>,
    hdr_enabled: &mut ResMut<HdrEnabled>,
    graph_states: &Query<&GraphState>,
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
            // Convert from Z-up to Y-up
            let pos = [viewport.pos.x, viewport.pos.z, -viewport.pos.y];
            camera.insert(Transform {
                translation: Vec3::from_array(pos),
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
            ui_state.insert_tile(Tile::Pane(Pane::Viewport(pane)), parent_id, viewport.active)
        }
        conduit::well_known::Panel::VSplit(split) => {
            let tile_id = ui_state.insert_tile(
                Tile::Container(Container::new_linear(
                    egui_tiles::LinearDir::Vertical,
                    vec![],
                )),
                parent_id,
                false,
            );
            split.panels.iter().for_each(|panel| {
                spawn_panel(
                    panel,
                    tile_id,
                    asset_server,
                    ui_state,
                    meshes,
                    materials,
                    render_layer_alloc,
                    commands,
                    entity_map,
                    grid_cell,
                    metadata_store,
                    hdr_enabled,
                    graph_states,
                );
            });
            tile_id
        }
        conduit::well_known::Panel::HSplit(split) => {
            let tile_id = ui_state.insert_tile(
                Tile::Container(Container::new_linear(
                    egui_tiles::LinearDir::Horizontal,
                    vec![],
                )),
                parent_id,
                false,
            );
            split.panels.iter().for_each(|panel| {
                spawn_panel(
                    panel,
                    tile_id,
                    asset_server,
                    ui_state,
                    meshes,
                    materials,
                    render_layer_alloc,
                    commands,
                    entity_map,
                    grid_cell,
                    metadata_store,
                    hdr_enabled,
                    graph_states,
                );
            });
            tile_id
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

            let graph_id = commands
                .spawn(GraphBundle::new(render_layer_alloc, entities, None))
                .id();
            let graph = GraphPane::new(graph_id, graph_states.iter().len());
            ui_state.insert_tile(Tile::Pane(Pane::Graph(graph)), parent_id, false)
        }
    }
}

pub fn shortcuts(kbd: Res<ButtonInput<KeyCode>>, mut ui_state: ResMut<TileState>) {
    // tab switching
    if kbd.pressed(KeyCode::ControlLeft) && kbd.just_pressed(KeyCode::Tab) {
        // TODO(sphw): we should have a more intelligent focus system
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
