use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    ecs::system::{SystemParam, SystemState},
    input::keyboard::Key,
    prelude::*,
};
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion, OrbitConstraint};
use bevy_egui::{
    EguiContexts,
    egui::{self, Color32, CornerRadius, Frame, RichText, Stroke, Ui, Visuals, vec2},
};
use bevy_render::{
    camera::{Exposure, PhysicalCameraParameters},
    view::RenderLayers,
};
use egui::UiBuilder;
use egui_tiles::{Container, Tile, TileId, Tiles};
use impeller2::types::ComponentId;
use impeller2_wkt::{Dashboard, Graph, Viewport};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};

use super::{
    SelectedObject, ViewportRect,
    actions::ActionTileWidget,
    button::{EImageButton, ETileButton},
    colors::{self, get_scheme, with_opacity},
    command_palette::{CommandPaletteState, palette_items},
    dashboard::{DashboardWidget, spawn_dashboard},
    hierarchy::{Hierarchy, HierarchyContent},
    images,
    inspector::{InspectorContent, InspectorIcons},
    monitor::{MonitorPane, MonitorWidget},
    plot::{GraphBundle, GraphState, PlotWidget},
    query_plot::QueryPlotData,
    query_table::{QueryTableData, QueryTablePane, QueryTableWidget},
    schematic::{graph_label, viewport_label},
    video_stream::{IsTileVisible, VideoDecoderHandle},
    widgets::{WidgetSystem, WidgetSystemExt},
};
use crate::{
    EqlContext, GridHandle, MainCamera,
    object_3d::{EditableEQL, compile_eql_expr},
    plugins::{
        LogicalKeyState,
        navigation_gizmo::{RenderLayerAlloc, spawn_gizmo},
    },
    ui::dashboard::NodeUpdaterParams,
};

#[derive(Clone)]
pub struct TileIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
    pub scrub: egui::TextureId,
    pub tile_3d_viewer: egui::TextureId,
    pub tile_graph: egui::TextureId,
    pub subtract: egui::TextureId,
    pub setting: egui::TextureId,
    pub search: egui::TextureId,
    pub chart: egui::TextureId,
    pub chevron: egui::TextureId,
    pub plot: egui::TextureId,
    pub viewport: egui::TextureId,
    pub container: egui::TextureId,
    pub entity: egui::TextureId,
}

#[derive(Resource, Clone)]
pub struct TileState {
    pub tree: egui_tiles::Tree<Pane>,
    pub tree_actions: smallvec::SmallVec<[TreeAction; 4]>,
    pub graphs: HashMap<TileId, Entity>,
}

#[derive(Resource, Default)]
pub struct ViewportContainsPointer(pub bool);
#[derive(Clone)]
pub struct ActionTilePane {
    pub entity: Entity,
    pub label: String,
}

#[derive(Clone)]
pub struct TreePane {
    pub entity: Entity,
}

#[derive(Clone)]
pub struct DashboardPane {
    pub entity: Entity,
    pub label: String,
}

impl TileState {
    pub fn insert_tile(
        &mut self,
        tile: Tile<Pane>,
        parent_id: Option<TileId>,
        active: bool,
    ) -> Option<TileId> {
        let parent_id = if let Some(id) = parent_id {
            id
        } else {
            let root_id = self.tree.root().or_else(|| {
                self.tree = egui_tiles::Tree::new_tabs("tab_tree", vec![]);
                self.tree.root()
            })?;

            if let Some(Tile::Container(Container::Linear(linear))) = self.tree.tiles.get(root_id) {
                if let Some(center) = linear.children.get(linear.children.len() / 2) {
                    *center
                } else {
                    root_id
                }
            } else {
                root_id
            }
        };

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

    pub fn create_viewport_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddViewport(tile_id));
    }

    pub fn create_viewport_tile_empty(&mut self) {
        self.tree_actions.push(TreeAction::AddViewport(None));
    }

    pub fn create_monitor_tile(&mut self, component_id: ComponentId, tile_id: Option<TileId>) {
        self.tree_actions
            .push(TreeAction::AddMonitor(tile_id, component_id));
    }

    pub fn create_action_tile(
        &mut self,
        button_name: String,
        lua_code: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddActionTile(tile_id, button_name, lua_code));
    }

    pub fn create_query_table_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddQueryTable(tile_id));
    }

    pub fn create_query_plot_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddQueryPlot(tile_id));
    }

    pub fn create_video_stream_tile(
        &mut self,
        msg_id: [u8; 2],
        label: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions
            .push(TreeAction::AddVideoStream(tile_id, msg_id, label));
    }

    pub fn create_dashboard_tile(
        &mut self,
        dashboard: impeller2_wkt::Dashboard,
        label: String,
        tile_id: Option<TileId>,
    ) {
        self.tree_actions.push(TreeAction::AddDashboard(
            tile_id,
            Box::new(dashboard),
            label,
        ));
    }

    pub fn create_hierarchy_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddHierarchy(tile_id));
    }

    pub fn create_inspector_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions.push(TreeAction::AddInspector(tile_id));
    }

    pub fn create_tree_tile(&mut self, tile_id: Option<TileId>) {
        self.tree_actions
            .push(TreeAction::AddSchematicTree(tile_id));
    }

    pub fn create_sidebars_layout(&mut self) {
        self.tree_actions.push(TreeAction::AddSidebars);
    }

    pub fn is_empty(&self) -> bool {
        self.tree.active_tiles().is_empty()
    }

    pub fn clear(&mut self, commands: &mut Commands, _selected_object: &mut SelectedObject) {
        for (tile_id, tile) in self.tree.tiles.iter() {
            match tile {
                Tile::Pane(Pane::Viewport(viewport)) => {
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
                Tile::Pane(Pane::Graph(graph)) => {
                    commands.entity(graph.id).despawn();
                    if let Some(graph_id) = self.graphs.get(tile_id) {
                        commands.entity(*graph_id).despawn();
                        self.graphs.remove(tile_id);
                    }
                }

                Tile::Pane(Pane::VideoStream(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::QueryPlot(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::SchematicTree(pane)) => {
                    commands.entity(pane.entity).despawn();
                }
                Tile::Pane(Pane::Dashboard(dashboard)) => {
                    commands.entity(dashboard.entity).despawn();
                }
                _ => {}
            }

            // if selected_object.is_tile_selected(*tile_id) {
            //     *selected_object = SelectedObject::None;
            // }
        }

        if let Some(root_id) = self.tree.root() {
            if let Some(Tile::Container(root)) = self.tree.tiles.get_mut(root_id) {
                root.retain(|_| false);
            };
        };
    }
}

#[derive(Clone)]
pub enum Pane {
    Viewport(ViewportPane),
    Graph(GraphPane),
    Monitor(MonitorPane),
    QueryTable(QueryTablePane),
    QueryPlot(super::query_plot::QueryPlotPane),
    ActionTile(ActionTilePane),
    VideoStream(super::video_stream::VideoStreamPane),
    Dashboard(DashboardPane),
    Hierarchy,
    Inspector,
    SchematicTree(TreePane),
}

impl Pane {
    fn title(
        &self,
        graph_states: &Query<&GraphState>,
        dashboards: &Query<&Dashboard<Entity>>,
    ) -> String {
        match self {
            Pane::Graph(pane) => {
                if let Ok(graph_state) = graph_states.get(pane.id) {
                    return graph_state.label.to_string();
                }
                pane.label.to_string()
            }
            Pane::Viewport(viewport) => viewport.label.to_string(),
            Pane::Monitor(monitor) => monitor.label.to_string(),
            Pane::QueryTable(..) => "Query".to_string(),
            Pane::QueryPlot(query_plot) => {
                if let Ok(graph_state) = graph_states.get(query_plot.entity) {
                    return graph_state.label.to_string();
                }
                "Query Plot".to_string()
            }
            Pane::ActionTile(action) => action.label.to_string(),
            Pane::VideoStream(video_stream) => video_stream.label.to_string(),
            Pane::Dashboard(dashboard) => {
                if let Ok(dash) = dashboards.get(dashboard.entity) {
                    return dash
                        .root
                        .label
                        .as_deref()
                        .unwrap_or("Dashboard")
                        .to_string();
                }
                "Dashboard".to_string()
            }
            Pane::Hierarchy => "Entities".to_string(),
            Pane::Inspector => "Inspector".to_string(),
            Pane::SchematicTree(_) => "Tree".to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        ui: &mut Ui,
        //state: &mut TileLayout<'_, '_>,
        icons: &TileIcons,
        world: &mut World,
        tree_actions: &mut SmallVec<[TreeAction; 4]>,
    ) -> egui_tiles::UiResponse {
        let content_rect = ui.available_rect_before_wrap();
        match self {
            Pane::Graph(pane) => {
                pane.rect = Some(content_rect);

                ui.add_widget_with::<PlotWidget>(world, "graph", (pane.id, icons.scrub));

                egui_tiles::UiResponse::None
            }
            Pane::Viewport(pane) => {
                pane.rect = Some(content_rect);
                egui_tiles::UiResponse::None
            }
            Pane::Monitor(pane) => {
                ui.add_widget_with::<MonitorWidget>(world, "monitor", pane.clone());
                egui_tiles::UiResponse::None
            }
            Pane::QueryTable(pane) => {
                ui.add_widget_with::<QueryTableWidget>(world, "sql", pane.clone());
                egui_tiles::UiResponse::None
            }
            Pane::QueryPlot(pane) => {
                pane.rect = Some(content_rect);
                ui.add_widget_with::<super::query_plot::QueryPlotWidget>(
                    world,
                    "query_plot",
                    pane.clone(),
                );
                egui_tiles::UiResponse::None
            }
            Pane::ActionTile(pane) => {
                ui.add_widget_with::<ActionTileWidget>(world, "action_tile", pane.entity);
                egui_tiles::UiResponse::None
            }
            Pane::VideoStream(pane) => {
                ui.add_widget_with::<super::video_stream::VideoStreamWidget>(
                    world,
                    "video_stream",
                    pane.clone(),
                );
                egui_tiles::UiResponse::None
            }
            Pane::Dashboard(pane) => {
                ui.add_widget_with::<DashboardWidget>(world, "dashboard", pane.entity);
                egui_tiles::UiResponse::None
            }
            Pane::Hierarchy => {
                ui.add_widget_with::<HierarchyContent>(
                    world,
                    "hierarchy_content",
                    Hierarchy {
                        search: icons.search,
                        entity: icons.entity,
                        chevron: icons.chevron,
                    },
                );
                egui_tiles::UiResponse::None
            }
            Pane::Inspector => {
                let inspector_icons = InspectorIcons {
                    chart: icons.chart,
                    add: icons.add,
                    subtract: icons.subtract,
                    setting: icons.setting,
                    search: icons.search,
                };
                let actions = ui.add_widget_with::<InspectorContent>(
                    world,
                    "inspector_content",
                    (inspector_icons, true),
                );
                tree_actions.extend(actions);
                egui_tiles::UiResponse::None
            }
            Pane::SchematicTree(tree_pane) => {
                let tree_icons = super::schematic::tree::TreeIcons {
                    chevron: icons.chevron,
                    search: icons.search,
                    container: icons.container,
                    plot: icons.plot,
                    viewport: icons.viewport,
                    add: icons.add,
                };
                ui.add_widget_with::<super::schematic::tree::TreeWidget>(
                    world,
                    "tree",
                    (tree_icons, tree_pane.entity),
                );
                egui_tiles::UiResponse::None
            }
        }
    }
}

#[derive(Default, Clone)]
pub struct ViewportPane {
    pub camera: Option<Entity>,
    pub nav_gizmo: Option<Entity>,
    pub nav_gizmo_camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
    pub label: String,
}

impl ViewportPane {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        commands: &mut Commands,
        asset_server: &Res<AssetServer>,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
        eql_ctx: &eql::Context,
        viewport: &Viewport,
        label: String,
    ) -> Self {
        let mut main_camera_layers = RenderLayers::default();
        let mut grid_layers = RenderLayers::none();
        if let Some(grid_layer) = render_layer_alloc.alloc() {
            main_camera_layers = main_camera_layers.with(grid_layer);
            grid_layers = grid_layers.with(grid_layer);
        }

        let grid_visibility = if viewport.show_grid {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        let grid_id = commands
            .spawn((
                bevy_infinite_grid::InfiniteGridBundle {
                    settings: bevy_infinite_grid::InfiniteGridSettings {
                        minor_line_color: Color::srgba(1.0, 1.0, 1.0, 0.02),
                        major_line_color: Color::srgba(1.0, 1.0, 1.0, 0.05),
                        z_axis_color: crate::ui::colors::bevy::GREEN,
                        x_axis_color: crate::ui::colors::bevy::RED,
                        fadeout_distance: 50_000.0,
                        scale: 0.1,
                        ..Default::default()
                    },
                    visibility: grid_visibility,
                    ..Default::default()
                },
                grid_layers,
            ))
            .id();

        let parent = commands
            .spawn((
                GlobalTransform::default(),
                Transform::from_translation(Vec3::new(5.0, 5.0, 10.0))
                    .looking_at(Vec3::ZERO, Vec3::Y),
                impeller2_wkt::WorldPos::default(),
            ))
            .id();
        let pos = viewport
            .pos
            .as_ref()
            .map(|eql| {
                let compiled_expr = eql_ctx.parse_str(eql).ok().map(compile_eql_expr);
                EditableEQL {
                    eql: eql.to_string(),
                    compiled_expr,
                }
            })
            .unwrap_or_default();
        let look_at = viewport
            .look_at
            .as_ref()
            .map(|eql| {
                let compiled_expr = eql_ctx.parse_str(eql).ok().map(compile_eql_expr);
                EditableEQL {
                    eql: eql.to_string(),
                    compiled_expr,
                }
            })
            .unwrap_or_default();

        let mut camera = commands.spawn((
            Transform::default(),
            Camera3d::default(),
            Camera {
                hdr: false,
                clear_color: ClearColorConfig::Default,
                order: 1,
                ..Default::default()
            },
            Projection::Perspective(PerspectiveProjection {
                fov: viewport.fov.to_radians(),
                ..Default::default()
            }),
            Tonemapping::TonyMcMapface,
            Exposure::from_physical_camera(PhysicalCameraParameters {
                aperture_f_stops: 2.8,
                shutter_speed_s: 1.0 / 200.0,
                sensitivity_iso: 400.0,
                // full frame sensor height
                sensor_height: 24.0 / 1000.0,
            }),
            main_camera_layers,
            MainCamera,
            big_space::GridCell::<i128>::default(),
            EditorCam {
                orbit_constraint: OrbitConstraint::Fixed {
                    up: Vec3::Y,
                    can_pass_tdc: false,
                },
                last_anchor_depth: 2.0,
                ..Default::default()
            },
            GridHandle { grid: grid_id },
            crate::ui::inspector::viewport::Viewport::new(parent, pos, look_at),
            ChildOf(parent),
        ));

        camera.insert(Bloom { ..default() });
        camera.insert(EnvironmentMapLight {
            diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
            specular_map: asset_server.load("embedded://elodin_editor/assets/specular.ktx2"),
            intensity: 2000.0,
            ..Default::default()
        });

        let camera = camera.id();

        let (nav_gizmo, nav_gizmo_camera) =
            spawn_gizmo(camera, commands, meshes, materials, render_layer_alloc);
        Self {
            camera: Some(camera),
            nav_gizmo,
            nav_gizmo_camera,
            rect: None,
            label,
        }
    }
}

#[derive(Clone)]
pub struct GraphPane {
    pub id: Entity,
    pub label: String,
    pub rect: Option<egui::Rect>,
}

impl GraphPane {
    pub fn new(graph_id: Entity, label: String) -> Self {
        Self {
            id: graph_id,
            label,
            rect: None,
        }
    }
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            tree: egui_tiles::Tree::new_tabs("tab_tree", vec![]),
            tree_actions: SmallVec::new(),
            graphs: HashMap::new(),
        }
    }
}

struct TreeBehavior<'w> {
    icons: TileIcons,
    tree_actions: SmallVec<[TreeAction; 4]>,
    world: &'w mut World,
}

#[derive(Clone)]
pub enum TreeAction {
    AddViewport(Option<TileId>),
    AddGraph(Option<TileId>, Option<GraphBundle>),
    AddMonitor(Option<TileId>, ComponentId),
    AddQueryTable(Option<TileId>),
    AddQueryPlot(Option<TileId>),
    AddActionTile(Option<TileId>, String, String),
    AddVideoStream(Option<TileId>, [u8; 2], String),
    AddDashboard(Option<TileId>, Box<impeller2_wkt::Dashboard>, String),
    AddHierarchy(Option<TileId>),
    AddInspector(Option<TileId>),
    AddSchematicTree(Option<TileId>),
    AddSidebars,
    DeleteTab(TileId),
    SelectTile(TileId),
}

enum TabState {
    Selected,
    Inactive,
}

impl egui_tiles::Behavior<Pane> for TreeBehavior<'_> {
    fn on_edit(&mut self, _edit_action: egui_tiles::EditAction) {}

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        let mut query =
            SystemState::<(Query<&GraphState>, Query<&Dashboard<Entity>>)>::new(self.world);
        let (graphs, dashes) = query.get(self.world);
        pane.title(&graphs, &dashes).into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        pane.ui(ui, &self.icons, self.world, &mut self.tree_actions)
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn tab_ui(
        &mut self,
        tiles: &mut Tiles<Pane>,
        ui: &mut Ui,
        id: egui::Id,
        tile_id: egui_tiles::TileId,
        state: &egui_tiles::TabState,
    ) -> egui::Response {
        let tab_state = if state.active {
            TabState::Selected
        } else {
            TabState::Inactive
        };
        let text = self.tab_title_for_tile(tiles, tile_id);
        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 11.0;
        let galley = text.into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, font_id);
        let x_margin = self.tab_title_spacing(ui.visuals());
        let (_, rect) = ui.allocate_space(vec2(
            galley.size().x + x_margin * 4.0,
            ui.available_height(),
        ));
        let text_rect = rect
            .shrink2(vec2(x_margin * 4.0, 0.0))
            .translate(vec2(-3.0 * x_margin, 0.0));
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());

        if ui.is_rect_visible(rect) && !state.is_being_dragged {
            let scheme = get_scheme();
            let bg_color = match tab_state {
                TabState::Selected => scheme.text_primary,
                TabState::Inactive => scheme.bg_secondary,
            };

            let text_color = match tab_state {
                TabState::Selected => scheme.bg_secondary,
                TabState::Inactive => with_opacity(scheme.text_primary, 0.6),
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
                        TabState::Inactive => scheme.text_primary,
                        TabState::Selected => scheme.bg_primary,
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
                egui::Stroke::new(1.0, scheme.border_primary),
            );

            ui.painter().vline(
                rect.right(),
                rect.y_range(),
                egui::Stroke::new(1.0, scheme.border_primary),
            );
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn on_tab_button(
        &mut self,
        _tiles: &Tiles<Pane>,
        tile_id: TileId,
        button_response: egui::Response,
    ) -> egui::Response {
        if button_response.middle_clicked() {
            self.tree_actions.push(TreeAction::DeleteTab(tile_id));
        } else if button_response.clicked() {
            self.tree_actions.push(TreeAction::SelectTile(tile_id));
        }
        button_response
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        32.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        get_scheme().bg_secondary
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, get_scheme().text_primary)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::with_opacity(get_scheme().text_primary, 0.6)
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = get_scheme().text_primary;
        frame.corner_radius = CornerRadius::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = self.tab_title_for_tile(tiles, tile_id);
            let text = text.text();
            ui.label(
                RichText::new(text)
                    .color(get_scheme().bg_secondary)
                    .size(11.0),
            );
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
        let mut layout = SystemState::<TileLayout>::new(self.world);
        let mut layout = layout.get_mut(self.world);

        let top_bar_rect = ui.available_rect_before_wrap();
        ui.painter().hline(
            top_bar_rect.x_range(),
            top_bar_rect.bottom(),
            egui::Stroke::new(1.0, get_scheme().border_primary),
        );

        ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        ui.style_mut().visuals.widgets.active.bg_stroke = Stroke::NONE;
        ui.add_space(5.0);
        let resp = ui.add(EImageButton::new(self.icons.add).scale(1.4, 1.4));
        if resp.clicked() {
            layout
                .cmd_palette_state
                .open_page(move || palette_items::create_tiles(tile_id));
        }
    }
}

#[derive(SystemParam)]
pub struct TileSystem<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
    ui_state: Res<'w, TileState>,
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

        let icons = TileIcons {
            add: contexts.add_image(images.icon_add.clone_weak()),
            close: contexts.add_image(images.icon_close.clone_weak()),
            scrub: contexts.add_image(images.icon_scrub.clone_weak()),
            tile_3d_viewer: contexts.add_image(images.icon_tile_3d_viewer.clone_weak()),
            tile_graph: contexts.add_image(images.icon_tile_graph.clone_weak()),

            subtract: contexts.add_image(images.icon_subtract.clone_weak()),
            chart: contexts.add_image(images.icon_chart.clone_weak()),
            setting: contexts.add_image(images.icon_setting.clone_weak()),
            search: contexts.add_image(images.icon_search.clone_weak()),
            chevron: contexts.add_image(images.icon_chevron_right.clone_weak()),
            plot: contexts.add_image(images.icon_plot.clone_weak()),
            viewport: contexts.add_image(images.icon_viewport.clone_weak()),
            container: contexts.add_image(images.icon_container.clone_weak()),
            entity: contexts.add_image(images.icon_entity.clone_weak()),
        };

        let is_empty_tile_tree = ui_state.is_empty() && ui_state.tree_actions.is_empty();

        egui::CentralPanel::default()
            .frame(Frame {
                fill: if is_empty_tile_tree {
                    get_scheme().bg_secondary
                } else {
                    colors::TRANSPARENT
                },
                ..Default::default()
            })
            .show_inside(ui, |ui| {
                if is_empty_tile_tree {
                    ui.add_widget_with::<TileLayoutEmpty>(
                        world,
                        "tile_layout_empty",
                        icons.clone(),
                    );
                } else {
                    ui.add_widget_with::<TileLayout>(world, "tile_layout", icons.clone());
                }
            });
    }
}

#[derive(SystemParam)]
pub struct TileLayoutEmpty<'w> {
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
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
        let mut state_mut = state.get_mut(world);

        let icons = args;

        let button_height = 160.0;
        let button_width = 240.0;
        let button_spacing = 20.0;

        let desired_size = egui::vec2(button_width * 3.0 + button_spacing, button_height);

        ui.allocate_new_ui(
            UiBuilder::new().max_rect(egui::Rect::from_center_size(
                ui.max_rect().center(),
                desired_size,
            )),
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
                        state_mut
                            .cmd_palette_state
                            .open_item(palette_items::create_viewport(None));
                    }

                    let create_graph_btn = ui.add(
                        ETileButton::new("Graph", icons.add)
                            .description("Point Graph")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_graph_btn.clicked() {
                        state_mut
                            .cmd_palette_state
                            .open_item(palette_items::create_graph(None));
                    }

                    let create_monitor_btn = ui.add(
                        ETileButton::new("Monitor", icons.add)
                            .description("Monitor")
                            .width(button_width)
                            .height(160.0),
                    );

                    if create_monitor_btn.clicked() {
                        state_mut
                            .cmd_palette_state
                            .open_item(palette_items::create_monitor(None));
                    }
                });
            },
        );
    }
}

#[derive(SystemParam)]
pub struct TileLayout<'w, 's> {
    commands: Commands<'w, 's>,
    selected_object: ResMut<'w, SelectedObject>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    viewport_contains_pointer: ResMut<'w, ViewportContainsPointer>,
    editor_cam: Query<'w, 's, &'static mut EditorCam, With<MainCamera>>,
    cmd_palette_state: ResMut<'w, CommandPaletteState>,
    eql_ctx: Res<'w, EqlContext>,
    node_updater_params: NodeUpdaterParams<'w, 's>,
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
        world.resource_scope::<TileState, _>(|world, mut ui_state| {
            let icons = args;

            let mut tree_actions = {
                let tab_diffs = std::mem::take(&mut ui_state.tree_actions);
                let mut behavior = TreeBehavior {
                    icons,
                    world,
                    tree_actions: tab_diffs,
                };
                ui_state.tree.ui(&mut behavior, ui);

                let TreeBehavior { tree_actions, .. } = behavior;
                tree_actions
            };
            let mut state_mut = state.get_mut(world);
            state_mut.viewport_contains_pointer.0 = ui.ui_contains_pointer();

            for mut editor_cam in state_mut.editor_cam.iter_mut() {
                editor_cam.enabled_motion = EnabledMotion {
                    pan: state_mut.viewport_contains_pointer.0,
                    orbit: state_mut.viewport_contains_pointer.0,
                    zoom: state_mut.viewport_contains_pointer.0,
                }
            }

            for diff in tree_actions.drain(..) {
                match diff {
                    TreeAction::DeleteTab(tile_id) => {
                        let Some(tile) = ui_state.tree.tiles.get(tile_id) else {
                            continue;
                        };

                        if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
                            if let Some(camera) = viewport.camera {
                                state_mut.commands.entity(camera).despawn();
                            }
                            if let Some(nav_gizmo_camera) = viewport.nav_gizmo_camera {
                                state_mut.commands.entity(nav_gizmo_camera).despawn();
                            }
                            if let Some(nav_gizmo) = viewport.nav_gizmo {
                                state_mut.commands.entity(nav_gizmo).despawn();
                            }
                        };

                        if let egui_tiles::Tile::Pane(Pane::Graph(graph)) = tile {
                            state_mut.commands.entity(graph.id).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::ActionTile(action)) = tile {
                            state_mut.commands.entity(action.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::VideoStream(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::QueryPlot(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::QueryTable(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        if let egui_tiles::Tile::Pane(Pane::SchematicTree(pane)) = tile {
                            state_mut.commands.entity(pane.entity).despawn();
                        };

                        ui_state.tree.remove_recursively(tile_id);

                        if let Some(graph_id) = ui_state.graphs.get(&tile_id) {
                            state_mut.commands.entity(*graph_id).despawn();
                            ui_state.graphs.remove(&tile_id);
                        }
                    }
                    TreeAction::AddViewport(parent_tile_id) => {
                        let viewport = Viewport::default();
                        let label = viewport_label(&viewport);
                        let viewport_pane = ViewportPane::spawn(
                            &mut state_mut.commands,
                            &state_mut.asset_server,
                            &mut state_mut.meshes,
                            &mut state_mut.materials,
                            &mut state_mut.render_layer_alloc,
                            &state_mut.eql_ctx.0,
                            &viewport,
                            label,
                        );

                        if let Some(tile_id) = ui_state.insert_tile(
                            Tile::Pane(Pane::Viewport(viewport_pane)),
                            parent_tile_id,
                            true,
                        ) {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddGraph(parent_tile_id, graph_bundle) => {
                        let graph_label = graph_label(&Graph::default());

                        let graph_bundle = if let Some(graph_bundle) = graph_bundle {
                            graph_bundle
                        } else {
                            GraphBundle::new(
                                &mut state_mut.render_layer_alloc,
                                BTreeMap::default(),
                                graph_label.clone(),
                            )
                        };
                        let graph_id = state_mut.commands.spawn(graph_bundle).id();

                        let graph = GraphPane::new(graph_id, graph_label.clone());
                        let pane = Pane::Graph(graph);

                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            *state_mut.selected_object = SelectedObject::Graph { graph_id };
                            ui_state.tree.make_active(|id, _| id == tile_id);
                            ui_state.graphs.insert(tile_id, graph_id);
                        }
                    }
                    TreeAction::AddMonitor(parent_tile_id, component_id) => {
                        let monitor = MonitorPane::new("Monitor".to_string(), component_id);

                        let pane = Pane::Monitor(monitor);
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddVideoStream(parent_tile_id, msg_id, label) => {
                        let entity = state_mut
                            .commands
                            .spawn((
                                super::video_stream::VideoStream {
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
                        let pane = Pane::VideoStream(super::video_stream::VideoStreamPane {
                            entity,
                            label: label.clone(),
                        });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddDashboard(parent_tile_id, dashboard, label) => {
                        let entity = match spawn_dashboard(
                            &dashboard,
                            &state_mut.eql_ctx.0,
                            &mut state_mut.commands,
                            &state_mut.node_updater_params,
                        ) {
                            Ok(entity) => entity,
                            Err(_) => {
                                // Handle error - create a default entity or show error message
                                state_mut.commands.spawn(bevy::ui::Node::default()).id()
                            }
                        };
                        let pane = Pane::Dashboard(DashboardPane { entity, label });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }

                    TreeAction::SelectTile(tile_id) => {
                        ui_state.tree.make_active(|id, _| id == tile_id);
                    }
                    TreeAction::AddActionTile(parent_tile_id, button_name, lua_code) => {
                        let entity = state_mut
                            .commands
                            .spawn(super::actions::ActionTile {
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
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddQueryTable(parent_tile_id) => {
                        let entity = state_mut.commands.spawn(QueryTableData::default()).id();
                        let pane = Pane::QueryTable(QueryTablePane { entity });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddQueryPlot(parent_tile_id) => {
                        let graph_bundle = GraphBundle::new(
                            &mut state_mut.render_layer_alloc,
                            BTreeMap::default(),
                            "Query Plot".to_string(),
                        );
                        let entity = state_mut
                            .commands
                            .spawn(QueryPlotData::default())
                            .insert(graph_bundle)
                            .id();
                        let pane = Pane::QueryPlot(super::query_plot::QueryPlotPane {
                            entity,
                            rect: None,
                        });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            *state_mut.selected_object = SelectedObject::Graph { graph_id: entity };
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddHierarchy(parent_tile_id) => {
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(Pane::Hierarchy), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddInspector(parent_tile_id) => {
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(Pane::Inspector), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddSchematicTree(parent_tile_id) => {
                        let entity = state_mut
                            .commands
                            .spawn(super::schematic::tree::TreeWidgetState::default())
                            .id();
                        let pane = Pane::SchematicTree(TreePane { entity });
                        if let Some(tile_id) =
                            ui_state.insert_tile(Tile::Pane(pane), parent_tile_id, true)
                        {
                            ui_state.tree.make_active(|id, _| id == tile_id);
                        }
                    }
                    TreeAction::AddSidebars => {
                        let hierarchy = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Hierarchy));
                        let inspector = ui_state.tree.tiles.insert_new(Tile::Pane(Pane::Inspector));

                        let mut linear = egui_tiles::Linear::new(
                            egui_tiles::LinearDir::Horizontal,
                            vec![hierarchy, inspector],
                        );
                        if let Some(root) = ui_state.tree.root() {
                            linear.children.insert(1, root);
                            linear.shares.set_share(hierarchy, 0.2);
                            linear.shares.set_share(root, 0.6);
                            linear.shares.set_share(inspector, 0.2);
                        }
                        let root = ui_state
                            .tree
                            .tiles
                            .insert_new(Tile::Container(Container::Linear(linear)));
                        ui_state.tree.root = Some(root);
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
                            if let Ok(mut cam) = state_mut.commands.get_entity(cam) {
                                cam.try_insert(ViewportRect(viewport.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(cam) {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::Hierarchy => {}
                    Pane::Inspector => {}
                    Pane::SchematicTree(_) => {}
                    Pane::Graph(graph) => {
                        if active_tiles.contains(tile_id) {
                            if let Ok(mut cam) = state_mut.commands.get_entity(graph.id) {
                                cam.try_insert(ViewportRect(graph.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(graph.id) {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::Monitor(_) => {}
                    Pane::QueryTable(_) => {}
                    Pane::QueryPlot(query_plot) => {
                        if active_tiles.contains(tile_id) {
                            if let Ok(mut cam) = state_mut.commands.get_entity(query_plot.entity) {
                                cam.try_insert(ViewportRect(query_plot.rect));
                            }
                        } else if let Ok(mut cam) = state_mut.commands.get_entity(query_plot.entity)
                        {
                            cam.try_insert(ViewportRect(None));
                        }
                    }
                    Pane::ActionTile(_) => {}
                    Pane::VideoStream(stream) => {
                        if let Ok(mut stream) = state_mut.commands.get_entity(stream.entity) {
                            stream.try_insert(IsTileVisible(active_tiles.contains(tile_id)));
                        }
                    }

                    Pane::Dashboard(dash) => {
                        if let Ok(mut stream) = state_mut.commands.get_entity(dash.entity) {
                            stream.try_insert(IsTileVisible(active_tiles.contains(tile_id)));
                        }
                    }
                }
            }
        })
    }
}

pub fn shortcuts(key_state: Res<LogicalKeyState>, mut ui_state: ResMut<TileState>) {
    // tab switching
    if key_state.pressed(&Key::Control) && key_state.just_pressed(&Key::Tab) {
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
        let offset = if key_state.pressed(&Key::Shift) {
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
