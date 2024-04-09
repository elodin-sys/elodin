use std::collections::BTreeMap;

use bevy::ecs::system::SystemParam;
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::query::QueryData,
    prelude::*,
    render::camera::Viewport,
};
use bevy_egui::{
    egui::{self, Color32, Label, Margin, RichText},
    EguiContexts,
};
use bevy_infinite_grid::InfiniteGrid;
use big_space::GridCell;
use conduit::{
    bevy::{ColumnPayloadMsg, ComponentValueMap, MaxTick, Received, Tick, TimeStep},
    query::MetadataStore,
    well_known::EntityMetadata,
    ComponentId, ControlMsg, EntityId, GraphId,
};
use egui_tiles::TileId;

use crate::{GridHandle, MainCamera};

use self::{
    utils::MarginSides,
    widgets::{button::EImageButton, hierarchy, inspector, modal::modal_graph, timeline},
};

pub mod colors;
pub mod images;
mod theme;
mod tiles;
pub mod utils;
pub mod widgets;

#[derive(Resource, Default)]
pub struct Paused(pub bool);

#[derive(Resource, Default)]
pub struct ShowStats(pub bool);

#[derive(Resource, Default, Debug)]
pub enum SelectedObject {
    #[default]
    None,
    Entity(EntityPair),
    Viewport {
        camera: Entity,
        tile_id: TileId,
    },
    Graph {
        tile_id: TileId,
        label: String,
        graph_id: GraphId,
    },
}

impl SelectedObject {
    pub fn is_entity_selected(&self, id: conduit::EntityId) -> bool {
        matches!(self, SelectedObject::Entity(pair) if pair.conduit == id)
    }

    pub fn is_tile_selected(&self, tile_id: TileId) -> bool {
        self.tile_id() == Some(tile_id)
    }

    pub fn tile_id(&self) -> Option<TileId> {
        match self {
            SelectedObject::None => None,
            SelectedObject::Entity(_) => None,
            SelectedObject::Viewport { tile_id, .. } => Some(*tile_id),
            SelectedObject::Graph { tile_id, .. } => Some(*tile_id),
        }
    }
}

#[derive(Resource, Default)]
pub struct HoveredEntity(pub Option<EntityPair>);

#[derive(Resource, Default)]
pub struct EntityFilter(pub String);

#[derive(Resource, Default)]
pub struct InspectorAnchor(pub Option<egui::Pos2>);

#[derive(Component)]
pub struct ViewportRect(pub Option<egui::Rect>);

#[derive(Clone, Copy, Debug)]
pub struct EntityPair {
    pub bevy: Entity,
    pub conduit: EntityId,
}

pub fn shortcuts(
    mut show_stats: ResMut<ShowStats>,
    mut paused: ResMut<Paused>,
    kbd: Res<ButtonInput<KeyCode>>,
) {
    if kbd.just_pressed(KeyCode::F12) {
        show_stats.0 = !show_stats.0;
    }

    if kbd.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }
}

pub type EntityData<'a> = (
    &'a EntityId,
    Entity,
    &'a mut ComponentValueMap,
    &'a EntityMetadata,
);

pub type EntityDataReadOnly<'a> = (
    &'a EntityId,
    Entity,
    &'a ComponentValueMap,
    &'a EntityMetadata,
);

#[derive(QueryData)]
#[query_data(mutable)]
pub struct CameraQuery {
    entity: Entity,
    camera: &'static mut Camera,
    projection: &'static mut Projection,
    transform: &'static mut Transform,
    global_transform: &'static mut GlobalTransform,
    grid_cell: &'static mut GridCell<i128>,
    parent: Option<&'static Parent>,
    grid_handle: Option<&'static GridHandle>,
    no_propagate_rot: Option<&'static big_space::propagation::NoPropagateRot>,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Paused>()
            .init_resource::<ShowStats>()
            .init_resource::<SelectedObject>()
            .init_resource::<HoveredEntity>()
            .init_resource::<EntityFilter>()
            .init_resource::<GraphsState>()
            .init_resource::<InspectorAnchor>()
            .init_resource::<tiles::TileState>()
            .init_resource::<SidebarState>()
            .init_resource::<FullscreenState>()
            .add_systems(Update, shortcuts)
            .add_systems(Update, render)
            .add_systems(Update, render_timeline.after(render))
            .add_systems(Update, modal_update_graph.after(render))
            .add_systems(Update, tiles::render_tiles.after(render_timeline))
            .add_systems(Update, render_viewport_ui.after(render_timeline))
            .add_systems(Update, set_camera_viewport.after(tiles::render_tiles))
            .add_systems(Update, sync_camera_grid_cell.after(tiles::render_tiles))
            .add_systems(Update, tiles::sync_viewports.after(render_timeline))
            .add_systems(Update, tiles::setup_default_tiles.after(render_timeline));
    }
}

pub fn modal_update_graph(
    mut contexts: EguiContexts,
    inspector_anchor: Res<InspectorAnchor>,
    entities_meta: Query<EntityData>,
    graph_states: ResMut<GraphsState>,
    metadata_store: Res<MetadataStore>,
    window: Query<&Window>,
    images: Local<images::Images>,
) {
    let modal_size = egui::vec2(280.0, 480.0);

    let modal_rect = if let Some(inspector_anchor) = inspector_anchor.0 {
        egui::Rect::from_min_size(
            egui::pos2(inspector_anchor.x - modal_size.x, inspector_anchor.y),
            modal_size,
        )
    } else {
        let window = window.single();
        egui::Rect::from_center_size(
            egui::pos2(
                window.resolution.width() / 2.0,
                window.resolution.height() / 2.0,
            ),
            modal_size,
        )
    };

    if let Some(graph_id) = graph_states.modal_graph {
        let close_icon = contexts.add_image(images.icon_close.clone_weak());

        modal_graph(
            contexts.ctx_mut(),
            modal_rect,
            close_icon,
            entities_meta,
            graph_states,
            metadata_store,
            graph_id,
        );
    }
}

type GraphStateComponent = Vec<(bool, egui::Color32)>;
type GraphStateEntity = BTreeMap<ComponentId, GraphStateComponent>;
type GraphState = BTreeMap<EntityId, GraphStateEntity>;

#[derive(Resource, Default, Clone, Debug)]
pub struct GraphsState {
    modal_graph: Option<GraphId>,
    modal_entity: Option<EntityId>,
    modal_component: Option<ComponentId>,
    graphs: BTreeMap<GraphId, GraphState>,
}

impl GraphsState {
    pub fn get_or_create_graph(&mut self, graph_id: &Option<GraphId>) -> (GraphId, &GraphState) {
        if let Some(graph_id) = graph_id {
            if !self.graphs.contains_key(graph_id) {
                self.graphs.insert(*graph_id, BTreeMap::new());
            }

            (*graph_id, self.graphs.get(graph_id).unwrap())
        } else {
            let new_graph_id = self
                .graphs
                .keys()
                .max()
                .map_or(GraphId(0), |lk| GraphId(lk.0 + 1));
            self.graphs.insert(new_graph_id, BTreeMap::new());

            (new_graph_id, self.graphs.get(&new_graph_id).unwrap())
        }
    }

    pub fn insert_component(
        &mut self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
        component_values: Vec<(bool, egui::Color32)>,
    ) {
        let mut graph = self
            .graphs
            .get(graph_id)
            .map_or(BTreeMap::new(), |graph| graph.clone());

        let mut entity = graph
            .get(entity_id)
            .map_or(BTreeMap::new(), |ec| ec.clone());

        entity.insert(*component_id, component_values);
        graph.insert(*entity_id, entity);

        self.graphs.insert(*graph_id, graph);
    }

    pub fn remove_graph(&mut self, graph_id: &GraphId) {
        self.graphs.remove(graph_id);

        if self.modal_graph == Some(*graph_id) {
            self.modal_graph = None;
            self.modal_entity = None;
            self.modal_component = None;
        }
    }

    pub fn remove_component(
        &mut self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) {
        let mut graph = self
            .graphs
            .get(graph_id)
            .map_or(BTreeMap::new(), |state| state.clone());

        let mut components = graph
            .get(entity_id)
            .map_or(BTreeMap::new(), |ec| ec.clone());

        components.remove(component_id);

        if components.is_empty() {
            graph.remove(entity_id);
        } else {
            graph.insert(*entity_id, components);
        }

        self.graphs.insert(*graph_id, graph);
    }

    pub fn contains_component(
        &self,
        graph_id: &GraphId,
        entity_id: &EntityId,
        component_id: &ComponentId,
    ) -> bool {
        if let Some(graph) = self.graphs.get(graph_id) {
            if let Some(entity) = graph.get(entity_id) {
                return entity.contains_key(component_id);
            }
        }

        false
    }
}

#[derive(Resource)]
pub struct SidebarState {
    pub left_open: bool,
    pub right_open: bool,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            left_open: true,
            right_open: true,
        }
    }
}

#[derive(Resource, Default)]
pub struct FullscreenState(pub bool);

#[derive(SystemParam)]
pub struct RenderArgs<'w, 's> {
    contexts: EguiContexts<'w, 's>,
    entity_filter: ResMut<'w, EntityFilter>,
    selected_object: ResMut<'w, SelectedObject>,
    graphs_state: ResMut<'w, GraphsState>,
    tile_state: ResMut<'w, tiles::TileState>,
    entities: Query<'w, 's, EntityData<'static>>,
    window: Query<'w, 's, &'static Window>,
    images: Local<'s, images::Images>,
    metadata_store: Res<'w, MetadataStore>,
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    commands: Commands<'w, 's>,
    inspector_anchor: ResMut<'w, InspectorAnchor>,
    entity_transform_query: Query<'w, 's, &'static GridCell<i128>, Without<MainCamera>>,
    column_payload_writer: EventWriter<'w, ColumnPayloadMsg>,
    sidebar_state: ResMut<'w, SidebarState>,
    fullscreen_state: ResMut<'w, FullscreenState>,
    grid_visibility: Query<'w, 's, &'static mut Visibility, With<InfiniteGrid>>,
}

pub fn render(args: RenderArgs) {
    let RenderArgs {
        mut contexts,
        entity_filter,
        mut selected_object,
        mut graphs_state,
        mut tile_state,
        mut entities,
        window,
        images,
        metadata_store,
        mut camera_query,
        mut commands,
        mut inspector_anchor,
        entity_transform_query,
        mut column_payload_writer,
        mut sidebar_state,
        mut fullscreen_state,
        mut grid_visibility,
    } = args;
    let Ok(window) = window.get_single() else {
        return;
    };
    let width = window.resolution.width();
    let height = window.resolution.height();

    theme::set_theme(contexts.ctx_mut());

    let icon_search = contexts.add_image(images.icon_search.clone_weak());
    let icon_side_bar_right = contexts.add_image(images.icon_side_bar_right.clone_weak());
    let icon_side_bar_left = contexts.add_image(images.icon_side_bar_left.clone_weak());
    let icon_fullscreen = contexts.add_image(images.icon_fullscreen.clone_weak());
    let icon_exit_fullscreen = contexts.add_image(images.icon_exit_fullscreen.clone_weak());
    let inspector_icons = inspector::InspectorIcons {
        chart: contexts.add_image(images.icon_chart.clone_weak()),
        add: contexts.add_image(images.icon_add.clone_weak()),
        subtract: contexts.add_image(images.icon_subtract.clone_weak()),
    };

    let titlebar_height = if cfg!(target_os = "macos") {
        52.0
    } else {
        16.0
    };
    let traffic_light_offset = if cfg!(target_os = "macos") { 72.0 } else { 0.0 };
    let titlebar_scale = if cfg!(target_os = "macos") { 1.4 } else { 1.3 };
    let titlebar_margin = if cfg!(target_os = "macos") { 8.0 } else { 4.0 };

    egui::TopBottomPanel::top("titlebar")
        .frame(
            egui::Frame {
                fill: colors::PRIMARY_ONYX,
                stroke: egui::Stroke::new(0.0, colors::BORDER_GREY),
                ..Default::default()
            }
            .inner_margin(Margin::same(titlebar_margin).left(16.0).right(16.0)),
        )
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            ui.set_height(titlebar_height - titlebar_margin * 2.0);
            ui.horizontal_centered(|ui| {
                ui.add_space(traffic_light_offset);
                if cfg!(target_family = "wasm") {
                    if ui
                        .add(
                            EImageButton::new(if fullscreen_state.bypass_change_detection().0 {
                                icon_exit_fullscreen
                            } else {
                                icon_fullscreen
                            })
                            .scale(titlebar_scale, titlebar_scale)
                            .bg_color(Color32::TRANSPARENT),
                        )
                        .clicked()
                    {
                        fullscreen_state.0 = !fullscreen_state.0;
                    }
                    ui.add_space(8.0);
                }
                if ui
                    .add(
                        EImageButton::new(icon_side_bar_left)
                            .scale(titlebar_scale, titlebar_scale)
                            .bg_color(Color32::TRANSPARENT),
                    )
                    .clicked()
                {
                    sidebar_state.left_open = !sidebar_state.left_open;
                };

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            EImageButton::new(icon_side_bar_right)
                                .scale(titlebar_scale, titlebar_scale)
                                .bg_color(Color32::TRANSPARENT),
                        )
                        .clicked()
                    {
                        sidebar_state.right_open = !sidebar_state.right_open;
                    };
                });
            });
        });

    if width * 0.75 > height {
        egui::SidePanel::new(egui::panel::Side::Left, "outline_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                inner_margin: Margin::same(4.0),
                ..Default::default()
            })
            .min_width(width * 0.15)
            .default_width(width * 0.20)
            .max_width(width * 0.35)
            .show_animated(contexts.ctx_mut(), sidebar_state.left_open, |ui| {
                let search_text = entity_filter.0.clone();

                hierarchy::header(ui, entity_filter, icon_search, false);

                hierarchy::entity_list(ui, &entities, &mut selected_object, &search_text);
            });

        egui::SidePanel::new(egui::panel::Side::Right, "inspector_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                ..Default::default()
            })
            .min_width(width * 0.15)
            .default_width(width * 0.25)
            .max_width(width * 0.35)
            .show_animated(contexts.ctx_mut(), sidebar_state.right_open, |ui| {
                inspector_anchor.0 = Some(ui.max_rect().min);
                inspector::inspector(
                    ui,
                    selected_object.as_ref(),
                    &mut entities,
                    &metadata_store,
                    &mut camera_query,
                    &mut commands,
                    &entity_transform_query,
                    &mut graphs_state,
                    &mut tile_state,
                    inspector_icons,
                    &mut column_payload_writer,
                    &mut grid_visibility,
                );
            });
    } else {
        egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "section_bottom")
            .resizable(true)
            .frame(egui::Frame::default())
            .default_height(200.0)
            .max_height(width * 0.5)
            .show(contexts.ctx_mut(), |ui| {
                let outline = egui::SidePanel::new(egui::panel::Side::Left, "outline_bottom")
                    .resizable(true)
                    .frame(egui::Frame {
                        fill: colors::PRIMARY_SMOKE,
                        stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                        inner_margin: Margin::same(4.0),
                        ..Default::default()
                    })
                    .min_width(width * 0.25)
                    .default_width(width * 0.4)
                    .max_width(width * 0.75)
                    .show_animated_inside(ui, sidebar_state.left_open, |ui| {
                        let search_text = entity_filter.0.clone();

                        hierarchy::header(ui, entity_filter, icon_search, true);

                        hierarchy::entity_list(ui, &entities, &mut selected_object, &search_text);

                        ui.allocate_space(ui.available_size());
                    });
                let outline_width = outline.map(|o| o.response.rect.width()).unwrap_or(0.0);
                egui::SidePanel::new(egui::panel::Side::Right, "inspector_bottom")
                    .resizable(false)
                    .frame(egui::Frame {
                        fill: colors::PRIMARY_SMOKE,
                        ..Default::default()
                    })
                    .exact_width(width - outline_width)
                    .show_animated_inside(ui, sidebar_state.right_open, |ui| {
                        inspector_anchor.0 = None;
                        inspector::inspector(
                            ui,
                            selected_object.as_ref(),
                            &mut entities,
                            &metadata_store,
                            &mut camera_query,
                            &mut commands,
                            &entity_transform_query,
                            &mut graphs_state,
                            &mut tile_state,
                            inspector_icons,
                            &mut column_payload_writer,
                            &mut grid_visibility,
                        );
                    });
            });
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_timeline(
    mut event: EventWriter<ControlMsg>,
    mut contexts: EguiContexts,
    mut paused: ResMut<Paused>,
    mut tick: ResMut<Tick>,
    max_tick: Res<MaxTick>,
    tick_time: Res<TimeStep>,
    images: Local<images::Images>,
) {
    theme::set_theme(contexts.ctx_mut());

    let timeline_icons = timeline::TimelineIcons {
        jump_to_start: contexts.add_image(images.icon_jump_to_start.clone_weak()),
        jump_to_end: contexts.add_image(images.icon_jump_to_end.clone_weak()),
        frame_forward: contexts.add_image(images.icon_frame_forward.clone_weak()),
        frame_back: contexts.add_image(images.icon_frame_back.clone_weak()),
        play: contexts.add_image(images.icon_play.clone_weak()),
        pause: contexts.add_image(images.icon_pause.clone_weak()),
        handle: contexts.add_image(images.icon_scrub.clone_weak()),
    };

    let sim_fps = 1.0 / tick_time.0.as_secs_f64();

    egui::TopBottomPanel::bottom("timeline")
        .frame(egui::Frame {
            fill: colors::PRIMARY_SMOKE,
            stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
            ..Default::default()
        })
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            timeline::timeline_area(
                ui,
                &mut paused,
                &max_tick,
                &mut tick,
                sim_fps,
                &mut event,
                timeline_icons,
            );
        });
}

#[allow(clippy::too_many_arguments)]
pub fn render_viewport_ui(
    mut contexts: EguiContexts,
    window: Query<&Window>,
    entities_meta: Query<EntityData>,
    show_stats: Res<ShowStats>,
    tick_time: Res<TimeStep>,
    diagnostics: Res<DiagnosticsStore>,
    hovered_entity: Res<HoveredEntity>,
) {
    let Ok(window) = window.get_single() else {
        return;
    };

    let hovered_entity_meta = if let Some(hovered_entity_pair) = hovered_entity.0 {
        entities_meta
            .iter()
            .find(|(id, _, _, _)| hovered_entity_pair.conduit == **id)
            .map(|(_, _, _, metadata)| metadata.to_owned())
    } else {
        None
    };

    if let Some(hovered_entity_meta) = hovered_entity_meta {
        contexts
            .ctx_mut()
            .set_cursor_icon(egui::CursorIcon::PointingHand);

        if let Some(cursor_pos) = window.cursor_position() {
            let offset = 16.0;
            let window_pos = egui::pos2(cursor_pos.x + offset, cursor_pos.y + offset);

            egui::Window::new("hovered_entity")
                .title_bar(false)
                .resizable(false)
                .frame(egui::Frame {
                    fill: colors::with_opacity(colors::PRIMARY_SMOKE, 0.5),
                    stroke: egui::Stroke::new(1.0, colors::with_opacity(colors::WHITE, 0.5)),
                    inner_margin: egui::Margin::symmetric(16.0, 8.0),
                    ..Default::default()
                })
                .fixed_pos(window_pos)
                .show(contexts.ctx_mut(), |ui| {
                    ui.add(Label::new(
                        RichText::new(hovered_entity_meta.name).color(Color32::WHITE),
                    ));
                });
        }
    }

    if show_stats.0 {
        let sim_fps = 1.0 / tick_time.0.as_secs_f64();

        let viewport_left_top = contexts.ctx_mut().available_rect().left_top();
        let viewport_margins = egui::vec2(16.0, 16.0);

        egui::Window::new("stats")
            .title_bar(false)
            .resizable(false)
            .frame(egui::Frame::default())
            .fixed_pos(viewport_left_top + viewport_margins)
            .show(contexts.ctx_mut(), |ui| {
                let render_fps_str = diagnostics
                    .get(&FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                    .map_or(" N/A".to_string(), |value| format!("{value:>4.0}"));

                ui.add(Label::new(
                    RichText::new(format!("FPS [VIEW]: {render_fps_str}")).color(Color32::WHITE),
                ));

                ui.add(Label::new(
                    RichText::new(format!("FPS [SIM]: {sim_fps:>4.0}")).color(Color32::WHITE),
                ));
            });
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
struct CameraViewportQuery {
    camera: &'static mut Camera,
    viewport_rect: &'static ViewportRect,
}

fn set_camera_viewport(
    window: Query<&Window>,
    egui_settings: Res<bevy_egui::EguiSettings>,
    mut main_camera_query: Query<CameraViewportQuery, With<MainCamera>>,
) {
    for CameraViewportQueryItem {
        mut camera,
        viewport_rect,
    } in main_camera_query.iter_mut()
    {
        let Some(available_rect) = viewport_rect.0 else {
            camera.order = 0;
            camera.is_active = false;
            camera.viewport = Some(Viewport {
                physical_position: UVec2::new(0, 0),
                physical_size: UVec2::new(1, 1),
                depth: 0.0..1.0,
            });

            continue;
        };
        camera.is_active = true;
        camera.order = 1;
        let Some(window) = window.iter().next() else {
            continue;
        };
        let scale_factor = window.scale_factor() * egui_settings.scale_factor;
        let viewport_pos = available_rect.left_top().to_vec2() * scale_factor;
        let viewport_size = available_rect.size() * scale_factor;
        if available_rect.size().x > window.width() || available_rect.size().y > window.height() {
            return;
        }
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(viewport_pos.x as u32, viewport_pos.y as u32),
            physical_size: UVec2::new(viewport_size.x as u32, viewport_size.y as u32),
            depth: 0.0..1.0,
        });
    }
}

fn sync_camera_grid_cell(
    mut query: Query<(Option<&Parent>, &mut GridCell<i128>), With<MainCamera>>,
    entity_transform_query: Query<&GridCell<i128>, (With<Received>, Without<MainCamera>)>,
) {
    for (parent, mut grid_cell) in query.iter_mut() {
        if let Some(parent) = parent {
            if let Ok(entity_cell) = entity_transform_query.get(parent.get()) {
                *grid_cell = *entity_cell;
            }
        }
    }
}
