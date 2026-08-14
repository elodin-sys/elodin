use std::collections::HashMap;

use crate::{
    GridHandle, TimeRangeBehavior,
    object_3d::Object3DState,
    plugins::render_layer_alloc::VIEW_CUBE_RENDER_LAYERS,
    ui::{
        HdrEnabled, actions, colors,
        colors::EColor,
        gauges, inspector, monitor, plot, query_plot, query_table,
        tiles::{self, Pane},
        timeline::{TelemetryMode, TimelineSettings},
        window::compute_window_title,
    },
    vector_arrow::ViewportArrow,
};
use bevy::{
    camera::visibility::RenderLayers, ecs::system::SystemParam, prelude::*, window::PrimaryWindow,
};
use bevy_geo_frames::{GeoFrame, GeoPosition};
use egui_tiles::{Tile, TileId};
use impeller2_bevy::ComponentMetadataRegistry;
use impeller2_wkt::{
    ActionPane, ComponentMonitor, ComponentPath, GeoPositionGauge, HorizonGauge, Line3d,
    OrientationGauge, Panel, Schematic, SchematicElem, Split, VectorArrow3d,
    VideoStream as WktVideoStream, Viewport, WindowSchematic, WorldMesh,
};

pub mod bindings;
pub use bindings::SchematicBindings;
pub mod tree;
pub use tree::*;
mod load;
pub use crate::plugins::kdl_document::{
    CurrentDocument, DocumentCleared, DocumentLoadFailed, DocumentLoaded, DocumentReloaded,
    InitialKdlPath, KdlDocumentSet, OpenDocumentRequest, SchematicDocumentAsset, SchematicWindow,
    WindowDocumentSave, apply_initial_kdl_path, sync_document_from_config,
};
pub use load::*;

#[derive(Resource, Debug, Clone, Deref, DerefMut)]
pub struct CurrentSchematic(pub Schematic);

#[derive(Debug, Clone)]
pub struct WindowSchematicEntry {
    pub window_id: tiles::WindowId,
    pub file_name: String,
    pub title: Option<String>,
    pub schematic: Schematic,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct CurrentWindowSchematics(pub Vec<WindowSchematicEntry>);

#[derive(SystemParam)]
pub struct SchematicParam<'w, 's> {
    pub query_tables: Query<'w, 's, &'static query_table::QueryTableData>,
    pub monitors: Query<'w, 's, &'static monitor::MonitorData>,
    pub geo_position_gauges: Query<'w, 's, &'static gauges::GeoPositionGaugeData>,
    pub orientation_gauges: Query<'w, 's, &'static gauges::OrientationGaugeData>,
    pub horizon_gauges: Query<'w, 's, &'static gauges::HorizonGaugeData>,
    pub eql_bindings: Query<'w, 's, &'static gauges::EqlBinding>,
    pub action_tiles: Query<'w, 's, &'static actions::ActionTile>,
    pub graph_states: Query<'w, 's, &'static plot::GraphState>,
    pub query_plots: Query<'w, 's, &'static query_plot::QueryPlotData>,
    pub viewports: Query<'w, 's, &'static inspector::viewport::Viewport>,
    pub projections: Query<'w, 's, &'static Projection>,
    pub viewport_configs: Query<'w, 's, &'static tiles::ViewportConfig>,
    pub camera_grids: Query<'w, 's, &'static GridHandle>,
    pub camera_layers: Query<'w, 's, &'static RenderLayers>,
    pub objects_3d: Query<'w, 's, (Entity, &'static Object3DState)>,
    pub lines_3d: Query<'w, 's, (Entity, &'static Line3d)>,
    pub world_meshes: Query<'w, 's, (Entity, &'static WorldMesh)>,
    pub vector_arrows: Query<
        'w,
        's,
        (
            Entity,
            &'static VectorArrow3d,
            Option<&'static ViewportArrow>,
        ),
    >,
    pub windows_state: Query<'w, 's, (&'static tiles::WindowState, &'static tiles::WindowId)>,
    pub primary_window: Single<'w, 's, Entity, With<PrimaryWindow>>,
    pub current_document: Res<'w, CurrentDocument>,
    pub video_streams: Query<'w, 's, &'static super::video_stream::VideoStream>,
    pub log_streams: Query<'w, 's, &'static super::log_stream::LogStreamState>,
    pub hdr_enabled: Res<'w, HdrEnabled>,
    pub timeline_settings: Res<'w, TimelineSettings>,
    pub time_range_behavior: Res<'w, TimeRangeBehavior>,
    pub telemetry_mode: Res<'w, TelemetryMode>,
    pub metadata: Res<'w, ComponentMetadataRegistry>,
    pub geo_positions: Query<'w, 's, &'static GeoPosition>,
    pub coordinate: Res<'w, crate::Coordinate>,
    pub geo_context: Res<'w, bevy_geo_frames::GeoContext>,
}

impl SchematicParam<'_, '_> {
    fn export_pane_name(&self, pane: &Pane) -> Option<String> {
        match pane {
            Pane::Viewport(viewport) => Some(viewport.name.clone()),
            Pane::Graph(graph) => self
                .graph_states
                .get(graph.id)
                .ok()
                .map(|state| state.label.clone()),
            Pane::Monitor(monitor) => Some(monitor.name.clone()),
            Pane::GeoPositionGauge(gauge)
            | Pane::OrientationGauge(gauge)
            | Pane::HorizonGauge(gauge) => Some(gauge.name.clone()),
            Pane::QueryTable(table) => Some(table.name.clone()),
            Pane::QueryPlot(plot) => self
                .graph_states
                .get(plot.entity)
                .ok()
                .map(|state| state.label.clone())
                .or_else(|| {
                    self.query_plots
                        .get(plot.entity)
                        .ok()
                        .map(|data| data.data.name.clone())
                }),
            Pane::ActionTile(action) => Some(action.name.clone()),
            Pane::SchematicTree(pane) => Some(pane.name.clone()),
            Pane::DataOverview(pane) => Some(pane.name.clone()),
            Pane::VideoStream(pane) => Some(pane.name.clone()),
            Pane::SensorView(pane) => Some(pane.name.clone()),
            Pane::LogStream(pane) => Some(pane.name.clone()),
        }
    }

    fn root_panels_from_state(
        &self,
        state: &tiles::TileState,
        bindings: &mut SchematicBindings,
    ) -> Vec<Panel> {
        let Some(root_id) = state.tree.root() else {
            return Vec::new();
        };

        match self.get_panel_from_state(state, root_id, bindings) {
            Some(Panel::Tabs(tabs)) => vec![Panel::Tabs(tabs)],
            Some(panel) => vec![panel],
            None => Vec::new(),
        }
    }

    pub fn get_panel(&self, tile_id: TileId, bindings: &mut SchematicBindings) -> Option<Panel> {
        self.windows_state
            .get(*self.primary_window)
            .ok()
            .and_then(|(window_state, _)| {
                self.get_panel_from_state(&window_state.tile_state, tile_id, bindings)
            })
    }

    pub fn get_panel_from_state(
        &self,
        state: &tiles::TileState,
        tile_id: TileId,
        bindings: &mut SchematicBindings,
    ) -> Option<Panel> {
        let tiles = &state.tree.tiles;
        let tile = tiles.get(tile_id)?;

        match tile {
            Tile::Pane(pane) => {
                let pane_name = self.export_pane_name(pane);
                match pane {
                    // ---- Viewport ----
                    Pane::Viewport(viewport) => {
                        let cam_entity = viewport.camera?;
                        let viewport_data = self.viewports.get(cam_entity).ok()?;
                        let fov = self
                            .projections
                            .get(cam_entity)
                            .ok()
                            .and_then(|projection| match projection {
                                Projection::Perspective(perspective) => {
                                    Some(perspective.fov.to_degrees())
                                }
                                _ => None,
                            })
                            .unwrap_or(45.0);

                        let vp_config = self.viewport_configs.get(cam_entity).ok();
                        let near = vp_config
                            .and_then(|c| c.configured_near)
                            .filter(|near| *near > 0.0);
                        let far = vp_config
                            .and_then(|c| c.configured_far)
                            .filter(|far| *far > 0.0);
                        let aspect = vp_config.and_then(|c| c.aspect);

                        let mut show_grid = false;
                        if let Ok(grid_handle) = self.camera_grids.get(cam_entity)
                            && let Ok(render_layers) = self.camera_layers.get(cam_entity)
                        {
                            show_grid =
                                render_layers.intersects(&RenderLayers::layer(grid_handle.layer));
                        }

                        let show_arrows = vp_config.map(|c| c.show_arrows).unwrap_or(true);
                        let create_frustum = vp_config.map(|c| c.create_frustum).unwrap_or(false);
                        let show_frustums = vp_config.map(|c| c.show_frustums).unwrap_or(false);
                        let frustums_color = vp_config
                            .map(|c| c.frustums_color)
                            .unwrap_or_else(impeller2_wkt::default_viewport_frustums_color);
                        let projection_color = vp_config
                            .map(|c| c.projection_color)
                            .unwrap_or_else(impeller2_wkt::default_viewport_projection_color);
                        let frustums_thickness = vp_config
                            .map(|c| c.frustums_thickness)
                            .unwrap_or_else(impeller2_wkt::default_viewport_frustums_thickness);
                        let show_view_cube = viewport.view_cube_layer.is_some();
                        let view_cube_frame = viewport.view_cube_layer.and_then(|layer| {
                            VIEW_CUBE_RENDER_LAYERS
                                .iter()
                                .find(|(_, l)| *l == layer)
                                .map(|(frame, _)| *frame)
                        });

                        let local_arrows: Vec<VectorArrow3d> = self
                            .vector_arrows
                            .iter()
                            .filter(|(_, _, viewport_arrow)| {
                                if let Some(viewport_arrow) = viewport_arrow {
                                    viewport_arrow.camera == cam_entity
                                } else {
                                    false
                                }
                            })
                            .map(|(_, arrow, _)| arrow.clone())
                            .collect();
                        let frame: Option<GeoFrame> = self
                            .geo_positions
                            .get(cam_entity)
                            .map(|geo_pos| geo_pos.0)
                            .ok();

                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, cam_entity);
                        Some(Panel::Viewport(Viewport {
                            fov,
                            near,
                            far,
                            aspect,
                            active: false,
                            show_grid,
                            show_arrows,
                            create_frustum,
                            show_frustums,
                            frustums_color,
                            projection_color,
                            frustums_thickness,
                            show_view_cube,
                            view_cube_frame,
                            // ViewportConfig does not yet track `effects`; default
                            // on so schematic dumps keep thruster particles visible.
                            effects: true,
                            hdr: self.hdr_enabled.0,
                            bloom: None,
                            // Like bloom, exposure is not read back from the
                            // live camera; hand-authored ev100 survives via
                            // CurrentSchematic, not this dump.
                            ev100: None,
                            name: pane_name,
                            pos: Some(viewport_data.pos.eql.clone()),
                            look_at: Some(viewport_data.look_at.eql.clone()),
                            up: (!viewport_data.up.eql.is_empty())
                                .then(|| viewport_data.up.eql.clone()),
                            smoothing: viewport_data.smoothing,
                            local_arrows,
                            frame,
                            node_id,
                        }))
                    }

                    // ---- Graph ----
                    Pane::Graph(graph) => {
                        let graph_state = self.graph_states.get(graph.id).ok()?;
                        let mut eql = String::new();
                        let mut colors: Vec<impeller2_wkt::Color> = vec![];
                        let mut parts: Vec<String> = Vec::new();

                        for (component_path, component_values) in &graph_state.components {
                            for (index, (enabled, color)) in component_values.iter().enumerate() {
                                if !*enabled {
                                    continue;
                                }
                                parts.push(component_expr(component_path, index, &self.metadata));
                                colors.push(impeller2_wkt::Color::from_color32(*color));
                            }
                        }

                        if !parts.is_empty() {
                            eql = parts.join(", ");
                        } else if !graph_state.label.is_empty() {
                            eql = graph_state.label.clone();
                        }

                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, graph.id);
                        Some(Panel::Graph(impeller2_wkt::Graph {
                            eql,
                            name: pane_name,
                            graph_type: graph_state.graph_type,
                            locked: graph_state.locked,
                            auto_y_range: graph_state.auto_y_range,
                            y_range: graph_state.y_range.clone(),
                            node_id,
                            colors,
                        }))
                    }

                    Pane::Monitor(monitor) => {
                        let monitor_data = self.monitors.get(monitor.entity).ok()?;
                        Some(Panel::ComponentMonitor(ComponentMonitor {
                            component_name: monitor_data.component_name.clone(),
                            name: pane_name,
                        }))
                    }

                    Pane::GeoPositionGauge(gauge) => {
                        let data = self.geo_position_gauges.get(gauge.entity).ok()?;
                        let binding = self.eql_bindings.get(gauge.entity).ok()?;
                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, gauge.entity);
                        Some(Panel::GeoPositionGauge(GeoPositionGauge {
                            eql: binding.eql.clone(),
                            // Keep None so save omits `source=` and inheritance
                            // from `coordinate` survives a round-trip.
                            source: data.source,
                            display: data.display,
                            name: pane_name,
                            node_id,
                        }))
                    }

                    Pane::OrientationGauge(gauge) => {
                        let data = self.orientation_gauges.get(gauge.entity).ok()?;
                        let binding = self.eql_bindings.get(gauge.entity).ok()?;
                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, gauge.entity);
                        Some(Panel::OrientationGauge(OrientationGauge {
                            eql: binding.eql.clone(),
                            source: data.source,
                            display: data.display,
                            // None when identity so the default stays implicit.
                            reference: data.reference_kdl(),
                            name: pane_name,
                            node_id,
                        }))
                    }

                    Pane::HorizonGauge(gauge) => {
                        let data = self.horizon_gauges.get(gauge.entity).ok()?;
                        let binding = self.eql_bindings.get(gauge.entity).ok()?;
                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, gauge.entity);
                        Some(Panel::HorizonGauge(HorizonGauge {
                            eql: binding.eql.clone(),
                            source: data.source,
                            // None when identity so the default stays implicit.
                            reference: data.reference_kdl(),
                            name: pane_name,
                            node_id,
                        }))
                    }

                    Pane::QueryTable(query_table) => {
                        let query_table_data = self.query_tables.get(query_table.entity).ok()?;
                        let mut data = query_table_data.data.clone();
                        data.name = pane_name;
                        Some(Panel::QueryTable(data))
                    }

                    Pane::QueryPlot(plot) => {
                        let query_plot_data = self.query_plots.get(plot.entity).ok()?;
                        let node_id = impeller2_wkt::NodeId::next();
                        bindings.bind_ephemeral(node_id, plot.entity);
                        let mut qp = query_plot_data.data.clone();
                        qp.node_id = node_id;
                        if let Some(name) = pane_name {
                            qp.name = name;
                        }
                        Some(Panel::QueryPlot(qp))
                    }

                    Pane::ActionTile(action) => {
                        let action_tile = self.action_tiles.get(action.entity).ok()?;
                        Some(Panel::ActionPane(ActionPane {
                            name: pane_name.unwrap_or_else(|| action_tile.button_name.clone()),
                            lua: action_tile.lua.clone(),
                        }))
                    }

                    Pane::VideoStream(video_pane) => {
                        let video_stream = self.video_streams.get(video_pane.entity).ok()?;
                        Some(Panel::VideoStream(WktVideoStream {
                            msg_name: video_stream.msg_name.clone(),
                            name: pane_name,
                        }))
                    }
                    Pane::SensorView(sv_pane) => {
                        let video_stream = self.video_streams.get(sv_pane.entity).ok()?;
                        Some(Panel::SensorView(impeller2_wkt::SensorView {
                            msg_name: video_stream.msg_name.clone(),
                            name: pane_name,
                        }))
                    }
                    Pane::LogStream(ls_pane) => {
                        let log_state = self.log_streams.get(ls_pane.entity).ok()?;
                        Some(Panel::LogStream(impeller2_wkt::LogStream {
                            msg_name: log_state.msg_name.clone(),
                            name: pane_name,
                        }))
                    }
                    Pane::DataOverview(_) => Some(Panel::DataOverview(pane_name)),

                    // Structural panes
                    Pane::SchematicTree(_) => Some(Panel::SchematicTree(pane_name)),
                }
            }

            // ---- Containers ----
            Tile::Container(container) => match container {
                egui_tiles::Container::Tabs(t) => {
                    let mut tabs = vec![];
                    for child_id in &t.children {
                        if let Some(tab) = self.get_panel_from_state(state, *child_id, bindings) {
                            tabs.push(tab)
                        }
                    }
                    match tabs.len() {
                        0 => None,
                        1 => Some(tabs.remove(0)),
                        _ => Some(Panel::Tabs(tabs)),
                    }
                }

                egui_tiles::Container::Linear(linear) => {
                    let mut panels = Vec::new();
                    let mut shares = HashMap::new();
                    let name = state.get_container_title(tile_id).map(|s| s.to_string());

                    for child_id in &linear.children {
                        if let Some(panel) = self.get_panel_from_state(state, *child_id, bindings) {
                            if let Some((_, share)) =
                                linear.shares.iter().find(|(id, _)| *id == child_id)
                            {
                                shares.insert(panels.len(), *share);
                            }
                            panels.push(panel);
                        }
                    }

                    match panels.len() {
                        0 => None,
                        1 if name.is_none() => Some(panels.remove(0)),
                        _ => {
                            let split = Split {
                                panels,
                                shares,
                                active: false,
                                name,
                            };
                            match linear.dir {
                                egui_tiles::LinearDir::Horizontal => Some(Panel::HSplit(split)),
                                egui_tiles::LinearDir::Vertical => Some(Panel::VSplit(split)),
                            }
                        }
                    }
                }

                _ => None,
            },
        }
    }
}

pub fn tiles_to_schematic(
    param: SchematicParam,
    mut schematic: ResMut<CurrentSchematic>,
    mut window_schematics: ResMut<CurrentWindowSchematics>,
    mut bindings: ResMut<SchematicBindings>,
) {
    schematic.elems.clear();
    schematic.frame = param.coordinate.0;

    // Persist the GeoContext origin (radians -> degrees), omitting the
    // default origin so plain schematics stay unchanged.
    let origin = &param.geo_context.origin;
    let default_origin = bevy_geo_frames::GeoOrigin::default();
    schematic.origin = ((origin.latitude, origin.longitude, origin.altitude)
        != (
            default_origin.latitude,
            default_origin.longitude,
            default_origin.altitude,
        ))
        .then(|| impeller2_wkt::GeoOriginConfig {
            latitude: origin.latitude.to_degrees(),
            longitude: origin.longitude.to_degrees(),
            altitude: origin.altitude,
        });
    // The ellipsoid rides along with the origin: without it a lunar schematic
    // would reload as WGS84, breaking ECEF verticals and LLA conversions.
    schematic.body = load::ellipsoid_body(origin.ellipsoid);
    bindings.clear_ephemeral();

    if let Some(root_panels) =
        param
            .windows_state
            .get(*param.primary_window)
            .ok()
            .map(|(window_state, _)| {
                param.root_panels_from_state(&window_state.tile_state, &mut bindings)
            })
    {
        schematic
            .elems
            .extend(root_panels.into_iter().map(SchematicElem::Panel))
    }
    schematic
        .elems
        .extend(param.objects_3d.iter().map(|(entity, o)| {
            let mut obj = o.data.clone();
            let node_id = impeller2_wkt::NodeId::next();
            bindings.bind_ephemeral(node_id, entity);
            obj.node_id = node_id;
            SchematicElem::Object3d(obj)
        }));
    schematic
        .elems
        .extend(param.lines_3d.iter().map(|(entity, line)| {
            let mut l = line.clone();
            let node_id = impeller2_wkt::NodeId::next();
            bindings.bind_ephemeral(node_id, entity);
            l.node_id = node_id;
            SchematicElem::Line3d(l)
        }));
    schematic.elems.extend(
        param
            .vector_arrows
            .iter()
            .filter(|(_, _, viewport_arrow)| viewport_arrow.is_none())
            .map(|(entity, arrow, _)| {
                let mut a = arrow.clone();
                let node_id = impeller2_wkt::NodeId::next();
                bindings.bind_ephemeral(node_id, entity);
                a.node_id = node_id;
                SchematicElem::VectorArrow(a)
            }),
    );

    schematic
        .elems
        .extend(param.world_meshes.iter().map(|(entity, world_mesh)| {
            let mut wm = world_mesh.clone();
            let node_id = impeller2_wkt::NodeId::next();
            bindings.bind_ephemeral(node_id, entity);
            wm.node_id = node_id;
            SchematicElem::WorldMesh(wm)
        }));

    window_schematics.0.clear();
    let mut window_elems = Vec::new();
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    // A window loaded from the DB keeps its stored asset key on save, so
    // ingest-keyed sub-schematics (e.g. `windows/detail.kdl`, kept at their
    // original keys by `resolve_stored_asset_key`) are overwritten in place
    // instead of being re-keyed under `schematics/<stem>.kdl` — which would
    // strand the stored reference for other consumers. Seed the generated-name
    // counter with the stems of preserved `schematics/<stem>.kdl` keys so a
    // freshly created window can never claim the same key.
    for (state, window_id) in &param.windows_state {
        if window_id.is_primary() {
            continue;
        }
        if let Some(stem) = preserved_window_key(state)
            .as_deref()
            .and_then(schematics_key_stem)
        {
            name_counts.entry(stem.to_string()).or_insert(1);
        }
    }
    for (state, window_id) in &param.windows_state {
        let mut file_name: Option<String> = None;
        let mut window_title: Option<String> = None;

        if !window_id.is_primary() {
            let computed_title = compute_window_title(state);
            if computed_title != "Panel" {
                window_title = Some(computed_title);
            }
            let name = preserved_window_key(state).unwrap_or_else(|| {
                let base_stem = preferred_window_stem(state);
                let unique_stem = ensure_unique_stem(&mut name_counts, &base_stem);
                format!("{unique_stem}.kdl")
            });
            file_name = Some(name);

            let mut win_schematic = Schematic::default();
            win_schematic.elems.extend(
                param
                    .root_panels_from_state(&state.tile_state, &mut bindings)
                    .into_iter()
                    .map(SchematicElem::Panel),
            );
            if let Some(file_name) = &file_name {
                window_schematics.0.push(WindowSchematicEntry {
                    window_id: *window_id,
                    file_name: file_name.clone(),
                    title: window_title.clone(),
                    schematic: win_schematic,
                });
            }
        }

        window_elems.push(SchematicElem::Window(WindowSchematic {
            title: window_title.clone(),
            path: file_name,
            screen: state.descriptor.screen.map(|idx| idx as u32),
            screen_rect: state.descriptor.screen_rect,
        }));
    }

    schematic.elems.extend(window_elems);
    let mut timeline: impeller2_wkt::TimelineConfig = (*param.timeline_settings).into();
    timeline.range = param.time_range_behavior.to_schematic_range();
    schematic.timeline = Some(timeline);
    schematic.telemetry_mode = param.telemetry_mode.0;
    if let Ok((state, _)) = param.windows_state.get(*param.primary_window)
        && let Some(mode) = state.descriptor.mode.clone()
    {
        let selection = colors::current_selection();
        schematic.theme = Some(impeller2_wkt::ThemeConfig {
            mode: Some(mode),
            scheme: Some(selection.scheme),
        });
    }
}

pub struct SchematicPlugin;

impl Plugin for SchematicPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CurrentSchematic(Default::default()))
            .insert_resource(CurrentWindowSchematics::default())
            .init_resource::<SchematicBindings>()
            .init_resource::<load::PendingWindowSchematics>()
            .init_resource::<load::PendingDataOverview>()
            .add_plugins(load::plugin)
            .add_systems(PostUpdate, tiles_to_schematic)
            .add_systems(
                PostUpdate,
                apply_initial_kdl_path
                    .pipe(sync_document_from_config)
                    .before(tiles_to_schematic),
            )
            .add_systems(
                PreUpdate,
                (
                    load::apply_document_cleared,
                    load::retry_pending_data_overview,
                    load::retry_pending_object_3d_spawns,
                    load::apply_document_loaded.before(crate::ui::sync_windows),
                    load::apply_document_reloaded.before(crate::ui::sync_windows),
                    load::apply_pending_window_schematics.before(crate::ui::sync_windows),
                    load::show_document_command_failures,
                    load::show_document_load_failures,
                )
                    .after(KdlDocumentSet::AssetEvents),
            );
    }
}

/// The DB asset key a window was loaded from (a `db:<key>` descriptor path),
/// if any. Preserved on save so the window overwrites its stored asset instead
/// of forking to a newly generated `schematics/<stem>.kdl` key.
fn preserved_window_key(state: &tiles::WindowState) -> Option<String> {
    let path = state.descriptor.path.as_ref()?.to_str()?;
    impeller2_kdl::db_asset_name(path)
}

/// The stem of a single-level `schematics/<stem>.kdl` key, i.e. the namespace
/// generated window names are keyed into.
fn schematics_key_stem(key: &str) -> Option<&str> {
    let stem = key.strip_prefix("schematics/")?.strip_suffix(".kdl")?;
    (!stem.is_empty() && !stem.contains('/')).then_some(stem)
}

fn preferred_window_stem(state: &tiles::WindowState) -> String {
    if let Some(title) = state.descriptor.title.as_deref() {
        let stem = sanitize_to_stem(title);
        if !stem.is_empty() {
            return stem;
        }
    }
    if let Some(stem) = state
        .descriptor
        .path
        .as_ref()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
    {
        let stem = sanitize_to_stem(stem);
        if !stem.is_empty() {
            return stem;
        }
    }
    "window".to_string()
}

pub fn sanitize_to_stem(input: &str) -> String {
    let mut stem = String::new();
    let mut last_dash = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            stem.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if (matches!(ch, '-' | '_') || ch.is_whitespace()) && !last_dash && !stem.is_empty()
        {
            stem.push('-');
            last_dash = true;
        }
    }
    stem.trim_matches('-').to_string()
}

fn ensure_unique_stem(counts: &mut HashMap<String, usize>, stem: &str) -> String {
    let base = if stem.is_empty() { "window" } else { stem };
    let entry = counts.entry(base.to_string()).or_insert(0);
    let current = *entry;
    *entry += 1;
    if current == 0 {
        base.to_string()
    } else {
        format!("{base}-{}", current + 1)
    }
}

fn component_expr(
    component_path: &ComponentPath,
    index: usize,
    metadata: &ComponentMetadataRegistry,
) -> String {
    // Full component path string (e.g., "drone.rate_pid_state")
    let base = component_path.to_string();

    if let Some(meta) = metadata.0.get(&component_path.id)
        && let Some(name) = meta
            .element_names()
            .split(',')
            .map(|s| s.trim())
            .nth(index)
            .filter(|s| !s.is_empty())
    {
        // If element name itself contains dots or non-identifier chars,
        // prefer index notation for compatibility with the EQL loader.
        let simple = name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if simple && !name.contains('.') {
            if base.ends_with(name) {
                return base.to_string();
            }
            return format!("{base}.{name}");
        } else {
            return format!("{base}[{index}]");
        }
    }

    format!("{base}[{index}]")
}

/// The `value * scale + offset` an expression applies to a component element.
///
/// Consumers that plot a bare element (2D graphs, monitors) ignore this;
/// `line_3d` applies it so `ball.pos[0] + 1.5` offsets the trail rather than
/// silently rendering the unshifted component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementAffine {
    pub scale: f64,
    pub offset: f64,
}

impl Default for ElementAffine {
    fn default() -> Self {
        Self {
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl ElementAffine {
    pub fn apply(self, value: f64) -> f64 {
        value * self.scale + self.offset
    }

    pub fn is_identity(self) -> bool {
        self == Self::default()
    }

    fn scaled(self, k: f64) -> Self {
        Self {
            scale: self.scale * k,
            offset: self.offset * k,
        }
    }

    fn shifted(self, k: f64) -> Self {
        Self {
            scale: self.scale,
            offset: self.offset + k,
        }
    }
}

/// Fold an expression made only of float literals into its value.
fn const_value(expr: &eql::Expr) -> Option<f64> {
    match expr {
        eql::Expr::FloatLiteral(f) => Some(*f),
        eql::Expr::BinaryOp(left, right, op) => {
            let (left, right) = (const_value(left)?, const_value(right)?);
            Some(match op {
                eql::BinaryOp::Add => left + right,
                eql::BinaryOp::Sub => left - right,
                eql::BinaryOp::Mul => left * right,
                eql::BinaryOp::Div => left / right,
            })
        }
        _ => None,
    }
}

pub trait EqlExt {
    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)>;
    fn to_graph_component_affines(&self) -> Vec<(ComponentPath, usize, ElementAffine)>;
    /// First geo-frame converter in the expression, if any (`ecef_to_ned`, …).
    /// Schematic load attaches SQL-backed `QueryPlotData` when this is `Some`.
    fn frame_conversion_name(&self) -> Option<&'static str>;
}

impl EqlExt for eql::Expr {
    /// Name of the first geo-frame converter in the expression, if any.
    fn frame_conversion_name(&self) -> Option<&'static str> {
        match self {
            eql::Expr::Formula(formula, expr) => {
                if formula.frame_conversion().is_some() {
                    Some(formula.name())
                } else {
                    expr.frame_conversion_name()
                }
            }
            eql::Expr::ArrayAccess(expr, _) => expr.frame_conversion_name(),
            eql::Expr::Tuple(exprs) => exprs.iter().find_map(|e| e.frame_conversion_name()),
            eql::Expr::BinaryOp(left, right, _) => left
                .frame_conversion_name()
                .or_else(|| right.frame_conversion_name()),
            _ => None,
        }
    }

    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)> {
        self.to_graph_component_affines()
            .into_iter()
            .map(|(path, index, _)| (path, index))
            .collect()
    }

    fn to_graph_component_affines(&self) -> Vec<(ComponentPath, usize, ElementAffine)> {
        match self {
            eql::Expr::ComponentPart(component_part) => {
                let Some(component) = &component_part.component else {
                    return vec![];
                };
                (0..component.element_names.len())
                    .map(|i| {
                        (
                            ComponentPath::from_name(&component_part.name),
                            i,
                            ElementAffine::default(),
                        )
                    })
                    .collect()
            }
            eql::Expr::ArrayAccess(expr, i) => {
                // Handle array access - recursively get components from the inner expression
                match &**expr {
                    eql::Expr::ComponentPart(component_part) => {
                        vec![(
                            ComponentPath::from_name(&component_part.name),
                            *i,
                            ElementAffine::default(),
                        )]
                    }
                    // For formulas or binary ops, extract components recursively
                    _ => expr.to_graph_component_affines(),
                }
            }
            eql::Expr::Tuple(exprs) => exprs
                .iter()
                .flat_map(|expr| expr.to_graph_component_affines().into_iter())
                .collect(),
            eql::Expr::BinaryOp(left, right, op) => {
                // Arithmetic against a constant stays exactly representable, so
                // fold it into the affine. Anything else (component against
                // component, non-affine ops) falls back to listing both sides'
                // components untransformed.
                if let Some(k) = const_value(right) {
                    let folded = match op {
                        eql::BinaryOp::Add => Some(Folded::Shift(k)),
                        eql::BinaryOp::Sub => Some(Folded::Shift(-k)),
                        eql::BinaryOp::Mul => Some(Folded::Scale(k)),
                        eql::BinaryOp::Div if k != 0.0 => Some(Folded::Scale(1.0 / k)),
                        eql::BinaryOp::Div => None,
                    };
                    if let Some(folded) = folded {
                        return folded.apply_to(left.to_graph_component_affines());
                    }
                } else if let Some(k) = const_value(left) {
                    // `k - expr` is `-expr + k`; `k / expr` is not affine.
                    let folded = match op {
                        eql::BinaryOp::Add => Some(vec![Folded::Shift(k)]),
                        eql::BinaryOp::Sub => Some(vec![Folded::Scale(-1.0), Folded::Shift(k)]),
                        eql::BinaryOp::Mul => Some(vec![Folded::Scale(k)]),
                        eql::BinaryOp::Div => None,
                    };
                    if let Some(folded) = folded {
                        let mut components = right.to_graph_component_affines();
                        for step in folded {
                            components = step.apply_to(components);
                        }
                        return components;
                    }
                }
                let mut components = left.to_graph_component_affines();
                components.extend(right.to_graph_component_affines());
                components
            }
            eql::Expr::Formula(_, expr) => {
                // Extract components from the formula's receiver/operand. The
                // formula itself is not affine, so the transform is dropped.
                expr.to_graph_component_affines()
                    .into_iter()
                    .map(|(path, index, _)| (path, index, ElementAffine::default()))
                    .collect()
            }
            _ => vec![],
        }
    }
}

/// A constant folded out of a [`eql::Expr::BinaryOp`], ready to compose onto the
/// other operand's affines.
enum Folded {
    Scale(f64),
    Shift(f64),
}

impl Folded {
    fn apply_to(
        &self,
        components: Vec<(ComponentPath, usize, ElementAffine)>,
    ) -> Vec<(ComponentPath, usize, ElementAffine)> {
        components
            .into_iter()
            .map(|(path, index, affine)| {
                let affine = match self {
                    Folded::Scale(k) => affine.scaled(*k),
                    Folded::Shift(k) => affine.shifted(*k),
                };
                (path, index, affine)
            })
            .collect()
    }
}

#[cfg(test)]
mod element_affine_tests {
    use super::*;
    use std::sync::Arc;

    /// `<name>[<index>]`, the shape `line_3d` axes come in.
    fn element(name: &str, index: usize) -> eql::Expr {
        eql::Expr::ArrayAccess(
            Box::new(eql::Expr::ComponentPart(Arc::new(eql::ComponentPart {
                name: name.to_string(),
                id: impeller2::types::ComponentId::new(name),
                component: None,
                children: Default::default(),
            }))),
            index,
        )
    }

    fn binary(left: eql::Expr, right: eql::Expr, op: eql::BinaryOp) -> eql::Expr {
        eql::Expr::BinaryOp(Box::new(left), Box::new(right), op)
    }

    fn affines(expr: &eql::Expr) -> Vec<ElementAffine> {
        expr.to_graph_component_affines()
            .into_iter()
            .map(|(_, _, affine)| affine)
            .collect()
    }

    fn only_affine(expr: &eql::Expr) -> ElementAffine {
        let affines = affines(expr);
        assert_eq!(affines.len(), 1, "{affines:?}");
        affines[0]
    }

    #[test]
    fn bare_element_is_identity() {
        assert!(only_affine(&element("ball.pos", 0)).is_identity());
    }

    #[test]
    fn constant_offsets_fold_in() {
        // `pos[0] + 1.5` is what separates two otherwise-coincident trails.
        let shifted = binary(
            element("ball.pos", 0),
            eql::Expr::FloatLiteral(1.5),
            eql::BinaryOp::Add,
        );
        assert_eq!(only_affine(&shifted).apply(10.0), 11.5);

        // Addition commutes.
        let flipped = binary(
            eql::Expr::FloatLiteral(1.5),
            element("ball.pos", 0),
            eql::BinaryOp::Add,
        );
        assert_eq!(only_affine(&flipped).apply(10.0), 11.5);

        let subtracted = binary(
            element("ball.pos", 0),
            eql::Expr::FloatLiteral(1.5),
            eql::BinaryOp::Sub,
        );
        assert_eq!(only_affine(&subtracted).apply(10.0), 8.5);

        // `k - expr` negates the element.
        let negated = binary(
            eql::Expr::FloatLiteral(1.5),
            element("ball.pos", 0),
            eql::BinaryOp::Sub,
        );
        assert_eq!(only_affine(&negated).apply(10.0), -8.5);
    }

    #[test]
    fn constant_scales_fold_in() {
        let scaled = binary(
            element("ball.pos", 0),
            eql::Expr::FloatLiteral(3.0),
            eql::BinaryOp::Mul,
        );
        assert_eq!(only_affine(&scaled).apply(10.0), 30.0);

        let divided = binary(
            element("ball.pos", 0),
            eql::Expr::FloatLiteral(4.0),
            eql::BinaryOp::Div,
        );
        assert_eq!(only_affine(&divided).apply(10.0), 2.5);
    }

    #[test]
    fn nested_constants_compose_in_order() {
        // `(pos[0] + 1) * 2` scales the existing offset too.
        let expr = binary(
            binary(
                element("ball.pos", 0),
                eql::Expr::FloatLiteral(1.0),
                eql::BinaryOp::Add,
            ),
            eql::Expr::FloatLiteral(2.0),
            eql::BinaryOp::Mul,
        );
        assert_eq!(only_affine(&expr).apply(10.0), 22.0);
    }

    #[test]
    fn division_by_zero_is_not_folded() {
        let expr = binary(
            element("ball.pos", 0),
            eql::Expr::FloatLiteral(0.0),
            eql::BinaryOp::Div,
        );
        assert!(only_affine(&expr).is_identity());
    }

    #[test]
    fn component_arithmetic_stays_untransformed() {
        // Two components can't collapse to one affine, so both are listed as-is
        // (the pre-existing behavior).
        let expr = binary(element("a.pos", 0), element("b.pos", 1), eql::BinaryOp::Add);
        let affines = affines(&expr);
        assert_eq!(affines.len(), 2);
        assert!(affines.iter().all(|a| a.is_identity()));
    }

    #[test]
    fn to_graph_components_matches_the_affine_traversal() {
        // The plain accessor must stay a projection of the affine one; graphs and
        // monitors rely on its exact ordering.
        let expr = eql::Expr::Tuple(vec![
            eql::Expr::FloatLiteral(0.0),
            binary(
                element("ball.pos", 0),
                eql::Expr::FloatLiteral(1.5),
                eql::BinaryOp::Add,
            ),
            element("ball.pos", 1),
        ]);
        let plain = expr.to_graph_components();
        let with_affine = expr.to_graph_component_affines();
        assert_eq!(plain.len(), 2);
        assert_eq!(
            plain,
            with_affine
                .iter()
                .map(|(path, index, _)| (path.clone(), *index))
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod frame_conversion_tests {
    use super::*;
    use impeller2::schema::Schema;
    use impeller2::types::{ComponentId, PrimType, Timestamp};
    use std::sync::Arc;

    fn ctx() -> eql::Context {
        let component = Arc::new(eql::Component::new(
            "rocket.world_pos".to_string(),
            ComponentId::new("rocket.world_pos"),
            Schema::new(PrimType::F64, vec![7u64]).unwrap(),
        ));
        eql::Context::from_leaves([component], Timestamp(0), Timestamp(1000))
    }

    #[test]
    fn detects_converter_under_element_tuple() {
        let ctx = ctx();
        let expr = ctx
            .parse_str(
                "(rocket.world_pos[4], rocket.world_pos[5], rocket.world_pos[6]).ecef_to_ned()",
            )
            .unwrap();
        assert_eq!(expr.frame_conversion_name(), Some("ecef_to_ned"));
        // SeriesStore extraction would yield the raw ECEF elements; the graph
        // loader routes converters through SQL-backed QueryPlotData instead.
        assert_eq!(expr.to_graph_components().len(), 3);
    }

    #[test]
    fn plain_tuple_has_no_conversion() {
        let ctx = ctx();
        let expr = ctx
            .parse_str("(rocket.world_pos[4], rocket.world_pos[5])")
            .unwrap();
        assert_eq!(expr.frame_conversion_name(), None);
    }
}

#[cfg(test)]
mod window_key_tests {
    use super::*;
    use std::path::PathBuf;

    fn window_state(path: Option<&str>) -> tiles::WindowState {
        tiles::WindowState {
            descriptor: tiles::WindowDescriptor {
                path: path.map(PathBuf::from),
                ..Default::default()
            },
            graph_entities: Vec::new(),
            tile_state: tiles::TileState::default(),
            ui_state: Default::default(),
        }
    }

    #[test]
    fn preserved_window_key_keeps_db_keys_only() {
        // Ingest-keyed and editor-keyed stored windows keep their exact key.
        assert_eq!(
            preserved_window_key(&window_state(Some("db:windows/detail.kdl"))).as_deref(),
            Some("windows/detail.kdl")
        );
        assert_eq!(
            preserved_window_key(&window_state(Some("db:schematics/detail.kdl"))).as_deref(),
            Some("schematics/detail.kdl")
        );
        // Local file paths and pathless (new) windows get generated names.
        assert_eq!(
            preserved_window_key(&window_state(Some("/abs/detail.kdl"))),
            None
        );
        assert_eq!(preserved_window_key(&window_state(None)), None);
    }

    #[test]
    fn schematics_key_stem_extracts_single_level_stems() {
        assert_eq!(schematics_key_stem("schematics/detail.kdl"), Some("detail"));
        assert_eq!(schematics_key_stem("schematics/sub/detail.kdl"), None);
        assert_eq!(schematics_key_stem("windows/detail.kdl"), None);
        assert_eq!(schematics_key_stem("schematics/.kdl"), None);
    }
}
