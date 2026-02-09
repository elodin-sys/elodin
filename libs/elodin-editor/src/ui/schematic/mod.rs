use std::collections::HashMap;

use crate::{
    GridHandle,
    object_3d::Object3DState,
    ui::{
        HdrEnabled, actions, colors,
        colors::EColor,
        inspector, monitor, plot, query_plot, query_table,
        tiles::{self, Pane},
        window::compute_window_title,
    },
    vector_arrow::ViewportArrow,
};
use bevy::{ecs::system::SystemParam, prelude::*, window::PrimaryWindow};
use egui_tiles::{Tile, TileId};
use impeller2_bevy::ComponentMetadataRegistry;
use impeller2_wkt::{
    ActionPane, ComponentMonitor, ComponentPath, Dashboard, Line3d, Panel, Schematic,
    SchematicElem, Split, VectorArrow3d, Viewport, WindowSchematic,
};
use bevy_geo_frames::{GeoFrame, GeoPosition};

pub mod tree;
pub use tree::*;
mod load;
pub use load::*;

#[derive(Resource, Debug, Clone, Deref, DerefMut)]
pub struct CurrentSchematic(pub Schematic<Entity>);

#[derive(Debug, Clone)]
pub struct SecondarySchematic {
    pub file_name: String,
    pub title: Option<String>,
    pub schematic: Schematic<Entity>,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct CurrentSecondarySchematics(pub Vec<SecondarySchematic>);

#[derive(SystemParam)]
pub struct SchematicParam<'w, 's> {
    pub query_tables: Query<'w, 's, &'static query_table::QueryTableData>,
    pub monitors: Query<'w, 's, &'static monitor::MonitorData>,
    pub action_tiles: Query<'w, 's, &'static actions::ActionTile>,
    pub graph_states: Query<'w, 's, &'static plot::GraphState>,
    pub query_plots: Query<'w, 's, &'static query_plot::QueryPlotData>,
    pub viewports: Query<'w, 's, &'static inspector::viewport::Viewport>,
    pub viewport_configs: Query<'w, 's, &'static tiles::ViewportConfig>,
    pub camera_grids: Query<'w, 's, &'static GridHandle>,
    pub grid_visibility: Query<'w, 's, &'static Visibility>,
    pub objects_3d: Query<'w, 's, (Entity, &'static Object3DState)>,
    pub lines_3d: Query<'w, 's, (Entity, &'static Line3d)>,
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
    pub dashboards: Query<'w, 's, &'static Dashboard<Entity>>,
    pub hdr_enabled: Res<'w, HdrEnabled>,
    pub metadata: Res<'w, ComponentMetadataRegistry>,
    pub geo_positions: Query<'w, 's, &'static GeoPosition>,
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
            Pane::Dashboard(dashboard) => Some(dashboard.name.clone()),
            Pane::SchematicTree(pane) => Some(pane.name.clone()),
            Pane::DataOverview(pane) => Some(pane.name.clone()),
            Pane::VideoStream(_) => None,
        }
    }

    fn root_panels_from_state(&self, state: &tiles::TileState) -> Vec<Panel<Entity>> {
        let Some(root_id) = state.tree.root() else {
            return Vec::new();
        };

        match self.get_panel_from_state(state, root_id) {
            Some(Panel::Tabs(tabs)) => vec![Panel::Tabs(tabs)],
            Some(panel) => vec![panel],
            None => Vec::new(),
        }
    }

    pub fn get_panel(&self, tile_id: TileId) -> Option<Panel<Entity>> {
        self.windows_state
            .get(*self.primary_window)
            .ok()
            .and_then(|(window_state, _)| {
                self.get_panel_from_state(&window_state.tile_state, tile_id)
            })
    }

    pub fn get_panel_from_state(
        &self,
        state: &tiles::TileState,
        tile_id: TileId,
    ) -> Option<Panel<Entity>> {
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
                        let mut show_grid = false;
                        if let Ok(grid_handle) = self.camera_grids.get(cam_entity)
                            && let Ok(visibility) = self.grid_visibility.get(grid_handle.grid)
                        {
                            show_grid = matches!(*visibility, Visibility::Visible);
                        }

                        let show_arrows = self
                            .viewport_configs
                            .get(cam_entity)
                            .map(|config| config.show_arrows)
                            .unwrap_or(true);

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
                        let frame: Option<GeoFrame> =
                            self.geo_positions.get(cam_entity)
                            .map(|geo_pos| geo_pos.0).ok();

                        Some(Panel::Viewport(Viewport {
                            fov: 45.0,
                            active: false,
                            show_grid,
                            show_arrows,
                            hdr: self.hdr_enabled.0,
                            name: pane_name,
                            pos: Some(viewport_data.pos.eql.clone()),
                            look_at: Some(viewport_data.look_at.eql.clone()),
                            local_arrows,
                            aux: cam_entity,
                            frame,
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

                        Some(Panel::Graph(impeller2_wkt::Graph {
                            eql,
                            name: pane_name,
                            graph_type: graph_state.graph_type,
                            locked: graph_state.locked,
                            auto_y_range: graph_state.auto_y_range,
                            y_range: graph_state.y_range.clone(),
                            aux: graph.id,
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

                    Pane::QueryTable(query_table) => {
                        let query_table_data = self.query_tables.get(query_table.entity).ok()?;
                        let mut data = query_table_data.data.clone();
                        data.name = pane_name;
                        Some(Panel::QueryTable(data))
                    }

                    Pane::QueryPlot(plot) => {
                        let query_plot = self.query_plots.get(plot.entity).ok()?;
                        let mut query_plot = query_plot.data.map_aux(|_| plot.entity);
                        if let Some(name) = pane_name {
                            query_plot.name = name;
                        }
                        Some(Panel::QueryPlot(query_plot))
                    }

                    Pane::ActionTile(action) => {
                        let action_tile = self.action_tiles.get(action.entity).ok()?;
                        Some(Panel::ActionPane(ActionPane {
                            name: pane_name.unwrap_or_else(|| action_tile.button_name.clone()),
                            lua: action_tile.lua.clone(),
                        }))
                    }

                    // Not exported
                    Pane::VideoStream(_) => None,
                    Pane::DataOverview(_) => Some(Panel::DataOverview(pane_name)),

                    // Structural panes
                    Pane::SchematicTree(_) => Some(Panel::SchematicTree(pane_name)),

                    // Dashboard
                    Pane::Dashboard(dash) => {
                        let mut dashboard = self.dashboards.get(dash.entity).ok()?.clone();
                        if let Some(name) = pane_name {
                            dashboard.root.name = Some(name);
                        }
                        Some(Panel::Dashboard(Box::new(dashboard)))
                    }
                }
            }

            // ---- Containers ----
            Tile::Container(container) => match container {
                egui_tiles::Container::Tabs(t) => {
                    let mut tabs = vec![];
                    for child_id in &t.children {
                        if let Some(tab) = self.get_panel_from_state(state, *child_id) {
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
                        if let Some(panel) = self.get_panel_from_state(state, *child_id) {
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
    mut secondary: ResMut<CurrentSecondarySchematics>,
) {
    schematic.elems.clear();

    if let Some(root_panels) = param
        .windows_state
        .get(*param.primary_window)
        .ok()
        .map(|(window_state, _)| param.root_panels_from_state(&window_state.tile_state))
    {
        schematic
            .elems
            .extend(root_panels.into_iter().map(SchematicElem::Panel))
    }
    schematic.elems.extend(
        param
            .objects_3d
            .iter()
            .map(|(entity, o)| o.data.map_aux(|_| entity))
            .map(SchematicElem::Object3d),
    );
    schematic.elems.extend(
        param
            .lines_3d
            .iter()
            .map(|(entity, line)| SchematicElem::Line3d(line.map_aux(|_| entity))),
    );
    schematic.elems.extend(
        param
            .vector_arrows
            .iter()
            .filter(|(_, _, viewport_arrow)| viewport_arrow.is_none())
            .map(|(entity, arrow, _)| SchematicElem::VectorArrow(arrow.map_aux(|_| entity))),
    );

    secondary.0.clear();
    let mut window_elems = Vec::new();
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for (state, window_id) in &param.windows_state {
        let mut file_name: Option<String> = None;
        let mut window_title: Option<String> = None;

        if !window_id.is_primary() {
            let computed_title = compute_window_title(state);
            if computed_title != "Panel" {
                window_title = Some(computed_title);
            }
            let base_stem = preferred_secondary_stem(state);
            let unique_stem = ensure_unique_stem(&mut name_counts, &base_stem);
            file_name = Some(format!("{unique_stem}.kdl"));

            let mut window_schematic = Schematic::default();
            window_schematic.elems.extend(
                param
                    .root_panels_from_state(&state.tile_state)
                    .into_iter()
                    .map(SchematicElem::Panel),
            );
            if let Some(file_name) = &file_name {
                secondary.0.push(SecondarySchematic {
                    file_name: file_name.clone(),
                    title: window_title.clone(),
                    schematic: window_schematic,
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
            .insert_resource(CurrentSecondarySchematics::default())
            .add_systems(PostUpdate, tiles_to_schematic)
            .add_systems(PostUpdate, sync_schematic.before(tiles_to_schematic))
            .init_resource::<SchematicLiveReloadRx>()
            .add_systems(PreUpdate, load::schematic_live_reload);
    }
}

fn preferred_secondary_stem(state: &tiles::WindowState) -> String {
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
    "secondary".to_string()
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
    let base = if stem.is_empty() { "secondary" } else { stem };
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

pub trait EqlExt {
    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)>;
}

impl EqlExt for eql::Expr {
    fn to_graph_components(&self) -> Vec<(ComponentPath, usize)> {
        match self {
            eql::Expr::ComponentPart(component_part) => {
                let Some(component) = &component_part.component else {
                    return vec![];
                };
                (0..component.element_names.len())
                    .map(|i| (ComponentPath::from_name(&component_part.name), i))
                    .collect()
            }
            eql::Expr::ArrayAccess(expr, i) => {
                // Handle array access - recursively get components from the inner expression
                match &**expr {
                    eql::Expr::ComponentPart(component_part) => {
                        vec![(ComponentPath::from_name(&component_part.name), *i)]
                    }
                    // For formulas or binary ops, extract components recursively
                    _ => expr.to_graph_components(),
                }
            }
            eql::Expr::Tuple(exprs) => exprs
                .iter()
                .flat_map(|expr| expr.to_graph_components().into_iter())
                .collect(),
            eql::Expr::BinaryOp(left, right, _) => {
                // Extract components from both operands
                let mut components = left.to_graph_components();
                components.extend(right.to_graph_components());
                components
            }
            eql::Expr::Formula(_, expr) => {
                // Extract components from the formula's receiver/operand
                expr.to_graph_components()
            }
            _ => vec![],
        }
    }
}
