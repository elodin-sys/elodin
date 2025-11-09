use std::collections::HashMap;

use crate::{
    GridHandle,
    object_3d::Object3DState,
    ui::{
        HdrEnabled, actions,
        colors::EColor,
        inspector, plot, query_plot, query_table,
        tiles::{self, Pane},
    },
};
use bevy::{ecs::system::SystemParam, prelude::*};
use egui_tiles::{Tile, TileId};
use impeller2_bevy::ComponentMetadataRegistry;
use impeller2_wkt::{
    ActionPane, ComponentMonitor, ComponentPath, Dashboard, Line3d, Panel, Schematic,
    SchematicElem, Split, VectorArrow3d, Viewport, WindowSchematic,
};

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
    pub action_tiles: Query<'w, 's, &'static actions::ActionTile>,
    pub graph_states: Query<'w, 's, &'static plot::GraphState>,
    pub query_plots: Query<'w, 's, &'static query_plot::QueryPlotData>,
    pub viewports: Query<'w, 's, &'static inspector::viewport::Viewport>,
    pub camera_grids: Query<'w, 's, &'static GridHandle>,
    pub grid_visibility: Query<'w, 's, &'static Visibility>,
    pub objects_3d: Query<'w, 's, (Entity, &'static Object3DState)>,
    pub lines_3d: Query<'w, 's, (Entity, &'static Line3d)>,
    pub vector_arrows: Query<'w, 's, (Entity, &'static VectorArrow3d)>,
    pub windows: Res<'w, tiles::WindowManager>,
    pub dashboards: Query<'w, 's, &'static Dashboard<Entity>>,
    pub hdr_enabled: Res<'w, HdrEnabled>,
    pub metadata: Res<'w, ComponentMetadataRegistry>,
}

impl SchematicParam<'_, '_> {
    pub fn get_panel(&self, tile_id: TileId) -> Option<Panel<Entity>> {
        self.get_panel_from_state(self.windows.main(), tile_id)
    }

    pub fn get_panel_from_state(
        &self,
        state: &tiles::TileState,
        tile_id: TileId,
    ) -> Option<Panel<Entity>> {
        let tiles = &state.tree.tiles;
        let tile = tiles.get(tile_id)?;

        match tile {
            Tile::Pane(pane) => match pane {
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

                    Some(Panel::Viewport(Viewport {
                        fov: 45.0,
                        active: false,
                        show_grid,
                        hdr: self.hdr_enabled.0,
                        name: Some(viewport.label.clone()),
                        pos: Some(viewport_data.pos.eql.clone()),
                        look_at: Some(viewport_data.look_at.eql.clone()),
                        aux: cam_entity,
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
                        name: Some(graph_state.label.clone()),
                        graph_type: graph_state.graph_type,
                        auto_y_range: graph_state.auto_y_range,
                        y_range: graph_state.y_range.clone(),
                        aux: graph.id,
                        colors,
                    }))
                }

                Pane::Monitor(monitor) => Some(Panel::ComponentMonitor(ComponentMonitor {
                    component_name: monitor.component_name.clone(),
                })),

                Pane::QueryTable(query_table) => {
                    let query_table = self.query_tables.get(query_table.entity).ok()?;
                    Some(Panel::QueryTable(query_table.data.clone()))
                }

                Pane::QueryPlot(plot) => {
                    let query_plot = self.query_plots.get(plot.entity).ok()?;
                    Some(Panel::QueryPlot(query_plot.data.map_aux(|_| plot.entity)))
                }

                Pane::ActionTile(action) => {
                    let action_tile = self.action_tiles.get(action.entity).ok()?;
                    Some(Panel::ActionPane(ActionPane {
                        label: action_tile.button_name.clone(),
                        lua: action_tile.lua.clone(),
                    }))
                }

                // Not exported
                Pane::VideoStream(_) => None,

                // Structural panes
                Pane::Hierarchy => Some(Panel::Hierarchy),
                Pane::Inspector => Some(Panel::Inspector),
                Pane::SchematicTree(_) => Some(Panel::SchematicTree),

                // Dashboard
                Pane::Dashboard(dash) => {
                    let dashboard = self.dashboards.get(dash.entity).ok()?;
                    Some(Panel::Dashboard(Box::new(dashboard.clone())))
                }
            },

            // ---- Containers ----
            Tile::Container(container) => match container {
                egui_tiles::Container::Tabs(t) => {
                    let mut tabs = vec![];
                    for tile_id in &t.children {
                        if let Some(tab) = self.get_panel_from_state(state, *tile_id) {
                            tabs.push(tab)
                        }
                    }
                    Some(Panel::Tabs(tabs))
                }

                egui_tiles::Container::Linear(linear) => {
                    let mut panels = Vec::new();
                    let mut shares = HashMap::new();

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

                    if panels.is_empty() {
                        return None;
                    }

                    let name = state.get_container_title(tile_id).map(|s| s.to_string());

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
    if let Some(tile_id) = param.windows.main().tree.root() {
        schematic
            .elems
            .extend(param.get_panel(tile_id).map(SchematicElem::Panel))
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
            .map(|(entity, arrow)| SchematicElem::VectorArrow(arrow.map_aux(|_| entity))),
    );

    secondary.0.clear();
    let mut window_elems = Vec::new();
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for state in param.windows.secondary() {
        let base_stem = preferred_secondary_stem(state);
        let unique_stem = ensure_unique_stem(&mut name_counts, &base_stem);
        let file_name = format!("{unique_stem}.kdl");

        let mut window_schematic = Schematic::default();
        if let Some(root_id) = state.tile_state.tree.root()
            && let Some(panel) = param.get_panel_from_state(&state.tile_state, root_id)
        {
            window_schematic.elems.push(SchematicElem::Panel(panel));
        }
        secondary.0.push(SecondarySchematic {
            file_name: file_name.clone(),
            title: state.descriptor.title.clone(),
            schematic: window_schematic,
        });
        window_elems.push(SchematicElem::Window(WindowSchematic {
            title: state.descriptor.title.clone(),
            path: Some(file_name),
            screen: state.descriptor.screen.map(|idx| idx as u32),
            screen_rect: state.descriptor.screen_rect,
        }));
    }

    let primary_layout = param.windows.primary_layout();
    if primary_layout.requested_screen.is_some() || primary_layout.requested_rect.is_some() {
        window_elems.push(SchematicElem::Window(WindowSchematic {
            title: None,
            path: None,
            screen: primary_layout.requested_screen.map(|idx| idx as u32),
            screen_rect: primary_layout.requested_rect,
        }));
    }

    schematic.elems.extend(window_elems);
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

fn preferred_secondary_stem(state: &tiles::SecondaryWindowState) -> String {
    if let Some(title) = state.descriptor.title.as_deref() {
        let stem = sanitize_to_stem(title);
        if !stem.is_empty() {
            return stem;
        }
    }
    if let Some(stem) = state.descriptor.path.file_stem().and_then(|s| s.to_str()) {
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
                let eql::Expr::ComponentPart(component_part) = &**expr else {
                    return vec![];
                };
                vec![(ComponentPath::from_name(&component_part.name), *i)]
            }
            eql::Expr::Tuple(exprs) => exprs
                .iter()
                .flat_map(|expr| expr.to_graph_components().into_iter())
                .collect(),
            _ => vec![],
        }
    }
}
