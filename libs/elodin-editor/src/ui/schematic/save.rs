//! Serialize the current egui_tiles layout to WKT panels, then to KDL.
//!
//! Note: `impeller2_wkt::Graph` has a non-optional `eql: String`, so the KDL
//! serializer will always emit a positional string. When `eql` is empty,
//! this appears as `graph "" name=...`. Hiding that empty string requires
//! a change in the serializer or making `eql` optional in the WKT model.

use std::collections::HashMap;

use egui_tiles::{Container, Tile, TileId};
use impeller2_kdl::serialize_schematic;
use impeller2_wkt::{
    ActionPane, Color, ComponentMonitor, Dashboard, Graph as WktGraph, GraphType, Panel, QueryPlot,
    QueryTable, QueryType, Schematic, SchematicElem, Split, Viewport as WktViewport,
};

use crate::ui::tiles::{Pane, TileState};

/// Entry point used by the UI to dump the current layout as a KDL string.
pub fn serialize_current_layout(ui_state: &TileState) -> String {
    let mut schematic: Schematic<()> = Schematic::default();

    if let Some(root) = ui_state.tree.root() {
        if let Some(panel) = panel_from_tile(ui_state, root) {
            schematic.elems.push(SchematicElem::Panel(panel));
        }
    }

    serialize_schematic(&schematic)
}

/// Returns true if `s` looks like an EQL-like component path (e.g., `a.b`, `a.b[2]`).
fn looks_like_eql_label(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let has_path_hint = s.contains('.') || s.contains('[');
    let valid_chars = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '[' || c == ']');
    has_path_hint && valid_chars
}

#[inline]
fn round1(x: f32) -> f32 {
    (x * 10.0).round() / 10.0
}

/// Recursively converts an egui_tiles node into a `Panel<()>`.
fn panel_from_tile(ui_state: &TileState, id: TileId) -> Option<Panel<()>> {
    let tiles = &ui_state.tree.tiles;
    let tile = tiles.get(id)?;

    match tile {
        // ---- Tabs ----
        Tile::Container(Container::Tabs(tabs)) => {
            // If there is only one child, flatten: return the child panel directly.
            if tabs.children.len() == 1 {
                return panel_from_tile(ui_state, tabs.children[0]);
            }

            let mut children = Vec::new();
            for &child in &tabs.children {
                if let Some(p) = panel_from_tile(ui_state, child) {
                    children.push(p);
                }
            }
            Some(Panel::Tabs(children))
        }

        // ---- Linear (HSplit / VSplit) ----
        Tile::Container(Container::Linear(linear)) => {
            // Collect children as Panels
            let mut children = Vec::new();
            for &child in &linear.children {
                if let Some(p) = panel_from_tile(ui_state, child) {
                    children.push(p);
                }
            }

            // Container custom title (editable in UI) -> Split.name
            let name = ui_state.get_container_title(id).map(|s| s.to_string());

            // Collect explicit shares and round to one decimal
            let mut shares: HashMap<usize, f32> = HashMap::new();
            for (i, &child) in linear.children.iter().enumerate() {
                let mut found: Option<f32> = None;
                for (tid, s) in linear.shares.iter() {
                    if *tid == child {
                        found = Some(*s);
                        break;
                    }
                }
                if let Some(s) = found {
                    shares.insert(i, round1(s));
                }
            }

            let split = Split {
                active: false,
                name,
                panels: children,
                shares,
            };

            Some(match linear.dir {
                egui_tiles::LinearDir::Horizontal => Panel::HSplit(split),
                egui_tiles::LinearDir::Vertical => Panel::VSplit(split),
            })
        }

        // ---- Grid (not exported yet) ----
        Tile::Container(Container::Grid(_)) => {
            // Not supported for the moment
            None
        }

        // ---- Panes ----
        Tile::Pane(pane) => Some(match pane {
            // Viewport: export the label as `name`; other fields default for now
            Pane::Viewport(vp) => Panel::Viewport(WktViewport {
                name: if vp.label.is_empty() {
                    None
                } else {
                    Some(vp.label.clone())
                },
                fov: 45.0,
                active: false,
                show_grid: false,
                hdr: false,
                pos: None,
                look_at: None,
                aux: (),
            }),

            // Graph:
            // - If label looks like EQL, put it in `eql` (no name).
            // - Else, keep label as `name` and leave `eql` empty (serializer will emit `""`).
            Pane::Graph(g) => {
                let (eql, name) = if looks_like_eql_label(&g.label) {
                    (g.label.clone(), None)
                } else if g.label.is_empty() {
                    (String::new(), None)
                } else {
                    (String::new(), Some(g.label.clone()))
                };

                Panel::Graph(WktGraph {
                    eql,
                    name,
                    graph_type: GraphType::Line,
                    auto_y_range: true,
                    y_range: 0.0..1.0,
                    aux: (),
                })
            }

            // Monitor: export component id
            Pane::Monitor(m) => Panel::ComponentMonitor(ComponentMonitor {
                component_id: m.component_id,
            }),

            // Query Table: placeholder without the actual query for now
            Pane::QueryTable(_qt) => Panel::QueryTable(QueryTable {
                query: String::new(),
                query_type: QueryType::EQL,
            }),

            // Query Plot: placeholder with defaults
            Pane::QueryPlot(_qp) => Panel::QueryPlot(QueryPlot {
                label: "Query Plot".to_string(),
                query: String::new(),
                refresh_interval: std::time::Duration::from_millis(1000),
                auto_refresh: false,
                color: Color::rgb(1.0, 1.0, 1.0),
                query_type: QueryType::EQL,
                aux: (),
            }),

            // Action tile: keep the visible label; script will be added later
            Pane::ActionTile(a) => Panel::ActionPane(ActionPane {
                label: a.label.clone(),
                lua: String::new(),
            }),

            // Dashboard: placeholder (we’ll reconstruct later)
            Pane::Dashboard(_d) => Panel::Dashboard(Box::new(Dashboard::default())),

            // Structural panes
            Pane::Hierarchy => Panel::Hierarchy,
            Pane::Inspector => Panel::Inspector,
            Pane::SchematicTree(_) => Panel::SchematicTree,

            // Not exported yet
            Pane::VideoStream(_) => return None,
        }),
    }
}
