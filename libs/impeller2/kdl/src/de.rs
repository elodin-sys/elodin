use impeller2_wkt::{Color, Schematic, SchematicElem, VectorArrow3d};
use kdl::{KdlDocument, KdlNode};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use impeller2_wkt::*;

use crate::KdlSchematicError;

pub fn parse_schematic(input: &str) -> Result<Schematic, KdlSchematicError> {
    let doc = input
        .parse::<KdlDocument>()
        .map_err(|source| KdlSchematicError::ParseError {
            source,
            src: input.to_string(),
            span: (0, input.len()).into(),
        })?;

    let mut schematic = Schematic::default();

    for node in doc.nodes() {
        let elem = parse_schematic_elem(node, input)?;
        schematic.elems.push(elem);
    }

    Ok(schematic)
}

fn parse_schematic_elem(node: &KdlNode, src: &str) -> Result<SchematicElem, KdlSchematicError> {
    match node.name().value() {
        "tabs" | "hsplit" | "vsplit" | "viewport" | "graph" | "component_monitor"
        | "action_pane" | "query_table" | "query_plot" | "inspector" | "hierarchy"
        | "schematic_tree" | "dashboard" => Ok(SchematicElem::Panel(parse_panel(node, src)?)),
        "object_3d" => Ok(SchematicElem::Object3d(parse_object_3d(node, src)?)),
        "line_3d" => Ok(SchematicElem::Line3d(parse_line_3d(node, src)?)),
        "vector_arrow" => Ok(SchematicElem::VectorArrow(parse_vector_arrow(node, src)?)),
        _ => Err(KdlSchematicError::UnknownNode {
            node_type: node.name().to_string(),
            src: src.to_string(),
            span: node.span(),
        }),
    }
}

fn parse_panel(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    match node.name().value() {
        "tabs" => {
            let mut panels = Vec::new();
            if let Some(children) = node.children() {
                for child in children.nodes() {
                    panels.push(parse_panel(child, src)?);
                }
            }
            Ok(Panel::Tabs(panels))
        }
        "hsplit" => parse_split(node, src, true),
        "vsplit" => parse_split(node, src, false),
        "viewport" => parse_viewport(node),
        "graph" => parse_graph(node, src),
        "component_monitor" => parse_component_monitor(node, src),
        "action_pane" => parse_action_pane(node, src),
        "query_table" => parse_query_table(node),
        "query_plot" => parse_query_plot(node, src),
        "inspector" => Ok(Panel::Inspector),
        "hierarchy" => Ok(Panel::Hierarchy),
        "schematic_tree" => Ok(Panel::SchematicTree),
        "dashboard" => parse_dashboard(node),
        _ => Err(KdlSchematicError::UnknownNode {
            node_type: node.name().to_string(),
            src: src.to_string(),
            span: node.span(),
        }),
    }
}

fn parse_split(node: &KdlNode, src: &str, is_horizontal: bool) -> Result<Panel, KdlSchematicError> {
    let mut panels = Vec::new();
    let mut shares = HashMap::new();

    if let Some(children) = node.children() {
        for (i, child) in children.nodes().iter().enumerate() {
            panels.push(parse_panel(child, src)?);

            // Look for share property on child
            if let Some(share_val) = child.get("share")
                && let Some(share) = share_val.as_float()
            {
                shares.insert(i, share as f32);
            }
        }
    }

    let active = node
        .get("active")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let name = node
        .get("name")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let split = Split {
        panels,
        shares,
        active,
        name,
    };

    if is_horizontal {
        Ok(Panel::HSplit(split))
    } else {
        Ok(Panel::VSplit(split))
    }
}

fn parse_viewport(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let fov = node.get("fov").and_then(|v| v.as_float()).unwrap_or(45.0) as f32;

    let active = node
        .get("active")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let name = node
        .get("name")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());
    let show_grid = node
        .get("show_grid")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let hdr = node.get("hdr").and_then(|v| v.as_bool()).unwrap_or(false);

    let pos = node
        .get("pos")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let look_at = node
        .get("look_at")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    Ok(Panel::Viewport(Viewport {
        fov,
        active,
        show_grid,
        hdr,
        name,
        pos,
        look_at,
        aux: (),
    }))
}

fn parse_graph(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let eql = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "eql".to_string(),
            node: "graph".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let name = node
        .get("name")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let graph_type = node
        .get("type")
        .and_then(|v| v.as_string())
        .map(|s| match s {
            "line" => GraphType::Line,
            "point" => GraphType::Point,
            "bar" => GraphType::Bar,
            _ => GraphType::Line,
        })
        .unwrap_or(GraphType::Line);

    let auto_y_range = node
        .get("auto_y_range")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let y_range = if let (Some(y_min), Some(y_max)) = (
        node.get("y_min").and_then(|v| v.as_float()),
        node.get("y_max").and_then(|v| v.as_float()),
    ) {
        y_min..y_max
    } else {
        0.0..1.0
    };
    let colors: Vec<_> = parse_color_children_from_node(node).collect();

    Ok(Panel::Graph(Graph {
        eql,
        name,
        graph_type,
        auto_y_range,
        y_range,
        aux: (),
        colors,
    }))
}

fn parse_component_monitor(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let component_name = node
        .get("component_name")
        .and_then(|v| v.as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "component_name".to_string(),
            node: "component_monitor".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?;

    Ok(Panel::ComponentMonitor(ComponentMonitor {
        component_name: component_name.to_string(),
    }))
}

fn parse_action_pane(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let label = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "label".to_string(),
            node: "action_pane".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let lua = node
        .get("lua")
        .and_then(|v| v.as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "lua".to_string(),
            node: "action_pane".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    Ok(Panel::ActionPane(ActionPane { label, lua }))
}

fn parse_query_table(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let query = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .unwrap_or_default()
        .to_string();

    let query_type = node
        .get("type")
        .and_then(|v| v.as_string())
        .map(|s| match s {
            "sql" => QueryType::SQL,
            "eql" => QueryType::EQL,
            _ => QueryType::EQL,
        })
        .unwrap_or(QueryType::EQL);

    Ok(Panel::QueryTable(QueryTable { query, query_type }))
}

fn parse_query_plot(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let label = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "label".to_string(),
            node: "query_plot".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let query = node
        .get("query")
        .and_then(|v| v.as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "query".to_string(),
            node: "query_plot".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let refresh_interval = node
        .get("refresh_interval")
        .and_then(|v| v.as_integer())
        .map(|ms| Duration::from_millis(ms as u64))
        .unwrap_or(Duration::from_secs(1));

    let auto_refresh = node
        .get("auto_refresh")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let color = parse_color_from_node_or_children(node, None).unwrap_or(Color::WHITE);

    let query_type = node
        .get("type")
        .and_then(|v| v.as_string())
        .map(|s| match s {
            "sql" => QueryType::SQL,
            "eql" => QueryType::EQL,
            _ => QueryType::EQL,
        })
        .unwrap_or(QueryType::EQL);

    Ok(Panel::QueryPlot(QueryPlot {
        label,
        query,
        refresh_interval,
        auto_refresh,
        color,
        query_type,
        aux: (),
    }))
}

fn parse_object_3d(node: &KdlNode, src: &str) -> Result<Object3D, KdlSchematicError> {
    let eql = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "eql".to_string(),
            node: "object_3d".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let mesh = if let Some(children) = node.children() {
        parse_object_3d_mesh(children.nodes().first(), src)?
    } else {
        return Err(KdlSchematicError::MissingProperty {
            property: "mesh".to_string(),
            node: "object_3d".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    };

    Ok(Object3D { eql, mesh, aux: () })
}

fn parse_object_3d_mesh(
    node: Option<&KdlNode>,
    src: &str,
) -> Result<Object3DMesh, KdlSchematicError> {
    let node = node.ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "mesh".to_string(),
        node: "object_3d".to_string(),
        src: src.to_string(),
        span: (0, 0).into(),
    })?;

    match node.name().value() {
        "glb" => {
            let path = node
                .get("path")
                .and_then(|v| v.as_string())
                .ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "path".to_string(),
                    node: "glb".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })?;

            Ok(Object3DMesh::Glb(path.to_string()))
        }
        "sphere" => {
            let radius = node
                .get("radius")
                .and_then(|v| v.as_float())
                .ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "radius".to_string(),
                    node: "sphere".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })? as f32;

            let mesh = Mesh::Sphere { radius };
            let material = parse_material_from_node(node).unwrap_or(Material::color(1.0, 1.0, 1.0));

            Ok(Object3DMesh::Mesh { mesh, material })
        }
        "box" => {
            let x = node.get("x").and_then(|v| v.as_float()).ok_or_else(|| {
                KdlSchematicError::MissingProperty {
                    property: "x".to_string(),
                    node: "box".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                }
            })? as f32;

            let y = node.get("y").and_then(|v| v.as_float()).ok_or_else(|| {
                KdlSchematicError::MissingProperty {
                    property: "y".to_string(),
                    node: "box".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                }
            })? as f32;

            let z = node.get("z").and_then(|v| v.as_float()).ok_or_else(|| {
                KdlSchematicError::MissingProperty {
                    property: "z".to_string(),
                    node: "box".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                }
            })? as f32;

            let mesh = Mesh::Box { x, y, z };
            let material = parse_material_from_node(node).unwrap_or(Material::color(1.0, 1.0, 1.0));

            Ok(Object3DMesh::Mesh { mesh, material })
        }
        "cylinder" => {
            let radius = node
                .get("radius")
                .and_then(|v| v.as_float())
                .ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "radius".to_string(),
                    node: "cylinder".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })? as f32;

            let height = node
                .get("height")
                .and_then(|v| v.as_float())
                .ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "height".to_string(),
                    node: "cylinder".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })? as f32;

            let mesh = Mesh::Cylinder { radius, height };
            let material = parse_material_from_node(node).unwrap_or(Material::color(1.0, 1.0, 1.0));

            Ok(Object3DMesh::Mesh { mesh, material })
        }
        "plane" => {
            let size = node
                .get("size")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(10.0);

            let width = node
                .get("width")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(size);

            let depth = node
                .get("depth")
                .and_then(|v| v.as_float())
                .map(|v| v as f32)
                .unwrap_or(size);

            let mesh = Mesh::Plane { width, depth };
            let material = parse_material_from_node(node).unwrap_or(Material::color(1.0, 1.0, 1.0));

            Ok(Object3DMesh::Mesh { mesh, material })
        }
        "ellipsoid" => {
            let scale = node
                .get("scale")
                .and_then(|v| v.as_string())
                .map(str::to_string)
                .unwrap_or_else(impeller2_wkt::default_ellipsoid_scale_expr);

            let color = parse_color_from_node_or_children(node, None)
                .unwrap_or_else(impeller2_wkt::default_ellipsoid_color);

            Ok(Object3DMesh::Ellipsoid { scale, color })
        }
        _ => Err(KdlSchematicError::UnknownNode {
            node_type: node.name().value().to_string(),
            src: src.to_string(),
            span: node.span(),
        }),
    }
}

fn parse_line_3d(node: &KdlNode, src: &str) -> Result<Line3d, KdlSchematicError> {
    let eql = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "eql".to_string(),
            node: "line_3d".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let line_width = node
        .get("line_width")
        .and_then(|v| v.as_float())
        .unwrap_or(1.0) as f32;

    let color = parse_color_from_node_or_children(node, None).unwrap_or(Color::WHITE);

    let perspective = node
        .get("perspective")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(Line3d {
        eql,
        line_width,
        color,
        perspective,
        aux: (),
    })
}

fn parse_vector_arrow(node: &KdlNode, src: &str) -> Result<VectorArrow3d, KdlSchematicError> {
    let vector = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "vector".to_string(),
            node: "vector_arrow".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let origin = node
        .get("origin")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let scale = match node.entry("scale") {
        None => 1.0,
        Some(entry) => {
            let value = entry.value();
            if let Some(value) = value.as_float() {
                value
            } else if let Some(value) = value.as_integer() {
                value as f64
            } else {
                return Err(KdlSchematicError::InvalidValue {
                    property: "scale".to_string(),
                    node: "vector_arrow".to_string(),
                    expected: "a numeric value".to_string(),
                    src: src.to_string(),
                    span: entry.span(),
                });
            }
        }
    };

    let name = node
        .get("name")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let body_frame = node
        .get("body_frame")
        .or_else(|| node.get("in_body_frame"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let normalize = node
        .get("normalize")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let color = parse_color_from_node_or_children(node, None).unwrap_or(Color::WHITE);

    Ok(VectorArrow3d {
        vector,
        origin,
        scale,
        name,
        color,
        body_frame,
        normalize,
        aux: (),
    })
}

fn parse_color_from_node_or_children(node: &KdlNode, color_tag: Option<&str>) -> Option<Color> {
    // First try to parse color from the node itself
    if let Some(color) = parse_color_from_node(node) {
        return Some(color);
    }

    let color_tag = color_tag.unwrap_or("color");
    // If no color found on the node, look for color child nodes
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == color_tag
                && let Some(color) = parse_color_from_node(child)
            {
                return Some(color);
            }
        }
    }

    None
}

fn color_component_from_integer(value: i64) -> Option<f32> {
    if (0..=255).contains(&value) {
        Some((value as f32) / 255.0)
    } else {
        None
    }
}

fn parse_color_component_value(value: &kdl::KdlValue) -> Option<f32> {
    if let Some(integer) = value.as_integer() {
        let Ok(integer) = i64::try_from(integer) else {
            return None;
        };
        color_component_from_integer(integer)
    } else {
        None
    }
}

fn parse_color_component_str(value: &str) -> Option<f32> {
    value
        .parse::<i64>()
        .ok()
        .and_then(color_component_from_integer)
}

fn parse_named_color(name: &str) -> Option<Color> {
    match name {
        "black" => Some(Color::BLACK),
        "white" => Some(Color::WHITE),
        "blue" => Some(Color::BLUE),
        "orange" => Some(Color::ORANGE),
        "yellow" => Some(Color::YELLOW),
        "yalk" => Some(Color::YALK),
        "pink" => Some(Color::PINK),
        "cyan" => Some(Color::CYAN),
        "gray" => Some(Color::GRAY),
        "green" => Some(Color::GREEN),
        "mint" => Some(Color::MINT),
        "turquoise" => Some(Color::TURQUOISE),
        "slate" => Some(Color::SLATE),
        "pumpkin" => Some(Color::PUMPKIN),
        "yolk" => Some(Color::YOLK),
        "peach" => Some(Color::PEACH),
        "reddish" => Some(Color::REDDISH),
        "hyperblue" => Some(Color::HYPERBLUE),
        _ => None,
    }
}

fn parse_color_from_node(node: &KdlNode) -> Option<Color> {
    // First try to read from positional arguments (0,1,2,3)
    let entries = node.entries();
    let positional_entries: Vec<_> = entries.iter().filter(|e| e.name().is_none()).collect();

    if positional_entries.len() >= 3
        && let (Some(r), Some(g), Some(b)) = (
            parse_color_component_value(positional_entries[0].value()),
            parse_color_component_value(positional_entries[1].value()),
            parse_color_component_value(positional_entries[2].value()),
        )
    {
        let a = positional_entries
            .get(3)
            .and_then(|entry| parse_color_component_value(entry.value()))
            .unwrap_or(1.0);

        return Some(Color::rgba(r, g, b, a));
    }

    if let Some(first) = positional_entries.first()
        && let Some(name) = first.value().as_string()
        && let Some(mut color) = parse_named_color(name)
    {
        if let Some(alpha_entry) = positional_entries.get(1)
            && let Some(alpha) = parse_color_component_value(alpha_entry.value())
        {
            color.a = alpha;
        }
        return Some(color);
    }

    if let Some(color_value) = node.get("color").and_then(|v| v.as_string()) {
        if color_value.starts_with('(') && color_value.ends_with(')') {
            let values: Vec<&str> = color_value[1..color_value.len() - 1]
                .split(',')
                .map(|s| s.trim())
                .collect();

            if values.len() >= 3
                && let (Some(r), Some(g), Some(b)) = (
                    parse_color_component_str(values[0]),
                    parse_color_component_str(values[1]),
                    parse_color_component_str(values[2]),
                )
            {
                let a = values
                    .get(3)
                    .and_then(|v| parse_color_component_str(v))
                    .unwrap_or(1.0);

                return Some(Color::rgba(r, g, b, a));
            }
        }

        // Fall back to named colors
        parse_named_color(color_value)
    } else {
        None
    }
}

fn parse_color_children_from_node(node: &KdlNode) -> impl Iterator<Item = Color> {
    node.children()
        .into_iter()
        .flat_map(|c| c.nodes())
        .filter_map(parse_color_from_node)
}

fn parse_material_from_node(node: &KdlNode) -> Option<Material> {
    parse_color_from_node_or_children(node, None).map(|color| Material { base_color: color })
}

fn parse_dashboard(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let root = parse_dashboard_node(node)?;

    Ok(Panel::Dashboard(Box::new(Dashboard { root, aux: () })))
}

fn parse_dashboard_node(node: &KdlNode) -> Result<DashboardNode<()>, KdlSchematicError> {
    let display = node
        .get("display")
        .and_then(|v| v.as_string())
        .and_then(|s| Display::from_str(s).ok())
        .unwrap_or_default();

    let box_sizing = node
        .get("box_sizing")
        .and_then(|v| v.as_string())
        .and_then(|s| BoxSizing::from_str(s).ok())
        .unwrap_or_default();

    let position_type = node
        .get("position_type")
        .and_then(|v| v.as_string())
        .and_then(|s| PositionType::from_str(s).ok())
        .unwrap_or_default();

    let overflow = node
        .get("overflow")
        .and_then(parse_overflow_from_value)
        .unwrap_or(Overflow {
            x: OverflowAxis::Visible,
            y: OverflowAxis::Visible,
        });

    let overflow_clip_margin = node
        .get("overflow_clip_margin")
        .and_then(parse_overflow_clip_margin_from_value)
        .unwrap_or(OverflowClipMargin {
            visual_box: OverflowClipBox::ContentBox,
            margin: 0.0,
        });

    let left = node
        .get("left")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let right = node
        .get("right")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let top = node
        .get("top")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let bottom = node
        .get("bottom")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let width = node
        .get("width")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let height = node
        .get("height")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let min_width = node
        .get("min_width")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let min_height = node
        .get("min_height")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let max_width = node
        .get("max_width")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let max_height = node
        .get("max_height")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);

    let aspect_ratio = node
        .get("aspect_ratio")
        .and_then(|v| v.as_float())
        .map(|f| f as f32);

    let align_items = node
        .get("align_items")
        .and_then(|v| v.as_string())
        .and_then(|s| AlignItems::from_str(s).ok())
        .unwrap_or_default();

    let justify_items = node
        .get("justify_items")
        .and_then(|v| v.as_string())
        .and_then(|s| JustifyItems::from_str(s).ok())
        .unwrap_or_default();

    let align_self = node
        .get("align_self")
        .and_then(|v| v.as_string())
        .and_then(|s| AlignSelf::from_str(s).ok())
        .unwrap_or_default();

    let justify_self = node
        .get("justify_self")
        .and_then(|v| v.as_string())
        .and_then(|s| JustifySelf::from_str(s).ok())
        .unwrap_or_default();

    let align_content = node
        .get("align_content")
        .and_then(|v| v.as_string())
        .and_then(|s| AlignContent::from_str(s).ok())
        .unwrap_or_default();

    let justify_content = node
        .get("justify_content")
        .and_then(|v| v.as_string())
        .and_then(|s| JustifyContent::from_str(s).ok())
        .unwrap_or_default();

    let flex_direction = node
        .get("flex_direction")
        .and_then(|v| v.as_string())
        .and_then(|s| FlexDirection::from_str(s).ok())
        .unwrap_or_default();

    let flex_wrap = node
        .get("flex_wrap")
        .and_then(|v| v.as_string())
        .and_then(|s| FlexWrap::from_str(s).ok())
        .unwrap_or_default();

    let flex_grow = node
        .get("flex_grow")
        .and_then(|v| v.as_float())
        .map(|f| f as f32)
        .unwrap_or(0.0);
    let flex_shrink = node
        .get("flex_shrink")
        .and_then(|v| v.as_float())
        .map(|f| f as f32)
        .unwrap_or(1.0);
    let flex_basis = node
        .get("flex_basis")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);

    let row_gap = node
        .get("row_gap")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);
    let column_gap = node
        .get("column_gap")
        .map(parse_val_from_value)
        .unwrap_or(Val::Auto);

    let text = node
        .get("text")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let font_size = node
        .get("font_size")
        .and_then(|v| v.as_float())
        .map(|f| f as f32)
        .unwrap_or(16.0);

    let label = node
        .get("label")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let mut margin = UiRect::default();
    let mut padding = UiRect::default();
    let mut border = UiRect::default();
    let mut color = Color::TRANSPARENT;
    let mut text_color = Color::WHITE;

    let mut children = Vec::new();
    if let Some(child_nodes) = node.children() {
        for child in child_nodes.nodes() {
            match child.name().value() {
                "text_color" => {
                    text_color = parse_color_from_node(child).unwrap_or(Color::WHITE);
                }
                "bg" | "background" => {
                    color = parse_color_from_node(child).unwrap_or(Color::TRANSPARENT);
                }
                "margin" => {
                    margin = parse_ui_rect_from_node(child).unwrap_or_default();
                }
                "padding" => {
                    padding = parse_ui_rect_from_node(child).unwrap_or_default();
                }
                "border" => {
                    border = parse_ui_rect_from_node(child).unwrap_or_default();
                }
                _ => {
                    children.push(parse_dashboard_node(child)?);
                }
            }
        }
    }

    Ok(DashboardNode {
        label,
        display,
        box_sizing,
        position_type,
        overflow,
        overflow_clip_margin,
        left,
        right,
        top,
        bottom,
        width,
        height,
        min_width,
        min_height,
        max_width,
        max_height,
        aspect_ratio,
        align_items,
        justify_items,
        align_self,
        justify_self,
        align_content,
        justify_content,
        margin,
        padding,
        border,
        flex_direction,
        flex_wrap,
        flex_grow,
        flex_shrink,
        flex_basis,
        row_gap,
        column_gap,
        children,
        color,
        text,
        font_size,
        text_color,
        aux: (),
    })
}

fn parse_overflow_axis(s: &str) -> OverflowAxis {
    match s {
        "visible" => OverflowAxis::Visible,
        "clip" => OverflowAxis::Clip,
        "hidden" => OverflowAxis::Hidden,
        "scroll" => OverflowAxis::Scroll,
        _ => Default::default(),
    }
}

fn parse_val_from_value(value: &kdl::KdlValue) -> Val {
    if let Some(s) = value.as_string() {
        match s {
            "auto" => Val::Auto,
            s if s.ends_with("px") => {
                let px = s.trim_end_matches("px").trim();
                Val::Px(px.to_string())
            }
            s if s.ends_with("%") => {
                let percent = s.trim_end_matches("%").trim();
                Val::Percent(percent.to_string())
            }
            s if s.ends_with("vw") => {
                let vw = s.trim_end_matches("vw").trim();
                Val::Vw(vw.to_string())
            }
            s if s.ends_with("vh") => {
                let vh = s.trim_end_matches("vh").trim();
                Val::Vh(vh.to_string())
            }
            s if s.ends_with("vmin") => {
                let vmin = s.trim_end_matches("vmin").trim();
                Val::VMin(vmin.to_string())
            }
            s if s.ends_with("vmax") => {
                let vmax = s.trim_end_matches("vmax").trim();
                Val::VMax(vmax.to_string())
            }
            _ => Val::Auto,
        }
    } else if let Some(f) = value.as_float() {
        Val::Px((f as f32).to_string())
    } else if let Some(i) = value.as_integer() {
        Val::Px((i as f32).to_string())
    } else {
        Val::Auto
    }
}

fn parse_overflow_from_value(value: &kdl::KdlValue) -> Option<Overflow> {
    if let Some(s) = value.as_string() {
        let axis = parse_overflow_axis(s);
        Some(Overflow { x: axis, y: axis })
    } else {
        None
    }
}

fn parse_overflow_clip_margin_from_value(value: &kdl::KdlValue) -> Option<OverflowClipMargin> {
    if value.as_string().is_some() {
        Some(OverflowClipMargin {
            visual_box: OverflowClipBox::ContentBox,
            margin: 0.0,
        })
    } else {
        None
    }
}

fn parse_ui_rect_from_node(node: &kdl::KdlNode) -> Option<UiRect> {
    let left = node
        .get("left")
        .map(parse_val_from_value)
        .unwrap_or_default();
    let right = node
        .get("right")
        .map(parse_val_from_value)
        .unwrap_or_default();
    let top = node
        .get("top")
        .map(parse_val_from_value)
        .unwrap_or_default();
    let bottom = node
        .get("bottom")
        .map(parse_val_from_value)
        .unwrap_or_default();

    Some(UiRect {
        left,
        right,
        top,
        bottom,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_viewport() {
        let kdl = r#"viewport name="main" fov=60.0 active=#true show_grid=#true"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] {
            assert_eq!(viewport.name, Some("main".to_string()));
            assert_eq!(viewport.fov, 60.0);
            assert!(viewport.active);
            assert!(viewport.show_grid);
        } else {
            panic!("Expected viewport panel");
        }
    }

    #[test]
    fn test_parse_graph() {
        let kdl = r#"graph "a.world_pos" name="Position Graph" type="line""#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Graph(graph)) = &schematic.elems[0] {
            assert_eq!(graph.eql, "a.world_pos");
            assert_eq!(graph.name, Some("Position Graph".to_string()));
            assert_eq!(graph.graph_type, GraphType::Line);
        } else {
            panic!("Expected graph panel");
        }
    }

    #[test]
    fn test_parse_graph_colors() {
        let kdl = r#"
graph "rocket.fins[2], rocket.fins[3]" {
    color 255 0 0
    color 0 255 0
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Graph(graph)) = &schematic.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 2);
        assert_eq!(graph.colors[0], Color::rgb(1.0, 0.0, 0.0));
        assert_eq!(graph.colors[1], Color::rgb(0.0, 1.0, 0.0));
    }

    /// I would like for this test to pass in the future. That is, I want the
    /// parsing to fail because color is given no positional arguments, but it
    /// looks sensible given its keyword arguments. Currently this will parse
    /// without error and the sphere will be white instead of red.
    #[ignore]
    #[test]
    fn test_parse_object_3d_sphere_old() {
        let kdl = r#"
object_3d "a.world_pos" {
    sphere radius=0.2 {
        color r=1.0 g=0.0 b=0.0
    }
}
    "#;
        assert!(parse_schematic(kdl).is_err());
    }

    #[test]
    fn test_parse_object_3d_sphere() {
        let kdl = r#"
object_3d "a.world_pos" {
    sphere radius=0.2 {
        color 255 0 0
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "a.world_pos");
            if let Object3DMesh::Mesh { mesh, material } = &obj.mesh {
                if let Mesh::Sphere { radius } = mesh {
                    assert_eq!(*radius, 0.2);
                } else {
                    panic!("Expected sphere mesh");
                }
                assert_eq!(material.base_color.r, 1.0);
                assert_eq!(material.base_color.g, 0.0);
                assert_eq!(material.base_color.b, 0.0);
            } else {
                panic!("Expected mesh object");
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_plane() {
        let kdl = r#"
object_3d "a.world_pos" {
    plane width=12.5 depth=8.0 {
        color 0 255 0
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };

        assert_eq!(obj.eql, "a.world_pos");

        let Object3DMesh::Mesh { mesh, material } = &obj.mesh else {
            panic!("Expected mesh object");
        };

        let Mesh::Plane { width, depth } = mesh else {
            panic!("Expected plane mesh");
        };

        assert!((*width - 12.5).abs() < f32::EPSILON);
        assert!((*depth - 8.0).abs() < f32::EPSILON);
        assert_eq!(material.base_color.r, 0.0);
        assert_eq!(material.base_color.g, 1.0);
        assert_eq!(material.base_color.b, 0.0);
    }

    #[test]
    fn test_parse_object_3d_plane_size_default() {
        let kdl = r#"
object_3d "a.world_pos" {
    plane size=4.0 {
        color 0 0 255
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };

        let Object3DMesh::Mesh { mesh, material } = &obj.mesh else {
            panic!("Expected mesh object");
        };

        let Mesh::Plane { width, depth } = mesh else {
            panic!("Expected plane mesh");
        };

        assert!((*width - 4.0).abs() < f32::EPSILON);
        assert!((*depth - 4.0).abs() < f32::EPSILON);
        assert_eq!(material.base_color.r, 0.0);
        assert_eq!(material.base_color.g, 0.0);
        assert_eq!(material.base_color.b, 1.0);
    }

    #[test]
    fn test_parse_object_3d_ellipsoid() {
        let kdl = r#"
object_3d "rocket.world_pos" {
    ellipsoid scale="rocket.scale" {
        color 64 128 255 96
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "rocket.world_pos");
            match &obj.mesh {
                Object3DMesh::Ellipsoid { scale, color } => {
                    assert_eq!(scale, "rocket.scale");
                    assert!((color.r - 64.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.g - 128.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.b - 1.0).abs() < f32::EPSILON);
                    assert!((color.a - 96.0 / 255.0).abs() < f32::EPSILON);
                }
                _ => panic!("Expected ellipsoid mesh"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_tabs_with_children() {
        let kdl = r#"
tabs {
    viewport name="camera1" fov=45.0
    graph "data.position" name="Position"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Tabs(tabs)) = &schematic.elems[0] {
            assert_eq!(tabs.len(), 2);

            if let Panel::Viewport(viewport) = &tabs[0] {
                assert_eq!(viewport.name, Some("camera1".to_string()));
            } else {
                panic!("Expected viewport in first tab");
            }

            if let Panel::Graph(graph) = &tabs[1] {
                assert_eq!(graph.eql, "data.position");
                assert_eq!(graph.name, Some("Position".to_string()));
            } else {
                panic!("Expected graph in second tab");
            }
        } else {
            panic!("Expected tabs panel");
        }
    }

    #[test]
    fn test_parse_line_3d() {
        let kdl = r#"line_3d "trajectory" line_width=2.0 color="mint" perspective=#false"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Line3d(line) = &schematic.elems[0] {
            assert_eq!(line.eql, "trajectory");
            assert_eq!(line.line_width, 2.0);
            assert_eq!(line.color.r, Color::MINT.r);
            assert_eq!(line.color.g, Color::MINT.g);
            assert_eq!(line.color.b, Color::MINT.b);
            assert!(!line.perspective);
        } else {
            panic!("Expected line_3d");
        }
    }

    #[test]
    fn test_parse_vector_arrow() {
        let kdl = r#"
vector_arrow "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]" origin="ball.world_pos" scale=1.5 name="Velocity" body_frame=#true normalize=#true {
    color 0 0 255
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::VectorArrow(arrow) = &schematic.elems[0] {
            assert_eq!(
                arrow.vector,
                "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]"
            );
            assert_eq!(arrow.origin.as_deref(), Some("ball.world_pos"));
            assert_eq!(arrow.scale, 1.5);
            assert_eq!(arrow.name.as_deref(), Some("Velocity"));
            assert!(arrow.body_frame);
            assert!(arrow.normalize);
            assert_eq!(arrow.color.b, 1.0);
        } else {
            panic!("Expected vector_arrow");
        }
    }

    #[test]
    fn test_parse_vector_arrow_in_body_frame_alias() {
        let kdl = r#"
vector_arrow "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]" in_body_frame=#true
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::VectorArrow(arrow) = &schematic.elems[0] {
            assert!(arrow.body_frame);
            assert!(!arrow.normalize);
        } else {
            panic!("Expected vector_arrow");
        }
    }

    #[test]
    fn test_parse_vector_arrow_integer_scale() {
        let kdl = r#"
vector_arrow "a.vector" scale=2 {
    color "mint"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::VectorArrow(arrow) = &schematic.elems[0] {
            assert_eq!(arrow.scale, 2.0);
        } else {
            panic!("Expected vector_arrow");
        }
    }

    #[test]
    fn test_parse_vector_arrow_invalid_scale() {
        let kdl = r#"vector_arrow "a.vector" scale="big""#;

        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "scale");
                assert_eq!(node, "vector_arrow");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_complex_example() {
        let kdl = r#"
tabs {
    viewport fov=45.0 active=#true show_grid=#false hdr=#true
    graph "a.world_pos" name="a world_pos"
}

object_3d "a.world_pos" {
    sphere radius=0.2
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 2);

        // Check tabs panel
        if let SchematicElem::Panel(Panel::Tabs(tabs)) = &schematic.elems[0] {
            assert_eq!(tabs.len(), 2);
        } else {
            panic!("Expected tabs panel");
        }

        // Check object_3d
        if let SchematicElem::Object3d(obj) = &schematic.elems[1] {
            assert_eq!(obj.eql, "a.world_pos");
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_component_monitor() {
        let kdl = r#"component_monitor component_name="a.world_pos""#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Panel(Panel::ComponentMonitor(monitor)) = &schematic.elems[0] {
            assert_eq!(monitor.component_name, "a.world_pos");
        } else {
            panic!("Expected component_monitor");
        }
    }

    #[test]
    fn test_parse_dashboard_with_font_and_color() {
        let kdl = r#"
dashboard label="Styled Dashboard" {
    node display="flex" flex_direction="column" text="Hello World" font_size=24.0  {
        text_color color="blue"
        node width="100px" height="50px" text="Child Text" font_size=12.0 {
            text_color color="mint"
        }
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Dashboard(dashboard)) = &schematic.elems[0] {
            assert_eq!(dashboard.root.label, Some("Styled Dashboard".to_string()));
            assert_eq!(dashboard.root.children.len(), 1);

            let node = &dashboard.root.children[0];
            assert!(matches!(node.display, Display::Flex));
            assert_eq!(node.font_size, 24.0);
            assert_eq!(node.text_color, Color::rgb(0.0, 0.0, 1.0));
            assert_eq!(node.text, Some("Hello World".to_string()));

            assert_eq!(node.children.len(), 1);
            let child_node = &node.children[0];
            assert_eq!(child_node.font_size, 12.0);
            assert_eq!(child_node.text_color, Color::MINT);
            assert_eq!(child_node.text, Some("Child Text".to_string()));
        } else {
            panic!("Expected dashboard");
        }
    }

    #[test]
    fn test_parse_dashboard() {
        let kdl = r#"
dashboard label="Test Dashboard" {
    node display="flex" flex_direction="column" {
        text "Hello World"
        node width="100px" height="50px" {
            text "Child Text"
        }
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Dashboard(dashboard)) = &schematic.elems[0] {
            assert_eq!(dashboard.root.label, Some("Test Dashboard".to_string()));
            assert_eq!(dashboard.root.children.len(), 1);

            let node = &dashboard.root.children[0];
            assert!(matches!(node.display, Display::Flex));
            assert!(matches!(node.flex_direction, FlexDirection::Column));
            assert_eq!(node.children.len(), 2);
        } else {
            panic!("Expected dashboard panel");
        }
    }

    #[test]
    fn test_parse_mesh_example() {
        let kdl = r#"
tabs {
    viewport fov=45.0 active=#true show_grid=#false hdr=#true
    graph "a.world_pos" name="a world_pos"
}

object_3d "a.world_pos" {
    glb path="hi"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 2);

        // Check tabs panel
        if let SchematicElem::Panel(Panel::Tabs(tabs)) = &schematic.elems[0] {
            assert_eq!(tabs.len(), 2);
        } else {
            panic!("Expected tabs panel");
        }

        // Check object_3d
        if let SchematicElem::Object3d(obj) = &schematic.elems[1] {
            assert_eq!(obj.eql, "a.world_pos");
            match &obj.mesh {
                Object3DMesh::Glb(s) => assert_eq!(s.as_str(), "hi"),
                _ => panic!("Expected glb"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_color_tuple_rgba() {
        let kdl = r#"
object_3d "test" {
    sphere radius=0.2 color="(1.0, 0.5, 0.0, 0.8)"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            if let Object3DMesh::Mesh { material, .. } = &obj.mesh {
                assert_eq!(material.base_color, Color::WHITE);
            } else {
                panic!("Expected mesh object");
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_color_tuple_rgb() {
        let kdl = r#"
object_3d "test" {
    sphere radius=0.2 color="(0.0, 1.0, 0.0)"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            if let Object3DMesh::Mesh { material, .. } = &obj.mesh {
                assert_eq!(material.base_color, Color::WHITE);
            } else {
                panic!("Expected mesh object");
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_color_tuple_rgb_no_quotes() {
        let kdl = r#"
object_3d "test" {
    sphere radius=0.2 {
       color 0 255 0
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            if let Object3DMesh::Mesh { material, .. } = &obj.mesh {
                assert_eq!(material.base_color.r, 0.0);
                assert_eq!(material.base_color.g, 1.0);
                assert_eq!(material.base_color.b, 0.0);
                assert_eq!(material.base_color.a, 1.0); // Should default to 1.0
            } else {
                panic!("Expected mesh object");
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_color_children_from_node() {
        let kdl = r#"
node {
    color 255 0 0
    color 0 255 0
    color 0 0 255
    other_node "not a color"
}
"#;
        let doc = kdl.parse::<KdlDocument>().unwrap();
        let node = &doc.nodes()[0];

        let colors = parse_color_children_from_node(node).collect::<Vec<_>>();

        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0].r, 1.0);
        assert_eq!(colors[0].g, 0.0);
        assert_eq!(colors[0].b, 0.0);
        assert_eq!(colors[1].r, 0.0);
        assert_eq!(colors[1].g, 1.0);
        assert_eq!(colors[1].b, 0.0);
        assert_eq!(colors[2].r, 0.0);
        assert_eq!(colors[2].g, 0.0);
        assert_eq!(colors[2].b, 1.0);
    }

    #[test]
    fn test_parse_color_children_from_sphere() {
        let kdl = r#"
sphere radius=0.2 {
        color 255 0 255
    }
"#;
        let doc = kdl.parse::<KdlDocument>().unwrap();
        let node = &doc.nodes()[0];
        assert_eq!(node.name().to_string(), "sphere");
        assert_eq!(node.children().iter().count(), 1);
        let color_doc = node.children().iter().next().copied().unwrap();
        let s = format!("{}", color_doc);
        assert_eq!("color 255 0 255", s.as_str().trim());
        let color_node = color_doc.get("color").unwrap();
        assert_eq!(color_node.entries().len(), 3);
        let mut entries = color_node.entries().iter().filter(|e| e.name().is_none());
        assert_eq!(entries.clone().count(), 3);
        let r = entries.next().and_then(|e| {
            e.value()
                .as_float()
                .or_else(|| e.value().as_integer().map(|x| x as f64))
        });
        assert_eq!(r, Some(255.0));
        let color = parse_color_from_node(color_node);
        assert!(color.is_some());

        let colors = parse_color_children_from_node(node).collect::<Vec<_>>();

        assert_eq!(colors.len(), 1);
        assert_eq!(colors[0].r, 1.0);
        assert_eq!(colors[0].g, 0.0);
        assert_eq!(colors[0].b, 1.0);
        assert_eq!(colors[0].a, 1.0);
    }

    #[test]
    fn test_parse_named_color_yalk() {
        let kdl = r#"
graph "value" {
    color yalk
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        let SchematicElem::Panel(Panel::Graph(graph)) = &schematic.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 1);
        let color = graph.colors[0];
        assert_eq!(color, Color::YALK);
    }

    #[test]
    fn test_parse_named_color_with_alpha() {
        let kdl = r#"
graph "value" {
    color yalk 120
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        let SchematicElem::Panel(Panel::Graph(graph)) = &schematic.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 1);
        let color = graph.colors[0];
        assert_eq!(color.r, Color::YALK.r);
        assert_eq!(color.g, Color::YALK.g);
        assert_eq!(color.b, Color::YALK.b);
        assert!((color.a - (120.0 / 255.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_error_report() {
        use miette::{GraphicalReportHandler, GraphicalTheme};
        let kdl = r#"
blah
graph "value" {
    color yalk
}
"#;
        let err = parse_schematic(kdl).unwrap_err();
        assert_eq!("Unknown node type 'blah'", format!("{}", err));
        let reporter = GraphicalReportHandler::new_themed(GraphicalTheme::unicode_nocolor());
        let mut b = String::new();
        reporter.render_report(&mut b, &err).unwrap();
        assert_eq!(
            "kdl_schematic::unknown_node\n\n  × Unknown node type 'blah'\n   ╭─[2:1]\n 1 │ \n 2 │ blah\n   · ──┬─\n   ·   ╰── unknown node\n 3 │ graph \"value\" {\n   ╰────\n",
            &b
        );
    }
}
