use crate::color_names::color_from_name;
use bevy_geo_frames::{GeoFrame, RotationKind};
use impeller2_wkt::{
    ArrowThickness, Color, Schematic, SchematicElem, ThemeConfig, VectorArrow3d, WindowSchematic,
};
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
        if node.name().value() == "skybox" {
            schematic.skybox = Some(parse_skybox(node, input)?);
            continue;
        }
        if node.name().value() == "environment" {
            schematic.environment = Some(parse_environment(node, input)?);
            continue;
        }
        if node.name().value() == "telemetry_mode" {
            schematic.telemetry_mode = parse_telemetry_mode(node, input)?;
            continue;
        }

        let elem = parse_schematic_elem(node, input)?;
        match elem {
            SchematicElem::Theme(theme) => schematic.theme = Some(theme),
            SchematicElem::Timeline(timeline) => schematic.timeline = Some(timeline),
            SchematicElem::Coordinate(coordinate) => {
                schematic.frame = Some(coordinate.frame);
                schematic.origin = coordinate.origin;
            }
            other => schematic.elems.push(other),
        }
    }

    Ok(schematic)
}

fn parse_skybox(node: &KdlNode, src: &str) -> Result<SkyboxConfig, KdlSchematicError> {
    Ok(SkyboxConfig {
        name: require_name(node, src)?,
    })
}

/// Parses the top-level `environment` node:
///
/// ```kdl
/// environment {
///     sun azimuth=320.0 elevation=32.0 illuminance=130000.0 shadows=#true
///     ambient scale=0.02
///     sky color="black"
/// }
/// ```
fn parse_environment(node: &KdlNode, src: &str) -> Result<EnvironmentConfig, KdlSchematicError> {
    let mut config = EnvironmentConfig::default();
    let Some(children) = node.children() else {
        return Ok(config);
    };
    let float_prop = |child: &KdlNode, prop: &str| -> Option<f32> {
        child
            .get(prop)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|v| v as f32)
    };
    for child in children.nodes() {
        match child.name().value() {
            "sun" => {
                let mut sun = SunConfig::default();
                if let Some(azimuth) = float_prop(child, "azimuth") {
                    sun.azimuth_deg = azimuth;
                }
                if let Some(elevation) = float_prop(child, "elevation") {
                    sun.elevation_deg = elevation;
                }
                if let Some(illuminance) = float_prop(child, "illuminance") {
                    sun.illuminance = illuminance;
                }
                if let Some(shadows) = bool_prop(child, "shadows") {
                    sun.shadows = shadows;
                }
                config.sun = Some(sun);
            }
            "ambient" => {
                config.ambient_scale = float_prop(child, "scale").ok_or_else(|| {
                    KdlSchematicError::MissingProperty {
                        property: "scale".to_string(),
                        node: "ambient".to_string(),
                        src: src.to_string(),
                        span: child.span(),
                    }
                })?;
            }
            "sky" => {
                config.sky_color =
                    Some(parse_named_color_field(child, "color").ok_or_else(|| {
                        KdlSchematicError::InvalidValue {
                            property: "color".to_string(),
                            node: "sky".to_string(),
                            expected: "a named color or tuple string like \"(0,0,0)\"".to_string(),
                            src: src.to_string(),
                            span: child.span(),
                        }
                    })?);
            }
            "atmosphere" => {
                let mut atmosphere = AtmosphereConfig::default();
                if let Some(origin) = parse_tuple3::<f64>(child, "origin") {
                    atmosphere.origin = origin;
                }
                if let Some(inner) = float_prop(child, "inner_radius") {
                    atmosphere.inner_radius = inner;
                }
                if let Some(outer) = float_prop(child, "outer_radius") {
                    atmosphere.outer_radius = outer;
                }
                if let Some(albedo) = parse_tuple3::<f32>(child, "ground_albedo") {
                    atmosphere.ground_albedo = albedo;
                }
                if atmosphere.outer_radius <= atmosphere.inner_radius {
                    return Err(KdlSchematicError::InvalidValue {
                        property: "outer_radius".to_string(),
                        node: "atmosphere".to_string(),
                        expected: "outer_radius must be greater than inner_radius".to_string(),
                        src: src.to_string(),
                        span: child.span(),
                    });
                }
                config.atmosphere = Some(atmosphere);
            }
            other => {
                return Err(KdlSchematicError::UnknownNode {
                    node_type: format!("environment.{other}"),
                    src: src.to_string(),
                    span: child.span(),
                });
            }
        }
    }
    Ok(config)
}

fn parse_telemetry_mode(node: &KdlNode, src: &str) -> Result<bool, KdlSchematicError> {
    let Some(entry) = node.entries().first() else {
        return Err(KdlSchematicError::MissingProperty {
            property: "value".to_string(),
            node: "telemetry_mode".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    };
    if entry.name().is_some() {
        return Err(KdlSchematicError::InvalidValue {
            property: "telemetry_mode".to_string(),
            node: "telemetry_mode".to_string(),
            expected: "a positional bool like telemetry_mode #true".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    }
    entry
        .value()
        .as_bool()
        .ok_or_else(|| KdlSchematicError::InvalidValue {
            property: "telemetry_mode".to_string(),
            node: "telemetry_mode".to_string(),
            expected: "#true or #false".to_string(),
            src: src.to_string(),
            span: node.span(),
        })
}

fn parse_schematic_elem(node: &KdlNode, src: &str) -> Result<SchematicElem, KdlSchematicError> {
    match node.name().value() {
        "tabs" | "hsplit" | "vsplit" | "viewport" | "graph" | "component_monitor"
        | "action_pane" | "query_table" | "query_plot" | "inspector" | "hierarchy"
        | "schematic_tree" | "data_overview" => Ok(SchematicElem::Panel(parse_panel(node, src)?)),
        "window" => Ok(SchematicElem::Window(parse_window(node, src)?)),
        "theme" => Ok(SchematicElem::Theme(parse_theme(node, src)?)),
        "timeline" => Ok(SchematicElem::Timeline(parse_timeline(node, src)?)),
        "object_3d" => Ok(SchematicElem::Object3d(parse_object_3d(node, src)?)),
        "line_3d" => Ok(SchematicElem::Line3d(parse_line_3d(node, src)?)),
        "vector_arrow" => Ok(SchematicElem::VectorArrow(parse_vector_arrow(node, src)?)),
        "world_mesh" => Ok(SchematicElem::WorldMesh(parse_world_mesh(node, src)?)),
        "coordinate" => Ok(SchematicElem::Coordinate(parse_coordinate(node, src)?)),
        _ => Err(KdlSchematicError::UnknownNode {
            node_type: node.name().to_string(),
            src: src.to_string(),
            span: node.span(),
        }),
    }
}

fn parse_window(node: &KdlNode, src: &str) -> Result<WindowSchematic, KdlSchematicError> {
    let path = node
        .get("path")
        .or_else(|| node.get("file"))
        .or_else(|| node.get("name"))
        .and_then(|v| v.as_string())
        .map(|raw| {
            let path = raw.trim();
            if path.is_empty() {
                return Err(KdlSchematicError::InvalidValue {
                    property: "path".to_string(),
                    node: node.name().to_string(),
                    expected: "a non-empty relative path".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                });
            }
            if path.contains('{') || path.contains('}') {
                return Err(KdlSchematicError::InvalidValue {
                    property: "path".to_string(),
                    node: node.name().to_string(),
                    expected: "a path without braces".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                });
            }
            Ok(path.to_string())
        })
        .transpose()?;

    let title = node
        .get("title")
        .or_else(|| node.get("display"))
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());

    let screen_idx = node
        .get("screen")
        .and_then(|value| value.as_integer())
        .map(|value| value as u32);

    let mut screen_rect = None;
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == "rect" {
                screen_rect = Some(parse_window_rect(child, src)?);
                break;
            }
        }
    }

    Ok(WindowSchematic {
        title,
        path,
        screen: screen_idx,
        screen_rect,
    })
}

fn parse_theme(node: &KdlNode, _src: &str) -> Result<ThemeConfig, KdlSchematicError> {
    let mode = node
        .get("mode")
        .and_then(|v| v.as_string())
        .or_else(|| {
            node.entries()
                .first()
                .and_then(|entry| entry.value().as_string())
        })
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());

    let scheme = node
        .get("scheme")
        .and_then(|v| v.as_string())
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());

    Ok(ThemeConfig { mode, scheme })
}

fn parse_timeline(node: &KdlNode, src: &str) -> Result<TimelineConfig, KdlSchematicError> {
    const TIMELINE_PROPERTIES: &[&str] =
        &["played_color", "future_color", "follow_latest", "range"];
    const TIMELINE_CHILDREN: &[&str] = &["played_color", "future_color"];

    for entry in node.entries() {
        let Some(name) = entry.name() else {
            return Err(KdlSchematicError::InvalidValue {
                property: "timeline".to_string(),
                node: "timeline".to_string(),
                expected: "named properties only".to_string(),
                src: src.to_string(),
                span: node.span(),
            });
        };

        if !TIMELINE_PROPERTIES.contains(&name.value()) {
            return Err(KdlSchematicError::InvalidValue {
                property: name.value().to_string(),
                node: "timeline".to_string(),
                expected: format!("one of: {}", TIMELINE_PROPERTIES.join(", ")),
                src: src.to_string(),
                span: node.span(),
            });
        }
    }

    if let Some(children) = node.children() {
        for child in children.nodes() {
            if !TIMELINE_CHILDREN.contains(&child.name().value()) {
                return Err(KdlSchematicError::InvalidValue {
                    property: child.name().value().to_string(),
                    node: "timeline".to_string(),
                    expected: format!("one of: {}", TIMELINE_CHILDREN.join(", ")),
                    src: src.to_string(),
                    span: child.span(),
                });
            }
        }
    }

    let parse_timeline_color = |names: &[&str], default: Color| {
        if let Some(value) = names.iter().find_map(|name| node.get(*name)) {
            let raw = value
                .as_string()
                .ok_or_else(|| KdlSchematicError::InvalidValue {
                    property: names[0].to_string(),
                    node: "timeline".to_string(),
                    expected: "a named color or tuple string like \"(255,255,0,200)\"".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })?;
            parse_color_from_text(raw).ok_or_else(|| KdlSchematicError::InvalidValue {
                property: names[0].to_string(),
                node: "timeline".to_string(),
                expected: "a named color or tuple string like \"(255,255,0,200)\"".to_string(),
                src: src.to_string(),
                span: node.span(),
            })
        } else if let Some(child) = node.children().and_then(|children| {
            children
                .nodes()
                .iter()
                .find(|child| names.contains(&child.name().value()))
        }) {
            parse_color_from_node(child).ok_or_else(|| KdlSchematicError::InvalidValue {
                property: names[0].to_string(),
                node: "timeline".to_string(),
                expected: "a named color, RGBA values, or tuple string".to_string(),
                src: src.to_string(),
                span: child.span(),
            })
        } else {
            Ok(default)
        }
    };

    let played_color = parse_timeline_color(&["played_color"], default_timeline_played_color())?;
    let future_color = parse_timeline_color(&["future_color"], default_timeline_future_color())?;

    let follow_latest = node
        .get("follow_latest")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let range = if let Some(value) = node.get("range") {
        let raw = value
            .as_string()
            .ok_or_else(|| KdlSchematicError::InvalidValue {
                property: "range".to_string(),
                node: "timeline".to_string(),
                expected: "a string like \"last_5s\" or \"full\"".to_string(),
                src: src.to_string(),
                span: node.span(),
            })?;
        if !is_valid_timeline_range(raw) {
            return Err(KdlSchematicError::InvalidValue {
                property: "range".to_string(),
                node: "timeline".to_string(),
                expected: "full, last_5s, last_15s, last_30s, last_1m, last_5m, last_15m, last_30m, last_1h, last_6h, last_12h, last_24h, or last_<N>s".to_string(),
                src: src.to_string(),
                span: node.span(),
            });
        }
        Some(raw.trim().to_string())
    } else {
        None
    };

    Ok(TimelineConfig {
        played_color,
        future_color,
        follow_latest,
        range,
    })
}

fn is_valid_timeline_range(raw: &str) -> bool {
    let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
    matches!(
        normalized.as_str(),
        "full"
            | "full_range"
            | "fullrange"
            | "last_5s"
            | "5s"
            | "last_15s"
            | "15s"
            | "last_30s"
            | "30s"
            | "last_1m"
            | "1m"
            | "last_60s"
            | "60s"
            | "last_5m"
            | "5m"
            | "last_15m"
            | "15m"
            | "last_30m"
            | "30m"
            | "last_1h"
            | "1h"
            | "last_6h"
            | "6h"
            | "last_12h"
            | "12h"
            | "last_24h"
            | "24h"
    ) || {
        let secs = normalized
            .strip_prefix("last_")
            .unwrap_or(&normalized)
            .strip_suffix('s')
            .and_then(|n| n.parse::<u64>().ok());
        secs.is_some()
    }
}

fn parse_coordinate(node: &KdlNode, src: &str) -> Result<CoordinateConfig, KdlSchematicError> {
    let frame_str = node
        .get("frame")
        .and_then(|v| v.as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "frame".to_string(),
            node: "coordinate".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?;

    let frame = GeoFrame::from_str(frame_str).map_err(|_| KdlSchematicError::InvalidValue {
        property: "frame".to_string(),
        node: "coordinate".to_string(),
        expected: "ENU, NED, or ECEF".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;

    let get_number = |property: &str| -> Result<Option<f64>, KdlSchematicError> {
        node.get(property)
            .map(|v| {
                v.as_float()
                    .or_else(|| v.as_integer().map(|i| i as f64))
                    .ok_or_else(|| KdlSchematicError::InvalidValue {
                        property: property.to_string(),
                        node: "coordinate".to_string(),
                        expected: "a number".to_string(),
                        src: src.to_string(),
                        span: node.span(),
                    })
            })
            .transpose()
    };

    let lat = get_number("lat")?;
    let lon = get_number("lon")?;
    let alt = get_number("alt")?;
    let origin = match (lat, lon) {
        (Some(latitude), Some(longitude)) => Some(GeoOriginConfig {
            latitude,
            longitude,
            altitude: alt.unwrap_or(0.0),
        }),
        (None, None) if alt.is_none() => None,
        _ => {
            return Err(KdlSchematicError::InvalidValue {
                property: "lat/lon".to_string(),
                node: "coordinate".to_string(),
                expected: "both lat and lon (alt optional)".to_string(),
                src: src.to_string(),
                span: node.span(),
            });
        }
    };

    Ok(CoordinateConfig { frame, origin })
}

fn parse_world_mesh(node: &KdlNode, src: &str) -> Result<WorldMesh, KdlSchematicError> {
    let region = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .or_else(|| node.get("region").and_then(|v| v.as_string()))
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "region".to_string(),
            node: "world_mesh".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .to_string();

    let lod_count = node
        .get("lod_count")
        .and_then(|v| v.as_integer())
        .map(|v| {
            u32::try_from(v).map_err(|_| KdlSchematicError::InvalidValue {
                property: "lod_count".to_string(),
                node: "world_mesh".to_string(),
                expected: format!("an integer between 0 and {}", u32::MAX),
                src: src.to_string(),
                span: node.span(),
            })
        })
        .transpose()?;

    let translate = parse_tuple3::<f64>(node, "translate");

    let frame = node
        .get("frame")
        .and_then(|v| v.as_string())
        .and_then(|s| GeoFrame::from_str(s).ok());

    let visible = node
        .get("visible")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    Ok(WorldMesh {
        region,
        lod_count,
        translate,
        frame,
        visible,
        node_id: NodeId::default(),
    })
}

fn parse_name(node: &KdlNode) -> Option<String> {
    node.get("name")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
}

fn require_name(node: &KdlNode, src: &str) -> Result<String, KdlSchematicError> {
    parse_name(node).ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "name".to_string(),
        node: node.name().to_string(),
        src: src.to_string(),
        span: node.span(),
    })
}

fn parse_window_rect(node: &KdlNode, src: &str) -> Result<WindowRect, KdlSchematicError> {
    if node.entries().len() != 4 {
        return Err(KdlSchematicError::InvalidValue {
            property: "rect".to_string(),
            node: node.name().to_string(),
            expected: "rect requires four numeric entries (x%, y%, width%, height%)".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    }

    let mut values = [0f64; 4];
    for (idx, entry) in node.entries().iter().enumerate() {
        if let Some(value) = entry.value().as_float() {
            values[idx] = value;
        } else if let Some(value) = entry.value().as_integer() {
            values[idx] = value as f64;
        } else {
            return Err(KdlSchematicError::InvalidValue {
                property: "rect".to_string(),
                node: node.name().to_string(),
                expected: "rect entries must be numeric percentages".to_string(),
                src: src.to_string(),
                span: node.span(),
            });
        }
    }

    Ok(WindowRect {
        x: clamp_percent(values[0]),
        y: clamp_percent(values[1]),
        width: clamp_percent(values[2]),
        height: clamp_percent(values[3]),
    })
}

fn clamp_percent(value: f64) -> u32 {
    value.round().clamp(0.0, 100.0) as u32
}

fn parse_panel(node: &KdlNode, kdl_src: &str) -> Result<Panel, KdlSchematicError> {
    match node.name().value() {
        "tabs" => {
            let mut panels = Vec::new();
            if let Some(children) = node.children() {
                for child in children.nodes() {
                    panels.push(parse_panel(child, kdl_src)?);
                }
            }
            Ok(Panel::Tabs(panels))
        }
        "hsplit" => parse_split(node, kdl_src, true),
        "vsplit" => parse_split(node, kdl_src, false),
        "viewport" => parse_viewport(node, kdl_src),
        "graph" => parse_graph(node, kdl_src),
        "component_monitor" => parse_component_monitor(node, kdl_src),
        "action_pane" => parse_action_pane(node, kdl_src),
        "query_table" => parse_query_table(node),
        "query_plot" => parse_query_plot(node, kdl_src),
        "inspector" => Ok(Panel::Inspector),
        "hierarchy" => Ok(Panel::Hierarchy),
        "schematic_tree" => Ok(Panel::SchematicTree(parse_name(node))),
        "data_overview" => Ok(Panel::DataOverview(parse_name(node))),
        "video_stream" => parse_video_stream(node),
        "sensor_view" => parse_sensor_view(node),
        "log_stream" => parse_log_stream(node),
        _ => Err(KdlSchematicError::UnknownNode {
            node_type: node.name().to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        }),
    }
}

fn parse_split(
    node: &KdlNode,
    kdl_src: &str,
    is_horizontal: bool,
) -> Result<Panel, KdlSchematicError> {
    let mut panels = Vec::new();
    let mut shares = HashMap::new();

    if let Some(children) = node.children() {
        for (i, child) in children.nodes().iter().enumerate() {
            panels.push(parse_panel(child, kdl_src)?);

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

    let name = parse_name(node);

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

/// Bool value parser that also accepts case-insensitive `"true"`/`"false"`
/// strings. In KDL 2.0 only `#true`/`#false` are booleans; a bare `True`
/// parses as a string, which previously made `hdr=True` silently false.
fn parse_bool_value(value: &kdl::KdlValue) -> Option<bool> {
    if let Some(b) = value.as_bool() {
        return Some(b);
    }
    match value.as_string()?.to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn bool_prop(node: &KdlNode, prop: &str) -> Option<bool> {
    node.get(prop).and_then(parse_bool_value)
}

fn parse_viewport(node: &KdlNode, kdl_src: &str) -> Result<Panel, KdlSchematicError> {
    let fov = node.get("fov").and_then(|v| v.as_float()).unwrap_or(45.0) as f32;
    let near = node
        .get("near")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32);
    let far = node
        .get("far")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32);
    let aspect = node
        .get("aspect")
        .or_else(|| node.get("aspect_ratio"))
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32);

    if let Some(near) = near
        && near <= 0.0
    {
        return Err(KdlSchematicError::InvalidValue {
            property: "near".to_string(),
            node: "viewport".to_string(),
            expected: "near must be > 0".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        });
    }
    if let Some(far) = far
        && far <= 0.0
    {
        return Err(KdlSchematicError::InvalidValue {
            property: "far".to_string(),
            node: "viewport".to_string(),
            expected: "far must be > 0".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        });
    }
    if let (Some(near), Some(far)) = (near, far)
        && far <= near
    {
        return Err(KdlSchematicError::InvalidValue {
            property: "far".to_string(),
            node: "viewport".to_string(),
            expected: "far must be > near".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        });
    }
    if let Some(aspect) = aspect
        && aspect <= 0.0
    {
        return Err(KdlSchematicError::InvalidValue {
            property: "aspect".to_string(),
            node: "viewport".to_string(),
            expected: "aspect must be > 0".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        });
    }

    let active = bool_prop(node, "active").unwrap_or(false);

    let name = parse_name(node);
    let show_grid = bool_prop(node, "show_grid").unwrap_or(false);

    let show_arrows = bool_prop(node, "show_arrows").unwrap_or(true);

    let create_frustum = bool_prop(node, "create_frustum").unwrap_or(false);

    let show_frustums = node
        .get("show_frustums")
        .or_else(|| node.get("show_frustum"))
        .and_then(parse_bool_value)
        .unwrap_or(false);

    let frustums_color = if let Some(value) = node.get("frustums_color") {
        parse_viewport_color(value).ok_or_else(|| KdlSchematicError::InvalidValue {
            property: "frustums_color".to_string(),
            node: "viewport".to_string(),
            expected: "a named color or tuple string like \"(255,255,0,200)\"".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        })?
    } else {
        default_viewport_frustums_color()
    };
    let projection_color = if let Some(value) = node.get("projection_color") {
        parse_viewport_color(value).ok_or_else(|| KdlSchematicError::InvalidValue {
            property: "projection_color".to_string(),
            node: "viewport".to_string(),
            expected: "a named color or tuple string like \"(255,255,0,200)\"".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        })?
    } else {
        default_viewport_projection_color()
    };
    let frustums_thickness = node
        .get("frustums_thickness")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32)
        .unwrap_or_else(default_viewport_frustums_thickness);
    if frustums_thickness <= 0.0 {
        return Err(KdlSchematicError::InvalidValue {
            property: "frustums_thickness".to_string(),
            node: "viewport".to_string(),
            expected: "frustums_thickness must be > 0".to_string(),
            src: kdl_src.to_string(),
            span: node.span(),
        });
    }

    let show_view_cube = bool_prop(node, "show_view_cube").unwrap_or(true);
    let effects = bool_prop(node, "effects").unwrap_or(true);

    let hdr = bool_prop(node, "hdr").unwrap_or(false);
    let bloom = parse_viewport_bloom(node, kdl_src)?;
    let ev100 = node
        .get("ev100")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32);

    let pos = node
        .get("pos")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

    let look_at = node
        .get("look_at")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

    let frame = node
        .get("frame")
        .and_then(|v| v.as_string())
        .and_then(|s| GeoFrame::from_str(s).ok());
    let up = node
        .get("up")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string())
        .and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

    let mut local_arrows = Vec::new();
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == "vector_arrow" {
                local_arrows.push(parse_vector_arrow(child, kdl_src)?);
            }
        }
    }

    Ok(Panel::Viewport(Viewport {
        fov,
        near,
        far,
        aspect,
        active,
        show_grid,
        show_arrows,
        create_frustum,
        show_frustums,
        frustums_color,
        projection_color,
        frustums_thickness,
        show_view_cube,
        effects,
        hdr,
        bloom,
        ev100,
        name,
        pos,
        look_at,
        frame,
        up,
        local_arrows,
        node_id: NodeId::default(),
    }))
}

fn parse_viewport_bloom(
    node: &KdlNode,
    src: &str,
) -> Result<Option<BloomConfig>, KdlSchematicError> {
    let Some(children) = node.children() else {
        return Ok(None);
    };
    let Some(bloom_node) = children
        .nodes()
        .iter()
        .find(|n| n.name().value() == "bloom")
    else {
        return Ok(None);
    };

    let preset = match bloom_node.get("preset").and_then(|v| v.as_string()) {
        None | Some("natural") => BloomPreset::Natural,
        Some("old_school") => BloomPreset::OldSchool,
        Some(_) => {
            return Err(KdlSchematicError::InvalidValue {
                property: "preset".to_string(),
                node: "bloom".to_string(),
                expected: "\"natural\" or \"old_school\"".to_string(),
                src: src.to_string(),
                span: bloom_node.span(),
            });
        }
    };

    let float_prop = |prop: &str| {
        bloom_node
            .get(prop)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|v| v as f32)
    };
    let config = BloomConfig {
        preset,
        intensity: float_prop("intensity"),
        threshold: float_prop("threshold"),
        threshold_softness: float_prop("threshold_softness"),
    };

    for (prop, value) in [
        ("intensity", config.intensity),
        ("threshold", config.threshold),
        ("threshold_softness", config.threshold_softness),
    ] {
        if let Some(value) = value
            && value < 0.0
        {
            return Err(KdlSchematicError::InvalidValue {
                property: prop.to_string(),
                node: "bloom".to_string(),
                expected: "a non-negative number".to_string(),
                src: src.to_string(),
                span: bloom_node.span(),
            });
        }
    }

    Ok(Some(config))
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

    let name = parse_name(node);

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

    let locked = node.get("lock").and_then(|v| v.as_bool()).unwrap_or(false);

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
        locked,
        auto_y_range,
        y_range,
        node_id: NodeId::default(),
        colors,
    }))
}

fn parse_component_monitor(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let name = parse_name(node);
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
        name,
    }))
}

fn parse_action_pane(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let name = require_name(node, src)?;

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

    Ok(Panel::ActionPane(ActionPane { name, lua }))
}

fn parse_video_stream(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let msg_name = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .unwrap_or_default()
        .to_string();

    let name = parse_name(node);

    Ok(Panel::VideoStream(VideoStream { msg_name, name }))
}

fn parse_sensor_view(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let msg_name = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .unwrap_or_default()
        .to_string();

    let name = parse_name(node);

    Ok(Panel::SensorView(SensorView { msg_name, name }))
}

fn parse_log_stream(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let msg_name = node
        .entries()
        .iter()
        .find(|e| e.name().is_none())
        .and_then(|e| e.value().as_string())
        .unwrap_or_default()
        .to_string();

    let name = parse_name(node);

    Ok(Panel::LogStream(LogStream { msg_name, name }))
}

fn parse_query_table(node: &KdlNode) -> Result<Panel, KdlSchematicError> {
    let name = parse_name(node);
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

    Ok(Panel::QueryTable(QueryTable {
        name,
        query,
        query_type,
    }))
}

fn parse_query_plot(node: &KdlNode, src: &str) -> Result<Panel, KdlSchematicError> {
    let name = require_name(node, src)?;

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

    // Parse plot mode: "timeseries" (default) or "xy"
    let plot_mode = node
        .get("mode")
        .and_then(|v| v.as_string())
        .map(|s| match s.to_lowercase().as_str() {
            "xy" => PlotMode::XY,
            "timeseries" | "time_series" | "time-series" => PlotMode::TimeSeries,
            _ => PlotMode::TimeSeries,
        })
        .unwrap_or(PlotMode::TimeSeries);

    // Parse optional axis labels
    let x_label = node
        .get("x_label")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    let y_label = node
        .get("y_label")
        .and_then(|v| v.as_string())
        .map(|s| s.to_string());

    Ok(Panel::QueryPlot(QueryPlot {
        name,
        query,
        refresh_interval,
        auto_refresh,
        color,
        query_type,
        plot_mode,
        x_label,
        y_label,
        node_id: NodeId::default(),
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

    let frame = node
        .get("frame")
        .and_then(|v| v.as_string())
        .and_then(|s| GeoFrame::from_str(s).ok());
    let frame_orientation = parse_optional_geo_frame(node, "frame_orientation", "object_3d", src)?;
    let orientation = parse_rotation_kind(node, src)?;
    let mut icon = None;
    let mut mesh_visibility_range = None;
    let mut thrusters = Vec::new();

    let mesh = if let Some(children) = node.children() {
        let children_nodes = children.nodes();
        let mesh_node = children_nodes
            .iter()
            .find(|child| is_object_3d_mesh_node(child.name().value()));
        let mut parsed_mesh = parse_object_3d_mesh(mesh_node, src)?;

        if let Some(mn) = mesh_node {
            mesh_visibility_range = parse_visibility_range_from_children(mn);
        }

        let mut animations = Vec::new();
        for child in children_nodes {
            let child_name = child.name().value();
            if child_name == "animate" {
                let joint_name = child
                    .get("joint")
                    .and_then(|v| v.as_string())
                    .ok_or_else(|| KdlSchematicError::MissingProperty {
                        property: "joint".to_string(),
                        node: "animate".to_string(),
                        src: src.to_string(),
                        span: child.span(),
                    })?
                    .to_string();

                let eql_expr = child
                    .get("rotation_vector")
                    .and_then(|v| v.as_string())
                    .ok_or_else(|| KdlSchematicError::MissingProperty {
                        property: "rotation_vector".to_string(),
                        node: "animate".to_string(),
                        src: src.to_string(),
                        span: child.span(),
                    })?
                    .to_string();

                animations.push(JointAnimation {
                    joint_name,
                    eql_expr,
                });
            } else if child_name == "icon" {
                icon = Some(parse_object_3d_icon(child, src)?);
            } else if child_name == "thruster" {
                thrusters.push(parse_thruster(child, src)?);
            }
        }

        if !animations.is_empty()
            && let Object3DMesh::Glb {
                path,
                scale,
                translate,
                rotate,
                emissivity,
                glow,
                glow_color,
                ..
            } = parsed_mesh
        {
            parsed_mesh = Object3DMesh::Glb {
                path,
                scale,
                translate,
                rotate,
                animations,
                emissivity,
                glow,
                glow_color,
            };
        }

        parsed_mesh
    } else {
        return Err(KdlSchematicError::MissingProperty {
            property: "mesh".to_string(),
            node: "object_3d".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    };

    Ok(Object3D {
        eql,
        mesh,
        frame,
        frame_orientation,
        orientation,
        icon,
        thrusters,
        mesh_visibility_range,
        node_id: NodeId::default(),
    })
}

fn parse_optional_geo_frame(
    node: &KdlNode,
    property: &str,
    node_name: &str,
    src: &str,
) -> Result<Option<GeoFrame>, KdlSchematicError> {
    match node.get(property).and_then(|v| v.as_string()) {
        None => Ok(None),
        Some(value) => {
            GeoFrame::from_str(value)
                .map(Some)
                .map_err(|_| KdlSchematicError::InvalidValue {
                    property: property.to_string(),
                    node: node_name.to_string(),
                    expected: "ENU, NED, or ECEF".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })
        }
    }
}

fn parse_rotation_kind(node: &KdlNode, src: &str) -> Result<RotationKind, KdlSchematicError> {
    match node.get("orientation").and_then(|v| v.as_string()) {
        None => Ok(RotationKind::default()),
        Some(value) => match value.to_ascii_lowercase().as_str() {
            "relative" => Ok(RotationKind::Relative),
            "absolute" => Ok(RotationKind::Absolute),
            _ => Err(KdlSchematicError::InvalidValue {
                property: "orientation".to_string(),
                node: "object_3d".to_string(),
                expected: r#""relative" or "absolute""#.to_string(),
                src: src.to_string(),
                span: node.span(),
            }),
        },
    }
}

fn is_object_3d_mesh_node(name: &str) -> bool {
    matches!(
        name,
        "glb" | "sphere" | "box" | "cylinder" | "plane" | "ellipsoid"
    )
}

fn parse_thruster(node: &KdlNode, src: &str) -> Result<Thruster, KdlSchematicError> {
    let position = parse_tuple3::<f32>(node, "position").ok_or_else(|| {
        KdlSchematicError::MissingProperty {
            property: "position".to_string(),
            node: "thruster".to_string(),
            src: src.to_string(),
            span: node.span(),
        }
    })?;
    let direction = parse_tuple3::<f32>(node, "direction");
    let intensity = node
        .get("intensity")
        .and_then(|v| v.as_string())
        .ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "intensity".to_string(),
            node: "thruster".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?
        .trim()
        .to_string();
    if intensity.is_empty() {
        return Err(KdlSchematicError::InvalidValue {
            property: "intensity".to_string(),
            node: "thruster".to_string(),
            expected: "a non-empty EQL expression".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    }

    let scale = match node.entry("scale") {
        None => Thruster::default_scale(),
        Some(entry) => {
            let value = entry.value();
            if let Some(value) = value.as_float() {
                value as f32
            } else if let Some(value) = value.as_integer() {
                value as f32
            } else {
                return Err(KdlSchematicError::InvalidValue {
                    property: "scale".to_string(),
                    node: "thruster".to_string(),
                    expected: "a numeric value".to_string(),
                    src: src.to_string(),
                    span: entry.span(),
                });
            }
        }
    };

    let light = node
        .children()
        .and_then(|children| {
            children
                .nodes()
                .iter()
                .find(|child| child.name().value() == "light")
        })
        .map(|child| parse_thruster_light(child, src))
        .transpose()?;

    // Repeated `effect "path"` child nodes: stacked layers rendered from the
    // same emitter (e.g. a camera-facing halo over a streaked core).
    let mut extra_effects = Vec::new();
    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() != "effect" {
                continue;
            }
            let path = child
                .entries()
                .iter()
                .find(|entry| entry.name().is_none())
                .and_then(|entry| entry.value().as_string())
                .ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "effect path (positional string)".to_string(),
                    node: "effect".to_string(),
                    src: src.to_string(),
                    span: child.span(),
                })?;
            extra_effects.push(path.to_string());
        }
    }

    Ok(Thruster {
        name: parse_name(node),
        body_frame: bool_prop(node, "body_frame").unwrap_or(false),
        position,
        direction,
        intensity,
        effect: node
            .get("effect")
            .and_then(|v| v.as_string())
            .map(|s| s.to_string())
            .unwrap_or_else(Thruster::default_effect),
        extra_effects,
        emission_rate: node
            .get("emission_rate")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|v| v as f32),
        cutoff: node
            .get("cutoff")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|v| v as f32)
            .unwrap_or_else(Thruster::default_cutoff),
        scale,
        light,
    })
}

fn parse_thruster_light(node: &KdlNode, src: &str) -> Result<ThrusterLight, KdlSchematicError> {
    let float_prop = |name: &str| {
        node.get(name)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .map(|v| v as f32)
    };
    let color =
        parse_tuple3::<f32>(node, "color").ok_or_else(|| KdlSchematicError::MissingProperty {
            property: "color".to_string(),
            node: "light".to_string(),
            src: src.to_string(),
            span: node.span(),
        })?;
    let intensity = float_prop("intensity").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "intensity".to_string(),
        node: "light".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    Ok(ThrusterLight {
        color,
        intensity,
        range: float_prop("range").unwrap_or_else(ThrusterLight::default_range),
        offset: float_prop("offset").unwrap_or(0.0),
        spot_angle: float_prop("spot_angle"),
        shadows: bool_prop(node, "shadows").unwrap_or(false),
    })
}

fn parse_object_3d_icon(node: &KdlNode, src: &str) -> Result<Object3DIcon, KdlSchematicError> {
    let has_path = node.get("path").and_then(|v| v.as_string()).is_some();
    let has_builtin = node.get("builtin").and_then(|v| v.as_string()).is_some();

    let source = if has_path && has_builtin {
        return Err(KdlSchematicError::MissingProperty {
            property: "path OR builtin (not both)".to_string(),
            node: "icon".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    } else if let Some(path) = node.get("path").and_then(|v| v.as_string()) {
        Object3DIconSource::Path(path.to_string())
    } else if let Some(name) = node.get("builtin").and_then(|v| v.as_string()) {
        Object3DIconSource::Builtin(name.to_string())
    } else {
        return Err(KdlSchematicError::MissingProperty {
            property: "path or builtin".to_string(),
            node: "icon".to_string(),
            src: src.to_string(),
            span: node.span(),
        });
    };

    let color = parse_color_from_node_or_children(node, None).unwrap_or_else(default_icon_color);

    let size = node
        .get("size")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32)
        .unwrap_or_else(default_icon_size);

    let visibility_range = parse_visibility_range_from_children(node);

    Ok(Object3DIcon {
        source,
        color,
        size,
        visibility_range,
    })
}

fn parse_visibility_range_from_children(node: &KdlNode) -> Option<VisRange> {
    let children = node.children()?;
    let vr_node = children
        .nodes()
        .iter()
        .find(|n| n.name().value() == "visibility_range")?;

    let min = vr_node
        .get("min")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32)
        .unwrap_or(0.0);
    let max = vr_node
        .get("max")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32)
        .unwrap_or(f32::MAX);
    let fade_distance = vr_node
        .get("fade_distance")
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        .map(|v| v as f32)
        .unwrap_or(0.0);

    Some(VisRange {
        min,
        max,
        fade_distance,
    })
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

    fn numeric_prop(node: &KdlNode, prop: &str) -> Option<f64> {
        node.get(prop)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
    }

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
                })?
                .to_string();

            let scale = node.get("scale").and_then(|v| v.as_float()).unwrap_or(1.0) as f32;

            let translate = parse_tuple3::<f32>(node, "translate").unwrap_or((0.0, 0.0, 0.0));
            let rotate = parse_tuple3::<f32>(node, "rotate").unwrap_or((0.0, 0.0, 0.0));
            let emissivity = numeric_prop(node, "emissivity")
                .map(|v| v as f32)
                .unwrap_or(0.0);
            let glow = numeric_prop(node, "glow").map(|v| v as f32).unwrap_or(0.0);
            let glow_color = node.get("glow_color").and_then(parse_viewport_color);

            Ok(Object3DMesh::Glb {
                path,
                scale,
                translate,
                rotate,
                animations: Vec::new(), // Animations are parsed at object_3d level, not glb level
                emissivity,
                glow,
                glow_color,
            })
        }
        "sphere" => {
            let radius =
                numeric_prop(node, "radius").ok_or_else(|| KdlSchematicError::MissingProperty {
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
            let x = numeric_prop(node, "x").ok_or_else(|| KdlSchematicError::MissingProperty {
                property: "x".to_string(),
                node: "box".to_string(),
                src: src.to_string(),
                span: node.span(),
            })? as f32;

            let y = numeric_prop(node, "y").ok_or_else(|| KdlSchematicError::MissingProperty {
                property: "y".to_string(),
                node: "box".to_string(),
                src: src.to_string(),
                span: node.span(),
            })? as f32;

            let z = numeric_prop(node, "z").ok_or_else(|| KdlSchematicError::MissingProperty {
                property: "z".to_string(),
                node: "box".to_string(),
                src: src.to_string(),
                span: node.span(),
            })? as f32;

            let mesh = Mesh::Box { x, y, z };
            let material = parse_material_from_node(node).unwrap_or(Material::color(1.0, 1.0, 1.0));

            Ok(Object3DMesh::Mesh { mesh, material })
        }
        "cylinder" => {
            let radius =
                numeric_prop(node, "radius").ok_or_else(|| KdlSchematicError::MissingProperty {
                    property: "radius".to_string(),
                    node: "cylinder".to_string(),
                    src: src.to_string(),
                    span: node.span(),
                })? as f32;

            let height =
                numeric_prop(node, "height").ok_or_else(|| KdlSchematicError::MissingProperty {
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
            let size = numeric_prop(node, "size").map(|v| v as f32).unwrap_or(10.0);

            let width = numeric_prop(node, "width")
                .map(|v| v as f32)
                .unwrap_or(size);

            let depth = numeric_prop(node, "depth")
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

            let error_covariance_cholesky = node
                .get("error_covariance_cholesky")
                .and_then(|v| v.as_string())
                .map(str::to_string);

            let error_confidence_interval = node
                .get("error_confidence_interval")
                .and_then(|v| v.as_float())
                .map(|f| f as f32)
                .unwrap_or_else(impeller2_wkt::default_ellipsoid_confidence_interval);

            let show_grid = node
                .get("show_grid")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(impeller2_wkt::default_ellipsoid_show_grid);

            let grid_color = node
                .children()
                .and_then(|c| c.nodes().iter().find(|n| n.name().value() == "grid_color"))
                .and_then(parse_color_from_node)
                .unwrap_or_else(impeller2_wkt::default_ellipsoid_grid_color);

            Ok(Object3DMesh::Ellipsoid {
                scale,
                color,
                error_covariance_cholesky,
                error_confidence_interval,
                show_grid,
                grid_color,
            })
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

    let color = parse_color_from_node_or_children(node, None);
    let future_color = parse_named_color_field(node, "future_color");

    let perspective = node
        .get("perspective")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let frame = node
        .get("frame")
        .and_then(|v| v.as_string())
        .and_then(|s| GeoFrame::from_str(s).ok());

    Ok(Line3d {
        eql,
        line_width,
        color,
        future_color,
        perspective,
        frame,
        node_id: NodeId::default(),
    })
}

/// Parses a named color field (e.g. `future_color`) from a property string or a
/// matching child node. Unlike [`parse_color_from_node_or_children`], this never
/// reads the node's own positional color args.
fn parse_named_color_field(node: &KdlNode, field: &str) -> Option<Color> {
    if let Some(text) = node.get(field).and_then(|v| v.as_string())
        && let Some(color) = parse_color_from_text(text)
    {
        return Some(color);
    }

    if let Some(children) = node.children() {
        for child in children.nodes() {
            if child.name().value() == field
                && let Some(color) = parse_color_from_node(child)
            {
                return Some(color);
            }
        }
    }

    None
}

fn parse_arrow_thickness(node: &KdlNode, src: &str) -> Result<ArrowThickness, KdlSchematicError> {
    let Some(entry) = node.entry("arrow_thickness") else {
        return Ok(ArrowThickness::default());
    };

    let value = entry.value();
    if let Some(value) = value.as_float() {
        return Ok(ArrowThickness::new(value as f32));
    }
    if let Some(value) = value.as_integer() {
        return Ok(ArrowThickness::new(value as f32));
    }
    if let Some(value) = value.as_string() {
        if let Ok(parsed) = value.parse::<f32>() {
            return Ok(ArrowThickness::new(parsed));
        }

        return Err(KdlSchematicError::InvalidValue {
            property: "arrow_thickness".to_string(),
            node: "vector_arrow".to_string(),
            expected: "a numeric value for arrow_thickness (e.g. 1.000)".to_string(),
            src: src.to_string(),
            span: entry.span(),
        });
    }

    Err(KdlSchematicError::InvalidValue {
        property: "arrow_thickness".to_string(),
        node: "vector_arrow".to_string(),
        expected: "a numeric value for arrow_thickness (e.g. 1.000)".to_string(),
        src: src.to_string(),
        span: entry.span(),
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

    let name = parse_name(node);

    let body_frame = node
        .get("body_frame")
        .or_else(|| node.get("in_body_frame"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let normalize = node
        .get("normalize")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let show_name = node
        .get("show_name")
        .or_else(|| node.get("display_name"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let thickness = parse_arrow_thickness(node, src)?;

    let label_position = match node.entry("label_position") {
        None => LabelPosition::None,
        Some(entry) => {
            let value = entry.value();
            if let Some(s) = value.as_string() {
                if let Some(prior) = s.strip_suffix("m") {
                    match prior.parse() {
                        Ok(v) => LabelPosition::Absolute(v),
                        Err(e) => {
                            return Err(KdlSchematicError::InvalidValue {
                                property: "label_position".to_string(),
                                node: "vector_arrow".to_string(),
                                expected: format!(
                                    "a numeric value before the meter marker 'm' but had error: {e}"
                                ),
                                src: src.to_string(),
                                span: entry.span(),
                            });
                        }
                    }
                } else {
                    match s.parse::<f32>() {
                        Ok(v) => {
                            if (0.0..=1.0).contains(&v) {
                                LabelPosition::Proportionate(v)
                            } else {
                                return Err(KdlSchematicError::InvalidValue {
                                    property: "label_position".to_string(),
                                    node: "vector_arrow".to_string(),
                                    expected: format!(
                                        "a numeric value between [0,1] but was {v:.2}"
                                    ),
                                    src: src.to_string(),
                                    span: entry.span(),
                                });
                            }
                        }
                        Err(e) => {
                            return Err(KdlSchematicError::InvalidValue {
                                property: "label_position".to_string(),
                                node: "vector_arrow".to_string(),
                                expected: format!("a numeric value expected but had error: {e}"),
                                src: src.to_string(),
                                span: entry.span(),
                            });
                        }
                    }
                }
            } else {
                let label_position = if let Some(value) = value.as_float() {
                    value as f32
                } else if let Some(value) = value.as_integer() {
                    value as f32
                } else {
                    return Err(KdlSchematicError::InvalidValue {
                        property: "label_position".to_string(),
                        node: "vector_arrow".to_string(),
                        expected: "a numeric value between 0.0 and 1.0".to_string(),
                        src: src.to_string(),
                        span: entry.span(),
                    });
                };
                LabelPosition::Proportionate(label_position.clamp(0.0, 1.0))
            }
        }
    };

    let color = parse_color_from_node_or_children(node, None).unwrap_or(Color::WHITE);

    let frame = node
        .get("frame")
        .and_then(|v| v.as_string())
        .and_then(|s| GeoFrame::from_str(s).ok());

    Ok(VectorArrow3d {
        vector,
        origin,
        scale,
        name,
        color,
        body_frame,
        normalize,
        show_name,
        thickness,
        label_position,
        frame,
        node_id: NodeId::default(),
    })
}

fn parse_color_from_node_or_children(node: &KdlNode, color_tag: Option<&str>) -> Option<Color> {
    // First try to parse color from the node itself
    if let Some(color) = parse_color_from_node(node) {
        return Some(color);
    }

    // If no color found on the node, look for color child nodes
    if let Some(children) = node.children() {
        for child in children.nodes() {
            let name = child.name().value();
            let matches_tag = if let Some(color_tag) = color_tag {
                name == color_tag
            } else {
                // Accept both “color” and the British “colour” spelling for compatibility
                matches!(name, "color" | "colour")
            };

            if matches_tag && let Some(color) = parse_color_from_node(child) {
                return Some(color);
            }
        }
    }

    None
}

fn parse_tuple3<T: FromStr>(node: &KdlNode, property: &str) -> Option<(T, T, T)> {
    let value_str = node.get(property).and_then(|v| v.as_string())?;

    // Parse string like "(1.0, 2.0, 3.0)" or "(1, 2, 3)"
    let trimmed = value_str.trim();
    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return None;
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();

    if parts.len() != 3 {
        return None;
    }

    let x = parts[0].trim().parse::<T>().ok()?;
    let y = parts[1].trim().parse::<T>().ok()?;
    let z = parts[2].trim().parse::<T>().ok()?;

    Some((x, y, z))
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

fn parse_color_from_text(value: &str) -> Option<Color> {
    let trimmed = value.trim();

    if let Some(named) = parse_named_color(trimmed) {
        return Some(named);
    }

    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return None;
    }

    let values: Vec<&str> = trimmed[1..trimmed.len() - 1]
        .split(',')
        .map(|s| s.trim())
        .collect();
    if values.len() < 3 {
        return None;
    }

    let (Some(r), Some(g), Some(b)) = (
        parse_color_component_str(values[0]),
        parse_color_component_str(values[1]),
        parse_color_component_str(values[2]),
    ) else {
        return None;
    };

    let a = values
        .get(3)
        .and_then(|v| parse_color_component_str(v))
        .unwrap_or(1.0);
    Some(Color::rgba(r, g, b, a))
}

fn parse_viewport_color(value: &kdl::KdlValue) -> Option<Color> {
    value.as_string().and_then(parse_color_from_text)
}

fn parse_named_color(name: &str) -> Option<Color> {
    color_from_name(name)
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

fn parse_emissivity_value(value: &kdl::KdlValue) -> Option<f32> {
    let parsed = if let Some(number) = value.as_float() {
        number as f32
    } else if let Some(integer) = value.as_integer() {
        let Ok(integer) = i64::try_from(integer) else {
            return None;
        };
        integer as f32
    } else if let Some(text) = value.as_string() {
        text.parse::<f32>().ok()?
    } else {
        return None;
    };

    Some(parsed.clamp(0.0, 1.0))
}

fn parse_material_from_node(node: &KdlNode) -> Option<Material> {
    parse_color_from_node_or_children(node, None).map(|color| {
        let emissivity = node
            .get("emissivity")
            .and_then(parse_emissivity_value)
            .unwrap_or(0.0);
        Material {
            base_color: color,
            emissivity,
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::ser::serialize_schematic;

    use super::*;

    #[test]
    fn test_parse_timeline_config() {
        let kdl = r#"timeline played_color="mint" future_color="white" follow_latest=#true"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert!(schematic.elems.is_empty());
        let timeline = schematic
            .timeline
            .expect("timeline config should be parsed");
        assert_eq!(timeline.played_color, Color::MINT);
        assert_eq!(timeline.future_color, Color::WHITE);
        assert!(timeline.follow_latest);
    }

    #[test]
    fn test_parse_timeline_range_and_telemetry_mode() {
        let kdl = r#"
timeline follow_latest=#true range="last_5s"
telemetry_mode #true
"#;
        let schematic = parse_schematic(kdl).unwrap();
        let timeline = schematic
            .timeline
            .expect("timeline config should be parsed");
        assert_eq!(timeline.range.as_deref(), Some("last_5s"));
        assert!(schematic.telemetry_mode);
    }

    #[test]
    fn test_parse_timeline_unknown_range_is_rejected() {
        let err = parse_schematic(r#"timeline range="last_3days""#).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("range") || msg.contains("last_3days"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_parse_skybox_config() {
        let schematic = parse_schematic(r#"skybox name="desert_night""#).unwrap();

        assert!(schematic.elems.is_empty());
        assert_eq!(
            schematic
                .skybox
                .expect("skybox config should be parsed")
                .name,
            "desert_night"
        );
    }

    #[test]
    fn test_parse_rc_jet_schematic_has_no_skybox() {
        let kdl = include_str!("../../../../examples/rc-jet/main.py");
        let kdl = kdl
            .split("world.schematic(")
            .nth(1)
            .and_then(|rest| rest.split("\"\"\"").nth(1))
            .expect("rc-jet schematic string");
        let schematic = parse_schematic(kdl).expect("rc-jet schematic should parse");
        assert!(
            schematic.skybox.is_none(),
            "rc-jet schematic must not embed a skybox; activate via the command palette"
        );
    }

    #[test]
    fn test_parse_apollo_lander_schematic_keeps_truth_as_trail_only() {
        let kdl = include_str!("../../../../examples/apollo-lander/apollo-lander.kdl");
        let schematic = parse_schematic(kdl).expect("apollo lander schematic should parse");

        let object_eqls: Vec<_> = schematic
            .elems
            .iter()
            .filter_map(|elem| match elem {
                SchematicElem::Object3d(obj) => Some(obj.eql.as_str()),
                _ => None,
            })
            .collect();

        assert!(object_eqls.contains(&"lander.world_pos"));
        assert!(
            !object_eqls.contains(&"lander_truth.world_pos"),
            "lander_truth should remain replay data/trail, not a second rendered LM"
        );
        assert!(schematic.elems.iter().any(|elem| matches!(
            elem,
            SchematicElem::Line3d(line) if line.eql == "lander_truth.world_pos"
        )));
    }

    #[test]
    fn test_parse_timeline_unknown_properties_are_rejected() {
        let kdl = r#"timeline unexpected=#true"#;
        assert!(parse_schematic(kdl).is_err());
    }

    #[test]
    fn test_parse_timeline_child_color() {
        let kdl = r#"
timeline follow_latest=#true {
  played_color mint
  future_color hyperblue
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        let timeline = schematic
            .timeline
            .expect("timeline config should be parsed");
        assert_eq!(timeline.played_color, Color::MINT);
        assert_eq!(timeline.future_color, Color::HYPERBLUE);
        assert!(timeline.follow_latest);
    }

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
            assert!(viewport.show_view_cube);
        } else {
            panic!("Expected viewport panel");
        }
    }

    #[test]
    fn test_parse_viewport_bool_string_values() {
        // Bare identifiers like `True` are strings in KDL 2.0, not booleans;
        // they previously parsed as false silently.
        for (kdl, expected_hdr) in [
            (r#"viewport hdr=True"#, true),
            (r#"viewport hdr="true""#, true),
            (r#"viewport hdr=#true"#, true),
            (r#"viewport hdr=False"#, false),
            (r#"viewport hdr=#false"#, false),
        ] {
            let schematic = parse_schematic(kdl).unwrap();
            let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
                panic!("Expected viewport panel");
            };
            assert_eq!(viewport.hdr, expected_hdr, "kdl: {kdl}");
        }
    }

    #[test]
    fn test_parse_viewport_with_view_cube_disabled() {
        let kdl = r#"viewport show_view_cube=#false"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] {
            assert!(!viewport.show_view_cube);
            assert!(viewport.effects);
        } else {
            panic!("Expected viewport panel");
        }
    }

    #[test]
    fn test_parse_viewport_effects_disabled() {
        let kdl = r#"viewport effects=#false"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] {
            assert!(!viewport.effects);
        } else {
            panic!("Expected viewport panel");
        }
    }

    #[test]
    fn test_parse_viewport_show_frustums() {
        let kdl = r#"viewport show_frustums=#true"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert!(viewport.show_frustums);
    }

    #[test]
    fn test_parse_viewport_create_frustum() {
        let kdl = r#"viewport create_frustum=#true"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert!(viewport.create_frustum);
    }

    #[test]
    fn test_parse_viewport_show_frustum_legacy() {
        let kdl = r#"viewport show_frustum=#true"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert!(viewport.show_frustums);
    }

    #[test]
    fn test_parse_viewport_near_far() {
        let kdl = r#"viewport near=0.05 far=500.0"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert_eq!(viewport.near, Some(0.05));
        assert_eq!(viewport.far, Some(500.0));
    }

    #[test]
    fn test_parse_viewport_aspect() {
        let kdl = r#"viewport aspect=1.7778"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert_eq!(viewport.aspect, Some(1.7778));
    }

    #[test]
    fn test_parse_viewport_rejects_invalid_near_far_pair() {
        let kdl = r#"viewport near=1.0 far=0.5"#;
        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "far");
                assert_eq!(node, "viewport");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_rejects_invalid_aspect() {
        let kdl = r#"viewport aspect=0.0"#;
        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "aspect");
                assert_eq!(node, "viewport");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_frustums_style() {
        let kdl = r#"viewport show_frustums=#true frustums_color="yalk" projection_color="mint" frustums_thickness=0.012"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport panel");
        };
        assert!(viewport.show_frustums);
        assert_eq!(viewport.frustums_color, Color::YALK);
        assert_eq!(viewport.projection_color, Color::MINT);
        assert!((viewport.frustums_thickness - 0.012).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_viewport_rejects_invalid_frustums_thickness() {
        let kdl = r#"viewport frustums_thickness=0.0"#;
        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "frustums_thickness");
                assert_eq!(node, "viewport");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_rejects_invalid_frustums_color() {
        let kdl = r#"viewport frustums_color="not_a_color""#;
        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "frustums_color");
                assert_eq!(node, "viewport");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_rejects_invalid_projection_color() {
        let kdl = r#"viewport projection_color="not_a_color""#;
        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "projection_color");
                assert_eq!(node, "viewport");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_rejects_partial_color_tuple() {
        let kdl = r#"viewport frustums_color="(255,)""#;
        let err = parse_schematic(kdl).unwrap_err();
        match err {
            KdlSchematicError::InvalidValue { property, .. } => {
                assert_eq!(property, "frustums_color");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_viewport_rejects_two_component_color_tuple() {
        let kdl = r#"viewport frustums_color="(255,128)""#;
        let err = parse_schematic(kdl).unwrap_err();
        match err {
            KdlSchematicError::InvalidValue { property, .. } => {
                assert_eq!(property, "frustums_color");
            }
            other => panic!("Expected invalid value error, got {other:?}"),
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
    fn test_parse_coordinate_with_origin() {
        let kdl = r#"coordinate frame="NED" lat=34.72 lon=-86.64 alt=180"#;
        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.frame, Some(GeoFrame::NED));
        assert_eq!(
            schematic.origin,
            Some(GeoOriginConfig {
                latitude: 34.72,
                longitude: -86.64,
                altitude: 180.0,
            })
        );
    }

    #[test]
    fn test_parse_coordinate_origin_alt_defaults_to_zero() {
        let kdl = r#"coordinate frame="ENU" lat=10.5 lon=20.25"#;
        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(
            schematic.origin,
            Some(GeoOriginConfig {
                latitude: 10.5,
                longitude: 20.25,
                altitude: 0.0,
            })
        );
    }

    #[test]
    fn test_parse_coordinate_without_origin() {
        let schematic = parse_schematic(r#"coordinate frame="NED""#).unwrap();
        assert_eq!(schematic.frame, Some(GeoFrame::NED));
        assert_eq!(schematic.origin, None);
    }

    #[test]
    fn test_parse_coordinate_lat_without_lon_is_an_error() {
        assert!(parse_schematic(r#"coordinate frame="NED" lat=34.72"#).is_err());
        assert!(parse_schematic(r#"coordinate frame="NED" lon=-86.64"#).is_err());
        assert!(parse_schematic(r#"coordinate frame="NED" alt=100.0"#).is_err());
    }

    #[test]
    fn test_parse_object_3d_with_ned_frame() {
        let kdl = r#"
object_3d frame="NED" "ball.world_pos" {
    sphere radius=0.2 {
        color orange
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "ball.world_pos");
            assert!(matches!(obj.frame, Some(GeoFrame::NED)));
            assert_eq!(obj.orientation, RotationKind::Relative);
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_with_absolute_orientation() {
        let kdl = r#"
object_3d frame="NED" orientation=absolute "ball.world_pos" {
    glb path="compass.glb"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "ball.world_pos");
            assert!(matches!(obj.frame, Some(GeoFrame::NED)));
            assert_eq!(obj.orientation, RotationKind::Absolute);
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_invalid_orientation_is_an_error() {
        let kdl = r#"
object_3d frame="NED" orientation=world "ball.world_pos" {
    sphere radius=0.2
}
"#;
        assert!(parse_schematic(kdl).is_err());
    }

    #[test]
    fn test_roundtrip_object_3d_absolute_orientation() {
        let original = r#"
object_3d frame="NED" orientation=absolute "ball.world_pos" {
    sphere radius=0.2 {
        color orange
    }
}
"#;
        let parsed = parse_schematic(original).unwrap();
        let serialized = crate::ser::serialize_schematic(&parsed);
        let reparsed = parse_schematic(&serialized).unwrap();
        if let SchematicElem::Object3d(obj) = &reparsed.elems[0] {
            assert_eq!(obj.orientation, RotationKind::Absolute);
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_with_frame_orientation() {
        let kdl = r#"
object_3d frame="ECEF" frame_orientation="NED" "(0,0,0,1, 0,0,0)" {
    sphere radius=0.2 {
        color orange
    }
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.frame, Some(GeoFrame::ECEF));
            assert_eq!(obj.frame_orientation, Some(GeoFrame::NED));
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_invalid_frame_orientation_is_an_error() {
        let kdl = r#"
object_3d frame_orientation="ECI" "ball.world_pos" {
    sphere radius=0.2
}
"#;
        assert!(parse_schematic(kdl).is_err());
    }

    #[test]
    fn test_roundtrip_object_3d_frame_orientation() {
        let original = r#"
object_3d frame="ECEF" frame_orientation="NED" "ball.world_pos" {
    sphere radius=0.2 {
        color orange
    }
}
"#;
        let parsed = parse_schematic(original).unwrap();
        let serialized = crate::ser::serialize_schematic(&parsed);
        let reparsed = parse_schematic(&serialized).unwrap();
        if let SchematicElem::Object3d(obj) = &reparsed.elems[0] {
            assert_eq!(obj.frame, Some(GeoFrame::ECEF));
            assert_eq!(obj.frame_orientation, Some(GeoFrame::NED));
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_with_enu_frame() {
        let kdl = r#"
object_3d "entity.world_pos" frame="ENU" {
    glb path="model.glb"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "entity.world_pos");
            assert!(matches!(obj.frame, Some(GeoFrame::ENU)));
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_default_frame() {
        let kdl = r#"
object_3d "a.world_pos" {
    sphere radius=0.5
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert!(obj.frame.is_none());
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_vector_thruster() {
        let kdl = r#"
object_3d lander.world_pos {
    sphere radius=0.1
    thruster name="DPS" body_frame=#true position="(0, -0.55, 0)" intensity=lander.main_thrust_viz
}
"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };
        assert_eq!(obj.thrusters.len(), 1);
        let thruster = &obj.thrusters[0];
        assert!(thruster.vector_intensity());
        assert_eq!(thruster.direction, None);
        assert_eq!(thruster.intensity, "lander.main_thrust_viz");
        assert_eq!(thruster.position, (0.0, -0.55, 0.0));
    }

    #[test]
    fn test_parse_object_3d_scalar_thruster_indexed_intensity() {
        // Indexed EQL must be quoted: KDL 2.0 bare strings cannot contain `[`/`]`.
        let kdl = r#"
object_3d lander.world_pos {
    sphere radius=0.1
    thruster name="rcs_0" effect="cold_gas" body_frame=#true position="(2.15, 0.85, 1.45)" direction="(-1, 0, 0)" intensity="lander.rcs_thruster_viz[0]" emission_rate=140.0 cutoff=0.006
}
"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };
        let thruster = &obj.thrusters[0];
        assert!(!thruster.vector_intensity());
        assert_eq!(thruster.intensity, "lander.rcs_thruster_viz[0]");
        assert_eq!(thruster.direction, Some((-1.0, 0.0, 0.0)));
        assert_eq!(thruster.effect, "cold_gas");
    }

    #[test]
    fn test_parse_viewport_with_frame() {
        let kdl =
            r#"viewport name="main" frame="NED" pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,3)""#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] {
            assert_eq!(viewport.name, Some("main".to_string()));
            assert!(matches!(viewport.frame, Some(GeoFrame::NED)));
        } else {
            panic!("Expected viewport panel");
        }
    }

    #[test]
    fn test_parse_line_3d_with_frame() {
        let kdl = r#"line_3d frame="NED" "ball.world_pos" line_width=2.0"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::Line3d(line) = &schematic.elems[0] {
            assert_eq!(line.eql, "ball.world_pos");
            assert!(matches!(line.frame, Some(GeoFrame::NED)));
        } else {
            panic!("Expected line_3d");
        }
    }

    #[test]
    fn test_parse_vector_arrow_with_frame() {
        let kdl = r#"vector_arrow frame="ENU" "ball.velocity" origin="ball.world_pos""#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);
        if let SchematicElem::VectorArrow(arrow) = &schematic.elems[0] {
            assert_eq!(arrow.vector, "ball.velocity");
            assert!(matches!(arrow.frame, Some(GeoFrame::ENU)));
        } else {
            panic!("Expected vector_arrow");
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
    fn test_parse_object_3d_plane_integer_dimensions() {
        let kdl = r#"
object_3d "a.world_pos" {
    plane width=100 depth=200 {
        color 0 255 0
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };

        let Object3DMesh::Mesh { mesh, .. } = &obj.mesh else {
            panic!("Expected mesh object");
        };

        let Mesh::Plane { width, depth } = mesh else {
            panic!("Expected plane mesh");
        };

        assert!((*width - 100.0).abs() < f32::EPSILON);
        assert!((*depth - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_object_3d_integer_primitive_dimensions() {
        type MeshCheck = fn(&Mesh) -> bool;
        let cases: [(&str, MeshCheck); 3] = [
            (
                r#"object_3d "a.world_pos" { sphere radius=50 }"#,
                |mesh: &Mesh| match mesh {
                    Mesh::Sphere { radius } => (*radius - 50.0).abs() < f32::EPSILON,
                    _ => false,
                },
            ),
            (
                r#"object_3d "a.world_pos" { box x=1 y=2 z=3 }"#,
                |mesh: &Mesh| match mesh {
                    Mesh::Box { x, y, z } => {
                        (*x - 1.0).abs() < f32::EPSILON
                            && (*y - 2.0).abs() < f32::EPSILON
                            && (*z - 3.0).abs() < f32::EPSILON
                    }
                    _ => false,
                },
            ),
            (
                r#"object_3d "a.world_pos" { cylinder radius=4 height=5 }"#,
                |mesh: &Mesh| match mesh {
                    Mesh::Cylinder { radius, height } => {
                        (*radius - 4.0).abs() < f32::EPSILON && (*height - 5.0).abs() < f32::EPSILON
                    }
                    _ => false,
                },
            ),
        ];

        for (kdl, validate) in cases {
            let schematic = parse_schematic(kdl).unwrap();
            let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
                panic!("Expected object_3d");
            };
            let Object3DMesh::Mesh { mesh, .. } = &obj.mesh else {
                panic!("Expected mesh object");
            };
            assert!(validate(mesh));
        }
    }

    #[test]
    fn test_parse_object_3d_glb_emissivity() {
        let kdl = r#"object_3d "a.world_pos" { glb path="moon.glb" emissivity=0.5 }"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };
        let Object3DMesh::Glb { emissivity, .. } = &obj.mesh else {
            panic!("Expected glb");
        };
        assert!((*emissivity - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_object_3d_glb_glow() {
        let kdl = r#"object_3d "a.world_pos" { glb path="moon.glb" glow=2.5 glow_color="cyan" }"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };
        let Object3DMesh::Glb {
            glow, glow_color, ..
        } = &obj.mesh
        else {
            panic!("Expected glb");
        };

        assert!((*glow - 2.5).abs() < f32::EPSILON);
        assert_eq!(*glow_color, Some(Color::CYAN));

        let serialized = serialize_schematic(&schematic);
        assert!(serialized.contains("glow=2.5"));
        assert!(
            serialized.contains("glow_color=cyan") || serialized.contains("glow_color=\"cyan\"")
        );
    }

    #[test]
    fn test_parse_viewport_bloom() {
        let kdl = r#"
viewport hdr=#true {
    bloom preset="old_school" intensity=0.4 threshold=1.0 threshold_softness=0.2
}
"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Panel(Panel::Viewport(viewport)) = &schematic.elems[0] else {
            panic!("Expected viewport");
        };
        let bloom = viewport.bloom.as_ref().expect("expected bloom config");
        assert_eq!(bloom.preset, BloomPreset::OldSchool);
        assert_eq!(bloom.intensity, Some(0.4));
        assert_eq!(bloom.threshold, Some(1.0));
        assert_eq!(bloom.threshold_softness, Some(0.2));

        let serialized = serialize_schematic(&schematic);
        assert!(serialized.contains("bloom"));
        assert!(
            serialized.contains("preset=old_school")
                || serialized.contains("preset=\"old_school\"")
        );
        assert!(serialized.contains("intensity=0.4"));
        assert!(serialized.contains("threshold=1.0"));
        assert!(serialized.contains("threshold_softness=0.2"));
    }

    #[test]
    fn test_parse_object_3d_material_emissivity() {
        let kdl = r#"
object_3d "a.world_pos" {
    sphere radius=0.2 emissivity=0.25 {
        color yellow 128
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            let Object3DMesh::Mesh { material, .. } = &obj.mesh else {
                panic!("Expected mesh object");
            };
            assert!((material.base_color.r - Color::YELLOW.r).abs() < f32::EPSILON);
            assert!((material.base_color.g - Color::YELLOW.g).abs() < f32::EPSILON);
            assert!((material.base_color.b - Color::YELLOW.b).abs() < f32::EPSILON);
            assert!((material.base_color.a - 128.0 / 255.0).abs() < f32::EPSILON);
            assert!((material.emissivity - 0.25).abs() < f32::EPSILON);
        } else {
            panic!("Expected object_3d");
        }
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
                Object3DMesh::Ellipsoid {
                    scale,
                    color,
                    error_covariance_cholesky,
                    error_confidence_interval,
                    show_grid,
                    grid_color: _,
                } => {
                    assert_eq!(scale, "rocket.scale");
                    assert!((color.r - 64.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.g - 128.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.b - 1.0).abs() < f32::EPSILON);
                    assert!(error_covariance_cholesky.is_none());
                    assert!((*error_confidence_interval - 70.0).abs() < f32::EPSILON);
                    assert!(!*show_grid);
                    assert!((color.a - 96.0 / 255.0).abs() < f32::EPSILON);
                }
                _ => panic!("Expected ellipsoid mesh"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_ellipsoid_error_covariance() {
        let kdl = r#"
object_3d "satellite.world_pos" {
    ellipsoid error_covariance_cholesky="(1, 0, 1, 0, 0, 1)" error_confidence_interval=95.0 show_grid=#true {
        color 200 200 0 120
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "satellite.world_pos");
            match &obj.mesh {
                Object3DMesh::Ellipsoid {
                    scale: _,
                    color,
                    error_covariance_cholesky,
                    error_confidence_interval,
                    show_grid,
                    grid_color: _,
                } => {
                    assert_eq!(
                        error_covariance_cholesky.as_deref(),
                        Some("(1, 0, 1, 0, 0, 1)")
                    );
                    assert!((*error_confidence_interval - 95.0).abs() < f32::EPSILON);
                    assert!(*show_grid);
                    assert!((color.r - 200.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.a - 120.0 / 255.0).abs() < f32::EPSILON);
                }
                _ => panic!("Expected ellipsoid mesh"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_material_emissivity_property() {
        let kdl = r#"
object_3d "a.world_pos" {
    sphere radius=0.2 emissivity=0.5 {
        color 255 0 0
    }
}
"#;

        let schematic = parse_schematic(kdl).unwrap();
        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            let Object3DMesh::Mesh { material, .. } = &obj.mesh else {
                panic!("Expected mesh object");
            };
            assert!((material.base_color.r - 1.0).abs() < f32::EPSILON);
            assert!((material.emissivity - 0.5).abs() < f32::EPSILON);
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
            let color = line.color.expect("color");
            assert_eq!(color.r, Color::MINT.r);
            assert_eq!(color.g, Color::MINT.g);
            assert_eq!(color.b, Color::MINT.b);
            assert!(!line.perspective);
        } else {
            panic!("Expected line_3d");
        }
    }

    #[test]
    fn test_parse_line_3d_color_optional() {
        // Explicit KDL color must be honored.
        let schematic = parse_schematic(r#"line_3d "traj" { color 0 255 0 }"#).unwrap();
        let SchematicElem::Line3d(line) = &schematic.elems[0] else {
            panic!("Expected line_3d");
        };
        assert_eq!(line.color, Some(Color::GREEN));

        // No color falls back to the timeline trail colors (None at parse time).
        let schematic = parse_schematic(r#"line_3d "traj""#).unwrap();
        let SchematicElem::Line3d(line) = &schematic.elems[0] else {
            panic!("Expected line_3d");
        };
        assert_eq!(line.color, None);
        assert_eq!(line.future_color, None);
    }

    #[test]
    fn test_parse_line_3d_future_color() {
        // Played and future colors can be set independently from KDL.
        let schematic =
            parse_schematic(r#"line_3d "traj" { color 0 255 0; future_color 255 255 255 }"#)
                .unwrap();
        let SchematicElem::Line3d(line) = &schematic.elems[0] else {
            panic!("Expected line_3d");
        };
        assert_eq!(line.color, Some(Color::GREEN));
        assert_eq!(line.future_color, Some(Color::WHITE));

        // `future_color` also accepts a named color as a child node.
        let schematic =
            parse_schematic(r#"line_3d "traj" { color green; future_color green }"#).unwrap();
        let SchematicElem::Line3d(line) = &schematic.elems[0] else {
            panic!("Expected line_3d");
        };
        assert_eq!(line.color, Some(Color::GREEN));
        assert_eq!(line.future_color, Some(Color::GREEN));
    }

    #[test]
    fn test_parse_vector_arrow() {
        let kdl = r#"
vector_arrow "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]" origin="ball.world_pos" scale=1.5 name="Velocity" body_frame=#true normalize=#true show_name=#false arrow_thickness=1.23456 {
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
            assert!(!arrow.show_name);
            assert!(
                (arrow.thickness.value() - 1.235).abs() < 1e-6,
                "unexpected arrow_thickness {}",
                arrow.thickness.value()
            );
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
            assert!(arrow.show_name);
        } else {
            panic!("Expected vector_arrow");
        }
    }

    #[test]
    fn test_parse_vector_arrow_rejects_string_thickness() {
        let kdl = r#"vector_arrow "a.vector" arrow_thickness="not_a_number""#;

        let err = parse_schematic(kdl).unwrap_err();

        match err {
            KdlSchematicError::InvalidValue { property, node, .. } => {
                assert_eq!(property, "arrow_thickness");
                assert_eq!(node, "vector_arrow");
            }
            _ => panic!("Expected InvalidValue error"),
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
                Object3DMesh::Glb {
                    path,
                    scale,
                    translate,
                    rotate,
                    animations,
                    ..
                } => {
                    assert_eq!(path.as_str(), "hi");
                    assert_eq!(*scale, 1.0);
                    assert_eq!(*translate, (0.0, 0.0, 0.0));
                    assert_eq!(*rotate, (0.0, 0.0, 0.0));
                    assert!(animations.is_empty());
                }
                _ => panic!("Expected glb"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_glb_with_animations() {
        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="flappy-rocket.glb"
    animate joint="Root.Fin_0" rotation_vector="(0, 3.14/2, 0)"
    animate joint="Root.Fin_1" rotation_vector="(0, 3.14/2, 0)"
    animate joint="Root.Fin_2" rotation_vector="rocket.fin_deflect"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            assert_eq!(obj.eql, "rocket.world_pos");
            match &obj.mesh {
                Object3DMesh::Glb {
                    path,
                    scale,
                    translate,
                    rotate,
                    animations,
                    ..
                } => {
                    assert_eq!(path.as_str(), "flappy-rocket.glb");
                    assert_eq!(*scale, 1.0);
                    assert_eq!(*translate, (0.0, 0.0, 0.0));
                    assert_eq!(*rotate, (0.0, 0.0, 0.0));
                    assert_eq!(animations.len(), 3);

                    assert_eq!(animations[0].joint_name, "Root.Fin_0");
                    assert_eq!(animations[0].eql_expr, "(0, 3.14/2, 0)");

                    assert_eq!(animations[1].joint_name, "Root.Fin_1");
                    assert_eq!(animations[1].eql_expr, "(0, 3.14/2, 0)");

                    assert_eq!(animations[2].joint_name, "Root.Fin_2");
                    assert_eq!(animations[2].eql_expr, "rocket.fin_deflect");
                }
                _ => panic!("Expected glb"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_parse_object_3d_glb_joint_animation_with_cast_in_rotation_vector() {
        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="model.glb"
    animate joint="Root.Fin_0" rotation_vector="(0, test_fixture0.actual_position.cast(f32)/1000.0 - 22, 0)"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();
        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("Expected object_3d");
        };
        let Object3DMesh::Glb { animations, .. } = &obj.mesh else {
            panic!("Expected glb");
        };
        assert_eq!(animations.len(), 1);
        assert_eq!(animations[0].joint_name, "Root.Fin_0");
        assert_eq!(
            animations[0].eql_expr,
            "(0, test_fixture0.actual_position.cast(f32)/1000.0 - 22, 0)"
        );
    }

    #[test]
    fn test_parse_object_3d_glb_without_animations() {
        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="rocket.glb"
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        assert_eq!(schematic.elems.len(), 1);

        if let SchematicElem::Object3d(obj) = &schematic.elems[0] {
            match &obj.mesh {
                Object3DMesh::Glb { animations, .. } => {
                    assert!(animations.is_empty());
                }
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
    fn test_parse_named_color_red() {
        let kdl = r#"
graph "value" {
    color red
}
"#;
        let schematic = parse_schematic(kdl).unwrap();

        let SchematicElem::Panel(Panel::Graph(graph)) = &schematic.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 1);
        assert_eq!(graph.colors[0], Color::RED);
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
    fn test_parse_world_mesh_lod_count() {
        let kdl = r#"world_mesh "death_valley" lod_count=7"#;
        let schematic = parse_schematic(kdl).unwrap();

        let SchematicElem::WorldMesh(world_mesh) = &schematic.elems[0] else {
            panic!("Expected world_mesh elem");
        };
        assert_eq!(world_mesh.region, "death_valley");
        assert_eq!(world_mesh.lod_count, Some(7));
    }

    #[test]
    fn test_parse_world_mesh_rejects_negative_lod_count() {
        let kdl = r#"world_mesh "globe" lod_count=-1"#;
        let err = parse_schematic(kdl).unwrap_err();
        assert!(matches!(
            err,
            KdlSchematicError::InvalidValue { ref property, .. } if property == "lod_count"
        ));
    }

    #[test]
    fn test_parse_world_mesh_rejects_overflowing_lod_count() {
        let kdl = r#"world_mesh "globe" lod_count=4294967296"#;
        let err = parse_schematic(kdl).unwrap_err();
        assert!(matches!(
            err,
            KdlSchematicError::InvalidValue { ref property, .. } if property == "lod_count"
        ));
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
