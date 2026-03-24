use crate::color_names::{color_to_ints, name_from_color};
use impeller2_wkt::*;
use kdl::{KdlDocument, KdlEntry, KdlNode};

// Default precision for float properties emitted to KDL.
const KDL_FLOAT_PRECISION: u32 = 6;

pub fn serialize_schematic(schematic: &Schematic) -> String {
    let mut doc = KdlDocument::new();

    if let Some(theme) = schematic.theme.as_ref() {
        doc.nodes_mut().push(serialize_theme(theme));
    }
    if let Some(timeline) = schematic.timeline.as_ref() {
        doc.nodes_mut().push(serialize_timeline(timeline));
    }

    for elem in &schematic.elems {
        let node = serialize_schematic_elem(elem);
        doc.nodes_mut().push(node);
    }

    doc.autoformat();
    let mut s = doc.to_string();
    s.truncate(s.trim_end().len());
    s
}

fn serialize_schematic_elem(elem: &SchematicElem) -> KdlNode {
    match elem {
        SchematicElem::Panel(panel) => serialize_panel(panel),
        SchematicElem::Object3d(obj) => serialize_object_3d(obj),
        SchematicElem::Line3d(line) => serialize_line_3d(line),
        SchematicElem::VectorArrow(arrow) => serialize_vector_arrow(arrow),
        SchematicElem::Window(window) => serialize_window(window),
        SchematicElem::Theme(theme) => serialize_theme(theme),
        SchematicElem::Timeline(timeline) => serialize_timeline(timeline),
    }
}

fn serialize_panel(panel: &Panel) -> KdlNode {
    match panel {
        Panel::Tabs(panels) => {
            let mut node = KdlNode::new("tabs");
            let mut children = KdlDocument::new();

            for panel in panels {
                children.nodes_mut().push(serialize_panel(panel));
            }

            node.set_children(children);
            node
        }
        Panel::HSplit(split) => serialize_split(split, true),
        Panel::VSplit(split) => serialize_split(split, false),
        Panel::Viewport(viewport) => serialize_viewport(viewport),
        Panel::Graph(graph) => serialize_graph(graph),
        Panel::ComponentMonitor(monitor) => serialize_component_monitor(monitor),
        Panel::ActionPane(action_pane) => serialize_action_pane(action_pane),
        Panel::QueryTable(query_table) => serialize_query_table(query_table),
        Panel::QueryPlot(query_plot) => serialize_query_plot(query_plot),
        Panel::Inspector => KdlNode::new("inspector"),
        Panel::Hierarchy => KdlNode::new("hierarchy"),
        Panel::SchematicTree(name) => {
            let mut node = KdlNode::new("schematic_tree");
            push_optional_name_prop(&mut node, name.as_deref());
            node
        }
        Panel::DataOverview(name) => {
            let mut node = KdlNode::new("data_overview");
            push_optional_name_prop(&mut node, name.as_deref());
            node
        }
        Panel::VideoStream(video_stream) => serialize_video_stream(video_stream),
        Panel::SensorView(sensor_view) => serialize_sensor_view(sensor_view),
        Panel::LogStream(log_stream) => serialize_log_stream(log_stream),
    }
}

fn push_name_prop(node: &mut KdlNode, name: &str) {
    node.entries_mut().push(KdlEntry::new_prop("name", name));
}

fn round_float(value: f64, precision: u32) -> f64 {
    if !value.is_finite() {
        return value;
    }

    let factor = 10_f64.powi(precision as i32);
    let rounded = (value * factor).round() / factor;

    if rounded == 0.0 && rounded.is_sign_negative() {
        0.0
    } else {
        rounded
    }
}

fn round_float_default(value: f64) -> f64 {
    round_float(value, KDL_FLOAT_PRECISION)
}

fn push_rounded_float_prop(node: &mut KdlNode, name: &str, value: f64) {
    node.entries_mut()
        .push(KdlEntry::new_prop(name, round_float_default(value)));
}

fn push_optional_name_prop(node: &mut KdlNode, name: Option<&str>) {
    if let Some(name) = name {
        push_name_prop(node, name);
    }
}

fn serialize_video_stream(video_stream: &VideoStream) -> KdlNode {
    let mut node = KdlNode::new("video_stream");
    node.entries_mut()
        .push(KdlEntry::new(video_stream.msg_name.as_str()));
    push_optional_name_prop(&mut node, video_stream.name.as_deref());
    node
}

fn serialize_sensor_view(sensor_view: &SensorView) -> KdlNode {
    let mut node = KdlNode::new("sensor_view");
    node.entries_mut()
        .push(KdlEntry::new(sensor_view.msg_name.as_str()));
    push_optional_name_prop(&mut node, sensor_view.name.as_deref());
    node
}

fn serialize_log_stream(log_stream: &LogStream) -> KdlNode {
    let mut node = KdlNode::new("log_stream");
    node.entries_mut()
        .push(KdlEntry::new(log_stream.msg_name.as_str()));
    push_optional_name_prop(&mut node, log_stream.name.as_deref());
    node
}

fn serialize_split(split: &Split, is_horizontal: bool) -> KdlNode {
    let node_name = if is_horizontal { "hsplit" } else { "vsplit" };
    let mut node = KdlNode::new(node_name);

    if split.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
    }

    push_optional_name_prop(&mut node, split.name.as_deref());

    let mut children = KdlDocument::new();

    for (i, panel) in split.panels.iter().enumerate() {
        let mut child_node = serialize_panel(panel);

        if let Some(&share) = split.shares.get(&i) {
            push_rounded_float_prop(&mut child_node, "share", share as f64);
        }

        children.nodes_mut().push(child_node);
    }

    node.set_children(children);
    node
}

fn serialize_viewport(viewport: &Viewport) -> KdlNode {
    let mut node = KdlNode::new("viewport");

    push_optional_name_prop(&mut node, viewport.name.as_deref());

    if viewport.fov != 45.0 {
        push_rounded_float_prop(&mut node, "fov", viewport.fov as f64);
    }

    if let Some(near) = viewport.near {
        push_rounded_float_prop(&mut node, "near", near as f64);
    }

    if let Some(far) = viewport.far {
        push_rounded_float_prop(&mut node, "far", far as f64);
    }

    if let Some(aspect) = viewport.aspect {
        push_rounded_float_prop(&mut node, "aspect", aspect as f64);
    }

    if let Some(ref pos) = viewport.pos {
        node.entries_mut()
            .push(KdlEntry::new_prop("pos", pos.clone()));
    }

    if let Some(ref look_at) = viewport.look_at {
        node.entries_mut()
            .push(KdlEntry::new_prop("look_at", look_at.clone()));
    }

    if let Some(frame) = viewport.frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("frame", <&str>::from(frame)));
    }
    if let Some(ref up) = viewport.up {
        node.entries_mut()
            .push(KdlEntry::new_prop("up", up.clone()));
    }

    if viewport.hdr {
        node.entries_mut().push(KdlEntry::new_prop("hdr", true));
    }

    if viewport.show_grid {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_grid", true));
    }

    if !viewport.show_arrows {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_arrows", false));
    }

    if viewport.create_frustum {
        node.entries_mut()
            .push(KdlEntry::new_prop("create_frustum", true));
    }

    if viewport.show_frustums {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_frustums", true));
    }

    if viewport.frustums_color != default_viewport_frustums_color() {
        if let Some(name) = name_from_color(&viewport.frustums_color) {
            node.entries_mut()
                .push(KdlEntry::new_prop("frustums_color", name));
        } else {
            let (r, g, b, a) = color_to_ints(&viewport.frustums_color);
            if a == 255 {
                node.entries_mut().push(KdlEntry::new_prop(
                    "frustums_color",
                    format!("({r},{g},{b})"),
                ));
            } else {
                node.entries_mut().push(KdlEntry::new_prop(
                    "frustums_color",
                    format!("({r},{g},{b},{a})"),
                ));
            }
        }
    }

    if (viewport.frustums_thickness - default_viewport_frustums_thickness()).abs() > f32::EPSILON {
        push_rounded_float_prop(
            &mut node,
            "frustums_thickness",
            viewport.frustums_thickness as f64,
        );
    }

    if !viewport.show_view_cube {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_view_cube", false));
    }

    if viewport.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
    }

    if !viewport.local_arrows.is_empty() {
        let mut children = node.children().cloned().unwrap_or_else(KdlDocument::new);
        for arrow in &viewport.local_arrows {
            children.nodes_mut().push(serialize_vector_arrow(arrow));
        }
        node.set_children(children);
    }

    node
}

fn serialize_window(window: &WindowSchematic) -> KdlNode {
    let mut node = KdlNode::new("window");
    if let Some(path) = &window.path {
        node.entries_mut()
            .push(KdlEntry::new_prop("path", path.clone()));
    }

    if let Some(title) = &window.title {
        node.entries_mut()
            .push(KdlEntry::new_prop("title", title.clone()));
    }

    if let Some(idx) = window.screen {
        node.entries_mut()
            .push(KdlEntry::new_prop("screen", i128::from(idx)));
    }

    if let Some(rect) = window.screen_rect {
        let mut rect_node = KdlNode::new("rect");
        rect_node
            .entries_mut()
            .push(KdlEntry::new(i128::from(rect.x)));
        rect_node
            .entries_mut()
            .push(KdlEntry::new(i128::from(rect.y)));
        rect_node
            .entries_mut()
            .push(KdlEntry::new(i128::from(rect.width)));
        rect_node
            .entries_mut()
            .push(KdlEntry::new(i128::from(rect.height)));

        let mut children = node.children().cloned().unwrap_or_else(KdlDocument::new);
        children.nodes_mut().push(rect_node);
        node.set_children(children);
    }

    node
}

fn serialize_theme(theme: &ThemeConfig) -> KdlNode {
    let mut node = KdlNode::new("theme");
    if let Some(mode) = &theme.mode {
        node.entries_mut()
            .push(KdlEntry::new_prop("mode", mode.clone()));
    }
    if let Some(scheme) = &theme.scheme {
        node.entries_mut()
            .push(KdlEntry::new_prop("scheme", scheme.clone()));
    }
    node
}

fn serialize_timeline(timeline: &TimelineConfig) -> KdlNode {
    let mut node = KdlNode::new("timeline");

    if timeline.played_color != default_timeline_played_color() {
        if let Some(name) = name_from_color(&timeline.played_color) {
            node.entries_mut()
                .push(KdlEntry::new_prop("played_color", name));
        } else {
            let (r, g, b, a) = color_to_ints(&timeline.played_color);
            if a == 255 {
                node.entries_mut()
                    .push(KdlEntry::new_prop("played_color", format!("({r},{g},{b})")));
            } else {
                node.entries_mut().push(KdlEntry::new_prop(
                    "played_color",
                    format!("({r},{g},{b},{a})"),
                ));
            }
        }
    }

    if timeline.future_color != default_timeline_future_color() {
        if let Some(name) = name_from_color(&timeline.future_color) {
            node.entries_mut()
                .push(KdlEntry::new_prop("future_color", name));
        } else {
            let (r, g, b, a) = color_to_ints(&timeline.future_color);
            if a == 255 {
                node.entries_mut()
                    .push(KdlEntry::new_prop("future_color", format!("({r},{g},{b})")));
            } else {
                node.entries_mut().push(KdlEntry::new_prop(
                    "future_color",
                    format!("({r},{g},{b},{a})"),
                ));
            }
        }
    }

    if timeline.follow_latest {
        node.entries_mut()
            .push(KdlEntry::new_prop("follow_latest", true));
    }

    node
}

fn serialize_graph(graph: &Graph) -> KdlNode {
    let mut node = KdlNode::new("graph");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(graph.eql.clone()));

    push_optional_name_prop(&mut node, graph.name.as_deref());

    match graph.graph_type {
        GraphType::Line => {} // Default, don't serialize
        GraphType::Point => {
            node.entries_mut().push(KdlEntry::new_prop("type", "point"));
        }
        GraphType::Bar => {
            node.entries_mut().push(KdlEntry::new_prop("type", "bar"));
        }
    }

    if !graph.auto_y_range {
        node.entries_mut()
            .push(KdlEntry::new_prop("auto_y_range", false));
    }

    if graph.locked {
        node.entries_mut().push(KdlEntry::new_prop("lock", true));
    }

    // Only serialize y_range if auto_y_range is false and range is not default
    if !graph.auto_y_range && (graph.y_range.start != 0.0 || graph.y_range.end != 1.0) {
        push_rounded_float_prop(&mut node, "y_min", graph.y_range.start);
        push_rounded_float_prop(&mut node, "y_max", graph.y_range.end);
    }

    for color in &graph.colors {
        serialize_color_to_node(&mut node, color);
    }

    node
}

fn serialize_component_monitor(monitor: &ComponentMonitor) -> KdlNode {
    let mut node = KdlNode::new("component_monitor");
    push_optional_name_prop(&mut node, monitor.name.as_deref());
    node.entries_mut().push(KdlEntry::new_prop(
        "component_name",
        monitor.component_name.clone(),
    ));
    node
}

fn serialize_action_pane(action_pane: &ActionPane) -> KdlNode {
    let mut node = KdlNode::new("action_pane");

    push_name_prop(&mut node, &action_pane.name);

    node.entries_mut()
        .push(KdlEntry::new_prop("lua", action_pane.lua.clone()));

    node
}

fn serialize_query_table(query_table: &QueryTable) -> KdlNode {
    let mut node = KdlNode::new("query_table");

    push_optional_name_prop(&mut node, query_table.name.as_deref());

    // Add the query as the first unnamed entry
    if !query_table.query.is_empty() {
        node.entries_mut()
            .push(KdlEntry::new(query_table.query.clone()));
    }

    match query_table.query_type {
        QueryType::EQL => {} // Default, don't serialize
        QueryType::SQL => {
            node.entries_mut().push(KdlEntry::new_prop("type", "sql"));
        }
    }

    node
}

fn serialize_query_plot(query_plot: &QueryPlot) -> KdlNode {
    let mut node = KdlNode::new("query_plot");

    push_name_prop(&mut node, &query_plot.name);

    node.entries_mut()
        .push(KdlEntry::new_prop("query", query_plot.query.clone()));

    // Only serialize refresh_interval if it's not the default (1 second)
    if query_plot.refresh_interval.as_millis() != 1000 {
        node.entries_mut().push(KdlEntry::new_prop(
            "refresh_interval",
            query_plot.refresh_interval.as_millis() as i128,
        ));
    }

    if query_plot.auto_refresh {
        node.entries_mut()
            .push(KdlEntry::new_prop("auto_refresh", true));
    }

    serialize_color_to_node(&mut node, &query_plot.color);

    match query_plot.query_type {
        QueryType::EQL => {} // Default, don't serialize
        QueryType::SQL => {
            node.entries_mut().push(KdlEntry::new_prop("type", "sql"));
        }
    }

    // Serialize plot mode (only if not default TimeSeries)
    match query_plot.plot_mode {
        PlotMode::TimeSeries => {} // Default, don't serialize
        PlotMode::XY => {
            node.entries_mut().push(KdlEntry::new_prop("mode", "xy"));
        }
    }

    // Serialize optional axis labels
    if let Some(ref x_label) = query_plot.x_label {
        node.entries_mut()
            .push(KdlEntry::new_prop("x_label", x_label.clone()));
    }

    if let Some(ref y_label) = query_plot.y_label {
        node.entries_mut()
            .push(KdlEntry::new_prop("y_label", y_label.clone()));
    }

    node
}

fn serialize_object_3d(obj: &Object3D) -> KdlNode {
    let mut node = KdlNode::new("object_3d");

    node.entries_mut().push(KdlEntry::new(obj.eql.clone()));

    // Add frame attribute if not default (Bevy)
    if let Some(frame) = obj.frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("frame", <&str>::from(frame)));
    }

    let mut children = KdlDocument::new();
    let (mut mesh_node, sibling_nodes) = serialize_object_3d_mesh(&obj.mesh);

    if let Some(vr) = &obj.mesh_visibility_range {
        serialize_visibility_range_to_node(&mut mesh_node, vr);
    }

    children.nodes_mut().push(mesh_node);
    for sibling in sibling_nodes {
        children.nodes_mut().push(sibling);
    }

    if let Some(icon) = &obj.icon {
        children.nodes_mut().push(serialize_object_3d_icon(icon));
    }

    node.set_children(children);

    node
}

fn serialize_object_3d_icon(icon: &Object3DIcon) -> KdlNode {
    use impeller2_wkt::{Object3DIconSource, default_icon_size};

    let mut node = KdlNode::new("icon");

    match &icon.source {
        Object3DIconSource::Path(path) => {
            node.entries_mut()
                .push(KdlEntry::new_prop("path", path.clone()));
        }
        Object3DIconSource::Builtin(name) => {
            node.entries_mut()
                .push(KdlEntry::new_prop("builtin", name.clone()));
        }
    }

    let is_default_color =
        icon.color.r == 1.0 && icon.color.g == 1.0 && icon.color.b == 1.0 && icon.color.a == 1.0;
    if !is_default_color {
        serialize_color_to_node(&mut node, &icon.color);
    }

    if (icon.size - default_icon_size()).abs() > f32::EPSILON {
        node.entries_mut()
            .push(KdlEntry::new_prop("size", icon.size as f64));
    }

    if let Some(vr) = &icon.visibility_range {
        serialize_visibility_range_to_node(&mut node, vr);
    }

    node
}

fn serialize_visibility_range_to_node(node: &mut KdlNode, vr: &impeller2_wkt::VisRange) {
    let mut vr_node = KdlNode::new("visibility_range");

    if vr.min > 0.0 {
        push_rounded_float_prop(&mut vr_node, "min", vr.min as f64);
    }
    if vr.max < f32::MAX {
        push_rounded_float_prop(&mut vr_node, "max", vr.max as f64);
    }
    if vr.fade_distance > 0.0 {
        push_rounded_float_prop(&mut vr_node, "fade_distance", vr.fade_distance as f64);
    }

    if let Some(existing_children) = node.children_mut().as_mut() {
        existing_children.nodes_mut().push(vr_node);
    } else {
        let mut doc = KdlDocument::new();
        doc.nodes_mut().push(vr_node);
        node.set_children(doc);
    }
}

/// Returns (mesh_node, sibling_nodes) where sibling_nodes are nodes that should be
/// siblings of the mesh node in the object_3d children (e.g., animate nodes)
fn serialize_object_3d_mesh(mesh: &Object3DMesh) -> (KdlNode, Vec<KdlNode>) {
    match mesh {
        Object3DMesh::Glb {
            path,
            scale,
            translate,
            rotate,
            animations,
        } => {
            let mut node = KdlNode::new("glb");
            node.entries_mut()
                .push(KdlEntry::new_prop("path", path.clone()));
            if *scale != 1.0 {
                push_rounded_float_prop(&mut node, "scale", *scale as f64);
            }
            if *translate != (0.0, 0.0, 0.0) {
                let tuple_str = format!("({}, {}, {})", translate.0, translate.1, translate.2);
                node.entries_mut()
                    .push(KdlEntry::new_prop("translate", tuple_str));
            }
            if *rotate != (0.0, 0.0, 0.0) {
                let tuple_str = format!("({}, {}, {})", rotate.0, rotate.1, rotate.2);
                node.entries_mut()
                    .push(KdlEntry::new_prop("rotate", tuple_str));
            }
            // Build animate nodes as siblings (not children of glb)
            let mut animate_nodes = Vec::new();
            for anim in animations {
                let mut anim_node = kdl::KdlNode::new("animate");
                anim_node
                    .entries_mut()
                    .push(kdl::KdlEntry::new_prop("joint", anim.joint_name.clone()));
                anim_node.entries_mut().push(kdl::KdlEntry::new_prop(
                    "rotation_vector",
                    anim.eql_expr.clone(),
                ));
                animate_nodes.push(anim_node);
            }
            (node, animate_nodes)
        }
        Object3DMesh::Mesh { mesh, material } => {
            let node = match mesh {
                Mesh::Sphere { radius } => {
                    let mut node = KdlNode::new("sphere");
                    push_rounded_float_prop(&mut node, "radius", *radius as f64);
                    serialize_material_to_node(&mut node, material);
                    node
                }
                Mesh::Box { x, y, z } => {
                    let mut node = KdlNode::new("box");
                    push_rounded_float_prop(&mut node, "x", *x as f64);
                    push_rounded_float_prop(&mut node, "y", *y as f64);
                    push_rounded_float_prop(&mut node, "z", *z as f64);
                    serialize_material_to_node(&mut node, material);
                    node
                }
                Mesh::Cylinder { radius, height } => {
                    let mut node = KdlNode::new("cylinder");
                    push_rounded_float_prop(&mut node, "radius", *radius as f64);
                    push_rounded_float_prop(&mut node, "height", *height as f64);
                    serialize_material_to_node(&mut node, material);
                    node
                }
                Mesh::Plane { width, depth } => {
                    let mut node = KdlNode::new("plane");
                    push_rounded_float_prop(&mut node, "width", *width as f64);
                    push_rounded_float_prop(&mut node, "depth", *depth as f64);
                    serialize_material_to_node(&mut node, material);
                    node
                }
            };
            (node, Vec::new())
        }
        Object3DMesh::Ellipsoid {
            scale,
            color,
            error_covariance_cholesky,
            error_confidence_interval,
            show_grid,
            grid_color,
        } => {
            let mut node = KdlNode::new("ellipsoid");
            if let Some(cholesky) = error_covariance_cholesky {
                node.entries_mut().push(KdlEntry::new_prop(
                    "error_covariance_cholesky",
                    cholesky.clone(),
                ));
                if *error_confidence_interval
                    != impeller2_wkt::default_ellipsoid_confidence_interval()
                {
                    node.entries_mut().push(KdlEntry::new_prop(
                        "error_confidence_interval",
                        *error_confidence_interval as f64,
                    ));
                }
            } else {
                node.entries_mut()
                    .push(KdlEntry::new_prop("scale", scale.clone()));
            }
            if *show_grid {
                node.entries_mut()
                    .push(KdlEntry::new_prop("show_grid", true));
            }
            if color != &default_ellipsoid_color() {
                serialize_color_to_node(&mut node, color);
            }
            if *show_grid && *grid_color != impeller2_wkt::default_ellipsoid_grid_color() {
                serialize_color_to_node_named(&mut node, grid_color, Some("grid_color"));
            }

            (node, Vec::new())
        }
    }
}

fn serialize_line_3d(line: &Line3d) -> KdlNode {
    let mut node = KdlNode::new("line_3d");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(line.eql.clone()));

    if line.line_width != 1.0 {
        push_rounded_float_prop(&mut node, "line_width", line.line_width as f64);
    }

    if let Some(frame) = line.frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("frame", <&str>::from(frame)));
    }

    serialize_color_to_node(&mut node, &line.color);

    if !line.perspective {
        node.entries_mut()
            .push(KdlEntry::new_prop("perspective", false));
    }

    node
}

fn serialize_vector_arrow(arrow: &VectorArrow3d) -> KdlNode {
    let mut node = KdlNode::new("vector_arrow");
    node.entries_mut().push(KdlEntry::new(arrow.vector.clone()));

    if let Some(origin) = &arrow.origin {
        node.entries_mut()
            .push(KdlEntry::new_prop("origin", origin.clone()));
    }

    if (arrow.scale - 1.0).abs() > f64::EPSILON {
        push_rounded_float_prop(&mut node, "scale", arrow.scale);
    }

    push_optional_name_prop(&mut node, arrow.name.as_deref());

    if arrow.body_frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("body_frame", true));
    }

    if arrow.normalize {
        node.entries_mut()
            .push(KdlEntry::new_prop("normalize", true));
    }

    if !arrow.show_name {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_name", false));
    }

    let thickness = arrow.thickness.value();
    if (thickness - ArrowThickness::default().value()).abs() > f32::EPSILON {
        push_rounded_float_prop(
            &mut node,
            "arrow_thickness",
            ArrowThickness::round_to_precision(thickness) as f64,
        );
    }

    match arrow.label_position {
        LabelPosition::None => {}
        LabelPosition::Proportionate(label_position) => {
            node.entries_mut().push(KdlEntry::new_prop(
                "label_position",
                format!("{:.2}", label_position),
            ));
        }
        LabelPosition::Absolute(length) => {
            node.entries_mut().push(KdlEntry::new_prop(
                "label_position",
                format!("{:.2}m", length),
            ));
        }
    }

    if let Some(frame) = arrow.frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("frame", <&str>::from(frame)));
    }

    serialize_color_to_node(&mut node, &arrow.color);

    node
}

fn serialize_color_to_node(node: &mut KdlNode, color: &Color) {
    serialize_color_to_node_named(node, color, None)
}

fn serialize_color_to_node_named(node: &mut KdlNode, color: &Color, name: Option<&str>) {
    let mut color_node = KdlNode::new(name.unwrap_or("color"));

    let (r, g, b, a) = color_to_ints(color);
    if let Some(named) = name_from_color(color) {
        color_node.entries_mut().push(KdlEntry::new(named));
        if a != 255 {
            color_node.entries_mut().push(KdlEntry::new(a));
        }
    } else {
        color_node.entries_mut().push(KdlEntry::new(r));
        color_node.entries_mut().push(KdlEntry::new(g));
        color_node.entries_mut().push(KdlEntry::new(b));
        if a != 255 {
            color_node.entries_mut().push(KdlEntry::new(a));
        }
    }

    if let Some(existing_children) = node.children_mut().as_mut() {
        existing_children.nodes_mut().push(color_node);
    } else {
        let mut doc = KdlDocument::new();
        doc.nodes_mut().push(color_node);
        node.set_children(doc);
    }
}

fn serialize_material_to_node(node: &mut KdlNode, material: &Material) {
    let emissivity = material.emissivity.clamp(0.0, 1.0);
    if emissivity > 0.0 {
        push_rounded_float_prop(node, "emissivity", emissivity as f64);
    }
    serialize_color_to_node(node, &material.base_color);
}

#[cfg(test)]
mod tests {

    use bevy_geo_frames::GeoFrame;
    use super::*;
    use crate::parse_schematic;

    const COLOR_EPSILON: f32 = 1.0 / 255.0 + 1e-6;

    fn assert_color_close(actual: Color, expected: Color) {
        assert!(
            (actual.r - expected.r).abs() <= COLOR_EPSILON,
            "expected r ~= {} got {}",
            expected.r,
            actual.r
        );
        assert!(
            (actual.g - expected.g).abs() <= COLOR_EPSILON,
            "expected g ~= {} got {}",
            expected.g,
            actual.g
        );
        assert!(
            (actual.b - expected.b).abs() <= COLOR_EPSILON,
            "expected b ~= {} got {}",
            expected.b,
            actual.b
        );
        assert!(
            (actual.a - expected.a).abs() <= COLOR_EPSILON,
            "expected a ~= {} got {}",
            expected.a,
            actual.a
        );
    }

    #[test]
    fn test_serialize_timeline_config() {
        let schematic = Schematic {
            timeline: Some(TimelineConfig {
                played_color: Color::MINT,
                future_color: Color::HYPERBLUE,
                follow_latest: true,
            }),
            ..Default::default()
        };

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert!(serialized.contains("timeline"));
        assert!(serialized.contains("played_color="));
        assert!(serialized.contains("future_color="));
        assert!(serialized.contains("follow_latest=#true"));

        let timeline = parsed.timeline.expect("timeline config should roundtrip");
        assert_eq!(timeline.played_color, Color::MINT);
        assert_eq!(timeline.future_color, Color::HYPERBLUE);
        assert!(timeline.follow_latest);
    }

    #[test]
    fn test_serialize_simple_viewport() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Viewport(Viewport {
                name: Some("main".to_string()),
                fov: 60.0,
                near: None,
                far: None,
                aspect: None,
                active: true,
                show_grid: true,
                show_arrows: true,
                create_frustum: false,
                show_frustums: false,
                frustums_color: default_viewport_frustums_color(),
                frustums_thickness: default_viewport_frustums_thickness(),
                show_view_cube: true,
                hdr: false,
                pos: None,
                look_at: None,
                frame: None,
                up: None,
                local_arrows: Vec::new(),
                node_id: NodeId::default(),
            })));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Viewport(viewport)) = &parsed.elems[0] {
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
    fn test_viewport_property_order() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Viewport(Viewport {
                name: Some("main".to_string()),
                fov: 60.0,
                near: Some(0.05),
                far: Some(500.0),
                aspect: Some(1.7778),
                active: true,
                show_grid: true,
                show_arrows: false,
                create_frustum: true,
                show_frustums: true,
                frustums_color: Color::YALK,
                frustums_thickness: 0.012,
                show_view_cube: false,
                hdr: true,
                pos: Some("(0,0,0,0, 1,2,3)".to_string()),
                look_at: Some("(0,0,0,0, 0,0,0)".to_string()),
                frame: None,
                up: None,
                local_arrows: Vec::new(),
                node_id: NodeId::default(),
            })));

        let serialized = serialize_schematic(&schematic);
        let viewport_line = serialized
            .lines()
            .find(|line| line.trim_start().starts_with("viewport"))
            .expect("viewport line missing");

        let properties = [
            "name=",
            "fov=",
            "near=",
            "far=",
            "aspect=",
            "pos=",
            "look_at=",
            "hdr=",
            "show_grid=",
            "show_arrows=",
            "create_frustum=",
            "show_frustums=",
            "frustums_color=",
            "frustums_thickness=",
            "show_view_cube=",
            "active=",
        ];
        let mut indices = Vec::with_capacity(properties.len());
        for property in properties {
            let idx = viewport_line
                .find(property)
                .unwrap_or_else(|| panic!("{property} missing in `{viewport_line}`"));
            indices.push(idx);
        }

        for window in indices.windows(2) {
            assert!(
                window[0] < window[1],
                "expected viewport properties in order name → fov → near → far → aspect → pos → look_at → hdr → show_grid → show_arrows → create_frustum → show_frustums → frustums_color → frustums_thickness → show_view_cube → active: `{viewport_line}`"
            );
        }
    }

    #[test]
    fn test_serialize_graph() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Graph(Graph {
                eql: "a.world_pos".to_string(),
                name: Some("Position Graph".to_string()),
                graph_type: GraphType::Line,
                locked: false,
                auto_y_range: true,
                y_range: 0.0..1.0,
                node_id: NodeId::default(),
                colors: vec![],
            })));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Graph(graph)) = &parsed.elems[0] {
            assert_eq!(graph.eql, "a.world_pos");
            assert_eq!(graph.name, Some("Position Graph".to_string()));
            assert_eq!(graph.graph_type, GraphType::Line);
        } else {
            panic!("Expected graph panel");
        }
    }

    #[test]
    fn test_serialize_graph_with_colors() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Graph(Graph {
                eql: "rocket.fins[2], rocket.fins[3]".to_string(),
                name: None,
                graph_type: GraphType::Line,
                locked: false,
                auto_y_range: true,
                y_range: 0.0..1.0,
                node_id: NodeId::default(),
                colors: vec![Color::rgb(1.0, 0.0, 0.0), Color::rgb(0.0, 1.0, 0.0)],
            })));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        let SchematicElem::Panel(Panel::Graph(graph)) = &parsed.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 2);
        assert_eq!(graph.colors[0], Color::rgb(1.0, 0.0, 0.0));
        assert_eq!(graph.colors[1], Color::rgb(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_roundtrip_named_color_red_is_serialized() {
        let original = r#"
graph "value" {
    color red
}
"#;

        let parsed = parse_schematic(original).unwrap();
        let serialized = serialize_schematic(&parsed);

        assert!(
            serialized.contains("color red"),
            "serialized output should emit named red, got:\n{serialized}"
        );

        let reparsed = parse_schematic(&serialized).unwrap();
        let SchematicElem::Panel(Panel::Graph(graph)) = &reparsed.elems[0] else {
            panic!("Expected graph panel");
        };
        assert_eq!(graph.colors.len(), 1);
        assert_eq!(graph.colors[0], Color::RED);
    }

    #[test]
    fn test_serialize_object_3d_sphere() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "a.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Sphere { radius: 0.2 },
                material: Material::with_color(Color::rgb(1.0, 0.0, 0.0)),
            },
            icon: None,
            mesh_visibility_range: None,
            frame:None,
            node_id: NodeId::default(),
        }));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &parsed.elems[0] {
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
    fn test_serialize_object_3d_plane() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "a.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Plane {
                    width: 15.0,
                    depth: 20.0,
                },
                material: Material::with_color(Color::rgb(0.0, 0.5, 1.0)),
            },
            icon: None,
            mesh_visibility_range: None,
            frame:None,
            node_id: NodeId::default(),
        }));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        let SchematicElem::Object3d(obj) = &parsed.elems[0] else {
            panic!("Expected object_3d");
        };

        let Object3DMesh::Mesh { mesh, material } = &obj.mesh else {
            panic!("Expected mesh object");
        };

        let Mesh::Plane { width, depth } = mesh else {
            panic!("Expected plane mesh");
        };

        assert!((*width - 15.0).abs() < f32::EPSILON);
        assert!((*depth - 20.0).abs() < f32::EPSILON);
        assert_eq!(material.base_color.r, 0.0);
        assert!((material.base_color.g - 128.0 / 255.0).abs() < f32::EPSILON);
        assert_eq!(material.base_color.b, 1.0);
    }

    #[test]
    fn test_serialize_object_3d_material_emissivity() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "a.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Sphere { radius: 0.2 },
                material: Material::color_with_emissivity(1.0, 1.0, 0.0, 0.25),
            },
            icon: None,
            mesh_visibility_range: None,
            frame: None,
            node_id: NodeId::default(),
        }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("emissivity=0.25"),
            "serialized output should expose emissivity on the mesh node, got:\n{serialized}"
        );

        let parsed = parse_schematic(&serialized).unwrap();
        let SchematicElem::Object3d(obj) = &parsed.elems[0] else {
            panic!("Expected object_3d");
        };
        let Object3DMesh::Mesh { material, .. } = &obj.mesh else {
            panic!("Expected mesh object");
        };
        assert!((material.emissivity - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialize_object_3d_ellipsoid() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "rocket.world_pos".to_string(),
            mesh: Object3DMesh::Ellipsoid {
                scale: "rocket.scale".to_string(),
                color: Color::rgba(64.0 / 255.0, 128.0 / 255.0, 1.0, 96.0 / 255.0),
                error_covariance_cholesky: None,
                error_confidence_interval: impeller2_wkt::default_ellipsoid_confidence_interval(),
                show_grid: impeller2_wkt::default_ellipsoid_show_grid(),
                grid_color: impeller2_wkt::default_ellipsoid_grid_color(),
            },
            icon: None,
            mesh_visibility_range: None,
            frame: None,
            node_id: NodeId::default(),
        }));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &parsed.elems[0] {
            assert_eq!(obj.eql, "rocket.world_pos");
            match &obj.mesh {
                Object3DMesh::Ellipsoid {
                    scale,
                    color,
                    error_covariance_cholesky,
                    error_confidence_interval: _,
                    show_grid: _,
                    grid_color: _,
                } => {
                    assert_eq!(scale, "rocket.scale");
                    assert!((color.r - 64.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.g - 128.0 / 255.0).abs() < f32::EPSILON);
                    assert!((color.b - 1.0).abs() < f32::EPSILON);
                    assert!((color.a - 96.0 / 255.0).abs() < f32::EPSILON);
                    assert!(error_covariance_cholesky.is_none());
                }
                _ => panic!("Expected ellipsoid mesh"),
            }
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_serialize_object_3d_with_frame() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "ball.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Sphere { radius: 0.2 },
                material: Material::with_color(Color::ORANGE),
            },
            frame: Some(GeoFrame::NED),
            mesh_visibility_range: None,
            icon: None,
            aux: (),
        }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("frame=NED") || serialized.contains(r#"frame="NED""#),
            "serialized output should contain frame=NED, got:\n{serialized}"
        );

        let parsed = parse_schematic(&serialized).unwrap();
        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &parsed.elems[0] {
            assert_eq!(obj.eql, "ball.world_pos");
            assert!(matches!(obj.frame, Some(GeoFrame::NED)));
        } else {
            panic!("Expected object_3d");
        }
    }

    #[test]
    fn test_serialize_object_3d_default_frame_not_serialized() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "entity.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Sphere { radius: 0.5 },
                material: Material::with_color(Color::WHITE),
            },
            frame: None, // Default (no frame)
            icon: None,
            mesh_visibility_range: None,
            aux: (),
        }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            !serialized.contains("frame="),
            "default None frame should not be serialized, got:\n{serialized}"
        );
    }

    #[test]
    fn test_serialize_viewport_with_frame() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Viewport(Viewport {
                name: Some("main".to_string()),
                fov: 45.0,
                active: false,
                show_grid: false,
                show_arrows: true,
                hdr: false,
                pos: Some("(0,0,0,0, 8,2,4)".to_string()),
                look_at: None,
                frame: Some(GeoFrame::NED),
                local_arrows: Vec::new(),
                aspect: None,
                aux: (),
                ..Default::default()
            })));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("frame=NED") || serialized.contains(r#"frame="NED""#),
            "serialized output should contain frame=NED, got:\n{serialized}"
        );
    }

    #[test]
    fn test_serialize_line_3d_with_frame() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Line3d(Line3d {
            eql: "ball.world_pos".to_string(),
            line_width: 2.0,
            color: Color::WHITE,
            perspective: true,
            frame: Some(GeoFrame::ENU),
            aux: (),
        }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("frame=ENU") || serialized.contains(r#"frame="ENU""#),
            "serialized output should contain frame=ENU, got:\n{serialized}"
        );
    }

    #[test]
    fn test_serialize_vector_arrow_with_frame() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::VectorArrow(VectorArrow3d {
                vector: "ball.velocity".to_string(),
                origin: Some("ball.world_pos".to_string()),
                scale: 1.0,
                name: None,
                color: Color::WHITE,
                body_frame: false,
                normalize: false,
                show_name: true,
                thickness: ArrowThickness::default(),
                label_position: LabelPosition::None,
                frame: Some(GeoFrame::ECEF),
                aux: (),
            }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("frame=ECEF") || serialized.contains(r#"frame="ECEF""#),
            "serialized output should contain frame=ECEF, got:\n{serialized}"
        );
    }

    #[test]
    fn test_serialize_tabs_with_children() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Panel(Panel::Tabs(vec![
            Panel::Viewport(Viewport {
                name: Some("camera1".to_string()),
                fov: 45.0,
                near: None,
                far: None,
                aspect: None,
                active: false,
                show_grid: false,
                show_arrows: true,
                create_frustum: false,
                show_frustums: false,
                frustums_color: default_viewport_frustums_color(),
                frustums_thickness: default_viewport_frustums_thickness(),
                show_view_cube: true,
                hdr: false,
                pos: None,
                look_at: None,
                frame: None,
                up: None,
                local_arrows: Vec::new(),
                node_id: NodeId::default(),
            }),
            Panel::Graph(Graph {
                eql: "data.position".to_string(),
                name: Some("Position".to_string()),
                graph_type: GraphType::Line,
                locked: false,
                auto_y_range: true,
                y_range: 0.0..1.0,
                node_id: NodeId::default(),
                colors: vec![],
            }),
        ])));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Tabs(tabs)) = &parsed.elems[0] {
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
    fn test_serialize_line_3d() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Line3d(Line3d {
            eql: "trajectory".to_string(),
            line_width: 2.0,
            color: Color::MINT,
            perspective: false,
            frame: None,
            node_id: NodeId::default(),
        }));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Line3d(line) = &parsed.elems[0] {
            assert_eq!(line.eql, "trajectory");
            assert_eq!(line.line_width, 2.0);
            assert_color_close(line.color, Color::MINT);
            assert!(!line.perspective);
        } else {
            panic!("Expected line_3d");
        }
    }

    #[test]
    fn test_serialize_vector_arrow() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::VectorArrow(VectorArrow3d {
                vector: "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]".to_string(),
                origin: Some("ball.world_pos".to_string()),
                scale: 2.5,
                name: Some("Velocity".to_string()),
                color: Color::BLUE,
                body_frame: true,
                normalize: true,
                show_name: false,
                thickness: ArrowThickness::new(1.23456),
                label_position: LabelPosition::None,
                frame: None,
                node_id: NodeId::default(),
            }));

        let serialized = serialize_schematic(&schematic);
        assert!(
            serialized.contains("arrow_thickness=1.235"),
            "arrow_thickness should serialize as a numeric value rounded to 3 decimals: {serialized}"
        );
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::VectorArrow(arrow) = &parsed.elems[0] {
            assert_eq!(
                arrow.vector,
                "ball.world_vel[3],ball.world_vel[4],ball.world_vel[5]"
            );
            assert_eq!(arrow.origin.as_deref(), Some("ball.world_pos"));
            assert_eq!(arrow.scale, 2.5);
            assert_eq!(arrow.name.as_deref(), Some("Velocity"));
            assert!(arrow.body_frame);
            assert!(arrow.normalize);
            assert!(!arrow.show_name);
            assert!(
                (arrow.thickness.value() - 1.235).abs() < 1e-6,
                "unexpected thickness after roundtrip {}",
                arrow.thickness.value()
            );
            assert_color_close(arrow.color, Color::BLUE);
        } else {
            panic!("Expected vector_arrow");
        }
    }

    #[test]
    fn test_roundtrip_complex_example() {
        let original_kdl = r#"
tabs {
    viewport fov=45.0 active=#true show_grid=#false hdr=#true
    graph "a.world_pos" name="a world_pos"
}

object_3d "a.world_pos" {
    sphere radius=0.2 {
        color mint
    }
}
"#;

        let parsed = parse_schematic(original_kdl).unwrap();
        let serialized = serialize_schematic(&parsed);
        assert!(
            serialized.contains("color mint") || serialized.contains("color 135 222 158"),
            "serialized output should mention either the mint name or its RGBA components, got:\n{serialized}"
        );
        let reparsed = parse_schematic(&serialized).unwrap();

        // Check that the structure is preserved
        assert_eq!(parsed.elems.len(), reparsed.elems.len());
    }

    #[test]
    fn test_roundtrip_complex_example_color_tuple() {
        let original_kdl = r#"
tabs {
    viewport fov=45.0 active=#true show_grid=#false hdr=#true
    graph "a.world_pos" name="a world_pos"
}

object_3d "a.world_pos" {
    sphere radius=0.2 {
        color 255 0 255
    }
}
"#;
        let parsed = parse_schematic(original_kdl).unwrap();
        let serialized = serialize_schematic(&parsed);
        // NOTE: fov and grid are dropped because they are the default value.
        //
        //viewport hdr=#true show_grid=#false active=#true
        assert_eq!(
            r#"
tabs {
    viewport hdr=#true active=#true
    graph a.world_pos name="a world_pos"
}
object_3d a.world_pos {
    sphere radius=0.2 {
        color 255 0 255
    }
}"#
            .trim(),
            serialized
        );
        let reparsed = parse_schematic(&serialized).unwrap();

        // Check that the structure is preserved
        assert_eq!(parsed.elems.len(), reparsed.elems.len());
    }

    #[test]
    fn test_roundtrip_rocket_example() {
        let original_kdl = r#"graph "rocket.fin_deflect[0]" name=Fin "#;
        let parsed = parse_schematic(original_kdl).unwrap();
        let serialized = serialize_schematic(&parsed);
        assert_eq!(r#"graph "rocket.fin_deflect[0]" name=Fin"#, serialized);
        let reparsed = parse_schematic(&serialized).unwrap();
        assert_eq!(parsed.elems.len(), reparsed.elems.len());
    }

    #[test]
    fn test_roundtrip_glb_animations() {
        use impeller2_wkt::{Object3DMesh, SchematicElem};
        let original = r#"
object_3d "rocket.world_pos" {
    glb path="rocket.glb"
    animate joint="Root.Fin_0" rotation_vector="(0, 1.0, 0)"
}
"#;
        let parsed = parse_schematic(original).unwrap();
        let SchematicElem::Object3d(parsed_obj) = &parsed.elems[0] else {
            panic!("Expected Object3d in parsed")
        };
        let Object3DMesh::Glb {
            animations: parsed_anims,
            ..
        } = &parsed_obj.mesh
        else {
            panic!("Expected Glb mesh in parsed")
        };
        assert_eq!(parsed_anims.len(), 1);
        let serialized = serialize_schematic(&parsed);
        let reparsed = parse_schematic(&serialized).unwrap();
        let SchematicElem::Object3d(obj) = &reparsed.elems[0] else {
            panic!()
        };
        let Object3DMesh::Glb { animations, .. } = &obj.mesh else {
            panic!()
        };
        assert_eq!(animations.len(), 1, "serialized:\n{serialized}");
    }
}
