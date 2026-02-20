use crate::color_names::{color_to_ints, name_from_color};
use impeller2_wkt::*;
use kdl::{KdlDocument, KdlEntry, KdlNode};

// Default precision for float properties emitted to KDL.
const KDL_FLOAT_PRECISION: u32 = 6;

pub fn serialize_schematic<T>(schematic: &Schematic<T>) -> String {
    let mut doc = KdlDocument::new();

    if let Some(theme) = schematic.theme.as_ref() {
        doc.nodes_mut().push(serialize_theme(theme));
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

fn serialize_schematic_elem<T>(elem: &SchematicElem<T>) -> KdlNode {
    match elem {
        SchematicElem::Panel(panel) => serialize_panel(panel),
        SchematicElem::Object3d(obj) => serialize_object_3d(obj),
        SchematicElem::Line3d(line) => serialize_line_3d(line),
        SchematicElem::VectorArrow(arrow) => serialize_vector_arrow(arrow),
        SchematicElem::Window(window) => serialize_window(window),
        SchematicElem::Theme(theme) => serialize_theme(theme),
    }
}

fn serialize_panel<T>(panel: &Panel<T>) -> KdlNode {
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
        Panel::Dashboard(dashboard) => serialize_dashboard(dashboard),
        Panel::VideoStream(video_stream) => serialize_video_stream(video_stream),
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

fn serialize_split<T>(split: &Split<T>, is_horizontal: bool) -> KdlNode {
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

fn serialize_viewport<T>(viewport: &Viewport<T>) -> KdlNode {
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

    if let Some(ref pos) = viewport.pos {
        node.entries_mut()
            .push(KdlEntry::new_prop("pos", pos.clone()));
    }

    if let Some(ref look_at) = viewport.look_at {
        node.entries_mut()
            .push(KdlEntry::new_prop("look_at", look_at.clone()));
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

fn serialize_graph<T>(graph: &Graph<T>) -> KdlNode {
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

fn serialize_query_plot<T>(query_plot: &QueryPlot<T>) -> KdlNode {
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

fn serialize_object_3d<T>(obj: &Object3D<T>) -> KdlNode {
    let mut node = KdlNode::new("object_3d");

    node.entries_mut().push(KdlEntry::new(obj.eql.clone()));

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
        Object3DMesh::Ellipsoid { scale, color } => {
            let mut node = KdlNode::new("ellipsoid");
            node.entries_mut()
                .push(KdlEntry::new_prop("scale", scale.clone()));

            if color != &default_ellipsoid_color() {
                serialize_color_to_node(&mut node, color);
            }

            (node, Vec::new())
        }
    }
}

fn serialize_line_3d<T>(line: &Line3d<T>) -> KdlNode {
    let mut node = KdlNode::new("line_3d");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(line.eql.clone()));

    if line.line_width != 1.0 {
        push_rounded_float_prop(&mut node, "line_width", line.line_width as f64);
    }

    serialize_color_to_node(&mut node, &line.color);

    if !line.perspective {
        node.entries_mut()
            .push(KdlEntry::new_prop("perspective", false));
    }

    node
}

fn serialize_vector_arrow<T>(arrow: &VectorArrow3d<T>) -> KdlNode {
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

fn serialize_dashboard<T>(dashboard: &Dashboard<T>) -> KdlNode {
    let mut node = serialize_dashboard_node(&dashboard.root);
    node.set_name("dashboard");

    node
}

fn serialize_dashboard_node<T>(dashboard_node: &DashboardNode<T>) -> KdlNode {
    let mut node = KdlNode::new("node");

    serialize_dashboard_node_properties(&mut node, dashboard_node);

    let mut children = KdlDocument::new();

    // Add special child nodes for margin, padding, border if they're not default
    if !is_ui_rect_default(&dashboard_node.margin) {
        let mut margin_node = KdlNode::new("margin");
        serialize_ui_rect_to_node(&mut margin_node, &dashboard_node.margin);
        children.nodes_mut().push(margin_node);
    }

    if !is_ui_rect_default(&dashboard_node.padding) {
        let mut padding_node = KdlNode::new("padding");
        serialize_ui_rect_to_node(&mut padding_node, &dashboard_node.padding);
        children.nodes_mut().push(padding_node);
    }

    if !is_ui_rect_default(&dashboard_node.border) {
        let mut border_node = KdlNode::new("border");
        serialize_ui_rect_to_node(&mut border_node, &dashboard_node.border);
        children.nodes_mut().push(border_node);
    }

    if dashboard_node.color.a > 0.0 {
        let mut bg_node = KdlNode::new("bg");
        serialize_color_to_node(&mut bg_node, &dashboard_node.color);
        children.nodes_mut().push(bg_node);
    }

    // let mut text_color_node = KdlNode::new("text_color");
    // serialize_color_to_node_named(&mut text_color_node, &dashboard_node.text_color, Some("text_color"));
    // children.nodes_mut().push(text_color_node);

    // Add regular children
    for child in &dashboard_node.children {
        children.nodes_mut().push(serialize_dashboard_node(child));
    }

    node.set_children(children);

    serialize_color_to_node_named(&mut node, &dashboard_node.text_color, Some("text_color"));
    node
}

fn serialize_dashboard_node_properties<T>(node: &mut KdlNode, dashboard_node: &DashboardNode<T>) {
    if let Some(name) = dashboard_node.name.as_ref() {
        node.entries_mut()
            .push(KdlEntry::new_prop("name", name.as_str()));
    }

    if !matches!(dashboard_node.display, Display::Flex) {
        node.entries_mut().push(KdlEntry::new_prop(
            "display",
            <&str>::from(dashboard_node.display),
        ));
    }

    if !matches!(dashboard_node.box_sizing, BoxSizing::BorderBox) {
        node.entries_mut().push(KdlEntry::new_prop(
            "box_sizing",
            <&str>::from(dashboard_node.box_sizing),
        ));
    }

    if !matches!(dashboard_node.position_type, PositionType::Relative) {
        node.entries_mut().push(KdlEntry::new_prop(
            "position_type",
            <&str>::from(dashboard_node.position_type),
        ));
    }

    // Serialize Val properties
    serialize_val_prop(node, "left", &dashboard_node.left);
    serialize_val_prop(node, "right", &dashboard_node.right);
    serialize_val_prop(node, "top", &dashboard_node.top);
    serialize_val_prop(node, "bottom", &dashboard_node.bottom);
    serialize_val_prop(node, "width", &dashboard_node.width);
    serialize_val_prop(node, "height", &dashboard_node.height);
    serialize_val_prop(node, "min_width", &dashboard_node.min_width);
    serialize_val_prop(node, "min_height", &dashboard_node.min_height);
    serialize_val_prop(node, "max_width", &dashboard_node.max_width);
    serialize_val_prop(node, "max_height", &dashboard_node.max_height);

    if let Some(aspect_ratio) = dashboard_node.aspect_ratio {
        push_rounded_float_prop(node, "aspect_ratio", aspect_ratio as f64);
    }

    // Serialize alignment properties
    if !matches!(dashboard_node.align_items, AlignItems::Default) {
        node.entries_mut().push(KdlEntry::new_prop(
            "align_items",
            <&str>::from(dashboard_node.align_items),
        ));
    }

    if !matches!(dashboard_node.justify_items, JustifyItems::Default) {
        node.entries_mut().push(KdlEntry::new_prop(
            "justify_items",
            <&str>::from(dashboard_node.justify_items),
        ));
    }

    if !matches!(dashboard_node.align_self, AlignSelf::Auto) {
        node.entries_mut().push(KdlEntry::new_prop(
            "align_self",
            <&str>::from(dashboard_node.align_self),
        ));
    }

    if !matches!(dashboard_node.justify_self, JustifySelf::Auto) {
        node.entries_mut().push(KdlEntry::new_prop(
            "justify_self",
            <&str>::from(dashboard_node.justify_self),
        ));
    }

    if !matches!(dashboard_node.align_content, AlignContent::Default) {
        node.entries_mut().push(KdlEntry::new_prop(
            "align_content",
            <&str>::from(dashboard_node.align_content),
        ));
    }

    if !matches!(dashboard_node.justify_content, JustifyContent::Default) {
        node.entries_mut().push(KdlEntry::new_prop(
            "justify_content",
            <&str>::from(dashboard_node.justify_content),
        ));
    }

    if !matches!(dashboard_node.flex_direction, FlexDirection::Row) {
        node.entries_mut().push(KdlEntry::new_prop(
            "flex_direction",
            <&str>::from(dashboard_node.flex_direction),
        ));
    }

    if !matches!(dashboard_node.flex_wrap, FlexWrap::NoWrap) {
        node.entries_mut().push(KdlEntry::new_prop(
            "flex_wrap",
            <&str>::from(dashboard_node.flex_wrap),
        ));
    }

    if dashboard_node.flex_grow != 0.0 {
        push_rounded_float_prop(node, "flex_grow", dashboard_node.flex_grow as f64);
    }

    if dashboard_node.flex_shrink != 1.0 {
        push_rounded_float_prop(node, "flex_shrink", dashboard_node.flex_shrink as f64);
    }

    serialize_val_prop(node, "flex_basis", &dashboard_node.flex_basis);
    serialize_val_prop(node, "row_gap", &dashboard_node.row_gap);
    serialize_val_prop(node, "column_gap", &dashboard_node.column_gap);

    if let Some(ref text) = dashboard_node.text {
        node.entries_mut()
            .push(KdlEntry::new_prop("text", text.clone()));
    }

    if dashboard_node.font_size != 16.0 {
        push_rounded_float_prop(node, "font_size", dashboard_node.font_size as f64);
    }
}

fn serialize_val_prop(node: &mut KdlNode, prop_name: &str, val: &Val) {
    match val {
        Val::Auto => {}
        Val::Px(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}px", s)));
        }
        Val::Percent(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}%", s)));
        }
        Val::Vw(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}vw", s)));
        }
        Val::Vh(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}vh", s)));
        }
        Val::VMin(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}vmin", s)));
        }
        Val::VMax(s) => {
            node.entries_mut()
                .push(KdlEntry::new_prop(prop_name, format!("{}vmax", s)));
        }
    }
}

fn serialize_ui_rect_to_node(node: &mut KdlNode, ui_rect: &UiRect) {
    serialize_val_prop(node, "left", &ui_rect.left);
    serialize_val_prop(node, "right", &ui_rect.right);
    serialize_val_prop(node, "top", &ui_rect.top);
    serialize_val_prop(node, "bottom", &ui_rect.bottom);
}

fn is_ui_rect_default(ui_rect: &UiRect) -> bool {
    matches!(ui_rect.left, Val::Px(ref s) if s == "0.0")
        && matches!(ui_rect.right, Val::Px(ref s) if s == "0.0")
        && matches!(ui_rect.top, Val::Px(ref s) if s == "0.0")
        && matches!(ui_rect.bottom, Val::Px(ref s) if s == "0.0")
}

#[cfg(test)]
mod tests {
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
    fn test_serialize_simple_viewport() {
        let mut schematic = Schematic::default();
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Viewport(Viewport {
                name: Some("main".to_string()),
                fov: 60.0,
                near: None,
                far: None,
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
                up: None,
                local_arrows: Vec::new(),
                aux: (),
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
                up: None,
                local_arrows: Vec::new(),
                aux: (),
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
                "expected viewport properties in order name → fov → near → far → pos → look_at → hdr → show_grid → show_arrows → create_frustum → show_frustums → frustums_color → frustums_thickness → show_view_cube → active: `{viewport_line}`"
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
                aux: (),
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
                aux: (),
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
            aux: (),
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
            aux: (),
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
            aux: (),
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
            },
            icon: None,
            mesh_visibility_range: None,
            aux: (),
        }));

        let serialized = serialize_schematic(&schematic);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Object3d(obj) = &parsed.elems[0] {
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
    fn test_serialize_tabs_with_children() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Panel(Panel::Tabs(vec![
            Panel::Viewport(Viewport {
                name: Some("camera1".to_string()),
                fov: 45.0,
                near: None,
                far: None,
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
                up: None,
                local_arrows: Vec::new(),
                aux: (),
            }),
            Panel::Graph(Graph {
                eql: "data.position".to_string(),
                name: Some("Position".to_string()),
                graph_type: GraphType::Line,
                locked: false,
                auto_y_range: true,
                y_range: 0.0..1.0,
                aux: (),
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
            aux: (),
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
                aux: (),
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
    fn test_serialize_dashboard_with_font_and_color() {
        let mut schematic = Schematic::default();
        let dashboard = Dashboard {
            root: DashboardNode {
                name: Some("Styled Dashboard".to_string()),
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                text: Some("Hello World".to_string()),
                font_size: 24.0,
                text_color: Color::TURQUOISE,
                children: vec![DashboardNode {
                    width: Val::Px("100.0".to_string()),
                    height: Val::Px("50.0".to_string()),
                    text: Some("Child Text".to_string()),
                    font_size: 12.0,
                    text_color: Color::MINT,
                    ..Default::default()
                }],
                ..Default::default()
            },
            aux: (),
        };
        schematic
            .elems
            .push(SchematicElem::Panel(Panel::Dashboard(Box::new(dashboard))));

        let serialized = serialize_schematic(&schematic);
        println!("{}", serialized);
        let parsed = parse_schematic(&serialized).unwrap();

        assert_eq!(parsed.elems.len(), 1);
        if let SchematicElem::Panel(Panel::Dashboard(dashboard)) = &parsed.elems[0] {
            assert_eq!(dashboard.root.name, Some("Styled Dashboard".to_string()));
            assert_eq!(dashboard.root.font_size, 24.0);
            assert_color_close(dashboard.root.text_color, Color::TURQUOISE);
            assert_eq!(dashboard.root.text, Some("Hello World".to_string()));

            assert_eq!(dashboard.root.children.len(), 1);
            let child_node = &dashboard.root.children[0];
            assert_eq!(child_node.font_size, 12.0);
            assert_color_close(child_node.text_color, Color::MINT);
            assert_eq!(child_node.text, Some("Child Text".to_string()));
        } else {
            panic!("Expected dashboard");
        }
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
