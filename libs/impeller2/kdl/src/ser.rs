use impeller2_wkt::*;
use kdl::{KdlDocument, KdlEntry, KdlNode};

pub fn serialize_schematic<T>(schematic: &Schematic<T>) -> String {
    let mut doc = KdlDocument::new();

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
        Panel::SchematicTree => KdlNode::new("schematic_tree"),
        Panel::Dashboard(dashboard) => serialize_dashboard(dashboard),
    }
}

fn serialize_split<T>(split: &Split<T>, is_horizontal: bool) -> KdlNode {
    let node_name = if is_horizontal { "hsplit" } else { "vsplit" };
    let mut node = KdlNode::new(node_name);

    if split.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
    }

    if let Some(ref name) = split.name {
        node.entries_mut()
            .push(KdlEntry::new_prop("name", name.clone()));
    }

    let mut children = KdlDocument::new();

    for (i, panel) in split.panels.iter().enumerate() {
        let mut child_node = serialize_panel(panel);

        if let Some(&share) = split.shares.get(&i) {
            child_node
                .entries_mut()
                .push(KdlEntry::new_prop("share", share as f64));
        }

        children.nodes_mut().push(child_node);
    }

    node.set_children(children);
    node
}

fn serialize_viewport<T>(viewport: &Viewport<T>) -> KdlNode {
    let mut node = KdlNode::new("viewport");

    if let Some(ref name) = viewport.name {
        node.entries_mut()
            .push(KdlEntry::new_prop("name", name.clone()));
    }

    if viewport.fov != 45.0 {
        node.entries_mut()
            .push(KdlEntry::new_prop("fov", viewport.fov as f64));
    }

    if let Some(ref pos) = viewport.pos {
        node.entries_mut()
            .push(KdlEntry::new_prop("pos", pos.clone()));
    }

    if let Some(ref look_at) = viewport.look_at {
        node.entries_mut()
            .push(KdlEntry::new_prop("look_at", look_at.clone()));
    }

    if viewport.hdr {
        node.entries_mut().push(KdlEntry::new_prop("hdr", true));
    }

    if viewport.show_grid {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_grid", true));
    }

    if viewport.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
    }

    node
}

fn serialize_window(window: &WindowSchematic) -> KdlNode {
    let mut node = KdlNode::new("window");
    node.entries_mut()
        .push(KdlEntry::new_prop("path", window.path.clone()));

    if let Some(title) = &window.title {
        node.entries_mut()
            .push(KdlEntry::new_prop("title", title.clone()));
    }

    node
}

fn serialize_graph<T>(graph: &Graph<T>) -> KdlNode {
    let mut node = KdlNode::new("graph");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(graph.eql.clone()));

    if let Some(ref name) = graph.name {
        node.entries_mut()
            .push(KdlEntry::new_prop("name", name.clone()));
    }

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

    // Only serialize y_range if auto_y_range is false and range is not default
    if !graph.auto_y_range && (graph.y_range.start != 0.0 || graph.y_range.end != 1.0) {
        node.entries_mut()
            .push(KdlEntry::new_prop("y_min", graph.y_range.start));
        node.entries_mut()
            .push(KdlEntry::new_prop("y_max", graph.y_range.end));
    }

    for color in &graph.colors {
        serialize_color_to_node(&mut node, color);
    }

    node
}

fn serialize_component_monitor(monitor: &ComponentMonitor) -> KdlNode {
    let mut node = KdlNode::new("component_monitor");
    node.entries_mut().push(KdlEntry::new_prop(
        "component_name",
        monitor.component_name.clone(),
    ));
    node
}

fn serialize_action_pane(action_pane: &ActionPane) -> KdlNode {
    let mut node = KdlNode::new("action_pane");

    // Add the label as the first unnamed entry
    node.entries_mut()
        .push(KdlEntry::new(action_pane.label.clone()));

    node.entries_mut()
        .push(KdlEntry::new_prop("lua", action_pane.lua.clone()));

    node
}

fn serialize_query_table(query_table: &QueryTable) -> KdlNode {
    let mut node = KdlNode::new("query_table");

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

    // Add the label as the first unnamed entry
    node.entries_mut()
        .push(KdlEntry::new(query_plot.label.clone()));

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

    node
}

fn serialize_object_3d<T>(obj: &Object3D<T>) -> KdlNode {
    let mut node = KdlNode::new("object_3d");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(obj.eql.clone()));

    let mut children = KdlDocument::new();
    children
        .nodes_mut()
        .push(serialize_object_3d_mesh(&obj.mesh));
    node.set_children(children);

    node
}

fn serialize_object_3d_mesh(mesh: &Object3DMesh) -> KdlNode {
    match mesh {
        Object3DMesh::Glb(path) => {
            let mut node = KdlNode::new("glb");
            node.entries_mut()
                .push(KdlEntry::new_prop("path", path.clone()));
            node
        }
        Object3DMesh::Mesh { mesh, material } => match mesh {
            Mesh::Sphere { radius } => {
                let mut node = KdlNode::new("sphere");
                node.entries_mut()
                    .push(KdlEntry::new_prop("radius", *radius as f64));
                serialize_material_to_node(&mut node, material);
                node
            }
            Mesh::Box { x, y, z } => {
                let mut node = KdlNode::new("box");
                node.entries_mut().push(KdlEntry::new_prop("x", *x as f64));
                node.entries_mut().push(KdlEntry::new_prop("y", *y as f64));
                node.entries_mut().push(KdlEntry::new_prop("z", *z as f64));
                serialize_material_to_node(&mut node, material);
                node
            }
            Mesh::Cylinder { radius, height } => {
                let mut node = KdlNode::new("cylinder");
                node.entries_mut()
                    .push(KdlEntry::new_prop("radius", *radius as f64));
                node.entries_mut()
                    .push(KdlEntry::new_prop("height", *height as f64));
                serialize_material_to_node(&mut node, material);
                node
            }
            Mesh::Plane { width, depth } => {
                let mut node = KdlNode::new("plane");
                node.entries_mut()
                    .push(KdlEntry::new_prop("width", *width as f64));
                node.entries_mut()
                    .push(KdlEntry::new_prop("depth", *depth as f64));
                serialize_material_to_node(&mut node, material);
                node
            }
        },
        Object3DMesh::Ellipsoid { scale, color } => {
            let mut node = KdlNode::new("ellipsoid");
            node.entries_mut()
                .push(KdlEntry::new_prop("scale", scale.clone()));

            if color != &default_ellipsoid_color() {
                serialize_color_to_node(&mut node, color);
            }

            node
        }
    }
}

fn serialize_line_3d<T>(line: &Line3d<T>) -> KdlNode {
    let mut node = KdlNode::new("line_3d");

    // Add the EQL query as the first unnamed entry
    node.entries_mut().push(KdlEntry::new(line.eql.clone()));

    if line.line_width != 1.0 {
        node.entries_mut()
            .push(KdlEntry::new_prop("line_width", line.line_width as f64));
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
        node.entries_mut()
            .push(KdlEntry::new_prop("scale", arrow.scale));
    }

    if let Some(name) = &arrow.name {
        node.entries_mut()
            .push(KdlEntry::new_prop("name", name.clone()));
    }

    if arrow.body_frame {
        node.entries_mut()
            .push(KdlEntry::new_prop("body_frame", true));
    }

    if arrow.normalize {
        node.entries_mut()
            .push(KdlEntry::new_prop("normalize", true));
    }

    serialize_color_to_node(&mut node, &arrow.color);

    node
}

fn serialize_color_to_node(node: &mut KdlNode, color: &Color) {
    serialize_color_to_node_named(node, color, None)
}

fn serialize_color_to_node_named(node: &mut KdlNode, color: &Color, name: Option<&str>) {
    let mut color_node = KdlNode::new(name.unwrap_or("color"));

    // Add r, g, b as positional arguments
    let r = float_color_component_to_int(color.r);
    let g = float_color_component_to_int(color.g);
    let b = float_color_component_to_int(color.b);
    color_node.entries_mut().push(KdlEntry::new(r));
    color_node.entries_mut().push(KdlEntry::new(g));
    color_node.entries_mut().push(KdlEntry::new(b));

    // Add alpha if it's not 1.0 (default)
    let a = float_color_component_to_int(color.a);
    if a != 255 {
        color_node.entries_mut().push(KdlEntry::new(a));
    }

    // Add the color node as a child
    if let Some(existing_children) = node.children_mut().as_mut() {
        existing_children.nodes_mut().push(color_node);
    } else {
        let mut doc = KdlDocument::new();
        doc.nodes_mut().push(color_node);
        node.set_children(doc);
    }
}

fn float_color_component_to_int(component: f32) -> i128 {
    let clamped = component.clamp(0.0, 1.0);
    (clamped * 255.0).round() as i128
}

fn serialize_material_to_node(node: &mut KdlNode, material: &Material) {
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
    if let Some(label) = dashboard_node.label.as_ref() {
        node.entries_mut()
            .push(KdlEntry::new_prop("label", label.as_str()));
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
        node.entries_mut()
            .push(KdlEntry::new_prop("aspect_ratio", aspect_ratio as f64));
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
        node.entries_mut().push(KdlEntry::new_prop(
            "flex_grow",
            dashboard_node.flex_grow as f64,
        ));
    }

    if dashboard_node.flex_shrink != 1.0 {
        node.entries_mut().push(KdlEntry::new_prop(
            "flex_shrink",
            dashboard_node.flex_shrink as f64,
        ));
    }

    serialize_val_prop(node, "flex_basis", &dashboard_node.flex_basis);
    serialize_val_prop(node, "row_gap", &dashboard_node.row_gap);
    serialize_val_prop(node, "column_gap", &dashboard_node.column_gap);

    if let Some(ref text) = dashboard_node.text {
        node.entries_mut()
            .push(KdlEntry::new_prop("text", text.clone()));
    }

    if dashboard_node.font_size != 16.0 {
        node.entries_mut().push(KdlEntry::new_prop(
            "font_size",
            dashboard_node.font_size as f64,
        ));
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
                active: true,
                show_grid: true,
                hdr: false,
                pos: None,
                look_at: None,
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
                active: true,
                show_grid: true,
                hdr: true,
                pos: Some("(0,0,0,0, 1,2,3)".to_string()),
                look_at: Some("(0,0,0,0, 0,0,0)".to_string()),
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
            "pos=",
            "look_at=",
            "hdr=",
            "show_grid=",
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
                "expected viewport properties in order name → fov → pos → look_at → hdr → show_grid → active: `{viewport_line}`"
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
    fn test_serialize_object_3d_sphere() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "a.world_pos".to_string(),
            mesh: Object3DMesh::Mesh {
                mesh: Mesh::Sphere { radius: 0.2 },
                material: Material {
                    base_color: Color::rgb(1.0, 0.0, 0.0),
                },
            },
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
                material: Material {
                    base_color: Color::rgb(0.0, 0.5, 1.0),
                },
            },
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
    fn test_serialize_object_3d_ellipsoid() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Object3d(Object3D {
            eql: "rocket.world_pos".to_string(),
            mesh: Object3DMesh::Ellipsoid {
                scale: "rocket.scale".to_string(),
                color: Color::rgba(64.0 / 255.0, 128.0 / 255.0, 1.0, 96.0 / 255.0),
            },
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
                active: false,
                show_grid: false,
                hdr: false,
                pos: None,
                look_at: None,
                aux: (),
            }),
            Panel::Graph(Graph {
                eql: "data.position".to_string(),
                name: Some("Position".to_string()),
                graph_type: GraphType::Line,
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
                aux: (),
            }));

        let serialized = serialize_schematic(&schematic);
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
    sphere radius=0.20000000298023224 {
        color 135 222 158
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
    sphere radius=0.20000000298023224 {
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
                label: Some("Styled Dashboard".to_string()),
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
            assert_eq!(dashboard.root.label, Some("Styled Dashboard".to_string()));
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
}
