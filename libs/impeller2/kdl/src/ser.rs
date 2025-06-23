use impeller2_wkt::{
    ActionPane, Color, ComponentMonitor, Graph, GraphType, Line3d, Material, Mesh, Object3D,
    Object3DMesh, Panel, QueryPlot, QueryTable, QueryType, Schematic, SchematicElem, Split,
    Viewport,
};
use kdl::{KdlDocument, KdlEntry, KdlNode};

pub fn serialize_schematic<T>(schematic: &Schematic<T>) -> String {
    let mut doc = KdlDocument::new();

    for elem in &schematic.elems {
        let node = serialize_schematic_elem(elem);
        doc.nodes_mut().push(node);
    }

    doc.autoformat();
    doc.to_string()
}

fn serialize_schematic_elem<T>(elem: &SchematicElem<T>) -> KdlNode {
    match elem {
        SchematicElem::Panel(panel) => serialize_panel(panel),
        SchematicElem::Object3d(obj) => serialize_object_3d(obj),
        SchematicElem::Line3d(line) => serialize_line_3d(line),
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
        Panel::Dashboard(_) => todo!(),
    }
}

fn serialize_split<T>(split: &Split<T>, is_horizontal: bool) -> KdlNode {
    let node_name = if is_horizontal { "hsplit" } else { "vsplit" };
    let mut node = KdlNode::new(node_name);

    if split.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
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

    if viewport.active {
        node.entries_mut().push(KdlEntry::new_prop("active", true));
    }

    if viewport.show_grid {
        node.entries_mut()
            .push(KdlEntry::new_prop("show_grid", true));
    }

    if viewport.hdr {
        node.entries_mut().push(KdlEntry::new_prop("hdr", true));
    }

    if let Some(ref pos) = viewport.pos {
        node.entries_mut()
            .push(KdlEntry::new_prop("pos", pos.clone()));
    }

    if let Some(ref look_at) = viewport.look_at {
        node.entries_mut()
            .push(KdlEntry::new_prop("look_at", look_at.clone()));
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

    node
}

fn serialize_component_monitor(monitor: &ComponentMonitor) -> KdlNode {
    let mut node = KdlNode::new("component_monitor");
    node.entries_mut().push(KdlEntry::new_prop(
        "component_id",
        monitor.component_id.to_string(),
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
            node.entries_mut().push(KdlEntry::new(path.clone()));
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
        },
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

fn serialize_color_to_node(node: &mut KdlNode, color: &Color) {
    node.entries_mut()
        .push(KdlEntry::new_prop("r", color.r as f64));
    node.entries_mut()
        .push(KdlEntry::new_prop("g", color.g as f64));
    node.entries_mut()
        .push(KdlEntry::new_prop("b", color.b as f64));
}

fn serialize_material_to_node(node: &mut KdlNode, material: &Material) {
    serialize_color_to_node(node, &material.base_color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_schematic;
    use impeller2::types::ComponentId;

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
            assert_eq!(viewport.active, true);
            assert_eq!(viewport.show_grid, true);
        } else {
            panic!("Expected viewport panel");
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
            assert_eq!(line.color.r, Color::MINT.r);
            assert_eq!(line.color.g, Color::MINT.g);
            assert_eq!(line.color.b, Color::MINT.b);
            assert_eq!(line.perspective, false);
        } else {
            panic!("Expected line_3d");
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
    sphere radius=0.2
}
"#;

        let parsed = parse_schematic(original_kdl).unwrap();
        let serialized = serialize_schematic(&parsed);
        let reparsed = parse_schematic(&serialized).unwrap();

        // Check that the structure is preserved
        assert_eq!(parsed.elems.len(), reparsed.elems.len());
    }
}
