//! Sparse layout overlay: split shares and window rects only (Phase 4 / FR-9).
//!
//! The Python source owns structure and EQL; the editor writes this artifact
//! after the user drags splits. `apply_overlay` merges it at build/watch time.

use impeller2_wkt::{Panel, Schematic, SchematicElem, Split, WindowRect, WindowSchematic};
use kdl::{KdlDocument, KdlEntry, KdlNode};

use crate::KdlSchematicError;

/// Conventional sibling of `schematics/main.kdl`.
pub const DEFAULT_OVERLAY_KEY: &str = "schematics/main.overlay.kdl";

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayoutOverlay {
    pub schematic: Option<String>,
    pub splits: Vec<SplitShare>,
    pub windows: Vec<WindowLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitShare {
    pub path: String,
    pub child: usize,
    pub share: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowLayout {
    pub title: Option<String>,
    pub path: Option<String>,
    pub rect: WindowRect,
}

/// `schematics/main.kdl` → `schematics/main.overlay.kdl`.
pub fn overlay_asset_key(schematic_key: &str) -> String {
    schematic_key
        .strip_suffix(".kdl")
        .map(|stem| format!("{stem}.overlay.kdl"))
        .unwrap_or_else(|| format!("{schematic_key}.overlay.kdl"))
}

pub fn parse_overlay(input: &str) -> Result<LayoutOverlay, KdlSchematicError> {
    let doc = input
        .parse::<KdlDocument>()
        .map_err(|source| KdlSchematicError::ParseError {
            source,
            src: input.to_string(),
            span: (0, input.len()).into(),
        })?;

    let mut overlay = LayoutOverlay::default();
    for node in doc.nodes() {
        match node.name().value() {
            "layout" => parse_layout_node(node, input, &mut overlay)?,
            other => {
                return Err(KdlSchematicError::UnknownNode {
                    node_type: other.to_string(),
                    src: input.to_string(),
                    span: node.span(),
                });
            }
        }
    }
    Ok(overlay)
}

fn parse_layout_node(
    node: &KdlNode,
    src: &str,
    overlay: &mut LayoutOverlay,
) -> Result<(), KdlSchematicError> {
    if let Some(schematic) = string_prop(node, "schematic") {
        overlay.schematic = Some(schematic);
    }
    let Some(children) = node.children() else {
        return Ok(());
    };
    for child in children.nodes() {
        match child.name().value() {
            "split" => overlay.splits.push(parse_split_share(child, src)?),
            "window" => overlay.windows.push(parse_window_layout(child, src)?),
            other => {
                return Err(KdlSchematicError::UnknownNode {
                    node_type: other.to_string(),
                    src: src.to_string(),
                    span: child.span(),
                });
            }
        }
    }
    Ok(())
}

fn parse_split_share(node: &KdlNode, src: &str) -> Result<SplitShare, KdlSchematicError> {
    let path = string_prop(node, "path").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "path".to_string(),
        node: "split".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    let child = int_prop(node, "child").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "child".to_string(),
        node: "split".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    let share = float_prop(node, "share").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "share".to_string(),
        node: "split".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    Ok(SplitShare {
        path,
        child: child as usize,
        share: share as f32,
    })
}

fn parse_window_layout(node: &KdlNode, src: &str) -> Result<WindowLayout, KdlSchematicError> {
    let x = int_prop(node, "x").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "x".to_string(),
        node: "window".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    let y = int_prop(node, "y").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "y".to_string(),
        node: "window".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    let width = int_prop(node, "width").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "width".to_string(),
        node: "window".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    let height = int_prop(node, "height").ok_or_else(|| KdlSchematicError::MissingProperty {
        property: "height".to_string(),
        node: "window".to_string(),
        src: src.to_string(),
        span: node.span(),
    })?;
    Ok(WindowLayout {
        title: string_prop(node, "title"),
        path: string_prop(node, "path"),
        rect: WindowRect {
            x: x as u32,
            y: y as u32,
            width: width as u32,
            height: height as u32,
        },
    })
}

pub fn serialize_overlay(overlay: &LayoutOverlay) -> String {
    let mut layout = KdlNode::new("layout");
    if let Some(schematic) = overlay.schematic.as_deref() {
        layout
            .entries_mut()
            .push(KdlEntry::new_prop("schematic", schematic));
    }
    let mut children = KdlDocument::new();
    for split in &overlay.splits {
        let mut node = KdlNode::new("split");
        node.entries_mut()
            .push(KdlEntry::new_prop("path", split.path.as_str()));
        node.entries_mut()
            .push(KdlEntry::new_prop("child", split.child as i128));
        node.entries_mut()
            .push(KdlEntry::new_prop("share", f64::from(split.share)));
        children.nodes_mut().push(node);
    }
    for window in &overlay.windows {
        let mut node = KdlNode::new("window");
        if let Some(title) = window.title.as_deref() {
            node.entries_mut().push(KdlEntry::new_prop("title", title));
        }
        if let Some(path) = window.path.as_deref() {
            node.entries_mut().push(KdlEntry::new_prop("path", path));
        }
        node.entries_mut()
            .push(KdlEntry::new_prop("x", window.rect.x as i128));
        node.entries_mut()
            .push(KdlEntry::new_prop("y", window.rect.y as i128));
        node.entries_mut()
            .push(KdlEntry::new_prop("width", window.rect.width as i128));
        node.entries_mut()
            .push(KdlEntry::new_prop("height", window.rect.height as i128));
        children.nodes_mut().push(node);
    }
    layout.set_children(children);
    let mut doc = KdlDocument::new();
    doc.nodes_mut().push(layout);
    doc.to_string()
}

/// Collect layout-only state from a schematic (what the editor should save).
pub fn extract_overlay(schematic: &Schematic) -> LayoutOverlay {
    let mut overlay = LayoutOverlay::default();
    let mut panel_i = 0usize;
    for elem in &schematic.elems {
        match elem {
            SchematicElem::Panel(panel) => {
                collect_split_shares(panel, &panel_i.to_string(), &mut overlay.splits);
                panel_i += 1;
            }
            SchematicElem::Window(window) => {
                if let Some(rect) = window.screen_rect {
                    overlay.windows.push(WindowLayout {
                        title: window.title.clone(),
                        path: window.path.clone(),
                        rect,
                    });
                }
            }
            _ => {}
        }
    }
    overlay
}

fn collect_split_shares(panel: &Panel, path: &str, out: &mut Vec<SplitShare>) {
    match panel {
        Panel::HSplit(split) | Panel::VSplit(split) => {
            collect_from_split(split, path, out);
        }
        Panel::Tabs(tabs) => {
            for (i, child) in tabs.iter().enumerate() {
                collect_split_shares(child, &format!("{path}/{i}"), out);
            }
        }
        _ => {}
    }
}

fn collect_from_split(split: &Split, path: &str, out: &mut Vec<SplitShare>) {
    let mut keys: Vec<usize> = split.shares.keys().copied().collect();
    keys.sort_unstable();
    for child in keys {
        if let Some(&share) = split.shares.get(&child) {
            out.push(SplitShare {
                path: path.to_string(),
                child,
                share,
            });
        }
    }
    for (i, child) in split.panels.iter().enumerate() {
        collect_split_shares(child, &format!("{path}/{i}"), out);
    }
}

/// Merge overlay layout onto `schematic`. Unknown paths are ignored so a
/// rebuilt tree does not fail the watch loop.
pub fn apply_overlay(schematic: &mut Schematic, overlay: &LayoutOverlay) {
    let mut panel_i = 0usize;
    for elem in &mut schematic.elems {
        match elem {
            SchematicElem::Panel(panel) => {
                apply_split_shares(panel, &panel_i.to_string(), overlay);
                panel_i += 1;
            }
            SchematicElem::Window(window) => {
                if let Some(layout) = matching_window(window, overlay) {
                    window.screen_rect = Some(layout.rect);
                }
            }
            _ => {}
        }
    }
}

fn matching_window<'a>(
    window: &WindowSchematic,
    overlay: &'a LayoutOverlay,
) -> Option<&'a WindowLayout> {
    overlay.windows.iter().find(|layout| {
        match (&layout.title, &window.title, &layout.path, &window.path) {
            (Some(a), Some(b), _, _) if a == b => true,
            (_, _, Some(a), Some(b)) if a == b => true,
            (None, _, None, _) => overlay.windows.len() == 1,
            _ => false,
        }
    })
}

fn apply_split_shares(panel: &mut Panel, path: &str, overlay: &LayoutOverlay) {
    match panel {
        Panel::HSplit(split) | Panel::VSplit(split) => {
            for entry in overlay.splits.iter().filter(|s| s.path == path) {
                split.shares.insert(entry.child, entry.share);
            }
            for (i, child) in split.panels.iter_mut().enumerate() {
                apply_split_shares(child, &format!("{path}/{i}"), overlay);
            }
        }
        Panel::Tabs(tabs) => {
            for (i, child) in tabs.iter_mut().enumerate() {
                apply_split_shares(child, &format!("{path}/{i}"), overlay);
            }
        }
        _ => {}
    }
}

fn string_prop(node: &KdlNode, name: &str) -> Option<String> {
    node.get(name)
        .and_then(|v| v.as_string())
        .map(str::to_string)
}

fn int_prop(node: &KdlNode, name: &str) -> Option<i128> {
    node.get(name).and_then(|v| v.as_integer())
}

fn float_prop(node: &KdlNode, name: &str) -> Option<f64> {
    node.get(name)
        .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_schematic, serialize_schematic};

    fn sample_schematic() -> Schematic {
        parse_schematic(
            r#"
            tabs {
                hsplit name=Flight {
                    viewport name=Chase share=0.55
                    vsplit share=0.45 {
                        graph drone.imu.accel
                    }
                }
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn overlay_key_replaces_kdl_suffix() {
        assert_eq!(
            overlay_asset_key("schematics/main.kdl"),
            "schematics/main.overlay.kdl"
        );
    }

    #[test]
    fn extract_apply_roundtrip_shares() {
        let original = sample_schematic();
        let mut overlay = extract_overlay(&original);
        assert!(
            overlay
                .splits
                .iter()
                .any(|s| s.path == "0/0" && s.child == 0 && (s.share - 0.55).abs() < 1e-5)
        );

        for share in &mut overlay.splits {
            if share.path == "0/0" && share.child == 0 {
                share.share = 0.3;
            }
        }

        let mut rebuilt = parse_schematic(&serialize_schematic(&original)).unwrap();
        apply_overlay(&mut rebuilt, &overlay);
        let text = serialize_overlay(&overlay);
        let parsed = parse_overlay(&text).unwrap();
        assert_eq!(parsed.splits.len(), overlay.splits.len());

        let again = extract_overlay(&rebuilt);
        let chase = again
            .splits
            .iter()
            .find(|s| s.path == "0/0" && s.child == 0)
            .unwrap();
        assert!((chase.share - 0.3).abs() < 1e-5);
    }

    #[test]
    fn apply_ignores_unknown_paths() {
        let mut schematic = sample_schematic();
        let overlay = LayoutOverlay {
            splits: vec![SplitShare {
                path: "9/9".into(),
                child: 0,
                share: 0.1,
            }],
            ..LayoutOverlay::default()
        };
        apply_overlay(&mut schematic, &overlay);
        assert_eq!(schematic, sample_schematic());
    }

    #[test]
    fn window_rect_extract_apply() {
        let mut schematic = Schematic::default();
        schematic.elems.push(SchematicElem::Window(WindowSchematic {
            title: Some("Aux".into()),
            path: None,
            screen: None,
            screen_rect: Some(WindowRect {
                x: 1,
                y: 2,
                width: 100,
                height: 80,
            }),
        }));
        let overlay = extract_overlay(&schematic);
        assert_eq!(overlay.windows.len(), 1);

        let mut empty = Schematic::default();
        empty.elems.push(SchematicElem::Window(WindowSchematic {
            title: Some("Aux".into()),
            ..WindowSchematic::default()
        }));
        apply_overlay(&mut empty, &overlay);
        match &empty.elems[0] {
            SchematicElem::Window(w) => assert_eq!(w.screen_rect.unwrap().width, 100),
            _ => panic!("expected window"),
        }
    }
}
