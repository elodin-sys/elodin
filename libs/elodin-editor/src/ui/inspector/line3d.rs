use bevy_egui::egui::{self, RichText};
use impeller2_wkt::{Color, Line3d, SchematicElem};

use crate::ui::colors::{EColor, get_scheme};
use crate::ui::inspector::color_popup;
use crate::ui::plot_3d::gpu::DEFAULT_FUTURE_TRAIL_ALPHA;
use crate::ui::schematic::CurrentSchematic;

/// Field editors for a single `line_3d`. `played_timeline`/`future_timeline` are
/// the inherited timeline colors shown in the swatches when the line has no
/// per-line override. Returns `true` if any value changed.
pub fn line3d_controls(
    ui: &mut egui::Ui,
    line: &mut Line3d,
    played_timeline: Color,
    future_timeline: Color,
) -> bool {
    let scheme = get_scheme();
    let mut changed = false;

    // Sober gray piping for the box-like inputs (drag value, checkbox).
    ui.visuals_mut().widgets.inactive.bg_stroke = egui::Stroke::new(1.0, scheme.border_primary);

    ui.label(RichText::new("Line Width").color(scheme.text_secondary));
    changed |= ui
        .add(
            egui::DragValue::new(&mut line.line_width)
                .speed(0.05)
                .range(0.1..=100.0),
        )
        .changed();
    ui.separator();

    // Played color: the swatch shows the per-line override, else the inherited
    // timeline color. Editing it materializes the override.
    let mut played = line.color.unwrap_or(played_timeline);
    if color_square(ui, "Played Color", &mut played) {
        line.color = Some(played);
        changed = true;
    }

    // Future color: shows the override, else the inherited timeline future color
    // (white by default) at the default fade. The picker alpha sets opacity.
    let mut future = line.future_color.unwrap_or(Color {
        a: DEFAULT_FUTURE_TRAIL_ALPHA,
        ..future_timeline
    });
    if color_square(ui, "Future Color", &mut future) {
        line.future_color = Some(future);
        changed = true;
    }
    ui.separator();

    changed |= ui
        .checkbox(&mut line.perspective, "Perspective (screen-space width)")
        .changed();

    changed
}

/// A plain color square below its label that opens the shared color popup.
/// Returns `true` only when the popup actually edits the color, so it never
/// spuriously materializes an inherited color from a redraw.
fn color_square(ui: &mut egui::Ui, label: &str, color: &mut Color) -> bool {
    let scheme = get_scheme();
    let mut egui_color = color.into_color32();

    ui.label(RichText::new(label).color(scheme.text_secondary));
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter().rect(
            rect,
            egui::CornerRadius::same(3),
            egui_color,
            egui::Stroke::new(1.0, scheme.border_primary),
            egui::StrokeKind::Inside,
        );
    }
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    ui.separator();

    let popup_id = ui.auto_id_with(("line3d_color", label));
    if resp.clicked() {
        egui::Popup::toggle_id(ui.ctx(), popup_id);
    }

    let before = egui_color;
    color_popup(ui, &mut egui_color, popup_id, &resp);
    if egui_color != before {
        *color = Color::from_color32(egui_color);
        true
    } else {
        false
    }
}

/// Mirror the edited fields back into `CurrentSchematic` so they survive a
/// schematic save/reload (rendering reads the live component directly).
pub fn persist_to_schematic(schematic: &mut CurrentSchematic, line: &Line3d) {
    for elem in &mut schematic.elems {
        if let SchematicElem::Line3d(elem) = elem
            && elem.node_id == line.node_id
        {
            elem.line_width = line.line_width;
            elem.color = line.color;
            elem.future_color = line.future_color;
            elem.perspective = line.perspective;
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_wkt::{NodeId, Schematic};

    fn line(node_id: u64, eql: &str) -> Line3d {
        Line3d {
            eql: eql.to_string(),
            line_width: 1.0,
            color: None,
            future_color: None,
            perspective: false,
            frame: None,
            node_id: NodeId(node_id),
        }
    }

    #[test]
    fn persist_updates_only_the_matching_line() {
        let mut schematic = CurrentSchematic(Schematic {
            elems: vec![
                SchematicElem::Line3d(line(1, "a")),
                SchematicElem::Line3d(line(2, "b")),
            ],
            ..Default::default()
        });

        let mut edited = line(2, "b");
        edited.line_width = 5.0;
        edited.color = Some(Color::rgba(1.0, 0.0, 0.0, 0.5));
        edited.future_color = Some(Color::rgba(0.0, 1.0, 0.0, 0.25));
        edited.perspective = true;
        persist_to_schematic(&mut schematic, &edited);

        let SchematicElem::Line3d(first) = &schematic.elems[0] else {
            panic!("expected line3d");
        };
        assert_eq!(first.line_width, 1.0);
        assert!(first.color.is_none());

        let SchematicElem::Line3d(second) = &schematic.elems[1] else {
            panic!("expected line3d");
        };
        assert_eq!(second.line_width, 5.0);
        assert_eq!(second.color, Some(Color::rgba(1.0, 0.0, 0.0, 0.5)));
        assert_eq!(second.future_color, Some(Color::rgba(0.0, 1.0, 0.0, 0.25)));
        assert!(second.perspective);
    }
}
