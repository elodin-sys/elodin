//! Paints a camera's up marker over the pane showing that camera's image, so
//! the image orientation reads the same there as on the frustum drawn in 3D.

use egui::Color32;
use impeller2_wkt::FrustumUpMarker;

/// Marker proportions, as fractions of half the image height, so the painted
/// marker keeps the weight it has on the 3D frustum.
const STROKE: f32 = 0.02;
const TRIANGLE_HALF_BASE: f32 = 0.13;
const TRIANGLE_HEIGHT: f32 = 0.20;
/// Image-origin ball radius, relative to the stroke width.
const BALL_SCALE: f32 = 1.5;

const MIN_STROKE: f32 = 2.0;
const MAX_STROKE: f32 = 7.0;

/// Cap on the triangle's reference, in points. The bar of `Highlight` spans the
/// top edge and so belongs to the pane's scale, but a triangle that keeps
/// growing with a large 3D viewport stops reading as an annotation and starts
/// looking like a shape in the scene.
const TRIANGLE_MAX_REFERENCE: f32 = 160.0;

fn stroke_width(reference: f32) -> f32 {
    (reference * STROKE).clamp(MIN_STROKE, MAX_STROKE)
}

/// Paints `marker` along the top of `image`.
///
/// `Highlight` runs a bar across the top edge with a ball on the left, marking
/// the corner that holds pixel (0, 0) — the same corner the frustum's ball sits
/// on. `Triangle` points at the top edge from just inside the image, since a
/// triangle standing on that edge would fall outside the pane.
pub fn paint_up_marker(
    painter: &egui::Painter,
    image: egui::Rect,
    marker: FrustumUpMarker,
    color: Color32,
) {
    let reference = image.height() * 0.5;

    match marker {
        FrustumUpMarker::None => {}
        FrustumUpMarker::Highlight => {
            let stroke_width = stroke_width(reference);
            // Inset by the ball radius so neither the bar nor the ball is
            // half-clipped by the image border.
            let ball_radius = stroke_width * BALL_SCALE;
            let y = image.top() + ball_radius;
            let origin = egui::pos2(image.left() + ball_radius, y);
            painter.line_segment(
                [origin, egui::pos2(image.right() - ball_radius, y)],
                egui::Stroke::new(stroke_width, color),
            );
            painter.circle_filled(origin, ball_radius, color);
        }
        FrustumUpMarker::Triangle => {
            let reference = reference.min(TRIANGLE_MAX_REFERENCE);
            let stroke_width = stroke_width(reference);
            let stroke = egui::Stroke::new(stroke_width, color);

            let apex_y = image.top() + stroke_width * 0.5;
            let base_y = apex_y + reference * TRIANGLE_HEIGHT;
            let half_base = reference * TRIANGLE_HALF_BASE;
            let center_x = image.center().x;

            let apex = egui::pos2(center_x, apex_y);
            painter.line_segment([egui::pos2(center_x - half_base, base_y), apex], stroke);
            painter.line_segment([apex, egui::pos2(center_x + half_base, base_y)], stroke);
            // Strokes are flat-capped, so round the joint the two sides share.
            painter.circle_filled(apex, stroke_width * 0.5, color);
        }
    }
}
