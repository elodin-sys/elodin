//! Paints a camera's up marker over the pane showing that camera's image, so
//! the image orientation reads the same there as on the frustum drawn in 3D.

use egui::Color32;
use impeller2_wkt::FrustumUpMarker;

/// Bar thickness, as a fraction of half the image height, so the painted marker
/// keeps the weight it has on the 3D frustum.
const STROKE: f32 = 0.02;
/// Image-origin ball radius, relative to the stroke width.
const BALL_SCALE: f32 = 1.5;

const MIN_STROKE: f32 = 2.0;
const MAX_STROKE: f32 = 7.0;

/// Paints `marker` along the top of `image`.
///
/// The bar takes `frustum_color`, which is what tells the panes of several
/// cameras apart; the ball takes `origin_color` and marks the corner holding
/// pixel (0, 0) — the same corner the frustum's ball sits on.
pub fn paint_up_marker(
    painter: &egui::Painter,
    image: egui::Rect,
    marker: FrustumUpMarker,
    frustum_color: Color32,
    origin_color: Color32,
) {
    if marker != FrustumUpMarker::Highlight {
        return;
    }

    let stroke_width = (image.height() * 0.5 * STROKE).clamp(MIN_STROKE, MAX_STROKE);
    // Inset by the ball radius so neither the bar nor the ball is half-clipped
    // by the image border.
    let ball_radius = stroke_width * BALL_SCALE;
    let y = image.top() + ball_radius;
    let origin = egui::pos2(image.left() + ball_radius, y);

    painter.line_segment(
        [origin, egui::pos2(image.right() - ball_radius, y)],
        egui::Stroke::new(stroke_width, frustum_color),
    );
    painter.circle_filled(origin, ball_radius, origin_color);
}
