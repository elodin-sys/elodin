//! Geo-position gauge: three labelled coordinate values converted from a
//! `source` frame into a selectable `display` coordinate system (NED / ENU /
//! ECEF / LLA). Values only — attitude lives in the orientation gauge.

use bevy::{ecs::system::SystemParam, math::DVec3, prelude::*};
use bevy_egui::egui;
use bevy_geo_frames::{GeoContext, GeoFrame, ecef_to_lla_deg};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp, DisplayFrame};
use nox::ArrayBuf;

use crate::WorldPosExt;
use crate::object_3d::ComponentArrayExt;

use super::{EqlBinding, GaugePane};
use crate::ui::{
    colors::get_scheme,
    theme,
    widgets::{SystemStateExt, WidgetSystem},
};

/// Backing data for a geo-position gauge pane; the EQL lives in the sibling
/// [`EqlBinding`] component.
#[derive(Component)]
pub struct GeoPositionGaugeData {
    /// Frame the EQL position is expressed in.
    ///
    /// `None` means inherit the schematic global [`crate::Coordinate`] (same
    /// as omitting `source` in KDL). Resolved at display/export via
    /// [`Self::effective_source`].
    pub source: Option<GeoFrame>,
    pub display: DisplayFrame,
}

impl GeoPositionGaugeData {
    pub fn new(source: Option<GeoFrame>, display: DisplayFrame) -> Self {
        Self { source, display }
    }

    /// Concrete source frame: explicit override, else schematic `coordinate`,
    /// else ENU (same fallback as viewport / view-cube when both are unset).
    pub fn effective_source(&self, coordinate: Option<GeoFrame>) -> GeoFrame {
        self.source.or(coordinate).unwrap_or(GeoFrame::ENU)
    }
}

#[derive(SystemParam)]
pub struct GeoPositionGaugeWidget<'w, 's> {
    gauges: Query<'w, 's, (&'static mut GeoPositionGaugeData, &'static EqlBinding)>,
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
    telemetry_cache: Res<'w, TelemetryCache>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    geo_context: Res<'w, GeoContext>,
    coordinate: Res<'w, crate::Coordinate>,
}

impl WidgetSystem for GeoPositionGaugeWidget<'_, '_> {
    type Args = GaugePane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        pane: Self::Args,
    ) -> Self::Output {
        let GeoPositionGaugeWidget {
            mut gauges,
            entity_map,
            values,
            telemetry_cache,
            current_timestamp,
            geo_context,
            coordinate,
        } = state.params_mut(world);
        let Ok((mut data, binding)) = gauges.get_mut(pane.entity) else {
            return;
        };

        let ts = current_timestamp.0;
        let value = binding.resolve(&entity_map, &values, &telemetry_cache, ts);
        // Keep inherit (`source = None`) live against `Coordinate` changes.
        let source = data.effective_source(coordinate.0);
        let combo_id = egui::Id::new(("geo_position_gauge_display", pane.entity));

        // Value cards read like the component monitor; the in-panel display
        // dropdown mirrors the orientation gauge so the frame can be switched
        // without opening the inspector.
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                {
                    let style = ui.style_mut();
                    theme::configure_input_with_border(style);
                    style
                        .text_styles
                        .iter_mut()
                        .for_each(|(_, font)| font.size = 10.0);
                }
                egui::ComboBox::from_id_salt(combo_id)
                    .selected_text(data.display.as_str())
                    .width(86.0)
                    .show_ui(ui, |ui| {
                        for frame in [
                            DisplayFrame::NED,
                            DisplayFrame::ENU,
                            DisplayFrame::ECEF,
                            DisplayFrame::LLA,
                        ] {
                            ui.selectable_value(&mut data.display, frame, frame.as_str());
                        }
                    });

                // Read `display` after the ComboBox so a change applies this frame.
                let display = data.display;
                let labels = display_labels(display);
                let out = value
                    .as_ref()
                    .and_then(component_value_to_position)
                    .map(|pos_src| convert(pos_src, source, display, &geo_context));
                let cards: Vec<(String, String)> = labels
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let value = out
                            .map(|v| fmt_val(v[i]))
                            .unwrap_or_else(|| "—".to_string());
                        ((*label).to_string(), value)
                    })
                    .collect();

                ui.add_space(4.0);
                render_position_cards(ui, &cards);
            });
    }
}

/// Lay the coordinate values out as cards: a single horizontal row that fills
/// the width when the pane is expanded, stacking vertically when it is too
/// narrow to read a row.
fn render_position_cards(ui: &mut egui::Ui, cards: &[(String, String)]) {
    if cards.is_empty() {
        return;
    }
    // Below ~90px per card a horizontal row gets too cramped to read; stack.
    let cols_n = if ui.available_width() >= 90.0 * cards.len() as f32 {
        cards.len()
    } else {
        1
    };
    ui.columns(cols_n, |cols| {
        for (i, (label, value)) in cards.iter().enumerate() {
            position_card(&mut cols[i % cols_n], label, value);
        }
    });
}

/// A single labelled value card (component-monitor styling, gauge sizing).
fn position_card(ui: &mut egui::Ui, label: &str, value: &str) {
    let scheme = get_scheme();
    egui::Frame::NONE
        .stroke(egui::Stroke::new(1.0, scheme.border_primary))
        .inner_margin(egui::Margin::symmetric(8, 6))
        .outer_margin(egui::Margin::same(2))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(label)
                    .size(11.0)
                    .monospace()
                    .color(scheme.text_secondary),
            );
            ui.label(
                egui::RichText::new(value)
                    .size(15.0)
                    .monospace()
                    .color(scheme.text_primary),
            );
        });
}

/// Extract a position (metres) from a component value for the position gauge.
///
/// Accepts only:
/// - a bare 3-vector (`F32`/`F64` with exactly three elements), or
/// - a SpatialTransform / [`WorldPos`](impeller2_wkt::WorldPos) (`F64`, ≥7
///   elements: quat + position).
///
/// Rejects other lengths (e.g. 4-element fin deflections) so the gauge does not
/// treat arbitrary trailing floats as coordinates and invent NED/LLA values.
fn component_value_to_position(value: &ComponentValue) -> Option<DVec3> {
    if let Some(wp) = value.as_world_pos() {
        return Some(wp.pos());
    }
    match value {
        ComponentValue::F32(array) => {
            let data = array.buf.as_buf();
            (data.len() == 3).then(|| DVec3::new(data[0] as f64, data[1] as f64, data[2] as f64))
        }
        ComponentValue::F64(array) => {
            let data = array.buf.as_buf();
            (data.len() == 3).then(|| DVec3::new(data[0], data[1], data[2]))
        }
        _ => None,
    }
}

/// Convert a position from `source` into the `display` coordinate system.
/// For LLA the result is `(lat_deg, lon_deg, alt_m)`.
fn convert(pos_src: DVec3, source: GeoFrame, display: DisplayFrame, ctx: &GeoContext) -> DVec3 {
    match display.geo_frame() {
        Some(frame) => frame._M_(&source, ctx).transform_point3(pos_src),
        None => {
            let ecef = GeoFrame::ECEF._M_(&source, ctx).transform_point3(pos_src);
            let (lat, lon, alt) = ecef_to_lla_deg(ecef, &ctx.origin.ellipsoid);
            DVec3::new(lat, lon, alt)
        }
    }
}

/// Axis labels for the three value cards, matching the display frame.
fn display_labels(display: DisplayFrame) -> [&'static str; 3] {
    match display {
        DisplayFrame::NED => ["N", "E", "D"],
        DisplayFrame::ENU => ["E", "N", "U"],
        DisplayFrame::ECEF => ["X", "Y", "Z"],
        DisplayFrame::LLA => ["Lat", "Lon", "Alt"],
    }
}

/// Format a coordinate compactly without ever dropping integer digits.
///
/// Truncating a fixed-precision string (the previous approach) silently lost
/// significant digits for larger magnitudes; instead pick the decimal count
/// from the magnitude so the string stays short while the value stays correct.
fn fmt_val(v: f64) -> String {
    if !v.is_finite() {
        return "—".to_string();
    }
    let mag = v.abs();
    let decimals = if mag >= 100_000.0 {
        1 // large metres (e.g. ECEF): sub-metre precision is noise here
    } else if mag >= 1_000.0 {
        3
    } else if mag >= 1.0 {
        5 // degrees (lat/lon) keep ~1 m; local metres keep mm
    } else {
        6
    };
    format!("{v:.decimals$}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_geo_frames::GeoOrigin;
    use nox::{Array, Dyn};

    fn f64_value(values: &[f64]) -> ComponentValue {
        ComponentValue::F64(
            Array::<f64, Dyn>::from_shape_vec(smallvec::smallvec![values.len()], values.to_vec())
                .expect("f64 buffer"),
        )
    }

    #[test]
    fn position_from_bare_xyz_vector() {
        let v = f64_value(&[1.0, 2.0, 3.0]);
        assert_eq!(
            component_value_to_position(&v),
            Some(DVec3::new(1.0, 2.0, 3.0))
        );
    }

    #[test]
    fn position_from_spatial_transform() {
        // quat (x,y,z,w) + pos — same layout as WorldPos / SpatialTransform.
        let v = f64_value(&[0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0]);
        assert_eq!(
            component_value_to_position(&v),
            Some(DVec3::new(10.0, 20.0, 30.0))
        );
    }

    #[test]
    fn non_pose_multi_element_is_not_a_position() {
        // e.g. fin deflections: last three must not become fake metres.
        let fins = f64_value(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(component_value_to_position(&fins), None);
        assert_eq!(component_value_to_position(&f64_value(&[1.0, 2.0])), None);
        assert_eq!(
            component_value_to_position(&f64_value(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            None
        );
    }

    #[test]
    fn convert_ecef_to_ned_and_lla_at_origin() {
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(34.72, -86.64, 180.0));

        // ECEF position of the NED origin itself.
        let ecef = GeoFrame::ECEF
            ._M_(&GeoFrame::NED, &ctx)
            .transform_point3(DVec3::ZERO);

        // ECEF -> NED at the origin collapses to ~0.
        let ned = convert(ecef, GeoFrame::ECEF, DisplayFrame::NED, &ctx);
        assert!(ned.length() < 1e-6, "expected ~0 NED, got {ned:?}");

        // ECEF -> LLA recovers the origin's geodetic coordinates (deg/deg/m).
        let lla = convert(ecef, GeoFrame::ECEF, DisplayFrame::LLA, &ctx);
        assert!((lla.x - 34.72).abs() < 1e-6, "lat {}", lla.x);
        assert!((lla.y + 86.64).abs() < 1e-6, "lon {}", lla.y);
        assert!((lla.z - 180.0).abs() < 1e-3, "alt {}", lla.z);
    }

    #[test]
    fn convert_ned_offset_round_trips_through_ecef() {
        let ctx = GeoContext::from(GeoOrigin::new_from_degrees(0.0, 0.0, 0.0));
        let ned_in = DVec3::new(100.0, -50.0, 25.0);
        // NED -> ECEF, then display ECEF back as NED should recover the input.
        let ecef = GeoFrame::ECEF
            ._M_(&GeoFrame::NED, &ctx)
            .transform_point3(ned_in);
        let ned_back = convert(ecef, GeoFrame::ECEF, DisplayFrame::NED, &ctx);
        assert!(
            (ned_back - ned_in).length() < 1e-6,
            "round-trip mismatch: {ned_back:?} vs {ned_in:?}"
        );
    }

    #[test]
    fn fmt_val_never_drops_integer_digits() {
        // Negative longitude used to lose trailing digits via string truncation;
        // now it keeps every integer digit and the sign.
        assert_eq!(fmt_val(-80.6043), "-80.60430");
        assert_eq!(fmt_val(28.6084), "28.60840");
        // Large ECEF magnitudes keep all integer digits (no mid-number loss).
        assert_eq!(fmt_val(6378137.0), "6378137.0");
        assert_eq!(fmt_val(-6378137.0), "-6378137.0");
        // Small values keep fine precision.
        assert_eq!(fmt_val(0.123456789), "0.123457");
        assert_eq!(fmt_val(f64::NAN), "—");
    }

    #[test]
    fn display_labels_match_frame_axes() {
        assert_eq!(display_labels(DisplayFrame::NED), ["N", "E", "D"]);
        assert_eq!(display_labels(DisplayFrame::ENU), ["E", "N", "U"]);
        assert_eq!(display_labels(DisplayFrame::ECEF), ["X", "Y", "Z"]);
        assert_eq!(display_labels(DisplayFrame::LLA), ["Lat", "Lon", "Alt"]);
    }

    #[test]
    fn effective_source_inherits_coordinate_then_enu() {
        let inherit = GeoPositionGaugeData::new(None, DisplayFrame::NED);
        assert_eq!(inherit.effective_source(Some(GeoFrame::NED)), GeoFrame::NED);
        assert_eq!(inherit.effective_source(None), GeoFrame::ENU);

        let explicit = GeoPositionGaugeData::new(Some(GeoFrame::ECEF), DisplayFrame::NED);
        assert_eq!(
            explicit.effective_source(Some(GeoFrame::ENU)),
            GeoFrame::ECEF
        );
    }
}
