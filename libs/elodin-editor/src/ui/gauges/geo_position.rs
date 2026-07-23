//! Geo-position gauge: three labelled coordinate values converted from a
//! `source` frame into a selectable `display` coordinate system (NED / ENU /
//! ECEF / LLA). Values only — attitude lives in the orientation gauge.

use bevy::{ecs::system::SystemParam, math::DVec3, prelude::*};
use bevy_egui::egui;
use bevy_geo_frames::{GeoContext, GeoFrame, ecef_to_lla_deg};
use impeller2_bevy::{EntityMap, TelemetryCache};
use impeller2_wkt::{ComponentValue, CurrentTimestamp, DisplayFrame};

use super::{EqlBinding, GaugePane};
use crate::ui::widgets::{SystemStateExt, WidgetSystem};

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
        let title = super::gauge_title(&binding.eql, &pane.name);
        // Keep inherit (`source = None`) live against `Coordinate` changes.
        let source = data.effective_source(coordinate.0);
        let combo_id = egui::Id::new(("geo_position_gauge_display", pane.entity));

        // Value cards read like the component monitor; the in-panel display
        // dropdown mirrors the orientation gauge so the frame can be switched
        // without opening the inspector.
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(super::GAUGE_PANEL_MARGIN))
            .show(ui, |ui| {
                super::gauge_header(ui, &title);

                super::style_gauge_combo(ui);
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
                // Reuse the component monitor's fixed-size wrapping cards so the
                // number frames read identically across both panels.
                crate::ui::monitor::render_value_cards(ui, &cards);
            });
    }
}

/// Extract a position (metres) from a component value for the position gauge.
///
/// Accepts only (in `F32` or `F64`):
/// - a bare 3-vector (exactly three elements), or
/// - a SpatialTransform / [`WorldPos`](impeller2_wkt::WorldPos) (≥7 elements:
///   quat `[x, y, z, w]` + position `[x, y, z]`).
///
/// Rejects other lengths (e.g. 4-element fin deflections) so the gauge does not
/// treat arbitrary trailing floats as coordinates and invent NED/LLA values.
fn component_value_to_position(value: &ComponentValue) -> Option<DVec3> {
    let data = super::component_buf_f64(value)?;
    // world_pos-style pose: position is elements 4..7 (after the head quaternion).
    if data.len() >= 7 {
        return Some(DVec3::new(data[4], data[5], data[6]));
    }
    (data.len() == 3).then(|| DVec3::new(data[0], data[1], data[2]))
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

    fn f32_value(values: &[f32]) -> ComponentValue {
        ComponentValue::F32(
            Array::<f32, Dyn>::from_shape_vec(smallvec::smallvec![values.len()], values.to_vec())
                .expect("f32 buffer"),
        )
    }

    #[test]
    fn position_from_f32_vector_and_pose() {
        // Bare F32 3-vector.
        assert_eq!(
            component_value_to_position(&f32_value(&[1.0, 2.0, 3.0])),
            Some(DVec3::new(1.0, 2.0, 3.0))
        );
        // F32 world_pos-style pose: position is elements 4..7.
        assert_eq!(
            component_value_to_position(&f32_value(&[0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0])),
            Some(DVec3::new(10.0, 20.0, 30.0))
        );
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
