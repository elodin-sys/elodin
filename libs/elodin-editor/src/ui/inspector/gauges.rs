use bevy::ecs::system::SystemParam;
use bevy::math::DQuat;
use bevy::prelude::{Entity, Query, Res};
use bevy_egui::egui;
use bevy_geo_frames::GeoFrame;
use impeller2_wkt::DisplayFrame;

use crate::{
    EqlContext,
    ui::{
        colors::get_scheme,
        gauges::{EqlBinding, GeoPositionGaugeData, OrientationGaugeData},
        inspector::{eql_autocomplete, inspector_text_field},
        theme,
        widgets::WidgetSystem,
    },
};

/// Shared EQL + source-frame section of both gauge inspectors.
fn eql_and_source_ui(
    ui: &mut egui::Ui,
    eql_context: &EqlContext,
    binding: &mut EqlBinding,
    source: &mut Option<GeoFrame>,
    eql_hint: &str,
) {
    ui.label(egui::RichText::new("EQL").color(get_scheme().text_secondary));
    ui.add_space(8.0);
    theme::configure_input_with_border(ui.style_mut());
    let res = ui.add(inspector_text_field(&mut binding.eql, eql_hint));
    eql_autocomplete(ui, &eql_context.0, &res, &mut binding.eql);

    ui.add_space(12.0);
    ui.label(egui::RichText::new("SOURCE FRAME").color(get_scheme().text_secondary));
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        // None = inherit schematic `coordinate` (omit `source` in KDL).
        ui.selectable_value(source, None, "Inherit");
        for frame in [GeoFrame::ECEF, GeoFrame::NED, GeoFrame::ENU] {
            ui.selectable_value(source, Some(frame), geo_frame_label(frame));
        }
    });
}

#[derive(SystemParam)]
pub struct InspectorGeoPositionGauge<'w, 's> {
    gauges: Query<'w, 's, (&'static mut GeoPositionGaugeData, &'static mut EqlBinding)>,
    eql_context: Res<'w, EqlContext>,
}

impl WidgetSystem for InspectorGeoPositionGauge<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorGeoPositionGauge {
            mut gauges,
            eql_context,
        } = state.get_mut(world);
        let Ok((mut gauge, mut binding)) = gauges.get_mut(entity) else {
            return;
        };

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                eql_and_source_ui(
                    ui,
                    &eql_context,
                    &mut binding,
                    &mut gauge.source,
                    "Position EQL (i.e a.world_pos)",
                );

                ui.add_space(12.0);
                ui.label(egui::RichText::new("DISPLAY FRAME").color(get_scheme().text_secondary));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for display in [
                        DisplayFrame::ECEF,
                        DisplayFrame::NED,
                        DisplayFrame::ENU,
                        DisplayFrame::LLA,
                    ] {
                        ui.selectable_value(&mut gauge.display, display, display.as_str());
                    }
                });
            });
    }
}

#[derive(SystemParam)]
pub struct InspectorOrientationGauge<'w, 's> {
    gauges: Query<'w, 's, (&'static mut OrientationGaugeData, &'static mut EqlBinding)>,
    eql_context: Res<'w, EqlContext>,
}

impl WidgetSystem for InspectorOrientationGauge<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorOrientationGauge {
            mut gauges,
            eql_context,
        } = state.get_mut(world);
        let Ok((mut gauge, mut binding)) = gauges.get_mut(entity) else {
            return;
        };

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                eql_and_source_ui(
                    ui,
                    &eql_context,
                    &mut binding,
                    &mut gauge.source,
                    "Attitude EQL (i.e a.world_pos or a.q)",
                );

                ui.add_space(12.0);
                ui.label(egui::RichText::new("DISPLAY FRAME").color(get_scheme().text_secondary));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    for display in [GeoFrame::ECEF, GeoFrame::NED, GeoFrame::ENU] {
                        ui.selectable_value(
                            &mut gauge.display,
                            Some(display),
                            geo_frame_label(display),
                        );
                    }
                });

                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("REFERENCE ATTITUDE (X Y Z W)")
                        .color(get_scheme().text_secondary),
                );
                ui.add_space(4.0);
                // Body→source quaternion the gimbal treats as neutral;
                // normalized on edit so hand-typed values stay a rotation.
                let q = &mut gauge.reference;
                let mut parts = [q.x, q.y, q.z, q.w];
                let mut changed = false;
                ui.horizontal(|ui| {
                    for part in &mut parts {
                        changed |= ui
                            .add(egui::DragValue::new(part).speed(0.01).range(-1.0..=1.0))
                            .changed();
                    }
                    if ui.button("Reset").clicked() {
                        parts = [0.0, 0.0, 0.0, 1.0];
                        changed = true;
                    }
                });
                if changed {
                    let next = DQuat::from_xyzw(parts[0], parts[1], parts[2], parts[3]);
                    if next.length() > 1e-9 {
                        *q = next.normalize();
                    }
                }
            });
    }
}

fn geo_frame_label(frame: GeoFrame) -> &'static str {
    match frame {
        GeoFrame::ECEF => "ECEF",
        GeoFrame::NED => "NED",
        GeoFrame::ENU => "ENU",
    }
}
