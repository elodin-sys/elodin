use bevy::ecs::system::SystemParam;
use bevy::prelude::{Entity, Query, Res};
use bevy_egui::egui;
use bevy_geo_frames::GeoFrame;
use impeller2_wkt::DisplayFrame;

use crate::{
    EqlContext,
    ui::{
        colors::get_scheme,
        inspector::{eql_autocomplete, inspector_text_field},
        spatial_gauge::SpatialGaugeData,
        theme,
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorSpatialGauge<'w, 's> {
    monitors: Query<'w, 's, &'static mut SpatialGaugeData>,
    eql_context: Res<'w, EqlContext>,
}

impl WidgetSystem for InspectorSpatialGauge<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorSpatialGauge {
            mut monitors,
            eql_context,
        } = state.get_mut(world);
        let Ok(mut monitor) = monitors.get_mut(entity) else {
            return;
        };

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("POSITION (EQL)").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                theme::configure_input_with_border(ui.style_mut());
                let res = ui.add(inspector_text_field(
                    &mut monitor.eql,
                    "Position EQL (i.e a.world_pos)",
                ));
                eql_autocomplete(ui, &eql_context.0, &res, &mut monitor.eql);

                ui.add_space(12.0);
                ui.label(egui::RichText::new("SOURCE FRAME").color(get_scheme().text_secondary));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    // None = inherit schematic `coordinate` (omit `source` in KDL).
                    ui.selectable_value(&mut monitor.source, None, "Inherit");
                    for frame in [GeoFrame::ECEF, GeoFrame::NED, GeoFrame::ENU] {
                        ui.selectable_value(
                            &mut monitor.source,
                            Some(frame),
                            geo_frame_label(frame),
                        );
                    }
                });

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
                        ui.selectable_value(&mut monitor.display, display, display.as_str());
                    }
                });
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
