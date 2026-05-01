use bevy::ecs::{
    system::{Query, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::Entity;
use bevy_egui::egui::{self, Align};

use crate::{
    sensor_camera::SensorCameraConfigs,
    ui::{
        button::EButton,
        colors::{EColor, get_scheme},
        label::ELabel,
        utils::MarginSides,
        video_stream::VideoStream,
        widgets::WidgetSystem,
    },
};

use super::{color_popup, empty_inspector};

#[derive(SystemParam)]
pub struct InspectorSensorCamera<'w, 's> {
    streams: Query<'w, 's, &'static VideoStream>,
    configs: ResMut<'w, SensorCameraConfigs>,
}

impl WidgetSystem for InspectorSensorCamera<'_, '_> {
    type Args = (Entity, String);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let scheme = get_scheme();
        let (stream_entity, title) = args;
        let InspectorSensorCamera {
            streams,
            mut configs,
        } = state.get_mut(world);

        let Ok(stream) = streams.get(stream_entity) else {
            ui.add(empty_inspector());
            return;
        };
        let Some(config) = configs
            .0
            .iter_mut()
            .find(|config| config.camera_name == stream.msg_name)
        else {
            ui.add(empty_inspector());
            return;
        };

        ui.spacing_mut().item_spacing.y = 8.0;
        let title = title.trim();
        let title = if title.is_empty() {
            config.camera_name.as_str()
        } else {
            title
        };
        ui.add(
            ELabel::new(title)
                .padding(egui::Margin::same(8).bottom(24.0))
                .bottom_stroke(egui::Stroke {
                    color: scheme.border_primary,
                    width: 1.0,
                })
                .margin(egui::Margin::same(0).bottom(16.0)),
        );

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("CAMERA").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(&config.camera_name);
                    });
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("RESOLUTION").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(format!("{}x{}", config.width, config.height));
                    });
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("FOV").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(format!("{:.1}", config.fov_degrees));
                    });
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SHOW ELLIPSOIDS").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.checkbox(&mut config.show_ellipsoids, "");
                    });
                });
            });

        ui.separator();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                let create_button_width = 88.0;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("FRUSTUM").color(scheme.text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        let frustum_created = config.create_frustum;
                        let button = if frustum_created {
                            EButton::red("DELETE")
                        } else {
                            EButton::highlight("CREATE")
                        };
                        if ui.add(button.width(create_button_width)).clicked() {
                            config.create_frustum = !frustum_created;
                        }
                    });
                });

                if config.create_frustum {
                    ui.add_space(8.0);

                    let mut frustums_color = config.frustums_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FRUSTUM COLOR").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(frustums_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("sensor_camera_frustums_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut frustums_color, color_id, &swatch);
                        });
                    });
                    config.frustums_color = impeller2_wkt::Color::from_color32(frustums_color);

                    ui.add_space(8.0);
                    let mut projection_color = config.projection_color.into_color32();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("PROJ. 2D COLOR").color(scheme.text_secondary),
                        );
                        ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                            let swatch = ui.add(
                                egui::Button::new("")
                                    .fill(projection_color)
                                    .stroke(egui::Stroke::new(1.0, scheme.border_primary))
                                    .corner_radius(egui::CornerRadius::same(10))
                                    .min_size(egui::vec2(20.0, 20.0)),
                            );
                            let color_id = ui.auto_id_with("sensor_camera_projection_color");
                            if swatch.clicked() {
                                egui::Popup::toggle_id(ui.ctx(), color_id);
                            }
                            color_popup(ui, &mut projection_color, color_id, &swatch);
                        });
                    });
                    config.projection_color = impeller2_wkt::Color::from_color32(projection_color);

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("THICKNESS").color(scheme.text_secondary));
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            let mut thickness = config.frustums_thickness;
                            if ui
                                .add(egui::DragValue::new(&mut thickness).speed(0.001))
                                .changed()
                            {
                                config.frustums_thickness = thickness.max(0.0001);
                            }
                        });
                    });
                }
            });
    }
}
