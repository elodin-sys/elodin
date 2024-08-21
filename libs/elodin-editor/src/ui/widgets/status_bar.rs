use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::{
        system::{Res, SystemParam, SystemState},
        world::World,
    },
};
use impeller::bevy::{MaxTick, Tick, TimeStep};

use crate::ui::colors;

use super::RootWidgetSystem;

#[derive(SystemParam)]
pub struct StatusBar<'w> {
    tick_time: Res<'w, TimeStep>,
    current_tick: Res<'w, Tick>,
    max_tick: Res<'w, MaxTick>,
    diagnostics: Res<'w, DiagnosticsStore>,
}

impl RootWidgetSystem for StatusBar<'_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let tick_time = state_mut.tick_time;
        let current_tick = state_mut.current_tick;
        let max_tick = state_mut.max_tick;
        let diagnostics = state_mut.diagnostics;

        egui::TopBottomPanel::bottom("status_bar")
            .frame(egui::Frame {
                fill: colors::PRIMARY_ONYX,
                inner_margin: egui::Margin::symmetric(16.0, 4.0),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let style = ui.style_mut();
                    style.spacing.item_spacing = [20.0, 8.0].into();

                    // Status

                    ui.add(editor_status_label(
                        tick_time.0.as_secs_f64(),
                        current_tick.0,
                        max_tick.0,
                    ));

                    // Editor FPS

                    let render_fps_str = diagnostics
                        .get(&FrameTimeDiagnosticsPlugin::FPS)
                        .and_then(|diagnostic_fps| diagnostic_fps.smoothed())
                        .map_or(" N/A".to_string(), |value| format!("{value:>6.1}"));

                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("FPS {render_fps_str}"))
                            .text_style(egui::TextStyle::Small)
                            .color(colors::PRIMARY_CREAME_6),
                    ));

                    // Simulator TPS

                    let sim_fps = if tick_time.0.as_secs_f64() > 0.0 {
                        format!("{:>6.1}", 1.0 / tick_time.0.as_secs_f64())
                    } else {
                        String::from("N/A")
                    };

                    ui.add(egui::Label::new(
                        egui::RichText::new(format!("TPS {sim_fps}"))
                            .text_style(egui::TextStyle::Small)
                            .color(colors::PRIMARY_CREAME_6),
                    ));
                });
            });
    }
}

fn editor_status_label_ui(
    ui: &mut egui::Ui,
    tick_time: f64,
    current_tick: u64,
    max_tick: u64,
) -> egui::Response {
    let style = ui.style_mut();
    let font_id = egui::TextStyle::Small.resolve(style);

    let text_color = colors::PRIMARY_CREAME_6;

    let (status_label, status_color) = if tick_time > 0.0 {
        if current_tick == max_tick {
            (String::from("SIMULATING"), colors::HYPERBLUE_DEFAULT)
        } else {
            (String::from("CONNECTED"), colors::MINT_DEFAULT)
        }
    } else {
        (String::from("DISCONNECTED"), colors::PEACH_DEFAULT)
    };

    // Set widget size and allocate space

    let galley = ui
        .painter()
        .layout_no_wrap(status_label.clone(), font_id.clone(), text_color);
    let circle_diameter = galley.size().y / 2.0;
    let spacing = circle_diameter * 1.5;

    let desired_size = egui::vec2(circle_diameter + spacing + galley.size().x, galley.size().y);

    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    // Paint the UI
    if ui.is_rect_visible(rect) {
        // Background
        let circle_radius = circle_diameter / 2.0;
        ui.painter().circle_filled(
            egui::pos2(rect.left_center().x + circle_radius, rect.left_center().y),
            circle_radius,
            status_color,
        );

        // Label
        ui.painter().text(
            egui::pos2(
                rect.left_center().x + circle_diameter + spacing,
                rect.left_center().y,
            ),
            egui::Align2::LEFT_CENTER,
            status_label,
            font_id,
            text_color,
        );
    }

    response
}

pub fn editor_status_label(tick_time: f64, current_tick: u64, max_tick: u64) -> impl egui::Widget {
    move |ui: &mut egui::Ui| editor_status_label_ui(ui, tick_time, current_tick, max_tick)
}
