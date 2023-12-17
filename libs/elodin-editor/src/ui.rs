use bevy::prelude::*;
use bevy_egui::{
    egui::{self, epaint::Shadow, style::WidgetVisuals, Color32, Margin, Rounding, Stroke, Ui},
    EguiContexts,
};
use elodin::{EntityQuery, Picked, SimState};
use nalgebra::Vector3;

const LIGHT_BLUE: Color32 = Color32::from_rgb(184, 204, 255);
const DARK_BLUE: Color32 = Color32::from_rgb(0x1F, 0x2C, 0x4C);
const ORANGE: Color32 = Color32::from_rgb(0xD2, 0x58, 0x00);

fn set_theme(context: &mut egui::Context) {
    let mut style = (*context.style()).clone();
    style.visuals.window_rounding = 0.0.into();
    style.visuals.window_shadow = Shadow::NONE;
    style.visuals.window_fill = Color32::TRANSPARENT;
    style.visuals.window_stroke = Stroke::new(1.0, LIGHT_BLUE);
    style.visuals.panel_fill = Color32::TRANSPARENT;
    style.visuals.override_text_color = Some(LIGHT_BLUE);
    style.visuals.selection.bg_fill = ORANGE;
    style.visuals.selection.stroke = Stroke::new(1.0, ORANGE);
    style.visuals.slider_trailing_fill = true;

    style.visuals.faint_bg_color = Color32::TRANSPARENT;
    style.visuals.extreme_bg_color = Color32::TRANSPARENT;

    set_widget_visuals(&mut style.visuals.widgets.active);
    style.visuals.widgets.active.bg_stroke = Stroke::new(1.0, ORANGE);
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, ORANGE);

    set_widget_visuals(&mut style.visuals.widgets.hovered);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ORANGE);

    set_widget_visuals(&mut style.visuals.widgets.inactive);

    set_widget_visuals(&mut style.visuals.widgets.open);
    style.visuals.widgets.open.bg_stroke = Stroke::new(1.0, ORANGE);

    style.spacing.window_margin = Margin::symmetric(10., 10.);

    // let mut fonts = FontDefinitions::default();
    // fonts.font_data.insert(
    //     "berkeley".to_owned(),
    //     FontData::from_static(include_bytes!("../assets/BerkeleyMono-Regular.ttf")),
    // );

    // fonts
    //     .families
    //     .get_mut(&FontFamily::Proportional)
    //     .unwrap()
    //     .insert(0, "berkeley".to_owned());

    // fonts
    //     .families
    //     .get_mut(&FontFamily::Monospace)
    //     .unwrap()
    //     .insert(0, "berkeley".to_owned());

    //context.set_fonts(fonts);

    context.set_style(style)
}

fn set_widget_visuals(visuals: &mut WidgetVisuals) {
    visuals.bg_stroke = Stroke::new(1.0, LIGHT_BLUE);
    visuals.fg_stroke = Stroke::new(1.0, LIGHT_BLUE);
    visuals.bg_fill = DARK_BLUE;
    visuals.expansion = 1.0;
    visuals.weak_bg_fill = Color32::TRANSPARENT;

    visuals.rounding = Rounding::ZERO;
}

// pub(crate) fn ui_system(mut contexts: EguiContexts, mut editables: ResMut<Editables>) {
//     set_theme(contexts.ctx_mut());
//     egui::Window::new("Inputs")
//         .title_bar(false)
//         .resizable(false)
//         .show(contexts.ctx_mut(), |ui| {
//             for editable in &mut editables.0 {
//                 editable.build(ui);
//             }
//         });
// }

pub(crate) fn picked_system(
    mut contexts: EguiContexts,
    picked: Query<(EntityQuery, &Picked, Entity)>,
) {
    set_theme(contexts.ctx_mut());
    egui::Window::new("picked components")
        .title_bar(false)
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            picked
                .iter()
                .filter(|(_, picked, _)| picked.0)
                .for_each(|(entity, _, e)| {
                    ui.collapsing(format!("entity {:?}", e), |ui| {
                        vec3_component(ui, "pos (m/s)", &entity.world_pos.0.pos);
                        vec3_component(ui, "vel (m/s)", &entity.world_vel.0.vel);
                        let euler_angles = vec_from_tuple(entity.world_pos.0.att.euler_angles());
                        vec3_component(ui, "euler angle (rad)", &euler_angles);
                        vec3_component(ui, "ang vel (m/s)", &entity.world_vel.0.vel);
                    });
                })
        });
}

pub(crate) fn timeline_system(
    mut contexts: EguiContexts,
    mut sim_state: ResMut<SimState>,
    window: Query<&Window>,
) {
    let window = window.single();
    let width = window.resolution.width();
    let height = window.resolution.height();
    egui::Window::new("timeline")
        .title_bar(false)
        .resizable(false)
        .fixed_size(egui::vec2(500.0, 50.0))
        .fixed_pos(egui::pos2(width / 2.0 - 250.0, height - 100.0))
        .show(contexts.ctx_mut(), |ui| {
            ui.horizontal(|ui| {
                let paused = sim_state.paused;
                ui.toggle_value(&mut sim_state.paused, if paused { "⏵" } else { "⏸" });
                let max_count = sim_state.history_count.saturating_sub(1);
                ui.spacing_mut().slider_width = 450.0;
                ui.add(egui::Slider::new(
                    &mut sim_state.history_index,
                    0..=max_count,
                ));
            })
        });
}

fn vec_from_tuple(tuple: (f64, f64, f64)) -> Vector3<f64> {
    Vector3::new(tuple.0, tuple.1, tuple.2)
}

fn vec3_component(ui: &mut Ui, label: &str, vec3: &Vector3<f64>) {
    ui.horizontal(|ui| {
        ui.label(label);
        let x = format!("{:+.5}", vec3.x);
        let y = format!("{:+.5}", vec3.y);
        let z = format!("{:+.5}", vec3.z);
        ui.add_sized(
            egui::vec2(70., 16.),
            egui::TextEdit::singleline(&mut x.as_str()),
        );
        ui.add_sized(
            egui::vec2(70., 16.),
            egui::TextEdit::singleline(&mut y.as_str()),
        );
        ui.add_sized(
            egui::vec2(70., 16.),
            egui::TextEdit::singleline(&mut z.as_str()),
        );
    });
}
