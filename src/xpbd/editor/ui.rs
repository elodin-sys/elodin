use super::Input;
use crate::xpbd::{
    builder::{Env, FromEnv},
    runner::SimRunnerEnv,
};
use bevy::prelude::*;
use bevy_egui::{
    egui::{
        self, epaint::Shadow, style::WidgetVisuals, Color32, FontData, FontDefinitions, FontFamily,
        Margin, Rounding, Stroke, Ui,
    },
    EguiContexts,
};
use std::ops::DerefMut;

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

    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "berkeley".to_owned(),
        FontData::from_static(include_bytes!("../../../assets/BerkeleyMono-Regular.ttf")),
    );

    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "berkeley".to_owned());

    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "berkeley".to_owned());

    context.set_fonts(fonts);

    context.set_style(style)
}

fn set_widget_visuals(visuals: &mut WidgetVisuals) {
    visuals.bg_stroke = Stroke::new(1.0, LIGHT_BLUE);
    visuals.fg_stroke = Stroke::new(1.0, LIGHT_BLUE);
    visuals.bg_fill = DARK_BLUE;
    visuals.expansion = 1.0;
    visuals.weak_bg_fill = Color32::TRANSPARENT;

    visuals.rounding = Rounding::none();
}

pub(crate) fn ui_system(mut contexts: EguiContexts, mut editables: ResMut<Editables>) {
    set_theme(contexts.ctx_mut());
    egui::Window::new("Inputs")
        .title_bar(false)
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            for editable in &mut editables.0 {
                editable.build(ui);
            }
        });
}

impl Editable for Input {
    fn build(&mut self, ui: &mut Ui) {
        let mut num = self.0.load();
        ui.add(egui::Slider::new(num.deref_mut(), -1.25..=1.25).text("input"));
    }
}

pub trait Editable: Send + Sync {
    fn build(&mut self, ui: &mut Ui);
}

impl<F: Editable + Clone + Resource + Default> FromEnv<SimRunnerEnv> for F {
    type Item<'a> = F;

    fn from_env(env: <SimRunnerEnv as Env>::Param<'_>) -> Self::Item<'_> {
        env.app
            .world
            .get_resource::<F>()
            .expect("missing resource")
            .clone()
    }

    fn init(env: &mut SimRunnerEnv) {
        let f = F::default();
        let mut editables = env
            .app
            .world
            .get_resource_or_insert_with(|| Editables(vec![]));
        editables.0.push(Box::new(f.clone()));
        env.app.world.insert_resource(f);
    }
}

#[derive(Resource, Default)]
pub struct Editables(Vec<Box<dyn Editable>>);
