use bevy_egui::egui::{
    self, epaint::Shadow, FontData, FontDefinitions, FontFamily, Margin, Rounding, Stroke, Style,
};

use super::colors::{self, with_opacity};

pub fn set_theme(context: &mut egui::Context) {
    let mut style = (*context.style()).clone();

    style.spacing.item_spacing = egui::vec2(0., 0.);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, colors::BORDER_GREY);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, colors::CREMA);
    style.visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(1.0, with_opacity(colors::CREMA, 0.6));
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, colors::BORDER_GREY);

    style.visuals.widgets.noninteractive.fg_stroke.color = colors::CREMA;
    style.visuals.widgets.inactive.fg_stroke.color = colors::CREMA;
    style.visuals.widgets.active.fg_stroke.color = colors::CREMA;

    style.visuals.widgets.noninteractive.bg_fill = colors::STONE_950;
    style.visuals.widgets.active.bg_fill = colors::STONE_950;
    style.visuals.widgets.hovered.bg_fill = colors::STONE_950;
    style.visuals.widgets.open.bg_fill = colors::STONE_950;
    style.visuals.widgets.inactive.bg_fill = colors::STONE_950;

    style.visuals.widgets.noninteractive.weak_bg_fill = colors::STONE_950;
    style.visuals.widgets.active.weak_bg_fill = colors::STONE_950;
    style.visuals.widgets.hovered.weak_bg_fill = colors::STONE_950;
    style.visuals.widgets.open.weak_bg_fill = colors::STONE_950;
    style.visuals.widgets.inactive.weak_bg_fill = colors::STONE_950;

    style.visuals.menu_rounding = rounding_xxs();
    style.visuals.window_stroke = Stroke::new(1.0, colors::ONYX_8);
    style.visuals.window_shadow = Shadow {
        extrusion: 4.0,
        color: colors::with_opacity(colors::BLACK, 0.25),
    };
    style.visuals.window_fill = colors::STONE_950;

    style.spacing.menu_margin = Margin::same(8.0);
    style.spacing.window_margin = Margin::same(8.0);

    context.set_fonts(configure_default_fonts());

    context.set_style(style);
}

fn rounding_xxs() -> Rounding {
    Rounding::same(2.0)
}

fn configure_default_fonts() -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "ibm_plex_mono_medium".to_owned(),
        FontData::from_static(include_bytes!(
            "../assets/fonts/IBMPlexMono-Medium_ss04.ttf"
        )),
    );

    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .insert(0, "ibm_plex_mono_medium".to_owned());

    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .push("ibm_plex_mono_medium".to_owned());

    fonts
}

pub fn configure_combo_box(style: &mut Style) {
    style.spacing.item_spacing = [16.0, 16.0].into();
    style.spacing.button_padding = [16.0, 16.0].into();
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, colors::CREMA);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, colors::CREMA);
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, colors::CREMA);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, colors::ONYX_8);
    style.visuals.widgets.hovered.bg_fill = colors::CREMA;
    style.visuals.widgets.hovered.expansion = 0.0;
}

pub fn configure_combo_item(style: &mut Style) {
    style.spacing.interact_size.y = 37.0;
    style.spacing.item_spacing = [8.0, 8.0].into();
    style.spacing.button_padding = [8.0, 8.0].into();
    style.visuals.widgets.inactive.fg_stroke.color = with_opacity(colors::CREMA, 0.6);
    style.visuals.widgets.active.rounding = Rounding::ZERO;
    style.visuals.widgets.active.bg_stroke = Stroke::NONE;
    style.visuals.widgets.active.bg_fill = colors::CREMA;
    style.visuals.widgets.active.fg_stroke.color = colors::BLACK;
    style.visuals.widgets.active.weak_bg_fill = colors::CREMA;
    style.visuals.widgets.active.expansion = 0.0;
    style.visuals.widgets.inactive.rounding = Rounding::ZERO;
    style.visuals.widgets.inactive.expansion = 0.0;
    style.visuals.widgets.hovered.rounding = Rounding::ZERO;
    style.visuals.widgets.hovered.weak_bg_fill = colors::ONYX;
    style.visuals.widgets.hovered.bg_fill = colors::CREMA;
    style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
    style.visuals.widgets.hovered.expansion = 0.0;
    style.visuals.selection.bg_fill = colors::CREMA;
    style.visuals.selection.stroke.color = colors::BLACK;
}
