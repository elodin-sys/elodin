use bevy_egui::egui::{self, emath, FontData, FontDefinitions, FontFamily};

use super::colors;

pub fn set_theme(context: &mut egui::Context) {
    let mut style = (*context.style()).clone();

    style.spacing.item_spacing = emath::vec2(0., 0.);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, colors::BORDER_GREY);

    context.set_fonts(configure_default_fonts());

    context.set_style(style);
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
