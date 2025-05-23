use bevy_egui::egui::{self, CornerRadius, FontData, Margin, Stroke, Style, epaint::Shadow};
use egui::{
    Color32,
    epaint::text::{FontInsert, InsertFontFamily},
};

use crate::ui::colors::{self, get_scheme, with_opacity};

use super::colors::ColorExt;

pub fn set_theme(context: &mut egui::Context) {
    let mut style = (*context.style()).clone();
    let scheme = colors::get_scheme();

    style.spacing.item_spacing = egui::vec2(0., 0.);

    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, scheme.border_primary);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, scheme.text_primary);
    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, scheme.text_secondary);
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, scheme.border_primary);

    style.visuals.extreme_bg_color = scheme.bg_secondary;

    style.visuals.widgets.noninteractive.fg_stroke.color = scheme.text_primary;
    style.visuals.widgets.active.fg_stroke.color = scheme.text_primary;
    style.visuals.widgets.inactive.fg_stroke.color = scheme.text_primary;
    style.visuals.widgets.hovered.fg_stroke.color = scheme.text_primary;
    style.visuals.widgets.open.fg_stroke.color = scheme.text_primary;

    style.visuals.widgets.noninteractive.bg_fill = scheme.bg_secondary;
    style.visuals.widgets.active.bg_fill = scheme.bg_secondary;
    style.visuals.widgets.inactive.bg_fill = scheme.bg_secondary;
    style.visuals.widgets.hovered.bg_fill = scheme.bg_secondary;
    style.visuals.widgets.open.bg_fill = scheme.bg_secondary;

    style.visuals.widgets.noninteractive.weak_bg_fill = scheme.bg_secondary;
    style.visuals.widgets.active.weak_bg_fill = scheme.bg_secondary;
    style.visuals.widgets.inactive.weak_bg_fill = scheme.bg_secondary;
    style.visuals.widgets.hovered.weak_bg_fill = scheme.bg_secondary;
    style.visuals.widgets.open.weak_bg_fill = scheme.bg_secondary;

    style.visuals.menu_corner_radius = corner_radius_xs();
    style.visuals.window_corner_radius = corner_radius_xs();
    style.visuals.window_stroke = Stroke::new(1.0, Color32::TRANSPARENT);
    style.visuals.window_shadow = Shadow {
        color: scheme.shadow.opacity(0.2),
        blur: 4,
        offset: [0, 0],
        spread: 4,
    };
    style.visuals.popup_shadow = Shadow {
        color: scheme.shadow.opacity(0.2),
        blur: 8,
        offset: [0, 0],
        spread: 2,
    };
    style.visuals.window_fill = scheme.bg_secondary;

    style.spacing.menu_margin = Margin::same(0);
    style.spacing.window_margin = Margin::same(8);
    style.visuals.selection.bg_fill = scheme.highlight.opacity(0.6);
    style.visuals.selection.stroke.color = scheme.highlight;

    configure_default_fonts(context);

    context.set_style(style);
}

pub fn corner_radius_xs() -> CornerRadius {
    CornerRadius::same(2)
}

pub fn corner_radius_sm() -> CornerRadius {
    CornerRadius::same(4)
}

fn configure_default_fonts(ctx: &egui::Context) {
    ctx.add_font(FontInsert::new(
        "ibm_plex_mono_medium",
        FontData::from_static(include_bytes!(
            "../assets/fonts/IBMPlexMono-Medium_ss04.ttf"
        )),
        vec![
            InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Highest,
            },
            InsertFontFamily {
                family: egui::FontFamily::Monospace,
                priority: egui::epaint::text::FontPriority::Highest,
            },
        ],
    ));

    if cfg!(target_os = "windows") {
        let buf = std::fs::read("C:\\Windows\\Fonts\\SegoeIcons.ttf").unwrap();
        ctx.add_font(FontInsert::new(
            "segeo_icons",
            FontData::from_owned(buf),
            vec![InsertFontFamily {
                family: egui::FontFamily::Proportional,
                priority: egui::epaint::text::FontPriority::Lowest,
            }],
        ));
    }
}

pub fn configure_input_with_border(style: &mut Style) {
    let scheme = get_scheme();
    style.visuals.widgets.active.fg_stroke = Stroke::new(0.0, scheme.border_primary);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(0.0, scheme.border_primary);
    style.visuals.widgets.inactive.bg_fill = scheme.bg_secondary;
    style.visuals.widgets.hovered.bg_fill = scheme.border_primary;
}

pub fn configure_combo_box(style: &mut Style) {
    configure_input_with_border(style);
    style.spacing.interact_size.y = 34.0;
    style.spacing.item_spacing = [0.0, 0.0].into();
    style.spacing.button_padding = [8.0, 4.0].into();
    style.visuals.widgets.hovered.expansion = 0.0;
    style.visuals.menu_corner_radius = CornerRadius::ZERO;
    style.spacing.menu_margin = Margin::same(0);
}

pub fn configure_combo_item(style: &mut Style) {
    let scheme = get_scheme();
    style.spacing.interact_size.y = 34.0;
    style.spacing.item_spacing = [0.0, 0.0].into();
    style.spacing.button_padding = [8.0, 0.0].into();
    style.visuals.widgets.inactive.fg_stroke.color = with_opacity(scheme.text_primary, 0.6);
    style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.active.bg_stroke = Stroke::NONE;
    style.visuals.widgets.active.bg_fill = scheme.text_primary;
    style.visuals.widgets.active.fg_stroke.color = scheme.bg_primary;
    style.visuals.widgets.active.weak_bg_fill = scheme.text_primary;
    style.visuals.widgets.active.expansion = 0.0;
    style.visuals.widgets.inactive.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.inactive.expansion = 0.0;
    style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.hovered.weak_bg_fill = scheme.border_primary;
    style.visuals.widgets.hovered.bg_fill = scheme.text_primary;
    style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
    style.visuals.widgets.hovered.expansion = 0.0;
    style.visuals.selection.bg_fill = scheme.text_primary;
    style.visuals.selection.stroke.color = scheme.bg_primary;
}
