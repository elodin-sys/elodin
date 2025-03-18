use bevy_egui::egui::{self, CornerRadius, FontData, Margin, Stroke, Style, epaint::Shadow};
use egui::epaint::text::{FontInsert, InsertFontFamily};

use super::colors::{self, with_opacity};

pub fn set_theme(context: &mut egui::Context) {
    let mut style = (*context.style()).clone();

    style.spacing.item_spacing = egui::vec2(0., 0.);
    style.visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, colors::BORDER_GREY);
    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, colors::PRIMARY_CREAME);
    style.visuals.widgets.hovered.bg_stroke =
        egui::Stroke::new(1.0, with_opacity(colors::PRIMARY_CREAME, 0.6));
    style.visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, colors::BORDER_GREY);

    style.visuals.widgets.noninteractive.fg_stroke.color = colors::PRIMARY_CREAME;
    style.visuals.widgets.inactive.fg_stroke.color = colors::PRIMARY_CREAME;
    style.visuals.widgets.active.fg_stroke.color = colors::PRIMARY_CREAME;

    style.visuals.widgets.noninteractive.bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.active.bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.hovered.bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.open.bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.inactive.bg_fill = colors::PRIMARY_SMOKE;

    style.visuals.widgets.noninteractive.weak_bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.active.weak_bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.hovered.weak_bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.open.weak_bg_fill = colors::PRIMARY_SMOKE;
    style.visuals.widgets.inactive.weak_bg_fill = colors::PRIMARY_SMOKE;

    style.visuals.menu_corner_radius = corner_radius_xs();
    style.visuals.window_corner_radius = corner_radius_xs();
    style.visuals.window_stroke = Stroke::new(1.0, colors::PRIMARY_ONYX_8);
    style.visuals.window_shadow = Shadow {
        color: colors::with_opacity(colors::BLACK_BLACK_600, 0.25),
        blur: 4,
        offset: [0, 0],
        spread: 4,
    };
    style.visuals.window_fill = colors::PRIMARY_SMOKE;

    style.spacing.menu_margin = Margin::same(8);
    style.spacing.window_margin = Margin::same(8);
    style.visuals.selection.bg_fill = with_opacity(colors::HYPERBLUE_DEFAULT, 0.6);
    style.visuals.selection.stroke.color = colors::HYPERBLUE_DEFAULT;

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
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, colors::PRIMARY_ONYX_9);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, colors::PRIMARY_ONYX_9);
    style.visuals.widgets.inactive.bg_fill = colors::PRIMARY_ONYX;
    style.visuals.widgets.hovered.bg_fill = colors::PRIMARY_ONYX_8;
}

pub fn configure_combo_box(style: &mut Style) {
    configure_input_with_border(style);
    style.spacing.item_spacing = [16.0, 16.0].into();
    style.spacing.button_padding = [16.0, 16.0].into();
    style.visuals.widgets.hovered.expansion = 0.0;
}

pub fn configure_combo_item(style: &mut Style) {
    style.spacing.interact_size.y = 37.0;
    style.spacing.item_spacing = [8.0, 8.0].into();
    style.spacing.button_padding = [8.0, 8.0].into();
    style.visuals.widgets.inactive.fg_stroke.color = with_opacity(colors::PRIMARY_CREAME, 0.6);
    style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.active.bg_stroke = Stroke::NONE;
    style.visuals.widgets.active.bg_fill = colors::PRIMARY_CREAME;
    style.visuals.widgets.active.fg_stroke.color = colors::BLACK_BLACK_600;
    style.visuals.widgets.active.weak_bg_fill = colors::PRIMARY_CREAME;
    style.visuals.widgets.active.expansion = 0.0;
    style.visuals.widgets.inactive.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.inactive.expansion = 0.0;
    style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
    style.visuals.widgets.hovered.weak_bg_fill = colors::PRIMARY_ONYX_8;
    style.visuals.widgets.hovered.bg_fill = colors::PRIMARY_CREAME;
    style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
    style.visuals.widgets.hovered.expansion = 0.0;
    style.visuals.selection.bg_fill = colors::PRIMARY_CREAME;
    style.visuals.selection.stroke.color = colors::BLACK_BLACK_600;
}
