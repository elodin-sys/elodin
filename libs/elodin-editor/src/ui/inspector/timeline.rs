use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy_egui::egui;
use impeller2_wkt::Line3d;

use crate::ui::colors::get_scheme;
use crate::ui::timeline::TimelineSettings;
use crate::ui::widgets::WidgetSystem;
use crate::ui::{label::ELabel, utils::MarginSides};

use super::line3d::line3d_controls;
use super::node_color_picker;
use crate::ui::widgets::SystemStateExt;

#[derive(SystemParam)]
pub struct InspectorTimeline<'w, 's> {
    timeline_settings: ResMut<'w, TimelineSettings>,
    lines: Query<'w, 's, (Entity, &'static mut Line3d)>,
}

impl WidgetSystem for InspectorTimeline<'_, '_> {
    type Args = ();
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        _args: Self::Args,
    ) {
        let scheme = get_scheme();
        let InspectorTimeline {
            mut timeline_settings,
            mut lines,
        } = state.params_mut(world);

        ui.spacing_mut().item_spacing.y = 8.0;

        // ── Timeline: global played/future trail colors ──────────────────
        section_header(ui, "TIMELINE", scheme.border_primary);
        node_color_picker(ui, "PLAYED COLOR", &mut timeline_settings.played_color);
        node_color_picker(ui, "FUTURE COLOR", &mut timeline_settings.future_color);

        // ── Line 3D: per-line overrides for every `line_3d` in the schematic ─
        let mut lines: Vec<(Entity, Mut<Line3d>)> = lines.iter_mut().collect();
        if lines.is_empty() {
            return;
        }
        lines.sort_by(|(_, a), (_, b)| a.eql.cmp(&b.eql));

        ui.add_space(16.0);
        section_header(ui, "LINE 3D", scheme.border_primary);

        let single_line = lines.len() == 1;
        let played_tl = timeline_settings.played_color;
        let future_tl = timeline_settings.future_color;
        // Subtle white piping around each line so multiple `line_3d` entries read
        // as distinct, roomy cards.
        let border = egui::Stroke {
            width: 1.0,
            color: crate::ui::colors::with_opacity(egui::Color32::WHITE, 0.4),
        };
        for (entity, line) in lines.iter_mut() {
            egui::Frame::NONE
                .stroke(border)
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(12))
                .outer_margin(egui::Margin::symmetric(0, 6))
                .show(ui, |ui| {
                    egui::CollapsingHeader::new(
                        egui::RichText::new(line.eql.clone())
                            .monospace()
                            .color(scheme.text_primary),
                    )
                    .id_salt(("line3d_inspector", *entity))
                    .default_open(single_line)
                    .show(ui, |ui| {
                        // Edits mutate the live `Line3d`; `tiles_to_schematic`
                        // mirrors it into `CurrentSchematic` (and KDL on save).
                        line3d_controls(ui, line, played_tl, future_tl);
                    });
                });
        }
    }
}

/// Section title with an underline, used to separate Timeline from Line 3D.
fn section_header(ui: &mut egui::Ui, title: &str, border: egui::Color32) {
    ui.add(
        ELabel::new(title)
            .padding(egui::Margin::same(8).bottom(24.0))
            .bottom_stroke(egui::Stroke {
                color: border,
                width: 1.0,
            })
            .margin(egui::Margin::same(0).bottom(8.0)),
    );
}
