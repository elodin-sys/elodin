use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy_egui::egui::{self, Align};

use crate::ui::timeline::TimelineSettings;
use crate::ui::widgets::WidgetSystem;
use crate::ui::{label::ELabel, theme, utils::MarginSides};
use crate::{ReplayMode, ui::colors::get_scheme};

use super::node_color_picker;

#[derive(SystemParam)]
pub struct InspectorTimeline<'w> {
    timeline_settings: ResMut<'w, TimelineSettings>,
    replay_mode: Option<Res<'w, ReplayMode>>,
}

impl WidgetSystem for InspectorTimeline<'_> {
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
            replay_mode,
        } = state.get_mut(world);

        ui.spacing_mut().item_spacing.y = 8.0;
        ui.add(
            ELabel::new("Timeline")
                .padding(egui::Margin::same(8).bottom(24.0))
                .bottom_stroke(egui::Stroke {
                    color: scheme.border_primary,
                    width: 1.0,
                })
                .margin(egui::Margin::same(0).bottom(16.0)),
        );

        node_color_picker(ui, "ACCENT COLOR", &mut timeline_settings.accent_color);
        node_color_picker(
            ui,
            "FUTURE TRAIL COLOR",
            &mut timeline_settings.future_trail_color,
        );

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("FOLLOW LATEST IF STREAMING")
                            .color(scheme.text_secondary),
                    );
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        theme::configure_input_with_border(ui.style_mut());
                        ui.checkbox(&mut timeline_settings.follow_latest_if_streaming, "");
                    });
                });

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Accent color drives the timeline scrubber and LIVE state. Future trail color controls the 3D path segment ahead of the current playback position.",
                    )
                    .color(scheme.text_tertiary),
                );
            });

        if replay_mode.is_some() {
            ui.separator();
            ui.label(
                egui::RichText::new(
                    "Replay mode ignores the auto-LIVE startup preference for the current session.",
                )
                .color(scheme.text_tertiary),
            );
        }
    }
}
