use bevy::ecs::system::{SystemParam, SystemState};
use bevy::prelude::*;
use bevy_egui::egui;

use crate::ui::colors::get_scheme;
use crate::ui::timeline::TimelineSettings;
use crate::ui::widgets::WidgetSystem;
use crate::ui::{label::ELabel, utils::MarginSides};

use super::node_color_picker;

#[derive(SystemParam)]
pub struct InspectorTimeline<'w> {
    timeline_settings: ResMut<'w, TimelineSettings>,
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

        node_color_picker(
            ui,
            "PLAYHEAD & PLAYED TRAIL COLOR",
            &mut timeline_settings.accent_color,
        );
        node_color_picker(
            ui,
            "LATEST MARKER & FUTURE TRAIL COLOR",
            &mut timeline_settings.future_trail_color,
        );
    }
}
