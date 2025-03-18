use bevy::{
    ecs::system::SystemParam,
    prelude::{Entity, Query},
};
use egui::{Color32, CornerRadius, RichText, Stroke};

use crate::ui::{actions::ActionTile, colors, widgets::WidgetSystem};

#[derive(SystemParam)]
pub struct InspectorAction<'w, 's> {
    action_tiles: Query<'w, 's, &'static mut ActionTile>,
}

impl WidgetSystem for InspectorAction<'_, '_> {
    type Args = Entity;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut tile) = state.action_tiles.get_mut(entity) else {
            return;
        };

        let style = ui.style_mut();
        style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
        style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
        style.visuals.widgets.open.corner_radius = CornerRadius::ZERO;

        style.visuals.widgets.active.fg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
        style.visuals.widgets.active.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
        style.visuals.widgets.hovered.fg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
        style.visuals.widgets.hovered.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
        style.visuals.widgets.open.fg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
        style.visuals.widgets.open.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);

        style.spacing.button_padding = [16.0, 16.0].into();

        style.visuals.widgets.active.bg_fill = colors::SURFACE_SECONDARY;
        style.visuals.widgets.open.bg_fill = colors::SURFACE_SECONDARY;
        style.visuals.widgets.inactive.bg_fill = colors::SURFACE_SECONDARY;
        style.visuals.widgets.hovered.bg_fill = colors::SURFACE_SECONDARY;
        ui.add(
            egui::Label::new(
                RichText::new("BUTTON LABEL")
                    .color(colors::PRIMARY_ONYX_5)
                    .size(12.),
            )
            .selectable(false),
        );
        ui.add_space(16.);

        ui.add_sized(
            egui::vec2(ui.available_width(), 50.0),
            egui::TextEdit::singleline(&mut tile.button_name).margin(egui::Margin::same(16)),
        );
        ui.add_space(32.);
        ui.add(
            egui::Label::new(
                RichText::new("LUA CMD")
                    .color(colors::PRIMARY_ONYX_5)
                    .size(12.),
            )
            .selectable(false),
        );

        ui.add_space(16.);
        ui.add_sized(
            egui::vec2(ui.available_width(), 50.0),
            egui::TextEdit::singleline(&mut tile.lua)
                .margin(egui::Margin::same(16))
                .desired_width(0.0),
        );
    }
}
