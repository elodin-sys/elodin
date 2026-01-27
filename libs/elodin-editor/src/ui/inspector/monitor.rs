use bevy::ecs::system::SystemParam;
use bevy::prelude::{Entity, Query, Res};
use bevy_egui::egui;

use crate::{
    EqlContext,
    ui::{
        colors::get_scheme,
        inspector::{eql_autocomplete, inspector_text_field},
        monitor::MonitorData,
        theme,
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorMonitor<'w, 's> {
    monitors: Query<'w, 's, &'static mut MonitorData>,
    eql_context: Res<'w, EqlContext>,
}

impl WidgetSystem for InspectorMonitor<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorMonitor {
            mut monitors,
            eql_context,
        } = state.get_mut(world);
        let Ok(mut monitor) = monitors.get_mut(entity) else {
            return;
        };

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("COMPONENT").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                theme::configure_input_with_border(ui.style_mut());
                let res = ui.add(inspector_text_field(
                    &mut monitor.component_name,
                    "Component name (i.e a.world_pos)",
                ));
                eql_autocomplete(ui, &eql_context.0, &res, &mut monitor.component_name);
            });
    }
}
