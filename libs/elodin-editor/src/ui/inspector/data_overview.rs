use bevy::ecs::system::SystemParam;
use bevy::prelude::{Res, ResMut};
use bevy_egui::egui;

use impeller2::types::ComponentId;

use crate::{
    EqlContext,
    ui::{
        button::EColorButton,
        colors::{ColorExt, get_scheme},
        data_overview::{ComponentTimeRanges, component_to_table_name, row_color},
        inspector::color_popup,
        theme,
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorDataOverview<'w> {
    time_ranges: ResMut<'w, ComponentTimeRanges>,
    eql_context: Res<'w, EqlContext>,
}

fn collect_components(
    parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
    result: &mut Vec<(ComponentId, String, String)>,
) {
    for part in parts.values() {
        if let Some(component) = &part.component {
            let label = component.name.clone();
            let table_name = component_to_table_name(&label);
            result.push((part.id, label, table_name));
        }
        collect_components(&part.children, result);
    }
}

impl WidgetSystem for InspectorDataOverview<'_> {
    type Args = ();
    type Output = ();

    fn ui_system(
        _world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        (): Self::Args,
    ) -> Self::Output {
        let mut params = state.get_mut(_world);
        let mut entries: Vec<(ComponentId, String, String, bool)> = Vec::new();
        let mut components: Vec<(ComponentId, String, String)> = Vec::new();
        collect_components(&params.eql_context.0.component_parts, &mut components);
        for (component_id, label, table_name) in components {
            let has_data = params.time_ranges.ranges.contains_key(&table_name);
            entries.push((component_id, label, table_name, has_data));
        }
        entries.sort_by(|a, b| match (a.3, b.3) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.1.cmp(&b.1),
        });

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("COMPONENTS").color(get_scheme().text_secondary));
                ui.add_space(8.0);

                if entries.is_empty() {
                    ui.label(
                        egui::RichText::new("No components available")
                            .color(get_scheme().text_secondary),
                    );
                    return;
                }

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (index, (component_id, label, table_name, _)) in
                            entries.iter().enumerate()
                        {
                            let default_color = row_color(index);
                            let mut clear_tables = false;
                            {
                                let settings = params
                                    .time_ranges
                                    .row_settings
                                    .entry(*component_id)
                                    .or_default();
                                if settings.color.is_none() {
                                    settings.color = Some(default_color);
                                }
                                if settings.custom_name.is_none() {
                                    settings.custom_name = Some(label.clone());
                                }
                                let row_color =
                                    settings.color.unwrap_or_else(|| get_scheme().text_primary);
                                let row_color = if settings.enabled {
                                    row_color
                                } else {
                                    row_color.opacity(0.35)
                                };

                                ui.horizontal(|ui| {
                                    // 1. Checkbox
                                    let mut enabled_value = settings.enabled;
                                    if ui.checkbox(&mut enabled_value, "").changed() {
                                        settings.enabled = enabled_value;
                                        if !enabled_value {
                                            clear_tables = true;
                                        }
                                    }

                                    // 2. Editable name
                                    let name =
                                        settings.custom_name.get_or_insert_with(|| label.clone());
                                    theme::configure_input_with_border(ui.style_mut());
                                    let text_edit = egui::TextEdit::singleline(name)
                                        .text_color(row_color)
                                        .desired_width(ui.available_width() - 40.0);
                                    ui.add(text_edit);

                                    // 3. Color button
                                    let mut color_value = settings.color.unwrap_or(row_color);
                                    let color_id = ui.auto_id_with(("data_overview_color", index));
                                    let color_btn = ui.add(EColorButton::new(color_value));
                                    if color_btn.clicked() {
                                        egui::Popup::toggle_id(ui.ctx(), color_id);
                                    }
                                    if color_popup(ui, &mut color_value, color_id, &color_btn)
                                        .is_some()
                                        && Some(color_value) != settings.color
                                    {
                                        settings.color = Some(color_value);
                                    }
                                });
                            }
                            if clear_tables {
                                params.time_ranges.ranges.remove(table_name);
                                params.time_ranges.row_counts.remove(table_name);
                                params.time_ranges.sparklines.remove(table_name);
                            }
                        }
                    });
            });
    }
}
