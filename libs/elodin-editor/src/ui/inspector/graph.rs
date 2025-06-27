use std::{ops::Range, time::Duration};

use bevy::ecs::{
    entity::Entity,
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use impeller2_wkt::{ComponentPath, GraphType, QueryType};
use smallvec::SmallVec;

use egui::{Align, Color32};
use impeller2_bevy::ComponentMetadataRegistry;

use crate::{
    EqlContext,
    ui::{
        SettingModal, SettingModalState,
        button::{EButton, ECheckboxButton},
        colors::{EColor, get_scheme},
        inspector::{color_popup, eql_autocomplete, query},
        label::{self, label_with_buttons},
        plot::GraphState,
        query_plot::QueryPlotData,
        theme::{self, configure_input_with_border},
        utils::MarginSides,
        widgets::WidgetSystem,
    },
};

use super::InspectorIcons;

#[derive(SystemParam)]
pub struct InspectorGraph<'w, 's> {
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
    query_plots: Query<'w, 's, &'static mut QueryPlotData>,
    eql_context: Res<'w, EqlContext>,
}

impl WidgetSystem for InspectorGraph<'_, '_> {
    type Args = (InspectorIcons, Entity);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, graph_id) = args;

        let InspectorGraph {
            mut setting_modal_state,
            metadata_store,
            mut graph_states,
            mut query_plots,
            eql_context,
        } = state_mut;

        let graph_label_margin = egui::Margin::same(0).top(10.0).bottom(14.0);
        let Ok(mut graph_state) = graph_states.get_mut(graph_id) else {
            return;
        };
        let graph_state = &mut *graph_state;
        let query_plot = query_plots.get_mut(graph_id);

        if query_plot.is_ok() {
            label::editable_label_with_buttons(
                ui,
                [],
                &mut graph_state.label,
                get_scheme().text_primary,
                graph_label_margin,
            );
        } else {
            let [add_clicked] = label::editable_label_with_buttons(
                ui,
                [icons.add],
                &mut graph_state.label,
                get_scheme().text_primary,
                graph_label_margin,
            );
            if add_clicked {
                setting_modal_state.0 = Some(SettingModal::Graph(graph_id, None));
            }
        }

        ui.separator();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("WIDTH").color(get_scheme().text_secondary));
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.add(egui::DragValue::new(&mut graph_state.line_width).speed(0.2))
                    });
                });

                ui.add_space(8.0);
                ui.style_mut().spacing.slider_width = ui.available_size().x;
                ui.style_mut().visuals.widgets.inactive.bg_fill = get_scheme().border_primary;
                ui.add(egui::Slider::new(&mut graph_state.line_width, 1.0..=15.0).show_value(false))
            });
        ui.separator();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("GRAPH TYPE").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                theme::configure_combo_box(ui.style_mut());
                ui.style_mut().spacing.combo_width = ui.available_size().x;
                egui::ComboBox::from_id_salt("graph_type")
                    .selected_text(match graph_state.graph_type {
                        GraphType::Line => "Line",
                        GraphType::Point => "Point",
                        GraphType::Bar => "Bar",
                    })
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());
                        ui.selectable_value(&mut graph_state.graph_type, GraphType::Line, "Line");
                        ui.selectable_value(&mut graph_state.graph_type, GraphType::Point, "Point");
                        ui.selectable_value(&mut graph_state.graph_type, GraphType::Bar, "Bar");
                    });
            });
        ui.separator();
        auto_range(
            ui,
            "Y Bounds",
            &mut graph_state.auto_y_range,
            &mut graph_state.y_range,
        );
        if query_plot.is_ok() {
            ui.separator();
            auto_range(
                ui,
                "X Bounds",
                &mut graph_state.auto_x_range,
                &mut graph_state.x_range,
            );
        }
        ui.separator();

        if let Ok(mut query_plot) = query_plot {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(0, 8))
                .show(ui, |ui| {
                    let max_width = ui.available_width();
                    ui.style_mut().spacing.item_spacing = egui::vec2(0.0, 8.0);
                    ui.label(egui::RichText::new("QUERY TYPE").color(get_scheme().text_secondary));
                    ui.scope(|ui| {
                        theme::configure_combo_box(ui.style_mut());
                        ui.style_mut().spacing.combo_width = ui.available_size().x;
                        let prev_query_type = query_plot.data.query_type;
                        egui::ComboBox::from_id_salt("query_type")
                            .selected_text(match query_plot.data.query_type {
                                QueryType::EQL => "EQL",
                                QueryType::SQL => "SQL",
                            })
                            .show_ui(ui, |ui| {
                                theme::configure_combo_item(ui.style_mut());
                                ui.selectable_value(
                                    &mut query_plot.data.query_type,
                                    QueryType::EQL,
                                    "EQL",
                                );
                                ui.selectable_value(
                                    &mut query_plot.data.query_type,
                                    QueryType::SQL,
                                    "SQL",
                                );
                            });
                        if let (QueryType::EQL, QueryType::SQL) =
                            (prev_query_type, query_plot.data.query_type)
                        {
                            if let Ok(sql) = eql_context.0.sql(&query_plot.data.query) {
                                query_plot.data.query = sql;
                            }
                        }
                    });
                    ui.separator();
                    ui.label(egui::RichText::new("Query").color(get_scheme().text_secondary));
                    configure_input_with_border(ui.style_mut());
                    let query_type = query_plot.data.query_type;
                    let query_res = ui.add(query(&mut query_plot.data.query, query_type));
                    if query_type == QueryType::EQL {
                        eql_autocomplete(
                            ui,
                            &eql_context.0,
                            &query_res,
                            &mut query_plot.data.query,
                        );
                    }
                    let enter_key = query_res.lost_focus()
                        && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));
                    ui.separator();
                    ui.label(egui::RichText::new("Behavior").color(get_scheme().text_secondary));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut query_plot.data.auto_refresh, "Auto Refresh?");
                        ui.add_space(8.0);
                        let mut seconds = query_plot.data.refresh_interval.as_secs_f64();
                        if ui
                            .add_enabled(
                                query_plot.data.auto_refresh,
                                egui::DragValue::new(&mut seconds)
                                    .suffix("s")
                                    .speed(0.5)
                                    .range(0.001..=120.0),
                            )
                            .changed()
                        {
                            query_plot.data.refresh_interval = Duration::from_secs_f64(seconds);
                        }
                    });
                    ui.add_space(8.0);

                    if ui
                        .add_sized([max_width, 32.0], EButton::green("Refresh"))
                        .clicked()
                        || enter_key
                    {
                        query_plot.last_refresh = None;
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Color").color(get_scheme().text_secondary));
                    let color_id = ui.auto_id_with("color");
                    let btn_resp = ui.add(EButton::new("Set Color"));
                    if btn_resp.clicked() {
                        ui.memory_mut(|m| m.toggle_popup(color_id));
                    }

                    if ui.memory(|mem| mem.is_popup_open(color_id)) {
                        let mut color = query_plot.data.color.into_color32();
                        let popup_response =
                            color_popup(ui, &mut color, color_id, btn_resp.rect.min);
                        if !btn_resp.clicked()
                            && (ui.input(|i| i.key_pressed(egui::Key::Escape))
                                || popup_response.clicked_elsewhere())
                        {
                            ui.memory_mut(|mem| mem.close_popup());
                        }
                        query_plot.data.color = impeller2_wkt::Color::from_color32(color);
                    }
                });
        } else {
            let mut remove_list: SmallVec<[ComponentPath; 1]> = SmallVec::new();
            for (path, component) in graph_state.components.iter_mut() {
                let Some(metadata) = metadata_store.get(&path.id) else {
                    continue;
                };

                let component_label = metadata.name.clone();
                let element_names = metadata.element_names();

                let component_label_margin = egui::Margin::symmetric(0, 18);
                let [subtract_clicked] = label_with_buttons(
                    ui,
                    [icons.subtract],
                    component_label,
                    get_scheme().text_tertiary,
                    component_label_margin,
                );
                if subtract_clicked {
                    remove_list.push(path.clone());
                }

                component_value(ui, component, element_names);
            }
            for path in remove_list.into_iter() {
                graph_state.remove_component(&path);
            }
        }
    }
}

pub fn auto_range(ui: &mut egui::Ui, label: &str, auto_range: &mut bool, range: &mut Range<f64>) {
    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(0, 8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(label).color(get_scheme().text_secondary));
                ui.add_space(8.0);
                theme::configure_input_with_border(ui.style_mut());
                ui.checkbox(auto_range, "Auto Bounds?");
            });
            ui.add_space(8.0);
            ui.style_mut().visuals.widgets.hovered.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
            ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
            ui.style_mut().override_font_id =
                Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
            ui.horizontal_wrapped(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Min").color(get_scheme().text_secondary));
                    ui.add_space(16.0);
                    ui.add_enabled(
                        !*auto_range,
                        egui::DragValue::new(&mut range.start).speed(0.01),
                    );
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Max").color(get_scheme().text_secondary));
                    ui.add_space(16.0);
                    ui.add_enabled(
                        !*auto_range,
                        egui::DragValue::new(&mut range.end).speed(0.01),
                    );
                })
            })
        });
}

fn component_value(
    ui: &mut egui::Ui,
    component_values: &mut [(bool, egui::Color32)],
    element_names: &str,
) {
    let element_names = element_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(Option::Some)
        .chain(std::iter::repeat(None));
    ui.horizontal_wrapped(|ui| {
        for (index, ((enabled, color), element_name)) in
            component_values.iter_mut().zip(element_names).enumerate()
        {
            ui.style_mut().override_font_id =
                Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
            let label = element_name
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("[{index}]"));
            let value_toggle =
                ui.add(ECheckboxButton::new(label.to_string(), *enabled).on_color(*color));

            if value_toggle.clicked() {
                *enabled = !*enabled;
            }
            let color_id = ui.auto_id_with("color");
            if value_toggle.secondary_clicked() {
                ui.memory_mut(|mem| mem.toggle_popup(color_id));
            }
            if ui.memory(|mem| mem.is_popup_open(color_id)) {
                let popup_response = color_popup(ui, color, color_id, value_toggle.rect.min);
                if !value_toggle.secondary_clicked()
                    && (ui.input(|i| i.key_pressed(egui::Key::Escape))
                        || popup_response.clicked_elsewhere())
                {
                    ui.memory_mut(|mem| mem.close_popup());
                }
            }
        }
    });
}
