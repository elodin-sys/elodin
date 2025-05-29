use std::{ops::Range, time::Duration};

use bevy::ecs::{
    entity::Entity,
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use impeller2_wkt::GraphType;
use smallvec::SmallVec;

use egui::{
    Align, Color32,
    color_picker::{Alpha, color_picker_color32},
};
use impeller2::types::{ComponentId, EntityId};
use impeller2_bevy::ComponentMetadataRegistry;

use crate::ui::{
    EntityData, SettingModal, SettingModalState,
    colors::{self, ColorExt, get_scheme},
    theme::{self, configure_input_with_border},
    utils::MarginSides,
    widgets::{
        WidgetSystem,
        button::{EButton, ECheckboxButton, EColorButton},
        label::{self, label_with_buttons},
        plot::GraphState,
        sql_plot::SqlPlot,
    },
};

use super::InspectorIcons;

#[derive(SystemParam)]
pub struct InspectorGraph<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
    sql_plots: Query<'w, 's, &'static mut SqlPlot>,
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
            entities,
            mut setting_modal_state,
            metadata_store,
            mut graph_states,
            mut sql_plots,
        } = state_mut;

        let graph_label_margin = egui::Margin::same(0).top(10.0).bottom(14.0);
        let Ok(mut graph_state) = graph_states.get_mut(graph_id) else {
            return;
        };
        let graph_state = &mut *graph_state;
        let sql_plot = sql_plots.get_mut(graph_id);

        if sql_plot.is_ok() {
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
                setting_modal_state.0 = Some(SettingModal::Graph(graph_id, None, None));
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
        if sql_plot.is_ok() {
            ui.separator();
            auto_range(
                ui,
                "X Bounds",
                &mut graph_state.auto_x_range,
                &mut graph_state.x_range,
            );
        }
        ui.separator();

        if let Ok(mut sql_plot) = sql_plot {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(0, 8))
                .show(ui, |ui| {
                    let max_width = ui.available_width();
                    ui.style_mut().spacing.item_spacing = egui::vec2(0.0, 8.0);
                    ui.label(egui::RichText::new("Query").color(get_scheme().text_secondary));
                    configure_input_with_border(ui.style_mut());
                    let query_res = query(ui, &mut sql_plot.current_query);
                    let enter_key = query_res.lost_focus()
                        && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));
                    ui.separator();
                    ui.label(egui::RichText::new("Behavior").color(get_scheme().text_secondary));
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut sql_plot.auto_refresh, "Auto Refresh?");
                        ui.add_space(8.0);
                        let mut seconds = sql_plot.refresh_interval.as_secs_f64();
                        if ui
                            .add_enabled(
                                sql_plot.auto_refresh,
                                egui::DragValue::new(&mut seconds)
                                    .speed(0.5)
                                    .range(0.02..=120.0),
                            )
                            .changed()
                        {
                            sql_plot.refresh_interval = Duration::from_secs_f64(seconds);
                        }
                    });
                    ui.add_space(8.0);
                    if ui
                        .add_sized([max_width, 32.0], EButton::green("Refresh"))
                        .clicked()
                        || enter_key
                    {
                        sql_plot.last_refresh = None;
                    }
                });
        } else {
            let mut remove_list: SmallVec<[(EntityId, ComponentId); 1]> = SmallVec::new();
            for (entity_id, components) in &mut graph_state.entities {
                let entity = entities.iter().find(|(eid, _, _, _)| *eid == entity_id);

                if let Some((_, _, _, entity_metadata)) = entity {
                    let entity_label_margin = egui::Margin::same(0).top(18.0).bottom(4.0);
                    let [add_clicked] = label_with_buttons(
                        ui,
                        [icons.add],
                        &entity_metadata.name,
                        get_scheme().text_primary,
                        entity_label_margin,
                    );
                    if add_clicked {
                        setting_modal_state.0 =
                            Some(SettingModal::Graph(graph_id, Some(*entity_id), None));
                    }

                    for (component_id, component_values) in components.iter_mut() {
                        let Some(metadata) = metadata_store.get_metadata(component_id) else {
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
                            remove_list.push((*entity_id, *component_id));
                        }

                        component_value(ui, component_values, element_names);
                    }
                }
            }
            for (entity_id, component_id) in remove_list.into_iter() {
                graph_state.remove_component(&entity_id, &component_id);
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

pub fn query(ui: &mut egui::Ui, query: &mut String) -> egui::Response {
    let scheme = get_scheme();
    ui.scope(|ui| {
        ui.style_mut().visuals.widgets.inactive = egui::style::WidgetVisuals {
            bg_fill: scheme.bg_primary,
            weak_bg_fill: scheme.bg_primary,
            bg_stroke: egui::Stroke::NONE,
            corner_radius: theme::corner_radius_xs(),
            fg_stroke: egui::Stroke::new(1.0, scheme.text_primary),
            expansion: 0.0,
        };
        ui.style_mut().visuals.widgets.active = egui::style::WidgetVisuals {
            bg_stroke: egui::Stroke::new(1.0, scheme.highlight),
            ..ui.style_mut().visuals.widgets.inactive
        };
        ui.style_mut().visuals.widgets.hovered = egui::style::WidgetVisuals {
            bg_stroke: egui::Stroke::new(1.0, scheme.highlight.opacity(0.5)),
            ..ui.style_mut().visuals.widgets.inactive
        };
        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 12.0;
        ui.add(
            egui::TextEdit::singleline(query)
                .font(font_id)
                .hint_text("SQL Query (i.e select * from table)")
                .margin(8.0),
        )
    })
    .inner
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
                let popup_response = egui::Area::new(color_id)
                    .kind(egui::UiKind::Picker)
                    .order(egui::Order::Foreground)
                    .fixed_pos(value_toggle.rect.min)
                    .default_width(300.0)
                    .show(ui.ctx(), |ui| {
                        theme::configure_input_with_border(ui.style_mut());
                        ui.spacing_mut().slider_width = 275.;
                        ui.spacing_mut().button_padding = egui::vec2(6.0, 4.0);
                        ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);

                        ui.add_space(8.0);
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                for elem_color in &colors::ALL_COLORS_DARK[..24] {
                                    if ui.add(EColorButton::new(*elem_color)).clicked() {
                                        *color = *elem_color;
                                    }
                                }
                            });
                            ui.add_space(8.0);
                            color_picker_color32(ui, color, Alpha::Opaque);
                        });
                    })
                    .response;

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
