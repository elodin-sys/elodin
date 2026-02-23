use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
    time::Duration,
};

use bevy::ecs::{
    entity::Entity,
    system::{Local, Query, Res, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2_wkt::{ComponentPath, GraphType, QueryType};
use smallvec::SmallVec;

use egui::{Align, Color32};
use impeller2_bevy::{ComponentMetadataRegistry, ComponentSchemaRegistry};

use crate::{
    EqlContext,
    ui::{
        button::{EButton, EColorButton},
        colors::{EColor, get_color_by_index_all, get_scheme},
        inspector::{color_popup, eql_autocomplete, inspector_text_field, query, search},
        label::{self, label_with_buttons},
        plot::GraphState,
        query_plot::QueryPlotData,
        schematic::EqlExt,
        theme::{self, configure_input_with_border},
        utils::MarginSides,
        widgets::WidgetSystem,
    },
};

use super::InspectorIcons;

#[derive(Default)]
struct AddComponentState {
    filter: String,
    expression: String,
}

#[derive(SystemParam)]
pub struct InspectorGraph<'w, 's> {
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    schema_store: Res<'w, ComponentSchemaRegistry>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
    query_plots: Query<'w, 's, &'static mut QueryPlotData>,
    eql_context: Res<'w, EqlContext>,
    add_component_state: Local<'s, HashMap<Entity, AddComponentState>>,
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
            metadata_store,
            schema_store,
            mut graph_states,
            mut query_plots,
            eql_context,
            mut add_component_state,
        } = state_mut;

        let graph_label_margin = egui::Margin::same(0).top(10.0).bottom(14.0);
        let Ok(mut graph_state) = graph_states.get_mut(graph_id) else {
            return;
        };
        let graph_state = &mut *graph_state;
        let query_plot = query_plots.get_mut(graph_id);

        label::editable_label_with_buttons(
            ui,
            [],
            &mut graph_state.label,
            get_scheme().text_primary,
            graph_label_margin,
        );

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
            query_plot.data.name = graph_state.label.clone();
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
                            && let Ok(sql) = eql_context.0.sql(&query_plot.data.query)
                        {
                            query_plot.data.query = sql;
                        }
                    });
                    ui.separator();
                    ui.label(egui::RichText::new("QUERY").color(get_scheme().text_secondary));
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
                    let scheme_color = get_scheme().highlight;
                    let mut auto_color = query_plot.auto_color;
                    if ui.checkbox(&mut auto_color, "Use scheme color").changed() {
                        query_plot.auto_color = auto_color;
                        if auto_color {
                            query_plot.data.color =
                                impeller2_wkt::Color::from_color32(scheme_color);
                        }
                    }
                    let color_id = ui.auto_id_with("color");
                    let btn_resp = ui.add(EButton::new("Set Color"));

                    if btn_resp.clicked() {
                        egui::Popup::toggle_id(ui.ctx(), color_id);
                    }

                    let prev_color = query_plot.data.color.into_color32();
                    let mut color = prev_color;
                    if color_popup(ui, &mut color, color_id, &btn_resp).is_some()
                        && color != prev_color
                    {
                        query_plot.data.color = impeller2_wkt::Color::from_color32(color);
                        query_plot.auto_color = false;
                    }
                });
        } else {
            egui::Frame::NONE
                .inner_margin(egui::Margin::symmetric(0, 8))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("EXISTING COMPONENTS")
                            .color(get_scheme().text_secondary),
                    );
                    ui.add_space(8.0);
                    if graph_state.components.is_empty() {
                        ui.label(
                            egui::RichText::new("No component selected yet.")
                                .color(get_scheme().text_tertiary),
                        );
                    } else {
                        let mut remove_list: SmallVec<[ComponentPath; 1]> = SmallVec::new();
                        let mut first_component = true;
                        for (path, component) in graph_state.components.iter_mut() {
                            let Some(metadata) = metadata_store.get(&path.id) else {
                                continue;
                            };
                            if !first_component {
                                ui.add_space(8.0);
                                ui.separator();
                                ui.add_space(8.0);
                            }
                            first_component = false;

                            let component_label = metadata.name.clone();
                            let element_names = metadata.element_names();

                            let component_label_margin = egui::Margin::symmetric(0, 8);
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
                });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);

            add_component_widget(
                ui,
                icons.search,
                graph_state,
                &metadata_store,
                &schema_store,
                &eql_context.0,
                add_component_state.entry(graph_id).or_default(),
            );
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
    let row_count = component_values.len();
    ui.vertical(|ui| {
        for (index, ((enabled, color), element_name)) in
            component_values.iter_mut().zip(element_names).enumerate()
        {
            let label = element_name
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("[{index}]"));
            ui.horizontal(|ui| {
                ui.checkbox(enabled, "");

                let color_id = ui.auto_id_with(("component_color", label, index));
                let color_btn = ui.add(EColorButton::new(*color));
                if color_btn.clicked() {
                    egui::Popup::toggle_id(ui.ctx(), color_id);
                }
                color_popup(ui, color, color_id, &color_btn);
                ui.add_space(4.0);

                ui.style_mut().override_font_id =
                    Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
                ui.label(
                    element_name
                        .map(|name| name.to_string())
                        .unwrap_or_else(|| format!("[{index}]")),
                );
                ui.style_mut().override_font_id = None;
            });
            if index + 1 < row_count {
                ui.add_space(2.0);
            }
        }
    });
}

fn add_component_widget(
    ui: &mut egui::Ui,
    search_icon: egui::TextureId,
    graph_state: &mut GraphState,
    metadata_store: &ComponentMetadataRegistry,
    schema_store: &ComponentSchemaRegistry,
    eql_context: &eql::Context,
    add_state: &mut AddComponentState,
) {
    let mut component_names = Vec::new();
    collect_component_names(&eql_context.component_parts, &mut component_names);
    component_names.sort();
    component_names.dedup();

    let matcher = SkimMatcherV2::default().smart_case().use_cache(true);
    let filter = add_state.filter.trim();
    let mut matched_components = component_names
        .iter()
        .filter_map(|name| {
            if filter.is_empty() {
                Some((0, name.as_str()))
            } else {
                matcher
                    .fuzzy_match(name, filter)
                    .map(|score| (score, name.as_str()))
            }
        })
        .collect::<Vec<_>>();
    matched_components.sort_by(|(score_a, name_a), (score_b, name_b)| {
        score_b.cmp(score_a).then_with(|| name_a.cmp(name_b))
    });

    const MAX_SEARCH_RESULTS: usize = 3;

    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(0, 8))
        .show(ui, |ui| {
            ui.label(egui::RichText::new("ADD COMPONENT").color(get_scheme().text_secondary));
            ui.add_space(8.0);
            ui.label(egui::RichText::new("SEARCH").color(get_scheme().text_secondary));
            search(ui, &mut add_state.filter, search_icon);
            ui.add_space(8.0);

            if matched_components.is_empty() {
                ui.label(
                    egui::RichText::new("No component matches this search.")
                        .color(get_scheme().text_tertiary),
                );
            } else {
                let shown_results = matched_components.len().min(MAX_SEARCH_RESULTS);
                for (index, (_, component_name)) in matched_components
                    .iter()
                    .take(MAX_SEARCH_RESULTS)
                    .enumerate()
                {
                    ui.label((*component_name).to_string());
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                            if ui.add(EButton::highlight("ADD").width(88.0)).clicked() {
                                let _ = add_components_from_eql(
                                    graph_state,
                                    metadata_store,
                                    schema_store,
                                    eql_context,
                                    component_name,
                                );
                            }
                        });
                    });
                    if index + 1 < shown_results {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                    }
                }

                if matched_components.len() > MAX_SEARCH_RESULTS {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "Showing first {MAX_SEARCH_RESULTS} of {} results.",
                            matched_components.len()
                        ))
                        .color(get_scheme().text_tertiary),
                    );
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("ADD BY EQL").color(get_scheme().text_secondary));

            configure_input_with_border(ui.style_mut());
            let query_res = ui.add(inspector_text_field(
                &mut add_state.expression,
                "Advanced expression (e.g. drone.world_pos, drone.vel[2])",
            ));
            eql_autocomplete(ui, eql_context, &query_res, &mut add_state.expression);
            let enter_pressed =
                query_res.has_focus() && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));

            ui.add_space(8.0);
            let mut add_expression_clicked = false;
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                    add_expression_clicked =
                        ui.add(EButton::highlight("ADD").width(88.0)).clicked();
                });
            });

            if add_expression_clicked || enter_pressed {
                let query = add_state.expression.trim().to_string();
                if !query.is_empty() {
                    if add_components_from_eql(
                        graph_state,
                        metadata_store,
                        schema_store,
                        eql_context,
                        &query,
                    )
                    .unwrap_or(false)
                    {
                        add_state.expression.clear();
                    }
                }
            }
        });
}

fn collect_component_names(
    parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
    out: &mut Vec<String>,
) {
    for part in parts.values() {
        if let Some(component) = &part.component {
            out.push(component.name.clone());
        }
        collect_component_names(&part.children, out);
    }
}

fn add_components_from_eql(
    graph_state: &mut GraphState,
    metadata_store: &ComponentMetadataRegistry,
    schema_store: &ComponentSchemaRegistry,
    eql_context: &eql::Context,
    query: &str,
) -> Result<bool, String> {
    let expr = eql_context
        .parse_str(query)
        .map_err(|err| format!("Invalid EQL expression: {err}"))?;

    let mut requested_components = expr.to_graph_components();
    requested_components.sort();
    requested_components.dedup();

    if requested_components.is_empty() {
        return Err("The expression does not reference any plottable component.".to_string());
    }

    let mut requested_by_path: BTreeMap<ComponentPath, BTreeSet<usize>> = BTreeMap::new();
    for (path, index) in requested_components {
        requested_by_path.entry(path).or_default().insert(index);
    }

    let mut added_lines = 0;
    let mut next_color_index = count_enabled_lines(graph_state);

    for (path, indexes) in requested_by_path {
        let max_index = indexes.iter().copied().max().unwrap_or(0);
        if !graph_state.components.contains_key(&path) {
            let len = component_len(graph_state, metadata_store, schema_store, &path, max_index);
            graph_state.insert_component(path.clone(), default_component_values(&path, len));
        }

        let Some(component_values) = graph_state.components.get_mut(&path) else {
            continue;
        };

        if component_values.len() <= max_index {
            let current_len = component_values.len();
            component_values.extend(
                (current_len..=max_index)
                    .map(|i| (false, get_color_by_index_all(path.id.0 as usize + i))),
            );
        }

        for index in indexes {
            let Some((enabled, color)) = component_values.get_mut(index) else {
                continue;
            };
            if !*enabled {
                *enabled = true;
                *color = get_color_by_index_all(next_color_index);
                next_color_index += 1;
                added_lines += 1;
            }
        }
    }

    Ok(added_lines > 0)
}

fn count_enabled_lines(graph_state: &GraphState) -> usize {
    graph_state
        .components
        .values()
        .flat_map(|component| component.iter())
        .filter(|(enabled, _)| *enabled)
        .count()
}

fn component_len(
    graph_state: &GraphState,
    metadata_store: &ComponentMetadataRegistry,
    schema_store: &ComponentSchemaRegistry,
    path: &ComponentPath,
    required_index: usize,
) -> usize {
    if let Some(values) = graph_state.components.get(path) {
        return values.len().max(required_index + 1);
    }

    if let Some(schema) = schema_store.0.get(&path.id) {
        let len = schema.shape().iter().copied().product::<usize>();
        if len > 0 {
            return len.max(required_index + 1);
        }
    }

    if let Some(metadata) = metadata_store.get(&path.id) {
        let named_len = metadata
            .element_names()
            .split(',')
            .filter(|name| !name.trim().is_empty())
            .count();
        if named_len > 0 {
            return named_len.max(required_index + 1);
        }
    }

    required_index + 1
}

fn default_component_values(path: &ComponentPath, len: usize) -> Vec<(bool, Color32)> {
    (0..len)
        .map(|i| (false, get_color_by_index_all(path.id.0 as usize + i)))
        .collect()
}
