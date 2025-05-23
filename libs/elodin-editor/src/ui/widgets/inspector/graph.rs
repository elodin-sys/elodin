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
    colors::{self, get_scheme},
    theme,
    utils::MarginSides,
    widgets::{
        WidgetSystem,
        button::{ECheckboxButton, EColorButton},
        label::{self, label_with_buttons},
        plot::GraphState,
    },
};

use super::InspectorIcons;

#[derive(SystemParam)]
pub struct InspectorGraph<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    graph_states: Query<'w, 's, &'static mut GraphState>,
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
        } = state_mut;

        let graph_label_margin = egui::Margin::same(0).top(10.0).bottom(14.0);
        let Ok(mut graph_state) = graph_states.get_mut(graph_id) else {
            return;
        };

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

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Y Bounds").color(get_scheme().text_secondary));
                    ui.add_space(8.0);
                    theme::configure_input_with_border(ui.style_mut());
                    ui.checkbox(&mut graph_state.auto_y_range, "Auto Bounds?");
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
                            !graph_state.auto_y_range,
                            egui::DragValue::new(&mut graph_state.y_range.start).speed(0.01),
                        );
                    });
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Max").color(get_scheme().text_secondary));
                        ui.add_space(16.0);
                        ui.add_enabled(
                            !graph_state.auto_y_range,
                            egui::DragValue::new(&mut graph_state.y_range.end).speed(0.01),
                        );
                    })
                })
            });
        ui.separator();

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
