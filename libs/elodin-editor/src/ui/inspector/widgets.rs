use bevy_egui::egui;
use egui::color_picker::{Alpha, color_picker_color32};
use impeller2_wkt::QueryType;

use crate::ui::{
    button::{ECheckboxButton, EColorButton},
    colors::{self, ColorExt, EColor, get_scheme},
    theme::{self, configure_combo_box},
};

pub fn query(query: &mut String, ty: QueryType) -> impl egui::Widget {
    move |ui: &mut egui::Ui| {
        ui.add(inspector_text_field(
            query,
            match ty {
                QueryType::EQL => "EQL Query (i.e a.world_pos.x)",
                QueryType::SQL => "SQL Query (i.e select * from table)",
            },
        ))
    }
}
pub fn inspector_text_field(query: &mut String, hint_text: &str) -> impl egui::Widget {
    move |ui: &mut egui::Ui| {
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
                    .lock_focus(true)
                    .hint_text(hint_text)
                    .desired_width(ui.available_width() - 16.0)
                    .margin(8.0),
            )
        })
        .inner
    }
}

pub fn eql_autocomplete(
    ui: &mut egui::Ui,
    eql_context: &eql::Context,
    query_res: &egui::Response,
    current_query: &mut String,
) {
    let tab_pressed = ui
        .ctx()
        .input_mut(|i| i.consume_key(Default::default(), egui::Key::Tab));
    let suggestions = eql_context.get_string_suggestions(current_query);
    let id = ui.next_auto_id();
    let suggestion_memory_id = ui.id().with("eql_suggestion_index");

    if tab_pressed && !suggestions.is_empty() && query_res.has_focus() {
        let selected_index = ui.memory(|mem| {
            mem.data
                .get_temp::<usize>(suggestion_memory_id)
                .unwrap_or_default()
        });
        if let Some((_, patch)) = suggestions.get(selected_index) {
            *current_query = patch.clone();
            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), query_res.id) {
                let ccursor = egui::text::CCursor::new(current_query.chars().count());
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::one(ccursor)));
                state.store(ui.ctx(), query_res.id);
                ui.memory_mut(|memory| memory.request_focus(query_res.id));
            }
            ui.memory_mut(|mem| {
                mem.data.remove::<usize>(suggestion_memory_id);
            });
            egui::Popup::close_id(ui.ctx(), id);
        }
    }

    if query_res.has_focus() {
        // Handle keyboard navigation
        let mut selected_index = ui.memory_mut(|mem| {
            *mem.data
                .get_temp::<usize>(suggestion_memory_id)
                .get_or_insert(0)
        });

        // Handle arrow key navigation
        if ui.ctx().input(|i| i.key_pressed(egui::Key::ArrowDown)) && !suggestions.is_empty() {
            selected_index = (selected_index + 1) % suggestions.len();
            ui.memory_mut(|mem| {
                mem.data.insert_temp(suggestion_memory_id, selected_index);
            });
        }
        if ui.ctx().input(|i| i.key_pressed(egui::Key::ArrowUp)) && !suggestions.is_empty() {
            selected_index = selected_index.saturating_sub(1);
            ui.memory_mut(|mem| {
                mem.data.insert_temp(suggestion_memory_id, selected_index);
            });
        }

        ui.scope(|ui| {
            configure_combo_box(ui.style_mut());
            ui.style_mut().spacing.menu_margin = egui::Margin::same(4);
            //let target_res = &query_res.clone().with_new_rect(query_res.rect.expand(28.0));
            let target_res = query_res;
            // These Popup settings were copied from the following URL after
            // popup_below_widget was deprecated:
            // https://github.com/emilk/egui/blob/af96e0373c18477b77236e2bfc89735af007b1c2/crates/egui/src/containers/old_popup.rs#L189
            egui::Popup::from_response(target_res)
                .layout(egui::Layout::top_down_justified(egui::Align::LEFT))
                .open_memory(None)
                .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
                .id(id)
                .align(egui::RectAlign::BOTTOM_START)
                .width(target_res.rect.width())
                .show(|ui| {
                    ui.set_min_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .max_height(200.)
                        .show(ui, |ui| {
                            ui.style_mut().spacing.item_spacing = egui::vec2(0.0, 8.0);
                            for (i, (suggestion, patch)) in suggestions.iter().enumerate() {
                                let is_selected = i == selected_index;
                                let response = if is_selected {
                                    ui.colored_label(get_scheme().highlight, suggestion)
                                } else {
                                    ui.label(suggestion)
                                };

                                // Handle mouse click to apply suggestion
                                if response.clicked() {
                                    *current_query = patch.clone();
                                    ui.memory_mut(|mem| {
                                        mem.data.remove::<usize>(suggestion_memory_id);
                                    });
                                    egui::Popup::close_id(ui.ctx(), id);
                                }
                            }
                        })
                });
        });
        if !suggestions.is_empty() {
            egui::Popup::open_id(ui.ctx(), id);
        } else {
            ui.memory_mut(|mem| {
                mem.data.remove::<usize>(suggestion_memory_id);
            });
            if egui::Popup::is_id_open(ui.ctx(), id) {
                egui::Popup::close_id(ui.ctx(), id);
            }
        }
    }
}

pub fn color_popup(
    ui: &mut egui::Ui,
    color: &mut egui::Color32,
    color_id: egui::Id,
    target_res: &egui::Response,
) -> Option<egui::Response> {

    let inner_response = egui::Popup::new(color_id, ui.ctx().clone(), target_res, target_res.layer_id)
        .kind(egui::PopupKind::Popup)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .width(300.0)
        .open_memory(None)
        .show(|ui| {
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
                color_picker_color32(ui, color, Alpha::OnlyBlend);
            });
        });

    inner_response.map(|ir| ir.response)
}

pub fn search(
    ui: &mut egui::Ui,
    filter: &mut String,
    search_icon: egui::TextureId,
) -> egui::Response {
    ui.vertical(|ui| {
        egui::Frame::NONE
            .corner_radius(egui::CornerRadius::same(3))
            .inner_margin(egui::Margin::same(4))
            .fill(get_scheme().bg_secondary)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::widgets::Image::new(egui::load::SizedTexture::new(
                            search_icon,
                            [ui.spacing().interact_size.y, ui.spacing().interact_size.y],
                        ))
                        .tint(get_scheme().text_secondary),
                    );

                    let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                    font_id.size = 12.0;
                    ui.add(
                        egui::TextEdit::singleline(filter)
                            .desired_width(ui.available_width())
                            .frame(false)
                            .font(font_id),
                    );
                });
            });
    })
    .response
}

pub fn node_color_picker(ui: &mut egui::Ui, label: &str, color: &mut impeller2_wkt::Color) -> bool {
    let mut egui_color = color.into_color32();
    let res = ui.add(
        ECheckboxButton::new(label, true)
            .margin(egui::Margin::symmetric(0, 8))
            .on_color(egui_color)
            .text_color(get_scheme().text_secondary)
            .left_label(true),
    );

    let color_id = ui.auto_id_with("color");
    if res.clicked() {
        egui::Popup::toggle_id(ui.ctx(), color_id);
    }
    if egui::Popup::is_id_open(ui.ctx(), color_id) {
        if let Some(popup_response) = color_popup(
            ui,
            &mut egui_color,
            color_id,
            &res,
        ) {
            if !res.clicked()
                && (ui.input(|i| i.key_pressed(egui::Key::Escape))
                    || popup_response.clicked_elsewhere())
            {
                egui::Popup::close_id(ui.ctx(), color_id);
            }
        }
    }

    let new_color = impeller2_wkt::Color::from_color32(egui_color);
    let changed = new_color != *color;
    *color = new_color;
    ui.separator();
    changed
}

pub fn eql_textfield(
    ui: &mut egui::Ui,
    enabled: bool,
    eql_ctx: &eql::Context,
    eql: &mut String,
) -> egui::Response {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let eql_res = ui.add_enabled(enabled, query(eql, impeller2_wkt::QueryType::EQL));
        eql_autocomplete(ui, eql_ctx, &eql_res, eql);
        eql_res
    })
    .inner
}
