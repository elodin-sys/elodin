use bevy::ecs::{
    system::{Query, Res, ResMut, SystemParam, SystemState},
    world::World,
};
use conduit::bevy::MaxTick;

use crate::ui::{
    colors::{self, with_opacity},
    theme::rounding_xxs,
    utils,
    widgets::{button::EImageButton, plot::GraphState, WidgetSystem},
    SelectedObject, ViewportRange,
};

use super::{
    timeline_ranges::{TimelineRanges, TimelineRangesFocused},
    TimelineIcons,
};

#[derive(SystemParam)]
pub struct TimelineRangeListHeader<'w> {
    timeline_ranges: ResMut<'w, TimelineRanges>,
    max_tick: Res<'w, MaxTick>,
}

impl WidgetSystem for TimelineRangeListHeader<'_> {
    type Args = (TimelineIcons, egui::Vec2);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let (icons, desired_size) = args;

        let state_mut = state.get_mut(world);
        let mut timeline_ranges = state_mut.timeline_ranges;
        let max_tick = state_mut.max_tick;

        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

        ui.painter().rect(
            rect,
            egui::Rounding::ZERO,
            colors::PRIMARY_ONYX,
            ui.style().visuals.widgets.noninteractive.bg_stroke,
        );

        let inner_rect = rect.shrink2(egui::vec2(12.0, 6.0));

        // Paint the UI
        if ui.is_rect_visible(inner_rect) {
            // NOTE: scope is necessary to prevent button allocation bleeding out into other widgets
            ui.scope(|ui| {
                let item_spacing = 20.0;
                let button_size_scale = 1.2;
                let button_side = ui.spacing().interact_size.y * button_size_scale;

                let label_color = with_opacity(colors::PRIMARY_CREAME, 0.2);

                // Loop Icon

                let default_uv =
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                let image_rect = egui::Rect::from_center_size(
                    egui::pos2(
                        inner_rect.left() + (button_side / 2.0),
                        inner_rect.center().y,
                    ),
                    egui::vec2(button_side, button_side),
                )
                .shrink(3.0);

                ui.painter()
                    .image(icons.range_loop, image_rect, default_uv, label_color);

                ui.painter().vline(
                    inner_rect.left() + button_side + (item_spacing / 2.0),
                    inner_rect.y_range(),
                    ui.style().visuals.widgets.noninteractive.bg_stroke,
                );

                // Label

                let font_id = egui::TextStyle::Button.resolve(ui.style());
                let layout_job = utils::get_galley_layout_job(
                    String::from("RANGES"),
                    inner_rect.width() - ((button_side + item_spacing) * 2.0),
                    font_id,
                    label_color,
                );
                let galley = ui.fonts(|f| f.layout_job(layout_job));

                let text_rect_raw = inner_rect.shrink2(egui::vec2(button_side + item_spacing, 0.0));
                let text_rect =
                    egui::Align2::LEFT_CENTER.align_size_within_rect(galley.size(), text_rect_raw);
                ui.painter().galley(text_rect.min, galley, label_color);

                // Button

                ui.allocate_ui_at_rect(inner_rect, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let btn = ui.add(
                            EImageButton::new(icons.add)
                                .scale(button_size_scale, button_size_scale)
                                .image_tint(colors::PRIMARY_CREAME)
                                .bg_color(colors::TRANSPARENT),
                        );

                        if btn.clicked() {
                            timeline_ranges.create_range(max_tick.0);
                        }
                    });
                });
            });
        }
    }
}

#[derive(SystemParam)]
pub struct TimelineRangeList<'w, 's> {
    timeline_ranges: ResMut<'w, TimelineRanges>,
    timeline_ranges_focused: ResMut<'w, TimelineRangesFocused>,
    viewport_range: ResMut<'w, ViewportRange>,
    selected_object: Res<'w, SelectedObject>,
    graph_states: Query<'w, 's, &'static GraphState>,
}

impl WidgetSystem for TimelineRangeList<'_, '_> {
    type Args = (TimelineIcons, egui::Vec2);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let (icons, desired_size) = args;

        let state_mut = state.get_mut(world);
        let mut timeline_ranges = state_mut.timeline_ranges;
        let mut timeline_ranges_focused = state_mut.timeline_ranges_focused;
        let mut viewport_range = state_mut.viewport_range;
        let graph_states = state_mut.graph_states;

        let selected_graph_range_id = match state_mut.selected_object.to_owned() {
            SelectedObject::Graph { graph_id, .. } => {
                if let Ok(graph_state) = graph_states.get(graph_id) {
                    graph_state.range_id
                } else {
                    None
                }
            }
            _ => None,
        };

        let mut has_focus = false;

        let item_spacing = 16.0;
        let button_size_scale = 1.2;
        let button_side = ui.spacing().interact_size.y * button_size_scale;

        timeline_ranges.0.retain(|range_id, range| {
            let mut retain = true;

            let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

            let hovered = if let Some(pointer_pos) = ui.input(|i| i.pointer.latest_pos()) {
                rect.contains(pointer_pos)
            } else {
                false
            };

            let loop_is_on = viewport_range.0 == Some(*range_id);

            let (bg_color, bg_hover_color, text_color, btn_bg_color, btn_bg_hover_color) =
                if selected_graph_range_id.is_some_and(|rid| rid == *range_id) {
                    (
                        colors::PRIMARY_CREAME,
                        colors::PRIMARY_CREAME_9,
                        colors::PRIMARY_ONYX,
                        colors::PRIMARY_CREAME_8,
                        colors::PRIMARY_CREAME_6,
                    )
                } else {
                    (
                        colors::TRANSPARENT,
                        colors::PRIMARY_ONYX,
                        colors::PRIMARY_CREAME,
                        colors::BLACK_BLACK_600,
                        colors::PRIMARY_ONYX_9,
                    )
                };

            ui.painter().rect(
                rect,
                egui::Rounding::ZERO,
                if hovered { bg_hover_color } else { bg_color },
                ui.style().visuals.widgets.noninteractive.bg_stroke,
            );

            let inner_rect = rect.shrink2(egui::vec2(12.0, 6.0));

            // Paint the UI
            if ui.is_rect_visible(inner_rect) {
                ui.scope(|ui| {
                    // Label

                    ui.allocate_ui_at_rect(inner_rect, |ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            let btn = ui.add(
                                EImageButton::new(icons.range_loop)
                                    .scale(button_size_scale, button_size_scale)
                                    .image_tint(if loop_is_on {
                                        text_color
                                    } else {
                                        colors::TRANSPARENT
                                    })
                                    .hovered_bg_color(btn_bg_hover_color)
                                    .bg_color(if hovered {
                                        btn_bg_color
                                    } else {
                                        colors::TRANSPARENT
                                    }),
                            );

                            if btn.clicked() {
                                viewport_range.0 = if loop_is_on { None } else { Some(*range_id) };
                            }

                            ui.add_space(item_spacing);

                            let style = ui.style_mut();
                            style.visuals.selection.stroke.color = text_color;
                            style.visuals.widgets.hovered.bg_stroke.color =
                                with_opacity(text_color, 0.2);
                            style.visuals.widgets.hovered.rounding = rounding_xxs();
                            style.visuals.widgets.active.rounding = rounding_xxs();
                            style.visuals.extreme_bg_color = colors::TRANSPARENT;

                            let response = ui.add(
                                egui::TextEdit::singleline(&mut range.label)
                                    .char_limit(32)
                                    .text_color(text_color)
                                    .desired_width(
                                        inner_rect.width() - ((button_side + item_spacing) * 2.0),
                                    )
                                    .hint_text("Untitled Range"),
                            );

                            if response.has_focus() {
                                has_focus = true;
                            }
                        });
                    });

                    // Button

                    ui.allocate_ui_at_rect(inner_rect, |ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let remove_btn = ui.add(
                                EImageButton::new(icons.remove)
                                    .scale(button_size_scale, button_size_scale)
                                    .image_tint(text_color)
                                    .hovered_bg_color(btn_bg_hover_color)
                                    .bg_color(colors::TRANSPARENT),
                            );

                            if remove_btn.clicked() {
                                retain = false;
                            }
                        });
                    });
                });
            }

            retain
        });

        timeline_ranges_focused.0 = has_focus;
    }
}
