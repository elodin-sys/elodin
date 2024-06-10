use bevy::{
    ecs::{
        system::{Local, Res, ResMut, Resource, SystemParam, SystemState},
        world::World,
    },
    input::{keyboard::KeyCode, ButtonInput},
};
use bevy_egui::EguiContexts;
use egui::epaint::Shadow;
use nalgebra::clamp;

use crate::ui::{
    colors::{self, with_opacity},
    images, theme,
    utils::Shrink4,
};

use self::palette_items::{palette_help_items, palette_viewport_items, PaletteItemWrapper};

use super::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystem, WidgetSystemExt};

mod palette_items;

#[derive(Resource, Default)]
pub struct CommandPaletteState {
    pub filter: String,
    pub show: bool,
    pub input_focus: bool,
}

#[derive(SystemParam)]
pub struct CommandPalette<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    kbd: Res<'w, ButtonInput<KeyCode>>,
}

impl RootWidgetSystem for CommandPalette<'_> {
    type Args = ();
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        _ctx: &mut egui::Context,
        _args: Self::Args,
    ) {
        let clicked_elsewhere = world.add_root_widget::<PaletteWindow>("command_palette_window");

        let state_mut = state.get_mut(world);

        let mut command_palette_state = state_mut.command_palette_state;
        let kbd = state_mut.kbd;

        if kbd.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight])
            && kbd.just_pressed(KeyCode::KeyP)
        {
            command_palette_state.show = !command_palette_state.show;
            if command_palette_state.show {
                command_palette_state.input_focus = true;
            }
        }

        if kbd.just_pressed(KeyCode::Escape) || clicked_elsewhere {
            command_palette_state.show = false;
        }

        if !command_palette_state.show {
            command_palette_state.filter = "".to_string();
        }
    }
}

pub struct CommandPaletteIcons {
    pub link: egui::TextureId,
    // TODO: cmd_btn
}

#[derive(SystemParam)]
pub struct PaletteWindow<'w, 's> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    contexts: EguiContexts<'w, 's>,
    images: Local<'s, images::Images>,
}

impl RootWidgetSystem for PaletteWindow<'_, '_> {
    type Args = ();
    type Output = bool;

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        _args: Self::Args,
    ) -> Self::Output {
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let mut contexts = state_mut.contexts;
        let images = state_mut.images;

        let command_palette_icons = CommandPaletteIcons {
            link: contexts.add_image(images.icon_link.clone_weak()),
        };

        if !command_palette_state.show {
            command_palette_state.filter = "".to_string();
            return false;
        }

        let screen_rect = ctx.screen_rect();
        let palette_width = clamp(screen_rect.width() / 3.0, 400.0, 600.0);
        let palette_size = egui::vec2(palette_width, 800.0);
        let palette_min = egui::pos2(
            screen_rect.center().x - palette_width / 2.0,
            screen_rect.height() * 0.2,
        );

        let cmd_window = egui::Window::new("command_palette")
            .title_bar(false)
            .resizable(false)
            .fixed_size(palette_size)
            .fixed_pos(palette_min)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, with_opacity(colors::PRIMARY_CREAME, 0.005)),
                rounding: theme::rounding_xs(),
                shadow: Shadow {
                    color: colors::PRIMARY_SMOKE,
                    blur: 16.0,
                    offset: egui::vec2(3.0, 3.0),
                    spread: 3.0,
                },
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.style_mut().spacing.item_spacing = egui::vec2(0.0, 0.0);

                let search_has_focus =
                    ui.add_widget::<PaletteSearch>(world, "command_palette_search");

                ui.separator();

                ui.add_widget_with::<PaletteItems>(
                    world,
                    "command_palette_items",
                    (command_palette_icons, search_has_focus),
                );
            });

        let Some(cmd_window) = cmd_window else {
            return false;
        };

        cmd_window.response.clicked_elsewhere()
    }
}

#[derive(SystemParam)]
pub struct PaletteSearch<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
}

impl WidgetSystem for PaletteSearch<'_> {
    type Args = ();
    type Output = bool;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        _args: Self::Args,
    ) -> Self::Output {
        let state_mut = state.get_mut(world);

        let mut command_palette_state = state_mut.command_palette_state;

        let mut has_focus = false;

        egui::Frame::none()
            .inner_margin(egui::Margin::same(16.0))
            .show(ui, |ui| {
                let style = ui.style_mut();
                style.visuals.selection.stroke = egui::Stroke::NONE;
                style.visuals.widgets.hovered.bg_stroke = egui::Stroke::NONE;
                style.visuals.extreme_bg_color = colors::TRANSPARENT;

                let search_bar = ui.add(
                    egui::TextEdit::singleline(&mut command_palette_state.filter)
                        .hint_text("Search...")
                        .return_key(egui::KeyboardShortcut::new(
                            egui::Modifiers::NONE,
                            egui::Key::Escape,
                        ))
                        .desired_width(ui.available_width()),
                );

                if command_palette_state.input_focus {
                    search_bar.request_focus();
                    command_palette_state.input_focus = false;
                }

                has_focus = search_bar.has_focus();
            });

        has_focus
    }
}

#[derive(SystemParam)]
pub struct PaletteItems<'w> {
    command_palette_state: Res<'w, CommandPaletteState>,
    kbd: Res<'w, ButtonInput<KeyCode>>,
}

impl WidgetSystem for PaletteItems<'_> {
    type Args = (CommandPaletteIcons, bool);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let (icons, search_has_focus) = args;
        let state_mut = state.get_mut(world);
        let command_palette_state = state_mut.command_palette_state;
        let kbd = state_mut.kbd;

        let filter = command_palette_state.filter.clone();

        let palette_items_filtered = [palette_help_items(&filter), palette_viewport_items(&filter)]
            .into_iter()
            .flatten()
            .collect::<Vec<PaletteItemWrapper>>();

        let row_margin = egui::Margin::symmetric(16.0, 12.0);
        let row_height = ui.spacing().interact_size.y + row_margin.sum().y;
        let max_visible_rows = clamp(palette_items_filtered.len(), 0, 10);

        let mut request_focus = kbd.just_pressed(KeyCode::ArrowDown) && search_has_focus;
        let mut use_first_item = kbd.just_pressed(KeyCode::Enter) && search_has_focus;

        if max_visible_rows > 0 {
            egui::ScrollArea::vertical()
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show_rows(ui, row_height, max_visible_rows, |ui, row_range| {
                    for palette_item in palette_items_filtered[row_range].iter() {
                        let current_request_focus = !palette_item.group_label && request_focus;
                        let current_use_first_item = !palette_item.group_label && use_first_item;

                        (palette_item.widget)(
                            ui,
                            world,
                            (current_request_focus, current_use_first_item),
                            palette_item.match_indices.clone(),
                            &icons,
                            row_margin,
                        );

                        if current_request_focus {
                            request_focus = false;
                        }

                        if current_use_first_item {
                            use_first_item = false;
                        }
                    }
                });
        } else {
            egui::Frame::none().inner_margin(row_margin).show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Couldn't find anything...")
                        .color(colors::PRIMARY_CREAME_6),
                );
            });
        }
    }
}

#[must_use = "You should put this widget in an ui with `ui.add(widget);`"]
pub struct PaletteItem {
    label: String,
    matched_char_indices: Vec<usize>,
    icon: Option<egui::TextureId>,
    shortcut_label: Option<String>,

    margin: egui::Margin,

    inactive_bg_color: egui::Color32,
    active_bg_color: egui::Color32,
    hovered_bg_color: egui::Color32,

    text_color: egui::Color32,
}

impl PaletteItem {
    pub fn new(label: impl ToString, matched_char_indices: Vec<usize>) -> Self {
        Self {
            label: label.to_string(),
            matched_char_indices,
            icon: None,
            shortcut_label: None,

            margin: egui::Margin::symmetric(16.0, 8.0),

            inactive_bg_color: colors::PRIMARY_SMOKE,
            active_bg_color: colors::PRIMARY_ONYX,
            hovered_bg_color: with_opacity(colors::PRIMARY_CREAME, 0.1),

            text_color: colors::PRIMARY_CREAME,
        }
    }

    pub fn shortcut_label(mut self, shortcut_label: impl ToString) -> Self {
        self.shortcut_label = Some(shortcut_label.to_string());
        self
    }

    pub fn icon(mut self, icon: egui::TextureId) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    fn label_with_matched_chars(&self, font_id: egui::FontId) -> egui::text::LayoutJob {
        let mut layout_job = egui::text::LayoutJob::default();
        let text_format_default = egui::TextFormat::simple(font_id, self.text_color);
        let text_format_highlighted = egui::TextFormat {
            background: colors::YOLK_40,
            ..text_format_default.clone()
        };
        let mut peekable = self.matched_char_indices.iter().peekable();

        for (i, char) in self.label.char_indices() {
            let next_id = **peekable.peek().unwrap_or(&&self.label.len());

            if next_id == i {
                layout_job.append(&char.to_string(), 0.0, text_format_highlighted.clone());
                peekable.next();
            } else {
                layout_job.append(&char.to_string(), 0.0, text_format_default.clone());
            }
        }

        layout_job
    }

    fn render(&mut self, ui: &mut egui::Ui) -> egui::Response {
        // Set widget size and allocate space
        let desired_size = egui::vec2(
            ui.available_width(),
            ui.spacing().interact_size.y + self.margin.sum().y,
        );
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        // Paint the UI
        if ui.is_rect_visible(rect) {
            let style = ui.style_mut();
            style.visuals.widgets.inactive.bg_fill = self.inactive_bg_color;
            style.visuals.widgets.active.bg_fill = self.active_bg_color;
            style.visuals.widgets.hovered.bg_fill = self.hovered_bg_color;

            let font_id = egui::TextStyle::Button.resolve(ui.style());
            let visuals = ui.style().interact(&response);

            // Background

            ui.painter()
                .rect_filled(rect, egui::Rounding::ZERO, visuals.bg_fill);

            // Label

            let inner_rect = rect.shrink4(self.margin);

            if let Some(image_id) = self.icon {
                let default_uv =
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));

                let image_rect = egui::Rect::from_min_size(
                    inner_rect.min,
                    egui::vec2(ui.spacing().interact_size.y, ui.spacing().interact_size.y),
                )
                .shrink(2.0);

                ui.painter()
                    .image(image_id, image_rect, default_uv, self.text_color);

                let layout_job = self.label_with_matched_chars(font_id.clone());
                let galley = ui.fonts(|f| f.layout_job(layout_job));
                ui.painter().galley(
                    egui::pos2(image_rect.right_top().x + 8.0, image_rect.right_top().y),
                    galley,
                    self.text_color,
                );
            } else {
                let layout_job = self.label_with_matched_chars(font_id.clone());
                let galley = ui.fonts(|f| f.layout_job(layout_job));
                ui.painter()
                    .galley(inner_rect.left_top(), galley, self.text_color);
            }

            if let Some(shortcut_label) = &self.shortcut_label {
                ui.painter().text(
                    inner_rect.right_center(),
                    egui::Align2::RIGHT_CENTER,
                    shortcut_label.to_owned(),
                    font_id,
                    self.text_color,
                );
            }
        }

        response
    }
}

impl egui::Widget for PaletteItem {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}
