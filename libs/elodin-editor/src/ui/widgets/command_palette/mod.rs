use bevy::{
    ecs::{
        system::{Local, Res, ResMut, Resource, SystemParam, SystemState},
        world::World,
    },
    input::{keyboard::KeyCode, ButtonInput},
};
use bevy_egui::EguiContexts;
use egui::{epaint::Shadow, Margin, Modifiers};

use crate::ui::{
    colors::{self, with_opacity},
    images, theme,
    utils::{MarginSides, Shrink4},
};

use self::palette_items::{MatchedPaletteItem, PaletteEvent, PaletteIcon, PalettePage};

use super::{RootWidgetSystem, RootWidgetSystemExt, WidgetSystem, WidgetSystemExt};

mod palette_items;

#[derive(Resource, Default)]
pub struct CommandPaletteState {
    pub filter: String,
    pub show: bool,
    pub input_focus: bool,
    pub page_stack: Vec<PalettePage>,
    pub selected_index: usize,
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
        let cmd_pressed = if cfg!(target_os = "macos") {
            kbd.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight])
        } else {
            kbd.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight])
        };
        if cmd_pressed && kbd.just_pressed(KeyCode::KeyP) {
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

#[derive(Clone)]
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
            command_palette_state.page_stack.clear();
            return false;
        }

        let screen_rect = ctx.screen_rect();
        let palette_width = (screen_rect.width() / 2.0).clamp(500.0, 900.0);
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
                fill: colors::PRIMARY_ONYX,
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

                let (up, down) = ui.add_widget::<PaletteSearch>(world, "command_palette_search");

                ui.separator();

                ui.add_widget_with::<PaletteItems>(
                    world,
                    "command_palette_items",
                    (command_palette_icons, up, down),
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
    type Output = (bool, bool);

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
                ui.horizontal(|ui| {
                    let style = ui.style_mut();
                    style.visuals.selection.stroke = egui::Stroke::NONE;
                    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::NONE;
                    style.visuals.extreme_bg_color = colors::TRANSPARENT;
                    let len = command_palette_state.page_stack.len().saturating_sub(1);
                    for page in command_palette_state.page_stack[..len].iter() {
                        ui.style_mut().interaction.selectable_labels = false;
                        if let Some(label) = &page.label {
                            egui::Frame::none()
                                .fill(colors::BONE_DEFAULT)
                                .inner_margin(Margin::symmetric(8.0, 1.0))
                                .outer_margin(Margin::ZERO.right(6.0))
                                .rounding(3.0)
                                .show(ui, |ui| {
                                    let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                                    font_id.size = 11.0;
                                    ui.label(
                                        egui::RichText::new(label)
                                            .font(font_id)
                                            .color(colors::PRIMARY_ONYX),
                                    );
                                });
                        } else {
                            let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                            font_id.size = 16.0;
                            egui::Frame::none()
                                .fill(colors::BONE_DEFAULT)
                                .outer_margin(Margin::ZERO.right(6.0))
                                .inner_margin(Margin::symmetric(5.5, 0.0))
                                .rounding(3.0)
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new("←")
                                            .font(font_id)
                                            .color(colors::PRIMARY_ONYX),
                                    );
                                });
                        }
                    }

                    let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                    font_id.size = 13.0;

                    let mut popped_page = false;
                    if command_palette_state.filter.is_empty() {
                        ui.ctx().input(|i| {
                            if i.key_pressed(egui::Key::Backspace) {
                                popped_page = true;
                                command_palette_state.page_stack.pop();
                                command_palette_state.selected_index = 0;
                            }
                        })
                    }

                    let (up_pressed, down_pressed) = ui.ctx().input_mut(|i| {
                        (
                            i.consume_key(Modifiers::NONE, egui::Key::ArrowUp),
                            i.consume_key(Modifiers::NONE, egui::Key::ArrowDown),
                        )
                    });
                    let search_bar = ui.add(
                        egui::TextEdit::singleline(&mut command_palette_state.filter)
                            .hint_text("Type a command...")
                            .font(font_id)
                            .return_key(egui::KeyboardShortcut::new(
                                egui::Modifiers::NONE,
                                egui::Key::Escape,
                            ))
                            .desired_width(ui.available_width()),
                    );

                    if search_bar.changed() {
                        command_palette_state.selected_index = 0;
                    }

                    if command_palette_state.input_focus {
                        search_bar.request_focus();
                        command_palette_state.input_focus = false;
                    }

                    command_palette_state.input_focus |= popped_page;
                    has_focus = search_bar.has_focus();
                    (up_pressed, down_pressed)
                })
                .inner
            })
            .inner
    }
}

#[derive(SystemParam)]
pub struct PaletteItems<'w> {
    command_palette_state: ResMut<'w, CommandPaletteState>,
    kbd: Res<'w, ButtonInput<KeyCode>>,
}

impl WidgetSystem for PaletteItems<'_> {
    type Args = (CommandPaletteIcons, bool, bool);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let (icons, up_pressed, down_pressed) = args;
        let state_mut = state.get_mut(world);
        let mut command_palette_state = state_mut.command_palette_state;
        let kbd = state_mut.kbd.clone();
        let mut selected_index = command_palette_state.selected_index;
        let hit_enter = kbd.just_pressed(KeyCode::Enter);
        let filter = command_palette_state.filter.clone();

        if command_palette_state.page_stack.is_empty() {
            command_palette_state
                .page_stack
                .push(PalettePage::default());
        }
        let mut page = command_palette_state.page_stack.pop().unwrap();
        page.initialize(world);
        let mut palette_items_filtered = page.filter(&filter);
        let row_margin = egui::Margin::symmetric(16.0, 12.0);
        let row_height = ui.spacing().interact_size.y + row_margin.sum().y;
        let max_visible_rows = palette_items_filtered.len().clamp(0, 10);
        if down_pressed {
            selected_index =
                (selected_index + 1).min(palette_items_filtered.len().saturating_sub(1));
        }
        if up_pressed {
            selected_index = selected_index.saturating_sub(1);
        }

        if max_visible_rows > 0 {
            let res = egui::ScrollArea::vertical()
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show_rows(ui, row_height, max_visible_rows, |ui, row_range| {
                    let mut current_heading: Option<String> = None;
                    for (
                        i,
                        MatchedPaletteItem {
                            item,
                            match_indices,
                            ..
                        },
                    ) in palette_items_filtered.drain(row_range).enumerate()
                    {
                        if Some(&item.header) != current_heading.as_ref() {
                            egui::Frame::none().inner_margin(row_margin).show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(item.header.clone())
                                        .monospace()
                                        .color(colors::PRIMARY_CREAME_6),
                                );
                            });
                            current_heading = Some(item.header.clone());
                        }
                        let btn = ui.add({
                            let widget = PaletteItemWidget::new(
                                item.label.clone(),
                                match_indices,
                                i == selected_index,
                            )
                            .margin(row_margin);
                            match item.icon {
                                PaletteIcon::Link => widget.icon(icons.link),
                                PaletteIcon::None => widget,
                            }
                        });

                        if btn.clicked() || (i == selected_index && hit_enter) {
                            return Some(item.system.run((), world));
                        }
                    }
                    None
                });
            match res.inner {
                Some(PaletteEvent::Exit) => {
                    let mut state_mut = state.get_mut(world);
                    state_mut.command_palette_state.show = false;
                    state_mut.command_palette_state.selected_index = 0;
                }
                Some(PaletteEvent::NextPage {
                    next_page,
                    prev_page_label,
                }) => {
                    let mut state_mut = state.get_mut(world);
                    state_mut.command_palette_state.filter = "".to_string();
                    if let Some(prev_page_label) = prev_page_label {
                        page.label = Some(prev_page_label);
                    }
                    state_mut.command_palette_state.page_stack.push(page);
                    state_mut.command_palette_state.page_stack.push(next_page);
                    state_mut.command_palette_state.input_focus = true;
                    state_mut.command_palette_state.selected_index = 0;
                }
                None => {
                    let mut state_mut = state.get_mut(world);
                    state_mut.command_palette_state.page_stack.push(page);
                    state_mut.command_palette_state.selected_index = selected_index;
                }
            }
        } else {
            let mut state_mut = state.get_mut(world);
            state_mut.command_palette_state.page_stack.push(page);
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
pub struct PaletteItemWidget {
    label: String,
    matched_char_indices: Vec<usize>,
    icon: Option<egui::TextureId>,
    shortcut_label: Option<String>,

    margin: egui::Margin,

    inactive_bg_color: egui::Color32,
    active_bg_color: egui::Color32,
    hovered_bg_color: egui::Color32,

    text_color: egui::Color32,

    active: bool,
}

impl PaletteItemWidget {
    pub fn new(label: impl ToString, matched_char_indices: Vec<usize>, active: bool) -> Self {
        Self {
            label: label.to_string(),
            matched_char_indices,
            icon: None,
            shortcut_label: None,

            margin: egui::Margin::symmetric(16.0, 8.0),

            inactive_bg_color: colors::PRIMARY_ONYX,
            active_bg_color: with_opacity(colors::PRIMARY_CREAME, 0.05),
            hovered_bg_color: with_opacity(colors::PRIMARY_CREAME, 0.1),

            text_color: colors::PRIMARY_CREAME,
            active,
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
            style.visuals.widgets.inactive.bg_fill = if self.active {
                self.active_bg_color
            } else {
                self.inactive_bg_color
            };
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

impl egui::Widget for PaletteItemWidget {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        self.render(ui)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}
