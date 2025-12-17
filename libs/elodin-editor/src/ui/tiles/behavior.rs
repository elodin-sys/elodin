use super::*;
use bevy_egui::egui::{self, Color32, CornerRadius, RichText, Stroke, Ui, Visuals, vec2};
use egui::response::Flags;
use egui_tiles::{Behavior, Container, Tile, TileId, Tiles};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TabRole {
    Super,
    Normal,
}

#[derive(Clone)]
enum TabState {
    Selected,
    Inactive,
}

pub(super) struct TreeBehavior<'w> {
    pub icons: TileIcons,
    pub tree_actions: SmallVec<[TreeAction; 4]>,
    pub world: &'w mut World,
    pub container_titles: HashMap<TileId, String>,
    pub read_only: bool,
    pub target_window: Option<Entity>,
    pub root_id: Option<TileId>,
}

impl<'w> TreeBehavior<'w> {
    fn container_fallback_title(&mut self, tiles: &Tiles<Pane>, id: TileId) -> String {
        if let Some(title) = self.container_titles.get(&id) {
            info!(
                target: "tabs.title",
                ?id,
                root = Some(id) == self.root_id,
                source = "container_titles",
                title = %title
            );
            return title.clone();
        }
        if let Some(egui_tiles::Tile::Container(container)) = tiles.get(id) {
            match container {
                // Prefer a stable label for tab containers when none is explicitly set.
                Container::Tabs(_) => {
                    let title = if Some(id) == self.root_id {
                        String::new()
                    } else {
                        "New tab".to_string()
                    };
                    info!(
                        target: "tabs.title",
                        ?id,
                        root = Some(id) == self.root_id,
                        source = "fallback_tabs",
                        title = %title
                    );
                    return title;
                }
                Container::Linear(linear) => {
                    if linear.children.len() == 1 {
                        if let Some(child) = linear.children.first().copied() {
                            let text = self.tab_title_for_tile(tiles, child).text().to_string();
                            if Some(id) == self.root_id {
                                info!(
                                    target: "tabs.title",
                                    ?id,
                                    root = true,
                                    source = "fallback_linear_single",
                                    title = %text
                                );
                            }
                            return text;
                        }
                    }
                    // Default fallback label when no explicit title is set.
                    let title = "New tab".to_string();
                    if Some(id) == self.root_id {
                        info!(
                            target: "tabs.title",
                            ?id,
                            root = true,
                            source = "fallback_linear_multi",
                            title = %title
                        );
                    }
                    return title;
                }
                _ => {}
            }
            let derived = format!("{:?}", container.kind());
            if Some(id) == self.root_id {
                info!(
                    target: "tabs.title",
                    ?id,
                    root = true,
                    source = "fallback_kind",
                    title = %derived
                );
            }
            derived
        } else {
            if Some(id) == self.root_id {
                info!(
                    target: "tabs.title",
                    ?id,
                    root = true,
                    source = "fallback_missing",
                    title = "Container"
                );
            }
            "Container".to_owned()
        }
    }

    fn tab_role(&self, tiles: &Tiles<Pane>, tile_id: TileId) -> TabRole {
        // Hide chrome for the static sidebars (Hierarchy/Inspector).
        if let Some(Tile::Pane(pane)) = tiles.get(tile_id) {
            return match pane {
                Pane::Hierarchy | Pane::Inspector => TabRole::Super,
                _ => TabRole::Normal,
            };
        }
        TabRole::Normal
    }
}

impl egui_tiles::Behavior<Pane> for TreeBehavior<'_> {
    fn on_edit(&mut self, _edit_action: egui_tiles::EditAction) {}

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        let mut query =
            SystemState::<(Query<&GraphState>, Query<&Dashboard<Entity>>)>::new(self.world);
        let (graphs, dashes) = query.get(self.world);
        pane.title(&graphs, &dashes).into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        // Hide sidebar content completely when collapsed to a thin slice.
        if matches!(pane, Pane::Hierarchy | Pane::Inspector) && ui.available_size().x <= 20.0 {
            let size = ui.available_size();
            ui.allocate_space(size);
            return egui_tiles::UiResponse::None;
        }
        pane.ui(ui, &self.icons, self.world, &mut self.tree_actions)
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn tab_ui(
        &mut self,
        tiles: &mut Tiles<Pane>,
        ui: &mut Ui,
        id: egui::Id,
        tile_id: egui_tiles::TileId,
        state: &egui_tiles::TabState,
    ) -> egui::Response {
        let tab_role = self.tab_role(tiles, tile_id);
        let _clip = ui.clip_rect();
        let _avail = ui.available_rect_before_wrap();
        if matches!(tab_role, TabRole::Super) {
            // Hide sidebar tabs completely: no title, no background, no "+" chrome.
            let (_, rect) = ui.allocate_space(vec2(0.0, ui.available_height()));
            return ui.interact(rect, id, egui::Sense::hover());
        }
        let hide_title = false;
        let show_close = true;

        let tab_state = if state.active {
            TabState::Selected
        } else {
            TabState::Inactive
        };

        let persist_id = id.with(("rename_title", tile_id));
        let edit_flag_id = id.with(("rename_editing", tile_id));
        let edit_buf_id = id.with(("rename_buffer", tile_id));
        let mut is_editing = ui
            .ctx()
            .data(|d| d.get_temp::<bool>(edit_flag_id))
            .unwrap_or(false);

        let title_str: String =
            if let Some(custom) = ui.ctx().data(|d| d.get_temp::<String>(persist_id)) {
                custom
            } else if let Some(t) = self.container_titles.get(&tile_id) {
                t.clone()
            } else {
                match tiles.get(tile_id) {
                    Some(egui_tiles::Tile::Container(_)) => {
                        self.container_fallback_title(tiles, tile_id)
                    }
                    _ => self.tab_title_for_tile(tiles, tile_id).text().to_string(),
                }
            };
        if title_str.trim().is_empty() {
            let kind = match tiles.get(tile_id) {
                Some(Tile::Pane(p)) => match p {
                    Pane::Viewport(_) => "Viewport",
                    Pane::Graph(_) => "Graph",
                    Pane::Monitor(_) => "Monitor",
                    Pane::QueryTable(_) => "QueryTable",
                    Pane::QueryPlot(_) => "QueryPlot",
                    Pane::ActionTile(_) => "ActionTile",
                    Pane::VideoStream(_) => "VideoStream",
                    Pane::Dashboard(_) => "Dashboard",
                    Pane::Hierarchy => "Hierarchy",
                    Pane::Inspector => "Inspector",
                    Pane::SchematicTree(_) => "SchematicTree",
                },
                Some(Tile::Container(c)) => match c {
                    Container::Tabs(_) => "Tabs",
                    Container::Linear(_) => "Linear",
                    Container::Grid(_) => "Grid",
                },
                None => "Unknown",
            };
            info!(
                target: "tabs.title",
                ?tile_id,
                kind,
                has_custom_title = self.container_titles.contains_key(&tile_id),
                "tab title resolved to empty string"
            );
        }
        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 11.0;
        let mut galley = egui::WidgetText::from(title_str.clone()).into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            font_id.clone(),
        );

        let x_margin = self.tab_title_spacing(ui.visuals());
        let add_size = 0.0;
        let add_gap = 0.0;
        let close_size = if show_close { 14.0 } else { 0.0 };
        let close_gap = if show_close { x_margin * 1.5 } else { 0.0 };
        let tab_width = if hide_title {
            0.0
        } else {
            galley.size().x + x_margin * 2.0 + add_gap + add_size + close_gap + close_size
        };
        let (_, rect) = ui.allocate_space(vec2(tab_width, ui.available_height()));

        let (_parent_tabs_id, is_first_child) = tiles
            .iter()
            .find_map(|(id, tile)| {
                if let Tile::Container(Container::Tabs(tabs)) = tile {
                    if let Some(idx) = tabs.children.iter().position(|c| *c == tile_id) {
                        return Some((*id, idx == 0));
                    }
                }
                None
            })
            .unwrap_or((tile_id, false));

        let left_gap = if is_first_child { x_margin * 3.5 } else { 0.0 };
        let paint_rect = rect.shrink2(vec2(left_gap, 0.0));

        let text_rect = paint_rect;
        let response = {
            let mut resp = ui.interact(rect, id, egui::Sense::click_and_drag());
            let drag_distance = ui.input(|i| {
                let press = i.pointer.press_origin();
                let latest = i.pointer.latest_pos();
                press
                    .zip(latest)
                    .map(|(p, l)| p.distance(l))
                    .unwrap_or_default()
            });
            const DRAG_SLOP: f32 = 12.0;
            if drag_distance < DRAG_SLOP {
                resp.flags
                    .remove(Flags::DRAG_STARTED | Flags::DRAGGED | Flags::DRAG_STOPPED);
            }
            resp
        };

        if !self.read_only
            && !hide_title
            && state.active
            && response.double_clicked()
            && !is_editing
        {
            ui.ctx()
                .data_mut(|d| d.insert_temp(edit_buf_id, title_str.clone()));
            ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, true));
            is_editing = true;
        }

        if ui.is_rect_visible(rect) && !state.is_being_dragged {
            let is_pane_child = matches!(tiles.get(tile_id), Some(Tile::Pane(_)));

            let scheme = get_scheme();
            let bg_color = match tab_state {
                TabState::Selected if is_pane_child => Color32::WHITE,
                TabState::Selected => Color32::from_rgb(230, 230, 230),
                TabState::Inactive => Color32::from_rgb(0, 0, 0),
            };

            let text_color = match tab_state {
                TabState::Selected => Color32::BLACK,
                TabState::Inactive => Color32::from_rgb(230, 230, 230),
            };

            ui.painter().rect_filled(
                paint_rect,
                CornerRadius {
                    nw: 6,
                    ne: 6,
                    sw: 0,
                    se: 0,
                },
                bg_color,
            );
            if !self.read_only && is_editing {
                let label_rect =
                    egui::Align2::LEFT_CENTER.align_size_within_rect(galley.size(), text_rect);
                let edit_rect = label_rect.expand(1.0);

                let mut buf = ui.ctx().data_mut(|d| {
                    d.get_temp_mut_or::<String>(edit_buf_id, String::new())
                        .clone()
                });

                let resp = ui
                    .scope(|ui| {
                        ui.visuals_mut().override_text_color = Some(Color32::BLACK);
                        ui.put(
                            edit_rect,
                            egui::TextEdit::singleline(&mut buf)
                                .font(egui::TextStyle::Button)
                                .clip_text(true)
                                .desired_width(edit_rect.width())
                                .frame(false),
                        )
                    })
                    .inner;

                let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                let lost_focus = resp.lost_focus();

                if enter_pressed || lost_focus {
                    let new_title = buf.trim().to_owned();

                    ui.ctx().data_mut(|d| d.insert_temp(edit_flag_id, false));
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(edit_buf_id, new_title.clone()));
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(persist_id, new_title.clone()));
                    ui.memory_mut(|m| m.surrender_focus(resp.id));

                    if !self.read_only {
                        self.tree_actions
                            .push(TreeAction::RenameContainer(tile_id, new_title.clone()));
                    }

                    galley = egui::WidgetText::from(new_title).into_galley(
                        ui,
                        Some(egui::TextWrapMode::Extend),
                        f32::INFINITY,
                        font_id.clone(),
                    );

                    ui.painter().galley(
                        egui::Align2::LEFT_CENTER
                            .align_size_within_rect(galley.size(), text_rect)
                            .min,
                        galley.clone(),
                        text_color,
                    );
                } else {
                    ui.ctx().data_mut(|d| d.insert_temp(edit_buf_id, buf));
                    if !resp.has_focus() {
                        resp.request_focus();
                    }
                }
            } else {
                ui.painter().galley(
                    egui::Align2::LEFT_CENTER
                        .align_size_within_rect(galley.size(), text_rect)
                        .min,
                    galley.clone(),
                    text_color,
                );
            }

            if show_close {
                let close_rect = {
                    let base = egui::Align2::RIGHT_CENTER.align_size_within_rect(
                        vec2(close_size, close_size + 2.0),
                        paint_rect.shrink2(vec2(x_margin, 0.0)),
                    );
                    // Nudge slightly upward to sit higher than center.
                    base.translate(vec2(0.0, -2.0))
                };
                let close_response = ui.put(
                    close_rect,
                    EImageButton::new(self.icons.close)
                        .scale(1.3, 1.3)
                        .image_tint(match tab_state {
                            TabState::Inactive => scheme.text_primary,
                            TabState::Selected => scheme.bg_primary,
                        })
                        .bg_color(colors::TRANSPARENT)
                        .hovered_bg_color(colors::TRANSPARENT),
                );
                if close_response.clicked() {
                    self.tree_actions.push(TreeAction::DeleteTab(tile_id));
                }
            }

            ui.painter().hline(
                paint_rect.x_range(),
                paint_rect.bottom(),
                egui::Stroke::new(1.0, scheme.border_primary),
            );

            ui.painter().vline(
                paint_rect.right(),
                paint_rect.y_range(),
                egui::Stroke::new(1.0, scheme.border_primary),
            );
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn on_tab_button(
        &mut self,
        _tiles: &Tiles<Pane>,
        tile_id: TileId,
        button_response: egui::Response,
    ) -> egui::Response {
        if button_response.middle_clicked() && !self.read_only {
            self.tree_actions.push(TreeAction::DeleteTab(tile_id));
        } else if button_response.clicked() {
            self.tree_actions.push(TreeAction::SelectTile(tile_id));
        }
        button_response
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        32.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        Color32::from_rgb(0, 0, 0)
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            // Keep tab bars visible (titles and "+") even with a single tab.
            prune_empty_tabs: false,
            prune_single_child_tabs: false,
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, get_scheme().text_primary)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::with_opacity(get_scheme().text_primary, 0.6)
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = get_scheme().text_primary;
        frame.corner_radius = CornerRadius::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = if let Some(t) = self.container_titles.get(&tile_id) {
                egui::WidgetText::from(t.clone())
            } else {
                self.tab_title_for_tile(tiles, tile_id)
            };
            let text = text.text();
            ui.label(
                RichText::new(text)
                    .color(get_scheme().bg_secondary)
                    .size(11.0),
            );
        });
    }

    fn resize_stroke(
        &self,
        _style: &egui::Style,
        _resize_state: egui_tiles::ResizeState,
    ) -> Stroke {
        Stroke::NONE
    }

    fn top_bar_right_ui(
        &mut self,
        tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.read_only {
            return;
        }

        let is_sidebar_tabs =
            if let Some(Tile::Container(Container::Tabs(tabs))) = tiles.get(tile_id) {
                tabs.children.iter().all(|child| {
                    matches!(
                        tiles.get(*child),
                        Some(Tile::Pane(Pane::Hierarchy | Pane::Inspector))
                    )
                })
            } else {
                false
            };
        if is_sidebar_tabs {
            return;
        }

        let _title = self
            .container_titles
            .get(&tile_id)
            .cloned()
            .unwrap_or_else(|| self.container_fallback_title(tiles, tile_id));

        let mut layout = SystemState::<TileLayout>::new(self.world);
        let mut layout = layout.get_mut(self.world);

        let top_bar_rect = ui.available_rect_before_wrap();
        ui.painter().hline(
            top_bar_rect.x_range(),
            top_bar_rect.bottom(),
            egui::Stroke::new(1.0, get_scheme().border_primary),
        );

        ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        ui.style_mut().visuals.widgets.active.bg_stroke = Stroke::NONE;
        ui.add_space(5.0);
        let resp = ui.add(
            EImageButton::new(self.icons.add)
                .scale(1.4, 1.4)
                .image_tint(Color32::from_rgb(180, 180, 180))
                .bg_color(colors::TRANSPARENT)
                .hovered_bg_color(colors::TRANSPARENT),
        );
        if resp.clicked() {
            let tabs_id = tile_id;
            layout
                .cmd_palette_state
                .open_page(move || palette_items::create_tiles(tabs_id, true));
            layout.cmd_palette_state.target_window = self.target_window;
        }
    }
}
