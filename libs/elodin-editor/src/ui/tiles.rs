use bevy::prelude::*;
use bevy_egui::{
    egui::{self, Color32, Frame, Rounding, Stroke, Ui, Visuals},
    EguiContexts,
};

use egui_tiles::{TileId, Tiles};

use super::{colors, ViewportRect};
use crate::MainCamera;

#[derive(Resource)]
pub struct TileState {
    tree: egui_tiles::Tree<Pane>,
}

enum Pane {
    Viewport(ViewportPane),
    Graph,
}

impl Pane {
    fn title(&self) -> &str {
        match self {
            Pane::Graph => "GRAPH",
            Pane::Viewport(_) => "VIEWPORT",
        }
    }

    fn ui(&mut self, ui: &mut Ui) -> egui_tiles::UiResponse {
        match self {
            Pane::Graph => {
                ui.painter()
                    .rect(ui.max_rect(), 0.0, colors::BLACK, Stroke::NONE);
                egui_tiles::UiResponse::None
            }
            Pane::Viewport(pane) => {
                pane.rect = Some(ui.max_rect());
                egui_tiles::UiResponse::None
            }
        }
    }
}

#[derive(Default)]
struct ViewportPane {
    pub camera: Option<Entity>,
    pub rect: Option<egui::Rect>,
}

impl Default for TileState {
    fn default() -> Self {
        let panes = vec![Pane::Graph];

        Self {
            tree: egui_tiles::Tree::new_tabs("tab_tree", panes),
        }
    }
}

struct TreeBehavior {}

impl egui_tiles::Behavior<Pane> for TreeBehavior {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        pane.ui(ui)
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn tab_ui(
        &mut self,
        tiles: &Tiles<Pane>,
        ui: &mut Ui,
        id: egui::Id,
        tile_id: egui_tiles::TileId,
        active: bool,
        is_being_dragged: bool,
    ) -> egui::Response {
        use egui::*;
        let text = self.tab_title_for_tile(tiles, tile_id);
        let font_id = TextStyle::Button.resolve(ui.style());
        let galley = text.into_galley(ui, Some(false), f32::INFINITY, font_id);

        let x_margin = self.tab_title_spacing(ui.visuals());
        let (_, rect) = ui.allocate_space(vec2(
            galley.size().x + 2.0 * x_margin,
            ui.available_height(),
        ));
        let response = ui.interact(rect, id, Sense::click_and_drag());

        if ui.is_rect_visible(rect) && !is_being_dragged {
            let bg_color = if active {
                colors::BLACK
            } else {
                colors::STONE_950
            };
            ui.painter().rect(rect, 0.0, bg_color, Stroke::NONE);

            let text_color = if active {
                colors::CREMA
            } else {
                colors::CREMA_60
            };
            ui.painter().galley(
                egui::Align2::CENTER_CENTER
                    .align_size_within_rect(galley.size(), rect)
                    .min,
                galley,
                text_color,
            );
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        34.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        colors::STONE_950
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, colors::ORANGE_50)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::CREMA_60
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = colors::CREMA;
        frame.rounding = Rounding::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = self
                .tab_title_for_tile(tiles, tile_id)
                .color(colors::STONE_950);
            ui.label(text);
        });
    }
}
pub fn setup_default_tiles(
    mut ui_state: ResMut<TileState>,
    main_camera_query: Query<Entity, With<MainCamera>>,
) {
    let mut panes: Vec<_> = main_camera_query
        .iter()
        .map(|camera| {
            Pane::Viewport(ViewportPane {
                camera: Some(camera),
                rect: None,
            })
        })
        .collect();
    panes.push(Pane::Graph);
    ui_state.tree = egui_tiles::Tree::new_tabs("tab_tree", panes);
}

#[allow(clippy::too_many_arguments)]
pub fn render_tiles(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<TileState>,
    mut commands: Commands,
) {
    egui::CentralPanel::default()
        .frame(Frame {
            fill: Color32::TRANSPARENT,
            ..Default::default()
        })
        .show(contexts.ctx_mut(), |ui| {
            let mut behavior = TreeBehavior {};
            ui_state.tree.ui(&mut behavior, ui);
            let tiles = ui_state.tree.tiles.iter();
            let active_tiles = ui_state.tree.active_tiles();
            for (tile_id, tile) in tiles {
                let egui_tiles::Tile::Pane(pane) = tile else {
                    continue;
                };
                let Pane::Viewport(viewport) = pane else {
                    continue;
                };
                let Some(cam) = viewport.camera else { continue };
                if active_tiles.contains(tile_id) {
                    commands.entity(cam).insert(ViewportRect(viewport.rect));
                } else {
                    commands.entity(cam).insert(ViewportRect(None));
                }
            }
        });
}
