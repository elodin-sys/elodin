use bevy::prelude::*;
use bevy_egui::{
    egui::{self, vec2, Color32, Frame, Margin, RichText, Rounding, Stroke, Ui, Visuals},
    EguiContexts,
};
use big_space::propagation::NoPropagateRot;
use big_space::GridCell;
use conduit::bevy::EntityMap;
use egui_tiles::{Container, Tile, TileId, Tiles};

use super::{colors, images::Images, widgets::button::ImageButton, SelectedObject, ViewportRect};
use crate::{plugins::navigation_gizmo::RenderLayerAlloc, spawn_main_camera};

struct TabIcons {
    pub add: egui::TextureId,
    pub close: egui::TextureId,
}

#[derive(Resource)]
pub struct TileState {
    tree: egui_tiles::Tree<Pane>,
}

impl TileState {
    fn insert_pane(&mut self, pane: Pane, active: bool) -> Option<TileId> {
        let root = self.tree.root()?;
        self.insert_pane_with_parent(pane, root, active)
    }
    fn insert_pane_with_parent(
        &mut self,
        pane: Pane,
        parent: TileId,
        active: bool,
    ) -> Option<TileId> {
        let child = self.tree.tiles.insert_pane(pane);
        let parent = self.tree.tiles.get_mut(parent)?;
        let Tile::Container(container) = parent else {
            return None;
        };
        container.add_child(child);
        if active {
            if let Container::Tabs(tabs) = container {
                tabs.set_active(child);
            }
        }
        Some(child)
    }
}

enum Pane {
    Viewport(ViewportPane),
    Graph,
    Welcome,
}

impl Pane {
    fn title(&self) -> &str {
        match self {
            Pane::Graph => "GRAPH",
            Pane::Viewport(_) => "VIEWPORT",
            Pane::Welcome => "WELCOME",
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
            Pane::Welcome => {
                ui.painter()
                    .rect(ui.max_rect(), 0.0, colors::BLACK, Stroke::NONE);
                Frame::none().inner_margin(Margin::same(8.0)).fill(colors::BLACK).show(ui, |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.heading("Welcome to the Elodin Editor!");
                        ui.label("Get started by connecting to a simulator, and then adding a viewport or graph");
                    });
                });
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

impl ViewportPane {
    fn spawn(
        commands: &mut Commands,
        asset_server: &Res<AssetServer>,
        meshes: &mut ResMut<Assets<Mesh>>,
        materials: &mut ResMut<Assets<StandardMaterial>>,
        render_layer_alloc: &mut ResMut<RenderLayerAlloc>,
    ) -> Self {
        let camera = Some(spawn_main_camera(
            commands,
            asset_server,
            meshes,
            materials,
            render_layer_alloc,
        ));
        Self { camera, rect: None }
    }
}

impl Default for TileState {
    fn default() -> Self {
        let panes = vec![];

        Self {
            tree: egui_tiles::Tree::new_tabs("tab_tree", panes),
        }
    }
}

struct TreeBehavior<'a> {
    icons: TabIcons,
    tab_diffs: Vec<TabDiff>,
    selected_object: &'a mut SelectedObject,
}

enum TabDiff {
    Add { parent: TileId, pane: Pane },
    AddViewport(TileId),
    Delete(TileId),
}

enum TabState {
    Active,
    Selected,
    Inactive,
}

impl<'a> egui_tiles::Behavior<Pane> for TreeBehavior<'a> {
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
        let is_selected = self.selected_object.is_tile_selected(tile_id);
        let tab_state = if is_selected {
            TabState::Selected
        } else if active {
            TabState::Active
        } else {
            TabState::Inactive
        };
        let text = self.tab_title_for_tile(tiles, tile_id);
        let mut font_id = egui::TextStyle::Button.resolve(ui.style());
        font_id.size = 11.0;
        let galley = text.into_galley(ui, Some(false), f32::INFINITY, font_id);
        let x_margin = self.tab_title_spacing(ui.visuals());
        let (_, rect) = ui.allocate_space(vec2(
            galley.size().x + x_margin * 4.0,
            ui.available_height(),
        ));
        let text_rect = rect
            .shrink2(vec2(x_margin * 4.0, 0.0))
            .translate(vec2(-3.0 * x_margin, 0.0));
        let response = ui.interact(rect, id, egui::Sense::click_and_drag());

        ui.painter().vline(
            rect.right(),
            rect.y_range(),
            egui::Stroke::new(1.0, colors::BLACK),
        );

        if ui.is_rect_visible(rect) && !is_being_dragged {
            let bg_color = match tab_state {
                TabState::Active => colors::BLACK,
                TabState::Selected => colors::CREMA,
                TabState::Inactive => colors::STONE_950,
            };

            let text_color = match tab_state {
                TabState::Active => colors::CREMA,
                TabState::Selected => colors::STONE_950,
                TabState::Inactive => colors::with_opacity(colors::CREMA, 0.6),
            };

            ui.painter().rect(rect, 0.0, bg_color, Stroke::NONE);
            ui.painter().galley(
                egui::Align2::LEFT_CENTER
                    .align_size_within_rect(galley.size(), text_rect)
                    .min,
                galley,
                text_color,
            );
            ui.add_space(-3.0 * x_margin);
            let close_response = ui.add(
                ImageButton::new(self.icons.close)
                    .scale(1.3, 1.3)
                    .image_tint(match tab_state {
                        TabState::Active | TabState::Inactive => colors::CREMA,
                        TabState::Selected => colors::BLACK,
                    })
                    .bg_color(Color32::TRANSPARENT),
            );
            if close_response.clicked() {
                self.tab_diffs.push(TabDiff::Delete(tile_id));
            }
        }

        self.on_tab_button(tiles, tile_id, response)
    }

    fn on_tab_button(
        &mut self,
        tiles: &Tiles<Pane>,
        tile_id: TileId,
        button_response: egui::Response,
    ) -> egui::Response {
        if button_response.middle_clicked() {
            self.tab_diffs.push(TabDiff::Delete(tile_id));
        } else if button_response.clicked() {
            let Some(tile) = tiles.get(tile_id) else {
                return button_response;
            };
            match tile {
                Tile::Pane(Pane::Graph) => {
                    *self.selected_object = SelectedObject::Graph { tile_id };
                }
                Tile::Pane(Pane::Welcome) => {
                    *self.selected_object = SelectedObject::None;
                }
                Tile::Pane(Pane::Viewport(viewport)) => {
                    let Some(camera) = viewport.camera else {
                        return button_response;
                    };
                    *self.selected_object = SelectedObject::Viewport { tile_id, camera };
                }
                _ => {}
            }
        }
        button_response
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        32.0
    }

    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        colors::STONE_950
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            join_nested_linear_containers: true,
            ..Default::default()
        }
    }

    fn drag_preview_stroke(&self, _visuals: &Visuals) -> Stroke {
        Stroke::new(1.0, colors::ORANGE_50)
    }

    fn drag_preview_color(&self, _visuals: &Visuals) -> Color32 {
        colors::with_opacity(colors::CREMA, 0.6)
    }

    fn drag_ui(&mut self, tiles: &Tiles<Pane>, ui: &mut Ui, tile_id: TileId) {
        let mut frame = egui::Frame::popup(ui.style());
        frame.fill = colors::CREMA;
        frame.rounding = Rounding::ZERO;
        frame.stroke = Stroke::NONE;
        frame.shadow = egui::epaint::Shadow::NONE;
        frame.show(ui, |ui| {
            let text = self.tab_title_for_tile(tiles, tile_id);
            let text = text.text();
            ui.label(RichText::new(text).color(colors::STONE_950).size(11.0));
        });
    }

    fn top_bar_right_ui(
        &mut self,
        _tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        ui.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        ui.style_mut().visuals.widgets.active.bg_stroke = Stroke::NONE;
        ui.add_space(5.0);
        let mut resp = ui.add(ImageButton::new(self.icons.add).scale(1.4, 1.4));
        if resp.clicked() {
            resp.clicked = [false, true, false, false, false];
        }
        resp.context_menu(|ui| {
            ui.style_mut().spacing.item_spacing = vec2(16.0, 8.0);
            if ui.button("VIEWPORT").clicked() {
                self.tab_diffs.push(TabDiff::AddViewport(tile_id));
            }
            ui.separator();
            if ui.button("GRAPH").clicked() {
                self.tab_diffs.push(TabDiff::Add {
                    parent: tile_id,
                    pane: Pane::Graph,
                });
            }
        });
    }
}
pub fn setup_default_tiles(mut tile_state: ResMut<TileState>) {
    if tile_state.tree.active_tiles().is_empty() {
        tile_state.tree = egui_tiles::Tree::new_tabs("tab_tree", vec![Pane::Welcome]);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_tiles(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<TileState>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    images: Local<Images>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_layer_alloc: ResMut<RenderLayerAlloc>,
    mut selected_object: ResMut<SelectedObject>,
) {
    let icons = TabIcons {
        add: contexts.add_image(images.icon_add.clone_weak()),
        close: contexts.add_image(images.icon_close.clone_weak()),
    };

    egui::CentralPanel::default()
        .frame(Frame {
            fill: Color32::TRANSPARENT,
            ..Default::default()
        })
        .show(contexts.ctx_mut(), |ui| {
            let mut behavior = TreeBehavior {
                icons,
                tab_diffs: vec![],
                selected_object: selected_object.as_mut(),
            };
            ui_state.tree.ui(&mut behavior, ui);
            for diff in behavior.tab_diffs.drain(..) {
                match diff {
                    TabDiff::Add { parent, pane } => {
                        ui_state.insert_pane_with_parent(pane, parent, true);
                    }
                    TabDiff::Delete(tab_id) => {
                        let Some(tile) = ui_state.tree.tiles.get(tab_id) else {
                            continue;
                        };

                        if let egui_tiles::Tile::Pane(Pane::Viewport(viewport)) = tile {
                            if let Some(camera) = viewport.camera {
                                commands.entity(camera).despawn(); // TODO(sphw): garbage collect old nav-gizmos
                            }
                        };

                        if ui_state.tree.tiles.len() > 1 {
                            ui_state.tree.tiles.remove(tab_id);
                        }
                    }
                    TabDiff::AddViewport(parent) => {
                        let pane = Pane::Viewport(ViewportPane::spawn(
                            &mut commands,
                            &asset_server,
                            &mut meshes,
                            &mut materials,
                            &mut render_layer_alloc,
                        ));
                        ui_state.insert_pane_with_parent(pane, parent, true);
                    }
                }
            }
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
                    if let Some(mut cam) = commands.get_entity(cam) {
                        cam.insert(ViewportRect(viewport.rect));
                    }
                } else if let Some(mut cam) = commands.get_entity(cam) {
                    cam.insert(ViewportRect(None));
                }
            }
        });
}

#[derive(Component)]
pub struct SyncedViewport;

#[allow(clippy::too_many_arguments)]
pub fn sync_viewports(
    panels: Query<(Entity, &conduit::well_known::Panel), Without<SyncedViewport>>,
    asset_server: Res<AssetServer>,
    mut ui_state: ResMut<TileState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut render_layer_alloc: ResMut<RenderLayerAlloc>,
    mut commands: Commands,
    entity_map: Res<EntityMap>,
    grid_cell: Query<&GridCell<i128>>,
) {
    for (entity, panel) in panels.iter() {
        match panel {
            conduit::well_known::Panel::Viewport(viewport) => {
                let pane = ViewportPane::spawn(
                    &mut commands,
                    &asset_server,
                    &mut meshes,
                    &mut materials,
                    &mut render_layer_alloc,
                );
                let camera = pane.camera.expect("no camera spawned for viewport");
                let mut camera = commands.entity(camera);
                if let Some(parent) = viewport.track_entity {
                    if let Some(parent) = entity_map.0.get(&parent) {
                        if let Ok(grid_cell) = grid_cell.get(*parent) {
                            camera.insert(*grid_cell);
                        }
                        camera.set_parent(*parent);
                    }
                };
                camera.insert(Transform {
                    translation: Vec3::new(viewport.pos.x, viewport.pos.y, viewport.pos.z),
                    rotation: Quat::from_xyzw(
                        viewport.rotation.i,
                        viewport.rotation.j,
                        viewport.rotation.k,
                        viewport.rotation.w,
                    ),
                    ..Default::default()
                });
                if !viewport.track_rotation {
                    camera.insert(NoPropagateRot);
                } else {
                    camera.remove::<NoPropagateRot>();
                }
                let pane = Pane::Viewport(pane);
                ui_state.insert_pane(pane, viewport.active);
                commands.entity(entity).insert(SyncedViewport);
            }
        }
    }
}
