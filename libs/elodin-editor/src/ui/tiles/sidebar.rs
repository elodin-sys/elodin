use egui::UiBuilder;
use egui_tiles::{Container, Tile, Tiles};

use super::Pane;

pub const SIDEBAR_CONTENT_PAD_LEFT: f32 = 8.0;

pub fn sidebar_content_ui<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let mut rect = ui.max_rect();
    let width = rect.width();
    let height = rect.height();
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return add_contents(ui);
    }

    let pad = SIDEBAR_CONTENT_PAD_LEFT.min(width);
    if !pad.is_finite() || pad <= 0.0 {
        return add_contents(ui);
    }

    rect.min.x = (rect.min.x + pad).min(rect.max.x);
    ui.scope_builder(UiBuilder::new().max_rect(rect), add_contents)
        .inner
}

pub fn tab_add_visible(tiles: &Tiles<Pane>, tabs: &egui_tiles::Tabs) -> bool {
    // Hide "+" for Tabs containing only one Tabs child (wrapper Tabs)
    if tabs.children.len() == 1
        && matches!(
            tiles.get(tabs.children[0]),
            Some(Tile::Container(Container::Tabs(_)))
        )
    {
        return false;
    }
    true
}
