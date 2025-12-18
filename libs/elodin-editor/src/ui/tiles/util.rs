use egui_tiles::Tile;

use super::types::Pane;

// Returns true for user-visible content panes (excluding fixed sidebars).
pub fn is_content_pane(pane: &Pane) -> bool {
    matches!(
        pane,
        Pane::Viewport(_)
            | Pane::Graph(_)
            | Pane::Monitor(_)
            | Pane::QueryTable(_)
            | Pane::QueryPlot(_)
            | Pane::ActionTile(_)
            | Pane::VideoStream(_)
            | Pane::Dashboard(_)
            | Pane::SchematicTree(_)
    )
}

pub fn is_content_tile(tile: &Tile<Pane>) -> bool {
    matches!(tile, Tile::Pane(pane) if is_content_pane(pane))
}
