use super::{
    behavior::TreeBehavior,
    sidebar::{SidebarMaskState, collect_sidebar_gutter_updates},
    tile_actions::TreeAction,
    *,
};

// Run the egui tiles UI and capture resulting actions plus share updates from gutter drags.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_tree_and_collect_updates(
    world: &mut World,
    ui: &mut egui::Ui,
    tree: &mut egui_tiles::Tree<Pane>,
    icons: TileIcons,
    container_titles: HashMap<TileId, String>,
    read_only: bool,
    window: Option<Entity>,
    mask_state: &mut SidebarMaskState,
    tree_actions: SmallVec<[TreeAction; 4]>,
) -> (SmallVec<[TreeAction; 4]>, Vec<ShareUpdate>) {
    let mut behavior = TreeBehavior {
        icons,
        // This world here makes getting ui_state difficult.
        world,
        tree_actions,
        container_titles,
        read_only,
        target_window: window,
        root_id: tree.root(),
    };
    let _logged_diag = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(egui::Id::new(("center_tabs_diag", window))))
        .unwrap_or(false);
    let _window_id = window.and_then(|w| behavior.world.get::<WindowId>(w));
    let _max_rect = ui.max_rect();
    let _clip = ui.clip_rect();
    tree.ui(&mut behavior, ui);
    let window_width = ui.ctx().screen_rect().width();
    let gutter_width: f32 = (window_width * 0.02).max(12.0);
    let painter = ui.painter_at(ui.max_rect());
    let share_updates = collect_sidebar_gutter_updates(tree, ui, painter, gutter_width, mask_state);
    let TreeBehavior { tree_actions, .. } = behavior;
    (tree_actions, share_updates)
}

// Apply gutter-driven share adjustments back onto the stored tree.
pub(super) fn apply_share_updates(tree: &mut egui_tiles::Tree<Pane>, updates: &[ShareUpdate]) {
    for (container_id, left_id, right_id, left_share, right_share) in updates {
        if let Some(Tile::Container(Container::Linear(linear))) = tree.tiles.get_mut(*container_id)
        {
            linear.shares.set_share(*left_id, *left_share);
            linear.shares.set_share(*right_id, *right_share);
        }
    }
}
