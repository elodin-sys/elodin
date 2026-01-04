use super::{Pane, ShareUpdate, TileId};
use crate::ui::colors::get_scheme;
use egui::{Stroke, UiBuilder};
use egui_tiles::{Container, Tile, Tiles};

pub const MIN_SIDEBAR_FRACTION: f32 = 0.05;
pub const MIN_SIDEBAR_PX: f32 = 16.0;
pub const MIN_SIDEBAR_MASKED_PX: f32 = 4.0;
pub const MIN_OTHER_PX: f32 = 32.0;
pub const SIDEBAR_CONTENT_PAD_LEFT: f32 = 8.0;
pub const COLLAPSED_SHARE_FALLBACK: f32 = 0.01;
pub const DEFAULT_SIDEBAR_FRACTION: f32 = 0.15;
pub const MASK_THRESHOLD_MULTIPLIER: f32 = 1.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidebarKind {
    Hierarchy,
    Inspector,
}

#[derive(Clone, Copy, Debug)]
pub struct SidebarMaskState {
    pub hierarchy_masked: bool,
    pub inspector_masked: bool,
    pub last_hierarchy_share: Option<f32>,
    pub last_inspector_share: Option<f32>,
}

impl Default for SidebarMaskState {
    fn default() -> Self {
        Self {
            hierarchy_masked: true,
            inspector_masked: true,
            last_hierarchy_share: Some(0.2),
            last_inspector_share: Some(0.2),
        }
    }
}

impl SidebarMaskState {
    pub fn masked(&self, kind: SidebarKind) -> bool {
        match kind {
            SidebarKind::Hierarchy => self.hierarchy_masked,
            SidebarKind::Inspector => self.inspector_masked,
        }
    }

    pub fn set_masked(&mut self, kind: SidebarKind, masked: bool) {
        match kind {
            SidebarKind::Hierarchy => self.hierarchy_masked = masked,
            SidebarKind::Inspector => self.inspector_masked = masked,
        }
    }

    pub fn last_share(&self, kind: SidebarKind) -> Option<f32> {
        match kind {
            SidebarKind::Hierarchy => self.last_hierarchy_share,
            SidebarKind::Inspector => self.last_inspector_share,
        }
    }

    pub fn set_last_share(&mut self, kind: SidebarKind, share: Option<f32>) {
        match kind {
            SidebarKind::Hierarchy => self.last_hierarchy_share = share,
            SidebarKind::Inspector => self.last_inspector_share = share,
        }
    }
}

// Compute sidebar gutter interactions and return share updates to apply.
pub fn collect_sidebar_gutter_updates(
    tree: &mut egui_tiles::Tree<Pane>,
    ui: &mut egui::Ui,
    painter: egui::Painter,
    gutter_width: f32,
    mask_state: &mut SidebarMaskState,
) -> Vec<ShareUpdate> {
    struct GutterCtx<'a> {
        tree: &'a mut egui_tiles::Tree<Pane>,
        ui: &'a mut egui::Ui,
        painter: egui::Painter,
        gutter_width: f32,
        mask_state: &'a mut SidebarMaskState,
        share_updates: Vec<ShareUpdate>,
    }

    #[derive(Clone, Copy)]
    struct PairInfo {
        container_id: TileId,
        parent_rect: egui::Rect,
        left_id: TileId,
        left_rect: egui::Rect,
        left_kind: Option<SidebarKind>,
        right_id: TileId,
        right_rect: egui::Rect,
        right_kind: Option<SidebarKind>,
    }

    impl<'a> GutterCtx<'a> {
        fn sidebar_kind(&self, id: TileId) -> Option<SidebarKind> {
            match self.tree.tiles.get(id) {
                Some(Tile::Pane(pane)) => pane.sidebar_kind(),
                Some(Tile::Container(Container::Tabs(tabs))) => {
                    for child in tabs.children.iter() {
                        if let Some(kind) = self.sidebar_kind(*child) {
                            return Some(kind);
                        }
                    }
                    None
                }
                Some(Tile::Container(Container::Linear(linear))) => {
                    for child in linear.children.iter() {
                        if let Some(kind) = self.sidebar_kind(*child) {
                            return Some(kind);
                        }
                    }
                    None
                }
                Some(Tile::Container(Container::Grid(grid))) => {
                    for child in grid.children() {
                        if let Some(kind) = self.sidebar_kind(*child) {
                            return Some(kind);
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        fn apply_shares(
            &mut self,
            container_id: TileId,
            left_id: TileId,
            right_id: TileId,
            left: f32,
            right: f32,
        ) {
            if let Some(Tile::Container(Container::Linear(linear))) =
                self.tree.tiles.get_mut(container_id)
            {
                linear.shares.set_share(left_id, left);
                linear.shares.set_share(right_id, right);
            }
            self.share_updates
                .push((container_id, left_id, right_id, left, right));
            self.ui.ctx().request_repaint();
        }

        fn compute_sidebar_shares(
            &self,
            pair_sum: f32,
            min_other_share: f32,
            min_sidebar_share: f32,
            target_sidebar_share: f32,
            sidebar_on_left: bool,
        ) -> (f32, f32) {
            let max_sidebar_share = (pair_sum - min_other_share).max(0.01);
            let clamped = target_sidebar_share
                .max(min_sidebar_share.max(0.01))
                .min(max_sidebar_share);
            let left_share = if sidebar_on_left {
                clamped
            } else {
                pair_sum - clamped
            };
            let right_share = pair_sum - left_share;
            (left_share.max(0.01), right_share.max(0.01))
        }

        /// Compute the collapsed share for a sidebar based on gutter width and share ratio.
        fn compute_collapsed_share(
            &self,
            gutter_draw_width: f32,
            share_per_px: f32,
            pair_sum: f32,
        ) -> f32 {
            if share_per_px > 0.0 {
                gutter_draw_width * share_per_px
            } else {
                pair_sum * COLLAPSED_SHARE_FALLBACK
            }
        }

        /// Draw the gutter between sidebar and main content.
        fn draw_gutter(
            &mut self,
            pair: &PairInfo,
            gutter_draw_width: f32,
            left_sidebar: bool,
        ) -> egui::Rect {
            let gap = pair.right_rect.min.x - pair.left_rect.max.x;
            let mut center_x = (pair.left_rect.max.x + pair.right_rect.min.x) * 0.5;
            if gap < gutter_draw_width {
                let offset = (gutter_draw_width - gap).max(0.0) * 0.5;
                if left_sidebar {
                    center_x -= offset;
                } else {
                    center_x += offset;
                }
            }
            let half = gutter_draw_width * 0.5;
            center_x = center_x
                .max(pair.parent_rect.left() + half)
                .min(pair.parent_rect.right() - half);
            let gutter_rect = egui::Rect::from_min_max(
                egui::pos2(center_x - half, pair.parent_rect.top()),
                egui::pos2(center_x + half, pair.parent_rect.bottom()),
            );

            let gutter_color = get_scheme().border_primary;
            let fill = gutter_color;
            let stroke = Stroke::new(1.0, gutter_color);
            self.painter.rect_filled(gutter_rect, 0.0, fill);
            self.painter
                .rect_stroke(gutter_rect, 0.0, stroke, egui::StrokeKind::Inside);

            gutter_rect
        }

        /// Handle click events on the gutter to toggle sidebar visibility.
        #[allow(clippy::too_many_arguments)]
        fn handle_gutter_click(
            &mut self,
            pair: &PairInfo,
            sidebar_kind: SidebarKind,
            sidebar_on_left: bool,
            sidebar_masked: bool,
            share_left: f32,
            share_right: f32,
            pair_sum: f32,
            share_per_px: f32,
            min_sidebar_px: f32,
            min_sidebar_share: f32,
            min_other_share: f32,
            gutter_rect: egui::Rect,
            response: &egui::Response,
        ) -> bool {
            let click_inside_gutter = response.clicked_by(egui::PointerButton::Primary)
                && self
                    .ui
                    .input(|i| i.pointer.interact_pos())
                    .map(|p| gutter_rect.shrink(1.0).contains(p))
                    .unwrap_or(false);

            if !click_inside_gutter {
                return sidebar_masked;
            }

            let (left_share, right_share) = if sidebar_masked {
                let default_px =
                    (pair.parent_rect.width() * DEFAULT_SIDEBAR_FRACTION).max(min_sidebar_px);
                let restore_share = if share_per_px > 0.0 {
                    default_px * share_per_px
                } else {
                    pair_sum * DEFAULT_SIDEBAR_FRACTION
                };
                self.compute_sidebar_shares(
                    pair_sum,
                    min_other_share,
                    min_sidebar_share,
                    restore_share,
                    sidebar_on_left,
                )
            } else {
                let current_sidebar_share = if sidebar_on_left {
                    share_left
                } else {
                    share_right
                };
                self.mask_state
                    .set_last_share(sidebar_kind, Some(current_sidebar_share));

                let gutter_draw_width = self.gutter_width.max(MIN_SIDEBAR_MASKED_PX);
                let collapsed_share =
                    self.compute_collapsed_share(gutter_draw_width, share_per_px, pair_sum);

                self.compute_sidebar_shares(
                    pair_sum,
                    min_other_share,
                    collapsed_share,
                    collapsed_share,
                    sidebar_on_left,
                )
            };
            self.apply_shares(
                pair.container_id,
                pair.left_id,
                pair.right_id,
                left_share,
                right_share,
            );
            self.mask_state.set_masked(sidebar_kind, !sidebar_masked);
            !sidebar_masked
        }

        /// Handle drag events on the gutter to resize the sidebar.
        #[allow(clippy::too_many_arguments)]
        fn handle_gutter_drag(
            &mut self,
            pair: &PairInfo,
            sidebar_kind: SidebarKind,
            sidebar_on_left: bool,
            left_sidebar: bool,
            right_sidebar: bool,
            min_sidebar_share: f32,
            gutter_rect: egui::Rect,
            response: &egui::Response,
        ) {
            #[derive(Clone, Copy, Default)]
            struct DragState {
                left_width: f32,
                right_width: f32,
                start_x: f32,
                active: bool,
            }

            let id = self.ui.id().with((
                "sidebar_gutter",
                pair.container_id,
                pair.left_id,
                pair.right_id,
            ));

            let mut drag_state = self
                .ui
                .ctx()
                .data(|d| d.get_temp::<DragState>(id))
                .unwrap_or_default();

            let pointer_pos = self.ui.input(|i| i.pointer.interact_pos());
            let pointer_down = self.ui.input(|i| i.pointer.primary_down());

            if !drag_state.active && pointer_down && response.hovered() {
                let start_x = pointer_pos.map(|p| p.x).unwrap_or(gutter_rect.center().x);
                drag_state = DragState {
                    left_width: pair.left_rect.width(),
                    right_width: pair.right_rect.width(),
                    start_x,
                    active: true,
                };
                self.ui.ctx().data_mut(|d| d.insert_temp(id, drag_state));
            }

            if drag_state.active && pointer_down {
                let delta = pointer_pos.map(|p| p.x - drag_state.start_x).unwrap_or(0.0);
                let min_left = if left_sidebar {
                    MIN_SIDEBAR_MASKED_PX
                } else {
                    MIN_OTHER_PX
                };
                let min_right = if right_sidebar {
                    MIN_SIDEBAR_MASKED_PX
                } else {
                    MIN_OTHER_PX
                };
                let new_left = (drag_state.left_width + delta).max(min_left);
                let new_right = (drag_state.right_width - delta).max(min_right);
                if let Some(Tile::Container(Container::Linear(linear))) =
                    self.tree.tiles.get_mut(pair.container_id)
                {
                    let share_left = linear.shares[pair.left_id];
                    let share_right = linear.shares[pair.right_id];
                    let share_sum = share_left + share_right;
                    let current_sidebar_share = if sidebar_on_left {
                        share_left
                    } else {
                        share_right
                    };

                    let total = new_left + new_right;
                    let new_left_share = share_sum * new_left / total.max(1.0);
                    let new_right_share = share_sum - new_left_share;
                    let left_clamped = new_left_share.max(0.01);
                    let right_clamped = new_right_share.max(0.01);
                    self.apply_shares(
                        pair.container_id,
                        pair.left_id,
                        pair.right_id,
                        left_clamped,
                        right_clamped,
                    );

                    let new_sidebar_share = if sidebar_on_left {
                        new_left_share
                    } else {
                        new_right_share
                    };
                    let mask_threshold = min_sidebar_share * MASK_THRESHOLD_MULTIPLIER;
                    if new_sidebar_share <= mask_threshold && !self.mask_state.masked(sidebar_kind)
                    {
                        let prev_last = self.mask_state.last_share(sidebar_kind);
                        let should_update_last = current_sidebar_share > min_sidebar_share * 1.5;
                        let store_share = if should_update_last {
                            Some(current_sidebar_share)
                        } else {
                            prev_last
                        };
                        self.mask_state.set_last_share(sidebar_kind, store_share);
                        self.mask_state.set_masked(sidebar_kind, true);
                    }
                }
            }

            if drag_state.active && !pointer_down {
                self.ui.ctx().data_mut(|d| d.remove::<DragState>(id));
            }
        }

        fn process_pair(&mut self, pair: PairInfo) {
            let left_sidebar = pair.left_kind.is_some();
            let right_sidebar = pair.right_kind.is_some();
            if left_sidebar == right_sidebar {
                return;
            }
            let sidebar_kind = match (pair.left_kind, pair.right_kind) {
                (Some(k), _) => k,
                (_, Some(k)) => k,
                _ => return,
            };
            let gutter_draw_width = self.gutter_width.max(MIN_SIDEBAR_MASKED_PX);
            let sidebar_on_left = left_sidebar;
            let pair_width = pair.left_rect.width() + pair.right_rect.width();
            let (share_left, share_right) = match self.tree.tiles.get(pair.container_id) {
                Some(Tile::Container(Container::Linear(linear))) => {
                    (linear.shares[pair.left_id], linear.shares[pair.right_id])
                }
                _ => return,
            };
            let pair_sum = share_left + share_right;
            if pair_sum <= 0.0 {
                return;
            }
            let share_per_px = if pair_width > 0.0 {
                pair_sum / pair_width
            } else {
                0.0
            };
            let min_sidebar_px =
                (pair.parent_rect.width() * MIN_SIDEBAR_FRACTION).max(MIN_SIDEBAR_PX);
            let min_other_px = MIN_OTHER_PX;
            let min_sidebar_share = if share_per_px > 0.0 {
                min_sidebar_px * share_per_px
            } else {
                pair_sum * MIN_SIDEBAR_FRACTION
            };
            let min_other_share = if share_per_px > 0.0 {
                min_other_px * share_per_px
            } else {
                pair_sum * MIN_SIDEBAR_FRACTION
            };

            let mut sidebar_masked = self.mask_state.masked(sidebar_kind);
            if sidebar_masked {
                let collapsed_share =
                    self.compute_collapsed_share(gutter_draw_width, share_per_px, pair_sum);
                let (target_left, target_right) = self.compute_sidebar_shares(
                    pair_sum,
                    min_other_share,
                    collapsed_share,
                    collapsed_share,
                    sidebar_on_left,
                );
                if (share_left - target_left).abs() > 0.0001
                    || (share_right - target_right).abs() > 0.0001
                {
                    self.apply_shares(
                        pair.container_id,
                        pair.left_id,
                        pair.right_id,
                        target_left,
                        target_right,
                    );
                }
            }

            let gutter_rect = self.draw_gutter(&pair, gutter_draw_width, left_sidebar);
            let id = self.ui.id().with((
                "sidebar_gutter",
                pair.container_id,
                pair.left_id,
                pair.right_id,
            ));
            let response = self
                .ui
                .interact(gutter_rect, id, egui::Sense::click_and_drag())
                .on_hover_cursor(egui::CursorIcon::PointingHand);

            if response.hovered() {
                self.ui
                    .output_mut(|o| o.cursor_icon = egui::CursorIcon::PointingHand);
            }

            sidebar_masked = self.handle_gutter_click(
                &pair,
                sidebar_kind,
                sidebar_on_left,
                sidebar_masked,
                share_left,
                share_right,
                pair_sum,
                share_per_px,
                min_sidebar_px,
                min_sidebar_share,
                min_other_share,
                gutter_rect,
                &response,
            );

            if !sidebar_masked {
                self.handle_gutter_drag(
                    &pair,
                    sidebar_kind,
                    sidebar_on_left,
                    left_sidebar,
                    right_sidebar,
                    min_sidebar_share,
                    gutter_rect,
                    &response,
                );
            } else {
                #[derive(Clone, Copy, Default)]
                struct DragState {
                    _left_width: f32,
                    _right_width: f32,
                    _start_x: f32,
                    active: bool,
                }
                if self
                    .ui
                    .ctx()
                    .data(|d| d.get_temp::<DragState>(id))
                    .is_some_and(|ds| ds.active)
                {
                    self.ui.ctx().data_mut(|d| d.remove::<DragState>(id));
                }
            }
        }

        fn process_container(&mut self, container_id: TileId) {
            let parent_rect = match self.tree.tiles.rect(container_id) {
                Some(rect) => rect,
                None => return,
            };

            let visible_children: Vec<TileId> = match self.tree.tiles.get(container_id) {
                Some(Tile::Container(Container::Linear(linear))) => linear
                    .children
                    .iter()
                    .copied()
                    .filter(|child| self.tree.tiles.is_visible(*child))
                    .collect(),
                _ => Vec::new(),
            };

            if visible_children.len() < 2 {
                return;
            }

            let mut child_data = Vec::new();
            for child in &visible_children {
                if let Some(rect) = self.tree.tiles.rect(*child) {
                    let sidebar_kind = self.sidebar_kind(*child);
                    child_data.push((*child, rect, sidebar_kind));
                }
            }

            if child_data.len() < 2 {
                return;
            }

            for pair in child_data.windows(2) {
                let (left_id, left_rect, left_kind) = pair[0];
                let (right_id, right_rect, right_kind) = pair[1];
                self.process_pair(PairInfo {
                    container_id,
                    parent_rect,
                    left_id,
                    left_rect,
                    left_kind,
                    right_id,
                    right_rect,
                    right_kind,
                });
            }
        }

        fn run(mut self) -> Vec<ShareUpdate> {
            let linear_ids: Vec<_> = self
                .tree
                .tiles
                .iter()
                .filter_map(|(id, tile)| {
                    if matches!(tile, Tile::Container(Container::Linear(_))) {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect();

            for container_id in linear_ids {
                self.process_container(container_id);
            }
            self.share_updates
        }
    }

    let ctx = GutterCtx {
        tree,
        ui,
        painter,
        gutter_width,
        mask_state,
        share_updates: Vec::new(),
    };
    ctx.run()
}

// Apply gutter-driven share adjustments back onto the stored tree.
pub fn apply_share_updates(tree: &mut egui_tiles::Tree<Pane>, updates: &[ShareUpdate]) {
    for (container_id, left_id, right_id, left_share, right_share) in updates {
        if let Some(Tile::Container(Container::Linear(linear))) = tree.tiles.get_mut(*container_id)
        {
            linear.shares.set_share(*left_id, *left_share);
            linear.shares.set_share(*right_id, *right_share);
        }
    }
}

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
    ui.allocate_new_ui(UiBuilder::new().max_rect(rect), add_contents)
        .inner
}

pub fn tile_is_sidebar(tiles: &Tiles<Pane>, tile_id: TileId) -> bool {
    match tiles.get(tile_id) {
        Some(Tile::Pane(pane)) => pane.is_sidebar(),
        Some(Tile::Container(Container::Tabs(tabs))) => {
            !tabs.children.is_empty()
                && tabs
                    .children
                    .iter()
                    .all(|child| tile_is_sidebar(tiles, *child))
        }
        Some(Tile::Container(Container::Linear(linear))) => {
            !linear.children.is_empty()
                && linear
                    .children
                    .iter()
                    .all(|child| tile_is_sidebar(tiles, *child))
        }
        Some(Tile::Container(Container::Grid(grid))) => {
            let mut any = false;
            let all = grid.children().all(|child| {
                any = true;
                tile_is_sidebar(tiles, *child)
            });
            any && all
        }
        _ => false,
    }
}

pub fn tabs_are_sidebar_only(tiles: &Tiles<Pane>, tabs: &egui_tiles::Tabs) -> bool {
    !tabs.children.is_empty()
        && tabs
            .children
            .iter()
            .all(|child| tile_is_sidebar(tiles, *child))
}

pub fn tab_title_visible(tiles: &Tiles<Pane>, tile_id: TileId) -> bool {
    !tile_is_sidebar(tiles, tile_id)
}

pub fn tab_add_visible(tiles: &Tiles<Pane>, tabs: &egui_tiles::Tabs) -> bool {
    let active_is_sidebar = tabs
        .active
        .is_some_and(|active| tile_is_sidebar(tiles, active));
    !(active_is_sidebar || tabs_are_sidebar_only(tiles, tabs))
}
