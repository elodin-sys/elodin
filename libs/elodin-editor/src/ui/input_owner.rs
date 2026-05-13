use bevy::prelude::{Entity, Resource};
use bevy_egui::egui;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PointerOwner {
    #[default]
    None,
    Viewport {
        camera: Entity,
    },
    Graph {
        graph: Entity,
    },
    QueryPlot {
        graph: Entity,
    },
    NavGizmo {
        camera: Entity,
    },
    BlockedByUi {
        blocker: UiBlocker,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiBlocker {
    Monitor,
    Inspector,
    Timeline,
    TabBar,
    Popup,
    Modal,
    CommandPalette,
    OtherPanel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PointerOwnerPriority {
    Content,
    Panel,
    Overlay,
    Modal,
}

#[derive(Clone, Copy, Debug)]
pub struct PointerOwnerRegion {
    pub rect: egui::Rect,
    pub owner: PointerOwner,
    pub priority: PointerOwnerPriority,
    order: u64,
}

impl PointerOwnerRegion {
    fn contains(&self, pos: egui::Pos2) -> bool {
        self.rect.contains(pos)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowInputOwner {
    pub pointer_pos: Option<egui::Pos2>,
    pub owner: PointerOwner,
}

#[derive(Resource, Debug, Default)]
pub struct UiInputOwners {
    regions_by_window: HashMap<Entity, Vec<PointerOwnerRegion>>,
    owners_by_window: HashMap<Entity, WindowInputOwner>,
    next_order: u64,
}

impl UiInputOwners {
    pub fn begin_frame(&mut self) {
        self.regions_by_window.clear();
        self.owners_by_window.clear();
        self.next_order = 0;
    }

    pub fn begin_window(&mut self, window: Entity) {
        self.regions_by_window.remove(&window);
        self.owners_by_window.remove(&window);
        self.next_order = 0;
    }

    pub fn register_rect(
        &mut self,
        window: Entity,
        rect: egui::Rect,
        owner: PointerOwner,
        priority: PointerOwnerPriority,
    ) {
        if !is_usable_rect(rect) {
            return;
        }

        let order = self.next_order;
        self.next_order = self.next_order.saturating_add(1);
        self.regions_by_window
            .entry(window)
            .or_default()
            .push(PointerOwnerRegion {
                rect,
                owner,
                priority,
                order,
            });
    }

    pub fn register_content_rect(&mut self, window: Entity, rect: egui::Rect, owner: PointerOwner) {
        self.register_rect(window, rect, owner, PointerOwnerPriority::Content);
    }

    pub fn register_blocker_rect(
        &mut self,
        window: Entity,
        rect: egui::Rect,
        blocker: UiBlocker,
        priority: PointerOwnerPriority,
    ) {
        self.register_rect(
            window,
            rect,
            PointerOwner::BlockedByUi { blocker },
            priority,
        );
    }

    pub fn resolve_window(
        &mut self,
        window: Entity,
        pointer_pos: Option<egui::Pos2>,
    ) -> PointerOwner {
        let owner = pointer_pos
            .and_then(|pos| self.resolve_owner_at(window, pos))
            .unwrap_or(PointerOwner::None);

        self.owners_by_window
            .insert(window, WindowInputOwner { pointer_pos, owner });
        owner
    }

    pub fn resolve_owner_at(
        &self,
        window: Entity,
        pointer_pos: egui::Pos2,
    ) -> Option<PointerOwner> {
        self.regions_by_window.get(&window).and_then(|regions| {
            regions
                .iter()
                .filter(|region| region.contains(pointer_pos))
                .max_by_key(|region| (region.priority, region.order))
                .map(|region| region.owner)
        })
    }

    pub fn owner_for_window(&self, window: Entity) -> PointerOwner {
        self.owners_by_window
            .get(&window)
            .map(|owner| owner.owner)
            .unwrap_or(PointerOwner::None)
    }

    pub fn owner_at(&self, window: Entity, pointer_pos: egui::Pos2) -> PointerOwner {
        self.resolve_owner_at(window, pointer_pos)
            .unwrap_or(PointerOwner::None)
    }

    pub fn permits_viewport(&self, window: Entity, camera: Entity) -> bool {
        matches!(
            self.owner_for_window(window),
            PointerOwner::Viewport { camera: owner_camera } if owner_camera == camera
        )
    }

    pub fn permits_viewport_at(
        &self,
        window: Entity,
        camera: Entity,
        pointer_pos: egui::Pos2,
    ) -> bool {
        matches!(
            self.owner_at(window, pointer_pos),
            PointerOwner::Viewport { camera: owner_camera } if owner_camera == camera
        )
    }

    pub fn permits_graph(&self, window: Entity, graph: Entity) -> bool {
        matches!(
            self.owner_for_window(window),
            PointerOwner::Graph { graph: owner_graph }
                | PointerOwner::QueryPlot { graph: owner_graph } if owner_graph == graph
        )
    }

    pub fn permits_graph_at(&self, window: Entity, graph: Entity, pointer_pos: egui::Pos2) -> bool {
        matches!(
            self.owner_at(window, pointer_pos),
            PointerOwner::Graph { graph: owner_graph }
                | PointerOwner::QueryPlot { graph: owner_graph } if owner_graph == graph
        )
    }

    pub fn permits_nav_gizmo(&self, window: Entity, camera: Entity) -> bool {
        matches!(
            self.owner_for_window(window),
            PointerOwner::NavGizmo { camera: owner_camera } if owner_camera == camera
        )
    }

    pub fn is_blocked_by_ui(&self, window: Entity) -> bool {
        matches!(
            self.owner_for_window(window),
            PointerOwner::BlockedByUi { .. }
        )
    }
}

fn is_usable_rect(rect: egui::Rect) -> bool {
    rect.min.x.is_finite()
        && rect.min.y.is_finite()
        && rect.max.x.is_finite()
        && rect.max.y.is_finite()
        && rect.width() > 0.0
        && rect.height() > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(id: u64) -> Entity {
        Entity::from_bits(id)
    }

    fn rect(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
    }

    #[test]
    fn viewport_under_pointer_owns_input() {
        let window = entity(1);
        let camera = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Viewport { camera },
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(50.0, 50.0))),
            PointerOwner::Viewport { camera }
        );
        assert!(owners.permits_viewport(window, camera));
    }

    #[test]
    fn monitor_over_viewport_blocks_viewport_input() {
        let window = entity(1);
        let camera = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Viewport { camera },
        );
        owners.register_blocker_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            UiBlocker::Monitor,
            PointerOwnerPriority::Panel,
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(50.0, 50.0))),
            PointerOwner::BlockedByUi {
                blocker: UiBlocker::Monitor,
            }
        );
        assert!(!owners.permits_viewport(window, camera));
        assert!(owners.is_blocked_by_ui(window));
    }

    #[test]
    fn higher_priority_wins_even_if_registered_earlier() {
        let window = entity(1);
        let graph = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_blocker_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            UiBlocker::Popup,
            PointerOwnerPriority::Overlay,
        );
        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph { graph },
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(50.0, 50.0))),
            PointerOwner::BlockedByUi {
                blocker: UiBlocker::Popup,
            }
        );
        assert!(!owners.permits_graph(window, graph));
    }

    #[test]
    fn nav_gizmo_overlay_wins_over_underlying_viewport() {
        let window = entity(1);
        let camera = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 200.0, 200.0),
            PointerOwner::Viewport { camera },
        );
        owners.register_rect(
            window,
            rect(100.0, 0.0, 200.0, 100.0),
            PointerOwner::NavGizmo { camera },
            PointerOwnerPriority::Overlay,
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(150.0, 50.0))),
            PointerOwner::NavGizmo { camera }
        );
        assert!(owners.permits_nav_gizmo(window, camera));
        assert!(!owners.permits_viewport(window, camera));
    }

    #[test]
    fn later_region_wins_with_equal_priority() {
        let window = entity(1);
        let first = entity(2);
        let second = entity(3);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph { graph: first },
        );
        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph { graph: second },
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(50.0, 50.0))),
            PointerOwner::Graph { graph: second }
        );
    }

    #[test]
    fn outside_registered_regions_has_no_owner() {
        let window = entity(1);
        let camera = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Viewport { camera },
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(150.0, 150.0))),
            PointerOwner::None
        );
        assert!(!owners.permits_viewport(window, camera));
    }

    #[test]
    fn invalid_rects_are_ignored() {
        let window = entity(1);
        let graph = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(10.0, 10.0, 10.0, 20.0),
            PointerOwner::Graph { graph },
        );

        assert_eq!(
            owners.resolve_window(window, Some(egui::pos2(10.0, 15.0))),
            PointerOwner::None
        );
    }

    #[test]
    fn begin_frame_clears_previous_state() {
        let window = entity(1);
        let graph = entity(2);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph { graph },
        );
        owners.resolve_window(window, Some(egui::pos2(50.0, 50.0)));

        owners.begin_frame();

        assert_eq!(owners.owner_for_window(window), PointerOwner::None);
        assert_eq!(
            owners.resolve_owner_at(window, egui::pos2(50.0, 50.0)),
            None
        );
    }

    #[test]
    fn begin_window_clears_only_that_window() {
        let first_window = entity(1);
        let second_window = entity(2);
        let first_graph = entity(3);
        let second_graph = entity(4);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            first_window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph { graph: first_graph },
        );
        owners.register_content_rect(
            second_window,
            rect(0.0, 0.0, 100.0, 100.0),
            PointerOwner::Graph {
                graph: second_graph,
            },
        );
        owners.resolve_window(first_window, Some(egui::pos2(50.0, 50.0)));
        owners.resolve_window(second_window, Some(egui::pos2(50.0, 50.0)));

        owners.begin_window(first_window);

        assert_eq!(owners.owner_for_window(first_window), PointerOwner::None);
        assert_eq!(
            owners.resolve_owner_at(first_window, egui::pos2(50.0, 50.0)),
            None
        );
        assert_eq!(
            owners.owner_for_window(second_window),
            PointerOwner::Graph {
                graph: second_graph,
            }
        );
        assert_eq!(
            owners.resolve_owner_at(second_window, egui::pos2(50.0, 50.0)),
            Some(PointerOwner::Graph {
                graph: second_graph,
            })
        );
    }

    #[test]
    fn permits_at_uses_the_provided_position_without_resolving_window_state() {
        let window = entity(1);
        let graph = entity(2);
        let camera = entity(3);
        let mut owners = UiInputOwners::default();

        owners.register_content_rect(
            window,
            rect(0.0, 0.0, 100.0, 50.0),
            PointerOwner::Graph { graph },
        );
        owners.register_content_rect(
            window,
            rect(0.0, 50.0, 100.0, 100.0),
            PointerOwner::Viewport { camera },
        );

        assert!(owners.permits_graph_at(window, graph, egui::pos2(50.0, 25.0)));
        assert!(!owners.permits_graph_at(window, graph, egui::pos2(50.0, 75.0)));
        assert!(owners.permits_viewport_at(window, camera, egui::pos2(50.0, 75.0)));
        assert_eq!(owners.owner_for_window(window), PointerOwner::None);
    }
}
