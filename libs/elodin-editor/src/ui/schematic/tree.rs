use std::f32;

use crate::EqlContext;
use crate::ui::SelectedObject;
use crate::ui::colors::{ColorExt, get_scheme};
use crate::ui::dashboard::{DashboardNodePath, NodeUpdaterParams, spawn_node};
use crate::ui::inspector::dashboard::DashboardExt;
use crate::ui::inspector::search;
use crate::ui::widgets::WidgetSystem;

use super::CurrentSchematic;
use bevy::ecs::entity::Entity;
use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::system::{Commands, Res, SystemParam};
use bevy::prelude::{Component, Query, ResMut};
use egui::collapsing_header::CollapsingState;
use egui::load::SizedTexture;
use impeller2_wkt::{Dashboard, DashboardNode, Panel};
use smallvec::smallvec;

#[derive(SystemParam)]
pub struct TreeWidget<'w, 's> {
    schematic: ResMut<'w, CurrentSchematic>,
    state: Query<'w, 's, &'static mut TreeWidgetState>,
    selected_object: ResMut<'w, SelectedObject>,
    spawn_node_params: SpawnNodeParams<'w, 's>,
}

#[derive(SystemParam)]
struct SpawnNodeParams<'w, 's> {
    eql_ctx: Res<'w, EqlContext>,
    commands: Commands<'w, 's>,
    node_update_params: NodeUpdaterParams<'w, 's>,
    dashboards: Query<'w, 's, &'static mut Dashboard<Entity>>,
}

pub struct TreeIcons {
    pub chevron: egui::TextureId,
    pub search: egui::TextureId,
    pub viewport: egui::TextureId,
    pub plot: egui::TextureId,
    pub container: egui::TextureId,
    pub add: egui::TextureId,
}

#[derive(Component, Default)]
pub struct TreeWidgetState {
    filter: String,
}

impl WidgetSystem for TreeWidget<'_, '_> {
    type Args = (TreeIcons, Entity);

    type Output = ();

    fn ui_system(
        world: &mut bevy::ecs::world::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        (icons, entity): Self::Args,
    ) -> Self::Output {
        let TreeWidget {
            schematic,
            mut state,
            mut selected_object,
            mut spawn_node_params,
        } = state.get_mut(world);
        let Ok(mut tree_state) = state.get_mut(entity) else {
            return;
        };

        let max_rect = ui.max_rect();

        ui.painter()
            .rect_filled(max_rect, egui::CornerRadius::ZERO, get_scheme().bg_primary);

        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| search(ui, &mut tree_state.filter, icons.search));

        egui::ScrollArea::vertical().show(ui, |ui| {
            for elem in &schematic.elems {
                match elem {
                    impeller2_wkt::SchematicElem::Panel(p) => panel(
                        ui,
                        max_rect,
                        &icons,
                        p,
                        &mut selected_object,
                        &mut spawn_node_params,
                    ),
                    impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                        let selected = if Some(object_3d.aux) == selected_object.entity() {
                            *selected_object != SelectedObject::None
                        } else {
                            false
                        };
                        let branch_res = Branch::new(
                            object_3d.eql.clone(),
                            icons.viewport,
                            icons.chevron,
                            max_rect,
                        )
                        .leaf(true)
                        .selected(selected)
                        .show(ui, |_| {});
                        if branch_res.inner.clicked() {
                            *selected_object = SelectedObject::Object3D {
                                entity: object_3d.aux,
                            };
                        }
                    }
                    impeller2_wkt::SchematicElem::Line3d(_line3d) => {}
                }
            }
        });
    }
}

fn panel(
    ui: &mut egui::Ui,
    tree_rect: egui::Rect,
    icons: &TreeIcons,
    p: &Panel<Entity>,
    selected_object: &mut SelectedObject,
    spawn_node_params: &mut SpawnNodeParams,
) {
    let p = p.collapse();
    let icon = match p {
        Panel::Viewport(_) => icons.viewport,
        Panel::VSplit(_) | Panel::HSplit(_) => icons.container,
        Panel::Graph(_) => icons.plot,
        Panel::ComponentMonitor(_) => icons.viewport,
        Panel::ActionPane(_) => icons.viewport,
        Panel::QueryTable(_) => icons.viewport,
        Panel::QueryPlot(_) => icons.plot,
        Panel::Tabs(_) => icons.container,
        Panel::Inspector => icons.viewport,
        Panel::Hierarchy => icons.viewport,
        Panel::SchematicTree => icons.viewport,
        Panel::Dashboard(_) => icons.viewport,
    };
    let children = p.children();
    let selected = if p.aux().copied() == selected_object.entity() {
        *selected_object != SelectedObject::None
    } else {
        false
    };
    let leaf = if let Panel::Dashboard(d) = p {
        d.root.children.is_empty()
    } else {
        children.is_empty()
    };
    let mut branch = Branch::new(p.label().to_string(), icon, icons.chevron, tree_rect)
        .leaf(leaf)
        .selected(selected);
    if let Panel::Dashboard(_) = p {
        branch = branch.extra_icon(icons.add);
    }
    let branch_res = branch.show(ui, |ui| {
        if let Panel::Dashboard(d) = p {
            for (i, child) in d.root.children.iter().enumerate() {
                dashboard_node(
                    ui,
                    tree_rect,
                    child,
                    icons,
                    selected_object,
                    DashboardNodePath {
                        root: d.aux,
                        path: smallvec![i],
                    },
                    spawn_node_params,
                    d.aux,
                );
            }
        } else {
            for child in children {
                panel(
                    ui,
                    tree_rect,
                    icons,
                    child,
                    selected_object,
                    spawn_node_params,
                );
            }
        }
    });
    if branch_res.inner.clicked() {
        match p {
            Panel::Viewport(viewport) => {
                *selected_object = SelectedObject::Viewport {
                    camera: viewport.aux,
                }
            }
            Panel::Graph(graph) => {
                *selected_object = SelectedObject::Graph {
                    graph_id: graph.aux,
                }
            }
            Panel::QueryPlot(plot) => {
                *selected_object = SelectedObject::Graph { graph_id: plot.aux }
            }
            Panel::Dashboard(d) => {
                *selected_object = SelectedObject::DashboardNode { entity: d.aux }
            }
            _ => {}
        }
    }
    if branch_res.extra_clicked {
        if let Panel::Dashboard(d) = p {
            spawn_child_node(
                &DashboardNodePath {
                    root: d.aux,
                    path: smallvec![],
                },
                spawn_node_params,
                d.aux,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dashboard_node(
    ui: &mut egui::Ui,
    tree_rect: egui::Rect,
    node: &DashboardNode<Entity>,
    icons: &TreeIcons,
    selected_object: &mut SelectedObject,
    path: DashboardNodePath,
    spawn_node_params: &mut SpawnNodeParams,
    parent: Entity,
) {
    let children = &node.children;

    let selected = if Some(node.aux) == selected_object.entity() {
        *selected_object != SelectedObject::None
    } else {
        false
    };
    let branch_res = Branch::new(
        node.label.as_deref().unwrap_or("node").to_string(),
        icons.container,
        icons.chevron,
        tree_rect,
    )
    .leaf(children.is_empty())
    .extra_icon(icons.add)
    .selected(selected)
    .show(ui, |ui| {
        for (i, child) in children.iter().enumerate() {
            let mut path = path.clone();
            path.path.push(i);
            dashboard_node(
                ui,
                tree_rect,
                child,
                icons,
                selected_object,
                path,
                spawn_node_params,
                node.aux,
            );
        }
    });

    branch_res.inner.context_menu(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(4.0, 4.0);
        if ui.button("Add Child").clicked() {
            spawn_child_node(&path, spawn_node_params, parent);
            ui.close_menu();
        }
        if ui.button("Duplicate").clicked() {
            let Ok(mut dashboard) = spawn_node_params.dashboards.get_mut(path.root) else {
                ui.close_menu();
                return;
            };
            let mut parent_path = path.path.clone();
            parent_path.pop();
            let Some(parent_node) = dashboard.root.get_node_mut(&parent_path) else {
                ui.close_menu();
                return;
            };
            let parent_entity = parent_node.aux;
            let mut path = DashboardNodePath {
                root: path.root,
                path: parent_path,
            };
            path.path.push(parent_node.children.len());
            let mut commands = spawn_node_params.commands.spawn(ChildOf(parent_entity));
            if let Ok(child) = spawn_node(
                node,
                &spawn_node_params.eql_ctx.0,
                &mut commands,
                path.root,
                path.path,
                &spawn_node_params.node_update_params,
            ) {
                parent_node.children.push(child);
            }
            ui.close_menu();
        }
        if ui.button("Delete").clicked() {
            let Ok(mut dashboard) = spawn_node_params.dashboards.get_mut(path.root) else {
                ui.close_menu();
                return;
            };
            let Some(parent_node) = dashboard
                .root
                .get_node_mut(&path.path[..path.path.len().saturating_sub(1)])
            else {
                ui.close_menu();
                return;
            };
            let Some(index) = path.path.last() else {
                ui.close_menu();
                return;
            };
            let node = parent_node.children.remove(*index);
            if let Ok(mut e) = spawn_node_params.commands.get_entity(node.aux) {
                e.despawn();
            }
            if selected {
                *selected_object = SelectedObject::DashboardNode {
                    entity: parent_node.aux,
                };
            }
            ui.close_menu();
        }
    });

    if branch_res.extra_clicked {
        spawn_child_node(&path, spawn_node_params, parent);
    }

    if branch_res.inner.clicked() {
        *selected_object = SelectedObject::DashboardNode { entity: node.aux };
    }
}

fn spawn_child_node(
    path: &DashboardNodePath,
    spawn_node_params: &mut SpawnNodeParams,
    parent: Entity,
) {
    let mut path = path.clone();
    let SpawnNodeParams {
        eql_ctx,
        commands,
        node_update_params,
        dashboards,
    } = spawn_node_params;
    let mut commands = commands.spawn(ChildOf(parent));

    let Ok(mut dashboard) = dashboards.get_mut(path.root) else {
        return;
    };
    let Some(parent_node) = dashboard.root.get_node_mut(&path.path) else {
        return;
    };
    path.path.push(parent_node.children.len());
    if let Ok(child) = spawn_node::<()>(
        &Default::default(),
        &eql_ctx.0,
        &mut commands,
        path.root,
        path.path,
        node_update_params,
    ) {
        parent_node.children.push(child);
    }
}

pub struct Branch {
    pub label: String,
    pub icon: egui::TextureId,
    pub chevron: egui::TextureId,
    pub leaf: bool,
    pub tree_rect: egui::Rect,
    pub extra_icon: Option<egui::TextureId>,
    pub selected: bool,
}

impl Branch {
    pub fn new(
        label: String,
        icon: egui::TextureId,
        chevron: egui::TextureId,
        tree_rect: egui::Rect,
    ) -> Self {
        Self {
            label,
            icon,
            chevron,
            leaf: true,
            tree_rect,
            extra_icon: None,
            selected: false,
        }
    }

    pub fn leaf(mut self, leaf: bool) -> Self {
        self.leaf = leaf;
        self
    }

    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    pub fn extra_icon(mut self, icon: egui::TextureId) -> Self {
        self.extra_icon = Some(icon);
        self
    }

    pub fn show(self, ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) -> BranchResponse {
        let Branch {
            label,
            icon,
            chevron,
            leaf,
            tree_rect,
            selected,
            extra_icon,
        } = self;

        let id = ui.make_persistent_id((&label, &selected));
        let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, true);
        let chevron = SizedTexture::new(chevron, [18., 18.]);
        let icon = SizedTexture::new(icon, [12., 12.]);
        let extra_icon = extra_icon.map(|icon| SizedTexture::new(icon, [12., 12.]));
        ui.spacing_mut().icon_width_inner = 12.0;
        ui.visuals_mut().indent_has_left_vline = false;
        let scheme = get_scheme();
        let header_res = ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), 36.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                let inner_rect = ui.max_rect();
                let mut response = ui.interact(
                    egui::Rect::from_min_max(
                        egui::pos2(tree_rect.min.x, inner_rect.min.y),
                        egui::pos2(tree_rect.max.x, inner_rect.max.y),
                    ),
                    ui.next_auto_id(),
                    egui::Sense::CLICK | egui::Sense::HOVER,
                );
                ui.advance_cursor_after_rect(inner_rect);
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(tree_rect.min.x, inner_rect.min.y),
                        egui::pos2(tree_rect.max.x, inner_rect.max.y),
                    ),
                    egui::CornerRadius::ZERO,
                    if selected {
                        scheme.text_primary
                    } else if response.is_pointer_button_down_on() {
                        scheme.bg_secondary
                    } else if response.hovered() {
                        scheme.bg_secondary.opacity(0.75)
                    } else {
                        scheme.bg_primary
                    },
                );
                let openness = state.openness(ui.ctx());
                let icon_color = if selected {
                    scheme.bg_primary
                } else {
                    scheme.icon_primary
                };
                let chevron_end = if !leaf {
                    let chevron_rect = egui::Rect::from_center_size(
                        inner_rect.left_center() + egui::vec2(6.0 + 8.0, 0.0),
                        egui::vec2(18., 18.),
                    );

                    if let Some(pos) = response.interact_pointer_pos() {
                        if response.clicked() && chevron_rect.contains(pos) {
                            response.flags &= !egui::response::Flags::CLICKED;
                            state.toggle(ui);
                        }
                    }
                    egui::Image::from_texture(chevron)
                        .tint(icon_color)
                        .rotate(openness * f32::consts::FRAC_PI_2, egui::vec2(0.5, 0.5))
                        .paint_at(ui, chevron_rect);

                    chevron_rect.right_center()
                } else {
                    inner_rect.left_center() + egui::vec2(6.0, 0.0)
                };
                let icon_rect = egui::Rect::from_center_size(
                    chevron_end + egui::vec2(8.0 + 6.0, 0.0),
                    egui::vec2(12., 12.),
                );
                egui::Image::from_texture(icon)
                    .tint(icon_color)
                    .paint_at(ui, icon_rect);

                let mut font_id = egui::TextStyle::Button.resolve(ui.style());
                font_id.size = 12.0;
                ui.painter().text(
                    icon_rect.right_center() + egui::vec2(8.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    label,
                    font_id,
                    if selected {
                        scheme.bg_primary
                    } else {
                        scheme.text_primary
                    },
                );

                let mut extra_clicked = false;
                if let Some(extra_icon) = extra_icon {
                    let extra_rect = egui::Rect::from_center_size(
                        response.rect.right_center() - egui::vec2(6.0 + 8.0, 0.0),
                        egui::vec2(12., 12.),
                    );
                    egui::Image::from_texture(extra_icon)
                        .tint(icon_color)
                        .paint_at(ui, extra_rect);

                    if let Some(pos) = response.interact_pointer_pos() {
                        if response.clicked() && extra_rect.contains(pos) {
                            response.flags &= !egui::response::Flags::CLICKED;
                            extra_clicked = true;
                        }
                    }
                }

                BranchResponse {
                    inner: response.with_new_rect(inner_rect),
                    extra_clicked,
                }
            },
        );
        state.show_body_indented(&header_res.response, ui, content);
        header_res.inner
    }
}

pub struct BranchResponse {
    pub inner: egui::Response,
    pub extra_clicked: bool,
}
