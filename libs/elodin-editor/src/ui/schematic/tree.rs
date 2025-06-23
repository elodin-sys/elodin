use crate::ui::SelectedObject;
use crate::ui::colors::{ColorExt, get_scheme};
use crate::ui::inspector::entity::search;
use crate::ui::widgets::WidgetSystem;

use super::CurrentSchematic;
use bevy::ecs::entity::Entity;
use bevy::ecs::system::SystemParam;
use bevy::prelude::{Component, Query, ResMut};
use egui::collapsing_header::CollapsingState;
use egui::load::SizedTexture;
use impeller2_wkt::Panel;

#[derive(SystemParam)]
pub struct TreeWidget<'w, 's> {
    schematic: ResMut<'w, CurrentSchematic>,
    state: Query<'w, 's, &'static mut TreeWidgetState>,
    selected_object: ResMut<'w, SelectedObject>,
}

pub struct TreeIcons {
    pub chevron: egui::TextureId,
    pub search: egui::TextureId,
    pub viewport: egui::TextureId,
    pub plot: egui::TextureId,
    pub container: egui::TextureId,
}

#[derive(Component)]
pub struct TreeWidgetState {
    filter: String,
}

impl Default for TreeWidgetState {
    fn default() -> Self {
        Self {
            filter: String::new(),
        }
    }
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
                    impeller2_wkt::SchematicElem::Panel(p) => {
                        panel(ui, max_rect, &icons, p, &mut selected_object)
                    }
                    impeller2_wkt::SchematicElem::Object3d(object_3d) => {
                        let selected = if Some(object_3d.aux) == selected_object.entity() {
                            *selected_object != SelectedObject::None
                        } else {
                            false
                        };
                        let branch_res = branch(
                            ui,
                            &object_3d.eql,
                            icons.viewport,
                            icons.chevron,
                            true,
                            max_rect,
                            selected,
                            |_| {},
                        );
                        if branch_res.clicked() {
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
    };
    let children = p.children();
    let selected = if p.aux().copied() == selected_object.entity() {
        *selected_object != SelectedObject::None
    } else {
        false
    };
    let branch_res = branch(
        ui,
        p.label(),
        icon,
        icons.chevron,
        children.is_empty(),
        tree_rect,
        selected,
        |ui| {
            for child in children {
                panel(ui, tree_rect, icons, child, selected_object);
            }
        },
    );
    if branch_res.clicked() {
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
            _ => {}
        }
    }
}

pub fn branch(
    ui: &mut egui::Ui,
    label: &str,
    icon: egui::TextureId,
    chevron: egui::TextureId,
    leaf: bool,
    tree_rect: egui::Rect,
    selected: bool,
    content: impl FnOnce(&mut egui::Ui),
) -> egui::Response {
    let id = ui.next_auto_id();
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let chevron = SizedTexture::new(chevron, [18., 18.]);
    let icon = SizedTexture::new(icon, [12., 12.]);
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
            let chevron = chevron.clone();
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
                        response.flags = response.flags & !egui::response::Flags::CLICKED;
                        state.toggle(ui);
                    }
                }
                egui::Image::from_texture(chevron)
                    .tint(icon_color)
                    .rotate(openness * 3.14 / 2.0, egui::vec2(0.5, 0.5))
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
            response.with_new_rect(inner_rect)
        },
    );
    state.show_body_indented(&header_res.response, ui, content);
    header_res.inner
}
