use action::InspectorAction;
use bevy::ecs::{
    system::{Res, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use egui::CornerRadius;
use smallvec::SmallVec;

use crate::ui::{
    InspectorAnchor, SelectedObject,
    colors::{self, get_scheme},
    tiles::TreeAction,
};

use self::{entity::InspectorEntity, graph::InspectorGraph, viewport::InspectorViewport};

use super::{WidgetSystem, WidgetSystemExt};

pub mod action;
pub mod entity;
pub mod graph;
pub mod viewport;

pub struct InspectorIcons {
    pub chart: egui::TextureId,
    pub add: egui::TextureId,
    pub subtract: egui::TextureId,
    pub setting: egui::TextureId,
    pub search: egui::TextureId,
}

fn empty_inspector_ui(ui: &mut egui::Ui) -> egui::Response {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            let text = egui::RichText::new("SELECT AN ENTITY OR TABLE TO INSPECT")
                .color(colors::with_opacity(colors::WHITE, 0.1));
            ui.add(egui::Label::new(text));
        },
    )
    .response
}

pub fn empty_inspector() -> impl egui::Widget {
    move |ui: &mut egui::Ui| empty_inspector_ui(ui)
}

#[derive(SystemParam)]
pub struct InspectorContent<'w> {
    inspector_anchor: ResMut<'w, InspectorAnchor>,
    selected_object: Res<'w, SelectedObject>,
}

impl WidgetSystem for InspectorContent<'_> {
    type Args = (InspectorIcons, bool);
    type Output = SmallVec<[TreeAction; 4]>;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let state_mut = state.get_mut(world);

        let (icons, is_side_panel) = args;

        let mut inspector_anchor = state_mut.inspector_anchor;
        let selected_object = state_mut.selected_object.to_owned();

        inspector_anchor.0 = if is_side_panel {
            Some(ui.max_rect().min)
        } else {
            None
        };
        ui.painter()
            .rect_filled(ui.max_rect(), CornerRadius::ZERO, get_scheme().bg_primary);

        egui::ScrollArea::vertical()
            .max_width(ui.available_width())
            .show(ui, |ui| {
                egui::Frame::NONE
                    .fill(get_scheme().bg_primary)
                    .inner_margin(16.0)
                    .show(ui, |ui| {
                        ui.vertical(|ui| match selected_object {
                            SelectedObject::None => {
                                ui.add(empty_inspector());
                                Default::default()
                            }
                            SelectedObject::Entity(pair) => ui.add_widget_with::<InspectorEntity>(
                                world,
                                "inspector_entity",
                                (icons, pair),
                            ),
                            SelectedObject::Viewport { camera, .. } => {
                                ui.add_widget_with::<InspectorViewport>(
                                    world,
                                    "inspector_viewport",
                                    (icons, camera),
                                );
                                Default::default()
                            }
                            SelectedObject::Graph { graph_id, .. } => {
                                ui.add_widget_with::<InspectorGraph>(
                                    world,
                                    "inspector_graph",
                                    (icons, graph_id),
                                );
                                Default::default()
                            }
                            SelectedObject::Action { action_id, .. } => {
                                ui.add_widget_with::<InspectorAction>(
                                    world,
                                    "inspector_action",
                                    action_id,
                                );
                                Default::default()
                            }
                        })
                    })
                    .inner
                    .inner
            })
            .inner
    }
}
