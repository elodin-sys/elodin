use action::InspectorAction;
use bevy::ecs::{
    system::{SystemParam, SystemState},
    world::World,
};
use bevy::prelude::{Entity, Query};
use bevy_egui::egui;
use egui::CornerRadius;
use smallvec::SmallVec;

use super::widgets::{WidgetSystem, WidgetSystemExt};
use crate::ui::tiles::WindowState;
use crate::ui::{
    SelectedObject,
    colors::{self, get_scheme},
    tiles::TreeAction,
    tiles::sidebar::sidebar_content_ui,
};

pub mod action;
pub mod dashboard;
pub mod data_overview;
pub mod entity;
pub mod graph;
pub mod monitor;
pub mod object3d;
pub mod query_table;
pub mod viewport;

mod widgets;
pub use widgets::*;

use self::{
    dashboard::InspectorDashboardNode, data_overview::InspectorDataOverview,
    entity::InspectorEntity, graph::InspectorGraph, monitor::InspectorMonitor,
    object3d::InspectorObject3D, query_table::InspectorQueryTable, viewport::InspectorViewport,
};

pub struct InspectorIcons {
    pub chart: egui::TextureId,
    pub subtract: egui::TextureId,
    pub setting: egui::TextureId,
    pub search: egui::TextureId,
}

fn empty_inspector_ui(ui: &mut egui::Ui) -> egui::Response {
    ui.with_layout(
        egui::Layout::centered_and_justified(egui::Direction::TopDown),
        |ui| {
            let text = egui::RichText::new("SELECT AN ENTITY OR TABLE TO INSPECT")
                .color(colors::with_opacity(get_scheme().text_tertiary, 0.6));
            ui.add(egui::Label::new(text));
        },
    )
    .response
}

pub fn empty_inspector() -> impl egui::Widget {
    move |ui: &mut egui::Ui| empty_inspector_ui(ui)
}

#[derive(SystemParam)]
pub struct InspectorContent<'w, 's> {
    window_states: Query<'w, 's, &'static mut WindowState>,
}

impl WidgetSystem for InspectorContent<'_, '_> {
    type Args = (InspectorIcons, bool, Entity);
    type Output = SmallVec<[TreeAction; 4]>;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> Self::Output {
        let mut state_mut = state.get_mut(world);

        let (icons, is_side_panel, target_window) = args;
        let selected_object = {
            let Ok(mut window_state) = state_mut.window_states.get_mut(target_window) else {
                return Default::default();
            };
            let ui_state = &mut window_state.ui_state;
            ui_state.inspector_anchor.0 = if is_side_panel {
                Some(ui.max_rect().min)
            } else {
                None
            };
            ui_state.selected_object.clone()
        };
        ui.painter()
            .rect_filled(ui.max_rect(), CornerRadius::ZERO, get_scheme().bg_primary);

        sidebar_content_ui(ui, |ui| {
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
                                SelectedObject::Entity(pair) => ui
                                    .add_widget_with::<InspectorEntity>(
                                        world,
                                        "inspector_entity",
                                        (icons, pair),
                                    ),
                                SelectedObject::Viewport { camera, .. } => {
                                    ui.add_widget_with::<InspectorViewport>(
                                        world,
                                        "inspector_viewport",
                                        camera,
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
                                SelectedObject::QueryTable { table_id } => {
                                    ui.add_widget_with::<InspectorQueryTable>(
                                        world,
                                        "inspector_query_table",
                                        table_id,
                                    );
                                    Default::default()
                                }
                                SelectedObject::Monitor { monitor_id } => {
                                    ui.add_widget_with::<InspectorMonitor>(
                                        world,
                                        "inspector_monitor",
                                        monitor_id,
                                    );
                                    Default::default()
                                }
                                SelectedObject::DataOverview => {
                                    ui.add_widget_with::<InspectorDataOverview>(
                                        world,
                                        "inspector_data_overview",
                                        (),
                                    );
                                    Default::default()
                                }
                                SelectedObject::DataOverviewComponent { .. } => {
                                    ui.add_widget_with::<InspectorDataOverview>(
                                        world,
                                        "inspector_data_overview",
                                        (),
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
                                SelectedObject::Object3D { entity, .. } => {
                                    ui.add_widget_with::<InspectorObject3D>(
                                        world,
                                        "inspector_object3d",
                                        (icons, entity),
                                    );
                                    Default::default()
                                }
                                SelectedObject::DashboardNode { entity } => {
                                    ui.add_widget_with::<InspectorDashboardNode>(
                                        world,
                                        "inspector_dashboard_node",
                                        (entity, target_window),
                                    );
                                    Default::default()
                                }
                            })
                        })
                        .inner
                        .inner
                })
                .inner
        })
    }
}
