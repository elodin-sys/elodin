use bevy::{
    ecs::{
        event::EventWriter,
        query::{With, Without},
        system::{Commands, Query, Res, ResMut, SystemParam, SystemState},
        world::World,
    },
    render::view::Visibility,
};
use bevy_egui::egui;
use bevy_infinite_grid::InfiniteGrid;
use big_space::GridCell;
use conduit::{
    bevy::{ColumnPayloadMsg, MaxTick},
    query::MetadataStore,
};

use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        colors, tiles, CameraQuery, EntityData, GraphsState, InspectorAnchor, SelectedObject,
        SettingModalState, SidebarState,
    },
    MainCamera,
};

use super::{
    timeline::tagged_range::TaggedRanges, RootWidgetSystem, WidgetSystem, WidgetSystemExt,
};

pub mod entity;
pub mod graph;
pub mod viewport;

pub struct InspectorIcons {
    pub chart: egui::TextureId,
    pub add: egui::TextureId,
    pub subtract: egui::TextureId,
    pub setting: egui::TextureId,
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
pub struct Inspector<'w> {
    sidebar_state: ResMut<'w, SidebarState>,
}

impl WidgetSystem for Inspector<'_> {
    type Args = (InspectorIcons, f32);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, width) = args;
        let sidebar_state = state_mut.sidebar_state;

        egui::SidePanel::new(egui::panel::Side::Right, "inspector_bottom")
            .resizable(false)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                ..Default::default()
            })
            .exact_width(width)
            .show_animated_inside(ui, sidebar_state.right_open, |ui| {
                ui.add_widget_with::<InspectorContent>(world, "inspector_content", (icons, false));
            });
    }
}

impl RootWidgetSystem for Inspector<'_> {
    type Args = (InspectorIcons, f32);
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, width) = args;
        let sidebar_state = state_mut.sidebar_state;

        egui::SidePanel::new(egui::panel::Side::Right, "inspector_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                ..Default::default()
            })
            .min_width(width.min(1280.) * 0.15)
            .default_width(width.min(1280.) * 0.25)
            .max_width(width * 0.35)
            .show_animated(ctx, sidebar_state.right_open, |ui| {
                ui.add_widget_with::<InspectorContent>(world, "inspector_content", (icons, true));
            });
    }
}

#[derive(SystemParam)]
pub struct InspectorContent<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    graphs_state: ResMut<'w, GraphsState>,
    setting_modal_state: ResMut<'w, SettingModalState>,
    tagged_ranges: ResMut<'w, TaggedRanges>,
    max_tick: Res<'w, MaxTick>,
    tile_state: ResMut<'w, tiles::TileState>,
    metadata_store: Res<'w, MetadataStore>,
    commands: Commands<'w, 's>,
    camera_query: Query<'w, 's, CameraQuery, With<MainCamera>>,
    selected_object: ResMut<'w, SelectedObject>,
    entity_transform_query: Query<'w, 's, &'static GridCell<i128>, Without<MainCamera>>,
    column_payload_writer: EventWriter<'w, ColumnPayloadMsg>,
    grid_visibility: Query<'w, 's, &'static mut Visibility, With<InfiniteGrid>>,
    inspector_anchor: ResMut<'w, InspectorAnchor>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
}

impl WidgetSystem for InspectorContent<'_, '_> {
    type Args = (InspectorIcons, bool);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icons, is_side_panel) = args;

        let mut entities = state_mut.entities;
        let mut graphs_state = state_mut.graphs_state;
        let mut setting_modal_state = state_mut.setting_modal_state;
        let mut tagged_ranges = state_mut.tagged_ranges;
        let max_tick = state_mut.max_tick;
        let mut tile_state = state_mut.tile_state;
        let metadata_store = state_mut.metadata_store;
        let mut commands = state_mut.commands;
        let mut camera_query = state_mut.camera_query;
        let selected_object = state_mut.selected_object.as_ref();
        let entity_transform_query = state_mut.entity_transform_query;
        let mut column_payload_writer = state_mut.column_payload_writer;
        let mut grid_visibility = state_mut.grid_visibility;
        let mut inspector_anchor = state_mut.inspector_anchor;
        let mut render_layer_alloc = state_mut.render_layer_alloc;

        inspector_anchor.0 = if is_side_panel {
            Some(ui.max_rect().min)
        } else {
            None
        };

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Frame::none()
                .fill(colors::PRIMARY_SMOKE)
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.vertical(|ui| match selected_object {
                        SelectedObject::None => {
                            ui.add(empty_inspector());
                        }
                        SelectedObject::Entity(pair) => {
                            let Ok((entity_id, _, mut map, metadata)) = entities.get_mut(pair.bevy)
                            else {
                                ui.add(empty_inspector());
                                return;
                            };
                            entity::inspector(
                                ui,
                                metadata,
                                *entity_id,
                                map.as_mut(),
                                &metadata_store,
                                &mut graphs_state,
                                &mut tile_state,
                                icons.chart,
                                &mut column_payload_writer,
                                &mut commands,
                                &mut render_layer_alloc,
                            );
                        }
                        SelectedObject::Viewport { camera, .. } => {
                            let Ok(cam) = camera_query.get_mut(*camera) else {
                                ui.add(empty_inspector());
                                return;
                            };
                            viewport::inspector(
                                ui,
                                &entities,
                                cam,
                                &mut commands,
                                &entity_transform_query,
                                &mut grid_visibility,
                            );
                        }
                        SelectedObject::Graph {
                            label, graph_id, ..
                        } => {
                            graph::inspector(
                                ui,
                                graph_id,
                                label,
                                &entities,
                                &mut graphs_state,
                                &mut setting_modal_state,
                                &mut tagged_ranges,
                                max_tick,
                                &metadata_store,
                                icons,
                            );
                        }
                    })
                })
        });
    }
}
