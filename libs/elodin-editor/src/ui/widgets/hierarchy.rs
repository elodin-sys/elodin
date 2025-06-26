use crate::ui::ComponentId;
use bevy::ecs::{
    system::{Query, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::Entity;
use bevy_egui::egui;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2_bevy::ComponentPathRegistry;
use impeller2_wkt::ComponentMetadata;

use crate::ui::{EntityFilter, EntityPair, SelectedObject, colors::get_scheme, utils};

use super::{WidgetSystem, inspector::entity::search};

#[derive(SystemParam)]
pub struct HierarchyContent<'w, 's> {
    entity_filter: ResMut<'w, EntityFilter>,
    selected_object: ResMut<'w, SelectedObject>,
    entities: Query<'w, 's, (&'static ComponentId, Entity, &'static ComponentMetadata)>,
    path_reg: ResMut<'w, ComponentPathRegistry>,
}

impl WidgetSystem for HierarchyContent<'_, '_> {
    type Args = egui::TextureId;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        ui.painter().rect_filled(
            ui.max_rect(),
            egui::CornerRadius::ZERO,
            get_scheme().bg_primary,
        );

        let HierarchyContent {
            entity_filter,
            mut selected_object,
            entities,
            path_reg,
        } = state.get_mut(world);

        let icon_search = args;

        let search_text = entity_filter.0.clone();
        header(ui, entity_filter, icon_search);
        entity_list(ui, &entities, &mut selected_object, &path_reg, &search_text);
    }
}

pub fn header(
    ui: &mut egui::Ui,
    mut entity_filter: ResMut<EntityFilter>,
    search_icon: egui::TextureId,
) -> egui::Response {
    egui::Frame::NONE
        .inner_margin(egui::Margin::symmetric(8, 8))
        .show(ui, |ui| {
            search(ui, &mut entity_filter.0, search_icon);
        })
        .response
}

pub fn entity_list(
    ui: &mut egui::Ui,
    entities: &Query<(&'static ComponentId, Entity, &'static ComponentMetadata)>,
    selected_object: &mut ResMut<SelectedObject>,
    path_reg: &ComponentPathRegistry,
    entity_filter: &str,
) -> egui::Response {
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                let matcher = SkimMatcherV2::default().smart_case().use_cache(true);
                // TODO: Improve filter & sorting efficiency
                let mut filtered_entities = entities
                    .into_iter()
                    .filter_map(|(id, entity, metadata)| {
                        let path = path_reg.0.get(id)?;
                        if !path.is_top_level() {
                            return None;
                        };
                        if entity_filter.is_empty() {
                            Some((id.0 as i64, id, entity, metadata))
                        } else {
                            matcher
                                .fuzzy_match(&metadata.name, entity_filter)
                                .map(|score| (score, id, entity, metadata))
                        }
                    })
                    .collect::<Vec<_>>();
                filtered_entities.sort_by(|a, b| b.0.cmp(&a.0));

                for (_, entity_id, entity, metadata) in filtered_entities {
                    let selected = selected_object.is_entity_selected(*entity_id);
                    let list_item = ui.add(list_item(selected, metadata));

                    if list_item.clicked() {
                        if let SelectedObject::Entity(prev) = selected_object.as_ref() {
                            **selected_object = if prev.impeller == *entity_id {
                                SelectedObject::None
                            } else {
                                SelectedObject::Entity(EntityPair {
                                    bevy: entity,
                                    impeller: *entity_id,
                                })
                            };
                        } else {
                            **selected_object = SelectedObject::Entity(EntityPair {
                                bevy: entity,
                                impeller: *entity_id,
                            })
                        }
                    }
                }

                ui.allocate_space(ui.available_size());
            })
        })
        .inner
        .response
}

fn list_item_ui(ui: &mut egui::Ui, on: bool, metadata: &ComponentMetadata) -> egui::Response {
    let image_tint = get_scheme().text_primary;
    let image_tint_click = get_scheme().text_secondary;
    let background_color = if on {
        get_scheme().text_primary
    } else {
        get_scheme().bg_primary
    };
    let text_color = if on {
        get_scheme().bg_primary
    } else {
        get_scheme().text_primary
    };

    // Set widget size and allocate space
    let height_scale = 2.0;
    let desired_size = egui::vec2(
        ui.available_width(),
        ui.spacing().interact_size.y * height_scale,
    );
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    // Paint the UI
    if ui.is_rect_visible(rect) {
        let outer_margin = 1.0;
        let rect = rect.shrink(outer_margin);

        let style = ui.style_mut();
        let font_id = egui::TextStyle::Button.resolve(style);
        style.visuals.widgets.inactive.bg_fill = image_tint;
        style.visuals.widgets.hovered.bg_fill = image_tint;
        style.visuals.widgets.active.bg_fill = image_tint_click;

        let visuals = ui.style().interact(&response);

        // Background
        ui.painter().rect(
            rect,
            visuals.corner_radius,
            background_color,
            visuals.bg_stroke,
            egui::StrokeKind::Middle,
        );

        // Icon
        let left_center_pos = rect.left_center();
        let horizontal_margin = 20.0;
        let icon_side = 8.0;
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(left_center_pos.x + horizontal_margin, left_center_pos.y),
            egui::vec2(icon_side, icon_side),
        );
        ui.painter().rect(
            icon_rect,
            egui::CornerRadius::same(2),
            get_scheme().blue,
            egui::Stroke::NONE,
            egui::StrokeKind::Middle,
        );

        // Label
        let left_text_margin = horizontal_margin + 12.0 + icon_side;

        let layout_job = utils::get_galley_layout_job(
            metadata.name.to_owned(),
            ui.available_width() - left_text_margin - horizontal_margin,
            font_id,
            text_color,
        );
        let galley = ui.fonts(|f| f.layout_job(layout_job));
        let text_rect = egui::Align2::LEFT_CENTER.anchor_rect(egui::Rect::from_min_size(
            egui::pos2(left_center_pos.x + left_text_margin, left_center_pos.y),
            galley.size(),
        ));
        ui.painter().galley(text_rect.min, galley, text_color);
    }

    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

pub fn list_item(on: bool, metadata: &ComponentMetadata) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| list_item_ui(ui, on, metadata)
}
