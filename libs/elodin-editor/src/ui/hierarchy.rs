use crate::{EqlContext, ui::ComponentId};
use bevy::ecs::{
    system::{Query, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::Entity;
use bevy::prelude::Res;
use bevy_egui::egui;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2_bevy::{ComponentPathRegistry, EntityMap};
use impeller2_wkt::ComponentMetadata;
use std::collections::HashMap;

use crate::ui::{EntityFilter, EntityPair, SelectedObject, colors::get_scheme, utils};

use super::{inspector::entity::search, schematic::branch, widgets::WidgetSystem};

#[derive(SystemParam)]
pub struct HierarchyContent<'w, 's> {
    entity_filter: ResMut<'w, EntityFilter>,
    selected_object: ResMut<'w, SelectedObject>,
    entities: Query<'w, 's, (&'static ComponentId, Entity, &'static ComponentMetadata)>,
    eql_ctx: Res<'w, EqlContext>,
    entity_map: Res<'w, EntityMap>,
    path_reg: ResMut<'w, ComponentPathRegistry>,
}

pub struct HiearchyIcons {
    pub search: egui::TextureId,
    pub entity: egui::TextureId,
    pub chevron: egui::TextureId,
}

impl WidgetSystem for HierarchyContent<'_, '_> {
    type Args = HiearchyIcons;
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        icons: Self::Args,
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
            eql_ctx,
            entity_map,
        } = state.get_mut(world);

        let search_text = entity_filter.0.clone();
        header(ui, entity_filter, icons.search);
        entity_list(
            ui,
            &eql_ctx,
            &entity_map,
            &mut selected_object,
            &path_reg,
            &search_text,
            icons,
        );
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
    eql_ctx: &EqlContext,
    entity_map: &EntityMap,
    selected_object: &mut ResMut<SelectedObject>,
    path_reg: &ComponentPathRegistry,
    entity_filter: &str,
    icons: HiearchyIcons,
) -> egui::Response {
    let tree_rect = ui.max_rect();
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                let matcher = SkimMatcherV2::default().smart_case().use_cache(true);
                let (parts, trailing) =
                    filter_component_parts(&eql_ctx.0.component_parts, &matcher, &entity_filter);

                for (_, _, part) in parts {
                    component_part(
                        ui,
                        tree_rect,
                        &icons,
                        part,
                        entity_map,
                        &trailing,
                        &matcher,
                        selected_object,
                    );
                }

                ui.allocate_space(ui.available_size());
            })
        })
        .inner
        .response
}

fn component_part(
    ui: &mut egui::Ui,
    tree_rect: egui::Rect,
    icons: &HiearchyIcons,
    part: &eql::ComponentPart,
    entity_map: &EntityMap,
    filter: &str,
    matcher: &SkimMatcherV2,
    selected_object: &mut SelectedObject,
) {
    let selected = selected_object.is_entity_selected(part.id);
    let (filtered_entities, trailing) = filter_component_parts(&part.children, matcher, filter);
    let list_item = branch(
        ui,
        &part.name,
        icons.entity,
        icons.chevron,
        filtered_entities.is_empty(),
        tree_rect,
        selected,
        |ui| {
            for (_, _, part) in filtered_entities {
                component_part(
                    ui,
                    tree_rect,
                    icons,
                    part,
                    entity_map,
                    trailing,
                    matcher,
                    selected_object,
                );
            }
        },
    );

    if list_item.clicked() {
        let Some(entity) = entity_map.get(&part.id) else {
            return;
        };
        if let SelectedObject::Entity(prev) = selected_object {
            *selected_object = if prev.impeller == part.id {
                SelectedObject::None
            } else {
                SelectedObject::Entity(EntityPair {
                    bevy: *entity,
                    impeller: part.id,
                })
            };
        } else {
            *selected_object = SelectedObject::Entity(EntityPair {
                bevy: *entity,
                impeller: part.id,
            })
        }
    }
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

fn filter_component_parts<'a, 'b>(
    children: &'b HashMap<String, eql::ComponentPart>,
    matcher: &SkimMatcherV2,
    str: &'a str,
) -> (Vec<(i64, String, &'b eql::ComponentPart)>, &'a str) {
    let (start, trailing) = str.split_once(".").unwrap_or((str, ""));
    let mut children: Vec<_> = children
        .iter()
        .filter_map(|(name, child)| {
            if start.is_empty() {
                Some((0, name.to_string(), child))
            } else {
                let score = matcher.fuzzy_match(name, start)?;
                Some((score, name.to_string(), child))
            }
        })
        .collect();
    children.sort_by_key(|(score, _, child)| (*score, child.id.0));
    (children, trailing)
}
