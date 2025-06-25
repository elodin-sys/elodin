use crate::EqlContext;
use bevy::ecs::{
    system::{ResMut, SystemParam, SystemState},
    world::World,
};
use bevy::prelude::Res;
use bevy_egui::egui;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use impeller2_bevy::EntityMap;
use std::collections::HashMap;

use crate::ui::{EntityFilter, EntityPair, SelectedObject, colors::get_scheme};

use super::{inspector::search, schematic::Branch, widgets::WidgetSystem};

#[derive(SystemParam)]
pub struct HierarchyContent<'w> {
    entity_filter: ResMut<'w, EntityFilter>,
    selected_object: ResMut<'w, SelectedObject>,
    eql_ctx: Res<'w, EqlContext>,
    entity_map: Res<'w, EntityMap>,
}

pub struct Hierarchy {
    pub search: egui::TextureId,
    pub entity: egui::TextureId,
    pub chevron: egui::TextureId,
}

impl WidgetSystem for HierarchyContent<'_> {
    type Args = Hierarchy;
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
    entity_filter: &str,
    icons: Hierarchy,
) -> egui::Response {
    let tree_rect = ui.max_rect();
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                let matcher = SkimMatcherV2::default().smart_case().use_cache(true);
                let (parts, trailing) =
                    filter_component_parts(&eql_ctx.0.component_parts, &matcher, entity_filter);

                for (_, _, part) in parts {
                    component_part(
                        ui,
                        tree_rect,
                        &icons,
                        part,
                        entity_map,
                        trailing,
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

#[allow(clippy::too_many_arguments)]
fn component_part(
    ui: &mut egui::Ui,
    tree_rect: egui::Rect,
    icons: &Hierarchy,
    part: &eql::ComponentPart,
    entity_map: &EntityMap,
    filter: &str,
    matcher: &SkimMatcherV2,
    selected_object: &mut SelectedObject,
) {
    let selected = selected_object.is_entity_selected(part.id);
    let (filtered_entities, trailing) = filter_component_parts(&part.children, matcher, filter);
    let list_item = Branch::new(part.name.clone(), icons.entity, icons.chevron, tree_rect)
        .selected(selected)
        .leaf(filtered_entities.is_empty())
        .show(ui, |ui| {
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
        });

    if list_item.inner.clicked() {
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
