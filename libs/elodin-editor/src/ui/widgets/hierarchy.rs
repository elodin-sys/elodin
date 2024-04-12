use bevy::ecs::{
    system::{Query, ResMut, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui;
use conduit::well_known::EntityMetadata;

use crate::ui::{
    colors::{self, EColor},
    utils, EntityData, EntityDataReadOnly, EntityFilter, EntityPair, SelectedObject, SidebarState,
};

use super::{RootWidgetSystem, WidgetSystem, WidgetSystemExt};

#[derive(SystemParam)]
pub struct Hierarchy<'w> {
    sidebar_state: ResMut<'w, SidebarState>,
}

impl WidgetSystem for Hierarchy<'_> {
    type Args = (egui::TextureId, f32);
    type Output = f32;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) -> f32 {
        let state_mut = state.get_mut(world);

        let (icon_search, width) = args;
        let sidebar_state = state_mut.sidebar_state;

        let outline = egui::SidePanel::new(egui::panel::Side::Left, "outline_bottom")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                inner_margin: egui::Margin::same(4.0),
                ..Default::default()
            })
            .min_width(width * 0.25)
            .default_width(width * 0.4)
            .max_width(width * 0.75)
            .show_animated_inside(ui, sidebar_state.left_open, |ui| {
                ui.add_widget_with::<HierarchyContent>(
                    world,
                    "hierarchy_content",
                    (icon_search, true),
                );

                ui.allocate_space(ui.available_size());
            });

        outline.map(|o| o.response.rect.width()).unwrap_or(0.0)
    }
}

impl RootWidgetSystem for Hierarchy<'_> {
    type Args = (egui::TextureId, f32);
    type Output = ();

    fn ctx_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ctx: &mut egui::Context,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icon_search, width) = args;
        let sidebar_state = state_mut.sidebar_state;

        egui::SidePanel::new(egui::panel::Side::Left, "outline_side")
            .resizable(true)
            .frame(egui::Frame {
                fill: colors::PRIMARY_SMOKE,
                stroke: egui::Stroke::new(1.0, colors::BORDER_GREY),
                inner_margin: egui::Margin::same(4.0),
                ..Default::default()
            })
            .min_width(width * 0.15)
            .default_width(width * 0.20)
            .max_width(width * 0.35)
            .show_animated(ctx, sidebar_state.left_open, |ui| {
                ui.add_widget_with::<HierarchyContent>(
                    world,
                    "hierarchy_content",
                    (icon_search, false),
                );
            });
    }
}

#[derive(SystemParam)]
pub struct HierarchyContent<'w, 's> {
    entity_filter: ResMut<'w, EntityFilter>,
    selected_object: ResMut<'w, SelectedObject>,
    entities: Query<'w, 's, EntityData<'static>>,
}

impl WidgetSystem for HierarchyContent<'_, '_> {
    type Args = (egui::TextureId, bool);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let state_mut = state.get_mut(world);

        let (icon_search, compact) = args;
        let entity_filter = state_mut.entity_filter;
        let mut selected_object = state_mut.selected_object;
        let entities = state_mut.entities;

        let search_text = entity_filter.0.clone();
        header(ui, entity_filter, icon_search, compact);
        entity_list(ui, &entities, &mut selected_object, &search_text);
    }
}

pub fn header(
    ui: &mut egui::Ui,
    mut entity_filter: ResMut<EntityFilter>,
    search_icon: egui::TextureId,
    compact: bool,
) -> egui::Response {
    ui.vertical(|ui| {
        egui::Frame::none()
            .outer_margin(egui::Margin::same(16.0))
            .stroke(egui::Stroke::new(1.0, colors::BORDER_GREY))
            .rounding(egui::Rounding::same(3.0))
            .inner_margin(egui::Margin::same(8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.style_mut().spacing.item_spacing = egui::vec2(8.0, 0.0);

                    ui.add(
                        egui::widgets::Image::new(egui::load::SizedTexture::new(
                            search_icon,
                            [ui.spacing().interact_size.y, ui.spacing().interact_size.y],
                        ))
                        .tint(colors::with_opacity(colors::PRIMARY_CREAME, 0.4)),
                    );

                    ui.add(egui::TextEdit::singleline(&mut entity_filter.0).frame(false));
                });
            });

        if !compact {
            ui.separator();

            egui::Frame::none()
                .inner_margin(egui::Margin::symmetric(16.0, 16.0))
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new("ENTITIES")
                                .color(colors::with_opacity(colors::PRIMARY_CREAME, 0.4)),
                        )
                        .wrap(false),
                    );
                });
        }
    })
    .response
}

pub fn entity_list(
    ui: &mut egui::Ui,
    entities: &Query<EntityData>,
    selected_object: &mut ResMut<SelectedObject>,
    entity_filter: &str,
) -> egui::Response {
    egui::ScrollArea::both()
        .show(ui, |ui| {
            ui.vertical(|ui| {
                // TODO: Improve filter & sorting efficiency
                let mut filtered_entities = entities
                    .into_iter()
                    .filter(|(_, _, _, metadata)| {
                        metadata
                            .name
                            .to_lowercase()
                            .contains(&entity_filter.to_lowercase())
                    })
                    .collect::<Vec<EntityDataReadOnly>>();
                filtered_entities.sort_by(|a, b| a.0 .0.cmp(&b.0 .0));

                for (entity_id, entity, _, metadata) in filtered_entities {
                    let selected = selected_object.is_entity_selected(*entity_id);
                    let list_item = ui.add(list_item(selected, metadata));

                    if list_item.clicked() {
                        if let SelectedObject::Entity(prev) = selected_object.as_ref() {
                            **selected_object = if prev.conduit == *entity_id {
                                SelectedObject::None
                            } else {
                                SelectedObject::Entity(EntityPair {
                                    bevy: entity,
                                    conduit: *entity_id,
                                })
                            };
                        } else {
                            **selected_object = SelectedObject::Entity(EntityPair {
                                bevy: entity,
                                conduit: *entity_id,
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

fn list_item_ui(ui: &mut egui::Ui, on: bool, metadata: &EntityMetadata) -> egui::Response {
    let image_tint = colors::WHITE;
    let image_tint_click = colors::PRIMARY_ONYX_5;
    let background_color = if on {
        colors::WHITE
    } else {
        colors::PRIMARY_SMOKE
    };
    let text_color = if on {
        colors::PRIMARY_SMOKE
    } else {
        colors::WHITE
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
        ui.painter()
            .rect(rect, visuals.rounding, background_color, visuals.bg_stroke);

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
            egui::Rounding::same(2.0),
            metadata.color.into_color32(),
            egui::Stroke::NONE,
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

pub fn list_item(on: bool, metadata: &EntityMetadata) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| list_item_ui(ui, on, metadata)
}
