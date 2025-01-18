use std::{collections::BTreeMap, fmt::Display};

use bevy::ecs::{
    system::{Query, Res, ResMut, Resource, SystemParam, SystemState},
    world::World,
};
use bevy_egui::egui::{self, emath, Align, Color32, Layout, RichText};
use fuzzy_matcher::{skim::SkimMatcherV2, FuzzyMatcher};
use impeller2_bevy::{ComponentMetadataRegistry, ComponentValue, ElementValueMut};
use smallvec::SmallVec;

use crate::{
    plugins::navigation_gizmo::RenderLayerAlloc,
    ui::{
        colors::{self, with_opacity},
        tiles,
        utils::{format_num, MarginSides},
        widgets::{
            label,
            plot::{default_component_values, GraphBundle},
            WidgetSystem,
        },
        EntityData, EntityPair,
    },
};

use super::{empty_inspector, InspectorIcons};

#[derive(SystemParam)]
pub struct InspectorEntity<'w, 's> {
    entities: Query<'w, 's, EntityData<'static>>,
    tile_state: ResMut<'w, tiles::TileState>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
    render_layer_alloc: ResMut<'w, RenderLayerAlloc>,
    filter: ResMut<'w, ComponentFilter>,
}

impl WidgetSystem for InspectorEntity<'_, '_> {
    type Args = (InspectorIcons, EntityPair);
    type Output = ();

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        args: Self::Args,
    ) {
        let mut state_mut = state.get_mut(world);

        let (icons, pair) = args;

        let mut entities = state_mut.entities;
        let mut tile_state = state_mut.tile_state;
        let metadata_store = state_mut.metadata_store;
        let mut render_layer_alloc = state_mut.render_layer_alloc;
        let Ok((entity_id, _, mut component_value_map, metadata)) = entities.get_mut(pair.bevy)
        else {
            ui.add(empty_inspector());
            return;
        };

        let icon_chart = icons.chart;
        let entity_id = *entity_id;

        let mono_font = egui::TextStyle::Monospace.resolve(ui.style_mut());
        egui::Frame::none()
            .inner_margin(egui::Margin::ZERO.top(8.0))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        label::ELabel::new(&metadata.name)
                            .padding(egui::Margin::same(0.0).bottom(24.0))
                            .bottom_stroke(label::ELabel::DEFAULT_STROKE)
                            .margin(egui::Margin::same(0.0).bottom(26.0)),
                    );

                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(
                            RichText::new(entity_id.0.to_string())
                                .color(colors::PRIMARY_CREAME)
                                .font(mono_font.clone()),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("ENTITY ID")
                                .color(with_opacity(colors::PRIMARY_CREAME, 0.6))
                                .font(mono_font.clone()),
                        );
                    });
                });
            });

        search(ui, state_mut.filter.as_mut(), icons.search);

        let matcher = SkimMatcherV2::default().smart_case().use_cache(true);

        let mut components = component_value_map
            .0
            .keys()
            .filter_map(|id| {
                let metadata = metadata_store.get_metadata(id)?;
                let priority = metadata.metadata.priority();
                Some((*id, priority, metadata))
            })
            .filter_map(|(id, priority, metadata)| {
                if state_mut.filter.0.is_empty() {
                    Some((id, priority, metadata))
                } else {
                    matcher
                        .fuzzy_match(&metadata.name, &state_mut.filter.0)
                        .map(|score| (id, score, metadata))
                }
            })
            .filter(|(_, _, metadata)| !metadata.asset)
            .filter(|(_, priority, _)| *priority >= 0)
            .collect::<Vec<_>>();
        components.sort_by_key(|(id, priority, _)| (*priority, *id));

        ui.add_space(10.0);

        for (component_id, _, metadata) in components.into_iter().rev() {
            let component_value = component_value_map.0.get_mut(&component_id).unwrap();
            let label = metadata.name.clone();
            let element_names = metadata.element_names();

            ui.add(egui::Separator::default().spacing(32.0));

            let mut create_graph = false;

            let res = inspector_item_multi(
                ui,
                &label,
                element_names,
                component_value,
                icon_chart,
                &mut create_graph,
            );
            if res.changed() {
                // if let Ok(payload) = ColumnPayload::try_from_value_iter(
                //     0,
                //     std::iter::once(ColumnValue {
                //         entity_id,
                //         value: component_value.clone(),
                //     }),
                // ) {
                //     column_payload_writer.send(ColumnPayloadMsg {
                //         component_name: metadata.name.to_string(),
                //         component_type: component_value.ty(),
                //         payload,
                //     });
                // }
            }

            if create_graph {
                let values = default_component_values(&entity_id, &component_id, component_value);
                let entities = BTreeMap::from_iter(std::iter::once((
                    entity_id,
                    BTreeMap::from_iter(std::iter::once((component_id, values.clone()))),
                )));
                let bundle = GraphBundle::new(&mut render_layer_alloc, entities, None);
                tile_state.create_graph_tile(None, bundle);
            }
        }
    }
}

#[derive(Resource, Default)]
pub struct ComponentFilter(pub String);

pub fn search(
    ui: &mut egui::Ui,
    component_filter: &mut ComponentFilter,
    search_icon: egui::TextureId,
) -> egui::Response {
    ui.vertical(|ui| {
        egui::Frame::none()
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

                    ui.add(egui::TextEdit::singleline(&mut component_filter.0).frame(false));
                });
            });
    })
    .response
}

fn inspector_item_value_ui(
    ui: &mut egui::Ui,
    label: &str,
    value: ElementValueMut<'_>,
    size: egui::Vec2,
) -> egui::Response {
    let label_color = colors::with_opacity(colors::PRIMARY_CREAME, 0.4);
    ui.allocate_ui_with_layout(
        size,
        Layout::left_to_right(Align::Center).with_main_justify(true),
        |ui| {
            ui.with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                ui.vertical(|ui| {
                    ui.add_space(3.0);
                    ui.style_mut().override_font_id =
                        Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
                    ui.label(RichText::new(label).color(label_color))
                })
            });

            match value {
                ElementValueMut::U8(n) => comp_drag_value(ui, n),
                ElementValueMut::U16(n) => comp_drag_value(ui, n),
                ElementValueMut::U32(n) => comp_drag_value(ui, n),
                ElementValueMut::U64(n) => comp_drag_value(ui, n),
                ElementValueMut::I8(n) => comp_drag_value(ui, n),
                ElementValueMut::I16(n) => comp_drag_value(ui, n),
                ElementValueMut::I32(n) => comp_drag_value(ui, n),
                ElementValueMut::I64(n) => comp_drag_value(ui, n),
                ElementValueMut::F64(n) => comp_drag_value(ui, n),
                ElementValueMut::F32(n) => comp_drag_value(ui, n),
                ElementValueMut::Bool(b) => ui.checkbox(b, ""),
            }
        },
    )
    .inner
}

fn comp_drag_value<Num: emath::Numeric>(ui: &mut egui::Ui, value: &mut Num) -> egui::Response {
    ui.with_layout(Layout::top_down_justified(Align::RIGHT), |ui| {
        ui.style_mut().visuals.widgets.hovered.weak_bg_fill = Color32::TRANSPARENT;
        ui.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
        ui.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
        ui.style_mut().override_font_id = Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
        ui.add(egui::DragValue::new(value).custom_formatter(|v, _| format_num(v)))
    })
    .inner
}

pub fn inspector_item_value<'a>(
    label: &'a str,
    value: ElementValueMut<'a>,
    size: egui::Vec2,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| inspector_item_value_ui(ui, label, value, size)
}

fn inspector_item_multi(
    ui: &mut egui::Ui,
    label: &str,
    element_names: &str,
    values: &mut ComponentValue,
    icon_chart: egui::TextureId,
    create_graph: &mut bool,
) -> egui::Response {
    let element_names = element_names
        .split(',')
        .filter(|s| !s.is_empty())
        .map(Option::Some)
        .chain(std::iter::repeat(None));
    let resp = ui.vertical(|ui| {
        let [graph_clicked] = label::label_with_buttons(
            ui,
            [icon_chart],
            label,
            colors::PRIMARY_CREAME,
            egui::Margin::symmetric(0.0, 4.0).bottom(12.0),
        );
        *create_graph = graph_clicked;

        let item_spacing = egui::vec2(8.0, 8.0);

        let line_width = ui.available_size().x;
        let line_height = ui.spacing().interact_size.y * 1.4;

        let item_width_min = ui.spacing().interact_size.x * 2.4;
        let items_per_line = (line_width / item_width_min).floor();

        let necessary_spacing = (items_per_line - 1.0) * item_spacing.x;
        let item_width = (line_width - necessary_spacing) / items_per_line;

        let desired_size = egui::vec2(item_width - 1.0, line_height);

        ui.horizontal_wrapped(|ui| {
            ui.style_mut().spacing.item_spacing = item_spacing;
            values.indexed_iter_mut().zip(element_names).fold(
                None,
                |res: Option<egui::Response>, ((dim_i, value), element_name)| {
                    let label = element_name
                        .map(|name| name.to_string())
                        .unwrap_or_else(|| format!("{dim_i:?}"));

                    let new_res = ui.add(inspector_item_value(&label, value, desired_size));
                    if let Some(res) = res {
                        Some(res | new_res)
                    } else {
                        Some(new_res)
                    }
                },
            )
        })
    });
    if let Some(inner_resp) = resp.inner.inner {
        resp.response | inner_resp
    } else {
        resp.response
    }
}

pub struct DimIndexFormat(SmallVec<[usize; 4]>);
impl Display for DimIndexFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.0.iter().enumerate() {
            write!(f, "{x}")?;
            if i + 1 < self.0.len() {
                write!(f, ".")?;
            }
        }
        Ok(())
    }
}
