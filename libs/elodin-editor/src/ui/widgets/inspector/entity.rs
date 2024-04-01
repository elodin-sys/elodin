use std::fmt::Display;

use bevy::ecs::{
    event::EventWriter,
    system::{Res, ResMut},
};
use bevy_egui::egui::{self, emath, Align, Color32, Layout, RichText};

use conduit::{
    bevy::{ColumnPayloadMsg, ComponentValueMap},
    ndarray::{self, Dimension},
    query::MetadataStore,
    ser_de::ColumnValue,
    well_known::EntityMetadata,
    ColumnPayload, ComponentValue, ElementValueMut, EntityId,
};

use crate::ui::{
    colors::{self, with_opacity},
    tiles,
    utils::{self, MarginSides},
    widgets::{button::ImageButton, label::ELabel},
    GraphsState,
};

const SEPARATOR_SPACING: f32 = 32.0;
const LABEL_SPACING: f32 = 8.0;

#[derive(Default)]
pub struct ItemActions {
    create_graph: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn inspector(
    ui: &mut egui::Ui,
    metadata: &EntityMetadata,
    entity_id: EntityId,
    map: &mut ComponentValueMap,
    metadata_store: &Res<MetadataStore>,
    graphs_state: &mut ResMut<GraphsState>,
    tile_state: &mut ResMut<tiles::TileState>,
    icon_chart: egui::TextureId,
    column_payload_writer: &mut EventWriter<ColumnPayloadMsg>,
) {
    ui.add(
        ELabel::new(&metadata.name)
            .padding(egui::Margin::same(0.0).bottom(24.0))
            .bottom_stroke(ELabel::DEFAULT_STROKE)
            .margin(egui::Margin::same(0.0).bottom(26.0)),
    );

    let mono_font = egui::TextStyle::Monospace.resolve(ui.style_mut());
    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(0.0, 8.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("ENTITY ID")
                        .color(with_opacity(colors::PRIMARY_CREAME, 0.6))
                        .font(mono_font.clone()),
                );
                ui.vertical(|ui| {
                    ui.add_space(3.0);
                    ui.with_layout(egui::Layout::right_to_left(Align::Min), |ui| {
                        ui.label(
                            RichText::new(entity_id.0.to_string())
                                .color(colors::PRIMARY_CREAME)
                                .font(mono_font),
                        )
                    });
                });
            });
        });

    for (component_id, component_value) in map.0.iter_mut() {
        let label = utils::get_component_label(metadata_store, component_id);

        ui.add(egui::Separator::default().spacing(SEPARATOR_SPACING));

        let mut item_actions = ItemActions::default();

        let res = inspector_item_multi(
            ui,
            &label,
            component_value,
            LABEL_SPACING,
            icon_chart,
            &mut item_actions,
        );
        if res.changed() {
            if let Ok(payload) = ColumnPayload::try_from_value_iter(
                0,
                std::iter::once(ColumnValue {
                    entity_id,
                    value: component_value.clone(),
                }),
            ) {
                column_payload_writer.send(ColumnPayloadMsg {
                    component_id: *component_id,
                    component_type: component_value.ty(),
                    payload,
                });
            }
        }

        if item_actions.create_graph {
            let (graph_id, _) = graphs_state.get_or_create_graph(&None);
            let component_values = component_value
                .iter()
                .enumerate()
                .map(|_| (true, colors::get_random_color()))
                .collect::<Vec<(bool, egui::Color32)>>();

            graphs_state.insert_component(&graph_id, &entity_id, component_id, component_values);

            tile_state.create_graph_tile(graph_id);
        }
    }
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
        ui.add(egui::DragValue::new(value).max_decimals(4))
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

fn inspector_item_label_ui(
    ui: &mut egui::Ui,
    label: &str,
    icon_chart: egui::TextureId,
    item_actions: &mut ItemActions,
) -> egui::Response {
    egui::Frame::none()
        .outer_margin(egui::Margin::same(1.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let (label_rect, btn_rect) = utils::get_rects_from_relative_width(
                    ui.max_rect(),
                    0.8,
                    ui.spacing().interact_size.y,
                );

                ui.allocate_ui_at_rect(label_rect, |ui| {
                    let text = egui::RichText::new(label.to_string()).color(colors::PRIMARY_CREAME);
                    ui.add(egui::Label::new(text));
                });

                ui.allocate_ui_at_rect(btn_rect, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        let create_graph_btn = ui.add(
                            ImageButton::new(icon_chart)
                                .scale(1.1, 1.1)
                                .image_tint(colors::PRIMARY_CREAME)
                                .bg_color(colors::TRANSPARENT),
                        );

                        item_actions.create_graph = create_graph_btn.clicked();
                    });
                });
            })
        })
        .response
}

pub fn inspector_item_label<'a>(
    label: &'a str,
    icon_chart: egui::TextureId,
    item_actions: &'a mut ItemActions,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| inspector_item_label_ui(ui, label, icon_chart, item_actions)
}

fn inspector_item_multi(
    ui: &mut egui::Ui,
    label: &str,
    values: &mut ComponentValue,
    label_spacing: f32,
    icon_chart: egui::TextureId,
    item_actions: &mut ItemActions,
) -> egui::Response {
    let resp = ui.vertical(|ui| {
        ui.add(inspector_item_label(label, icon_chart, item_actions));

        ui.add_space(label_spacing);

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
            values
                .indexed_iter_mut()
                .fold(None, |res: Option<egui::Response>, (i, value)| {
                    let i = DimIndexFormat(i);
                    let label = format!("{i}");

                    let new_res = ui.add(inspector_item_value(&label, value, desired_size));
                    if let Some(res) = res {
                        Some(res | new_res)
                    } else {
                        Some(new_res)
                    }
                })
        })
    });
    if let Some(inner_resp) = resp.inner.inner {
        resp.response | inner_resp
    } else {
        resp.response
    }
}

pub struct DimIndexFormat(ndarray::Dim<ndarray::IxDynImpl>);
impl Display for DimIndexFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let view = self.0.as_array_view();
        for (i, x) in view.iter().enumerate() {
            write!(f, "{x}")?;
            if i + 1 < view.len() {
                write!(f, ".")?;
            }
        }
        Ok(())
    }
}
