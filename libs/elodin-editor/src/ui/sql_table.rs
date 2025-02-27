use std::time::Instant;

use arrow::{
    error::ArrowError,
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, In, Query},
};
use egui::{Color32, CornerRadius, RichText, Stroke};
use egui_extras::{Column, TableBuilder};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use super::{
    colors,
    widgets::{button::EButton, WidgetSystem},
};

#[derive(Clone)]
pub struct SQLTablePane {
    pub entity: Entity,
}

#[derive(Component, Default)]
pub struct SqlTable {
    current_query: String,
    state: SqlTableState,
}

#[derive(Default)]
pub enum SqlTableState {
    #[default]
    None,
    Requested(Instant),
    Results(Vec<RecordBatch>),
    Error(ErrorResponse),
}

#[derive(SystemParam)]
pub struct SqlTableWidget<'w, 's> {
    states: Query<'w, 's, &'static mut SqlTable>,
    commands: Commands<'w, 's>,
}

impl WidgetSystem for SqlTableWidget<'_, '_> {
    type Args = SQLTablePane;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        SQLTablePane { entity }: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut table) = state.states.get_mut(entity) else {
            return;
        };
        egui::Frame::NONE
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    let style = ui.style_mut();
                    style.visuals.widgets.active.corner_radius = CornerRadius::ZERO;
                    style.visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
                    style.visuals.widgets.open.corner_radius = CornerRadius::ZERO;

                    style.visuals.widgets.active.fg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
                    style.visuals.widgets.active.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
                    style.visuals.widgets.hovered.fg_stroke =
                        Stroke::new(0.0, Color32::TRANSPARENT);
                    style.visuals.widgets.hovered.bg_stroke =
                        Stroke::new(0.0, Color32::TRANSPARENT);
                    style.visuals.widgets.open.fg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
                    style.visuals.widgets.open.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);

                    style.spacing.button_padding = [16.0, 16.0].into();

                    style.visuals.widgets.active.bg_fill = colors::SURFACE_SECONDARY;
                    style.visuals.widgets.open.bg_fill = colors::SURFACE_SECONDARY;
                    style.visuals.widgets.inactive.bg_fill = colors::SURFACE_SECONDARY;
                    style.visuals.widgets.hovered.bg_fill = colors::SURFACE_SECONDARY;
                    let text_edit_width = ui.max_rect().width() - 132.0;
                    let text_edit_res = ui.add(
                        egui::TextEdit::singleline(&mut table.current_query)
                            .desired_width(text_edit_width)
                            .margin(egui::Margin::symmetric(16, 8)),
                    );
                    ui.add_space(16.0);
                    let query_res = ui.add_sized([55., 32.], EButton::green("QUERY"));
                    let enter_key = text_edit_res.lost_focus()
                        && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));
                    if query_res.clicked() || enter_key {
                        table.state = SqlTableState::Requested(Instant::now());
                        state.commands.send_req_reply(
                            SQLQuery(table.current_query.clone()),
                            move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                                  mut states: Query<&mut SqlTable>| {
                                let Ok(mut entity) = states.get_mut(entity) else {
                                    return;
                                };
                                match res {
                                    Ok(ipc) => {
                                        let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                        let collect = ipc
                                            .batches
                                            .into_iter()
                                            .filter_map(|batch| {
                                                let mut buffer =
                                                    arrow::buffer::Buffer::from(batch.into_owned());
                                                decoder.decode(&mut buffer).ok()?
                                            })
                                            .collect::<Vec<_>>();
                                        entity.state = SqlTableState::Results(collect);
                                    }
                                    Err(err) => {
                                        entity.state = SqlTableState::Error(err);
                                    }
                                }
                            },
                        );
                    }
                });
                ui.add_space(8.0);
                match &table.state {
                    SqlTableState::None => {}
                    SqlTableState::Requested(_) => {
                        ui.label("Loading");
                    }
                    SqlTableState::Results(batches) => {
                        egui::Frame::NONE
                            .stroke(Stroke::new(1.0, colors::PRIMARY_CREAME_5))
                            .outer_margin(egui::Margin::same(8))
                            .show(ui, |ui| {
                                ui.set_width(ui.max_rect().width() - 16.);
                                let options = FormatOptions::default();
                                ui.style_mut().spacing.item_spacing = [8., 8.].into();
                                ui.visuals_mut().clip_rect_margin = 8.;
                                if let Some(first_batch) = batches.first() {
                                    let schema = first_batch.schema();
                                    let mut table = TableBuilder::new(ui)
                                        .striped(true)
                                        .resizable(true)
                                        .auto_shrink(false);
                                    let count = schema.fields().len();
                                    for (i, _) in schema.fields().iter().enumerate() {
                                        table = table.column(if (i + 1) == count {
                                            Column::remainder()
                                        } else {
                                            Column::auto().at_least(100.0)
                                        });
                                    }

                                    let table = table.header(22.0, |mut header| {
                                        for field in schema.fields() {
                                            header.col(|ui| {
                                                ui.strong(field.name());
                                            });
                                        }
                                    });

                                    table.body(|mut body| {
                                        for batch in batches {
                                            let Ok(formatters) = batch
                                                .columns()
                                                .iter()
                                                .map(|c| {
                                                    ArrayFormatter::try_new(c.as_ref(), &options)
                                                })
                                                .collect::<Result<Vec<_>, ArrowError>>()
                                            else {
                                                continue;
                                            };
                                            for row_idx in 0..batch.num_rows() {
                                                body.row(20.0, |mut row| {
                                                    for formatter in &formatters {
                                                        row.col(|ui| {
                                                            let label = formatter
                                                                .value(row_idx)
                                                                .to_string();
                                                            ui.label(label);
                                                        });
                                                    }
                                                });
                                            }
                                        }
                                    });
                                }
                            });
                    }
                    SqlTableState::Error(error_response) => {
                        let label = RichText::new(&error_response.description)
                            .color(colors::REDDISH_DEFAULT);
                        ui.label(label);
                    }
                }
            });
    }
}

// pub fn handle_pkt(
//     InRef(pkt): InRef<OwnedPacket<PacketGrantR>>,
//     query: Query<'w, 's, &'static mut SqlTable>,
//     entity: Entity,
// ) {
//     let Ok(mut table) = query.get_mut(entity) else {
//         return;
//     };
//     match pkt {
//         impeller2::types::OwnedPacket::Msg(m) if m.id == ErrorResponse::ID => {
//             let m = m.parse::<ErrorResponse>()?;
//         }
//         impeller2::types::OwnedPacket::Msg(m) if m.id == M::Reply::ID => {
//             let m = m.parse::<M::Reply>()?;
//             Ok(m)
//         }
//         _ => {}
//     }
// }
