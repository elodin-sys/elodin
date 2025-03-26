use std::time::Instant;

use arrow::{
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, In, Query},
};
use egui::{RichText, Stroke};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use super::{
    colors::{self, ColorExt},
    theme,
    widgets::{WidgetSystem, button::EButton},
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
impl SqlTableState {
    pub fn push_result(&mut self, batch: RecordBatch) {
        match self {
            SqlTableState::Results(record_batches) => {
                record_batches.push(batch);
            }
            other => {
                *other = SqlTableState::Results(vec![batch]);
            }
        }
    }
}

pub struct SqlTableResults<'a> {
    batches: &'a [RecordBatch],
    formatters: Vec<Vec<ArrayFormatter<'a>>>,
}

impl<'a> SqlTableResults<'a> {
    pub fn from_record_batches(batches: &'a [RecordBatch], columns: usize) -> Self {
        let options = FormatOptions::default();
        let mut formatters = (0..columns).map(|_| vec![]).collect::<Vec<_>>();
        for batch in batches {
            for (i, col) in batch.columns().iter().enumerate() {
                let Ok(fmt) = ArrayFormatter::try_new(col.as_ref(), &options) else {
                    continue;
                };
                formatters[i].push(fmt);
            }
        }
        SqlTableResults {
            batches,
            formatters,
        }
    }
}

impl egui_table::TableDelegate for SqlTableResults<'_> {
    fn header_cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::HeaderCellInfo) {
        let Some(first_batch) = self.batches.first() else {
            return;
        };
        let schema = first_batch.schema();

        ui.painter().rect_filled(
            ui.max_rect(),
            egui::CornerRadius::ZERO,
            colors::BONE_DEFAULT,
        );
        for field in &schema.fields[cell.col_range.clone()] {
            egui::Frame::NONE
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.strong(RichText::new(field.name()).color(colors::PRIMARY_SMOKE));
                });
        }
    }

    fn cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::CellInfo) {
        let mut count: usize = 0;
        let mut batch_i = 0;
        for (i, batch) in self.batches.iter().enumerate() {
            let next_count = count + batch.num_rows();
            if (count..next_count).contains(&(cell.row_nr as usize)) {
                batch_i = i;
                break;
            }
            count = next_count;
        }
        let offset = cell.row_nr as usize - count;
        let formatters = &self.formatters[cell.col_nr];
        let formatter = &formatters[batch_i];
        if cell.row_nr % 2 == 0 {
            ui.painter().rect_filled(
                ui.max_rect(),
                egui::CornerRadius::ZERO,
                colors::SURFACE_SECONDARY.opacity(0.4),
            );
        }

        let label = formatter.value(offset).to_string();
        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(8, 0))
            .show(ui, |ui| {
                ui.label(label);
            });
    }

    fn row_top_offset(&self, _ctx: &egui::Context, _table_id: egui::Id, row_nr: u64) -> f32 {
        (24 * row_nr) as f32
    }
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
        ui.horizontal_top(|ui| {
            egui::Frame::NONE
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    let style = ui.style_mut();
                    style.visuals.widgets.active.corner_radius = theme::corner_radius_xs();
                    style.visuals.widgets.hovered.corner_radius = theme::corner_radius_xs();
                    style.visuals.widgets.open.corner_radius = theme::corner_radius_xs();

                    style.visuals.widgets.inactive.bg_stroke =
                        Stroke::new(1.0, colors::BORDER_GREY);
                    style.visuals.widgets.inactive.fg_stroke =
                        Stroke::new(1.0, colors::PRIMARY_CREAME);
                    style.visuals.widgets.hovered.bg_stroke =
                        Stroke::new(1.0, colors::HYPERBLUE_DEFAULT.opacity(0.5));

                    style.spacing.button_padding = [16.0, 16.0].into();

                    style.visuals.widgets.active.bg_fill = colors::PRIMARY_SMOKE;
                    style.visuals.widgets.open.bg_fill = colors::PRIMARY_SMOKE;
                    style.visuals.widgets.inactive.bg_fill = colors::SURFACE_SECONDARY;
                    style.visuals.widgets.hovered.bg_fill = colors::SURFACE_SECONDARY;
                    style.visuals.widgets.active.fg_stroke =
                        Stroke::new(1.0, colors::PRIMARY_CREAME);
                    let text_edit_width = ui.max_rect().width() - 104.0;
                    let text_edit_res = ui.add(
                        egui::TextEdit::singleline(&mut table.current_query)
                            .hint_text("Enter an SQL query - like `show tables`")
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
                                    return true;
                                };
                                match res {
                                    Ok(ipc) => {
                                        let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                        if let Some(batch) = ipc.batch {
                                            let mut buffer =
                                                arrow::buffer::Buffer::from(batch.into_owned());
                                            if let Some(batch) =
                                                decoder.decode(&mut buffer).ok().and_then(|b| b)
                                            {
                                                entity.state.push_result(batch);
                                                return false;
                                            }
                                        }
                                    }
                                    Err(err) => {
                                        entity.state = SqlTableState::Error(err);
                                    }
                                }
                                true
                            },
                        );
                    }
                });
        });
        match &mut table.state {
            SqlTableState::None => {}
            SqlTableState::Requested(_) => {
                ui.label("Loading");
            }
            SqlTableState::Results(batches) => {
                let Some(first_batch) = batches.first() else {
                    return;
                };
                let schema = first_batch.schema();
                let count = schema.fields().len();
                let rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
                let table = egui_table::Table::new()
                    .id_salt("table")
                    .num_rows(rows as u64)
                    .columns(vec![egui_table::Column::default(); count])
                    .num_sticky_cols(0)
                    .headers(vec![egui_table::HeaderRow::new(28.0)]);
                let mut del = SqlTableResults::from_record_batches(batches, count);
                table.show(ui, &mut del);
            }
            SqlTableState::Error(error_response) => {
                let label =
                    RichText::new(&error_response.description).color(colors::REDDISH_DEFAULT);
                ui.label(label);
            }
        }
    }
}
