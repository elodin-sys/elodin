use std::time::Instant;

use arrow::{
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use bevy::{ecs::system::SystemParam, prelude::{Component, Entity, Query}};
use egui::RichText;
use impeller2_wkt::{ErrorResponse, QueryTable};

use super::{
    PaneName,
    colors::{ColorExt, get_scheme},
    widgets::WidgetSystem,
};

#[derive(Clone)]
pub struct QueryTablePane {
    pub entity: Entity,
    pub name: PaneName,
}

#[derive(Component, Default)]
pub struct QueryTableData {
    pub data: QueryTable,
    pub state: QueryTableState,
}

#[derive(Default)]
pub enum QueryTableState {
    #[default]
    None,
    Requested(Instant),
    Results(Vec<RecordBatch>),
    Error(ErrorResponse),
}
impl QueryTableState {
    pub fn push_result(&mut self, batch: RecordBatch) {
        match self {
            QueryTableState::Results(record_batches) => {
                record_batches.push(batch);
            }
            other => {
                *other = QueryTableState::Results(vec![batch]);
            }
        }
    }
}

pub struct QueryTableResults<'a> {
    batches: &'a [RecordBatch],
    formatters: Vec<Vec<ArrayFormatter<'a>>>,
}

impl<'a> QueryTableResults<'a> {
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
        QueryTableResults {
            batches,
            formatters,
        }
    }
}

impl egui_table::TableDelegate for QueryTableResults<'_> {
    fn header_cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::HeaderCellInfo) {
        let Some(first_batch) = self.batches.first() else {
            return;
        };
        let schema = first_batch.schema();

        ui.painter().rect_filled(
            ui.max_rect(),
            egui::CornerRadius::ZERO,
            get_scheme().text_primary,
        );
        for field in &schema.fields[cell.col_range.clone()] {
            egui::Frame::NONE
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.strong(RichText::new(field.name()).color(get_scheme().bg_secondary));
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
        if cell.row_nr.is_multiple_of(2) {
            ui.painter().rect_filled(
                ui.max_rect(),
                egui::CornerRadius::ZERO,
                get_scheme().bg_secondary.opacity(0.4),
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
pub struct QueryTableWidget<'w, 's> {
    states: Query<'w, 's, &'static mut QueryTableData>,
}

impl WidgetSystem for QueryTableWidget<'_, '_> {
    type Args = QueryTablePane;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        QueryTablePane { entity, .. }: Self::Args,
    ) -> Self::Output {
        let QueryTableWidget { mut states } = state.get_mut(world);
        let Ok(mut table) = states.get_mut(entity) else {
            return;
        };

        match &mut table.state {
            QueryTableState::None => {
                ui.centered_and_justified(|ui| {
                    ui.label("Enter a query to view data");
                });
            }
            QueryTableState::Requested(_) => {
                ui.label("Loading");
            }
            QueryTableState::Results(batches) => {
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
                let mut del = QueryTableResults::from_record_batches(batches, count);
                table.show(ui, &mut del);
            }
            QueryTableState::Error(error_response) => {
                let label = RichText::new(&error_response.description).color(get_scheme().error);
                ui.label(label);
            }
        }
    }
}
