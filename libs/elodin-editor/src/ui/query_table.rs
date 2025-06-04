use std::time::Instant;

use arrow::{
    record_batch::RecordBatch,
    util::display::{ArrayFormatter, FormatOptions},
};
use bevy::{
    ecs::system::SystemParam,
    prelude::{Commands, Component, Entity, In, Query, Res},
};
use egui::{RichText, Stroke};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use crate::EqlContext;

use super::{
    colors::{ColorExt, get_scheme},
    theme,
    widgets::{
        WidgetSystem, button::EButton, inspector::graph::eql_autocomplete, query_plot::QueryType,
    },
};

#[derive(Clone)]
pub struct QueryTablePane {
    pub entity: Entity,
}

#[derive(Component, Default)]
pub struct QueryTable {
    pub current_query: String,
    pub state: QueryTableState,
    pub query_type: QueryType,
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
        if cell.row_nr % 2 == 0 {
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
    states: Query<'w, 's, &'static mut QueryTable>,
    eql_context: Res<'w, EqlContext>,
    commands: Commands<'w, 's>,
}

impl WidgetSystem for QueryTableWidget<'_, '_> {
    type Args = QueryTablePane;

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        QueryTablePane { entity }: Self::Args,
    ) -> Self::Output {
        let QueryTableWidget {
            mut states,
            eql_context,
            mut commands,
        } = state.get_mut(world);
        let Ok(mut table) = states.get_mut(entity) else {
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
                        Stroke::new(1.0, get_scheme().border_primary);
                    style.visuals.widgets.inactive.fg_stroke =
                        Stroke::new(1.0, get_scheme().text_primary);
                    style.visuals.widgets.hovered.bg_stroke =
                        Stroke::new(1.0, get_scheme().highlight.opacity(0.5));

                    style.spacing.button_padding = [16.0, 16.0].into();

                    style.visuals.widgets.active.bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.open.bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.inactive.bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.hovered.bg_fill = get_scheme().bg_primary;

                    style.visuals.widgets.active.weak_bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.open.weak_bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.inactive.weak_bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.hovered.weak_bg_fill = get_scheme().bg_primary;
                    style.visuals.widgets.active.fg_stroke =
                        Stroke::new(1.0, get_scheme().text_primary);
                    let text_edit_width = ui.max_rect().width() - 160.0;
                    let text_edit_res = ui.add(
                        egui::TextEdit::singleline(&mut table.current_query)
                            .hint_text("Enter an SQL query - like `show tables`")
                            .lock_focus(true)
                            .desired_width(text_edit_width)
                            .background_color(get_scheme().bg_primary)
                            .margin(egui::Margin::symmetric(16, 8)),
                    );

                    ui.add_space(8.0);

                    ui.scope(|ui| {
                        theme::configure_combo_box(ui.style_mut());
                        ui.style_mut().spacing.combo_width = ui.available_size().x;
                        let prev_query_type = table.query_type;
                        egui::ComboBox::from_id_salt("query_type")
                            .width(55.)
                            .selected_text(match table.query_type {
                                QueryType::EQL => "EQL",
                                QueryType::SQL => "SQL",
                            })
                            .show_ui(ui, |ui| {
                                theme::configure_combo_item(ui.style_mut());
                                ui.selectable_value(&mut table.query_type, QueryType::EQL, "EQL");
                                ui.selectable_value(&mut table.query_type, QueryType::SQL, "SQL");
                            });
                        if let (QueryType::EQL, QueryType::SQL) =
                            (prev_query_type, table.query_type)
                        {
                            if let Ok(sql) = eql_context.0.sql(&table.current_query) {
                                table.current_query = sql;
                            }
                        }
                    });
                    ui.add_space(8.0);
                    let query_res = ui.add_sized([55., 32.], EButton::highlight("QUERY"));

                    if table.query_type == QueryType::EQL {
                        eql_autocomplete(
                            ui,
                            &eql_context.0,
                            &text_edit_res
                                .clone()
                                .with_new_rect(text_edit_res.rect.expand2(egui::vec2(0.0, 8.0))),
                            &mut table.current_query,
                        );
                    }

                    let enter_key = text_edit_res.lost_focus()
                        && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));
                    if query_res.clicked() || enter_key {
                        table.state = QueryTableState::Requested(Instant::now());
                        let query = match table.query_type {
                            QueryType::SQL => table.current_query.to_string(),
                            QueryType::EQL => match eql_context.0.sql(&table.current_query) {
                                Ok(sql) => sql,
                                Err(err) => {
                                    table.state = QueryTableState::Error(ErrorResponse {
                                        description: err.to_string(),
                                    });
                                    return;
                                }
                            },
                        };
                        commands.send_req_reply(
                            SQLQuery(query),
                            move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                                  mut states: Query<&mut QueryTable>| {
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
                                        entity.state = QueryTableState::Error(err);
                                    }
                                }
                                true
                            },
                        );
                    }
                });
        });
        match &mut table.state {
            QueryTableState::None => {}
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
