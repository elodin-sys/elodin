use std::time::Instant;

use bevy::ecs::system::SystemParam;
use bevy::prelude::{Commands, Entity, In, Query, Res, ResMut};
use bevy_egui::egui;
use impeller2::types::Timestamp;
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, EarliestTimestamp, ErrorResponse, LastUpdated, QueryType, SQLQuery};

use crate::{
    EqlContext, SelectedTimeRange,
    ui::{
        colors::get_scheme,
        inspector::{eql_autocomplete, query},
        query_table::{QueryTableData, QueryTableState},
        theme,
        widgets::WidgetSystem,
    },
};

#[derive(SystemParam)]
pub struct InspectorQueryTable<'w, 's> {
    tables: Query<'w, 's, &'static mut QueryTableData>,
    eql_context: ResMut<'w, EqlContext>,
    selected_range: Res<'w, SelectedTimeRange>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    last_updated: Res<'w, LastUpdated>,
    commands: Commands<'w, 's>,
}

impl WidgetSystem for InspectorQueryTable<'_, '_> {
    type Args = Entity;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        entity: Self::Args,
    ) -> Self::Output {
        let InspectorQueryTable {
            mut tables,
            mut eql_context,
            selected_range,
            earliest_timestamp,
            last_updated,
            mut commands,
        } = state.get_mut(world);
        let Ok(mut table) = tables.get_mut(entity) else {
            return;
        };

        {
            use std::cmp::{max, min};

            let mut start = selected_range.0.start;
            let mut end = selected_range.0.end;
            let placeholder_start = Timestamp(i64::MIN);
            let placeholder_end = Timestamp(i64::MAX);

            if start == placeholder_start && end == placeholder_end {
                start = earliest_timestamp.0;
                end = last_updated.0;
            }

            if start > end {
                start = min(earliest_timestamp.0, last_updated.0);
                end = max(earliest_timestamp.0, last_updated.0);
            }

            eql_context.0.earliest_timestamp = start;
            eql_context.0.last_timestamp = end;
        }

        let context = &eql_context.0;

        egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(0, 8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("QUERY TYPE").color(get_scheme().text_secondary));
                ui.add_space(8.0);
                theme::configure_combo_box(ui.style_mut());
                ui.style_mut().spacing.combo_width = ui.available_size().x;
                let prev_query_type = table.data.query_type;
                egui::ComboBox::from_id_salt("query_table_type")
                    .selected_text(match table.data.query_type {
                        QueryType::EQL => "EQL",
                        QueryType::SQL => "SQL",
                    })
                    .show_ui(ui, |ui| {
                        theme::configure_combo_item(ui.style_mut());
                        ui.selectable_value(&mut table.data.query_type, QueryType::EQL, "EQL");
                        ui.selectable_value(&mut table.data.query_type, QueryType::SQL, "SQL");
                    });
                if let (QueryType::EQL, QueryType::SQL) = (prev_query_type, table.data.query_type)
                    && let Ok(sql) = context.sql(&table.data.query)
                {
                    table.data.query = sql;
                }
            });

        ui.separator();
        ui.label(egui::RichText::new("QUERY").color(get_scheme().text_secondary));
        theme::configure_input_with_border(ui.style_mut());
        let query_type = table.data.query_type;
        let query_res = ui.add(query(&mut table.data.query, query_type));
        if query_type == QueryType::EQL {
            eql_autocomplete(ui, context, &query_res, &mut table.data.query);
        }

        // Execute query when Enter is pressed (singleline input loses focus on Enter)
        let enter_pressed =
            query_res.lost_focus() && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));

        if enter_pressed && !table.data.query.is_empty() {
            table.state = QueryTableState::Requested(Instant::now());
            let query = match table.data.query_type {
                QueryType::SQL => table.data.query.to_string(),
                QueryType::EQL => match context.sql(&table.data.query) {
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
                      mut states: Query<&mut QueryTableData>| {
                    let Ok(mut entity) = states.get_mut(entity) else {
                        return true;
                    };
                    match res {
                        Ok(ipc) => {
                            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                            if let Some(batch) = ipc.batch {
                                let mut buffer = arrow::buffer::Buffer::from(batch.into_owned());
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
    }
}
