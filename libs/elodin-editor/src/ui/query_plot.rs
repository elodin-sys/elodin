use std::time::{Duration, Instant};

use arrow::{
    array::{
        Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};
use bevy::{
    asset::{Assets, Handle},
    ecs::{hierarchy::ChildOf, system::SystemParam},
    math::DVec2,
    prelude::{Commands, Component, Entity, In, Query, Res, ResMut},
    render::camera::Projection,
};
use egui::RichText;
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, QueryPlot, QueryType, SQLQuery};
use itertools::Itertools;

use crate::{
    EqlContext, SelectedObject, SelectedTimeRange, TimeRangeBehavior,
    ui::{
        colors::{ColorExt, EColor, get_scheme},
        plot::{
            GraphState, PlotBounds, PlotDataSource, TimeseriesPlot, XYLine, get_inner_rect,
            gpu::{LineBundle, LineConfig, LineHandle, LineUniform, LineWidgetWidth},
        },
        widgets::WidgetSystem,
    },
};
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp};

use super::plot::{Line, gpu};

#[derive(Clone)]
pub struct QueryPlotPane {
    pub entity: Entity,
    pub rect: Option<egui::Rect>,
    pub scrub_icon: Option<egui::TextureId>,
}

#[derive(Component)]
pub struct QueryPlotData {
    pub data: QueryPlot,
    pub state: QueryPlotState,
    pub xy_line_handle: Option<Handle<XYLine>>,
    pub line_entity: Option<Entity>,
    pub x_offset: f64,
    pub y_offset: f64,
    pub last_refresh: Option<Instant>,
    pub earliest_timestamp: Option<impeller2::types::Timestamp>, // Store earliest timestamp for relative time conversion
}

impl Default for QueryPlotData {
    fn default() -> Self {
        Self {
            data: QueryPlot {
                label: "Query Plot".to_string(),
                query: Default::default(),
                refresh_interval: Duration::from_millis(500),
                auto_refresh: Default::default(),
                color: impeller2_wkt::Color::from_color32(get_scheme().highlight),
                query_type: QueryType::EQL,
                aux: (),
            },
            state: Default::default(),
            xy_line_handle: Default::default(),
            line_entity: Default::default(),
            x_offset: Default::default(),
            y_offset: Default::default(),
            last_refresh: Some(Instant::now()),
            earliest_timestamp: None,
        }
    }
}

#[derive(Default)]
pub enum QueryPlotState {
    #[default]
    None,
    Requested(Instant),
    Results,
    Error(ErrorResponse),
}

impl QueryPlotData {
    fn process_record_batch(&mut self, batch: RecordBatch, xy_lines: &mut Assets<XYLine>) {
        if batch.num_columns() < 2 || batch.num_rows() == 0 {
            return;
        }

        let x_col = batch.column(0);
        let y_col = batch.column(1);

        // Convert timestamp X column to seconds for proper time axis display
        // Also track the earliest absolute timestamp for relative time conversion
        let (x_values, earliest_abs_timestamp): (Vec<f64>, Option<i64>) = match x_col.data_type() {
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let values: Vec<i64> = x_col
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .iter()
                    .map(|x| x.unwrap_or_default())
                    .collect();
                let earliest = values.iter().min().copied();
                let relative: Vec<f64> = values.iter().map(|&x| x as f64 / 1_000_000.0).collect();
                (relative, earliest)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let values: Vec<i64> = x_col
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap()
                    .iter()
                    .map(|x| x.unwrap_or_default())
                    .collect();
                let earliest = values.iter().min().copied();
                let relative: Vec<f64> =
                    values.iter().map(|&x| x as f64 / 1_000_000_000.0).collect();
                (relative, earliest)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                let values: Vec<i64> = x_col
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .unwrap()
                    .iter()
                    .map(|x| x.unwrap_or_default())
                    .collect();
                let earliest = values.iter().min().copied();
                let relative: Vec<f64> = values.iter().map(|&x| x as f64 / 1_000.0).collect();
                (relative, earliest)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                let values: Vec<i64> = x_col
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .unwrap()
                    .iter()
                    .map(|x| x.unwrap_or_default())
                    .collect();
                let earliest = values.iter().min().copied();
                let relative: Vec<f64> = values.iter().map(|&x| x as f64).collect();
                (relative, earliest)
            }
            _ => {
                // For non-timestamp types, use the existing array_iter logic
                let values: Vec<f64> = array_iter(x_col).collect();
                (values, None)
            }
        };

        // Store earliest timestamp if this is the first batch or if we found an earlier one
        if let Some(earliest) = earliest_abs_timestamp
            && (self.earliest_timestamp.is_none()
                || Some(impeller2::types::Timestamp(earliest)) < self.earliest_timestamp)
        {
            self.earliest_timestamp = Some(impeller2::types::Timestamp(earliest));
        }

        // Filter out NaN and infinite values for offset calculation
        let finite_x_values: Vec<f64> = x_values
            .iter()
            .copied()
            .filter(|&x| x.is_finite())
            .collect();
        let finite_y_values: Vec<f64> = array_iter(y_col).filter(|&y| y.is_finite()).collect();

        self.x_offset = finite_x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        self.y_offset = finite_y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if !self.x_offset.is_finite() {
            self.x_offset = 0.0;
        }
        if !self.y_offset.is_finite() {
            self.y_offset = 0.0;
        }

        let mut xy_line = XYLine {
            label: "SQL Data".to_string(),
            x_shard_alloc: None,
            y_shard_alloc: None,
            x_values: vec![],
            y_values: vec![],
        };

        // Only add finite x and y pairs to avoid breaking rendering
        let mut y_iter = array_iter(y_col);
        for x_value in x_values {
            if let Some(y_value) = y_iter.next()
                && x_value.is_finite()
                && y_value.is_finite()
            {
                xy_line.push_x_value((x_value - self.x_offset) as f32);
                xy_line.push_y_value((y_value - self.y_offset) as f32);
            }
        }

        let handle = xy_lines.add(xy_line);
        self.xy_line_handle = Some(handle);
        self.state = QueryPlotState::Results;
    }

    fn offset(&self) -> DVec2 {
        DVec2::new(self.x_offset, self.y_offset)
    }
}

pub fn sync_bounds_query(
    graph_state: &mut GraphState,
    data_bounds: PlotBounds,
    rect: egui::Rect,
    inner_rect: egui::Rect,
) -> PlotBounds {
    let outer_ratio = (rect.size() / inner_rect.size()).as_dvec2();
    let pan_offset = graph_state.pan_offset.as_dvec2() * DVec2::new(-1.0, 1.0);

    data_bounds
        .zoom_at(outer_ratio, DVec2::new(1.0, 0.5))
        .offset_by_norm(pan_offset)
        .zoom(graph_state.zoom_factor.as_dvec2())
        .normalize()
}

#[derive(SystemParam)]
pub struct QueryPlotWidget<'w, 's> {
    states: Query<'w, 's, &'static mut QueryPlotData>,
    graphs_state: Query<'w, 's, &'static mut GraphState>,
    eql_context: Res<'w, EqlContext>,
    commands: Commands<'w, 's>,
    xy_lines: ResMut<'w, Assets<XYLine>>,
    selected_time_range: Res<'w, SelectedTimeRange>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    selected_object: ResMut<'w, SelectedObject>,
    time_range_behavior: ResMut<'w, TimeRangeBehavior>,
}

trait Vec2Ext {
    fn as_dvec2(&self) -> DVec2;
}

impl Vec2Ext for egui::Vec2 {
    fn as_dvec2(&self) -> DVec2 {
        DVec2::new(self.x as f64, self.y as f64)
    }
}

impl WidgetSystem for QueryPlotWidget<'_, '_> {
    type Args = QueryPlotPane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        QueryPlotPane {
            entity, scrub_icon, ..
        }: Self::Args,
    ) -> Self::Output {
        // Use a default texture ID if scrub_icon is not provided
        // This should only happen during initialization, and will be set properly in the UI
        let scrub_icon = scrub_icon.unwrap_or(egui::TextureId::default());
        let mut state = state.get_mut(world);
        let Ok(mut plot) = state.states.get_mut(entity) else {
            return;
        };

        ui.vertical(|ui| {
            let should_refresh = if let Some(last_refresh) = plot.last_refresh {
                plot.data.auto_refresh && last_refresh.elapsed() > plot.data.refresh_interval
            } else {
                !plot.data.query.is_empty()
            };
            if should_refresh {
                plot.state = QueryPlotState::Requested(Instant::now());
                plot.last_refresh = Some(Instant::now());
                let query = match plot.data.query_type {
                    QueryType::SQL => plot.data.query.to_string(),
                    QueryType::EQL => {
                        // Parse the EQL expression
                        let expr = match state.eql_context.0.parse_str(&plot.data.query) {
                            Ok(expr) => expr,
                            Err(err) => {
                                plot.state = QueryPlotState::Error(ErrorResponse {
                                    description: err.to_string(),
                                });
                                return;
                            }
                        };
                        // Get time field for query plots (they need time as first column, value as second)
                        let time_field = match expr.to_sql_time_field() {
                            Ok(field) => field,
                            Err(err) => {
                                plot.state = QueryPlotState::Error(ErrorResponse {
                                    description: err.to_string(),
                                });
                                return;
                            }
                        };
                        // Generate SQL and prepend time column
                        match expr.to_sql(&state.eql_context.0) {
                            Ok(mut sql) => {
                                // Insert time column after "select " (case-insensitive)
                                if let Some(pos) = sql.to_lowercase().find("select ") {
                                    let after_select = pos + 7;
                                    sql.insert_str(after_select, &format!("{}, ", time_field));
                                } else {
                                    // Fallback: prepend time at the beginning
                                    sql = format!(
                                        "select {}, {}",
                                        time_field,
                                        sql.trim_start_matches("select ")
                                            .trim_start_matches("SELECT ")
                                    );
                                }
                                sql
                            }
                            Err(err) => {
                                plot.state = QueryPlotState::Error(ErrorResponse {
                                    description: err.to_string(),
                                });
                                return;
                            }
                        }
                    }
                };
                state.commands.send_req_reply(
                    SQLQuery(query),
                    move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                          mut states: Query<&mut QueryPlotData>,
                          mut xy_lines: ResMut<Assets<XYLine>>| {
                        let Ok(mut plot) = states.get_mut(entity) else {
                            return true;
                        };
                        match res {
                            Ok(ipc) => {
                                if let Some(batch) = ipc.batch {
                                    let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                    let mut buffer =
                                        arrow::buffer::Buffer::from(batch.into_owned());
                                    if let Some(batch) =
                                        decoder.decode(&mut buffer).ok().and_then(|b| b)
                                    {
                                        plot.process_record_batch(batch, &mut xy_lines);
                                        plot.state = QueryPlotState::Results;
                                        return false;
                                    }
                                }
                            }
                            Err(err) => {
                                plot.state = QueryPlotState::Error(err);
                            }
                        }
                        true
                    },
                );
            }

            if let Some(xy_line_handle) = plot.xy_line_handle.clone() {
                let Ok(mut graph_state) = state.graphs_state.get_mut(entity) else {
                    return;
                };

                // Store values we need before borrowing plot
                let query_label = plot.data.label.clone();
                let query_color = plot.data.color.into_color32();
                let offset_y = plot.offset().y;
                let earliest_timestamp = plot
                    .earliest_timestamp
                    .unwrap_or(state.earliest_timestamp.0);
                let selected_range = state.selected_time_range.0.clone();
                let current_timestamp = state.current_timestamp.0;

                // X-range is already relative (time starts from 0), Y-range needs offset subtracted
                let data_bounds = PlotBounds::new(
                    graph_state.x_range.start, // Already relative
                    graph_state.y_range.start,
                    graph_state.x_range.end, // Already relative
                    graph_state.y_range.end,
                )
                .offset(DVec2::new(0.0, -offset_y)); // Only subtract Y offset
                let rect = ui.max_rect();
                let inner_rect = get_inner_rect(ui.max_rect());
                let bounds = sync_bounds_query(&mut graph_state, data_bounds, rect, inner_rect);

                graph_state.widget_width = ui.max_rect().width() as f64;

                state
                    .commands
                    .entity(entity)
                    .try_insert(Projection::Orthographic(bounds.as_projection()));

                let line_entity = if let Some(entity) = plot.line_entity {
                    entity
                } else {
                    state.commands.spawn_empty().id()
                };
                state
                    .commands
                    .entity(line_entity)
                    .insert(LineBundle {
                        line: LineHandle::XY(xy_line_handle.clone()),
                        uniform: LineUniform::new(graph_state.line_width, query_color.into_bevy()),
                        config: LineConfig {
                            render_layers: graph_state.render_layers.clone(),
                        },
                        line_visible_range: graph_state.visible_range.clone(),
                        graph_type: graph_state.graph_type,
                    })
                    .insert(ChildOf(entity))
                    .insert(LineWidgetWidth(ui.max_rect().width() as usize));

                plot.line_entity = Some(line_entity);

                // Use TimeseriesPlot for unified rendering
                let plot_renderer = TimeseriesPlot::from_bounds_with_relative_time(
                    rect,
                    bounds,
                    selected_range,
                    earliest_timestamp,
                    current_timestamp,
                    true, // is_relative_time = true for query plots
                );

                let data_source = PlotDataSource::XY {
                    xy_lines: &state.xy_lines,
                    xy_line_handle,
                    query_label,
                    query_color,
                };

                plot_renderer.render(
                    ui,
                    data_source,
                    &mut graph_state,
                    &scrub_icon,
                    entity,
                    state.selected_object.as_mut(),
                    state.time_range_behavior.as_mut(),
                );
            }
            match &plot.state {
                QueryPlotState::None => {
                    ui.centered_and_justified(|ui| {
                        ui.label("Enter a query to plot data");
                    });
                }
                QueryPlotState::Requested(_instant) => {
                    ui.centered_and_justified(|ui| {
                        if plot.line_entity.is_none() {
                            ui.label("Loading...");
                        }
                    });
                }
                QueryPlotState::Results => {}
                QueryPlotState::Error(error_response) => {
                    ui.centered_and_justified(|ui| {
                        let label =
                            RichText::new(&error_response.description).color(get_scheme().error);
                        ui.label(label);
                    });
                }
            }
        });
    }
}

pub fn auto_bounds(
    mut graph_states: Query<(&mut GraphState, &mut QueryPlotData)>,
    line_handles: Query<&LineHandle>,
    mut lines: ResMut<Assets<Line>>,
    mut xy_lines: ResMut<Assets<XYLine>>,
) {
    for (mut graph_state, plot) in &mut graph_states {
        if let Some(entity) = plot.line_entity {
            let Ok(handle) = line_handles.get(entity) else {
                continue;
            };
            let Some(line) = handle.get(&mut lines, &mut xy_lines) else {
                continue;
            };
            if let gpu::LineMut::XY(xy) = line {
                if graph_state.auto_y_range {
                    let (min, max) = match xy.y_values.iter().flat_map(|c| c.cpu()).minmax() {
                        itertools::MinMaxResult::OneElement(elem) => (elem - 1.0, elem + 1.0),
                        itertools::MinMaxResult::MinMax(min, max) => (*min, *max),
                        itertools::MinMaxResult::NoElements => continue,
                    };

                    let min = min as f64 + plot.y_offset;
                    let max = max as f64 + plot.y_offset;

                    graph_state.y_range = min..max;
                }

                if graph_state.auto_x_range {
                    let (min, max) = match xy.x_values.iter().flat_map(|c| c.cpu()).minmax() {
                        itertools::MinMaxResult::OneElement(elem) => (elem - 1.0, elem + 1.0),
                        itertools::MinMaxResult::MinMax(min, max) => (*min, *max),
                        itertools::MinMaxResult::NoElements => continue,
                    };
                    // For X-axis (time), keep relative values (x_values already have offset subtracted)
                    // Don't add x_offset back - we want time to start from 0
                    let min = min as f64;
                    let max = max as f64;

                    graph_state.x_range = min..max;
                }
            }
        }
    }
}

pub fn array_iter(array_ref: &ArrayRef) -> Box<dyn Iterator<Item = f64> + '_> {
    match array_ref.data_type() {
        DataType::Float32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Float64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default()),
        ),
        DataType::Int32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Int64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::UInt32 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::UInt64 => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Second, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),

        DataType::Timestamp(TimeUnit::Millisecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => Box::new(
            array_ref
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .iter()
                .map(|x| x.unwrap_or_default() as f64),
        ),
        ty => {
            println!("Unsupported data type: {:?}", ty);
            Box::new(std::iter::empty())
        }
    }
}
