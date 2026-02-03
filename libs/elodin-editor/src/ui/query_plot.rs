use std::time::{Duration, Instant};

use arrow::{
    array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
    record_batch::RecordBatch,
};
use bevy::{
    asset::{Assets, Handle},
    camera::Projection,
    ecs::{hierarchy::ChildOf, system::SystemParam},
    math::DVec2,
    prelude::{Commands, Component, Entity, In, Query, Res, ResMut},
};
use egui::RichText;
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, PlotMode, QueryPlot, QueryType, SQLQuery};
use itertools::Itertools;

use crate::{
    EqlContext, SelectedTimeRange, TimeRangeBehavior,
    ui::{
        colors::{ColorExt, EColor, get_scheme},
        plot::{
            GraphState, PlotBounds, PlotDataSource, TimeseriesPlot, XYLine, get_inner_rect,
            gpu::{LineBundle, LineConfig, LineHandle, LineUniform, LineWidgetWidth},
        },
        tiles::WindowState,
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
    pub auto_color: bool,
    pub xy_line_handle: Option<Handle<XYLine>>,
    pub line_entity: Option<Entity>,
    pub x_offset: f64,
    pub y_offset: f64,
    pub last_refresh: Option<Instant>,
    pub earliest_timestamp: Option<impeller2::types::Timestamp>,
}

impl Default for QueryPlotData {
    fn default() -> Self {
        Self {
            data: QueryPlot {
                name: "Query Plot".to_string(),
                query: Default::default(),
                refresh_interval: Duration::from_millis(500),
                auto_refresh: Default::default(),
                color: impeller2_wkt::Color::from_color32(get_scheme().highlight),
                query_type: QueryType::EQL,
                plot_mode: PlotMode::TimeSeries,
                x_label: None,
                y_label: None,
                aux: (),
            },
            state: Default::default(),
            auto_color: true,
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

        let (x_values, earliest_abs_timestamp_micros): (Vec<f64>, Option<i64>) = match x_col
            .data_type()
        {
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
                // Already in microseconds, no conversion needed
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
                // Convert nanoseconds to microseconds
                (relative, earliest.map(|ns| ns / 1_000))
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
                // Convert milliseconds to microseconds
                (relative, earliest.map(|ms| ms * 1_000))
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
                // Convert seconds to microseconds
                (relative, earliest.map(|s| s * 1_000_000))
            }
            _ => {
                // For non-timestamp types, use the existing array_iter logic
                let values: Vec<f64> = array_iter(x_col).collect();
                (values, None)
            }
        };

        if let Some(earliest_micros) = earliest_abs_timestamp_micros
            && (self.earliest_timestamp.is_none()
                || Some(impeller2::types::Timestamp(earliest_micros)) < self.earliest_timestamp)
        {
            self.earliest_timestamp = Some(impeller2::types::Timestamp(earliest_micros));
        }

        let finite_x_values: Vec<f64> = x_values
            .iter()
            .copied()
            .filter(|&x| x.is_finite())
            .collect();

        // In XY mode, don't subtract x_offset so we preserve absolute coordinates
        // In TimeSeries mode, subtract x_offset to make time relative (starting from 0)
        let is_xy_mode = self.data.plot_mode == PlotMode::XY;
        if is_xy_mode {
            self.x_offset = 0.0;
        } else {
            self.x_offset = finite_x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if !self.x_offset.is_finite() {
                self.x_offset = 0.0;
            }
        }
        self.y_offset = 0.0;

        let mut xy_line = XYLine {
            label: "SQL Data".to_string(),
            x_shard_alloc: None,
            y_shard_alloc: None,
            x_values: vec![],
            y_values: vec![],
        };

        let mut y_iter = array_iter(y_col);
        let mut points: Vec<(f64, f64)> = Vec::new();

        for x_value in x_values {
            if let Some(y_value) = y_iter.next()
                && x_value.is_finite()
                && y_value.is_finite()
            {
                points.push((x_value, y_value));
            }
        }

        // Skip initial points that might be initialization artifacts.
        // Only do this heuristic for time-series mode, not XY mode where all points are meaningful.
        let skip_initial_points = if !is_xy_mode && points.len() > 2 {
            // Find how many initial points share the same timestamp.
            let first_time = points[0].0;
            let mut last_same = 0usize;

            // Count consecutive points with the same initial timestamp.
            for (i, (time, _)) in points.iter().enumerate().skip(1) {
                if (*time - first_time).abs() < 0.001 {
                    // Same timestamp (within floating point tolerance).
                    last_same = i;
                } else {
                    break; // Found a different timestamp, stop counting.
                }
            }

            if last_same > 0 && last_same + 1 < points.len() {
                // Only skip if we have data beyond the initial duplicates.
                last_same + 1
            } else if points.len() >= 3 {
                // No duplicate timestamps (or all timestamps are equal), but check for a huge
                // value jump that indicates initialization artifacts.
                let first_y = points[0].1;
                let second_y = points[1].1;
                if (second_y - first_y).abs() > 50.0 {
                    1 // Skip just the first point.
                } else {
                    0 // Keep all points.
                }
            } else {
                0
            }
        } else {
            0 // Not enough points to analyze, or XY mode where we keep all points.
        };

        // Add the points to the plot, skipping initial bad points if needed
        for (x_value, y_value) in points.into_iter().skip(skip_initial_points) {
            xy_line.push_x_value((x_value - self.x_offset) as f32);
            xy_line.push_y_value((y_value - self.y_offset) as f32);
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
    time_range_behavior: ResMut<'w, TimeRangeBehavior>,
    window_states: Query<'w, 's, &'static mut WindowState>,
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
    type Args = (QueryPlotPane, Entity);
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        (pane, target_window): Self::Args,
    ) -> Self::Output {
        let QueryPlotPane {
            entity, scrub_icon, ..
        } = pane;
        // Use a default texture ID if scrub_icon is not provided
        // This should only happen during initialization, and will be set properly in the UI
        let scrub_icon = scrub_icon.unwrap_or(egui::TextureId::default());
        let mut state = state.get_mut(world);
        let Ok(mut plot) = state.states.get_mut(entity) else {
            return;
        };

        if plot.auto_color {
            let scheme_color = get_scheme().highlight;
            if plot.data.color.into_color32() != scheme_color {
                plot.data.color = impeller2_wkt::Color::from_color32(scheme_color);
            }
        }

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
                let query_label = plot.data.name.clone();
                let query_color = plot.data.color.into_color32();
                let plot_mode = plot.data.plot_mode;
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
                // In XY mode, use numeric X-axis labels; otherwise use time labels
                let plot_renderer = match plot_mode {
                    PlotMode::XY => TimeseriesPlot::from_bounds_xy_mode(
                        rect,
                        bounds,
                        selected_range,
                        earliest_timestamp,
                        current_timestamp,
                    ),
                    PlotMode::TimeSeries => TimeseriesPlot::from_bounds_with_relative_time(
                        rect,
                        bounds,
                        selected_range,
                        earliest_timestamp,
                        current_timestamp,
                        true, // is_relative_time = true for query plots
                    ),
                };

                let data_source = PlotDataSource::XY {
                    xy_lines: &state.xy_lines,
                    xy_line_handle,
                    query_label,
                    query_color,
                };

                let Ok(mut window_state) = state.window_states.get_mut(target_window) else {
                    return;
                };
                plot_renderer.render(
                    ui,
                    data_source,
                    &mut graph_state,
                    &scrub_icon,
                    entity,
                    &mut window_state.ui_state.selected_object,
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
                    let (min, max) = if (max - min).abs() < f32::EPSILON {
                        (min - 1.0, max + 1.0)
                    } else {
                        (min, max)
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
                    let (min, max) = if (max - min).abs() < f32::EPSILON {
                        (min - 1.0, max + 1.0)
                    } else {
                        (min, max)
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
        DataType::FixedSizeList(_, list_size) => {
            let list_array = array_ref
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            let list_size = *list_size as usize;
            if list_size == 0 {
                Box::new(std::iter::empty())
            } else {
                let values = list_array.values();
                let inner_values: Vec<f64> = array_iter(values).collect();
                if inner_values.is_empty() {
                    println!("Unsupported list data type: {:?}", values.data_type());
                    Box::new(std::iter::empty())
                } else {
                    let len = list_array.len();
                    let mut min_vals = vec![f64::INFINITY; list_size];
                    let mut max_vals = vec![f64::NEG_INFINITY; list_size];
                    for row in 0..len {
                        if list_array.is_null(row) {
                            continue;
                        }
                        let base = row * list_size;
                        for i in 0..list_size {
                            if let Some(value) = inner_values.get(base + i)
                                && value.is_finite()
                            {
                                if *value < min_vals[i] {
                                    min_vals[i] = *value;
                                }
                                if *value > max_vals[i] {
                                    max_vals[i] = *value;
                                }
                            }
                        }
                    }
                    let mut selected_index = 0usize;
                    let mut best_range = f64::NEG_INFINITY;
                    for i in 0..list_size {
                        let min = min_vals[i];
                        let max = max_vals[i];
                        if min.is_finite() && max.is_finite() {
                            let range = max - min;
                            if range > best_range {
                                best_range = range;
                                selected_index = i;
                            }
                        }
                    }
                    Box::new((0..len).map(move |row| {
                        if list_array.is_null(row) {
                            0.0
                        } else {
                            inner_values
                                .get(row * list_size + selected_index)
                                .copied()
                                .unwrap_or_default()
                        }
                    }))
                }
            }
        }
        ty => {
            println!("Unsupported data type: {:?}", ty);
            Box::new(std::iter::empty())
        }
    }
}
