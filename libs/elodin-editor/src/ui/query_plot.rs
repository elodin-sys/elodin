use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use bevy::{
    asset::{Assets, Handle},
    camera::Projection,
    ecs::{hierarchy::ChildOf, system::SystemParam},
    math::DVec2,
    prelude::{Commands, Component, Entity, In, Query, Res, ResMut},
};
use egui::{Color32, RichText};
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
        sql_eql::{eql_to_sql_with_time, process_sql_record_batch},
        tiles::WindowState,
        timeline::TelemetryMode,
        widgets::WidgetSystem,
    },
};
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp};

use super::plot::{Line, gpu};
use crate::ui::widgets::SystemStateExt;

pub use crate::ui::sql_eql::array_iter;

#[derive(Clone)]
pub struct QueryPlotPane {
    pub entity: Entity,
    pub rect: Option<egui::Rect>,
    pub scrub_icon: Option<egui::TextureId>,
}

#[derive(Clone)]
pub struct QueryPlotSeries {
    pub handle: Handle<XYLine>,
    pub entity: Option<Entity>,
    pub color: Color32,
    pub label: String,
}

#[derive(Component)]
pub struct QueryPlotData {
    pub data: QueryPlot,
    pub state: QueryPlotState,
    pub auto_color: bool,
    /// Per-series colors from `graph` KDL `color` children; empty falls back to `data.color`.
    pub series_colors: Vec<Color32>,
    pub series: Vec<QueryPlotSeries>,
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
                node_id: Default::default(),
            },
            state: Default::default(),
            auto_color: true,
            series_colors: Vec::new(),
            series: Vec::new(),
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
    /// Rebuilds `series` from a SQL batch. Returns leftover line entities from a
    /// longer previous batch so the caller can despawn them.
    fn process_record_batch(
        &mut self,
        batch: RecordBatch,
        xy_lines: &mut Assets<XYLine>,
    ) -> Vec<Entity> {
        let default_color = self.data.color.into_color32();
        let Some(plot) = process_sql_record_batch(
            &batch,
            self.data.plot_mode,
            xy_lines,
            &self.series_colors,
            default_color,
        ) else {
            return Vec::new();
        };

        if let Some(earliest) = plot.earliest_timestamp
            && (self.earliest_timestamp.is_none() || Some(earliest) < self.earliest_timestamp)
        {
            self.earliest_timestamp = Some(earliest);
        }
        self.x_offset = plot.x_offset;
        self.y_offset = plot.y_offset;

        let mut old_entities: Vec<Option<Entity>> =
            self.series.drain(..).map(|s| s.entity).collect();
        self.series = plot
            .series
            .into_iter()
            .enumerate()
            .map(|(i, s)| QueryPlotSeries {
                handle: s.handle,
                entity: old_entities.get_mut(i).and_then(|e| e.take()),
                color: s.color,
                label: s.label,
            })
            .collect();
        self.state = QueryPlotState::Results;
        old_entities.into_iter().flatten().collect()
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
    telemetry_mode: Res<'w, TelemetryMode>,
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
        let mut state = state.params_mut(world);
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
                        match eql_to_sql_with_time(&state.eql_context.0, &plot.data.query) {
                            Ok(sql) => sql,
                            Err(err) => {
                                plot.state =
                                    QueryPlotState::Error(ErrorResponse { description: err });
                                return;
                            }
                        }
                    }
                };
                state.commands.send_req_reply(
                    SQLQuery(query),
                    move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                          mut states: Query<&mut QueryPlotData>,
                          mut xy_lines: ResMut<Assets<XYLine>>,
                          mut commands: Commands| {
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
                                        let leftover =
                                            plot.process_record_batch(batch, &mut xy_lines);
                                        for line_entity in leftover {
                                            commands.entity(line_entity).despawn();
                                        }
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

            if !plot.series.is_empty() {
                let Ok(mut graph_state) = state.graphs_state.get_mut(entity) else {
                    return;
                };

                // Store values we need before borrowing plot
                let query_label = plot.data.name.clone();
                let plot_mode = plot.data.plot_mode;
                let x_label = plot.data.x_label.clone();
                let y_label = plot.data.y_label.clone();
                let offset_y = plot.offset().y;
                let earliest_timestamp = plot
                    .earliest_timestamp
                    .unwrap_or(state.earliest_timestamp.0);
                let selected_range = state.selected_time_range.0.clone();
                let current_timestamp = state.current_timestamp.0;
                let series_snapshot: Vec<_> = plot
                    .series
                    .iter()
                    .map(|s| (s.handle.clone(), s.entity, s.color, s.label.clone()))
                    .collect();

                // X-range is already relative (time starts from 0), Y-range needs offset subtracted
                let data_bounds = PlotBounds::new(
                    graph_state.x_range.start, // Already relative
                    graph_state.y_range.start,
                    graph_state.x_range.end, // Already relative
                    graph_state.y_range.end,
                )
                .offset(DVec2::new(0.0, -offset_y)); // Only subtract Y offset
                let telemetry = state.telemetry_mode.0;
                let rect = ui.max_rect();
                let inner_rect = get_inner_rect(ui.max_rect(), telemetry);
                let bounds = sync_bounds_query(&mut graph_state, data_bounds, rect, inner_rect);

                graph_state.widget_width = ui.max_rect().width() as f64;
                let line_width = graph_state.line_width;
                let render_layers = graph_state.render_layers.clone();
                let visible_range = graph_state.visible_range.clone();
                let graph_type = graph_state.graph_type;
                let widget_width = ui.max_rect().width() as usize;

                state
                    .commands
                    .entity(entity)
                    .try_insert(Projection::Orthographic(bounds.as_projection()));

                let mut updated_entities = Vec::with_capacity(series_snapshot.len());
                let mut xy_series = Vec::with_capacity(series_snapshot.len());
                for (handle, existing, color, label) in series_snapshot {
                    let line_entity = existing.unwrap_or_else(|| state.commands.spawn_empty().id());
                    state
                        .commands
                        .entity(line_entity)
                        .insert(LineBundle {
                            line: LineHandle::XY(handle.clone()),
                            uniform: LineUniform::new(line_width, color.into_bevy()),
                            config: LineConfig {
                                render_layers: render_layers.clone(),
                            },
                            line_visible_range: visible_range.clone(),
                            graph_type,
                        })
                        .insert(ChildOf(entity))
                        .insert(LineWidgetWidth(widget_width));
                    updated_entities.push(line_entity);
                    xy_series.push(crate::ui::plot::XYPlotSeries {
                        handle,
                        label,
                        color,
                    });
                }
                for (series, entity) in plot.series.iter_mut().zip(updated_entities) {
                    series.entity = Some(entity);
                }

                // Use TimeseriesPlot for unified rendering
                // In XY mode, use numeric X-axis labels; otherwise use time labels
                let plot_renderer = match plot_mode {
                    PlotMode::XY => TimeseriesPlot::from_bounds_xy_mode(
                        rect,
                        bounds,
                        selected_range,
                        earliest_timestamp,
                        current_timestamp,
                        telemetry,
                    ),
                    PlotMode::TimeSeries => TimeseriesPlot::from_bounds_with_relative_time(
                        rect,
                        bounds,
                        selected_range,
                        earliest_timestamp,
                        current_timestamp,
                        true, // is_relative_time = true for query plots
                        telemetry,
                    ),
                }
                .with_labels(x_label, y_label);

                let data_source = PlotDataSource::XY {
                    xy_lines: &state.xy_lines,
                    query_label,
                    series: xy_series,
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
                    telemetry,
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
                        if plot.series.is_empty() {
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
        let mut y_min: Option<f32> = None;
        let mut y_max: Option<f32> = None;
        let mut x_min: Option<f32> = None;
        let mut x_max: Option<f32> = None;

        for series in &plot.series {
            let Some(entity) = series.entity else {
                continue;
            };
            let Ok(handle) = line_handles.get(entity) else {
                continue;
            };
            let Some(line) = handle.get(&mut lines, &mut xy_lines) else {
                continue;
            };
            let gpu::LineMut::XY(xy) = line else {
                continue;
            };
            if graph_state.auto_y_range {
                match xy.y_values.iter().flat_map(|c| c.cpu()).minmax() {
                    itertools::MinMaxResult::OneElement(elem) => {
                        y_min = Some(y_min.map_or(elem - 1.0, |m| m.min(elem - 1.0)));
                        y_max = Some(y_max.map_or(elem + 1.0, |m| m.max(elem + 1.0)));
                    }
                    itertools::MinMaxResult::MinMax(min, max) => {
                        y_min = Some(y_min.map_or(*min, |m| m.min(*min)));
                        y_max = Some(y_max.map_or(*max, |m| m.max(*max)));
                    }
                    itertools::MinMaxResult::NoElements => {}
                }
            }
            if graph_state.auto_x_range {
                match xy.x_values.iter().flat_map(|c| c.cpu()).minmax() {
                    itertools::MinMaxResult::OneElement(elem) => {
                        x_min = Some(x_min.map_or(elem - 1.0, |m| m.min(elem - 1.0)));
                        x_max = Some(x_max.map_or(elem + 1.0, |m| m.max(elem + 1.0)));
                    }
                    itertools::MinMaxResult::MinMax(min, max) => {
                        x_min = Some(x_min.map_or(*min, |m| m.min(*min)));
                        x_max = Some(x_max.map_or(*max, |m| m.max(*max)));
                    }
                    itertools::MinMaxResult::NoElements => {}
                }
            }
        }

        if graph_state.auto_y_range
            && let (Some(min), Some(max)) = (y_min, y_max)
        {
            let (min, max) = if (max - min).abs() < f32::EPSILON {
                (min - 1.0, max + 1.0)
            } else {
                (min, max)
            };
            graph_state.y_range = (min as f64 + plot.y_offset)..(max as f64 + plot.y_offset);
        }
        if graph_state.auto_x_range
            && let (Some(min), Some(max)) = (x_min, x_max)
        {
            let (min, max) = if (max - min).abs() < f32::EPSILON {
                (min - 1.0, max + 1.0)
            } else {
                (min, max)
            };
            graph_state.x_range = min as f64..max as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::Float64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;

    fn batch_xy(y_columns: usize) -> RecordBatch {
        let mut fields = vec![Field::new("x", DataType::Float64, false)];
        let mut columns: Vec<arrow::array::ArrayRef> =
            vec![Arc::new(Float64Array::from(vec![0.0, 1.0]))];
        for i in 0..y_columns {
            fields.push(Field::new(format!("y{i}"), DataType::Float64, false));
            columns.push(Arc::new(Float64Array::from(vec![i as f64, i as f64 + 1.0])));
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    #[test]
    fn shrinking_series_returns_leftover_line_entities() {
        let mut xy_lines = Assets::<XYLine>::default();
        let mut plot = QueryPlotData::default();
        plot.data.plot_mode = PlotMode::XY;

        assert!(
            plot.process_record_batch(batch_xy(3), &mut xy_lines)
                .is_empty()
        );
        assert_eq!(plot.series.len(), 3);

        let kept = Entity::from_bits(1);
        let extra_a = Entity::from_bits(2);
        let extra_b = Entity::from_bits(3);
        plot.series[0].entity = Some(kept);
        plot.series[1].entity = Some(extra_a);
        plot.series[2].entity = Some(extra_b);

        let leftover = plot.process_record_batch(batch_xy(1), &mut xy_lines);
        assert_eq!(plot.series.len(), 1);
        assert_eq!(plot.series[0].entity, Some(kept));
        assert_eq!(leftover, vec![extra_a, extra_b]);
    }
}
