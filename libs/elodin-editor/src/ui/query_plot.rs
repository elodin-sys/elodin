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
    EqlContext,
    ui::{
        colors::{ColorExt, EColor, get_scheme},
        plot::{
            AXIS_LABEL_MARGIN, GraphState, NOTCH_LENGTH, PlotBounds, STEPS_X_WIDTH_DIVISOR,
            STEPS_Y_HEIGHT_DIVISOR, XYLine, draw_borders, draw_y_axis, get_inner_rect,
            gpu::{LineBundle, LineConfig, LineHandle, LineUniform, LineWidgetWidth},
            pretty_round,
        },
        utils::format_num,
        widgets::WidgetSystem,
    },
};

use super::plot::{Line, gpu};

#[derive(Clone)]
pub struct QueryPlotPane {
    pub entity: Entity,
    pub rect: Option<egui::Rect>,
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

        self.x_offset = array_iter(x_col).fold(f64::INFINITY, f64::min);
        self.y_offset = array_iter(y_col).fold(f64::INFINITY, f64::min);

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

        for value in array_iter(x_col) {
            xy_line.push_x_value((value - self.x_offset) as f32);
        }

        for value in array_iter(y_col) {
            xy_line.push_y_value((value - self.y_offset) as f32);
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
        QueryPlotPane { entity, .. }: Self::Args,
    ) -> Self::Output {
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
                    QueryType::EQL => match state.eql_context.0.sql(&plot.data.query) {
                        Ok(sql) => sql,
                        Err(err) => {
                            plot.state = QueryPlotState::Error(ErrorResponse {
                                description: err.to_string(),
                            });
                            return;
                        }
                    },
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

            if let Some(xy_line_handle) = &plot.xy_line_handle {
                let Ok(mut graph_state) = state.graphs_state.get_mut(entity) else {
                    return;
                };

                let data_bounds = PlotBounds::new(
                    graph_state.x_range.start,
                    graph_state.y_range.start,
                    graph_state.x_range.end,
                    graph_state.y_range.end,
                )
                .offset(-plot.offset());
                let rect = ui.max_rect();
                let inner_rect = get_inner_rect(ui.max_rect());
                let bounds = sync_bounds_query(&mut graph_state, data_bounds, rect, inner_rect);

                graph_state.widget_width = ui.max_rect().width() as f64;

                state
                    .commands
                    .entity(entity)
                    .try_insert(Projection::Orthographic(bounds.as_projection()));

                let entity = if let Some(entity) = plot.line_entity {
                    entity
                } else {
                    state.commands.spawn_empty().id()
                };
                let line_entity = state
                    .commands
                    .entity(entity)
                    .insert(LineBundle {
                        line: LineHandle::XY(xy_line_handle.clone()),
                        uniform: LineUniform::new(
                            graph_state.line_width,
                            plot.data.color.into_color32().into_bevy(),
                        ),
                        config: LineConfig {
                            render_layers: graph_state.render_layers.clone(),
                        },
                        line_visible_range: graph_state.visible_range.clone(),
                        graph_type: graph_state.graph_type,
                    })
                    .insert(ChildOf(entity))
                    .insert(LineWidgetWidth(ui.max_rect().width() as usize))
                    .id();

                let mut steps_y = ((inner_rect.height() / STEPS_Y_HEIGHT_DIVISOR) as usize).max(1);
                if steps_y % 2 != 0 {
                    steps_y += 1;
                }

                let steps_x = ((inner_rect.width() / STEPS_X_WIDTH_DIVISOR) as usize).max(1);

                draw_borders(ui, rect, inner_rect);
                let axis_bounds = bounds.offset(plot.offset());
                draw_y_axis(ui, axis_bounds, steps_y, rect, inner_rect);
                draw_x_axis(ui, axis_bounds, steps_x, rect, inner_rect);

                plot.line_entity = Some(line_entity);
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

pub fn draw_x_axis(
    ui: &mut egui::Ui,
    bounds: PlotBounds,
    steps_x: usize,
    rect: egui::Rect,
    inner_rect: egui::Rect,
) {
    let border_stroke = egui::Stroke::new(1.0, get_scheme().border_primary);
    let scheme = get_scheme();
    let mut font_id = egui::TextStyle::Monospace.resolve(ui.style());
    font_id.size = 11.0;

    let draw_tick = |tick| {
        let value = DVec2::new(tick, bounds.min_y);
        let screen_pos = bounds.value_to_screen_pos(rect, value);
        let screen_pos = egui::pos2(screen_pos.x, inner_rect.max.y);
        ui.painter().line_segment(
            [screen_pos, screen_pos + egui::vec2(0.0, NOTCH_LENGTH)],
            border_stroke,
        );

        ui.painter().text(
            screen_pos + egui::vec2(0.0, NOTCH_LENGTH + AXIS_LABEL_MARGIN),
            egui::Align2::CENTER_TOP,
            format_num(tick),
            font_id.clone(),
            scheme.text_primary,
        );
    };

    if !bounds.min_x.is_finite() || !bounds.max_x.is_finite() {
        return;
    }

    if bounds.min_x <= 0.0 {
        let step_size = pretty_round((bounds.max_x - bounds.min_x) / steps_x as f64);
        if !step_size.is_normal() {
            return;
        }
        let mut i = 0.0;

        while i < bounds.max_x {
            draw_tick(i);
            i += step_size;
        }
        draw_tick(i);

        let mut i = 0.0;
        while i > bounds.min_x {
            draw_tick(i);
            i -= step_size;
        }
        draw_tick(i);
    } else {
        let step_size = pretty_round(bounds.width() / steps_x as f64);
        let steps_x = (0..=steps_x).map(|i| bounds.min_x + (i as f64) * step_size);

        for x_step in steps_x {
            draw_tick(x_step);
        }
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
                    let min = min as f64 + plot.x_offset;
                    let max = max as f64 + plot.x_offset;

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
