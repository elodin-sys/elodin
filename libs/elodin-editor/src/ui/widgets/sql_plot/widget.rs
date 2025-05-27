use std::time::Instant;

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
    prelude::{Commands, Component, Entity, In, Query, ResMut},
    render::camera::Projection,
};
use egui::{RichText, Stroke};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use crate::ui::{
    colors::{ColorExt, get_scheme},
    theme,
    utils::format_num,
    widgets::{
        WidgetSystem,
        button::EButton,
        plot::{
            AXIS_LABEL_MARGIN, CHUNK_LEN, GraphState, NOTCH_LENGTH, PlotBounds,
            STEPS_X_WIDTH_DIVISOR, STEPS_Y_HEIGHT_DIVISOR, SharedBuffer, XYLine, draw_borders,
            draw_y_axis, get_inner_rect,
            gpu::{LineBundle, LineConfig, LineHandle, LineUniform, LineWidgetWidth},
            pretty_round,
        },
    },
};

#[derive(Clone)]
pub struct SQLPlotPane {
    pub entity: Entity,
    pub graph_entity: Entity,
    pub rect: Option<egui::Rect>,
    pub label: String,
}

#[derive(Component, Default)]
pub struct SqlPlot {
    pub current_query: String,
    pub state: SqlPlotState,
    pub xy_line_handle: Option<Handle<XYLine>>,
    pub line_entity: Option<Entity>,
    pub x_offset: f64,
    pub y_offset: f64,
}

#[derive(Default)]
pub enum SqlPlotState {
    #[default]
    None,
    Requested(Instant),
    Results,
    Error(ErrorResponse),
}

impl SqlPlot {
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
            x_values: SharedBuffer::<f32, CHUNK_LEN>::default(),
            y_values: SharedBuffer::<f32, CHUNK_LEN>::default(),
        };

        for value in array_iter(x_col) {
            xy_line.x_values.push((value - self.x_offset) as f32);
        }

        for value in array_iter(y_col) {
            xy_line.y_values.push((value - self.y_offset) as f32);
        }

        let handle = xy_lines.add(xy_line);
        self.xy_line_handle = Some(handle);
        self.state = SqlPlotState::Results;
    }
}

pub fn sync_bounds_sql(
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
pub struct SqlPlotWidget<'w, 's> {
    states: Query<'w, 's, &'static mut SqlPlot>,
    graphs_state: Query<'w, 's, &'static mut GraphState>,
    xy_lines: ResMut<'w, Assets<XYLine>>,
    commands: Commands<'w, 's>,
}

trait Vec2Ext {
    fn as_dvec2(self) -> DVec2;
}

impl Vec2Ext for egui::Vec2 {
    fn as_dvec2(self) -> DVec2 {
        DVec2::new(self.x as f64, self.y as f64)
    }
}

impl WidgetSystem for SqlPlotWidget<'_, '_> {
    type Args = SQLPlotPane;
    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        SQLPlotPane {
            entity,
            graph_entity,
            ..
        }: Self::Args,
    ) -> Self::Output {
        let mut state = state.get_mut(world);
        let Ok(mut plot) = state.states.get_mut(entity) else {
            return;
        };

        ui.vertical(|ui| {
            ui.horizontal_top(|ui| {
                egui::Frame::NONE
                    .inner_margin(egui::Margin::same(4))
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

                        ui.add_space(60.0);
                        let text_edit_width = ui.max_rect().width() - 165.0;
                        let text_edit_res = ui.add(
                            egui::TextEdit::singleline(&mut plot.current_query)
                                .hint_text("Enter SQL query returning X,Y columns")
                                .desired_width(text_edit_width)
                                .background_color(get_scheme().bg_primary)
                                .margin(egui::Margin::symmetric(16, 8)),
                        );
                        ui.add_space(16.0);
                        let query_res = ui.add_sized([55., 32.], EButton::green("PLOT"));
                        let enter_key = text_edit_res.lost_focus()
                            && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));

                        if query_res.clicked() || enter_key {
                            plot.state = SqlPlotState::Requested(Instant::now());
                            state.commands.send_req_reply(
                                SQLQuery(plot.current_query.clone()),
                                move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                                      mut states: Query<&mut SqlPlot>,
                                      mut xy_lines: ResMut<Assets<XYLine>>| {
                                    let Ok(mut entity) = states.get_mut(entity) else {
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
                                                    entity.process_record_batch(batch, &mut *xy_lines);
                                                    entity.state = SqlPlotState::Results;
                                                    return false;
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            entity.state = SqlPlotState::Error(err);
                                        }
                                    }
                                    true
                                },
                            );
                        }
                    });
            });

            match &plot.state {
                SqlPlotState::None => {
                    ui.centered_and_justified(|ui| {
                        ui.label("Enter a SQL query to plot data");
                    });
                }
                SqlPlotState::Requested(_) => {
                    ui.centered_and_justified(|ui| {
                        ui.label("Loading...");
                    });
                }
                SqlPlotState::Results => {
                    if let Some(xy_line_handle) = &plot.xy_line_handle {
                        let Ok(mut graph_state) = state.graphs_state.get_mut(graph_entity) else {
                            return;
                        };

                        let Some(xy_line) = state.xy_lines.get_mut(xy_line_handle) else {
                            return;
                        };

                        let data_bounds = xy_line.plot_bounds();
                        let rect = ui.max_rect();
                        let inner_rect = get_inner_rect(ui.max_rect());
                        let bounds = sync_bounds_sql(
                            &mut graph_state,
                            data_bounds,
                            rect,
                            inner_rect
                        );

                        graph_state.widget_width = ui.max_rect().width() as f64;
                        graph_state.y_range = bounds.min_y..bounds.max_y;

                        state.commands
                            .entity(graph_entity)
                            .try_insert(Projection::Orthographic(bounds.as_projection()));


                        let entity = if let Some(entity) = plot.line_entity {
                            entity
                        }else{
                            state.commands.spawn_empty().id()
                        };
                        let line_entity = state.commands.entity(entity).insert(LineBundle {
                            line: LineHandle::XY(xy_line_handle.clone()),
                            uniform: LineUniform::new(
                                graph_state.line_width,
                                get_scheme().highlight.into_bevy(),
                            ),
                            config: LineConfig {
                                render_layers: graph_state.render_layers.clone(),
                            },
                            line_visible_range: graph_state.visible_range.clone(),
                            graph_type: graph_state.graph_type,
                        })
                        .insert(ChildOf(graph_entity))
                        .insert(LineWidgetWidth(ui.max_rect().width() as usize))
                        .id();


                        let mut steps_y = ((inner_rect.height() / STEPS_Y_HEIGHT_DIVISOR) as usize).max(1);
                        if steps_y % 2 != 0 {
                            steps_y += 1;
                        }

                        let steps_x = ((inner_rect.width() / STEPS_X_WIDTH_DIVISOR) as usize).max(1);

                        draw_borders(ui, rect, inner_rect);
                        draw_y_axis(ui, bounds, steps_y, rect, inner_rect, plot.y_offset as f32);
                        draw_x_axis(ui, bounds, steps_x, rect, inner_rect, plot.x_offset as f32);

                        plot.line_entity = Some(line_entity);

                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label("No data to plot");
                        });
                    }
                }
                SqlPlotState::Error(error_response) => {
                    ui.centered_and_justified(|ui| {
                        let label = RichText::new(&error_response.description).color(get_scheme().error);
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
    x_offset: f32,
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
            format_num(tick + x_offset as f64),
            font_id.clone(),
            scheme.text_primary,
        );
    };

    if !bounds.min_x.is_finite() || !bounds.max_x.is_finite() {
        return;
    }

    let step_size = pretty_round((bounds.max_x - bounds.min_x) / steps_x as f64);
    if !step_size.is_normal() {
        return;
    }

    let steps_x = (0..=steps_x)
        .map(|i| bounds.min_x + (i as f64) * step_size)
        .collect::<Vec<f64>>();

    for x_step in steps_x {
        draw_tick(x_step);
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
                .map(|x| x.unwrap_or_default() as f64),
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
