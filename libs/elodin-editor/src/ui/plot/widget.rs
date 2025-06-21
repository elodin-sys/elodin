use crate::{
    editor_cam_touch::*,
    ui::{theme::corner_radius_sm, utils::Shrink4},
};
use bevy::{
    asset::Assets,
    ecs::{
        entity::Entity,
        event::EventReader,
        query::With,
        system::{Commands, Local, Query, Res, SystemParam},
    },
    input::{
        ButtonInput,
        keyboard::Key,
        mouse::{MouseButton, MouseScrollUnit, MouseWheel},
    },
    math::{DVec2, Rect, Vec2},
    prelude::{Component, ResMut},
    render::camera::{Camera, OrthographicProjection, Projection, ScalingMode},
    window::{PrimaryWindow, Window},
};
use bevy_egui::egui::{self, Align, Layout};
use egui::{CornerRadius, Frame, Margin, RichText, Stroke};
use impeller2::types::Timestamp;
use impeller2_bevy::{ComponentMetadataRegistry, ComponentPath};
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp};
use std::time::{Duration, Instant};
use std::{
    fmt::Debug,
    ops::{Range, RangeInclusive},
};

use crate::{
    Offset, SelectedTimeRange, TimeRangeBehavior,
    plugins::LogicalKeyState,
    ui::{
        colors::{ColorExt, get_scheme, with_opacity},
        plot::{
            CollectedGraphData, GraphState, Line,
            gpu::{LineBundle, LineConfig, LineUniform},
        },
        time_label::{PrettyDuration, time_label},
        timeline::DurationExt,
        utils::format_num,
        widgets::WidgetSystem,
    },
};

use super::{
    PlotDataComponent, XYLine,
    gpu::{self, LineHandle, LineVisibleRange, LineWidgetWidth},
};

#[derive(SystemParam)]
pub struct PlotWidget<'w, 's> {
    collected_graph_data: ResMut<'w, CollectedGraphData>,
    graphs_state: Query<'w, 's, &'static mut GraphState>,
    lines: ResMut<'w, Assets<Line>>,
    commands: Commands<'w, 's>,
    selected_time_range: Res<'w, SelectedTimeRange>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    current_timestamp: Res<'w, CurrentTimestamp>,
    time_range_behavior: ResMut<'w, TimeRangeBehavior>,
    line_query: Query<'w, 's, &'static LineHandle>,
}

impl WidgetSystem for PlotWidget<'_, '_> {
    type Args = (Entity, egui::TextureId);

    type Output = ();

    fn ui_system(
        world: &mut bevy::prelude::World,
        state: &mut bevy::ecs::system::SystemState<Self>,
        ui: &mut egui::Ui,
        (id, scrub_icon): Self::Args,
    ) -> Self::Output {
        let PlotWidget {
            collected_graph_data,
            mut graphs_state,
            lines,
            mut commands,
            selected_time_range,
            earliest_timestamp,
            current_timestamp,
            mut time_range_behavior,
            line_query,
        } = state.get_mut(world);

        let Ok(mut graph_state) = graphs_state.get_mut(id) else {
            return;
        };

        let bounds = sync_bounds(
            &mut graph_state,
            selected_time_range.0.clone(),
            earliest_timestamp.0,
            ui.max_rect(),
            get_inner_rect(ui.max_rect()),
        );

        let line_visible_range = bounds.timestamp_range(earliest_timestamp.0);
        graph_state.visible_range = LineVisibleRange(line_visible_range.clone());
        graph_state.widget_width = ui.max_rect().width() as f64;

        commands
            .entity(id)
            .try_insert(Projection::Orthographic(bounds.as_projection()));

        TimeseriesPlot::from_bounds(
            ui.max_rect(),
            bounds,
            selected_time_range.0.clone(),
            earliest_timestamp.0,
            current_timestamp.0,
        )
        .render(
            ui,
            &lines,
            &line_query,
            &collected_graph_data,
            &mut graph_state,
            &scrub_icon,
            &mut time_range_behavior,
        );
    }
}

#[derive(Debug)]
pub struct TimeseriesPlot {
    selected_range: Range<Timestamp>,
    current_timestamp: Timestamp,
    earliest_timestamp: Timestamp,
    bounds: PlotBounds,
    rect: egui::Rect,
    inner_rect: egui::Rect,

    steps_x: usize,
    steps_y: usize,
}

pub const MARGIN: egui::Margin = egui::Margin {
    left: 60,
    right: 0,
    top: 35,
    bottom: 35,
};

pub const TICK_MARK_LINE_WIDTH: f32 = 1.0;
pub const TICK_MARK_ASPECT_RATIO: f32 = 12.0 / 30.0;
pub const NOTCH_LENGTH: f32 = 10.0;
pub const AXIS_LABEL_MARGIN: f32 = 5.0;
pub const Y_AXIS_LABEL_MARGIN: f32 = 10.0;
pub const Y_AXIS_FLAG_WIDTH: f32 = 70.0;
pub const Y_AXIS_FLAG_HEIGHT: f32 = 20.0;
pub const Y_AXIS_FLAG_MARGIN: f32 = 4.0;
pub const STEPS_Y_HEIGHT_DIVISOR: f32 = 50.0;
pub const STEPS_X_WIDTH_DIVISOR: f32 = 75.0;

pub const MODAL_WIDTH: f32 = 250.0;
pub const MODAL_MARGIN: f32 = 20.0;

pub const ZOOM_SENSITIVITY: f32 = 0.001;
pub const SCROLL_PIXELS_PER_LINE: f32 = 100.0;

pub fn get_inner_rect(rect: egui::Rect) -> egui::Rect {
    rect.shrink4(MARGIN)
}

impl TimeseriesPlot {
    pub fn from_bounds(
        rect: egui::Rect,
        bounds: PlotBounds,
        mut selected_range: Range<Timestamp>,
        earliest_timestamp: Timestamp,
        current_timestamp: Timestamp,
    ) -> Self {
        let inner_rect = get_inner_rect(rect);

        if selected_range.start == selected_range.end {
            selected_range.end += Duration::from_secs(10);
        }

        let mut steps_y = ((inner_rect.height() / STEPS_Y_HEIGHT_DIVISOR) as usize).max(1);
        if steps_y % 2 != 0 {
            steps_y += 1;
        }
        let steps_x = ((inner_rect.width() / STEPS_X_WIDTH_DIVISOR) as usize).max(1);

        Self {
            selected_range,
            current_timestamp,
            earliest_timestamp,

            bounds,
            rect,
            inner_rect,

            steps_x,
            steps_y,
        }
    }

    fn draw_x_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let step_size =
            hifitime::Duration::from_microseconds(self.bounds.width() / self.steps_x as f64)
                .segment_round();
        let step_size_micro = step_size.total_nanoseconds() / 1000;
        let step_size_float = step_size_micro as f64;
        if step_size_micro <= 0 {
            return;
        }
        let visible_time_range = self.visible_time_range();
        let start = self.selected_range.start;
        let end = visible_time_range.end;
        let start_count = (visible_time_range.start.0 - start.0) / step_size_micro as i64 - 1;
        let end_count = (end.0 - start.0) / step_size_micro as i64 + 1;

        for i in start_count..=end_count {
            let offset_float = step_size_float * i as f64;
            let offset = hifitime::Duration::from_microseconds(offset_float);

            let x_pos = self
                .bounds
                .value_to_screen_pos(
                    self.rect,
                    (
                        self.timestamp_to_x(Timestamp(start.0 + offset_float as i64)),
                        0.0,
                    )
                        .into(),
                )
                .x;

            ui.painter().line_segment(
                [
                    egui::pos2(x_pos, self.inner_rect.max.y),
                    egui::pos2(x_pos, self.inner_rect.max.y + (NOTCH_LENGTH)),
                ],
                egui::Stroke::new(1.0, get_scheme().border_primary),
            );

            ui.painter().text(
                egui::pos2(
                    x_pos,
                    self.inner_rect.max.y + (NOTCH_LENGTH + AXIS_LABEL_MARGIN),
                ),
                egui::Align2::CENTER_TOP,
                PrettyDuration(offset).to_string(),
                font_id.clone(),
                get_scheme().text_primary,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_modal(
        &self,
        ui: &mut egui::Ui,
        lines: &Assets<Line>,
        line_handles: &Query<&LineHandle>,
        graph_state: &GraphState,
        collected_graph_data: &CollectedGraphData,
        pointer_pos: egui::Pos2,
        timestamp: Timestamp,
    ) {
        let anchor_left = pointer_pos.x + MODAL_WIDTH + MODAL_MARGIN < self.rect.right(); // NOTE: might want to replace pointer_pos with x_offset from `render`

        let (pivot, fixed_pos) = if anchor_left {
            (
                egui::Align2::LEFT_TOP,
                egui::pos2(pointer_pos.x + MODAL_MARGIN, pointer_pos.y + MODAL_MARGIN),
            )
        } else {
            (
                egui::Align2::RIGHT_TOP,
                egui::pos2(pointer_pos.x - MODAL_MARGIN, pointer_pos.y + MODAL_MARGIN),
            )
        };

        egui::Window::new("plot_modal")
            .pivot(pivot)
            .title_bar(false)
            .resizable(false)
            .fixed_pos(fixed_pos)
            .fixed_size(egui::vec2(MODAL_WIDTH, self.inner_rect.height() / 2.))
            .frame(
                Frame::default()
                    .inner_margin(Margin::same(8))
                    .stroke(Stroke::new(1.0, get_scheme().border_primary))
                    .corner_radius(corner_radius_sm())
                    .fill(get_scheme().bg_secondary)
                    .shadow(egui::epaint::Shadow {
                        offset: [0, 0],
                        blur: 8,
                        spread: 4,
                        color: get_scheme().shadow.opacity(0.2),
                    }),
            )
            .show(ui.ctx(), |ui| {
                let time: hifitime::Epoch = timestamp.into();

                ui.add(time_label(time));
                let offset = hifitime::Duration::from_microseconds(
                    (timestamp.0 - self.selected_range.start.0) as f64,
                );
                ui.label(PrettyDuration(offset).to_string());
                let mut current_component_path: Option<&ComponentPath> = None;
                for ((component_path, line_index), (entity, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line_handle) = line_handle.as_timeseries() else {
                        continue;
                    };
                    let Some(line) = lines.get(line_handle) else {
                        continue;
                    };
                    if current_component_path != Some(component_path) {
                        ui.add_space(8.0);
                        ui.add(egui::Separator::default().grow(16.0 * 2.0));
                        ui.add_space(8.0);
                        current_component_path = Some(component_path);
                        if let Some(component_data) =
                            collected_graph_data.get_component(&component_path.id)
                        {
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(component_data.label.to_owned())
                                    .size(11.0)
                                    .color(with_opacity(get_scheme().text_primary, 0.6)),
                            );
                            ui.add_space(8.0);
                        }
                    }

                    let Some(line_data) = collected_graph_data
                        .get_line(&component_path.id, *line_index)
                        .and_then(|h| lines.get(h))
                    else {
                        continue;
                    };

                    ui.horizontal(|ui| {
                        ui.style_mut().override_font_id =
                            Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::click());
                        ui.painter().rect(
                            rect,
                            egui::CornerRadius::same(2),
                            *color,
                            egui::Stroke::NONE,
                            egui::StrokeKind::Middle,
                        );
                        ui.add_space(6.);
                        ui.label(RichText::new(line_data.label.clone()).size(11.0));
                        let value = line
                            .data
                            .get_nearest(timestamp)
                            .map(|(_time, x)| format!("{:.2}", x))
                            .unwrap_or_else(|| "N/A".to_string());
                        ui.with_layout(Layout::top_down_justified(Align::RIGHT), |ui| {
                            ui.add_space(3.0);
                            ui.label(RichText::new(value).size(11.0));
                            ui.add_space(3.0);
                        })
                    });
                }
            });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        ui: &mut egui::Ui,
        lines: &Assets<Line>,
        line_handles: &Query<&LineHandle>,
        collected_graph_data: &CollectedGraphData,
        graph_state: &mut GraphState,
        scrub_icon: &egui::TextureId,
        time_range_behavior: &mut TimeRangeBehavior,
    ) {
        let response = ui.allocate_rect(self.rect, egui::Sense::click_and_drag());
        let pointer_pos = ui.input(|i| i.pointer.latest_pos());

        response.context_menu(|ui| {
            if ui.button("Set Time Range to Viewport Bounds").clicked() {
                let start = Timestamp((self.bounds.min_x as i64) + self.earliest_timestamp.0);
                let end = Timestamp((self.bounds.max_x as i64) + self.earliest_timestamp.0);
                graph_state.zoom_factor = Vec2::new(1.0, 1.0);
                graph_state.pan_offset = Vec2::ZERO;
                *time_range_behavior = TimeRangeBehavior {
                    start: Offset::Fixed(start),
                    end: Offset::Fixed(end),
                };
                ui.close_menu();
            }
        });

        let mut font_id = egui::TextStyle::Monospace.resolve(ui.style());

        if graph_state.components.is_empty() {
            ui.painter().text(
                self.rect.center(),
                egui::Align2::CENTER_CENTER,
                "NO DATA POINTS SELECTED",
                font_id.clone(),
                get_scheme().text_primary,
            );

            return;
        }

        font_id.size = 11.0;

        draw_borders(ui, self.rect, self.inner_rect);

        self.draw_x_axis(ui, &font_id);
        draw_y_axis(ui, self.bounds, self.steps_y, self.rect, self.inner_rect);

        if let Some(pointer_pos) = pointer_pos {
            if self.inner_rect.contains(pointer_pos) && ui.ui_contains_pointer() {
                let plot_point = self.bounds.screen_pos_to_value(self.rect, pointer_pos);
                draw_y_axis_flag(ui, pointer_pos, plot_point.y, self.inner_rect, font_id);

                let inner_point_pos = pointer_pos - self.rect.min;
                let timestamp = Timestamp(
                    (((inner_point_pos.x / self.rect.width()) as f64 * self.bounds.width()
                        + self.bounds.min_x) as i64)
                        + self.earliest_timestamp.0,
                );

                draw_cursor(
                    ui,
                    pointer_pos,
                    inner_point_pos.x,
                    self.rect,
                    self.inner_rect,
                );

                // Draw highlight circles on lines

                for ((_, _), (entity, color)) in graph_state.enabled_lines.iter() {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line_handle) = line_handle.as_timeseries() else {
                        continue;
                    };
                    let Some(line) = lines.get(line_handle) else {
                        continue;
                    };
                    let Some((timestamp, y)) = line.data.get_nearest(timestamp) else {
                        continue;
                    };
                    let value = DVec2::new(self.timestamp_to_x(timestamp), *y as f64);
                    let pos = self.bounds.value_to_screen_pos(self.rect, value);
                    ui.painter().circle(
                        pos,
                        4.5,
                        get_scheme().bg_secondary,
                        egui::Stroke::new(2.0, *color),
                    );
                }

                self.draw_modal(
                    ui,
                    lines,
                    line_handles,
                    graph_state,
                    collected_graph_data,
                    pointer_pos,
                    timestamp,
                );
            }
        }

        if self.selected_range.contains(&self.current_timestamp) {
            let tick_pos = self
                .bounds
                .value_to_screen_pos(
                    self.rect,
                    (self.timestamp_to_x(self.current_timestamp), 0.0).into(),
                )
                .x;

            draw_tick_mark(ui, self.rect, self.inner_rect, tick_pos, *scrub_icon);
        }
    }

    fn timestamp_to_x(&self, timestamp: Timestamp) -> f64 {
        (timestamp.0 - self.earliest_timestamp.0) as f64
    }

    fn visible_time_range(&self) -> Range<Timestamp> {
        Timestamp(self.bounds.min_x as i64 + self.earliest_timestamp.0)
            ..Timestamp(self.bounds.max_x as i64 + self.earliest_timestamp.0)
    }
}

pub fn draw_y_axis(
    ui: &mut egui::Ui,
    bounds: PlotBounds,
    steps_y: usize,
    rect: egui::Rect,
    inner_rect: egui::Rect,
) {
    let border_stroke = egui::Stroke::new(1.0, get_scheme().border_primary);
    let scheme = get_scheme();
    let mut font_id = egui::TextStyle::Monospace.resolve(ui.style());
    font_id.size = 11.0;

    let draw_tick = |tick| {
        let value = DVec2::new(bounds.min_x, tick);
        let screen_pos = bounds.value_to_screen_pos(rect, value);
        let screen_pos = egui::pos2(inner_rect.min.x, screen_pos.y);
        ui.painter().line_segment(
            [screen_pos, screen_pos - egui::vec2(NOTCH_LENGTH, 0.0)],
            border_stroke,
        );

        ui.painter().text(
            screen_pos - egui::vec2(NOTCH_LENGTH + Y_AXIS_LABEL_MARGIN, 0.0),
            egui::Align2::RIGHT_CENTER,
            format_num(tick),
            font_id.clone(),
            scheme.text_primary,
        );
    };
    if !bounds.min_y.is_finite() || !bounds.max_y.is_finite() {
        return;
    }
    if bounds.min_y <= 0.0 {
        let step_size = pretty_round((bounds.max_y - bounds.min_y) / steps_y as f64);
        if !step_size.is_normal() {
            return;
        }
        let mut i = 0.0;

        while i < bounds.max_y {
            draw_tick(i);
            i += step_size;
        }
        draw_tick(i);

        let mut i = 0.0;
        while i > bounds.min_y {
            draw_tick(i);
            i -= step_size;
        }
        draw_tick(i);
    } else {
        let step_size = pretty_round(bounds.height() / steps_y as f64);
        let steps_y = (0..=steps_y).map(|i| bounds.min_y + (i as f64) * step_size);

        for y_step in steps_y {
            draw_tick(y_step);
        }
    }
}

pub fn draw_y_axis_flag(
    ui: &mut egui::Ui,
    pointer_pos: egui::Pos2,
    value: f64,
    border_rect: egui::Rect,
    font_id: egui::FontId,
) {
    let label_size = egui::vec2(Y_AXIS_FLAG_WIDTH, Y_AXIS_FLAG_HEIGHT);
    let label_margin = Y_AXIS_FLAG_MARGIN;

    let pointer_rect = egui::Rect::from_center_size(
        egui::pos2(
            border_rect.min.x - ((label_size.x / 2.0) + label_margin),
            pointer_pos.y,
        ),
        label_size,
    );

    ui.painter().rect(
        pointer_rect,
        egui::CornerRadius::same(2),
        get_scheme().text_primary,
        egui::Stroke::NONE,
        egui::StrokeKind::Middle,
    );

    ui.painter().text(
        pointer_rect.center(),
        egui::Align2::CENTER_CENTER,
        format_num(value),
        font_id,
        get_scheme().bg_secondary,
    );
}

pub fn draw_cursor(
    ui: &mut egui::Ui,
    pointer_pos: egui::Pos2,
    x_offset: f32,
    rect: egui::Rect,
    inner_rect: egui::Rect,
) {
    ui.painter().vline(
        x_offset + rect.min.x,
        0.0..=inner_rect.max.y,
        egui::Stroke::new(1.0, get_scheme().border_primary),
    );

    ui.painter().hline(
        inner_rect.min.x..=inner_rect.max.x,
        pointer_pos.y,
        egui::Stroke::new(1.0, get_scheme().border_primary),
    );
}

pub fn draw_borders(ui: &mut egui::Ui, rect: egui::Rect, inner_rect: egui::Rect) {
    // draw bg
    let border_bg_color = get_scheme().bg_secondary.opacity(0.9);
    let y_bg_rect = egui::Rect::from_min_max(rect.min, inner_rect.min).with_max_y(rect.max.y);
    ui.painter()
        .rect_filled(y_bg_rect, CornerRadius::ZERO, border_bg_color);
    let x_bg_rect = egui::Rect::from_min_max(inner_rect.max, rect.max).with_min_x(inner_rect.min.x);
    ui.painter()
        .rect_filled(x_bg_rect, CornerRadius::ZERO, border_bg_color);

    let border_stroke = egui::Stroke::new(1.0, get_scheme().border_primary);
    let left_border = [inner_rect.left_top(), inner_rect.left_bottom()];
    ui.painter().line_segment(left_border, border_stroke);

    let bottom_border = [inner_rect.left_bottom(), inner_rect.right_bottom()];
    ui.painter().line_segment(bottom_border, border_stroke);
}

pub fn draw_tick_mark(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    inner_rect: egui::Rect,
    tick_pos: f32,
    scrub_icon: egui::TextureId,
) {
    let scrub_height = 12.0 * TICK_MARK_LINE_WIDTH;
    let scrub_width = scrub_height * TICK_MARK_ASPECT_RATIO;

    ui.painter().vline(
        tick_pos,
        rect.min.y..=inner_rect.max.y,
        egui::Stroke::new(TICK_MARK_LINE_WIDTH, get_scheme().text_primary),
    );

    let scrub_center = egui::pos2(tick_pos, rect.min.y + (scrub_height * 0.5));
    let scrub_rect =
        egui::Rect::from_center_size(scrub_center, egui::vec2(scrub_width, scrub_height));

    ui.painter().image(
        scrub_icon,
        scrub_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        get_scheme().text_primary,
    );
}

pub fn sync_bounds(
    graph_state: &mut GraphState,
    selected_range: Range<Timestamp>,
    earliest_timestamp: Timestamp,
    rect: egui::Rect,
    inner_rect: egui::Rect,
) -> PlotBounds {
    let (y_min, y_max) = (graph_state.y_range.start, graph_state.y_range.end);
    let outer_ratio = (rect.size() / inner_rect.size()).as_dvec2();
    let pan_offset = graph_state.pan_offset.as_dvec2() * DVec2::new(-1.0, 1.0);
    PlotBounds::from_lines(&selected_range, earliest_timestamp, y_min, y_max)
        .zoom_at(outer_ratio, DVec2::new(1.0, 0.5)) // zoom the bounds out so the graph takes up the entire screen
        .offset_by_norm(pan_offset) // pan the bounds by the amount the cursor has moved
        .zoom(graph_state.zoom_factor.as_dvec2()) // zoom the bounds based on the current zoom factor
        .normalize() // clamp the bounds so max > min
}

pub fn auto_y_bounds(
    mut graph_states: Query<&mut GraphState>,
    selected_range: Res<SelectedTimeRange>,
    line_handles: Query<&LineHandle>,
    mut lines: ResMut<Assets<Line>>,
    mut xy_lines: ResMut<Assets<XYLine>>,
) {
    for mut graph_state in &mut graph_states {
        if graph_state.auto_y_range {
            let mut y_min: Option<f32> = None;
            let mut y_max: Option<f32> = None;
            for (entity, _) in graph_state.enabled_lines.values() {
                let Ok(handle) = line_handles.get(*entity) else {
                    continue;
                };
                let Some(line) = handle.get(&mut lines, &mut xy_lines) else {
                    continue;
                };
                if let gpu::LineMut::Timeseries(line) = line {
                    let summary = line.data.range_summary(selected_range.0.clone());
                    if let Some(min) = summary.min {
                        if let Some(y_min) = &mut y_min {
                            *y_min = y_min.min(min);
                        } else {
                            y_min = Some(min)
                        }
                    }
                    if let Some(max) = summary.max {
                        if let Some(y_max) = &mut y_max {
                            *y_max = y_max.max(max);
                        } else {
                            y_max = Some(max)
                        }
                    }
                }
            }

            graph_state.y_range =
                y_min.unwrap_or_default() as f64..y_max.unwrap_or_default() as f64;
        }
    }
}

pub fn sync_graphs(
    mut graph_states: Query<&mut GraphState>,
    metadata_store: Res<ComponentMetadataRegistry>,
    mut collected_graph_data: ResMut<CollectedGraphData>,
    mut commands: Commands,
) {
    for mut graph_state in &mut graph_states {
        let graph_state = &mut *graph_state;

        for (component_path, component_values) in &graph_state.components {
            let component_id = &component_path.id;
            let Some(component_metadata) = metadata_store.get_metadata(component_id) else {
                continue;
            };
            let component_label = component_metadata.name.clone();
            let element_names = component_metadata.element_names();
            let component = collected_graph_data
                .components
                .entry(*component_id)
                .or_insert_with(|| {
                    PlotDataComponent::new(
                        component_label,
                        element_names
                            .split(',')
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect(),
                    )
                });

            for (value_index, (enabled, color)) in component_values.iter().enumerate() {
                let entity = graph_state
                    .enabled_lines
                    .get_mut(&(component_path.clone(), value_index));

                let Some(line) = component.lines.get(&value_index) else {
                    continue;
                };

                match (entity, enabled) {
                    (None, true) => {
                        let entity = commands
                            .spawn(LineBundle {
                                line: LineHandle::Timeseries(line.clone()),
                                uniform: LineUniform::new(
                                    graph_state.line_width,
                                    color.into_bevy(),
                                ),
                                config: LineConfig {
                                    render_layers: graph_state.render_layers.clone(),
                                },
                                line_visible_range: graph_state.visible_range.clone(),
                                graph_type: graph_state.graph_type,
                            })
                            .insert(LineWidgetWidth(graph_state.widget_width as usize))
                            .id();
                        graph_state
                            .enabled_lines
                            .insert((component_path.clone(), value_index), (entity, *color));
                    }
                    (Some((entity, _)), false) => {
                        commands.entity(*entity).despawn();
                        graph_state
                            .enabled_lines
                            .remove(&(component_path.clone(), value_index));
                    }
                    (Some((entity, graph_state_color)), true) => {
                        *graph_state_color = *color;
                        commands
                            .entity(*entity)
                            .try_insert(LineUniform::new(graph_state.line_width, color.into_bevy()))
                            .try_insert(graph_state.graph_type)
                            .try_insert(LineWidgetWidth(graph_state.widget_width as usize))
                            .try_insert(graph_state.visible_range.clone());
                    }
                    (None, false) => {}
                }
            }
        }

        graph_state
            .enabled_lines
            .retain(|(component_path, index), _| {
                graph_state
                    .components
                    .get(component_path)
                    .and_then(|component| component.get(*index))
                    .is_some()
            });
    }
}

#[derive(Debug, Clone, Default, Copy)]
pub struct PlotBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl PlotBounds {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn rounded(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        let min_y = sigfig_round(min_y, 2);
        let max_y = sigfig_round(max_y, 2);
        let min_x = sigfig_round(min_x, 2);
        let max_x = sigfig_round(max_x, 2);
        Self::new(min_x, min_y, max_x, max_y)
    }

    pub fn from_lines(
        baseline: &Range<Timestamp>,
        earliest_timestamp: Timestamp,
        min_y: f64,
        max_y: f64,
    ) -> Self {
        let (min_x, max_x) = (
            (baseline.start.0 - earliest_timestamp.0) as f64,
            (baseline.end.0 - earliest_timestamp.0) as f64,
        );

        let min_y = sigfig_round(min_y, 2);
        let max_y = sigfig_round(max_y, 2);
        Self::new(min_x, min_y, max_x, max_y)
    }

    pub fn min(&self) -> DVec2 {
        DVec2::new(self.min_x, self.min_y)
    }

    pub fn max(&self) -> DVec2 {
        DVec2::new(self.max_x, self.max_y)
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn size(&self) -> DVec2 {
        DVec2::new(self.width(), self.height())
    }

    pub fn range_x_f32(&self) -> RangeInclusive<f32> {
        (self.min_x as f32)..=(self.max_x as f32)
    }

    pub fn range_y_f32(&self) -> RangeInclusive<f32> {
        (self.min_y as f32)..=(self.max_y as f32)
    }

    pub fn offset_by_norm(self, norm_coord: DVec2) -> Self {
        self.offset(norm_coord * self.size())
    }

    pub fn offset(mut self, offset: DVec2) -> Self {
        self.min_x += offset.x;
        self.max_x += offset.x;
        self.min_y += offset.y;
        self.max_y += offset.y;
        self
    }

    pub fn timestamp_range(&self, earliest_timestamp: Timestamp) -> Range<Timestamp> {
        let min_x = (self.min_x as i64) + earliest_timestamp.0;
        let max_x = (self.max_x as i64) + earliest_timestamp.0;
        Timestamp(min_x)..Timestamp(max_x)
    }

    pub fn zoom_at(self, zoom_factor: DVec2, anchor: DVec2) -> Self {
        let offset = self.size() * (zoom_factor - DVec2::ONE);

        Self {
            min_x: self.min_x + -offset.x * anchor.x,
            max_x: self.max_x + offset.x * (1.0 - anchor.x),
            min_y: self.min_y + -offset.y * anchor.y,
            max_y: self.max_y + offset.y * (1.0 - anchor.y),
        }
    }

    pub fn zoom(self, zoom_factor: DVec2) -> Self {
        let offset = self.size() * (zoom_factor - DVec2::ONE);
        Self {
            min_x: self.min_x - offset.x / 2.0,
            max_x: self.max_x + offset.x / 2.0,
            min_y: self.min_y - offset.y / 2.0,
            max_y: self.max_y + offset.y / 2.0,
        }
    }

    pub fn normalize(mut self) -> Self {
        if self.min_x >= self.max_x {
            self.min_x = self.max_x.min(self.min_x);
            self.max_x = self.min_x + 1.0;
        }
        if self.min_y >= self.max_y {
            self.min_y = self.max_y.min(self.min_y);
            self.max_y = self.min_y + 1.0;
        }
        self
    }

    pub fn as_projection(&self) -> OrthographicProjection {
        let viewport_origin =
            DVec2::new(-self.min_x / self.width(), -self.min_y / self.height()).as_vec2();
        OrthographicProjection {
            near: 0.0,
            far: 1000.0,
            viewport_origin,
            scaling_mode: ScalingMode::Fixed {
                width: self.width() as f32,
                height: self.height() as f32,
            },
            scale: 1.0,
            area: Rect::new(
                self.min_x as f32,
                self.min_y as f32,
                self.max_x as f32,
                self.max_y as f32,
            ),
        }
    }

    pub fn screen_pos_to_value(&self, screen_rect: egui::Rect, pos: egui::Pos2) -> DVec2 {
        let offset = (pos - screen_rect.min).as_dvec2();
        let screen_to_value = self.size() / screen_rect.size().as_dvec2();
        self.min() + offset * screen_to_value
    }

    pub fn value_to_screen_pos(&self, screen_rect: egui::Rect, value: DVec2) -> egui::Pos2 {
        let offset = value - self.min();
        let offset = egui::vec2(offset.x as f32, offset.y as f32);
        let size = self.size();
        let value_to_screen = screen_rect.size() / egui::vec2(size.x as f32, size.y as f32);
        let screen_offset = offset * value_to_screen;
        egui::pos2(
            screen_rect.min.x + screen_offset.x,
            screen_rect.max.y - screen_offset.y,
        )
    }
}

fn sigfig_round(x: f64, mut digits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    digits -= x.abs().log10().ceil() as i32;
    let y = (10.0f64).powi(digits);
    if x.is_sign_positive() {
        (y * x).ceil() / y
    } else {
        (y * x).floor() / y
    }
}

pub fn pretty_round(num: f64) -> f64 {
    let mut multiplier = 1.0;
    let mut n = num;

    // Handle negative numbers
    let is_negative = n < 0.0;
    n = n.abs();

    // Find the appropriate multiplier for the decimal places
    while n < 1.0 {
        n *= 10.0;
        multiplier *= 10.0;
    }

    // Round to nearest 5
    let rounded = (n * 2.0).round() / 2.0;
    let result = rounded / multiplier;

    if is_negative { -result } else { result }
}

pub fn graph_touch(
    mut query: Query<(&mut GraphState, &Camera)>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    touch_tracker: Res<TouchTracker>,
) {
    let Ok(window) = primary_window.single() else {
        return;
    };

    let touch_gestures = touch_tracker.get_touch_gestures();

    let midpoint = match touch_gestures {
        TouchGestures::OneFinger(one_finger) => one_finger.midpoint,
        TouchGestures::TwoFinger(two_finger) => two_finger.midpoint,
        _ => return,
    };

    for (mut graph_state, cam) in query.iter_mut() {
        let Some(viewport_rect) = cam.logical_viewport_rect() else {
            continue;
        };

        let Some(viewport) = &cam.viewport else {
            continue;
        };

        if !viewport_rect.contains(midpoint) {
            continue;
        }
        let area = (viewport_rect.height() * viewport_rect.width()).sqrt();

        match touch_gestures {
            // orbit
            TouchGestures::OneFinger(gesture) => {
                let delta_device_pixels = gesture.motion;
                let delta = delta_device_pixels / viewport_rect.size() * graph_state.zoom_factor;
                graph_state.pan_offset += delta;
            }
            TouchGestures::TwoFinger(gesture) => {
                let cursor_pos = midpoint * window.scale_factor();
                let scroll_offset = gesture.pinch / area * window.scale_factor();
                let old_scale = graph_state.zoom_factor;
                graph_state.zoom_factor *= 1. - scroll_offset;
                graph_state.zoom_factor = graph_state.zoom_factor.clamp(Vec2::ZERO, Vec2::ONE);

                let cursor_pos = (cursor_pos - viewport.physical_position.as_vec2())
                    - viewport.physical_size.as_vec2() / 2.;
                let cursor_normalized_screen_pos = cursor_pos / viewport.physical_size.as_vec2();
                let cursor_normalized_screen_pos = Vec2::new(
                    cursor_normalized_screen_pos.x,
                    cursor_normalized_screen_pos.y,
                );

                let delta = (old_scale - graph_state.zoom_factor) * cursor_normalized_screen_pos;

                graph_state.pan_offset -= delta;
            }
            TouchGestures::None => {}
        }
    }
}

pub fn zoom_graph(
    mut query: Query<(&mut GraphState, &Camera)>,
    scroll_events: EventReader<MouseWheel>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    key_state: Res<LogicalKeyState>,
) {
    let scroll_offset = scroll_offset_from_events(scroll_events);
    if scroll_offset == 0. {
        return;
    }

    let Ok(window) = primary_window.single() else {
        return;
    };

    let cursor_pos = window.physical_cursor_position();

    for (mut graph_state, camera) in &mut query {
        let Some(cursor_pos) = cursor_pos else {
            continue;
        };

        let Some(viewport) = &camera.viewport else {
            continue;
        };

        let physical_size = viewport.physical_size.as_vec2();
        let physical_pos = viewport.physical_position.as_vec2();
        let viewport_rect = Rect::from_corners(physical_pos, physical_pos + physical_size);

        if !viewport_rect.contains(cursor_pos) {
            continue;
        }

        let offset_mask = if key_state.pressed(&Key::Control) {
            Vec2::new(1.0, 0.0)
        } else if key_state.pressed(&Key::Shift) {
            Vec2::new(0.0, 1.0)
        } else {
            Vec2::new(1.0, 1.0)
        };

        let old_scale = graph_state.zoom_factor;
        graph_state.zoom_factor *= 1. - scroll_offset * ZOOM_SENSITIVITY * offset_mask;
        graph_state.zoom_factor = graph_state.zoom_factor.clamp(Vec2::ZERO, Vec2::ONE);

        let cursor_pos = (cursor_pos - viewport.physical_position.as_vec2())
            - viewport.physical_size.as_vec2() / 2.;
        let cursor_normalized_screen_pos = cursor_pos / viewport.physical_size.as_vec2();
        let cursor_normalized_screen_pos = Vec2::new(
            cursor_normalized_screen_pos.x,
            cursor_normalized_screen_pos.y,
        );

        let delta = (old_scale - graph_state.zoom_factor) * cursor_normalized_screen_pos;

        graph_state.pan_offset -= delta * offset_mask;
    }
}

#[derive(Component)]
pub struct LastPos(Option<Vec2>);

pub fn pan_graph(
    mut query: Query<(Entity, &mut GraphState, &Camera, Option<&LastPos>)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
    key_state: Res<LogicalKeyState>,
    mut commands: Commands,
) {
    let Ok(window) = primary_window.single() else {
        return;
    };

    let cursor_pos = window.physical_cursor_position();
    let Some(cursor_pos) = cursor_pos else { return };

    for (entity, mut graph_state, camera, last_pos) in &mut query {
        let last_pos = last_pos.and_then(|p| p.0);
        let Some(viewport) = &camera.viewport else {
            continue;
        };

        let physical_size = viewport.physical_size.as_vec2();
        let physical_pos = viewport.physical_position.as_vec2();
        let viewport_rect = Rect::from_corners(physical_pos, physical_pos + physical_size);

        if !viewport_rect.contains(cursor_pos) {
            if let Ok(mut e) = commands.get_entity(entity) {
                e.try_insert(LastPos(None));
            }
            continue;
        }

        if mouse_buttons.just_pressed(MouseButton::Left) {
            if let Ok(mut e) = commands.get_entity(entity) {
                e.try_insert(LastPos(Some(cursor_pos)));
            }
            continue;
        }

        if !mouse_buttons.pressed(MouseButton::Left) {
            if let Ok(mut e) = commands.get_entity(entity) {
                e.try_insert(LastPos(None));
            }
            continue;
        }

        let Some(last_pos) = last_pos else {
            continue;
        };

        let delta_device_pixels = cursor_pos - last_pos;

        let offset_mask = if key_state.pressed(&Key::Control) {
            Vec2::new(1.0, 0.0)
        } else if key_state.pressed(&Key::Shift) {
            Vec2::new(0.0, 1.0)
        } else {
            Vec2::new(1.0, 1.0)
        };

        let delta =
            delta_device_pixels / viewport_rect.size() * graph_state.zoom_factor * offset_mask;
        graph_state.pan_offset += delta;

        if let Ok(mut e) = commands.get_entity(entity) {
            e.try_insert(LastPos(Some(cursor_pos)));
        }
    }
}

pub fn reset_graph(
    mut last_click: Local<Option<Instant>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut query: Query<(&mut GraphState, &Camera)>,
    primary_window: Query<&Window, With<PrimaryWindow>>,
) {
    if mouse_buttons.just_released(MouseButton::Left) {
        *last_click = Some(Instant::now());
    }

    let Ok(window) = primary_window.single() else {
        return;
    };

    let cursor_pos = window.physical_cursor_position();
    let Some(cursor_pos) = cursor_pos else { return };

    if mouse_buttons.just_pressed(MouseButton::Left)
        && last_click
            .map(|t| t.elapsed() < Duration::from_millis(250))
            .unwrap_or_default()
    {
        for (mut graph_state, camera) in &mut query {
            let Some(viewport) = &camera.viewport else {
                continue;
            };

            let physical_size = viewport.physical_size.as_vec2();
            let physical_pos = viewport.physical_position.as_vec2();
            let viewport_rect = Rect::from_corners(physical_pos, physical_pos + physical_size);
            if !viewport_rect.contains(cursor_pos) {
                continue;
            }
            graph_state.pan_offset = Vec2::ZERO;
            graph_state.zoom_factor = Vec2::ONE;
        }
    }
}

fn scroll_offset_from_events(mut scroll_events: EventReader<MouseWheel>) -> f32 {
    let pixels_per_line = SCROLL_PIXELS_PER_LINE;
    scroll_events
        .read()
        .map(|ev| match ev.unit {
            MouseScrollUnit::Pixel => ev.y,
            MouseScrollUnit::Line => ev.y * pixels_per_line,
        })
        .sum::<f32>()
}

pub trait Vec2Ext {
    fn as_dvec2(&self) -> DVec2;
}

impl Vec2Ext for egui::Vec2 {
    fn as_dvec2(&self) -> DVec2 {
        DVec2::new(self.x as f64, self.y as f64)
    }
}
