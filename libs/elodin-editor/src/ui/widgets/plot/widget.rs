use crate::editor_cam_touch::*;
use bevy::{
    asset::Assets,
    ecs::{
        entity::Entity,
        event::EventReader,
        hierarchy::ChildOf,
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
use egui::{Color32, CornerRadius, Frame, Margin, Pos2, RichText, Stroke};
use impeller2::types::{ComponentId, EntityId, Timestamp};
use impeller2_bevy::{ComponentMetadataRegistry, EntityMap};
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp, EntityMetadata};
use std::time::{Duration, Instant};
use std::{
    fmt::Debug,
    ops::{Range, RangeInclusive},
};

use crate::{
    Offset, SelectedTimeRange, TimeRangeBehavior,
    plugins::LogicalKeyState,
    ui::{
        colors::{self, ColorExt, with_opacity},
        utils::format_num,
        widgets::{
            WidgetSystem,
            plot::{
                CollectedGraphData, GraphState, Line,
                gpu::{LineBundle, LineConfig, LineUniform},
            },
            time_label::{PrettyDuration, time_label},
            timeline::DurationExt,
        },
    },
};

use super::{
    PlotDataComponent, PlotDataEntity,
    gpu::{LineHandle, LineVisibleRange, LineWidgetWidth},
};

#[derive(SystemParam)]
pub struct PlotWidget<'w, 's> {
    collected_graph_data: ResMut<'w, CollectedGraphData>,
    graphs_state: Query<'w, 's, &'static mut GraphState>,
    lines: ResMut<'w, Assets<Line>>,
    commands: Commands<'w, 's>,
    entity_map: Res<'w, EntityMap>,
    entity_metadata: Query<'w, 's, &'static EntityMetadata>,
    metadata_store: Res<'w, ComponentMetadataRegistry>,
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
        let mut state = state.get_mut(world);

        let Ok(mut graph_state) = state.graphs_state.get_mut(id) else {
            return;
        };

        Plot::from_state(
            ui,
            &mut state.collected_graph_data,
            &mut graph_state,
            &mut state.lines,
            &mut state.commands,
            id,
            &state.entity_map,
            &state.entity_metadata,
            &state.metadata_store,
            state.selected_time_range.clone(),
            state.earliest_timestamp.0,
        )
        .current_timestamp(state.current_timestamp.0)
        .render(
            ui,
            &state.lines,
            &state.line_query,
            &state.collected_graph_data,
            &mut graph_state,
            &scrub_icon,
            &mut state.time_range_behavior,
        );
    }
}

#[derive(Debug)]
pub struct Plot {
    tick_range: Range<Timestamp>,
    current_timestamp: Timestamp,
    earliest_timestamp: Timestamp,
    bounds: PlotBounds,
    rect: egui::Rect,
    inner_rect: egui::Rect,

    steps_x: usize,
    steps_y: usize,

    notch_length: f32,
    axis_label_margin: f32,

    text_color: egui::Color32,
    border_stroke: egui::Stroke,

    show_modal: bool,
}

fn range_x_from_rect(rect: &egui::Rect) -> RangeInclusive<f32> {
    rect.max.x..=rect.min.x
}

fn range_y_from_rect(rect: &egui::Rect) -> RangeInclusive<f32> {
    rect.max.y..=rect.min.y
}

pub const MARGIN: egui::Margin = egui::Margin {
    left: 60,
    right: 0,
    top: 30,
    bottom: 40,
};

pub fn get_inner_rect(rect: egui::Rect) -> egui::Rect {
    egui::Rect {
        min: egui::pos2(
            rect.min.x + (MARGIN.left as f32),
            rect.min.y + (MARGIN.top as f32),
        ),
        max: egui::pos2(
            rect.max.x - (MARGIN.right as f32),
            rect.max.y - (MARGIN.bottom as f32),
        ),
    }
}

impl Plot {
    /// Calculate bounds and point positions based on the current UI allocation
    /// Should be run last, right before render
    #[allow(clippy::too_many_arguments)]
    pub fn from_state(
        ui: &egui::Ui,
        collected_graph_data: &mut CollectedGraphData,
        graph_state: &mut GraphState,
        lines: &mut Assets<Line>,
        commands: &mut Commands,
        graph_id: Entity,
        entity_map: &EntityMap,
        entity_metadata: &Query<&EntityMetadata>,
        metadata_store: &ComponentMetadataRegistry,
        selected_range: SelectedTimeRange,
        earliest_timestamp: Timestamp,
    ) -> Self {
        //let earliest_timestamp = selected_range.0.start;
        let rect = ui.max_rect();
        let inner_rect = get_inner_rect(rect);

        let mut tick_range = selected_range.0.clone();
        if tick_range.start == tick_range.end {
            tick_range.end += Duration::from_secs(10);
        }

        // calc bounds

        let (y_min, y_max) = if graph_state.auto_y_range {
            let mut y_min: Option<f32> = None;
            let mut y_max: Option<f32> = None;

            for (entity_id, components) in &graph_state.entities {
                if let Some(entity) = collected_graph_data.entities.get(entity_id) {
                    for (component_id, component_values) in components {
                        if let Some(component) = entity.components.get(component_id) {
                            for (value_index, (enabled, _)) in component_values.iter().enumerate() {
                                if *enabled {
                                    if let Some(line) = component
                                        .lines
                                        .get(&value_index)
                                        .and_then(|handle| lines.get_mut(handle))
                                    {
                                        let summary =
                                            line.data.range_summary(selected_range.0.clone());
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
                            }
                        }
                    }
                }
            }
            (
                y_min.unwrap_or_default() as f64,
                y_max.unwrap_or_default() as f64,
            )
        } else {
            (graph_state.y_range.start, graph_state.y_range.end)
        };
        graph_state.y_range = y_min..y_max;

        let mut steps_y = ((inner_rect.height() / 50.0) as usize).max(1);
        if steps_y % 2 != 0 {
            steps_y += 1;
        }
        let steps_x = ((inner_rect.width() / 75.0) as usize).max(1);

        let original_bounds = PlotBounds::from_lines(&tick_range, earliest_timestamp, y_min, y_max);

        //graph_state = (original_bounds.max_y..original_bounds.min_y)

        let bounds_size = DVec2::new(original_bounds.width(), original_bounds.height());
        let outer_ratio = rect.size() / inner_rect.size();
        let outer_ratio = Vec2::new(outer_ratio.x, outer_ratio.y).as_dvec2();
        let pan_offset = graph_state.pan_offset.as_dvec2() * bounds_size * DVec2::new(-1.0, 1.0);
        let mut bounds = original_bounds.clone().offset(pan_offset * outer_ratio);
        let new_bounds = bounds_size * graph_state.zoom_factor.as_dvec2();
        let offset = new_bounds - bounds_size;
        bounds.min_x -= offset.x / 2.0;
        bounds.max_x += offset.x / 2.0;
        bounds.min_y -= offset.y / 2.0;
        bounds.max_y += offset.y / 2.0;

        if bounds.min_x >= bounds.max_x {
            bounds.min_x = bounds.max_x.min(bounds.min_x);
            bounds.max_x = bounds.min_x + 1.0;
        }
        if bounds.min_y >= bounds.max_y {
            bounds.min_y = bounds.max_y.min(bounds.min_y);
            bounds.max_y = bounds.min_y + 1.0;
        }

        let y_range = bounds.max_y - bounds.min_y;
        let full_y_range = (rect.height() / inner_rect.height()) as f64 * y_range;
        let y_delta = full_y_range - y_range;
        let mut new_bounds = bounds.clone();
        new_bounds.min_y -= y_delta / 2.0;
        new_bounds.max_y -= y_delta / 2.0;
        let x_range = 1.0 / (bounds.max_x - bounds.min_x);
        let width = new_bounds.max_x - new_bounds.min_x;
        let viewport_origin = DVec2::new(
            (offset.x / 2.0 - pan_offset.x) * x_range
                - (selected_range.0.start.0 - earliest_timestamp.0) as f64 / width,
            -(new_bounds.min_y / full_y_range),
        );
        let viewport_origin = viewport_origin.as_vec2();
        let orthographic_projection = OrthographicProjection {
            near: 0.0,
            far: 1000.0,
            viewport_origin,
            scaling_mode: ScalingMode::Fixed {
                width: width as f32,
                height: full_y_range as f32,
            },
            scale: 1.0,
            area: Rect::new(
                new_bounds.min_x as f32,
                new_bounds.min_y as f32,
                new_bounds.max_x as f32,
                new_bounds.max_y as f32,
            ),
        };
        commands
            .entity(graph_id)
            .try_insert(Projection::Orthographic(orthographic_projection));

        let min_x = -viewport_origin.x as f64 * width;
        let max_x = width + min_x;

        let line_visible_range = Timestamp(min_x as i64 + earliest_timestamp.0)
            ..Timestamp(max_x as i64 + earliest_timestamp.0);
        let bounds = PlotBounds::new(min_x as f64, bounds.min_y, max_x as f64, bounds.max_y);

        // calc lines

        for (entity_id, components) in &graph_state.entities {
            let entity = collected_graph_data
                .entities
                .entry(*entity_id)
                .or_insert_with(|| {
                    let label = entity_map
                        .get(entity_id)
                        .and_then(|id| entity_metadata.get(*id).ok())
                        .map_or(format!("E[{}]", entity_id.0), |metadata| {
                            metadata.name.to_owned()
                        });
                    PlotDataEntity {
                        label,
                        components: Default::default(),
                    }
                });

            for (component_id, component_values) in components {
                let Some(component_metadata) = metadata_store.get_metadata(component_id) else {
                    continue;
                };
                let component_label = component_metadata.name.clone();
                let element_names = component_metadata.element_names();
                let component = entity.components.entry(*component_id).or_insert_with(|| {
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
                    let entity = graph_state.enabled_lines.get_mut(&(
                        *entity_id,
                        *component_id,
                        value_index,
                    ));

                    let Some(line) = component.lines.get(&value_index) else {
                        continue;
                    };

                    match (entity, enabled) {
                        (None, true) => {
                            let entity = commands
                                .spawn(LineBundle {
                                    line: LineHandle(line.clone()),
                                    uniform: LineUniform::new(
                                        graph_state.line_width,
                                        color.into_bevy(),
                                    ),
                                    config: LineConfig {
                                        render_layers: graph_state.render_layers.clone(),
                                    },
                                    line_visible_range: LineVisibleRange(
                                        line_visible_range.clone(),
                                    ),
                                    graph_type: graph_state.graph_type,
                                })
                                .insert(ChildOf(graph_id))
                                .insert(LineWidgetWidth(ui.max_rect().width() as usize))
                                .id();
                            graph_state
                                .enabled_lines
                                .insert((*entity_id, *component_id, value_index), (entity, *color));
                        }
                        (Some((entity, _)), false) => {
                            commands.entity(*entity).despawn();
                            graph_state.enabled_lines.remove(&(
                                *entity_id,
                                *component_id,
                                value_index,
                            ));
                        }
                        (Some((entity, graph_state_color)), true) => {
                            *graph_state_color = *color;
                            commands
                                .entity(*entity)
                                .try_insert(LineUniform::new(
                                    graph_state.line_width,
                                    color.into_bevy(),
                                ))
                                .try_insert(graph_state.graph_type)
                                .try_insert(LineWidgetWidth(ui.max_rect().width() as usize))
                                .try_insert(LineVisibleRange(line_visible_range.clone()));
                        }
                        (None, false) => {}
                    }
                }
            }
        }
        graph_state
            .enabled_lines
            .retain(|(entity_id, component_id, index), _| {
                graph_state
                    .entities
                    .get(entity_id)
                    .and_then(|entity| entity.get(component_id))
                    .and_then(|component| component.get(*index))
                    .is_some()
            });

        Self {
            tick_range,
            current_timestamp: Timestamp::EPOCH,
            bounds,
            rect,
            inner_rect,
            steps_x,
            steps_y,
            notch_length: 10.0,
            axis_label_margin: 5.0,
            text_color: colors::PRIMARY_CREAME,
            border_stroke: egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),

            show_modal: true,
            earliest_timestamp,
        }
    }

    pub fn current_timestamp(mut self, tick: Timestamp) -> Self {
        self.current_timestamp = tick;
        self
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
        let start = self.tick_range.start;
        let end = visible_time_range.end;
        let start_count = (visible_time_range.start.0 - start.0) / step_size_micro as i64 - 1;
        let end_count = (end.0 - start.0) / step_size_micro as i64 + 1;

        for i in start_count..=end_count {
            let offset_float = step_size_float * i as f64;
            let offset = hifitime::Duration::from_microseconds(offset_float);
            let x_pos = self.x_pos_from_timestamp(Timestamp(start.0 + offset_float as i64));

            ui.painter().line_segment(
                [
                    egui::pos2(x_pos, self.inner_rect.max.y),
                    egui::pos2(x_pos, self.inner_rect.max.y + (self.notch_length)),
                ],
                self.border_stroke,
            );

            ui.painter().text(
                egui::pos2(
                    x_pos,
                    self.inner_rect.max.y + (self.notch_length + self.axis_label_margin),
                ),
                egui::Align2::CENTER_TOP,
                PrettyDuration(offset).to_string(),
                font_id.clone(),
                self.text_color,
            );
        }
    }

    fn draw_y_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let draw_tick = |tick| {
            let mut y_position = PlotPoint::from_plot_point(self, self.bounds.min_x, tick).pos2;
            y_position.x = self.inner_rect.min.x;
            ui.painter().line_segment(
                [
                    egui::pos2(y_position.x, y_position.y),
                    egui::pos2(y_position.x - self.notch_length, y_position.y),
                ],
                self.border_stroke,
            );

            ui.painter().text(
                egui::pos2(
                    y_position.x - (self.notch_length + self.axis_label_margin),
                    y_position.y,
                ),
                egui::Align2::RIGHT_CENTER,
                format_num(tick),
                font_id.clone(),
                self.text_color,
            );
        };
        if !self.bounds.min_y.is_finite() || !self.bounds.max_y.is_finite() {
            return;
        }
        if self.bounds.min_y <= 0.0 {
            let step_size =
                pretty_round((self.bounds.max_y - self.bounds.min_y) / self.steps_y as f64);
            if !step_size.is_normal() {
                return;
            }
            let mut i = 0.0;

            while i < self.bounds.max_y {
                draw_tick(i);
                i += step_size;
            }
            draw_tick(i);

            let mut i = 0.0;
            while i > self.bounds.min_y {
                draw_tick(i);
                i -= step_size;
            }
            draw_tick(i);
        } else {
            let step_size = pretty_round(self.bounds.height() / self.steps_y as f64);
            let steps_y = (0..=self.steps_y)
                .map(|i| self.bounds.min_y + (i as f64) * step_size)
                .collect::<Vec<f64>>();

            for y_step in steps_y {
                draw_tick(y_step);
            }
        }
    }

    fn draw_y_axis_flag(
        &self,
        ui: &mut egui::Ui,
        pointer_plot_point: PlotPoint,
        border_rect: egui::Rect,
        font_id: egui::FontId,
    ) {
        let label_size = egui::vec2(70.0, 20.0);
        let label_margin = 4.0;

        let pointer_rect = egui::Rect::from_center_size(
            egui::pos2(
                border_rect.min.x - ((label_size.x / 2.0) + label_margin),
                pointer_plot_point.pos2.y,
            ),
            label_size,
        );

        ui.painter().rect(
            pointer_rect,
            egui::CornerRadius::same(2),
            colors::MINT_DEFAULT,
            egui::Stroke::NONE,
            egui::StrokeKind::Middle,
        );

        ui.painter().text(
            pointer_rect.center(),
            egui::Align2::CENTER_CENTER,
            format_num(pointer_plot_point.y),
            font_id,
            colors::PRIMARY_ONYX,
        );
    }

    fn draw_cursor(
        &self,
        ui: &mut egui::Ui,
        pointer_pos: egui::Pos2,
        x_offset: f32,
        border_rect: egui::Rect,
    ) {
        ui.painter().vline(
            x_offset + border_rect.min.x,
            0.0..=self.inner_rect.max.y,
            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
        );

        ui.painter().hline(
            border_rect.min.x..=border_rect.max.x,
            pointer_pos.y,
            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
        );
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
        let modal_width = 250.0;
        let margin = 20.0;

        let anchor_left = pointer_pos.x + modal_width + margin < self.rect.right(); // NOTE: might want to replace pointer_pos with x_offset from `render`

        let (pivot, fixed_pos) = if anchor_left {
            (
                egui::Align2::LEFT_TOP,
                egui::pos2(pointer_pos.x + margin, pointer_pos.y + margin),
            )
        } else {
            (
                egui::Align2::RIGHT_TOP,
                egui::pos2(pointer_pos.x - margin, pointer_pos.y + margin),
            )
        };

        egui::Window::new("plot_modal")
            .pivot(pivot)
            .title_bar(false)
            .resizable(false)
            .fixed_pos(fixed_pos)
            .fixed_size(egui::vec2(modal_width, self.inner_rect.height() / 2.))
            .frame(
                Frame::default()
                    .inner_margin(Margin::same(16))
                    .stroke(Stroke::new(1.0, colors::BORDER_GREY))
                    .corner_radius(CornerRadius::same(4))
                    .fill(colors::PRIMARY_SMOKE)
                    .shadow(egui::epaint::Shadow {
                        offset: [0, 5],
                        blur: 8,
                        spread: 2,
                        color: Color32::from_black_alpha(191),
                    }),
            )
            .show(ui.ctx(), |ui| {
                let time: hifitime::Epoch = timestamp.into();

                ui.add(time_label(time));
                let offset = hifitime::Duration::from_microseconds(
                    (timestamp.0 - self.tick_range.start.0) as f64,
                );
                ui.label(PrettyDuration(offset).to_string());
                let mut current_entity_id: Option<EntityId> = None;
                let mut current_component_id: Option<ComponentId> = None;
                for ((entity_id, component_id, line_index), (entity, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line) = lines.get(&line_handle.0) else {
                        continue;
                    };
                    if current_entity_id.as_ref() != Some(entity_id) {
                        ui.add_space(8.0);
                        ui.add(egui::Separator::default().grow(16.0 * 2.0));
                        ui.add_space(8.0);
                        current_entity_id = Some(*entity_id);
                        current_component_id = None;
                        if let Some(entity_data) = collected_graph_data.get_entity(entity_id) {
                            ui.label(egui::RichText::new(entity_data.label.to_owned()).size(13.0));
                        }
                    }

                    if current_component_id.as_ref() != Some(component_id) {
                        current_component_id = Some(*component_id);
                        if let Some(component_data) =
                            collected_graph_data.get_component(entity_id, component_id)
                        {
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(component_data.label.to_owned())
                                    .size(11.0)
                                    .color(with_opacity(colors::PRIMARY_CREAME, 0.6)),
                            );
                            ui.add_space(8.0);
                        }
                    }

                    let Some(line_data) = collected_graph_data
                        .get_line(entity_id, component_id, *line_index)
                        .and_then(|h| lines.get(h))
                    else {
                        continue;
                    };

                    ui.horizontal(|ui| {
                        ui.style_mut().override_font_id =
                            Some(egui::TextStyle::Monospace.resolve(ui.style_mut()));
                        ui.with_layout(Layout::top_down_justified(Align::LEFT), |ui| {
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(8.0, 8.0),
                                    egui::Sense::click(),
                                );
                                ui.painter().rect(
                                    rect,
                                    egui::CornerRadius::same(2),
                                    *color,
                                    egui::Stroke::NONE,
                                    egui::StrokeKind::Middle,
                                );
                                ui.add_space(6.);
                                ui.label(RichText::new(line_data.label.clone()).size(11.0));
                            })
                        });
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

    fn get_border_rect(&self, rect: egui::Rect) -> egui::Rect {
        egui::Rect {
            min: egui::pos2(
                rect.min.x + MARGIN.left as f32,
                rect.min.y + MARGIN.top as f32,
            ),
            max: egui::pos2(
                rect.max.x - MARGIN.right as f32,
                rect.max.y - MARGIN.bottom as f32,
            ),
        }
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

        let border_rect = self.get_border_rect(self.rect);

        // Style

        let mut font_id = egui::TextStyle::Monospace.resolve(ui.style());

        // Draw inner container

        if graph_state.enabled_lines.is_empty() {
            ui.painter().text(
                self.rect.center(),
                egui::Align2::CENTER_CENTER,
                "NO DATA POINTS SELECTED",
                font_id.clone(),
                colors::WHITE,
            );

            return;
        }

        font_id.size = 11.0;

        // Draw borders

        let left_border = [border_rect.left_top(), border_rect.left_bottom()];
        ui.painter().line_segment(left_border, self.border_stroke);

        let bottom_border = [border_rect.left_bottom(), border_rect.right_bottom()];
        ui.painter().line_segment(bottom_border, self.border_stroke);

        // Draw axis

        self.draw_x_axis(ui, &font_id);
        self.draw_y_axis(ui, &font_id);

        // Draw lines

        if let Some(pointer_pos) = pointer_pos {
            if self.inner_rect.contains(pointer_pos) && ui.ui_contains_pointer() {
                let pointer_plot_point = PlotPoint::from_plot_pos2(self, pointer_pos);

                self.draw_y_axis_flag(ui, pointer_plot_point, border_rect, font_id);

                let inner_point_pos = pointer_pos - self.inner_rect.min;
                let timestamp = Timestamp(
                    (((inner_point_pos.x / self.inner_rect.width()) as f64 * self.bounds.width()
                        + self.bounds.min_x) as i64)
                        + self.earliest_timestamp.0,
                );

                self.draw_cursor(ui, pointer_pos, inner_point_pos.x, self.inner_rect);

                // Draw highlight circles on lines

                for ((_entity_id, _component_id, _line_index), (entity, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line) = lines.get(&line_handle.0) else {
                        continue;
                    };
                    let Some((timestamp, y)) = line.data.get_nearest(timestamp) else {
                        continue;
                    };
                    let x = self.x_pos_from_timestamp(timestamp);
                    let y_offset = *y as f64 - self.bounds.min_y;
                    let y_range = self.bounds.max_y - self.bounds.min_y;
                    let y_pos = y_offset as f32 * (self.inner_rect.height() / y_range as f32);
                    let y_pos = self.inner_rect.max.y - y_pos;
                    let pos = Pos2::new(x, y_pos);
                    ui.painter().circle(
                        pos,
                        6.0,
                        colors::PRIMARY_SMOKE,
                        egui::Stroke::new(2.0, *color),
                    );
                }

                if self.show_modal {
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
        }

        // Draw a line for the current_tick
        if self.tick_range.contains(&self.current_timestamp) {
            let line_width = 1.0;
            let aspect_ratio = 12.0 / 30.0;

            let scrub_height = 12.0 * line_width;
            let scrub_width = scrub_height * aspect_ratio;

            let tick_pos = self.x_pos_from_timestamp(self.current_timestamp);
            ui.painter().vline(
                tick_pos,
                self.rect.min.y..=border_rect.max.y,
                egui::Stroke::new(line_width, colors::WHITE),
            );

            let scrub_center = egui::pos2(tick_pos, self.rect.min.y + (scrub_height * 0.5));
            let scrub_rect =
                egui::Rect::from_center_size(scrub_center, egui::vec2(scrub_width, scrub_height));

            ui.painter().image(
                *scrub_icon,
                scrub_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                colors::WHITE,
            );
        }
    }

    fn x_pos_from_timestamp(&self, timestamp: Timestamp) -> f32 {
        (self.inner_rect.width() as f64 / self.bounds.width()
            * ((timestamp.0 - self.earliest_timestamp.0) as f64 - self.bounds.min_x)) as f32
            + self.inner_rect.min.x
    }

    fn visible_time_range(&self) -> Range<Timestamp> {
        Timestamp(self.bounds.min_x as i64 + self.earliest_timestamp.0)
            ..Timestamp(self.bounds.max_x as i64 + self.earliest_timestamp.0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PlotBounds {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
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

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn range_x_f32(&self) -> RangeInclusive<f32> {
        (self.min_x as f32)..=(self.max_x as f32)
    }

    pub fn range_y_f32(&self) -> RangeInclusive<f32> {
        (self.min_y as f32)..=(self.max_y as f32)
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

fn pretty_round(num: f64) -> f64 {
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

#[derive(Debug, Clone)]
pub struct PlotPoint {
    #[allow(dead_code)] // we might want to use the x values again
    x: f64,
    y: f64,
    pos2: egui::Pos2,
}

impl PlotPoint {
    pub fn new(x: f64, y: f64, pos2: egui::Pos2) -> Self {
        Self { x, y, pos2 }
    }

    pub fn from_plot_pos2(plot: &Plot, pos: egui::Pos2) -> Self {
        Self::new(
            egui::remap(
                pos.x,
                range_x_from_rect(&plot.inner_rect),
                plot.bounds.range_x_f32(),
            ) as f64,
            egui::remap(
                pos.y,
                range_y_from_rect(&plot.inner_rect),
                plot.bounds.range_y_f32(),
            ) as f64,
            pos,
        )
    }

    pub fn from_plot_point(plot: &Plot, x: f64, y: f64) -> Self {
        Self::new(x, y, Self::pos2(plot, x, y))
    }

    fn pos2(plot: &Plot, x: f64, y: f64) -> egui::Pos2 {
        egui::pos2(
            egui::remap(
                x as f32,
                plot.bounds.range_x_f32(),
                range_x_from_rect(&plot.inner_rect),
            ),
            egui::remap(
                y as f32,
                plot.bounds.range_y_f32(),
                range_y_from_rect(&plot.inner_rect),
            ),
        )
    }
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
    const ZOOM_SENSITIVITY: f32 = 0.001;

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

        let viewport_rect = Rect::new(
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32 + viewport.physical_position.x as f32,
            viewport.physical_size.y as f32 + viewport.physical_position.y as f32,
        );
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

        let viewport_rect = Rect::new(
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32 + viewport.physical_position.x as f32,
            viewport.physical_size.y as f32 + viewport.physical_position.y as f32,
        );
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

            let viewport_rect = Rect::new(
                viewport.physical_position.x as f32,
                viewport.physical_position.y as f32,
                viewport.physical_size.x as f32 + viewport.physical_position.x as f32,
                viewport.physical_size.y as f32 + viewport.physical_position.y as f32,
            );
            if !viewport_rect.contains(cursor_pos) {
                continue;
            }
            graph_state.pan_offset = Vec2::ZERO;
            graph_state.zoom_factor = Vec2::ONE;
        }
    }
}

fn scroll_offset_from_events(mut scroll_events: EventReader<MouseWheel>) -> f32 {
    let pixels_per_line = 100.; // Maybe make configurable?
    scroll_events
        .read()
        .map(|ev| match ev.unit {
            MouseScrollUnit::Pixel => ev.y,
            MouseScrollUnit::Line => ev.y * pixels_per_line,
        })
        .sum::<f32>()
}
