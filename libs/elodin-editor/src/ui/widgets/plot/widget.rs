use bevy::{
    asset::{Assets, Handle},
    ecs::{
        entity::Entity,
        event::EventWriter,
        system::{Commands, Query},
    },
    math::{Rect, Vec2},
    render::camera::{OrthographicProjection, Projection, ScalingMode},
};
use bevy_egui::egui::{self, Align, Layout};
use conduit::{
    bevy::EntityMap, query::MetadataStore, well_known::EntityMetadata, ComponentId, ControlMsg,
    EntityId,
};
use egui::{vec2, Color32, Frame, Margin, Pos2, RichText, Rounding, Stroke};
use itertools::{Itertools, MinMaxResult};
use std::{
    fmt::Debug,
    ops::{Range, RangeInclusive},
};

use crate::{
    ui::widgets::plot::CollectedGraphData,
    ui::{
        colors::{self, with_opacity, ColorExt},
        utils::{self, format_num},
        widgets::{
            plot::gpu::{LineBundle, LineConfig, LineUniform},
            plot::GraphState,
            plot::Line,
            timeline::tagged_range::TaggedRange,
        },
    },
};

use super::{PlotDataComponent, PlotDataEntity};

#[derive(Debug)]
pub struct Plot {
    tick_range: Range<u64>,
    current_tick: u64,
    time_step: std::time::Duration,
    bounds: PlotBounds,
    rect: egui::Rect,
    inner_rect: egui::Rect,

    invert_x: bool,
    invert_y: bool,
    steps_x: usize,
    steps_y: usize,

    padding: egui::Margin,
    margin: egui::Margin,
    notch_length: f32,
    axis_label_margin: f32,

    text_color: egui::Color32,
    border_stroke: egui::Stroke,

    show_modal: bool,
    show_legend: bool,
}

fn range_x_from_rect(rect: &egui::Rect, invert: bool) -> RangeInclusive<f32> {
    if invert {
        rect.max.x..=rect.min.x
    } else {
        rect.min.x..=rect.max.x
    }
}

fn range_y_from_rect(rect: &egui::Rect, invert: bool) -> RangeInclusive<f32> {
    if invert {
        rect.max.y..=rect.min.y
    } else {
        rect.min.y..=rect.max.y
    }
}

pub fn get_inner_rect(rect: egui::Rect) -> egui::Rect {
    use crate::ui::utils::MarginSides;
    let adding: egui::Margin = egui::Margin::same(0.0).left(20.0).bottom(20.0);
    let margin: egui::Margin = egui::Margin::same(60.0).left(85.0).top(40.0);
    egui::Rect {
        min: egui::pos2(
            rect.min.x + (margin.left + adding.left),
            rect.min.y + (margin.top + adding.top),
        ),
        max: egui::pos2(
            rect.max.x - (margin.right + adding.right),
            rect.max.y - (margin.bottom + adding.bottom),
        ),
    }
}

impl Default for Plot {
    fn default() -> Self {
        Self::new()
    }
}

impl Plot {
    pub fn new() -> Self {
        Self {
            tick_range: Range::default(),
            current_tick: 0,
            time_step: std::time::Duration::from_secs_f64(1.0 / 60.0),
            bounds: PlotBounds::default(),
            rect: egui::Rect::ZERO,
            inner_rect: egui::Rect::ZERO,

            invert_x: false,
            invert_y: true,
            steps_x: 6,
            steps_y: 6,

            padding: egui::Margin::same(20.0),
            margin: egui::Margin::same(60.0),
            notch_length: 10.0,
            axis_label_margin: 10.0,

            text_color: colors::PRIMARY_CREAME,
            border_stroke: egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),

            show_modal: true,
            show_legend: false,
        }
    }

    /// Calculate bounds and point positions based on the current UI allocation
    /// Should be run last, right before render
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_lines(
        mut self,
        ui: &egui::Ui,
        collected_graph_data: &mut CollectedGraphData,
        graph_state: &mut GraphState,
        tagged_range: Option<&TaggedRange>,
        lines: &mut Assets<Line>,
        commands: &mut Commands,
        graph_id: Entity,
        entity_map: &EntityMap,
        entity_metadata: &Query<&EntityMetadata>,
        metadata_store: &MetadataStore,
        control_msg: &mut EventWriter<ControlMsg>,
    ) -> Self {
        self.rect = ui.max_rect();
        self.inner_rect = get_inner_rect(self.rect);

        // calc max_length

        let width_in_px = ui.available_width() * ui.ctx().pixels_per_point();
        let max_length = width_in_px.ceil() as usize * 2;

        if let Some(tagged_range) = tagged_range {
            let (a, b) = tagged_range.values;
            self.tick_range = if a > b { b..a } else { a..b };
        } else {
            self.tick_range = collected_graph_data.tick_range.clone();
        }

        // calc bounds

        let mut minmax_lines = vec![];

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
                                    minmax_lines.push(vec![line.min, line.max]);
                                    line.data.max_count = max_length;
                                }
                            }
                        }
                    }
                }
            }
        }

        self.bounds = PlotBounds::from_lines(&self.tick_range, &minmax_lines);
        let y_range = self.bounds.max_y - self.bounds.min_y;
        commands
            .entity(graph_id)
            .insert(Projection::Orthographic(OrthographicProjection {
                near: 0.0,
                far: 1000.0,
                viewport_origin: Vec2::new(0.0, -(self.bounds.min_y / y_range) as f32),
                scaling_mode: ScalingMode::Fixed {
                    width: (self.bounds.max_x - self.bounds.min_x) as f32,
                    height: (self.bounds.max_y - self.bounds.min_y) as f32,
                },
                scale: 1.0,
                area: Rect::new(
                    self.bounds.min_x as f32,
                    self.bounds.min_y as f32,
                    self.bounds.max_x as f32,
                    self.bounds.max_y as f32,
                ),
            }));

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
                let component_label = component_metadata.component_name();
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
                    let entity =
                        graph_state
                            .enabled_lines
                            .get(&(*entity_id, *component_id, value_index));

                    let Some(line) = component.lines.get(&value_index) else {
                        continue;
                    };
                    if *enabled {
                        if let Some(values) = lines.get_mut(line) {
                            values.data.current_range =
                                (self.tick_range.start as usize)..(self.tick_range.end as usize);
                            values.data.max_count = max_length;
                            for chunk in values.data.range() {
                                if !chunk.unfetched.is_empty() {
                                    let start =
                                        chunk.unfetched.min().expect("unexpected empty chunk")
                                            as u64;
                                    let mut end =
                                        chunk.unfetched.max().expect("unexpected empty chunk")
                                            as u64;
                                    if start == end {
                                        end += 1;
                                    }
                                    let start = start.max(self.tick_range.start);
                                    let end = end.min(self.tick_range.end) + 1;
                                    let time_range = start..end;

                                    if end.saturating_sub(start) > 0 {
                                        chunk.unfetched.remove_range(start as u32..end as u32);
                                        control_msg.send(ControlMsg::Query {
                                            time_range,
                                            query: conduit::Query {
                                                component_id: *component_id,
                                                with_component_ids: vec![],
                                                entity_ids: vec![*entity_id],
                                            },
                                        });
                                    }
                                }
                            }
                            values.data.recalculate_avg_data();
                        }
                    }

                    match (entity, enabled) {
                        (None, true) => {
                            let entity = commands
                                .spawn(LineBundle {
                                    line: line.clone(),
                                    uniform: LineUniform::new(
                                        graph_state.line_width,
                                        color.into_bevy(),
                                    ),
                                    config: LineConfig {
                                        render_layers: graph_state.render_layers,
                                    },
                                })
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
                        (Some((entity, _)), true) => {
                            commands.entity(*entity).insert(LineUniform::new(
                                graph_state.line_width,
                                color.into_bevy(),
                            ));
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

        self
    }

    pub fn time_step(mut self, step: std::time::Duration) -> Self {
        self.time_step = step;
        self
    }

    pub fn current_tick(mut self, tick: u64) -> Self {
        self.current_tick = tick;
        self
    }

    pub fn invert(mut self, x: bool, y: bool) -> Self {
        self.invert_x = x;
        self.invert_y = y;
        self
    }

    pub fn padding(mut self, padding: egui::Margin) -> Self {
        self.padding = padding;
        self
    }

    pub fn margin(mut self, margin: egui::Margin) -> Self {
        self.margin = margin;
        self
    }

    pub fn steps(mut self, x: usize, y: usize) -> Self {
        self.steps_x = x;
        self.steps_y = y;
        self
    }

    pub fn text_color(mut self, color: egui::Color32) -> Self {
        self.text_color = color;
        self
    }

    pub fn border_stroke(mut self, stroke: egui::Stroke) -> Self {
        self.border_stroke = stroke;
        self
    }

    fn draw_x_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let step_size = self.bounds.width() / self.steps_x as f64;
        let min_step_size = 1.0 / self.time_step.as_secs_f64();
        let step_size = (step_size / min_step_size).ceil() * min_step_size;
        let steps_x = (0..=self.steps_x)
            .map(|i| self.bounds.min_x + (i as f64) * step_size)
            .filter(|x| *x <= self.bounds.max_x)
            .collect::<Vec<f64>>();

        for x_step in steps_x {
            let x_position = PlotPoint::from_plot_point(self, x_step, self.bounds.min_y).pos2;

            ui.painter().line_segment(
                [
                    egui::pos2(x_position.x, x_position.y + self.padding.bottom),
                    egui::pos2(
                        x_position.x,
                        x_position.y + (self.padding.bottom + self.notch_length),
                    ),
                ],
                self.border_stroke,
            );

            let time = (self.time_step.as_secs_f64() * x_step).round() as usize;
            ui.painter().text(
                egui::pos2(
                    x_position.x,
                    x_position.y
                        + (self.padding.bottom + self.notch_length + self.axis_label_margin),
                ),
                egui::Align2::CENTER_TOP,
                utils::time_label(time, false),
                font_id.clone(),
                self.text_color,
            );
        }
    }

    fn draw_y_axis(&self, ui: &mut egui::Ui, font_id: &egui::FontId) {
        let step_size = self.bounds.height() / self.steps_y as f64;
        let steps_y = (0..=self.steps_y)
            .map(|i| self.bounds.min_y + (i as f64) * step_size)
            .collect::<Vec<f64>>();

        for y_step in steps_y {
            let y_position = PlotPoint::from_plot_point(self, self.bounds.min_x, y_step).pos2;

            ui.painter().line_segment(
                [
                    egui::pos2(y_position.x - self.padding.left, y_position.y),
                    egui::pos2(
                        y_position.x - (self.padding.left + self.notch_length),
                        y_position.y,
                    ),
                ],
                self.border_stroke,
            );

            ui.painter().text(
                egui::pos2(
                    y_position.x - (self.padding.left + self.notch_length + self.axis_label_margin),
                    y_position.y,
                ),
                egui::Align2::RIGHT_CENTER,
                format_num(y_step),
                font_id.clone(),
                self.text_color,
            );
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
            egui::Rounding::same(2.0),
            colors::MINT_DEFAULT,
            egui::Stroke::NONE,
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
            border_rect.min.y..=border_rect.max.y,
            egui::Stroke::new(1.0, colors::PRIMARY_ONYX_9),
        );

        // NOTE: HLine is not attached to points
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
        line_handles: &Query<&Handle<Line>>,
        graph_state: &GraphState,
        collected_graph_data: &CollectedGraphData,
        pointer_pos: egui::Pos2,
        tick: usize,
    ) {
        let modal_width = 200.0;
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
                    .inner_margin(Margin::same(16.0))
                    .stroke(Stroke::new(1.0, colors::BORDER_GREY))
                    .rounding(Rounding::same(4.0))
                    .fill(colors::PRIMARY_SMOKE)
                    .shadow(egui::epaint::Shadow {
                        offset: vec2(0., 5.0),
                        blur: 8.0,
                        spread: -2.0,
                        color: Color32::from_black_alpha(191),
                    }),
            )
            .show(ui.ctx(), |ui| {
                let time = self.time_step * tick as u32;
                let time_text = utils::time_label_ms(time.as_secs_f64());

                ui.label(time_text);
                let mut current_entity_id: Option<EntityId> = None;
                let mut current_component_id: Option<ComponentId> = None;
                for ((entity_id, component_id, line_index), (entity, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line) = lines.get(line_handle) else {
                        continue;
                    };
                    let index = tick / line.data.chunk_size;
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
                                    egui::Rounding::same(2.0),
                                    *color,
                                    egui::Stroke::NONE,
                                );
                                ui.add_space(6.);
                                ui.label(RichText::new(line_data.label.clone()).size(11.0));
                            })
                        });
                        let value = line
                            .data
                            .averaged_data
                            .get(index)
                            .map(|x| format!("{:.2}", x))
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

    fn draw_legend(&self, ui: &mut egui::Ui) {
        let legend_size = egui::vec2(200.0, 50.0);
        let legend_margin = 10.0;

        let modal_rect = egui::Rect::from_min_max(
            egui::pos2(
                self.inner_rect.max.x - (legend_size.x + legend_margin),
                self.inner_rect.max.y - (legend_size.y + legend_margin),
            ),
            egui::pos2(
                self.inner_rect.max.x - legend_margin,
                self.inner_rect.max.y - legend_margin,
            ),
        );

        ui.painter().rect(
            modal_rect,
            egui::Rounding::same(2.0),
            colors::with_opacity(colors::PRIMARY_SMOKE, 0.9),
            egui::Stroke::new(0.4, colors::PRIMARY_CREAME),
        );
    }

    fn get_border_rect(&self, rect: egui::Rect) -> egui::Rect {
        egui::Rect {
            min: egui::pos2(rect.min.x + self.margin.left, rect.min.y + self.margin.top),
            max: egui::pos2(
                rect.max.x - self.margin.right,
                rect.max.y - self.margin.bottom,
            ),
        }
    }

    pub fn render(
        &self,
        ui: &mut egui::Ui,
        lines: &Assets<Line>,
        line_handles: &Query<&Handle<Line>>,
        collected_graph_data: &CollectedGraphData,
        graph_state: &GraphState,
        scrub_icon: &egui::TextureId,
    ) {
        let _response = ui.allocate_rect(self.rect, egui::Sense::click_and_drag());
        let pointer_pos = ui.input(|i| i.pointer.latest_pos());

        let border_rect = self.get_border_rect(self.rect);

        // Style

        let style = ui.style();
        let font_id = egui::TextStyle::Button.resolve(style);

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
            if self.inner_rect.contains(pointer_pos) {
                let pointer_plot_point = PlotPoint::from_plot_pos2(self, pointer_pos);

                self.draw_y_axis_flag(ui, pointer_plot_point, border_rect, font_id);

                let x_offset = pointer_pos.x - self.inner_rect.min.x;
                let x_range = self.bounds.max_x - self.bounds.min_x;
                let x_pos = x_offset * (x_range as f32 / self.inner_rect.width());
                let x_tick = x_pos as usize;
                self.draw_cursor(ui, pointer_pos, x_offset, self.inner_rect);

                // Draw highlight circles on lines

                for ((_entity_id, _component_id, _line_index), (entity, color)) in
                    graph_state.enabled_lines.iter()
                {
                    let Ok(line_handle) = line_handles.get(*entity) else {
                        continue;
                    };
                    let Some(line) = lines.get(line_handle) else {
                        continue;
                    };
                    let x_index = x_tick / line.data.chunk_size;
                    let Some(y) = line.data.averaged_data.get(x_index) else {
                        continue;
                    };
                    let y_offset = *y as f64 - self.bounds.min_y;
                    let y_range = self.bounds.max_y - self.bounds.min_y;
                    let y_pos = y_offset as f32 * (self.inner_rect.height() / y_range as f32);
                    let y_pos = self.inner_rect.max.y - y_pos;
                    let pos = Pos2::new(pointer_pos.x, y_pos);
                    ui.painter().circle(
                        pos,
                        6.0,
                        colors::PRIMARY_SMOKE,
                        egui::Stroke::new(2.0, *color),
                    );
                }
            }
        }

        // Draw a line for the current_tick
        if self.tick_range.contains(&self.current_tick) {
            let line_width = 1.0;
            let aspect_ratio = 12.0 / 30.0;

            let scrub_height = 12.0 * line_width;
            let scrub_width = scrub_height * aspect_ratio;

            let tick_pos =
                PlotPoint::from_plot_point(self, self.current_tick as f64, self.bounds.min_y).pos2;
            ui.painter().vline(
                tick_pos.x,
                self.rect.min.y..=border_rect.max.y,
                egui::Stroke::new(line_width, colors::WHITE),
            );

            let scrub_center = egui::pos2(tick_pos.x, self.rect.min.y + (scrub_height * 0.5));
            let scrub_rect =
                egui::Rect::from_center_size(scrub_center, egui::vec2(scrub_width, scrub_height));

            ui.painter().image(
                *scrub_icon,
                scrub_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                colors::WHITE,
            );
        }

        if let Some(pointer_pos) = pointer_pos {
            if self.inner_rect.contains(pointer_pos) && self.show_modal && ui.ui_contains_pointer()
            {
                let x_offset = pointer_pos.x - self.inner_rect.min.x;
                let x_range = self.bounds.max_x - self.bounds.min_x;
                let x_pos = x_offset * (x_range as f32 / self.inner_rect.width());
                let x_tick = x_pos as usize;

                self.draw_modal(
                    ui,
                    lines,
                    line_handles,
                    graph_state,
                    collected_graph_data,
                    pointer_pos,
                    x_tick,
                );
            }
        }

        // Draw legend

        if self.show_legend {
            self.draw_legend(ui);
        }
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

    pub fn from_lines(baseline: &Range<u64>, lines: &[Vec<f64>]) -> Self {
        let (min_x, max_x) = (baseline.start as f64, baseline.end as f64);

        let minmax_y = lines.iter().flatten().minmax();

        let (min_y, max_y) = match minmax_y {
            MinMaxResult::MinMax(min_y, max_y) => (*min_y, *max_y),
            MinMaxResult::OneElement(y) => (*y, *y),
            _ => (0.0, 0.0),
        };

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
                range_x_from_rect(&plot.inner_rect, plot.invert_x),
                plot.bounds.range_x_f32(),
            ) as f64,
            egui::remap(
                pos.y,
                range_y_from_rect(&plot.inner_rect, plot.invert_y),
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
                range_x_from_rect(&plot.inner_rect, plot.invert_x),
            ),
            egui::remap(
                y as f32,
                plot.bounds.range_y_f32(),
                range_y_from_rect(&plot.inner_rect, plot.invert_y),
            ),
        )
    }
}
