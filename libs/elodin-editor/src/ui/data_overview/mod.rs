//! Data Overview Panel
//!
//! Provides a high-level timeline view showing all database components and their
//! data density across time. Useful for identifying gaps or disparities in data.

use std::collections::HashMap;
use std::time::Instant;

use arrow::array::{Array, Float64Array, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_egui::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::{
    CommandsExt, ComponentPathRegistry, ComponentSchemaRegistry, SimTimeStepFetch,
    SimTimeStepSource,
};
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery, SimulationTimeStep, SparklineQuery};

use crate::{
    EqlContext, SelectedTimeRange,
    ui::{SelectedObject, tiles::WindowState, widgets::WidgetSystem},
};

use super::{
    PaneName,
    colors::{ColorExt, get_scheme},
};

// Re-export for use in component collection
use crate::ui::widgets::SystemStateExt;
use eql;

/// Maximum number of points per series in each sparkline
const SPARKLINE_MAX_POINTS: usize = 10000;

/// Maximum concurrent queries to avoid RequestId overflow and queue overflow.
/// RequestId is u8 (max 255), and the BBQ queue can fill if too many large responses arrive.
/// Reduced to 120 to ensure responses are processed before the queue overflows.
const MAX_CONCURRENT_QUERIES: usize = 120;

/// A single series of data points (one field/element of a component)
#[derive(Clone, Debug, Default)]
pub struct SparklineSeries {
    /// Time-value pairs for this series
    pub points: Vec<(i64, f64)>,
}

/// Sparkline data for a single component - may have multiple series for vector types
#[derive(Clone, Debug, Default)]
pub struct SparklineData {
    /// Multiple series (one per field/element), each with time-value pairs
    pub series: Vec<SparklineSeries>,
    /// Min Y value across all series for scaling
    pub y_min: f64,
    /// Max Y value across all series for scaling  
    pub y_max: f64,
    /// Number of raw data points (before downsampling) - used to detect new data
    pub raw_point_count: usize,
}

/// Resource to cache component timestamp ranges and sparkline data from the database
#[derive(Resource, Default)]
pub struct ComponentTimeRanges {
    /// Map from component table name to (min_timestamp, max_timestamp)
    pub ranges: HashMap<String, (Timestamp, Timestamp)>,
    /// Map from component table name to row count
    pub row_counts: HashMap<String, usize>,
    /// Map from component table name to sparkline data
    pub sparklines: HashMap<String, SparklineData>,
    /// Number of queries still pending in current batch
    pub pending_queries: usize,
    /// Total number of queries across all batches
    pub total_queries: usize,
    /// Number of queries completed so far
    pub completed_queries: usize,
    /// State of the query
    pub state: TimeRangeQueryState,
    /// List of table names to query (for batched processing)
    pub tables_to_query: Vec<String>,
    /// Current batch index for processing
    pub current_batch: usize,
    pub row_settings: HashMap<ComponentId, DataOverviewRowSettings>,
    /// Bumped by [`ComponentTimeRanges::reset`] to disown replies in flight.
    generation: u64,
}

impl ComponentTimeRanges {
    /// Drop everything measured for the previous recording. Returning `state`
    /// to `NotStarted` is what re-arms [`trigger_time_range_queries`]; leaving
    /// stale ranges behind would let the timeline step keep the old
    /// recording's rate.
    pub fn reset(&mut self) {
        let disowned = self.generation.wrapping_add(1);
        *self = Self::default();
        self.generation = disowned;
    }

    /// Stamp to hand a query, so its reply can be matched back to the
    /// recording that asked for it.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether a reply still belongs to the current recording.
    pub fn accepts(&self, generation: u64) -> bool {
        self.generation == generation
    }
}

/// Account for one finished time-range query.
///
/// Reaching zero pending moves the batch on to sparklines. Note that a reply
/// arriving after a reset would satisfy `completed >= total` immediately,
/// since a reset leaves `total_queries` at zero — which is why callers must
/// check [`ComponentTimeRanges::accepts`] first. Moving off `NotStarted` is
/// unrecoverable: [`trigger_time_range_queries`] only ever starts from there.
fn finish_time_range_query(time_ranges: &mut ComponentTimeRanges) {
    time_ranges.pending_queries = time_ranges.pending_queries.saturating_sub(1);
    time_ranges.completed_queries += 1;

    if time_ranges.pending_queries == 0
        && time_ranges.completed_queries >= time_ranges.total_queries
    {
        time_ranges.state = TimeRangeQueryState::QueryingSparklines(Instant::now());
        time_ranges.current_batch = 0;
        time_ranges.completed_queries = 0;
    }
}

#[derive(Clone, Debug)]
pub struct DataOverviewRowSettings {
    pub enabled: bool,
    pub color: Option<Color32>,
    pub custom_name: Option<String>,
}

impl Default for DataOverviewRowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            color: None,
            custom_name: None,
        }
    }
}

#[derive(Default, Clone)]
pub enum TimeRangeQueryState {
    #[default]
    NotStarted,
    /// Phase 1: Querying time ranges and row counts
    QueryingTimeRanges(Instant),
    /// Phase 2: Querying sparkline data (after row counts are known)
    QueryingSparklines(Instant),
    Ready,
    Error(String),
}

/// Pane data for the DataOverview panel
#[derive(Clone)]
pub struct DataOverviewPane {
    pub name: PaneName,
    /// Cached screen rect for rendering
    pub rect: Option<egui::Rect>,
    /// Vertical scroll offset for component list
    pub scroll_offset: f32,
    /// Horizontal zoom factor (1.0 = fit to data, >1.0 = zoomed in)
    pub zoom_factor: f32,
    /// Horizontal pan offset in microseconds from the data start
    pub pan_offset_us: i64,
    /// Cached data time range (min, max) from component queries
    pub cached_data_range: Option<(Timestamp, Timestamp)>,
    /// Last known drag position for pan calculation
    pub last_drag_pos: Option<egui::Pos2>,
}

impl Default for DataOverviewPane {
    fn default() -> Self {
        Self {
            name: "Data Overview".to_string(),
            rect: None,
            scroll_offset: 0.0,
            zoom_factor: 1.0,
            pan_offset_us: 0,
            cached_data_range: None,
            last_drag_pos: None,
        }
    }
}

/// Summary of a component's data presence
#[derive(Clone, Debug)]
pub struct ComponentTimestampSummary {
    pub component_id: ComponentId,
    pub label: String,
    pub table_name: String,
    pub color: Color32,
    pub enabled: bool,
    pub timestamp_range: Option<(Timestamp, Timestamp)>,
    /// User-defined custom name for display (if set via inspector)
    pub custom_name: Option<String>,
}

/// Generate a distinct color for a given row index using golden ratio hue distribution
pub fn row_color(index: usize) -> Color32 {
    const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_034;
    let hue = ((index as f32) * GOLDEN_RATIO_CONJUGATE * 360.0) % 360.0;
    let (r, g, b) = hsl_to_rgb(hue, 0.7, 0.55);
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Convert HSL to RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_normalized = h / 360.0;

    let r = hue_to_rgb(p, q, h_normalized + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_normalized);
    let b = hue_to_rgb(p, q, h_normalized - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Convert component name to SQL table name using the same conversion as the database.
pub fn component_to_table_name(full_component_name: &str) -> String {
    // Full component name is like "GpsPosMessage1.VACC"
    // Table name is like "gps_pos_message_1_vacc"
    eql::sql_table_name(full_component_name)
}

/// Widget for rendering the Data Overview panel
#[derive(SystemParam)]
pub struct DataOverviewWidget<'w, 's> {
    eql_context: Res<'w, EqlContext>,
    selected_range: Res<'w, SelectedTimeRange>,
    time_ranges: ResMut<'w, ComponentTimeRanges>,
    commands: Commands<'w, 's>,
    window_states: Query<'w, 's, &'static mut WindowState>,
}

impl WidgetSystem for DataOverviewWidget<'_, '_> {
    type Args = (DataOverviewPane, Entity);
    type Output = DataOverviewPane;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        (mut pane, target_window): Self::Args,
    ) -> Self::Output {
        let mut params = state.params_mut(world);
        let Ok(mut window_state) = params.window_states.get_mut(target_window) else {
            return pane;
        };

        let scheme = get_scheme();
        let available_rect = ui.available_rect_before_wrap();
        pane.rect = Some(available_rect);

        // Constants for layout
        const LABEL_WIDTH: f32 = 280.0; // Wider to show full component paths
        const ROW_HEIGHT: f32 = 16.0;

        // Collect component info from EqlContext
        fn collect_components(
            parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
            result: &mut Vec<(ComponentId, String, String)>,
        ) {
            for part in parts.values() {
                if let Some(component) = &part.component {
                    // Use the full component name (e.g., "MfNavElodinEnumMessage.ypr_enu_2_body_deg")
                    // for both display label and table name generation
                    let full_name = component.name.clone();
                    let table_name = component_to_table_name(&full_name);
                    result.push((part.id, full_name, table_name));
                }
                collect_components(&part.children, result);
            }
        }

        let mut component_list: Vec<(ComponentId, String, String)> = Vec::new();
        collect_components(&params.eql_context.0.component_parts, &mut component_list);
        let mut enabled_tables: Vec<String> = Vec::new();
        for (component_id, _, table_name) in &component_list {
            let settings = params
                .time_ranges
                .row_settings
                .entry(*component_id)
                .or_default();
            if settings.enabled {
                enabled_tables.push(table_name.clone());
            }
        }

        // Phase 1 (time ranges) is sent by `dispatch_time_range_queries`, which
        // runs whether or not this panel is open.

        // Phase 2: Query sparkline data in batches
        if matches!(
            params.time_ranges.state,
            TimeRangeQueryState::QueryingSparklines(_)
        ) && params.time_ranges.pending_queries == 0
        {
            let batch_start = params.time_ranges.current_batch * MAX_CONCURRENT_QUERIES;
            let batch_end = (batch_start + MAX_CONCURRENT_QUERIES)
                .min(params.time_ranges.tables_to_query.len());

            if batch_start >= params.time_ranges.tables_to_query.len() {
                params.time_ranges.state = TimeRangeQueryState::Ready;
            } else {
                // Start next batch
                let batch_size = batch_end - batch_start;
                params.time_ranges.pending_queries = batch_size;
                params.time_ranges.current_batch += 1;

                let generation = params.time_ranges.generation();
                for table_name in params.time_ranges.tables_to_query[batch_start..batch_end].iter()
                {
                    let table_name_clone = table_name.clone();

                    // Use SparklineQuery with server-side LTTB downsampling.
                    // The server extracts just time + first scalar value and applies LTTB,
                    // returning at most SPARKLINE_MAX_POINTS data points.
                    // This is resilient to any data size - even millions of rows.
                    let query = SparklineQuery {
                        table_name: table_name.clone(),
                        max_points: SPARKLINE_MAX_POINTS as u32,
                    };

                    params.commands.send_req_reply(
                        query,
                        move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                              mut time_ranges: ResMut<ComponentTimeRanges>| {
                            if !time_ranges.accepts(generation) {
                                return true;
                            }
                            // Only process data batches; completion markers (batch: None) are ignored
                            // since the handler is removed immediately after processing data.
                            // This prevents request ID collisions when IDs wrap around.
                            match res {
                                Ok(ipc) => {
                                    if let Some(batch_data) = ipc.batch {
                                        // Data batch - process it (already downsampled by server)
                                        let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                        let mut buffer =
                                            arrow::buffer::Buffer::from(batch_data.into_owned());
                                        if let Some(batch) =
                                            decoder.decode(&mut buffer).ok().and_then(|b| b)
                                        {
                                            process_sparkline_result(
                                                &table_name_clone,
                                                &batch,
                                                &mut time_ranges.sparklines,
                                            );
                                        }
                                        // Decrement pending_queries now that we have the data
                                        time_ranges.pending_queries =
                                            time_ranges.pending_queries.saturating_sub(1);
                                        time_ranges.completed_queries += 1;
                                    }
                                    // If batch is None (completion marker), just ignore it
                                    // The handler will be removed either way
                                }
                                Err(e) => {
                                    // Error response - query is done
                                    eprintln!(
                                        "Sparkline {}: query error: {}",
                                        table_name_clone, e.description
                                    );
                                    time_ranges.pending_queries =
                                        time_ranges.pending_queries.saturating_sub(1);
                                    time_ranges.completed_queries += 1;
                                }
                            }
                            // Always return true to remove handler immediately, freeing the request ID
                            true
                        },
                    );
                }
            }
        }

        // Build summaries with cached time ranges
        let mut summaries: Vec<ComponentTimestampSummary> = Vec::new();
        for (component_id, label, table_name) in component_list.iter() {
            let timestamp_range = params.time_ranges.ranges.get(table_name).copied();
            let custom_name = params
                .time_ranges
                .row_settings
                .get(component_id)
                .and_then(|s| s.custom_name.clone());

            summaries.push(ComponentTimestampSummary {
                component_id: *component_id,
                label: label.clone(),
                table_name: table_name.clone(),
                color: Color32::WHITE, // Will be assigned after sorting
                enabled: true,
                timestamp_range,
                custom_name,
            });
        }

        // Sort: components with data first (alphabetically), then empty components (alphabetically)
        summaries.sort_by(|a, b| {
            match (a.timestamp_range.is_some(), b.timestamp_range.is_some()) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.label.cmp(&b.label),
            }
        });

        // Assign colors after sorting so adjacent rows have distinct colors
        for (index, summary) in summaries.iter_mut().enumerate() {
            let settings = params
                .time_ranges
                .row_settings
                .entry(summary.component_id)
                .or_default();
            if settings.color.is_none() {
                settings.color = Some(row_color(index));
            }
            summary.color = settings.color.unwrap_or_else(|| row_color(index));
            summary.enabled = settings.enabled;
            if !summary.enabled {
                summary.timestamp_range = None;
            }
        }

        // Calculate time range from actual component data ranges
        // This ensures the timeline scales to fit the actual data
        let data_time_range: Option<(Timestamp, Timestamp)> = summaries
            .iter()
            .filter_map(|s| s.timestamp_range)
            .filter(|(min, _)| min.0 > 0)
            .fold(None, |acc: Option<(Timestamp, Timestamp)>, (min, max)| {
                Some(match acc {
                    None => (min, max),
                    Some((a_min, a_max)) => {
                        (Timestamp(a_min.0.min(min.0)), Timestamp(a_max.0.max(max.0)))
                    }
                })
            });

        // Update cached data range when queries complete
        if matches!(params.time_ranges.state, TimeRangeQueryState::Ready)
            && data_time_range.is_some()
        {
            pane.cached_data_range = data_time_range;
        }

        // Use cached data range, falling back to selected_range if no data yet
        let (data_min, data_max) = pane
            .cached_data_range
            .or(data_time_range)
            .unwrap_or_else(|| (params.selected_range.0.start, params.selected_range.0.end));

        // Add a small margin (5%) to the data range for better visualization
        let span_us = data_max.0.saturating_sub(data_min.0);
        let base_span = span_us.max(1_000_000) as f64; // minimum 1 second
        let margin = (base_span * 0.05) as i64;
        let display_data_min = Timestamp(data_min.0.saturating_sub(margin));

        // Apply zoom factor (1.0 = fit all data, >1.0 = zoomed in)
        let zoomed_span = base_span / pane.zoom_factor.max(0.01) as f64;

        // Apply pan offset (clamped to prevent scrolling beyond data)
        let max_pan = (base_span - zoomed_span).max(0.0) as i64;
        pane.pan_offset_us = pane.pan_offset_us.clamp(0, max_pan);

        let display_start = Timestamp(display_data_min.0.saturating_add(pane.pan_offset_us));
        let display_end = Timestamp(display_start.0.saturating_add(zoomed_span as i64));

        let timeline_width = (available_rect.width() - LABEL_WIDTH).max(100.0);
        let time_span = (display_end.0 - display_start.0).max(1) as f64;
        let pixels_per_us = timeline_width as f64 / time_span;
        let timeline_start_x = available_rect.min.x + LABEL_WIDTH;
        let timeline_end_x = timeline_start_x + timeline_width;
        let response = ui.interact(
            available_rect,
            ui.id().with("data_overview"),
            Sense::click_and_drag(),
        );
        let click_pos = if response.clicked() {
            window_state.ui_state.selected_object = SelectedObject::DataOverview;
            response.interact_pointer_pos()
        } else {
            None
        };

        // Header with component count and query status
        let header_height = ROW_HEIGHT;
        let header_rect = Rect::from_min_size(
            available_rect.min,
            Vec2::new(available_rect.width(), header_height),
        );
        ui.painter()
            .rect_filled(header_rect, 0.0, scheme.bg_secondary);

        // Header text with status
        let status_text = match &params.time_ranges.state {
            TimeRangeQueryState::NotStarted => {
                format!("Components: {} (loading...)", summaries.len())
            }
            TimeRangeQueryState::QueryingTimeRanges(_) => {
                format!(
                    "Components: {} (scanning {}/{}...)",
                    summaries.len(),
                    params.time_ranges.completed_queries,
                    params.time_ranges.total_queries
                )
            }
            TimeRangeQueryState::QueryingSparklines(_) => {
                format!(
                    "Components: {} (loading data {}/{}...)",
                    summaries.len(),
                    params.time_ranges.completed_queries,
                    params.time_ranges.total_queries
                )
            }
            TimeRangeQueryState::Ready => {
                let with_data = summaries
                    .iter()
                    .filter(|s| s.timestamp_range.is_some())
                    .count();
                format!("Components: {} ({} with data)", summaries.len(), with_data)
            }
            TimeRangeQueryState::Error(e) => {
                format!("Components: {} (error: {})", summaries.len(), e)
            }
        };

        ui.painter().text(
            Pos2::new(available_rect.min.x + 8.0, available_rect.min.y + 2.0),
            egui::Align2::LEFT_TOP,
            status_text,
            egui::FontId::proportional(11.0),
            scheme.text_primary,
        );

        // Draw component rows
        let content_start_y = available_rect.min.y + header_height + 2.0;
        let content_height = available_rect.height() - header_height - 2.0;

        // Create a scrollable area for the component list
        let scroll_area = egui::ScrollArea::vertical()
            .max_height(content_height)
            .auto_shrink([false, false]);

        scroll_area.show(ui, |ui| {
            for (row_index, summary) in summaries.iter().enumerate() {
                let row_y = row_index as f32 * ROW_HEIGHT;
                let row_rect = Rect::from_min_size(
                    Pos2::new(
                        available_rect.min.x,
                        content_start_y + row_y - pane.scroll_offset,
                    ),
                    Vec2::new(available_rect.width(), ROW_HEIGHT),
                );

                // Skip if not visible
                if row_rect.max.y < available_rect.min.y || row_rect.min.y > available_rect.max.y {
                    continue;
                }

                let is_selected = matches!(
                    window_state.ui_state.selected_object,
                    SelectedObject::DataOverviewComponent { component_id }
                        if component_id == summary.component_id
                );
                if let Some(pos) = click_pos
                    && row_rect.contains(pos)
                {
                    window_state.ui_state.selected_object = SelectedObject::DataOverviewComponent {
                        component_id: summary.component_id,
                    };
                }

                // Alternate row background
                if row_index % 2 == 0 {
                    ui.painter().rect_filled(row_rect, 0.0, scheme.bg_primary);
                }
                if is_selected {
                    ui.painter()
                        .rect_filled(row_rect, 0.0, scheme.highlight.opacity(0.08));
                }

                // Draw label with component color
                let label_rect =
                    Rect::from_min_size(row_rect.min, Vec2::new(LABEL_WIDTH, ROW_HEIGHT));

                // Color indicator
                let indicator_color = if summary.enabled {
                    summary.color
                } else {
                    summary.color.opacity(0.35)
                };
                ui.painter().circle_filled(
                    Pos2::new(label_rect.min.x + 8.0, label_rect.center().y),
                    4.0,
                    indicator_color,
                );

                // Component name (use custom_name if set, otherwise original label)
                let mut label = summary
                    .custom_name
                    .clone()
                    .unwrap_or_else(|| summary.label.clone());
                if label.len() > 38 {
                    label.truncate(35);
                    label.push_str("...");
                }

                let label_color = if summary.enabled {
                    summary.color
                } else {
                    summary.color.opacity(0.35)
                };
                ui.painter().text(
                    Pos2::new(label_rect.min.x + 16.0, label_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(10.0),
                    label_color,
                );

                // Draw sparkline if we have data with points, otherwise fall back to bar
                let sparkline = params.time_ranges.sparklines.get(&summary.table_name);

                // Check if we have valid sparkline data with actual points to draw
                let has_sparkline_points = sparkline
                    .map(|s| s.series.iter().any(|series| !series.points.is_empty()))
                    .unwrap_or(false);

                if let Some(sparkline_data) = sparkline
                    && has_sparkline_points
                {
                    // Draw sparklines for each series (field/element)
                    let row_top = row_rect.min.y + 2.0;
                    let row_bottom = row_rect.max.y - 2.0;
                    let row_height = row_bottom - row_top;
                    let y_range = sparkline_data.y_max - sparkline_data.y_min;

                    let num_series = sparkline_data.series.len();

                    for (series_idx, series) in sparkline_data.series.iter().enumerate() {
                        if series.points.is_empty() {
                            continue;
                        }

                        // Generate a color variation for each series
                        let series_color = if num_series == 1 {
                            summary.color
                        } else {
                            // Vary the hue slightly for each series
                            let base_color = summary.color;
                            let (r, g, b, a) = (
                                base_color.r(),
                                base_color.g(),
                                base_color.b(),
                                base_color.a(),
                            );
                            // Shift brightness/saturation for each series
                            let factor = 0.7 + 0.3 * (series_idx as f32 / num_series.max(1) as f32);
                            Color32::from_rgba_unmultiplied(
                                (r as f32 * factor).min(255.0) as u8,
                                (g as f32 * factor).min(255.0) as u8,
                                (b as f32 * factor).min(255.0) as u8,
                                a,
                            )
                        };

                        // Build points for the polyline
                        let mut line_points: Vec<Pos2> = Vec::new();

                        for &(time, value) in &series.points {
                            // Calculate X position
                            let x_offset = (time - display_start.0) as f64;
                            let x = timeline_start_x + (x_offset * pixels_per_us) as f32;

                            // Skip points outside visible area
                            if x < timeline_start_x || x > timeline_end_x {
                                continue;
                            }

                            // Calculate Y position (inverted: higher values at top)
                            let y_normalized = if y_range > 0.0 {
                                (value - sparkline_data.y_min) / y_range
                            } else {
                                0.5
                            };
                            let y = row_bottom - (y_normalized as f32 * row_height);

                            line_points.push(Pos2::new(x, y));
                        }

                        // Draw the sparkline as a polyline
                        if line_points.len() >= 2 {
                            ui.painter().add(egui::Shape::line(
                                line_points,
                                Stroke::new(1.0_f32, series_color),
                            ));
                        } else if line_points.len() == 1 {
                            // Single point - draw a small circle
                            ui.painter()
                                .circle_filled(line_points[0], 2.0, series_color);
                        }
                    }
                }

                // Fallback: Draw bar if we have timestamp data but no sparkline points
                if !has_sparkline_points && let Some((start_ts, end_ts)) = summary.timestamp_range {
                    let start_offset = (start_ts.0 - display_start.0) as f64;
                    let end_offset = (end_ts.0 - display_start.0) as f64;

                    let start_x = timeline_start_x + (start_offset * pixels_per_us) as f32;
                    let end_x = timeline_start_x + (end_offset * pixels_per_us) as f32;

                    // Clip to the visible timeline area
                    let clipped_start_x = start_x.max(timeline_start_x);
                    let clipped_end_x = end_x.min(timeline_end_x);

                    // Only draw if there's a visible portion
                    if clipped_end_x > clipped_start_x {
                        let min_width = 2.0;
                        let bar_width = (clipped_end_x - clipped_start_x).max(min_width);

                        let bar_rect = Rect::from_min_max(
                            Pos2::new(clipped_start_x, row_rect.min.y + 3.0),
                            Pos2::new(clipped_start_x + bar_width, row_rect.max.y - 3.0),
                        );

                        ui.painter().rect_filled(bar_rect, 2.0, summary.color);
                    }
                }
            }
        });

        // Draw vertical separator between labels and timeline
        ui.painter().vline(
            timeline_start_x,
            available_rect.y_range(),
            Stroke::new(1.0_f32, scheme.border_primary),
        );

        // Check if pointer is over the timeline area (right of labels)
        let pointer_in_timeline = response
            .hover_pos()
            .map(|pos| pos.x > timeline_start_x)
            .unwrap_or(false);

        // Double-click to reset zoom and pan
        if response.double_clicked() {
            pane.zoom_factor = 1.0;
            pane.pan_offset_us = 0;
        }

        // Handle drag for panning
        if response.dragged() {
            let delta = response.drag_delta();

            if pointer_in_timeline {
                // Horizontal drag in timeline area = horizontal pan
                // Convert pixel delta to timestamp delta
                let us_per_pixel = time_span / timeline_width as f64;
                let pan_delta = (-delta.x as f64 * us_per_pixel) as i64;
                pane.pan_offset_us = (pane.pan_offset_us + pan_delta).max(0);
            }

            // Vertical drag anywhere = vertical scroll
            pane.scroll_offset = (pane.scroll_offset - delta.y).max(0.0);
        }

        // Zoom with scroll wheel when hovering over timeline
        if response.hovered() {
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta);

            if pointer_in_timeline && scroll_delta.y != 0.0 {
                // Horizontal zoom in timeline area
                const ZOOM_SENSITIVITY: f32 = 0.002;
                let zoom_delta = scroll_delta.y * ZOOM_SENSITIVITY;

                // Get pointer position relative to timeline for zoom centering
                let pointer_x = response
                    .hover_pos()
                    .map(|pos| pos.x - timeline_start_x)
                    .unwrap_or(timeline_width / 2.0);
                let pointer_ratio = pointer_x / timeline_width;

                let old_zoom = pane.zoom_factor;
                pane.zoom_factor = (pane.zoom_factor * (1.0 + zoom_delta)).clamp(1.0, 100.0);

                // Adjust pan to zoom towards pointer position
                if pane.zoom_factor != old_zoom {
                    let old_span = base_span / old_zoom as f64;
                    let new_span = base_span / pane.zoom_factor as f64;
                    let span_delta = old_span - new_span;
                    let pan_adjustment = (span_delta * pointer_ratio as f64) as i64;
                    pane.pan_offset_us = (pane.pan_offset_us + pan_adjustment).max(0);
                }
            } else if scroll_delta.y != 0.0 {
                // Vertical scroll in label area
                pane.scroll_offset = (pane.scroll_offset - scroll_delta.y).max(0.0);
            }
        }

        pane
    }
}

/// Process a single table's timestamp range and row count result
fn process_time_range_and_count(
    table_name: &str,
    batch: &RecordBatch,
    time_ranges: &mut ComponentTimeRanges,
) {
    if batch.num_rows() == 0 {
        return;
    }

    let schema = batch.schema();

    // Find column indices
    let min_col = schema.fields().iter().position(|f| f.name() == "min_time");
    let max_col = schema.fields().iter().position(|f| f.name() == "max_time");
    let count_col = schema.fields().iter().position(|f| f.name() == "row_count");

    // Extract time range
    if let (Some(min_idx), Some(max_idx)) = (min_col, max_col) {
        let min_array = batch.column(min_idx);
        let max_array = batch.column(max_idx);

        let min_timestamps = min_array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>();
        let max_timestamps = max_array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>();

        if let (Some(mins), Some(maxs)) = (min_timestamps, max_timestamps)
            && !mins.is_null(0)
            && !maxs.is_null(0)
        {
            let min_ts = Timestamp(mins.value(0));
            let max_ts = Timestamp(maxs.value(0));
            time_ranges
                .ranges
                .insert(table_name.to_string(), (min_ts, max_ts));
        }
    }

    // Extract row count
    if let Some(count_idx) = count_col {
        let count_array = batch.column(count_idx);

        // Try different integer types for count
        if let Some(counts) = count_array
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
        {
            if !counts.is_null(0) {
                time_ranges
                    .row_counts
                    .insert(table_name.to_string(), counts.value(0) as usize);
            }
        } else if let Some(counts) = count_array
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            && !counts.is_null(0)
        {
            time_ranges
                .row_counts
                .insert(table_name.to_string(), counts.value(0) as usize);
        }
    }
}

/// Extract f64 values from an Arrow array, returning multiple series for vector types
/// Returns Vec of series, where each series is a Vec of values (one per row)
fn extract_values_from_array(array: &dyn Array) -> Option<Vec<Vec<f64>>> {
    use arrow::array::{
        BooleanArray, FixedSizeListArray, Float32Array, Int8Array, Int16Array, Int32Array,
        Int64Array, ListArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };

    // Handle FixedSizeListArray (vector/matrix types) - return each element as a separate series
    // NOTE: We use f64::NAN for null rows to preserve array indices, ensuring 1:1
    // correspondence with the timestamp array.
    if let Some(list_array) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let values_array = list_array.values();
        let list_size = list_array.value_length() as usize;
        let num_rows = list_array.len();

        // Extract inner values as a flat array
        let inner_series = extract_values_from_array(values_array.as_ref())?;
        // Inner should be a single series with all values flattened
        let inner_values = inner_series.into_iter().next()?;

        // Split into separate series per element
        let mut series: Vec<Vec<f64>> = (0..list_size)
            .map(|_| Vec::with_capacity(num_rows))
            .collect();

        for row in 0..num_rows {
            if list_array.is_null(row) {
                // Push NAN for each element to preserve index alignment with timestamps
                for s in series.iter_mut() {
                    s.push(f64::NAN);
                }
                continue;
            }
            let start = row * list_size;
            for (elem_idx, s) in series.iter_mut().enumerate() {
                if start + elem_idx < inner_values.len() {
                    s.push(inner_values[start + elem_idx]);
                } else {
                    // Push NAN for missing data to preserve index alignment with timestamps
                    s.push(f64::NAN);
                }
            }
        }

        return Some(series);
    }

    // Handle ListArray (variable-size lists) - use first element of each row as a single series
    // For variable-size lists, we can't split into multiple series since sizes vary per row.
    // Instead, extract the first element of each list to create a single series.
    if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
        let values_array = list_array.values();
        let num_rows = list_array.len();

        // Extract inner values as a flat array
        let inner_series = extract_values_from_array(values_array.as_ref())?;
        let inner_values = inner_series.into_iter().next()?;

        // For each row, take the first element (or NAN if empty/null)
        let mut series = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            if list_array.is_null(row) {
                series.push(f64::NAN);
                continue;
            }
            let start = list_array.value_offsets()[row] as usize;
            let end = list_array.value_offsets()[row + 1] as usize;
            if start < end && start < inner_values.len() {
                series.push(inner_values[start]);
            } else {
                series.push(f64::NAN);
            }
        }

        return Some(vec![series]);
    }

    // Handle scalar numeric types - return as single series
    // NOTE: We use f64::NAN for null values to preserve array indices.
    // This ensures 1:1 correspondence with the timestamp array, allowing
    // sparklines to show gaps where data is missing.

    // Helper macro to extract values from a typed array
    macro_rules! extract_numeric {
        ($array:expr, $type:ty) => {
            if let Some(vals) = $array.as_any().downcast_ref::<$type>() {
                return Some(vec![
                    (0..vals.len())
                        .map(|i| {
                            if vals.is_null(i) {
                                f64::NAN
                            } else {
                                vals.value(i) as f64
                            }
                        })
                        .collect(),
                ]);
            }
        };
    }

    // Try all numeric types
    extract_numeric!(array, Float64Array);
    extract_numeric!(array, Float32Array);
    extract_numeric!(array, Int64Array);
    extract_numeric!(array, Int32Array);
    extract_numeric!(array, Int16Array);
    extract_numeric!(array, Int8Array);
    extract_numeric!(array, UInt64Array);
    extract_numeric!(array, UInt32Array);
    extract_numeric!(array, UInt16Array);
    extract_numeric!(array, UInt8Array);

    // Handle boolean as 0.0/1.0
    if let Some(vals) = array.as_any().downcast_ref::<BooleanArray>() {
        return Some(vec![
            (0..vals.len())
                .map(|i| {
                    if vals.is_null(i) {
                        f64::NAN
                    } else if vals.value(i) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        ]);
    }

    None
}

/// Process sparkline data result from SparklineQuery.
/// The server has already applied LTTB downsampling and returns a simple schema:
/// - time: TimestampMicrosecond
/// - value: Float64 (first scalar element from vector types)
fn process_sparkline_result(
    table_name: &str,
    batch: &RecordBatch,
    sparklines: &mut HashMap<String, SparklineData>,
) {
    if batch.num_rows() == 0 {
        return;
    }

    let schema = batch.schema();

    // Find the time column
    let time_col = schema.fields().iter().position(|f| f.name() == "time");
    let Some(time_idx) = time_col else {
        return; // Missing time column - skip silently
    };

    // Find the value column (from SparklineQuery it's always "value")
    let value_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "value" || (f.name() != "time" && f.name() != "rn"));
    let Some(value_idx) = value_idx else {
        return; // Missing data column - skip silently
    };

    let time_array = batch.column(time_idx);
    let value_array = batch.column(value_idx);

    // Try to get timestamps
    let Some(timestamps) = time_array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
    else {
        return; // Non-timestamp time column - skip silently
    };

    // For SparklineQuery, values are always Float64
    // Fall back to extract_values_from_array for backward compatibility with SQLQuery
    let values: Vec<f64> =
        if let Some(float_array) = value_array.as_any().downcast_ref::<Float64Array>() {
            float_array.values().to_vec()
        } else {
            // Fallback for complex types (backward compat with old SQLQuery results)
            let Some(value_series) = extract_values_from_array(value_array.as_ref()) else {
                return;
            };
            if value_series.is_empty() || value_series[0].is_empty() {
                return;
            }
            value_series[0].clone()
        };

    let point_count = timestamps.len();

    // Build points - server already did LTTB downsampling, no client-side downsampling needed
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut points = Vec::with_capacity(point_count);

    for (i, &value) in values.iter().enumerate() {
        if i >= timestamps.len() {
            break;
        }
        if timestamps.is_null(i) {
            continue;
        }
        // Skip NAN values - this creates gaps in the sparkline where data is missing
        if value.is_nan() {
            continue;
        }
        let time = timestamps.value(i);
        points.push((time, value));
        y_min = y_min.min(value);
        y_max = y_max.max(value);
    }

    // Ensure we have some Y range for flat lines
    if (y_max - y_min).abs() < 1e-10 {
        y_min -= 1.0;
        y_max += 1.0;
    }

    sparklines.insert(
        table_name.to_string(),
        SparklineData {
            series: vec![SparklineSeries { points }],
            y_min,
            y_max,
            raw_point_count: point_count,
        },
    );
}

/// System that triggers component time range queries when components become available.
/// Queries are run once on load - no periodic refresh.
/// Initialize time range queries by collecting table names; the batches
/// themselves are sent by [`dispatch_time_range_queries`].
pub fn trigger_time_range_queries(
    eql_context: Res<EqlContext>,
    mut time_ranges: ResMut<ComponentTimeRanges>,
) {
    // Only initialize when state is NotStarted
    if !matches!(time_ranges.state, TimeRangeQueryState::NotStarted) {
        return;
    }

    // Collect table names
    fn collect_table_names(
        parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
        result: &mut Vec<(ComponentId, String)>,
    ) {
        for part in parts.values() {
            if let Some(component) = &part.component {
                let table_name = component_to_table_name(&component.name);
                result.push((part.id, table_name));
            }
            collect_table_names(&part.children, result);
        }
    }

    let mut table_names: Vec<(ComponentId, String)> = Vec::new();
    collect_table_names(&eql_context.0.component_parts, &mut table_names);

    // Skip if no components yet
    if table_names.is_empty() {
        return;
    }

    time_ranges.state = TimeRangeQueryState::QueryingTimeRanges(Instant::now());
    time_ranges.total_queries = table_names.len();
    time_ranges.completed_queries = 0;
    time_ranges.current_batch = 0;
    time_ranges.tables_to_query = table_names.into_iter().map(|(_a, b)| b).collect();
}

/// Send the time range and row count queries, in batches bounded by
/// [`MAX_CONCURRENT_QUERIES`] because `RequestId` is a `u8`.
///
/// These ranges drive component filtering and the timeline step, neither of
/// which should depend on the Data Overview panel being open, so this runs as a
/// plain system rather than from the panel's widget.
pub fn dispatch_time_range_queries(
    mut time_ranges: ResMut<ComponentTimeRanges>,
    mut commands: Commands,
) {
    if !matches!(
        time_ranges.state,
        TimeRangeQueryState::QueryingTimeRanges(_)
    ) || time_ranges.pending_queries > 0
    {
        return;
    }
    let batch_start = time_ranges.current_batch * MAX_CONCURRENT_QUERIES;
    if batch_start >= time_ranges.tables_to_query.len() {
        return;
    }
    let batch_end = (batch_start + MAX_CONCURRENT_QUERIES).min(time_ranges.tables_to_query.len());
    time_ranges.pending_queries = batch_end - batch_start;
    time_ranges.current_batch += 1;

    // These requests are queued as commands and so outlive the frame. A
    // session reset in between cancels the registered handlers, but not ones
    // whose registration is still sitting in the command queue.
    let generation = time_ranges.generation();
    let batch: Vec<String> = time_ranges.tables_to_query[batch_start..batch_end].to_vec();
    for table_name in batch {
        let query = format!(
            "SELECT min(time) as min_time, max(time) as max_time, count(*) as row_count FROM {}",
            table_name
        );
        commands.send_req_reply(
            SQLQuery(query),
            move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                  mut time_ranges: ResMut<ComponentTimeRanges>| {
                if !time_ranges.accepts(generation) {
                    return true;
                }
                // Completion markers (batch: None) are ignored; the handler is
                // removed either way, freeing the request id.
                match res {
                    Ok(ipc) => {
                        if let Some(batch_data) = ipc.batch {
                            let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                            let mut buffer = arrow::buffer::Buffer::from(batch_data.into_owned());
                            if let Some(batch) = decoder.decode(&mut buffer).ok().and_then(|b| b) {
                                process_time_range_and_count(&table_name, &batch, &mut time_ranges);
                            }
                            finish_time_range_query(&mut time_ranges);
                        }
                    }
                    Err(_) => finish_time_range_query(&mut time_ranges),
                }
                true
            },
        );
    }
}

/// A recording that declares no `simulation_time_step` still has an implicit
/// one: the finest spacing at which any of its components was sampled, which is
/// the resolution the timeline can actually be advanced at.
///
/// Spacing is taken over each series as a whole rather than a leading window.
/// The opening samples of a real recording are an ingest burst — on FT27 flight
/// data the first 64 IMU samples imply 9 µs where the series really runs at
/// 1 ms — and since the finest component wins, the worst-biased one would
/// otherwise decide the result every time.
pub fn estimate_sim_time_step_from_ranges(
    time_ranges: Res<ComponentTimeRanges>,
    path_reg: Res<ComponentPathRegistry>,
    schema_reg: Res<ComponentSchemaRegistry>,
    mut fetch: ResMut<SimTimeStepFetch>,
    mut time_step: ResMut<SimulationTimeStep>,
) {
    if fetch.defers_to_declared(&path_reg, &schema_reg) {
        return;
    }
    // Deliberately not gated on the ranges changing. A DB that declares a rate
    // only releases the measured fallback once its fetch runs out of retries,
    // seconds after the ranges have gone quiet. Once a rate is published,
    // though, only fresh ranges can refine it.
    if fetch.source() == SimTimeStepSource::Estimated && !time_ranges.is_changed() {
        return;
    }
    let Some(micros) = finest_sample_spacing_micros(&time_ranges) else {
        return;
    };
    if let Some(dt) = fetch.record_estimate(micros) {
        time_step.0 = dt;
    }
}

/// Finest mean sample spacing across every measured component, in micros.
fn finest_sample_spacing_micros(time_ranges: &ComponentTimeRanges) -> Option<i64> {
    time_ranges
        .ranges
        .iter()
        .filter_map(|(table, (min, max))| {
            // A single sample spans no interval.
            let intervals =
                i64::try_from(time_ranges.row_counts.get(table)?.checked_sub(1)?).ok()?;
            let span = max.0.checked_sub(min.0)?;
            (intervals > 0 && span > 0).then(|| span / intervals)
        })
        .filter(|micros| *micros > 0)
        .min()
}

#[cfg(test)]
mod sample_spacing_tests {
    use super::*;

    fn ranges(entries: &[(&str, i64, i64, usize)]) -> ComponentTimeRanges {
        let mut time_ranges = ComponentTimeRanges::default();
        for (table, min, max, count) in entries {
            time_ranges
                .ranges
                .insert(table.to_string(), (Timestamp(*min), Timestamp(*max)));
            time_ranges.row_counts.insert(table.to_string(), *count);
        }
        time_ranges
    }

    #[test]
    fn the_finest_component_sets_the_step() {
        // FT27 flight data: a 1 kHz IMU alongside much slower channels.
        let time_ranges = ranges(&[
            ("mfimumessage_gyro", 0, 837_856_000, 837_857),
            ("cn0message_cn0", 0, 837_768_000, 56_814),
            ("controlmessage_aileron_cmd_deg", 0, 72_402_000, 69_352),
        ]);

        assert_eq!(finest_sample_spacing_micros(&time_ranges), Some(1_000));
    }

    #[test]
    fn a_leading_burst_does_not_shrink_the_step() {
        // Same series, whose first samples arrive microseconds apart before
        // settling: whole-series spacing must ignore that opening.
        let time_ranges = ranges(&[("mfimumessage_gyro", 1_000, 1_000_001_000, 1_000_001)]);

        assert_eq!(finest_sample_spacing_micros(&time_ranges), Some(1_000));
    }

    #[test]
    fn components_that_measure_nothing_are_skipped() {
        let time_ranges = ranges(&[
            ("empty", 0, 0, 0),
            ("one_sample", 500, 500, 1),
            ("all_at_once", 700, 700, 64),
            ("real", 0, 4_000, 5),
        ]);

        assert_eq!(finest_sample_spacing_micros(&time_ranges), Some(1_000));
        assert_eq!(finest_sample_spacing_micros(&ranges(&[])), None);
    }

    #[test]
    fn a_count_without_a_range_is_ignored() {
        let mut time_ranges = ranges(&[("real", 0, 4_000, 5)]);
        time_ranges.row_counts.insert("orphan".to_string(), 10_000);

        assert_eq!(finest_sample_spacing_micros(&time_ranges), Some(1_000));
    }

    #[test]
    fn a_reset_disowns_queries_already_in_flight() {
        let mut time_ranges = ranges(&[("gyro", 0, 1_000_000, 1_001)]);
        let in_flight = time_ranges.generation();
        assert!(time_ranges.accepts(in_flight));

        time_ranges.reset();

        assert!(
            !time_ranges.accepts(in_flight),
            "cancelling handlers cannot reach requests still sitting in the command queue"
        );
        assert!(time_ranges.accepts(time_ranges.generation()));
    }

    #[test]
    fn one_unguarded_reply_would_strand_the_new_recording() {
        // What the generation guard exists to prevent: a reset leaves
        // `total_queries` at zero, so the very first reply to land satisfies
        // `completed >= total` and moves the state on. Since
        // `trigger_time_range_queries` only ever starts from `NotStarted`,
        // nothing would re-measure the new recording.
        let mut time_ranges = ComponentTimeRanges::default();
        time_ranges.reset();
        assert!(matches!(time_ranges.state, TimeRangeQueryState::NotStarted));

        finish_time_range_query(&mut time_ranges);

        assert!(!matches!(
            time_ranges.state,
            TimeRangeQueryState::NotStarted
        ));
    }

    #[test]
    fn a_full_batch_moves_on_to_sparklines() {
        let mut time_ranges = ComponentTimeRanges {
            state: TimeRangeQueryState::QueryingTimeRanges(Instant::now()),
            pending_queries: 2,
            total_queries: 2,
            ..Default::default()
        };

        finish_time_range_query(&mut time_ranges);
        assert!(matches!(
            time_ranges.state,
            TimeRangeQueryState::QueryingTimeRanges(_)
        ));

        finish_time_range_query(&mut time_ranges);
        assert!(matches!(
            time_ranges.state,
            TimeRangeQueryState::QueryingSparklines(_)
        ));
    }

    fn estimate_app() -> App {
        let mut app = App::new();
        app.init_resource::<ComponentTimeRanges>()
            .init_resource::<ComponentPathRegistry>()
            .init_resource::<ComponentSchemaRegistry>()
            .init_resource::<SimTimeStepFetch>()
            .insert_resource(SimulationTimeStep(0.0))
            .add_systems(Update, estimate_sim_time_step_from_ranges);
        app
    }

    fn declare_a_rate(app: &mut App) {
        use impeller2::component::Component;
        let id = <SimulationTimeStep as Component>::COMPONENT_ID;
        let schema = impeller2::schema::Schema::new(impeller2::types::PrimType::F64, [0usize; 0])
            .expect("schema");
        app.world_mut()
            .resource_mut::<ComponentSchemaRegistry>()
            .0
            .insert(id, schema);
    }

    #[test]
    fn the_measured_rate_lands_after_the_declared_one_gives_up() {
        let mut app = estimate_app();
        declare_a_rate(&mut app);
        {
            let mut time_ranges = app.world_mut().resource_mut::<ComponentTimeRanges>();
            *time_ranges = ranges(&[("gyro", 0, 1_000_000, 1_001)]);
        }
        app.update();
        assert_eq!(
            app.world().resource::<SimulationTimeStep>().0,
            0.0,
            "a declared rate must not be overridden while it may still arrive"
        );

        // The declared rate drops out of the running, on a frame where the
        // ranges are untouched — exactly what the retry budget running out
        // looks like, seconds after the queries went quiet.
        app.world_mut()
            .resource_mut::<ComponentSchemaRegistry>()
            .0
            .clear();
        app.update();

        assert_eq!(app.world().resource::<SimulationTimeStep>().0, 0.001);
        assert_eq!(
            app.world().resource::<SimTimeStepFetch>().source(),
            SimTimeStepSource::Estimated
        );
    }

    #[test]
    fn later_query_batches_refine_the_rate() {
        let mut app = estimate_app();
        {
            let mut time_ranges = app.world_mut().resource_mut::<ComponentTimeRanges>();
            *time_ranges = ranges(&[("slow", 0, 1_000_000, 1_001)]);
        }
        app.update();
        assert_eq!(app.world().resource::<SimulationTimeStep>().0, 0.001);

        {
            let mut time_ranges = app.world_mut().resource_mut::<ComponentTimeRanges>();
            time_ranges
                .ranges
                .insert("fast".into(), (Timestamp(0), Timestamp(1_000_000)));
            time_ranges.row_counts.insert("fast".into(), 8_001);
        }
        app.update();

        assert_eq!(app.world().resource::<SimulationTimeStep>().0, 0.000125);
    }
}
