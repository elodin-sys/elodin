//! Data Overview Panel
//!
//! Provides a high-level timeline view showing all database components and their
//! data density across time. Useful for identifying gaps or disparities in data.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use arrow::array::{Array, Float64Array, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_egui::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use convert_case::{Case, Casing};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use crate::{EqlContext, SelectedTimeRange, ui::widgets::WidgetSystem};

use super::colors::get_scheme;

// Re-export for use in component collection
use eql;

/// Maximum number of points per series in each sparkline
const SPARKLINE_MAX_POINTS: usize = 2000;

/// Refresh interval for live data updates
const SPARKLINE_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

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
    /// Map from component table name to sparkline data
    pub sparklines: HashMap<String, SparklineData>,
    /// Number of queries still pending
    pub pending_queries: usize,
    /// Total number of queries sent
    pub total_queries: usize,
    /// State of the query
    pub state: TimeRangeQueryState,
}

#[derive(Default, Clone)]
pub enum TimeRangeQueryState {
    #[default]
    NotStarted,
    Querying(Instant),
    Ready,
    Error(String),
}

/// Pane data for the DataOverview panel
#[derive(Clone)]
pub struct DataOverviewPane {
    /// Cached screen rect for rendering
    pub rect: Option<egui::Rect>,
    /// Vertical scroll offset for component list
    pub scroll_offset: f32,
    /// Horizontal zoom factor (1.0 = fit to data, >1.0 = zoomed in)
    pub zoom_factor: f32,
    /// Horizontal pan offset in microseconds from the data start
    pub pan_offset_us: i64,
    /// Whether we've triggered the initial query
    pub query_triggered: bool,
    /// Last time we refreshed queries (for live updates)
    pub last_refresh: Option<Instant>,
    /// Cached data time range (min, max) from component queries
    pub cached_data_range: Option<(Timestamp, Timestamp)>,
    /// Last known drag position for pan calculation
    pub last_drag_pos: Option<egui::Pos2>,
}

impl Default for DataOverviewPane {
    fn default() -> Self {
        Self {
            rect: None,
            scroll_offset: 0.0,
            zoom_factor: 1.0,
            pan_offset_us: 0,
            query_triggered: false,
            last_refresh: None,
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
    pub timestamp_range: Option<(Timestamp, Timestamp)>,
}

/// Generate a distinct color for a given row index using golden ratio hue distribution
pub fn row_color(index: usize) -> Color32 {
    const GOLDEN_RATIO_CONJUGATE: f32 = 0.618033988749895;
    let hue = ((index as f32) * GOLDEN_RATIO_CONJUGATE * 360.0) % 360.0;
    let (r, g, b) = hsl_to_rgb(hue, 0.7, 0.55);
    Color32::from_rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

/// Convert HSL to RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
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
/// The database uses `to_case(Case::Snake)` followed by replacing '.' with '_'.
fn component_to_table_name(full_component_name: &str) -> String {
    // Full component name is like "GpsPosMessage1.VACC"
    // Table name is like "gps_pos_message_1_vacc"
    // Must match the conversion in libs/db/src/arrow/mod.rs
    full_component_name.to_case(Case::Snake).replace('.', "_")
}

/// Widget for rendering the Data Overview panel
#[derive(SystemParam)]
pub struct DataOverviewWidget<'w, 's> {
    eql_context: Res<'w, EqlContext>,
    selected_range: Res<'w, SelectedTimeRange>,
    time_ranges: ResMut<'w, ComponentTimeRanges>,
    commands: Commands<'w, 's>,
}

impl WidgetSystem for DataOverviewWidget<'_, '_> {
    type Args = DataOverviewPane;
    type Output = DataOverviewPane;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        mut pane: Self::Args,
    ) -> Self::Output {
        let mut params = state.get_mut(world);
        
        let scheme = get_scheme();
        let available_rect = ui.available_rect_before_wrap();
        pane.rect = Some(available_rect);

        // Constants for layout
        const LABEL_WIDTH: f32 = 280.0;  // Wider to show full component paths
        const ROW_HEIGHT: f32 = 16.0;

        // Collect component info from EqlContext
        fn collect_components(
            parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
            result: &mut Vec<(ComponentId, String, String)>,
        ) {
            for (_name, part) in parts {
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
        
        // Check if we should refresh queries for live data
        let should_refresh = pane.last_refresh
            .map(|t| t.elapsed() > SPARKLINE_REFRESH_INTERVAL)
            .unwrap_or(true);

        // Trigger query for timestamp ranges if not started or need refresh
        let should_query = (!pane.query_triggered && !component_list.is_empty())
            || (should_refresh 
                && matches!(params.time_ranges.state, TimeRangeQueryState::Ready)
                && !component_list.is_empty());
        
        if should_query {
            if matches!(params.time_ranges.state, TimeRangeQueryState::NotStarted | TimeRangeQueryState::Ready) {
                params.time_ranges.state = TimeRangeQueryState::Querying(Instant::now());
                // Each component needs 2 queries: time range + sparkline data
                params.time_ranges.total_queries = component_list.len() * 2;
                params.time_ranges.pending_queries = component_list.len() * 2;
                pane.last_refresh = Some(Instant::now());
                
                // Query each table individually to handle missing tables gracefully
                for (_, _, table_name) in component_list.iter() {
                    // Query 1: Get time range
                    let table_name_clone = table_name.clone();
                    let query = format!(
                        "SELECT min(time) as min_time, max(time) as max_time FROM {}",
                        table_name
                    );
                    
                    params.commands.send_req_reply(
                        SQLQuery(query),
                        move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                              mut time_ranges: ResMut<ComponentTimeRanges>| {
                            time_ranges.pending_queries = time_ranges.pending_queries.saturating_sub(1);
                            
                            if let Ok(ipc) = res {
                                let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                if let Some(batch_data) = ipc.batch {
                                    let mut buffer = arrow::buffer::Buffer::from(batch_data.into_owned());
                                    if let Some(batch) = decoder.decode(&mut buffer).ok().and_then(|b| b) {
                                        process_single_table_result(&table_name_clone, &batch, &mut time_ranges.ranges);
                                    }
                                }
                            }
                            
                            if time_ranges.pending_queries == 0 {
                                time_ranges.state = TimeRangeQueryState::Ready;
                            }
                            true
                        },
                    );
                    
                    // Query 2: Get sparkline data
                    // Query all data and downsample client-side to get even distribution
                    let table_name_clone2 = table_name.clone();
                    let sparkline_query = format!(
                        "SELECT * FROM {} ORDER BY time",
                        table_name
                    );
                    
                    params.commands.send_req_reply(
                        SQLQuery(sparkline_query),
                        move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                              mut time_ranges: ResMut<ComponentTimeRanges>| {
                            time_ranges.pending_queries = time_ranges.pending_queries.saturating_sub(1);
                            
                            if let Ok(ipc) = res {
                                let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                if let Some(batch_data) = ipc.batch {
                                    let mut buffer = arrow::buffer::Buffer::from(batch_data.into_owned());
                                    if let Some(batch) = decoder.decode(&mut buffer).ok().and_then(|b| b) {
                                        process_sparkline_result(&table_name_clone2, &batch, &mut time_ranges.sparklines);
                                    }
                                }
                            }
                            
                            if time_ranges.pending_queries == 0 {
                                time_ranges.state = TimeRangeQueryState::Ready;
                            }
                            true
                        },
                    );
                }
                pane.query_triggered = true;
            }
        }

        // Build summaries with cached time ranges
        let mut summaries: Vec<ComponentTimestampSummary> = Vec::new();
        for (component_id, label, table_name) in component_list.iter() {
            let timestamp_range = params.time_ranges.ranges.get(table_name).copied();
            
            summaries.push(ComponentTimestampSummary {
                component_id: *component_id,
                label: label.clone(),
                table_name: table_name.clone(),
                color: Color32::WHITE, // Will be assigned after sorting
                timestamp_range,
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
            summary.color = row_color(index);
        }

        // Calculate time range from actual component data ranges
        // This ensures the timeline scales to fit the actual data
        let data_time_range: Option<(Timestamp, Timestamp)> = summaries
            .iter()
            .filter_map(|s| s.timestamp_range)
            .fold(None, |acc: Option<(Timestamp, Timestamp)>, (min, max)| {
                Some(match acc {
                    None => (min, max),
                    Some((a_min, a_max)) => (
                        Timestamp(a_min.0.min(min.0)),
                        Timestamp(a_max.0.max(max.0)),
                    ),
                })
            });

        // Update cached data range when queries complete
        if matches!(params.time_ranges.state, TimeRangeQueryState::Ready) && data_time_range.is_some() {
            pane.cached_data_range = data_time_range;
        }

        // Use cached data range, falling back to selected_range if no data yet
        let (data_min, data_max) = pane.cached_data_range
            .or(data_time_range)
            .unwrap_or_else(|| (params.selected_range.0.start, params.selected_range.0.end));

        // Add a small margin (5%) to the data range for better visualization
        let base_span = (data_max.0 - data_min.0).max(1_000_000) as f64; // minimum 1 second
        let margin = (base_span * 0.05) as i64;
        let display_data_min = Timestamp(data_min.0.saturating_sub(margin));

        // Apply zoom factor (1.0 = fit all data, >1.0 = zoomed in)
        let zoomed_span = base_span / pane.zoom_factor.max(0.01) as f64;
        
        // Apply pan offset (clamped to prevent scrolling beyond data)
        let max_pan = (base_span - zoomed_span).max(0.0) as i64;
        pane.pan_offset_us = pane.pan_offset_us.clamp(0, max_pan);
        
        let display_start = Timestamp(display_data_min.0 + pane.pan_offset_us);
        let display_end = Timestamp(display_start.0 + zoomed_span as i64);

        let timeline_width = (available_rect.width() - LABEL_WIDTH).max(100.0);
        let time_span = (display_end.0 - display_start.0).max(1) as f64;
        let pixels_per_us = timeline_width as f64 / time_span;
        let timeline_start_x = available_rect.min.x + LABEL_WIDTH;
        let timeline_end_x = timeline_start_x + timeline_width;

        // Header with component count and query status
        let header_height = ROW_HEIGHT;
        let header_rect = Rect::from_min_size(
            available_rect.min,
            Vec2::new(available_rect.width(), header_height),
        );
        ui.painter().rect_filled(header_rect, 0.0, scheme.bg_secondary);
        
        // Header text with status
        let status_text = match &params.time_ranges.state {
            TimeRangeQueryState::NotStarted => format!("Components: {} (loading...)", summaries.len()),
            TimeRangeQueryState::Querying(_) => {
                let done = params.time_ranges.total_queries - params.time_ranges.pending_queries;
                format!("Components: {} (querying {}/{}...)", summaries.len(), done, params.time_ranges.total_queries)
            }
            TimeRangeQueryState::Ready => {
                let with_data = summaries.iter().filter(|s| s.timestamp_range.is_some()).count();
                format!("Components: {} ({} with data)", summaries.len(), with_data)
            }
            TimeRangeQueryState::Error(e) => format!("Components: {} (error: {})", summaries.len(), e),
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
                    Pos2::new(available_rect.min.x, content_start_y + row_y - pane.scroll_offset),
                    Vec2::new(available_rect.width(), ROW_HEIGHT),
                );

                // Skip if not visible
                if row_rect.max.y < available_rect.min.y || row_rect.min.y > available_rect.max.y {
                    continue;
                }

                // Alternate row background
                if row_index % 2 == 0 {
                    ui.painter().rect_filled(
                        row_rect,
                        0.0,
                        scheme.bg_primary,
                    );
                }

                // Draw label with component color
                let label_rect = Rect::from_min_size(
                    row_rect.min,
                    Vec2::new(LABEL_WIDTH, ROW_HEIGHT),
                );
                
                // Color indicator
                ui.painter().circle_filled(
                    Pos2::new(label_rect.min.x + 8.0, label_rect.center().y),
                    4.0,
                    summary.color,
                );

                // Component name (truncated if needed)
                let mut label = summary.label.clone();
                if label.len() > 38 {
                    label.truncate(35);
                    label.push_str("...");
                }
                
                ui.painter().text(
                    Pos2::new(label_rect.min.x + 16.0, label_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(10.0),
                    summary.color,
                );

                // Draw sparkline if we have data, otherwise fall back to bar
                let sparkline = params.time_ranges.sparklines.get(&summary.table_name);
                
                if let Some(sparkline_data) = sparkline {
                    if !sparkline_data.series.is_empty() {
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
                                let (r, g, b, a) = (base_color.r(), base_color.g(), base_color.b(), base_color.a());
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
                                    Stroke::new(1.0, series_color),
                                ));
                            } else if line_points.len() == 1 {
                                // Single point - draw a small circle
                                ui.painter().circle_filled(
                                    line_points[0],
                                    2.0,
                                    series_color,
                                );
                            }
                        }
                    }
                } else if let Some((start_ts, end_ts)) = summary.timestamp_range {
                    // Fallback: Draw bar if we have timestamp data but no sparkline
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
                        
                        ui.painter().rect_filled(
                            bar_rect,
                            2.0,
                            summary.color,
                        );
                    }
                }
            }
        });

        // Draw vertical separator between labels and timeline
        ui.painter().vline(
            timeline_start_x,
            available_rect.y_range(),
            Stroke::new(1.0, scheme.border_primary),
        );

        // Handle interactions
        let response = ui.interact(available_rect, ui.id().with("data_overview"), Sense::click_and_drag());
        
        // Check if pointer is over the timeline area (right of labels)
        let pointer_in_timeline = response.hover_pos()
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
                let pointer_x = response.hover_pos()
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

/// Process a single table's timestamp range result
fn process_single_table_result(
    table_name: &str,
    batch: &RecordBatch,
    ranges: &mut HashMap<String, (Timestamp, Timestamp)>,
) {
    if batch.num_rows() == 0 {
        return;
    }
    
    let schema = batch.schema();
    
    // Find column indices
    let min_col = schema.fields().iter().position(|f| f.name() == "min_time");
    let max_col = schema.fields().iter().position(|f| f.name() == "max_time");
    
    let (Some(min_idx), Some(max_idx)) = (min_col, max_col) else {
        return;
    };
    
    let min_array = batch.column(min_idx);
    let max_array = batch.column(max_idx);
    
    // Try timestamp arrays
    let min_timestamps = min_array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>();
    let max_timestamps = max_array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>();
    
    if let (Some(mins), Some(maxs)) = (min_timestamps, max_timestamps) {
        // Should only have one row with min/max
        if !mins.is_null(0) && !maxs.is_null(0) {
            let min_ts = Timestamp(mins.value(0));
            let max_ts = Timestamp(maxs.value(0));
            ranges.insert(table_name.to_string(), (min_ts, max_ts));
        }
    }
}

/// Extract f64 values from an Arrow array, returning multiple series for vector types
/// Returns Vec of series, where each series is a Vec of values (one per row)
fn extract_values_from_array(array: &dyn Array) -> Option<Vec<Vec<f64>>> {
    use arrow::array::{FixedSizeListArray, Float32Array, Int32Array, Int64Array, UInt32Array, UInt64Array};
    
    // Handle FixedSizeListArray (vector/matrix types) - return each element as a separate series
    if let Some(list_array) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let values_array = list_array.values();
        let list_size = list_array.value_length() as usize;
        let num_rows = list_array.len();
        
        // Extract inner values as a flat array
        let inner_series = extract_values_from_array(values_array.as_ref())?;
        // Inner should be a single series with all values flattened
        let inner_values = inner_series.into_iter().next()?;
        
        // Split into separate series per element
        let mut series: Vec<Vec<f64>> = (0..list_size).map(|_| Vec::with_capacity(num_rows)).collect();
        
        for row in 0..num_rows {
            if list_array.is_null(row) {
                continue;
            }
            let start = row * list_size;
            for (elem_idx, s) in series.iter_mut().enumerate() {
                if start + elem_idx < inner_values.len() {
                    s.push(inner_values[start + elem_idx]);
                }
            }
        }
        
        return Some(series);
    }
    
    // Handle scalar numeric types - return as single series
    let values: Option<Vec<f64>> = if let Some(vals) = array.as_any().downcast_ref::<Float64Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i)) }
        }).collect())
    } else if let Some(vals) = array.as_any().downcast_ref::<Float32Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i) as f64) }
        }).collect())
    } else if let Some(vals) = array.as_any().downcast_ref::<Int64Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i) as f64) }
        }).collect())
    } else if let Some(vals) = array.as_any().downcast_ref::<Int32Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i) as f64) }
        }).collect())
    } else if let Some(vals) = array.as_any().downcast_ref::<UInt64Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i) as f64) }
        }).collect())
    } else if let Some(vals) = array.as_any().downcast_ref::<UInt32Array>() {
        Some((0..vals.len()).filter_map(|i| {
            if vals.is_null(i) { None } else { Some(vals.value(i) as f64) }
        }).collect())
    } else {
        None
    };
    
    values.map(|v| vec![v])
}

/// Process sparkline data result from SQL query
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
        return;
    };
    
    // Find first non-time column for data
    let value_idx = schema.fields().iter().position(|f| f.name() != "time");
    let Some(value_idx) = value_idx else {
        return;
    };
    
    let time_array = batch.column(time_idx);
    let value_array = batch.column(value_idx);
    
    // Try to get timestamps
    let Some(timestamps) = time_array.as_any().downcast_ref::<TimestampMicrosecondArray>() else {
        return;
    };
    
    // Extract values from the data column (may be multiple series for vector types)
    let Some(value_series) = extract_values_from_array(value_array.as_ref()) else {
        return;
    };
    
    if value_series.is_empty() || value_series[0].is_empty() {
        return;
    }
    
    let raw_point_count = timestamps.len();
    
    // Check if we already have data with the same point count (no new data)
    if let Some(existing) = sparklines.get(table_name) {
        if existing.raw_point_count == raw_point_count {
            return; // No new data, skip update
        }
    }
    
    // Build points for each series and track global Y range
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    
    let mut all_series: Vec<Vec<(i64, f64)>> = value_series.iter()
        .map(|_| Vec::with_capacity(timestamps.len()))
        .collect();
    
    // Collect all points for each series
    for i in 0..timestamps.len() {
        if timestamps.is_null(i) {
            continue;
        }
        let time = timestamps.value(i);
        
        for (series_idx, values) in value_series.iter().enumerate() {
            if i < values.len() {
                let value = values[i];
                all_series[series_idx].push((time, value));
                y_min = y_min.min(value);
                y_max = y_max.max(value);
            }
        }
    }
    
    // Downsample each series to SPARKLINE_MAX_POINTS if needed
    let series: Vec<SparklineSeries> = all_series.into_iter().map(|all_points| {
        let points = if all_points.len() <= SPARKLINE_MAX_POINTS {
            all_points
        } else {
            let step = all_points.len() as f64 / SPARKLINE_MAX_POINTS as f64;
            let mut sampled = Vec::with_capacity(SPARKLINE_MAX_POINTS);
            for i in 0..SPARKLINE_MAX_POINTS {
                let idx = (i as f64 * step) as usize;
                if idx < all_points.len() {
                    sampled.push(all_points[idx]);
                }
            }
            // Always include the last point
            if let Some(&last) = all_points.last() {
                if sampled.last() != Some(&last) {
                    sampled.push(last);
                }
            }
            sampled
        };
        SparklineSeries { points }
    }).collect();
    
    // Ensure we have some Y range for flat lines
    if (y_max - y_min).abs() < 1e-10 {
        y_min -= 1.0;
        y_max += 1.0;
    }
    
    sparklines.insert(table_name.to_string(), SparklineData {
        series,
        y_min,
        y_max,
        raw_point_count,
    });
}

/// System that triggers component time range queries when components become available.
/// This ensures filtering happens even if the Data Overview panel isn't displayed.
/// The panel's own refresh logic handles periodic updates for live data.
pub fn trigger_time_range_queries(
    eql_context: Res<EqlContext>,
    mut time_ranges: ResMut<ComponentTimeRanges>,
    mut commands: Commands,
) {
    // Check if we should start queries or handle stale queries
    let should_start = match &time_ranges.state {
        TimeRangeQueryState::NotStarted => true,
        TimeRangeQueryState::Querying(start) => {
            // Timeout stale queries after 30 seconds and reset to NotStarted
            if start.elapsed() > Duration::from_secs(30) {
                time_ranges.state = TimeRangeQueryState::NotStarted;
                true
            } else {
                false
            }
        }
        TimeRangeQueryState::Ready => false, // Panel handles refresh
        TimeRangeQueryState::Error(_) => true,
    };
    
    if !should_start {
        return;
    }

    // Collect components
    fn collect_table_names(
        parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
        result: &mut Vec<String>,
    ) {
        for (_name, part) in parts {
            if let Some(component) = &part.component {
                let table_name = component_to_table_name(&component.name);
                result.push(table_name);
            }
            collect_table_names(&part.children, result);
        }
    }

    let mut table_names: Vec<String> = Vec::new();
    collect_table_names(&eql_context.0.component_parts, &mut table_names);

    // Skip if no components yet
    if table_names.is_empty() {
        return;
    }

    // Start querying
    time_ranges.state = TimeRangeQueryState::Querying(Instant::now());
    time_ranges.total_queries = table_names.len();
    time_ranges.pending_queries = table_names.len();

    // Query each table individually
    for table_name in table_names {
        let table_name_clone = table_name.clone();
        let query = format!(
            "SELECT min(time) as min_time, max(time) as max_time FROM {}",
            table_name
        );

        commands.send_req_reply(
            SQLQuery(query),
            move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                  mut time_ranges: ResMut<ComponentTimeRanges>| {
                time_ranges.pending_queries = time_ranges.pending_queries.saturating_sub(1);

                if let Ok(ipc) = res {
                    let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                    if let Some(batch_data) = ipc.batch {
                        let mut buffer = arrow::buffer::Buffer::from(batch_data.into_owned());
                        if let Some(batch) = decoder.decode(&mut buffer).ok().and_then(|b| b) {
                            process_single_table_result(&table_name_clone, &batch, &mut time_ranges.ranges);
                        }
                    }
                }

                if time_ranges.pending_queries == 0 {
                    time_ranges.state = TimeRangeQueryState::Ready;
                }
                true
            },
        );
    }
}
