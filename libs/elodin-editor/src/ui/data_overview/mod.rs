//! Data Overview Panel
//!
//! Provides a high-level timeline view showing all database components and their
//! data density across time. Useful for identifying gaps or disparities in data.

use std::collections::HashMap;
use std::time::Instant;

use arrow::array::{Array, TimestampMicrosecondArray};
use arrow::record_batch::RecordBatch;
use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_egui::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_bevy::CommandsExt;
use impeller2_wkt::{ArrowIPC, ErrorResponse, SQLQuery};

use crate::{EqlContext, SelectedTimeRange, ui::widgets::WidgetSystem};

use super::colors::get_scheme;

// Re-export for use in component collection
use eql;

/// Resource to cache component timestamp ranges from the database
#[derive(Resource, Default)]
pub struct ComponentTimeRanges {
    /// Map from component table name to (min_timestamp, max_timestamp)
    pub ranges: HashMap<String, (Timestamp, Timestamp)>,
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
#[derive(Clone, Default)]
pub struct DataOverviewPane {
    /// Cached screen rect for rendering
    pub rect: Option<egui::Rect>,
    /// Vertical scroll offset for component list
    pub scroll_offset: f32,
    /// Horizontal zoom level (pixels per microsecond)
    pub zoom: f64,
    /// Horizontal pan offset (timestamp)
    pub pan_offset: i64,
    /// Whether we've triggered the initial query
    pub query_triggered: bool,
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

/// Convert component name to SQL table name (lowercase with underscores)
fn component_to_table_name(full_component_name: &str) -> String {
    // Full component name is like "MfNavElodinEnumMessage.ypr_enu_2_body_deg"
    // Table name is like "mfnavelodinenumessage_ypr_enu_2_body_deg"
    full_component_name.to_lowercase().replace('.', "_")
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
        const LABEL_WIDTH: f32 = 150.0;
        const ROW_HEIGHT: f32 = 16.0;

        // Collect component info from EqlContext
        fn collect_components(
            parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
            result: &mut Vec<(ComponentId, String, String)>,
        ) {
            for (name, part) in parts {
                if let Some(component) = &part.component {
                    // Use the full component name (e.g., "MfNavElodinEnumMessage.ypr_enu_2_body_deg")
                    // to generate the table name (e.g., "mfnavelodinenumessage_ypr_enu_2_body_deg")
                    let table_name = component_to_table_name(&component.name);
                    result.push((part.id, name.clone(), table_name));
                }
                collect_components(&part.children, result);
            }
        }
        
        let mut component_list: Vec<(ComponentId, String, String)> = Vec::new();
        collect_components(&params.eql_context.0.component_parts, &mut component_list);
        
        // Trigger query for timestamp ranges if not started
        if !pane.query_triggered && !component_list.is_empty() {
            if matches!(params.time_ranges.state, TimeRangeQueryState::NotStarted) {
                params.time_ranges.state = TimeRangeQueryState::Querying(Instant::now());
                params.time_ranges.total_queries = component_list.len();
                params.time_ranges.pending_queries = component_list.len();
                
                // Query each table individually to handle missing tables gracefully
                for (_, _, table_name) in component_list.iter() {
                    let table_name_clone = table_name.clone();
                    let query = format!(
                        "SELECT min(time) as min_time, max(time) as max_time FROM {}",
                        table_name
                    );
                    
                    params.commands.send_req_reply(
                        SQLQuery(query),
                        move |In(res): In<Result<ArrowIPC<'static>, ErrorResponse>>,
                              mut time_ranges: ResMut<ComponentTimeRanges>| {
                            // Decrement pending count
                            time_ranges.pending_queries = time_ranges.pending_queries.saturating_sub(1);
                            
                            match res {
                                Ok(ipc) => {
                                    // Decode the Arrow IPC batch
                                    let mut decoder = arrow::ipc::reader::StreamDecoder::new();
                                    if let Some(batch_data) = ipc.batch {
                                        let mut buffer = arrow::buffer::Buffer::from(batch_data.into_owned());
                                        if let Some(batch) = decoder.decode(&mut buffer).ok().and_then(|b| b) {
                                            process_single_table_result(&table_name_clone, &batch, &mut time_ranges.ranges);
                                        }
                                    }
                                }
                                Err(_err) => {
                                    // Silently skip missing tables - this is expected for some components
                                }
                            }
                            
                            // Mark ready when all queries complete
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
        for (index, (component_id, label, table_name)) in component_list.iter().enumerate() {
            let color = row_color(index);
            let timestamp_range = params.time_ranges.ranges.get(table_name).copied();
            
            summaries.push(ComponentTimestampSummary {
                component_id: *component_id,
                label: label.clone(),
                table_name: table_name.clone(),
                color,
                timestamp_range,
            });
        }

        // Calculate time range - use selected_range to match the window timeline
        let time_range = &params.selected_range.0;
        let time_span = (time_range.end.0 - time_range.start.0).max(1) as f64;
        let timeline_width = (available_rect.width() - LABEL_WIDTH).max(100.0);
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
                if label.len() > 18 {
                    label.truncate(15);
                    label.push_str("...");
                }
                
                ui.painter().text(
                    Pos2::new(label_rect.min.x + 16.0, label_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(10.0),
                    summary.color,
                );

                // Draw bar if we have timestamp data
                if let Some((start_ts, end_ts)) = summary.timestamp_range {
                    // Calculate bar position relative to the selected time range
                    let start_offset = (start_ts.0 - time_range.start.0) as f64;
                    let end_offset = (end_ts.0 - time_range.start.0) as f64;
                    
                    let start_x = timeline_start_x + (start_offset * pixels_per_us) as f32;
                    let end_x = timeline_start_x + (end_offset * pixels_per_us) as f32;
                    
                    // Clip to the visible timeline area
                    let clipped_start_x = start_x.max(timeline_start_x);
                    let clipped_end_x = end_x.min(timeline_end_x);
                    
                    // Only draw if there's a visible portion
                    if clipped_end_x > clipped_start_x {
                        // Ensure minimum width for visibility
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
        
        if response.dragged() {
            let delta = response.drag_delta();
            pane.scroll_offset = (pane.scroll_offset - delta.y).max(0.0);
        }

        // Zoom with scroll wheel
        if response.hovered() {
            let scroll_delta = ui.input(|i| i.raw_scroll_delta);
            if scroll_delta.y != 0.0 {
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
