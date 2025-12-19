//! Data Overview Panel
//!
//! Provides a high-level timeline view showing all database components and their
//! data density across time. Useful for identifying gaps or disparities in data.

use bevy::{
    ecs::system::{SystemParam, SystemState},
    prelude::*,
};
use bevy_egui::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use impeller2::types::{ComponentId, Timestamp};
use impeller2_wkt::{EarliestTimestamp, LastUpdated};

use crate::{EqlContext, SelectedTimeRange, ui::widgets::WidgetSystem};

use super::{
    colors::get_scheme,
    plot::data::{CollectedGraphData, Line},
};

// Re-export for use in component collection
use eql;

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
}

/// Summary of a component's data presence
#[derive(Clone, Debug)]
pub struct ComponentTimestampSummary {
    pub component_id: ComponentId,
    pub label: String,
    pub color: Color32,
    pub timestamp_ranges: Vec<(Timestamp, Timestamp)>,
    pub point_count: usize,
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

/// Widget for rendering the Data Overview panel
#[derive(SystemParam)]
pub struct DataOverviewWidget<'w> {
    eql_context: Res<'w, EqlContext>,
    graph_data: Res<'w, CollectedGraphData>,
    lines: Res<'w, Assets<Line>>,
    selected_range: Res<'w, SelectedTimeRange>,
    earliest_timestamp: Res<'w, EarliestTimestamp>,
    last_updated: Res<'w, LastUpdated>,
}

impl WidgetSystem for DataOverviewWidget<'_> {
    type Args = DataOverviewPane;
    type Output = DataOverviewPane;

    fn ui_system(
        world: &mut World,
        state: &mut SystemState<Self>,
        ui: &mut egui::Ui,
        mut pane: Self::Args,
    ) -> Self::Output {
        let params = state.get(world);
        
        let scheme = get_scheme();
        let available_rect = ui.available_rect_before_wrap();
        pane.rect = Some(available_rect);

        // Constants for layout
        const LABEL_WIDTH: f32 = 150.0;
        const ROW_HEIGHT: f32 = 16.0;

        // Collect component summaries from EqlContext (all available components)
        let mut summaries: Vec<ComponentTimestampSummary> = Vec::new();
        
        // Recursively collect all leaf components from the component_parts tree
        fn collect_components(
            parts: &std::collections::BTreeMap<String, eql::ComponentPart>,
            summaries: &mut Vec<(ComponentId, String)>,
        ) {
            for (name, part) in parts {
                if part.component.is_some() {
                    // This is a leaf component
                    summaries.push((part.id, name.clone()));
                }
                // Recurse into children
                collect_components(&part.children, summaries);
            }
        }
        
        let mut component_list: Vec<(ComponentId, String)> = Vec::new();
        collect_components(&params.eql_context.0.component_parts, &mut component_list);
        
        // Get the database time range
        let db_earliest = params.earliest_timestamp.0;
        let db_latest = params.last_updated.0;
        let has_db_data = db_latest.0 > db_earliest.0;
        
        for (index, (component_id, label)) in component_list.iter().enumerate() {
            let color = row_color(index);
            
            // Try to get detailed timestamp data from CollectedGraphData if available
            let mut timestamp_ranges = Vec::new();
            let mut point_count = 0;
            
            if let Some(plot_data) = params.graph_data.components.get(component_id) {
                if let Some((_, line_handle)) = plot_data.lines.first_key_value() {
                    if let Some(line) = params.lines.get(line_handle) {
                        // Get summary from the line's data tree
                        let summary = line.data.range_summary(params.selected_range.0.clone());
                        if summary.len > 0 {
                            timestamp_ranges.push((summary.start_timestamp, summary.end_timestamp));
                            point_count = summary.len;
                        }
                    }
                }
            }
            
            // If no detailed data but database has data, show the full db range
            // This indicates the component exists in the database
            if timestamp_ranges.is_empty() && has_db_data {
                timestamp_ranges.push((db_earliest, db_latest));
            }

            summaries.push(ComponentTimestampSummary {
                component_id: *component_id,
                label: label.clone(),
                color,
                timestamp_ranges,
                point_count,
            });
        }

        // Calculate time range
        let time_range = &params.selected_range.0;
        let time_span = (time_range.end.0 - time_range.start.0).max(1) as f64;
        let timeline_width = (available_rect.width() - LABEL_WIDTH).max(100.0) as f64;
        let pixels_per_us = timeline_width / time_span;

        // Draw header
        let header_rect = Rect::from_min_size(
            available_rect.min,
            Vec2::new(available_rect.width(), ROW_HEIGHT + 4.0),
        );
        ui.painter().rect_filled(header_rect, 0.0, scheme.bg_secondary);
        
        // Header text
        ui.painter().text(
            Pos2::new(available_rect.min.x + 8.0, available_rect.min.y + 4.0),
            egui::Align2::LEFT_TOP,
            format!("Components: {}", summaries.len()),
            egui::FontId::proportional(11.0),
            scheme.text_primary,
        );

        // Timeline header with time markers
        let timeline_start_x = available_rect.min.x + LABEL_WIDTH;
        let num_markers = 5;
        for i in 0..=num_markers {
            let t = i as f64 / num_markers as f64;
            let x = timeline_start_x + (t * timeline_width as f64) as f32;
            let timestamp_us = time_range.start.0 + (t * time_span) as i64;
            let timestamp_s = timestamp_us as f64 / 1_000_000.0;
            
            ui.painter().text(
                Pos2::new(x, available_rect.min.y + 4.0),
                egui::Align2::CENTER_TOP,
                format!("{:.2}s", timestamp_s),
                egui::FontId::proportional(9.0),
                scheme.text_secondary,
            );
        }

        // Draw component rows
        let content_start_y = available_rect.min.y + ROW_HEIGHT + 8.0;
        let content_height = available_rect.height() - ROW_HEIGHT - 8.0;
        
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

                // Draw a simple presence bar for each timestamp range
                for (start, end) in &summary.timestamp_ranges {
                    let start_x = timeline_start_x 
                        + ((start.0 - time_range.start.0) as f64 * pixels_per_us) as f32;
                    let end_x = timeline_start_x 
                        + ((end.0 - time_range.start.0) as f64 * pixels_per_us) as f32;
                    
                    let bar_rect = Rect::from_min_max(
                        Pos2::new(start_x.max(timeline_start_x), row_rect.min.y + 4.0),
                        Pos2::new(end_x.min(timeline_start_x + timeline_width as f32), row_rect.max.y - 4.0),
                    );
                    
                    if bar_rect.width() > 0.0 {
                        ui.painter().rect_filled(
                            bar_rect,
                            2.0,
                            summary.color,
                        );
                    }
                }

                // Draw point count indicator
                if summary.point_count > 0 {
                    ui.painter().text(
                        Pos2::new(available_rect.max.x - 8.0, row_rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        format!("{}", summary.point_count),
                        egui::FontId::proportional(9.0),
                        scheme.text_secondary,
                    );
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
