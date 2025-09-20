use anyhow::Result;
use colored::*;
use impeller2::com_de::Decomponentize;
use impeller2::registry::HashMapRegistry;
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, OwnedTable, Timestamp};
use impeller2_stellar::{Client, SubStream};
use impeller2_wkt::{StreamReply, VTableMsg};
use stellarator::buf::Slice;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

use crate::discovery::DiscoveredComponent;

/// Stores telemetry values for display
#[derive(Debug, Clone)]
struct TelemetryValue {
    name: String,
    values: Vec<f64>,
    unit: String,
}

/// Processes incoming telemetry packets
pub struct TelemetryProcessor {
    components: HashMap<ComponentId, DiscoveredComponent>,
    packet_count: usize,
    last_display_time: Instant,
    latest_values: HashMap<ComponentId, TelemetryValue>,
    last_timestamp: Option<Timestamp>,
    /// Registry of VTables for decomponentizing packets
    vtable_registry: HashMapRegistry,
}

impl TelemetryProcessor {
    pub fn new(components: HashMap<ComponentId, DiscoveredComponent>) -> Self {
        Self {
            components,
            packet_count: 0,
            last_display_time: Instant::now(),
            latest_values: HashMap::new(),
            last_timestamp: None,
            vtable_registry: HashMapRegistry::default(),
        }
    }
    
    /// Process incoming packets from a subscription stream
    pub async fn process_sub_stream(&mut self, stream: &mut SubStream<'_, StreamReply<Slice<Vec<u8>>>>) -> Result<()> {
        println!("\n📡 Waiting for telemetry data...\n");
        println!("💡 {} to exit", "Press Ctrl+C".bright_black());
        
        // Clear screen once at the beginning for the dashboard
        let mut first_packet = true;
        
        loop {
            // Wait for packets (blocking)
            match stream.next().await {
                Ok(reply) => {
                    match reply {
                        StreamReply::Table(table) => {
                            self.handle_table(table)?;
                            
                            // Clear screen only on first packet, then just update
                            if first_packet {
                                print!("\x1B[2J");  // Clear screen once
                                first_packet = false;
                            }
                            self.display_telemetry();
                        }
                        StreamReply::VTable(vtable_msg) => {
                            self.handle_vtable(vtable_msg);
                        }
                    }
                }
                Err(e) => {
                    println!("\n❌ Error receiving packet: {}", e);
                    break;
                }
            }
        }
        
        if self.packet_count > 0 {
            println!("\n✅ Stream processing complete. Total packets: {}", 
                self.packet_count.to_string().green());
        }
        
        Ok(())
    }
    
    /// Process incoming packets from the stream (alternative method)
    pub async fn process_stream(&mut self, _client: &mut Client) -> Result<()> {
        // This method could be used if we had direct access to client.rx
        println!("\n📡 This method would process packets directly from client.rx");
        Ok(())
    }
    
    /// Handle a VTable message
    fn handle_vtable(&mut self, vtable_msg: VTableMsg) {
        debug!("Received VTable with ID: {:?}", vtable_msg.id);
        info!("Storing VTable for packet ID: {:?}", vtable_msg.id);
        
        // Store the VTable in our registry
        self.vtable_registry.map.insert(vtable_msg.id, vtable_msg.vtable);
    }
    
    /// Handle a table packet  
    fn handle_table(&mut self, table: OwnedTable<Slice<Vec<u8>>>) -> Result<()> {
        self.packet_count += 1;
        
        // Create extractor for this packet
        let mut extractor = TelemetryExtractor::new(&self.components, &mut self.latest_values);
        
        // Use the VTable registry to decomponentize the table
        match table.sink(&self.vtable_registry, &mut extractor) {
            Ok(Ok(())) => {
                // info!("Successfully extracted real data from table packet #{}", self.packet_count);
                
                // Update timestamp from latest values if available
                if let Some(_first_value) = self.latest_values.values().next() {
                    self.last_timestamp = Some(Timestamp(self.packet_count as i64 * 1000));
                }
            }
            Ok(Err(e)) => {
                debug!("Error in extractor: {:?}", e);
                // Don't update values if extraction fails
            }
            Err(e) => {
                if self.packet_count == 1 {
                    info!("Waiting for VTable definitions...");
                }
                debug!("VTable not found for packet ID {:?}: {}", table.id, e);
                // Don't update values if VTable not found
            }
        }
        
        debug!("Processed table packet #{} (ID: {:?})", self.packet_count, table.id);
        
        Ok(())
    }
    
    /// Display telemetry in a beautiful terminal format
    fn display_telemetry(&self) {
        // Move cursor to top-left without clearing (reduces flicker)
        print!("\x1B[1;1H");
        
        // Header
        println!("{}", "╔═══════════════════════════════════════════════════════════════════════════════╗".bright_blue());
        println!("{}  {}  {}", "║".bright_blue(), 
            "🚀 ROCKET TELEMETRY DASHBOARD - RAW VALUES".bright_white().bold(), 
            "║".bright_blue());
        println!("{}\n", "╚═══════════════════════════════════════════════════════════════════════════════╝".bright_blue());
        
        // Status bar
        println!("📡 {} | 📦 {} | ⏱️  {}", 
            "Connected".green().bold(),
            format!("Packets: {}", self.packet_count).cyan(),
            self.last_timestamp.map_or("--".to_string(), |t| format!("T: {}", t.0)).yellow()
        );
        println!("{}", "─".repeat(80).bright_black());
        
        // Group components by category
        let mut categories: HashMap<&str, Vec<(&ComponentId, &TelemetryValue)>> = HashMap::new();
        
        for (id, value) in &self.latest_values {
            let category = categorize_component(&value.name);
            categories.entry(category).or_insert_with(Vec::new).push((id, value));
        }
        
        // Display each category
        let category_order = ["🔥 Propulsion", "🎯 Control", "💨 Aerodynamics", "📍 Position/Motion", "📊 Other"];
        for category_name in &category_order {
            if let Some(mut components) = categories.remove(category_name) {
                println!("\n{}", category_name.bright_white().bold());
                println!("{}", "═".repeat(80).bright_black());
                
                // Sort components by name for consistent display
                components.sort_by_key(|(_, v)| &v.name);
                
                for (_, value) in components {
                    self.display_component(value);
                }
            }
        }
        
        // Footer
        println!("\n{}", "─".repeat(80).bright_black());
        println!("💡 {} to exit", "Press Ctrl+C".bright_black());
    }
    
    /// Display a single component's values  
    fn display_component(&self, value: &TelemetryValue) {
        let name = value.name.replace("rocket.", "").replace("Globals.", "");
        let formatted_name = format!("{:28}", name).bright_cyan();
        
        // Check if this is a buffer component and handle specially
        let formatted_values = if name.to_lowercase().contains("buffer") {
            format_buffer_values(&value.values)
        } else {
            format_values(&value.values)
        };
        
        let unit = if !value.unit.is_empty() {
            format!(" ({})", value.unit).bright_black().to_string()
        } else {
            String::new()
        };
        
        println!("  {}: {}{}", formatted_name, formatted_values, unit);
    }
    
    /// Handle a single packet (unused in current implementation)
    fn handle_packet(&mut self, packet: OwnedPacket<Slice<Vec<u8>>>) -> Result<()> {
        self.packet_count += 1;
        
        match packet {
            OwnedPacket::Table(table) => {
                self.handle_table(table)?;
            }
            OwnedPacket::TimeSeries(ts) => {
                self.process_time_series(ts)?;
            }
            OwnedPacket::Msg(_msg) => {
                debug!("Received message packet");
            }
        }
        
        Ok(())
    }
    
    /// Process time series data
    fn process_time_series(&mut self, ts: impeller2::types::OwnedTimeSeries<Slice<Vec<u8>>>) -> Result<()> {
        let timestamps = ts.timestamps()?;
        debug!("Received time series with {} timestamps", timestamps.len());
        Ok(())
    }
}

/// Helper to extract telemetry values from packets
struct TelemetryExtractor<'a> {
    components: &'a HashMap<ComponentId, DiscoveredComponent>,
    latest_values: &'a mut HashMap<ComponentId, TelemetryValue>,
}

impl<'a> TelemetryExtractor<'a> {
    fn new(
        components: &'a HashMap<ComponentId, DiscoveredComponent>,
        latest_values: &'a mut HashMap<ComponentId, TelemetryValue>,
    ) -> Self {
        Self {
            components,
            latest_values,
        }
    }
}

impl<'a> Decomponentize for TelemetryExtractor<'a> {
    type Error = anyhow::Error;
    
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        // Find the component definition
        if let Some(component) = self.components.get(&component_id) {
            // Extract values based on the component view type
            let values = match value {
                ComponentView::F64(array) => {
                    array.buf().to_vec()
                }
                ComponentView::F32(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::U64(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::I64(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::U32(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::I32(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::U16(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::I16(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::U8(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::I8(array) => {
                    array.buf().iter().map(|&v| v as f64).collect()
                }
                ComponentView::Bool(array) => {
                    array.buf().iter().map(|&v| if v { 1.0 } else { 0.0 }).collect()
                }
            };
            
            // Extract unit from metadata
            let unit = component.metadata.get("unit")
                .cloned()
                .unwrap_or_default();
            
            // Store the latest values
            self.latest_values.insert(component_id, TelemetryValue {
                name: component.name.clone(),
                values,
                unit,
            });
        } else {
            debug!("Unknown component ID: {:?}", component_id);
        }
        
        Ok(())
    }
}

/// Categorize component by name
fn categorize_component(name: &str) -> &'static str {
    if name.contains("motor") || name.contains("thrust") {
        "🔥 Propulsion"
    } else if name.contains("fin") || name.contains("pid") || name.contains("control") || name.contains("setpoint") {
        "🎯 Control"
    } else if name.contains("mach") || name.contains("pressure") || name.contains("angle") || name.contains("aero") || name.contains("wind") {
        "💨 Aerodynamics"
    } else if name.contains("world_pos") || name.contains("world_vel") || name.contains("accel") || name.contains("gravity") {
        "📍 Position/Motion"
    } else {
        "📊 Other"
    }
}

/// Format values for display - show all values
fn format_values(values: &[f64]) -> String {
    if values.is_empty() {
        "no data".bright_black().to_string()
    } else if values.len() == 1 {
        // Single value
        format_float(values[0])
    } else if values.len() > 20 {
        // Very large arrays might be internal buffers - show summary
        format!("[{} values]", values.len()).bright_black().to_string()
    } else {
        // Show all values in the array
        let vals: Vec<String> = values.iter()
            .map(|&v| format_float(v))
            .collect();
        
        // Format based on length for better display
        if values.len() <= 4 {
            format!("[{}]", vals.join(", "))
        } else {
            // For longer arrays, show on multiple lines if needed
            format!("[{}]", vals.join(", "))
        }
    }
}

/// Format buffer values - show concise representation
fn format_buffer_values(values: &[f64]) -> String {
    if values.is_empty() {
        "[empty buffer]".bright_black().to_string()
    } else {
        format!("[buffer: {} values]", values.len()).bright_black().to_string()
    }
}

/// Format a single float value - just show the raw value
fn format_float(value: f64) -> String {
    // Show raw values with appropriate precision
    if value.abs() < 0.0001 {
        format!("{:8.4}", value)
    } else if value.abs() < 1.0 {
        format!("{:8.4}", value)
    } else if value.abs() < 1000.0 {
        format!("{:8.2}", value)
    } else {
        format!("{:8.2e}", value)
    }
}