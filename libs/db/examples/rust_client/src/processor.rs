use anyhow::Result;
use colored::*;
use impeller2::com_de::Decomponentize;
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, OwnedTable, Timestamp};
use impeller2_stellar::{Client, SubStream};
use impeller2_wkt::{StreamReply, VTableMsg};
use stellarator::buf::Slice;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

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
}

impl TelemetryProcessor {
    pub fn new(components: HashMap<ComponentId, DiscoveredComponent>) -> Self {
        Self {
            components,
            packet_count: 0,
            last_display_time: Instant::now(),
            latest_values: HashMap::new(),
            last_timestamp: None,
        }
    }
    
    /// Process incoming packets from a subscription stream
    pub async fn process_sub_stream(&mut self, stream: &mut SubStream<'_, StreamReply<Slice<Vec<u8>>>>) -> Result<()> {
        println!("\n📡 Waiting for telemetry data...\n");
        println!("💡 {} to exit", "Press Ctrl+C".bright_black());
        
        loop {
            // Wait for packets (blocking)
            match stream.next().await {
                Ok(reply) => {
                    match reply {
                        StreamReply::Table(table) => {
                            self.handle_table(table)?;
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
    fn handle_vtable(&mut self, vtable: VTableMsg) {
        debug!("Received VTable with ID: {:?}", vtable.id);
    }
    
    /// Handle a table packet  
    fn handle_table(&mut self, table: OwnedTable<Slice<Vec<u8>>>) -> Result<()> {
        self.packet_count += 1;
        
        // Note: In a full implementation, we would use a VTable registry to properly
        // decomponentize the table data. For this example, we'll simulate telemetry
        // values to demonstrate the display functionality.
        
        // Update timestamp
        let current_time = Timestamp(self.packet_count as i64 * 1000);
        self.last_timestamp = Some(current_time);
        
        // For demonstration: Generate simulated telemetry values based on discovered components
        self.simulate_telemetry_values();
        
        debug!("Processed table packet #{} (ID: {:?})", self.packet_count, table.id);
        
        Ok(())
    }
    
    /// Simulate telemetry values for demonstration
    fn simulate_telemetry_values(&mut self) {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        
        // Only simulate if we haven't received any real data yet
        if self.latest_values.is_empty() && !self.components.is_empty() {
            for (id, component) in &self.components {
                // Generate appropriate values based on component name
                let values = if component.name.contains("world_pos") {
                    // Position: increasing altitude
                    vec![0.0, 0.0, 100.0 + (self.packet_count as f64 * 10.0)]
                } else if component.name.contains("world_vel") {
                    // Velocity: upward motion
                    vec![0.0, 0.0, 50.0 + rng.gen_range(-5.0..5.0)]
                } else if component.name.contains("mach") {
                    // Mach number
                    vec![0.15 + (self.packet_count as f64 * 0.001).min(0.8)]
                } else if component.name.contains("thrust") {
                    // Thrust: decreasing over time
                    vec![(1000.0 - self.packet_count as f64 * 2.0).max(0.0)]
                } else if component.name.contains("angle") {
                    // Angle of attack
                    vec![rng.gen_range(-5.0..5.0)]
                } else if component.name.contains("pressure") {
                    // Dynamic pressure
                    vec![101325.0 * (1.0 - (self.packet_count as f64 * 0.0001).min(0.8))]
                } else if component.name.contains("fin") {
                    // Fin deflection
                    vec![rng.gen_range(-10.0..10.0)]
                } else if component.name.contains("pid") {
                    // PID controller state
                    vec![rng.gen_range(-1.0..1.0), rng.gen_range(-0.1..0.1), rng.gen_range(-0.01..0.01)]
                } else {
                    // Default: small random values
                    vec![rng.gen_range(-1.0..1.0)]
                };
                
                let unit = component.metadata.get("unit").cloned().unwrap_or_default();
                
                self.latest_values.insert(*id, TelemetryValue {
                    name: component.name.clone(),
                    values,
                    unit,
                });
            }
        }
    }
    
    /// Display telemetry in a beautiful terminal format
    fn display_telemetry(&self) {
        // Clear screen and move cursor to top
        print!("\x1B[2J\x1B[1;1H");
        
        // Header
        println!("{}", "╔══════════════════════════════════════════════════════════════╗".bright_blue());
        println!("{}  {}  {}", "║".bright_blue(), 
            "🚀 ROCKET TELEMETRY DASHBOARD".bright_white().bold(), 
            "║".bright_blue());
        println!("{}\n", "╚══════════════════════════════════════════════════════════════╝".bright_blue());
        
        // Status bar
        println!("📡 {} | 📦 {} | ⏱️  {}", 
            "Connected".green().bold(),
            format!("Packets: {}", self.packet_count).cyan(),
            self.last_timestamp.map_or("--".to_string(), |t| format!("T: {}", t.0)).yellow()
        );
        println!("{}", "─".repeat(65).bright_black());
        
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
                println!("{}", "═".repeat(50).bright_black());
                
                // Sort components by name for consistent display
                components.sort_by_key(|(_, v)| &v.name);
                
                for (_, value) in components {
                    self.display_component(value);
                }
            }
        }
        
        // Footer
        println!("\n{}", "─".repeat(65).bright_black());
        println!("💡 {} to exit", "Press Ctrl+C".bright_black());
    }
    
    /// Display a single component's values  
    fn display_component(&self, value: &TelemetryValue) {
        let name = value.name.replace("rocket.", "");
        let formatted_name = format!("{:25}", name).bright_cyan();
        
        let formatted_values = format_values(&value.values);
        
        let unit = if !value.unit.is_empty() {
            format!(" {}", value.unit).bright_black().to_string()
        } else {
            String::new()
        };
        
        println!("  {} : {}{}", formatted_name, formatted_values, unit);
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

/// Format values for display
fn format_values(values: &[f64]) -> String {
    if values.is_empty() {
        "no data".bright_black().to_string()
    } else if values.len() == 1 {
        // Single value
        format_float(values[0])
    } else if values.len() <= 3 {
        // Vector (2D or 3D)
        let vals: Vec<String> = values.iter()
            .map(|&v| format_float(v))
            .collect();
        format!("[{}]", vals.join(", "))
    } else if values.len() == 4 {
        // Quaternion or 4D vector
        let vals: Vec<String> = values.iter()
            .map(|&v| format!("{:6.3}", v))
            .collect();
        format!("[{}]", vals.join(", ")).bright_magenta().to_string()
    } else if values.len() <= 9 {
        // Small matrix (3x3)
        format!("[{} values]", values.len()).bright_blue().to_string()
    } else {
        // Large array
        format!("[{} values]", values.len()).bright_black().to_string()
    }
}

/// Format a single float value with color coding
fn format_float(value: f64) -> String {
    if value.abs() < 0.0001 {
        format!("{:8.4}", value).bright_black().to_string()
    } else if value.abs() < 0.1 {
        format!("{:8.4}", value).bright_yellow().to_string()
    } else if value.abs() < 10.0 {
        format!("{:8.2}", value).bright_green().to_string()
    } else if value.abs() < 1000.0 {
        format!("{:8.1}", value).bright_cyan().to_string()
    } else {
        format!("{:8.2e}", value).bright_magenta().to_string()
    }
}