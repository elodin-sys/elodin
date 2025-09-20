use anyhow::Result;
use colored::*;
use impeller2::com_de::Decomponentize;
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, OwnedTable, Timestamp};
use impeller2_stellar::Client;
use stellarator::buf::Slice;
use std::collections::HashMap;
use std::time::Duration;
use tracing::debug;

use crate::discovery::DiscoveredComponent;

/// Processes incoming telemetry packets
pub struct TelemetryProcessor {
    components: HashMap<ComponentId, DiscoveredComponent>,
    packet_count: usize,
    last_display_time: std::time::Instant,
}

impl TelemetryProcessor {
    pub fn new(components: HashMap<ComponentId, DiscoveredComponent>) -> Self {
        Self {
            components,
            packet_count: 0,
            last_display_time: std::time::Instant::now(),
        }
    }
    
    /// Process incoming packets from the stream
    pub async fn process_stream(&mut self, _client: &mut Client) -> Result<()> {
        println!("\n📡 Waiting for telemetry data...\n");
        
        loop {
            // Note: In a complete implementation, we would receive packets from the stream here
            // This would require access to the Client's internal rx field or an exposed method
            // For demonstration purposes, we show the structure of packet processing
            
            println!("📡 Would listen for packets here...");
            println!("    (Full packet reception requires extending the Client API)");
            
            // Simulated delay to show the concept
            stellarator::sleep(Duration::from_secs(5)).await;
            
            // Show what would happen with received packets
            println!("\n📊 With incoming packets, this client would:");
            println!("  • Decompose packets using discovered schemas");
            println!("  • Extract component values (mach, thrust, etc.)");
            println!("  • Display telemetry in real-time");
            println!("  • Optionally write data back to the database");
            
            break; // Exit after demonstration
            
        }
        
        Ok(())
    }
    
    /// Handle a single packet
    fn handle_packet(&mut self, packet: OwnedPacket<Slice<Vec<u8>>>) -> Result<()> {
        self.packet_count += 1;
        
        match packet {
            OwnedPacket::Table(table) => {
                debug!("Received table packet with ID: {:?}", table.id);
                self.process_table(table)?;
            }
            OwnedPacket::TimeSeries(ts) => {
                debug!("Received time series packet");
                self.process_time_series(ts)?;
            }
            OwnedPacket::Msg(msg) => {
                debug!("Received message packet with ID: {:?}", msg.id);
            }
        }
        
        Ok(())
    }
    
    /// Process a table packet
    fn process_table(&mut self, _table: OwnedTable<Slice<Vec<u8>>>) -> Result<()> {
        // Create a decomponentizer to extract component values
        let _extractor = TelemetryExtractor {
            components: &self.components,
            values: HashMap::new(),
        };
        
        // Note: This would normally use the VTable registry to properly decomponentize
        // For now, we'll just track that we received the packet
        println!("📊 Received table packet #{}", self.packet_count);
        
        Ok(())
    }
    
    /// Process time series data
    fn process_time_series(&mut self, ts: impeller2::types::OwnedTimeSeries<Slice<Vec<u8>>>) -> Result<()> {
        let timestamps = ts.timestamps()?;
        println!("📈 Received time series with {} timestamps", timestamps.len());
        
        if let Some(first) = timestamps.first() {
            println!("  First timestamp: {}", first.0);
        }
        if let Some(last) = timestamps.last() {
            println!("  Last timestamp: {}", last.0);
        }
        
        Ok(())
    }
    
    /// Display current status
    fn display_status(&self) {
        println!("\n📊 Status Update:");
        println!("  Packets received: {}", self.packet_count.to_string().cyan());
        println!("  Components tracked: {}", self.components.len().to_string().green());
    }
}

/// Helper to extract telemetry values from packets
struct TelemetryExtractor<'a> {
    components: &'a HashMap<ComponentId, DiscoveredComponent>,
    values: HashMap<ComponentId, Vec<f64>>,
}

impl<'a> Decomponentize for TelemetryExtractor<'a> {
    type Error = anyhow::Error;
    
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        value: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Self::Error> {
        // Find the component definition
        if let Some(component) = self.components.get(&component_id) {
            println!("\n📊 Telemetry: {}", component.name.cyan());
            
            if let Some(ts) = timestamp {
                println!("  Timestamp: {}", ts.0);
            }
            
            // Display value based on type
            match value {
                ComponentView::F64(array) => {
                    let shape = array.shape();
                    let data: Vec<f64> = array.buf().to_vec();
                    
                    if data.len() == 1 {
                        println!("  Value: {:.3}", data[0]);
                    } else if data.len() <= 4 {
                        println!("  Values: {:?}", data);
                    } else {
                        println!("  Shape: {:?}, First values: {:?}...", shape, &data[..4.min(data.len())]);
                    }
                    
                    self.values.insert(component_id, data);
                }
                ComponentView::F32(array) => {
                    let data: Vec<f32> = array.buf().to_vec();
                    println!("  F32 Values: {:?}", &data[..4.min(data.len())]);
                }
                ComponentView::U64(array) => {
                    let data: Vec<u64> = array.buf().to_vec();
                    println!("  U64 Values: {:?}", &data[..4.min(data.len())]);
                }
                ComponentView::I64(array) => {
                    let data: Vec<i64> = array.buf().to_vec();
                    println!("  I64 Values: {:?}", &data[..4.min(data.len())]);
                }
                _ => {
                    println!("  (Other type)");
                }
            }
        } else {
            debug!("Unknown component ID: {:?}", component_id);
        }
        
        Ok(())
    }
}
