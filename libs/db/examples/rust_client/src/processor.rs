use anyhow::Result;
use colored::*;
use impeller2::com_de::Decomponentize;
use impeller2::types::{ComponentId, ComponentView, OwnedPacket, OwnedTable, Timestamp};
use impeller2_stellar::{Client, SubStream};
use impeller2_wkt::{StreamReply, VTableMsg};
use stellarator::buf::Slice;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};

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
    
    /// Process incoming packets from a subscription stream
    pub async fn process_sub_stream(&mut self, stream: &mut SubStream<'_, StreamReply<Slice<Vec<u8>>>>) -> Result<()> {
        println!("\n📡 Waiting for telemetry data...\n");
        
        loop {
            // Wait for packets (blocking)
            match stream.next().await {
                Ok(reply) => {
                    match reply {
                        StreamReply::Table(table) => {
                            self.handle_table(table)?;
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
            
            // Display status periodically
            if self.packet_count > 0 && self.packet_count % 100 == 0 {
                self.display_status();
            }
        }
        
        if self.packet_count > 0 {
            println!("\n✅ Stream processing complete. Total packets received: {}", 
                self.packet_count.to_string().green());
        }
        
        Ok(())
    }
    
    /// Process incoming packets from the stream (alternative method)
    pub async fn process_stream(&mut self, _client: &mut Client) -> Result<()> {
        // This method could be used if we had direct access to client.rx
        // For now, we use process_sub_stream instead
        println!("\n📡 This method would process packets directly from client.rx");
        Ok(())
    }
    
    /// Handle a VTable message
    fn handle_vtable(&mut self, vtable: VTableMsg) {
        info!("Received VTable with ID: {:?}", vtable.id);
        println!("🎯 VTable received (ID: {:?}) - Structure for decoding telemetry", vtable.id);
    }
    
    /// Handle a table packet  
    fn handle_table(&mut self, table: OwnedTable<Slice<Vec<u8>>>) -> Result<()> {
        self.packet_count += 1;
        
        if self.packet_count == 1 {
            println!("🎉 First telemetry packet received!");
            println!("  Table ID: {:?}", table.id);
            println!("  Data size: {} bytes", table.buf.len());
        }
        
        // Process the table using decomponentize
        // Note: This would normally use the VTable to properly extract components
        self.process_table(table)?;
        
        // Display periodic status
        let now = std::time::Instant::now();
        if now.duration_since(self.last_display_time) > Duration::from_secs(5) {
            self.display_status();
            self.last_display_time = now;
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
