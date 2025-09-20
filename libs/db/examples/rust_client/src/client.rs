use anyhow::{Context, Result};
use colored::*;
use impeller2::types::PacketId;
use impeller2_stellar::Client;
use impeller2_wkt::{Stream, StreamBehavior, VTableStream};
use tracing::info;

use crate::discovery::{discover_components, display_rocket_summary};

/// Demonstrates connection and dynamic discovery with the database
pub async fn demonstrate_connection(client: &mut Client) -> Result<()> {
    info!("Demonstrating Elodin-DB connectivity with dynamic discovery");
    
    // Discover what components are available in the database
    println!("\n🔍 Discovering registered components:");
    let components = discover_components(client).await?;
    
    // Display summary of rocket components
    display_rocket_summary(&components);
    
    if components.is_empty() {
        println!("\n⚠️  No components found in database!");
        println!("     Make sure rocket.py or another simulation is running first.");
        return Ok(());
    }
    
    // Subscribe to real-time stream
    println!("\n📡 Setting up real-time telemetry stream:");
    
    let stream = Stream {
        behavior: StreamBehavior::RealTime,
        id: 1,
    };
    
    client.send(&stream)
        .await
        .0
        .context("Failed to setup stream")?;
        
    println!("  {} Real-time stream configured", "✓".green());
    
    // Request a VTable stream for receiving structured data
    println!("\n📊 Requesting VTable stream:");
    
    let vtable_id: PacketId = [1, 0];
    let vtable_stream = VTableStream { id: vtable_id };
    
    // Start the stream subscription
    let _stream_handle = client.stream(&vtable_stream)
        .await
        .context("Failed to start VTable stream")?;
        
    println!("  {} VTable stream started with ID {:?}", "✓".green(), vtable_id);
    
    // Process incoming telemetry
    println!("\n✨ Setup complete! Ready to process telemetry.");
    
    if !components.is_empty() {
        // Create processor with discovered components
        let mut processor = crate::processor::TelemetryProcessor::new(components);
        
        // Process incoming packets
        processor.process_stream(client).await?;
    }
    
    info!("Client session complete");
    
    Ok(())
}