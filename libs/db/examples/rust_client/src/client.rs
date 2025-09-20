use anyhow::{Context, Result};
use colored::*;
use impeller2_stellar::Client;
use impeller2_wkt::{Stream, StreamBehavior};
use tracing::info;

use crate::discovery::{discover_components, display_rocket_summary};
use crate::processor::TelemetryProcessor;

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
    
    // Use the stream() method to create a subscription that we can continuously poll
    let mut sub_stream = client.stream(&stream)
        .await
        .context("Failed to setup stream subscription")?;
        
    println!("  {} Real-time stream subscription active", "✓".green());
    println!("\n✨ Setup complete! Listening for telemetry data...");
    println!("  (Press Ctrl+C to stop)\n");
    
    if !components.is_empty() {
        // Create processor with discovered components
        let mut processor = TelemetryProcessor::new(components);
        
        // Process incoming packets from the subscription
        processor.process_sub_stream(&mut sub_stream).await?;
    } else {
        // Even with no components, listen for packets
        println!("⚠️  No components discovered. Waiting for any data...");
        loop {
            match sub_stream.next().await {
                Ok(_reply) => {
                    println!("Received data packet!");
                    // Break after receiving first packet to show it's working
                    break;
                }
                Err(e) => {
                    println!("Error: {}", e);
                    break;
                }
            }
        }
    }
    
    info!("Client session complete");
    
    Ok(())
}