use anyhow::{Context, Result};
use impeller2_stellar::Client;
use impeller2_wkt::{Stream, StreamBehavior};
use tracing::info;

use crate::discovery::discover_components;
use crate::processor::TelemetryProcessor;
use crate::tui::run_tui;

/// Run the client with the TUI interface
pub async fn run_with_tui(client: &mut Client) -> Result<()> {
    info!("Starting Elodin-DB client with TUI");
    
    // Discover what components are available in the database
    info!("Discovering registered components...");
    let components = discover_components(client).await?;
    
    // Log summary to tracing (not to stdout since we're using TUI)
    if !components.is_empty() {
        info!("Discovered {} components", components.len());
        for component in components.values() {
            info!("  {} -> {:?}", component.name, component.schema);
        }
    }
    
    if components.is_empty() {
        eprintln!("⚠️  No components found in database!");
        eprintln!("     Make sure rocket.py or another simulation is running first.");
        return Ok(());
    }
    
    // Create a channel for telemetry data
    let (sender, receiver) = async_channel::bounded(1000);
    
    // Subscribe to real-time stream
    let stream = Stream {
        behavior: StreamBehavior::RealTime,
        id: 1,
    };
    
    // Create subscription stream
    let mut sub_stream = client.stream(&stream)
        .await
        .context("Failed to setup stream subscription")?;
    
    // Create processor with discovered components
    let mut processor = TelemetryProcessor::new(components);
    processor.set_telemetry_sender(sender.clone());
    
    // Run TUI and processor concurrently
    let tui_future = run_tui(receiver);
    let processor_future = processor.process_sub_stream(&mut sub_stream);
    
    // Race both futures - when either completes, we're done
    use futures_lite::future;
    
    match future::race(tui_future, processor_future).await {
        Ok(_) => info!("TUI or processor exited normally"),
        Err(e) => info!("Error: {}", e),
    }
    
    info!("Client session complete");
    Ok(())
}