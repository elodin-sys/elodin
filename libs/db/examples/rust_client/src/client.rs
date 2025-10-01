use anyhow::{Context, Result};
use impeller2_stellar::Client;
use impeller2_wkt::{Stream, StreamBehavior};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;
use tracing::{info, warn};

use crate::control::run_control_loop;
use crate::discovery::discover_components;
use crate::processor::TelemetryProcessor;
use crate::tui::{TelemetryRow, run_tui};

/// Run the client with connection resilience and TUI
pub async fn run_resilient(addr: SocketAddr) -> Result<()> {
    info!("Starting resilient Elodin-DB client for {}", addr);

    // Create channel for telemetry data
    let (sender, receiver) = async_channel::bounded(1000);

    // Send initial "waiting for connection" message
    let _ = sender.send(TelemetryRow::waiting_for_connection()).await;

    // Run the TUI and connection handler concurrently
    let tui_future = run_tui(receiver.clone());
    let connection_future = connection_loop(addr, sender);

    // Race both futures - when TUI exits (user presses 'q'), we're done
    use futures_lite::future;
    match future::race(tui_future, connection_future).await {
        Ok(_) => info!("Client exited normally"),
        Err(e) => warn!("Client error: {}", e),
    }

    info!("Client shutdown complete");
    Ok(())
}

/// Connection loop that handles retries and reconnections
async fn connection_loop(
    addr: SocketAddr,
    sender: async_channel::Sender<TelemetryRow>,
) -> Result<()> {
    let mut retry_count = 0;
    loop {
        retry_count += 1;
        info!("Connection attempt #{}", retry_count);

        match connect_and_stream(addr, sender.clone()).await {
            Ok(_) => {
                info!("Stream ended normally, will reconnect...");
            }
            Err(e) => {
                warn!("Connection/stream error: {}, will retry...", e);
            }
        }

        // Send "waiting for reconnection" message
        let _ = sender.send(TelemetryRow::waiting_for_reconnection()).await;

        // Wait before retrying
        stellarator::sleep(Duration::from_secs(2)).await;
    }
}

/// Connect to database and stream telemetry
async fn connect_and_stream(
    addr: SocketAddr,
    sender: async_channel::Sender<TelemetryRow>,
) -> Result<()> {
    // Try to connect with retries - we need two clients
    let mut attempt = 0;
    let (mut telemetry_client, mut control_client) = loop {
        attempt += 1;
        match futures_lite::future::try_zip(Client::connect(addr), Client::connect(addr)).await {
            Ok((client1, client2)) => {
                info!("Connected to database at {} (attempt #{})", addr, attempt);
                // Send "connected" indicator
                let _ = sender.send(TelemetryRow::connected()).await;
                break (client1, client2);
            }
            Err(e) => {
                if attempt == 1 {
                    info!("Waiting for database at {} - {}", addr, e);
                } else if attempt % 10 == 0 {
                    info!(
                        "Still waiting for database at {} (attempt #{}) - {}",
                        addr, attempt, e
                    );
                }
                stellarator::sleep(Duration::from_secs(1)).await;
            }
        }
    };

    // Run telemetry streaming and control loop concurrently
    let telemetry_future = stream_telemetry(&mut telemetry_client, sender);
    let control_future = run_control_loop(&mut control_client);

    // Race both futures - if either fails, we stop both
    use futures_lite::future;
    match future::race(telemetry_future, control_future).await {
        Ok(_) => {
            info!("Streaming completed normally");
            Ok(())
        }
        Err(e) => {
            warn!("Streaming/control error: {}", e);
            Err(e)
        }
    }
}

/// Stream telemetry from the connected client
async fn stream_telemetry(
    client: &mut Client,
    sender: async_channel::Sender<TelemetryRow>,
) -> Result<()> {
    info!("Starting telemetry stream");

    // Try to discover components, but don't fail if none are found
    info!("Discovering registered components...");
    let components = match discover_components(client).await {
        Ok(components) => {
            if !components.is_empty() {
                info!("Discovered {} components", components.len());
                for component in components.values() {
                    info!("  {} -> {:?}", component.name, component.schema);
                }
            } else {
                info!("No components found yet, waiting for simulation data...");
            }
            components
        }
        Err(e) => {
            info!(
                "Component discovery failed ({}), starting with empty component list",
                e
            );
            HashMap::new()
        }
    };

    // Subscribe to real-time stream
    let stream = Stream {
        behavior: StreamBehavior::RealTime,
        id: 1,
    };

    // Create subscription stream
    let mut sub_stream = client
        .stream(&stream)
        .await
        .context("Failed to setup stream subscription")?;

    // Create processor with discovered components (may be empty)
    let mut processor = TelemetryProcessor::new(components);
    processor.set_telemetry_sender(sender.clone());

    // Process the stream
    match processor.process_sub_stream(&mut sub_stream).await {
        Ok(_) => {
            info!("Stream processing completed normally");
            Ok(())
        }
        Err(e) => {
            warn!("Stream processing error: {}", e);
            Err(e)
        }
    }
}
