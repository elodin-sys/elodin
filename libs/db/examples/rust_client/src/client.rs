use anyhow::{Context, Result};
use colored::*;
use impeller2::types::{ComponentId, IntoLenPacket, PacketId};
use impeller2_stellar::Client;
use impeller2_wkt::{SetComponentMetadata, Stream, StreamBehavior, VTableStream};
use tracing::{debug, info};

/// Demonstrates basic connection and messaging with the database
pub async fn demonstrate_connection(client: &mut Client) -> Result<()> {
    info!("Demonstrating basic Elodin-DB connectivity");
    
    // Register some example rocket components
    println!("\n📝 Registering rocket components:");
    
    let components = vec![
        ("rocket.mach", "Mach number"),
        ("rocket.thrust", "Thrust force"),
        ("rocket.altitude", "Altitude"),
        ("rocket.velocity", "Velocity vector"),
    ];
    
    for (name, description) in components {
        let component_id = ComponentId::new(name);
        let metadata = SetComponentMetadata::new(component_id, name);
        
        // Send the metadata registration
        client.send(&metadata)
            .await
            .0
            .context(format!("Failed to register {}", name))?;
            
        println!("  {} Registered: {} - {}", "✓".green(), name.cyan(), description.dimmed());
        debug!("Registered component {} with ID {:?}", name, component_id);
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
    
    println!("\n{}", "✨ Successfully demonstrated Elodin-DB connectivity!".green().bold());
    println!("\n{}", "Next steps:".bold());
    println!("  1. Run the rocket.py simulation to generate data");
    println!("  2. Extend this client to process incoming packets");
    println!("  3. Add visualization and data analysis");
    
    info!("Connection demonstration complete");
    
    Ok(())
}