///! Simplified Rust client example for Elodin-DB
///! 
///! This demonstrates connecting to the database and subscribing to rocket telemetry
///! using basic TCP sockets and the Impeller2 protocol.

use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

/// FNV-1a 64-bit hash for component IDs
fn component_id(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Packet types in the Impeller2 protocol
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum PacketType {
    Msg = 0,
    Table = 1,
    TimeSeries = 2,
}

/// Packet header (8 bytes total)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct PacketHeader {
    len: u32,       // Length of packet (little-endian)
    ty: PacketType, // Packet type
    id: [u8; 2],    // Packet ID
    request_id: u8, // Request ID for correlation
}

/// Component metadata message
#[derive(Serialize, Deserialize, Debug)]
struct SetComponentMetadata {
    component_id: u64,
    name: String,
    #[serde(default)]
    metadata: std::collections::HashMap<String, String>,
    #[serde(default)]
    asset: bool,
}

/// Stream subscription message  
#[derive(Serialize, Deserialize, Debug)]
struct Stream {
    behavior: StreamBehavior,
    id: u64,
}

#[derive(Serialize, Deserialize, Debug)]
enum StreamBehavior {
    RealTime,
}

/// Simple VTable stream subscription
#[derive(Serialize, Deserialize, Debug)]
struct VTableStream {
    id: [u8; 2],
}

/// CLI arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "Simplified Rust client for Elodin-DB rocket telemetry")]
struct Args {
    /// Host address of the Elodin-DB server
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Port of the Elodin-DB server
    #[arg(short, long, default_value_t = 2240)]
    port: u16,
}

/// Rocket telemetry components
struct RocketComponents {
    components: Vec<(&'static str, u64)>,
}

impl RocketComponents {
    fn new() -> Self {
        let component_names = vec![
            "rocket.mach",
            "rocket.thrust", 
            "rocket.fin_deflect",
            "rocket.angle_of_attack",
            "rocket.dynamic_pressure",
            "rocket.world_pos",
            "rocket.world_vel",
            "rocket.aero_force",
            "rocket.center_of_gravity",
        ];
        
        let components = component_names
            .into_iter()
            .map(|name| (name, component_id(name)))
            .collect();
            
        Self { components }
    }
}

/// Send a message packet
fn send_msg<T: Serialize>(
    stream: &mut TcpStream,
    msg_id: [u8; 2],
    msg: &T,
) -> Result<()> {
    // Serialize message with postcard
    let payload = postcard::to_stdvec(msg)?;
    
    // Create packet header
    let header = PacketHeader {
        len: (8 + payload.len()) as u32,
        ty: PacketType::Msg,
        id: msg_id,
        request_id: 0,
    };
    
    // Write header
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const _ as *const u8,
            std::mem::size_of::<PacketHeader>(),
        )
    };
    stream.write_all(header_bytes)?;
    
    // Write payload
    stream.write_all(&payload)?;
    stream.flush()?;
    
    Ok(())
}

/// Main client implementation
struct RocketClient {
    stream: TcpStream,
    components: RocketComponents,
}

impl RocketClient {
    /// Connect to the database
    fn connect(host: &str, port: u16) -> Result<Self> {
        println!("{} Connecting to {}:{}...", "🔌".cyan(), host, port);
        
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr)
            .context("Failed to connect to Elodin-DB")?;
            
        // Set timeouts
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        
        Ok(Self {
            stream,
            components: RocketComponents::new(),
        })
    }
    
    /// Register component metadata
    fn register_components(&mut self) -> Result<()> {
        println!("{} Registering rocket components...", "📝".yellow());
        
        const SET_COMPONENT_METADATA_ID: [u8; 2] = [0x96, 0x23]; // Pre-computed FNV hash
        
        for (name, id) in &self.components.components {
            let metadata = SetComponentMetadata {
                component_id: *id,
                name: name.to_string(),
                metadata: Default::default(),
                asset: false,
            };
            
            send_msg(&mut self.stream, SET_COMPONENT_METADATA_ID, &metadata)?;
            println!("  {} Registered: {}", "✓".green(), name.cyan());
        }
        
        Ok(())
    }
    
    /// Subscribe to telemetry stream
    fn subscribe(&mut self) -> Result<()> {
        println!("{} Subscribing to telemetry streams...", "📡".yellow());
        
        // Subscribe to real-time stream
        const STREAM_ID: [u8; 2] = [0x6e, 0x1f]; // Pre-computed FNV hash
        let stream = Stream {
            behavior: StreamBehavior::RealTime,
            id: 1,
        };
        send_msg(&mut self.stream, STREAM_ID, &stream)?;
        
        // Subscribe to VTable stream
        const VTABLE_STREAM_ID: [u8; 2] = [0x4d, 0x30]; // Pre-computed FNV hash
        let vtable_stream = VTableStream {
            id: [1, 0],
        };
        send_msg(&mut self.stream, VTABLE_STREAM_ID, &vtable_stream)?;
        
        println!("{} Subscribed successfully!", "✓".green().bold());
        Ok(())
    }
    
    /// Process incoming telemetry (simplified)
    fn process_telemetry(&mut self) -> Result<()> {
        println!("\n{}", "╭─────────── Rocket Telemetry Status ───────────╮".cyan());
        println!("{}", "│                                               │".cyan());
        println!("{} {} {}", "│".cyan(), "Connection Status:".white(), "Connected ✓".green());
        println!("{} {} {}", "│".cyan(), "Database Host:".white(), format!("{}:{}", self.stream.peer_addr()?.ip(), self.stream.peer_addr()?.port()).yellow());
        println!("{} {} {}", "│".cyan(), "Components:".white(), format!("{} registered", self.components.components.len()).yellow());
        println!("{} {} {}", "│".cyan(), "Stream Mode:".white(), "Real-time".green());
        println!("{}", "│                                               │".cyan());
        println!("{}", "│ Waiting for telemetry data...                │".cyan());
        println!("{}", "│                                               │".cyan());
        println!("{}", "│ Note: This simplified client demonstrates    │".cyan());
        println!("{}", "│ the connection and registration process.     │".cyan());
        println!("{}", "│ Full packet processing would require the     │".cyan());
        println!("{}", "│ complete Impeller2 protocol implementation.  │".cyan());
        println!("{}", "│                                               │".cyan());
        println!("{}", "╰───────────────────────────────────────────────╯".cyan());
        
        // In a full implementation, we would:
        // 1. Read packet headers
        // 2. Parse table/time-series packets
        // 3. Decomponentize the data
        // 4. Display telemetry values
        
        // For now, just keep the connection alive
        loop {
            std::thread::sleep(Duration::from_secs(1));
            print!(".");
            std::io::stdout().flush()?;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("{}", "🚀 Elodin-DB Rust Client - Rocket Telemetry".bold().cyan());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".cyan());
    
    // Connect to database
    let mut client = RocketClient::connect(&args.host, args.port)?;
    println!("{} Connected successfully!", "✓".green().bold());
    
    // Register components
    client.register_components()?;
    
    // Subscribe to streams
    client.subscribe()?;
    
    // Process telemetry
    // Handle Ctrl+C gracefully
    let result = tokio::select! {
        res = tokio::task::spawn_blocking(move || client.process_telemetry()) => {
            res?
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\n{} Shutting down gracefully...", "⏹".red().bold());
            Ok(())
        }
    };
    
    println!("{} Client disconnected", "✓".green().bold());
    result
}
