use anyhow::Result;
use clap::Parser;
use impeller2_stellar::Client;
use std::net::SocketAddr;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod client;
mod discovery;
mod processor;
mod tui;

#[derive(Parser, Debug)]
#[command(author, version, about = "Rust client example for Elodin-DB rocket telemetry", long_about = None)]
struct Args {
    /// Host address of the Elodin-DB server
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Port of the Elodin-DB server
    #[arg(short, long, default_value_t = 2240)]
    port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[stellarator::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let filter = if args.verbose {
        "debug"
    } else {
        "info"
    };
    
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| filter.into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse address
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    
    info!("Connecting to Elodin-DB at {}", addr);

    // Connect to database
    match Client::connect(addr).await {
        Ok(mut client) => {
            info!("Connected to database!");
            
            // Run the client with the TUI
            client::run_with_tui(&mut client).await?;
        }
        Err(e) => {
            error!("Failed to connect: {}", e);
            eprintln!("\nConnection failed: {}", e);
            eprintln!("\nMake sure:");
            eprintln!("  1. elodin-db is running (elodin-db run [::]:2240 ~/.elodin/db)");
            eprintln!("  2. The address {}:{} is correct", args.host, args.port);
        }
    }

    Ok(())
}