use anyhow::Result;
use clap::Parser;
use colored::*;
use impeller2_stellar::Client;
use std::net::SocketAddr;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod client;
mod discovery;
mod processor;

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
    
    println!("\n{}", "🚀 Elodin-DB Rust Client Example".bold().green());
    println!("{}", "================================".green());
    
    info!("Connecting to Elodin-DB at {}", addr);

    // Connect to database
    match Client::connect(addr).await {
        Ok(mut client) => {
            println!("\n{} Connected to database!", "✓".green());
            
            // Simple demonstration of sending a message
            client::demonstrate_connection(&mut client).await?;
        }
        Err(e) => {
            error!("Failed to connect: {}", e);
            println!("\n{} Connection failed: {}", "✗".red(), e);
            println!("\nMake sure:");
            println!("  1. elodin-db is running (elodin-db run [::]:2240 ~/.elodin/db)");
            println!("  2. The address {}:{} is correct", args.host, args.port);
        }
    }

    Ok(())
}