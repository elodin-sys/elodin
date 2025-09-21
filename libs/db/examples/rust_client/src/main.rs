use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod client;
mod control;
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
    
    info!("Starting Elodin-DB client for {}", addr);

    // Run the client with connection retry and resilient TUI
    client::run_resilient(addr).await?;

    Ok(())
}