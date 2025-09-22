use anyhow::{Context, Result};
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
    /// Host address of the Elodin-DB server. Can be specified as 'host' or 'host:port'.
    /// If port is included here, it overrides the -p flag.
    /// Examples: "127.0.0.1", "[::1]", "localhost:2290", "[::]:2240"
    #[arg(short = 'H', long, default_value = "127.0.0.1:2240")]
    host: String,

    /// Port of the Elodin-DB server (ignored if port is specified in --host)
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
    let filter = if args.verbose { "debug" } else { "info" };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse address - support both "host" and "host:port" formats
    let addr = parse_address(&args.host, args.port)
        .with_context(|| format!("Failed to parse address from '{}'", args.host))?;

    info!("Starting Elodin-DB client for {}", addr);

    // Run the client with connection retry and resilient TUI
    client::run_resilient(addr).await?;

    Ok(())
}

/// Parse address from host string and optional port override
/// Supports formats like:
/// - "127.0.0.1" (uses port argument)
/// - "127.0.0.1:2290" (port in host string)
/// - "[::1]" (IPv6, uses port argument)
/// - "[::1]:2290" (IPv6 with port)
/// - "[::]:2240" (IPv6 any address with port)
/// - "localhost" (uses port argument)
/// - "localhost:2290" (port in host string)
fn parse_address(host_str: &str, default_port: u16) -> Result<SocketAddr> {
    // First, try to parse as a complete SocketAddr (handles "host:port" format)
    if let Ok(addr) = host_str.parse::<SocketAddr>() {
        return Ok(addr);
    }

    // Check if the host string contains a port
    // For IPv6, we need to handle the bracketed format specially
    let (host, port) = if host_str.starts_with('[') {
        // IPv6 address - look for "]:port" pattern
        if let Some(bracket_end) = host_str.find(']') {
            let host_part = &host_str[..=bracket_end];
            if bracket_end + 1 < host_str.len()
                && &host_str[bracket_end + 1..bracket_end + 2] == ":"
            {
                // Has port after bracket
                let port_str = &host_str[bracket_end + 2..];
                let port = port_str
                    .parse::<u16>()
                    .with_context(|| format!("Invalid port in '{}'", host_str))?;
                (host_part.to_string(), port)
            } else {
                // No port, use default
                (host_part.to_string(), default_port)
            }
        } else {
            // Invalid IPv6 format
            return Err(anyhow::anyhow!("Invalid IPv6 address format: {}", host_str));
        }
    } else if let Some(colon_pos) = host_str.rfind(':') {
        // For non-IPv6, check if there's a colon (could be port)
        // But need to be careful with hostnames that might contain colons
        let potential_port = &host_str[colon_pos + 1..];
        if potential_port.parse::<u16>().is_ok() {
            // Valid port number after colon
            let host_part = &host_str[..colon_pos];
            let port = potential_port.parse::<u16>()?;
            (host_part.to_string(), port)
        } else {
            // Not a valid port, treat whole string as hostname
            (host_str.to_string(), default_port)
        }
    } else {
        // No colon, use the whole string as host and default port
        (host_str.to_string(), default_port)
    };

    // Now parse the final address
    let addr_str = if host.starts_with('[') && host.ends_with(']') {
        // Already bracketed IPv6
        format!("{}:{}", host, port)
    } else if host.contains(':') {
        // Bare IPv6 address, needs brackets
        format!("[{}]:{}", host, port)
    } else {
        // IPv4 or hostname
        format!("{}:{}", host, port)
    };

    addr_str
        .parse::<SocketAddr>()
        .with_context(|| format!("Failed to parse address '{}'", addr_str))
}
