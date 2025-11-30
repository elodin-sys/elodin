//! RC Controller for BDX Jet Simulation
//!
//! Reads gamepad (FrSky X20 R5) and keyboard input, sends control commands
//! to the simulation via elodin-db.
//!
//! Usage:
//!     cargo run -p rc-jet-controller
//!     cargo run -p rc-jet-controller -- --host 127.0.0.1:2240
//!     cargo run -p rc-jet-controller -- --mode1  # EU stick mode

use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use impeller2_stellar::Client;
use std::net::SocketAddr;
use std::time::Duration;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod control;
mod input;

use control::ControlSender;
use input::{InputReader, StickMode};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "RC Controller for BDX Jet Simulation",
    long_about = None
)]
struct Args {
    /// Host address of the Elodin-DB server
    #[arg(short = 'H', long, default_value = "127.0.0.1:2240")]
    host: String,

    /// Use Mode 1 stick layout (EU/Asia: Left=Pitch/Yaw, Right=Throttle/Roll)
    #[arg(long)]
    mode1: bool,

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

    // Parse address
    let addr: SocketAddr = args
        .host
        .parse()
        .with_context(|| format!("Invalid address: {}", args.host))?;

    // Determine stick mode
    let stick_mode = if args.mode1 {
        StickMode::Mode1
    } else {
        StickMode::Mode2
    };

    print_banner(stick_mode);

    // Run controller
    run_controller(addr, stick_mode).await
}

fn print_banner(stick_mode: StickMode) {
    println!();
    println!(
        "{}",
        "╔═══════════════════════════════════════════════════════════╗".cyan()
    );
    println!(
        "{}",
        "║         🛩️  BDX JET RC CONTROLLER  🛩️                      ║".cyan()
    );
    println!(
        "{}",
        "╚═══════════════════════════════════════════════════════════╝".cyan()
    );
    println!();

    let mode_str = match stick_mode {
        StickMode::Mode2 => "Mode 2 (US): Left=Throttle/Yaw, Right=Pitch/Roll",
        StickMode::Mode1 => "Mode 1 (EU): Left=Pitch/Yaw, Right=Throttle/Roll",
    };
    println!("  {} {}", "Stick Mode:".bold(), mode_str);
    println!();
    println!("  {}", "Keyboard Controls:".bold());
    println!("    {} Throttle up/down", "W/S".green());
    println!("    {} Rudder left/right", "A/D".green());
    println!("    {} Elevator up/down (pitch)", "↑/↓".green());
    println!("    {} Aileron left/right (roll)", "←/→".green());
    println!();
}

async fn run_controller(addr: SocketAddr, stick_mode: StickMode) -> Result<()> {
    info!("Connecting to elodin-db at {}", addr);

    // Connect with retry
    let mut client = loop {
        match Client::connect(addr).await {
            Ok(client) => {
                println!("  {} Connected to {}", "✓".green(), addr);
                break client;
            }
            Err(e) => {
                println!(
                    "  {} Waiting for simulation at {} - {}",
                    "⏳".yellow(),
                    addr,
                    e
                );
                stellarator::sleep(Duration::from_secs(1)).await;
            }
        }
    };

    // Initialize control sender
    let mut control_sender = ControlSender::new();

    // Send VTable definition
    control_sender.send_vtable(&mut client).await?;
    stellarator::sleep(Duration::from_millis(100)).await;

    // Initialize input reader
    let mut input_reader = InputReader::new(stick_mode);

    println!();
    println!("  {} Streaming control inputs...", "📡".green());
    println!("  {} Press Ctrl+C to exit", "💡".dimmed());
    println!();

    // Main control loop
    let mut last_display = std::time::Instant::now();

    loop {
        // Read input
        let input = input_reader.read();

        // Send control
        control_sender.send_control(&mut client, input).await?;

        // Display current values periodically
        if last_display.elapsed() >= Duration::from_millis(100) {
            last_display = std::time::Instant::now();
            display_controls(&input);
        }

        // Small delay (~100Hz polling)
        stellarator::sleep(Duration::from_millis(10)).await;
    }
}

fn display_controls(input: &input::ControlInput) {
    // Clear line and print current values
    print!("\r  ");
    print!(
        "Throttle: {}%  ",
        format!("{:5.1}", input.throttle * 100.0).yellow()
    );
    print!(
        "Elevator: {:+6.1}°  ",
        format!("{:+6.1}", input.elevator.to_degrees()).cyan()
    );
    print!(
        "Aileron: {:+6.1}°  ",
        format!("{:+6.1}", input.aileron.to_degrees()).cyan()
    );
    print!(
        "Rudder: {:+6.1}°",
        format!("{:+6.1}", input.rudder.to_degrees()).cyan()
    );
    print!("     "); // Clear any trailing chars

    use std::io::Write;
    let _ = std::io::stdout().flush();
}
