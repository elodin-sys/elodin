//! Manual controller for the Betaflight SITL racing example.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use impeller2_stellar::Client;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod control;
mod input;

use control::ControlSender;
use input::{InputReader, StickMode};

#[derive(Debug, Parser)]
#[command(about = "Gamepad/keyboard input provider for Betaflight SITL")]
struct Args {
    /// Address of the Elodin DB server.
    #[arg(short = 'H', long, default_value = "127.0.0.1:2240")]
    host: String,
    /// Mode 1: left pitch/yaw, right throttle/roll (Mode 2 is the default).
    #[arg(long)]
    mode1: bool,
}

#[stellarator::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let address: SocketAddr = args
        .host
        .parse()
        .with_context(|| format!("invalid Elodin DB address: {}", args.host))?;
    let stick_mode = if args.mode1 {
        StickMode::Mode1
    } else {
        StickMode::Mode2
    };
    print_controls(stick_mode);
    run(address, stick_mode).await
}

async fn run(address: SocketAddr, stick_mode: StickMode) -> Result<()> {
    let mut client = loop {
        match Client::connect(address).await {
            Ok(client) => break client,
            Err(error) => {
                info!(%address, %error, "waiting for Elodin DB");
                stellarator::sleep(Duration::from_secs(1)).await;
            }
        }
    };

    let mut sender = ControlSender::new();
    sender.send_vtable(&mut client).await?;
    stellarator::sleep(Duration::from_millis(100)).await;

    let mut input = InputReader::new(stick_mode);
    let mut last_display = Instant::now();
    loop {
        let control = input.read();
        sender.send(&mut client, control).await?;
        if last_display.elapsed() >= Duration::from_millis(250) {
            last_display = Instant::now();
            eprint!(
                "\rroll={:+.2} pitch={:+.2} throttle={:.2} yaw={:+.2} arm={} angle={}    ",
                control.roll,
                control.pitch,
                control.throttle,
                control.yaw,
                control.armed,
                control.angle_mode,
            );
        }
        stellarator::sleep(Duration::from_millis(10)).await;
    }
}

fn print_controls(mode: StickMode) {
    println!("Betaflight SITL manual controller ({mode:?})");
    println!("  W/S             throttle up/down");
    println!("  Q/E or A/D      yaw left/right");
    println!("  arrow up/down   pitch forward/back");
    println!("  arrow left/right roll left/right");
    println!("  Shift+R / F     arm (at min throttle) / disarm");
    println!("  M               toggle ANGLE mode");
    println!("  Gamepad A/B/Y   arm / disarm / toggle ANGLE");
}
