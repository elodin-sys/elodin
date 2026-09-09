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
    /// Stream a deterministic bounded sequence for the physical sign audit.
    #[arg(long)]
    audit: bool,
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
    run(address, stick_mode, args.audit).await
}

async fn run(address: SocketAddr, stick_mode: StickMode, audit: bool) -> Result<()> {
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
    let sequence_start = Instant::now();
    let mut last_display = Instant::now();
    loop {
        let control = if audit {
            audit_input(sequence_start.elapsed().as_secs_f64())
        } else {
            input.read()
        };
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

/// The simulation begins roughly 2.5 seconds after this process connects due
/// to bridge startup and warmup. The long initial safe period leaves five full
/// simulation seconds for Betaflight's boot grace before requesting arm.
fn audit_input(elapsed: f64) -> input::ControlInput {
    use input::ControlInput;

    let mut control = ControlInput::safe();
    match elapsed {
        t if t < 8.0 => {}
        t if t < 9.0 => control.armed = true,
        t if t < 11.0 => {
            control.armed = true;
            control.throttle = 0.15;
        }
        t if t < 12.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.roll = 0.25;
        }
        t if t < 13.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.roll = -0.25;
        }
        t if t < 14.0 => {
            control.armed = true;
            control.throttle = 0.15;
        }
        t if t < 15.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.pitch = 0.25;
        }
        t if t < 16.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.pitch = -0.25;
        }
        t if t < 17.0 => {
            control.armed = true;
            control.throttle = 0.15;
        }
        t if t < 18.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.yaw = 0.25;
        }
        t if t < 19.0 => {
            control.armed = true;
            control.throttle = 0.15;
            control.yaw = -0.25;
        }
        t if t < 20.0 => {
            control.armed = true;
            control.throttle = 0.15;
        }
        t if t < 21.0 => {
            control.armed = true;
            control.throttle = 0.30;
        }
        t if t < 23.0 => control.armed = true,
        _ => {}
    }
    control
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_sequence_starts_and_ends_safe() {
        assert_eq!(audit_input(0.0), input::ControlInput::safe());
        assert_eq!(audit_input(24.0), input::ControlInput::safe());
    }

    #[test]
    fn audit_sequence_injects_every_control_axis_in_angle_mode() {
        let samples = [11.5, 14.5, 17.5, 20.5].map(audit_input);
        assert!(
            samples
                .iter()
                .all(|sample| sample.armed && sample.angle_mode)
        );
        assert!(samples[0].roll > 0.0);
        assert!(samples[1].pitch > 0.0);
        assert!(samples[2].yaw > 0.0);
        assert!(samples[3].throttle > 0.2);
    }
}
