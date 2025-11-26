use anyhow::Result;
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod backends;
mod config;
mod db_client;
mod layout;
mod osd_grid;
mod telemetry;

use backends::{Backend, DebugTerminalBackend, DisplayPortBackend};
use config::Config;
use db_client::DbClient;
use osd_grid::OsdGrid;
use telemetry::TelemetryProcessor;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Avatar OSD Service - MSP DisplayPort OSD for Walksnail VTX",
    long_about = None
)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Mode: 'debug' for terminal output, 'serial' for MSP DisplayPort
    #[arg(short, long, default_value = "debug")]
    mode: String,

    /// Serial port (overrides config, only for serial mode)
    #[arg(short, long)]
    serial_port: Option<String>,

    /// Database address (overrides config)
    #[arg(short, long)]
    db_addr: Option<String>,

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
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Avatar OSD Service starting...");

    // Load configuration
    let mut config = Config::from_file_or_default(&args.config)?;

    // Apply command-line overrides
    if let Some(db_addr) = args.db_addr {
        let addr: SocketAddr = db_addr.parse()?;
        config.db.host = addr.ip().to_string();
        config.db.port = addr.port();
    }

    if let Some(port) = args.serial_port {
        config.serial.port = port;
    }

    // Create OSD grid
    let grid = Arc::new(tokio::sync::RwLock::new(OsdGrid::new(
        config.osd.rows,
        config.osd.cols,
    )));

    // Initialize backend based on mode
    let backend: Box<dyn Backend> = match args.mode.as_str() {
        "debug" | "terminal" => {
            info!("Starting in debug/terminal mode");
            Box::new(DebugTerminalBackend::new())
        }
        "serial" | "displayport" => {
            info!(
                "Starting in serial/DisplayPort mode on {}",
                config.serial.port
            );
            Box::new(DisplayPortBackend::new(&config.serial)?)
        }
        _ => {
            anyhow::bail!("Unknown mode: {}. Use 'debug' or 'serial'", args.mode);
        }
    };

    // Create telemetry processor
    let telemetry_processor = Arc::new(TelemetryProcessor::new());

    // Connect to database and start streaming
    let db_addr: SocketAddr = format!("{}:{}", config.db.host, config.db.port).parse()?;
    let mut db_client = DbClient::new(db_addr, config.db.components.clone());

    // Create channel for telemetry updates
    let (tx, rx) = async_channel::bounded(100);

    // Run database streaming in background
    let telemetry_processor_clone = telemetry_processor.clone();
    let tx_clone = tx.clone();
    let db_task = stellarator::spawn(async move {
        db_client.connect_and_stream(telemetry_processor_clone, tx_clone).await
    });

    // Run OSD update loop
    let grid_clone = grid.clone();
    let config_clone = config.clone();
    let osd_task = stellarator::spawn(async move {
        run_osd_loop(rx, grid_clone, backend, telemetry_processor, config_clone).await
    });

    // Wait for either task to complete
    use futures_lite::future;
    match future::race(db_task, osd_task).await {
        Ok(Ok(())) => info!("Task completed successfully"),
        Ok(Err(e)) => info!("Task error: {}", e),
        Err(e) => info!("Task panic: {}", e),
    }

    Ok(())
}

async fn run_osd_loop(
    rx: async_channel::Receiver<()>,
    grid: Arc<tokio::sync::RwLock<OsdGrid>>,
    mut backend: Box<dyn Backend>,
    telemetry_processor: Arc<TelemetryProcessor>,
    config: Config,
) -> Result<()> {
    let frame_period = std::time::Duration::from_secs_f32(1.0 / config.osd.refresh_rate_hz);
    let mut last_frame = std::time::Instant::now();

    loop {
        // Check if we should render a new frame
        let now = std::time::Instant::now();
        let should_render = now.duration_since(last_frame) >= frame_period;

        // Wait for telemetry update or timeout
        let timeout = if should_render {
            std::time::Duration::from_millis(1)
        } else {
            frame_period.saturating_sub(now.duration_since(last_frame))
        };

        // Try to receive telemetry update with timeout
        let _ = futures_lite::future::or(
            async {
                let _ = rx.recv().await;
            },
            async {
                stellarator::sleep(timeout).await;
            },
        )
        .await;

        // Render frame if it's time
        if should_render {
            last_frame = now;

            // Get current telemetry state
            let state = telemetry_processor.get_state().await;

            // Update grid with layout
            {
                let mut grid = grid.write().await;
                layout::render(&mut grid, &state, &config.osd);
            }

            // Send to backend (with status if terminal backend)
            {
                let grid = grid.read().await;
                
                // Check if this is the debug terminal backend
                if let Some(terminal_backend) = backend.as_any_mut().downcast_mut::<DebugTerminalBackend>() {
                    terminal_backend.render_with_status(&grid, &state).await?;
                } else {
                    backend.render(&grid).await?;
                }
            }
        }
    }
}
