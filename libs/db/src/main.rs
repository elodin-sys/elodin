use std::{net::SocketAddr, path::PathBuf};

use clap::{Parser, Subcommand};
use elodin_db::Server;
use miette::IntoDiagnostic;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    #[command(about = "Run the Elodin database server")]
    Run(RunArgs),
    #[command(about = "Run a Lua script or launch a REPL")]
    Lua(impeller2_cli::Args),
}

#[derive(clap::Args, Clone, Debug)]
struct RunArgs {
    #[clap(default_value = "[::]:2240", help = "Address to bind the server to")]
    addr: SocketAddr,
    #[clap(help = "Path to the data directory")]
    path: Option<PathBuf>,
    #[clap(long, help = "Path to the configuration file")]
    pub config: Option<PathBuf>,
    #[cfg(feature = "axum")]
    #[clap(long, help = "Address to bind the HTTP server to")]
    http_addr: Option<SocketAddr>,
    #[clap(long, hide = true)]
    reset: bool,
}

fn main() -> miette::Result<()> {
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::builder().from_env_lossy()
    } else {
        EnvFilter::builder().parse_lossy("elodin_db=info")
    };

    let _ = tracing_subscriber::fmt::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
            "%Y-%m-%d %H:%M:%S%.3f".to_string(),
        ))
        .try_init();
    let args = Cli::parse();
    stellarator::run(|| async {
        match args.command {
            Commands::Run(RunArgs {
                addr,
                http_addr,
                path,
                config,
                reset,
            }) => {
                let path = path.unwrap_or_else(|| {
                    let dirs =
                        directories::ProjectDirs::from("systems", "elodin", "db").expect("no dirs");
                    dirs.data_dir().join("data")
                });
                if reset && path.exists() {
                    info!(?path, "resetting db");
                    std::fs::remove_dir_all(&path).unwrap_or_else(|_| {
                        tracing::warn!("failed to remove existing data directory");
                    });
                }
                info!(?path, "starting db");
                let server = Server::new(path, addr).into_diagnostic()?;
                let axum_db = server.db.clone();
                let db = stellarator::spawn(server.run());
                if let Some(http_addr) = http_addr {
                    stellarator::struc_con::tokio(move |_| async move {
                        elodin_db::axum::serve(http_addr, axum_db).await.unwrap()
                    });
                }
                if let Some(lua_config) = config {
                    let args = impeller2_cli::Args {
                        path: Some(lua_config),
                    };
                    impeller2_cli::run(args)
                        .await
                        .map_err(|e| miette::miette!(e))?;
                }
                db.await.unwrap().into_diagnostic()
            }
            Commands::Lua(args) => impeller2_cli::run(args)
                .await
                .map_err(|e| miette::miette!(e)),
        }
    })
}
