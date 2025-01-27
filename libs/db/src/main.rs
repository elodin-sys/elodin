use std::{net::SocketAddr, path::PathBuf};

use clap::{Parser, Subcommand};
use elodin_db::Server;
use miette::IntoDiagnostic;

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
    addr: SocketAddr,
    path: PathBuf,
}

fn main() -> miette::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Cli::parse();
    stellarator::run(|| async {
        match args.command {
            Commands::Run(RunArgs { addr, path }) => {
                let server = Server::new(path, addr).into_diagnostic()?;
                server.run().await.into_diagnostic()
            }
            Commands::Lua(args) => impeller2_cli::run(args)
                .await
                .map_err(|e| miette::miette!(e)),
        }
    })
}
