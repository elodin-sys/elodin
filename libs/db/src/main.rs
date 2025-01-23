use std::{net::SocketAddr, path::PathBuf};

use clap::Parser;
use elodin_db::Server;
use miette::IntoDiagnostic;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    addr: SocketAddr,
    path: PathBuf,
}

fn main() -> miette::Result<()> {
    tracing_subscriber::fmt::init();
    stellarator::run(|| async {
        let Args { addr, path } = Args::parse();
        let server = Server::new(path, addr).into_diagnostic()?;
        server.run().await.into_diagnostic()
    })
}
