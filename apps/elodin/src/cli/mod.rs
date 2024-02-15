use std::ffi::OsString;

use clap::{Parser, Subcommand};

mod auth;
mod editor;
mod monte_carlo;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(short, long, default_value = "https://app.elodin.systems")]
    url: String,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Obtain access credentials for your user account
    Login,
    /// Manage your Monte Carlo runs
    MonteCarlo(monte_carlo::Args),
    Editor(editor::Args),
}

impl Cli {
    pub fn from_os_args() -> Self {
        Self::parse()
    }

    pub fn from_args(args: &[OsString]) -> Self {
        Self::parse_from(args)
    }

    pub fn run(self) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(async {
            match &self.command {
                Commands::Login => self.login().await,
                Commands::MonteCarlo(args) => self.monte_carlo(args).await,
                Commands::Editor(args) => self.editor(args.clone()).await,
            }
        });
        if let Err(err) = res {
            eprintln!("Error: {:#}", err);
            std::process::exit(1);
        }
    }

    fn is_dev(&self) -> bool {
        self.url.ends_with("elodin.dev")
    }

    fn xdg_dirs(&self) -> xdg::BaseDirectories {
        let profile = if self.is_dev() { "dev" } else { "" };
        xdg::BaseDirectories::with_profile("elodin", profile).unwrap()
    }
}
