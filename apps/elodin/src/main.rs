use clap::{Parser, Subcommand};

mod auth;
mod editor;
mod monte_carlo;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
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

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if let Err(err) = cli.run().await {
        eprintln!("Error: {:#}", err);
        std::process::exit(1);
    }
}

impl Cli {
    async fn run(self) -> anyhow::Result<()> {
        match &self.command {
            Commands::Login => self.login().await,
            Commands::MonteCarlo(args) => self.monte_carlo(args).await,
            Commands::Editor(args) => self.editor(args.clone()).await,
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
