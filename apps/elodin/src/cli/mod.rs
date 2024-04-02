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
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Obtain access credentials for your user account
    Login,
    /// Manage your Monte Carlo runs
    MonteCarlo(monte_carlo::Args),
    /// Launch the Elodin editor (default)
    Editor(editor::Args),
}

impl Cli {
    pub fn from_os_args() -> Self {
        Self::parse()
    }

    pub fn run(self) {
        tracing_subscriber::fmt::init();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime failed to start");
        let res = match &self.command {
            Some(Commands::Login) => rt.block_on(self.login()),
            Some(Commands::MonteCarlo(args)) => rt.block_on(self.monte_carlo(args)),
            Some(Commands::Editor(args)) => self.editor(args.clone()),
            None => self.editor(editor::Args::default()),
        };
        if let Err(err) = res {
            eprintln!("Error: {:#}", err);
            std::process::exit(1);
        }
    }

    fn is_dev(&self) -> bool {
        self.url.ends_with("elodin.dev") || self.url.contains("localhost")
    }

    fn dirs(&self) -> Option<directories::ProjectDirs> {
        let app_name = if self.is_dev() { "cli-dev" } else { "cli" };
        directories::ProjectDirs::from("systems", "elodin", app_name)
    }
}
