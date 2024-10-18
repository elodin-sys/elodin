use clap::{Parser, Subcommand};
use miette::miette;
use miette::Context;
use miette::IntoDiagnostic;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

mod auth;
mod create;
mod editor;
mod license;
mod monte_carlo;

#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(short, long, default_value = "https://app.elodin.systems")]
    url: String,
    #[command(subcommand)]
    command: Option<Commands>,
    #[arg(long, hide = true)]
    markdown_help: bool,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Obtain access credentials for your user account
    Login,
    /// Manage your Monte Carlo runs
    MonteCarlo(monte_carlo::Args),
    /// Launch the Elodin editor (default)
    Editor(editor::Args),
    /// Run an Elodin simulaton in headless mode
    #[cfg(not(target_os = "windows"))]
    Run(editor::Args),
    /// Create template
    Create(create::Args),
}

impl Cli {
    pub fn from_os_args() -> Self {
        Self::parse()
    }

    pub fn run(self) -> miette::Result<()> {
        if self.markdown_help {
            clap_markdown::print_help_markdown::<Cli>();
            std::process::exit(0);
        }

        let filter = if std::env::var("RUST_LOG").is_ok() {
            EnvFilter::builder().from_env_lossy()
        } else {
            EnvFilter::builder().parse_lossy(
                "s10=info,elodin=info,impeller=info,nox_ecs=info,impeller::bevy=error,error",
            )
        };

        let _ = tracing_subscriber::fmt::fmt()
            .with_target(false)
            .with_env_filter(filter)
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(
                "%Y-%m-%d %H:%M:%S%.3f".to_string(),
            ))
            .try_init();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime failed to start");

        if let Err(err) = self.first_launch() {
            eprintln!("Error: {:#}", err);
            std::process::exit(1);
        }

        match self.command {
            // un-licensed commands
            Some(Commands::Login) | Some(Commands::Create(_)) | None => {}
            // licensed commands
            _ => {
                rt.block_on(self.verify_license_key())?;
            }
        }
        match &self.command {
            Some(Commands::Login) => rt.block_on(self.login()),
            Some(Commands::MonteCarlo(args)) => rt.block_on(self.monte_carlo(args)),
            Some(Commands::Editor(args)) => self.clone().editor(args.clone(), rt),
            #[cfg(not(target_os = "windows"))]
            Some(Commands::Run(args)) => self
                .run_sim(args, rt, CancellationToken::new())?
                .join()
                .map_err(|_| miette!("join error"))?,
            Some(Commands::Create(args)) => self.create_template(args).into_diagnostic(),
            None => self.clone().editor(editor::Args::default(), rt),
        }
    }

    fn first_launch(&self) -> miette::Result<()> {
        let dirs = self.dirs().into_diagnostic()?;
        let data_dir = dirs.data_dir();
        let is_first_launch = !data_dir.exists();
        std::fs::create_dir_all(data_dir)
            .into_diagnostic()
            .context("failed to create data directory")?;

        if is_first_launch {
            println!("This is your first use of the Elodin CLI!\n");

            println!("You can log in using this command:");
            println!("    elodin login\n");

            println!("Ensure the Elodin Python SDK is installed in your preferred Python virtual environment:");
            println!("    pip install -U elodin\n");

            println!("Check out our docs (at https://docs.elodin.systems) for more information.");
            std::process::exit(0);
        }

        Ok(())
    }

    fn is_dev(&self) -> bool {
        self.url.ends_with("elodin.dev") || self.url.contains("localhost")
    }

    fn dirs(&self) -> Result<directories::ProjectDirs, std::io::Error> {
        let app_name = if self.is_dev() { "cli-dev" } else { "cli" };
        directories::ProjectDirs::from("systems", "elodin", app_name).ok_or(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "failed to get data directory",
        ))
    }
}
