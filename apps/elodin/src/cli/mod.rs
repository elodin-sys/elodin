use clap::{Parser, Subcommand};
use miette::Context;
use miette::IntoDiagnostic;
use tracing_subscriber::{EnvFilter, fmt::time::ChronoLocal, prelude::*};
mod auth;
mod editor;
mod monte_carlo;

#[derive(Parser, Clone)]
#[command(
    author,
    version = concat!(env!("CARGO_PKG_VERSION"), "+", env!("GIT_HASH")),
    about,
    long_about = None
)]
pub struct Cli {
    #[arg(short, long, default_value = "https://app.elodin.systems")]
    url: String,
    /// OIDC issuer (Keycloak realm) used by `elodin login`.
    #[arg(
        long,
        env = "ELODIN_ISSUER",
        default_value = "https://auth.elodin.systems/realms/elodin",
        global = true
    )]
    issuer: String,
    /// Elodin Cloud API base URL used by authenticated commands.
    #[arg(
        long,
        env = "ELODIN_API_URL",
        default_value = "https://api.elodin.systems",
        global = true
    )]
    api_url: String,
    #[command(subcommand)]
    command: Option<Commands>,
    #[arg(long, hide = true)]
    markdown_help: bool,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Sign up for Elodin Cloud (creates your account + organization)
    Signup(auth::SignupArgs),
    /// Log in to Elodin Cloud through your browser (OIDC + PKCE)
    Login(auth::LoginArgs),
    /// Remove stored Elodin Cloud credentials and end the session
    Logout,
    /// Print the currently authenticated Elodin Cloud identity
    Whoami(auth::WhoamiArgs),
    /// Create and list Elodin Cloud projects
    Projects(auth::ProjectsArgs),
    /// Launch the Elodin editor (default)
    Editor(editor::Args),
    /// Run an Elodin simulation in headless mode
    #[cfg(not(target_os = "windows"))]
    Run(editor::Args),
    /// Run and manage Monte Carlo campaigns
    #[cfg(not(target_os = "windows"))]
    MonteCarlo(monte_carlo::Args),
    /// Start the headless sensor camera render server (managed by s10)
    #[cfg(not(target_os = "windows"))]
    RenderServer(editor::RenderServerArgs),
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

        // TracyClient binds its port from a C++ static constructor (before
        // main), so TRACY_PORT must be in the environment before the process
        // starts.  If it isn't set yet, re-exec with the correct port so the
        // Tracy client picks it up on the next load.
        #[cfg(all(feature = "tracy", not(target_os = "windows")))]
        if std::env::var("TRACY_PORT").is_err() {
            let port = match &self.command {
                Some(Commands::RenderServer(_)) => "8088",
                _ => "8087",
            };
            use std::os::unix::process::CommandExt;
            let exe = std::env::current_exe().expect("failed to get current exe path");
            let err = std::process::Command::new(exe)
                .args(&std::env::args().collect::<Vec<_>>()[1..])
                .env("TRACY_PORT", port)
                .exec();
            return Err(miette::miette!("re-exec failed: {err}"));
        }

        let filter = EnvFilter::try_from_default_env()
            .or_else(|_| {
                EnvFilter::try_new("s10=info,elodin=info,impeller=info,impeller::bevy=error,error")
            })
            .unwrap_or_else(|_| EnvFilter::new("info"));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_timer(ChronoLocal::new("%Y-%m-%d %H:%M:%S%.3f".to_string()));

        #[cfg(feature = "tracy")]
        let init_res = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(tracing_tracy::TracyLayer::default())
            .try_init();

        #[cfg(not(feature = "tracy"))]
        let init_res = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .try_init();

        if let Err(err) = init_res {
            eprintln!("warning: failed to install global tracing subscriber ({err})");
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime failed to start");

        // Auth commands manage their own config dir and must not be gated by the
        // first-launch onboarding (which exits early to nudge the Python SDK install).
        let is_auth_command = matches!(
            self.command,
            Some(Commands::Signup(_))
                | Some(Commands::Login(_))
                | Some(Commands::Logout)
                | Some(Commands::Whoami(_))
                | Some(Commands::Projects(_))
        );
        if !is_auth_command && let Err(err) = self.first_launch() {
            eprintln!("Error: {:#}", err);
            std::process::exit(1);
        }

        match &self.command {
            Some(Commands::Signup(args)) => self.clone().signup(args.clone(), rt),
            Some(Commands::Login(args)) => self.clone().login(args.clone(), rt),
            Some(Commands::Logout) => self.clone().logout(rt),
            Some(Commands::Whoami(args)) => self.clone().whoami(args.clone(), rt),
            Some(Commands::Projects(args)) => self.clone().projects(args.clone(), rt),
            Some(Commands::Editor(args)) => self.clone().editor(args.clone(), rt),
            #[cfg(not(target_os = "windows"))]
            Some(Commands::Run(args)) => self.clone().run_headless(args.clone(), rt),
            #[cfg(not(target_os = "windows"))]
            Some(Commands::MonteCarlo(args)) => self.clone().monte_carlo(args.clone(), rt),
            #[cfg(not(target_os = "windows"))]
            Some(Commands::RenderServer(args)) => self.clone().render_server(args.clone()),
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

            println!(
                "Ensure the Elodin Python SDK is installed in your preferred Python virtual environment:"
            );
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
