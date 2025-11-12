use clap::{Parser, Subcommand};
use miette::Context;
use miette::IntoDiagnostic;
use miette::miette;
use stellarator::util::CancelToken;
use tracing_subscriber::EnvFilter;

mod editor;

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
    /// Launch the Elodin editor (default)
    Editor(editor::Args),
    /// Run an Elodin simulaton in headless mode
    #[cfg(not(target_os = "windows"))]
    Run(editor::Args),
}

impl Cli {
    pub fn from_os_args() -> Self {
        Self::parse()
    }

    pub fn run(self) -> miette::Result<()> {
        self.ensure_xwayland_if_needed()?;
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

        match &self.command {
            Some(Commands::Editor(args)) => self.clone().editor(args.clone(), rt),
            #[cfg(not(target_os = "windows"))]
            Some(Commands::Run(args)) => self
                .run_sim(args, rt, CancelToken::new())?
                .join()
                .map_err(|_| miette!("join error"))?,
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

    #[cfg(target_os = "linux")]
    fn ensure_xwayland_if_needed(&self) -> miette::Result<()> {
        use std::{
            env,
            process::{Command, Stdio},
        };

        if env::var_os("ELODIN_FORCE_WAYLAND").is_some() || env::var_os("ELODIN_XWAYLAND").is_some()
        {
            return Ok(());
        }

        let is_wayland_session = env::var_os("WAYLAND_DISPLAY").is_some()
            || env::var("XDG_SESSION_TYPE")
                .map(|value| value.eq_ignore_ascii_case("wayland"))
                .unwrap_or(false);
        if !is_wayland_session {
            return Ok(());
        }

        let display_set = env::var_os("DISPLAY").is_some();
        if !display_set {
            eprintln!(
                "Warning: Wayland session detected but DISPLAY is unset; continuing under Wayland"
            );
            return Ok(());
        }

        let current_exe = std::env::current_exe().into_diagnostic()?;
        let mut env_overrides = env::vars_os().collect::<Vec<_>>();
        env_overrides.retain(|(key, _)| {
            key != "WAYLAND_DISPLAY" && key != "XDG_SESSION_TYPE" && key != "ELODIN_FORCE_WAYLAND"
        });
        env_overrides.push(("ELODIN_XWAYLAND".into(), "1".into()));

        let status = Command::new(current_exe)
            .args(std::env::args_os().skip(1))
            .env_clear()
            .envs(env_overrides)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .into_diagnostic()?;
        std::process::exit(status.code().unwrap_or(1));
    }

    #[cfg(not(target_os = "linux"))]
    fn ensure_xwayland_if_needed(&self) -> miette::Result<()> {
        Ok(())
    }
}
